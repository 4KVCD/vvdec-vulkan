// parse.slang's transform units: CABACReader::cu_residual and what it calls -- the transform tree, the transform
// units' coded block flags, QP deltas and chroma QP offsets, joint Cb-Cr, the residual coding (regular and transform
// skip's) into the planes of levels, LFNST's and MTS's indices -- with the TUs' records (TuRecord).

// --------------------------------------------------------------------------------------------------------------------
// the CU's TUs: their records' words in shared memory until the CU is done
// --------------------------------------------------------------------------------------------------------------------

static const int kMaxCuTus = 16;
groupshared uint s_tu[kMaxCuTus * 16];

uint tuWord( int t, int w )
{
  return uni( s_tu[t * 16 + w] );
}

void setTuWord( int t, int w, uint v )
{
  s_tu[t * 16 + w] = v;
}

bool tuLumaValid( int t )
{
  return ( tuWord( t, 1 ) & 0x10000u ) != 0;
}

bool tuChromaValid( int t )
{
  return ( tuWord( t, 3 ) & 0x10000u ) != 0;
}

bool tuCbf( int t, int comp )
{
  return ( ( tuWord( t, 4 ) >> uint( comp ) ) & 1u ) != 0;
}

void setTuCbf( int t, int comp, bool on )
{
  const uint w = tuWord( t, 4 ) & ~( 1u << uint( comp ) );
  setTuWord( t, 4, w | ( on ? 1u << uint( comp ) : 0u ) );
}

int tuJoint( int t )
{
  return int( ( tuWord( t, 4 ) >> 3 ) & 3u );
}

int tuMts( int t, int comp )
{
  const uint w = tuWord( t, 4 );
  return comp == 0 ? int( ( w >> 5 ) & 7u ) : ( comp == 1 ? int( ( w >> 8 ) & 1u ) : int( ( w >> 9 ) & 1u ) );
}

void setTuMts( int t, int comp, int v )
{
  uint w = tuWord( t, 4 );
  if( comp == 0 )
  {
    w = ( w & ~( 7u << 5 ) ) | ( uint( v ) & 7u ) << 5;
  }
  else if( comp == 1 )
  {
    w = ( w & ~( 1u << 8 ) ) | ( uint( v ) & 1u ) << 8;
  }
  else
  {
    w = ( w & ~( 1u << 9 ) ) | ( uint( v ) & 1u ) << 9;
  }
  setTuWord( t, 4, w );
}

int tuChType( int t )
{
  return int( ( tuWord( t, 4 ) >> 10 ) & 1u );
}

// a TU's block of a component: x, y, w, h in the component's samples
int4 tuBlock( int t, int comp )
{
  const uint p = comp == 0 ? tuWord( t, 0 ) : tuWord( t, 2 );
  const uint s = comp == 0 ? tuWord( t, 1 ) : tuWord( t, 3 );
  return int4( int( p & 0xffffu ), int( p >> 16 ), int( s & 0xffu ), int( ( s >> 8 ) & 0xffu ) );
}

void setMaxScanPos( int t, int comp, int x, int y )
{
  const uint shift = uint( comp ) * 8u;
  setTuWord( t, 6, ( tuWord( t, 6 ) & ~( 0xffu << shift ) ) | ( uint( x ) & 0xffu ) << shift );
  setTuWord( t, 7, ( tuWord( t, 7 ) & ~( 0xffu << shift ) ) | ( uint( y ) & 0xffu ) << shift );
}

// CodingStructure::addTU: luma's block (luma samples) and chroma's (chroma samples) when valid
int addTu( int4 luma, bool lumaValid, int4 chroma, bool chromaValid )
{
  if( g_numTus >= g_tuSlots || g_cu.numTus >= kMaxCuTus )
  {
    g_error = true;
    return -1;
  }
  const int t = g_cu.numTus;
  g_cu.numTus++;
  g_numTus++;
  setTuWord( t, 0, lumaValid ? uint( luma.x ) | uint( luma.y ) << 16 : 0u );
  setTuWord( t, 1, lumaValid ? uint( luma.z ) | uint( luma.w ) << 8 | 1u << 16 : 0u );
  setTuWord( t, 2, chromaValid ? uint( chroma.x ) | uint( chroma.y ) << 16 : 0u );
  setTuWord( t, 3, chromaValid ? uint( chroma.z ) | uint( chroma.w ) << 8 | 1u << 16 : 0u );
  setTuWord( t, 4, uint( g_p.chType ) << 10 );
  for( int w = 5; w < 16; w++ )
  {
    setTuWord( t, w, 0u );
  }
  setTuWord( t, 8, uint( g_cu.local ) );
  return t;
}

void writeTuRecords()
{
  for( int t = 0; t < g_cu.numTus; t++ )
  {
    uint* r = pc.tus + uint( g_ctu * g_tuSlots + g_cu.firstTu + t ) * 16u;
    for( int w = 0; w < 16; w++ )
    {
      r[w] = tuWord( t, w );
    }
  }
}

int4 chromaOf( int4 a )
{
  return int4( a.x >> g_csx, a.y >> g_csy, a.z >> g_csx, a.w >> g_csy );
}

// g_rsScanToZ: the max transform size's tiles in z-order
int rsScanToZ( int numLog2H, int i )
{
  if( numLog2H == 2 )
  {
    const int t[16] = { 0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15 };
    return t[i];
  }
  return i;
}

// CodingStructure::addEmptyTUs
void addEmptyTUs()
{
  const int4 a  = currArea();
  const bool cv = g_chromaFormat != 0;
  if( a.z > g_maxTbSize || a.w > g_maxTbSize )
  {
    const int numTilesH = max( 1, a.z >> g_log2MaxTb );
    const int numTilesV = max( 1, a.w >> g_log2MaxTb );
    const int numLog2H  = log2i( numTilesH );
    const int numLog2V  = log2i( numTilesV );
    const int4 c0       = chromaOf( a );
    for( int i = 0; i < numTilesH * numTilesV; i++ )
    {
      const int  zid = rsScanToZ( numLog2H, i );
      const int  y   = zid >> numLog2H;
      const int  x   = zid & ( ( 1 << numLog2H ) - 1 );
      const int4 l   = int4( a.x + ( a.z >> numLog2H ) * x, a.y + ( a.w >> numLog2V ) * y, a.z >> numLog2H, a.w >> numLog2V );
      const int4 c   = int4( c0.x + ( c0.z >> numLog2H ) * x, c0.y + ( c0.w >> numLog2V ) * y, c0.z >> numLog2H, c0.w >> numLog2V );
      addTu( l, true, c, cv );
    }
  }
  else
  {
    addTu( a, true, chromaOf( a ), cv );
  }
}

// --------------------------------------------------------------------------------------------------------------------
// cu_residual, the transform tree
// --------------------------------------------------------------------------------------------------------------------

// CU::checkAllowedSbt: a bit an SBT index
int checkAllowedSbt()
{
  if( !tool( kToolSbt ) || g_cu.predMode != MODE_INTER || ( g_cu.flags & kCiip ) != 0 )
  {
    return 0;
  }
  const int w = g_cu.lw, h = g_cu.lh;
  if( w > g_maxTbSize || h > g_maxTbSize )
  {
    return 0;
  }
  int allowed = 0;
  allowed |= ( w >= 8 ? 1 : 0 ) << SBT_VER_HALF;
  allowed |= ( h >= 8 ? 1 : 0 ) << SBT_HOR_HALF;
  allowed |= ( w >= 16 ? 1 : 0 ) << SBT_VER_QUAD;
  allowed |= ( h >= 16 ? 1 : 0 ) << SBT_HOR_QUAD;
  return allowed;
}

// CABACReader::sbt_mode
void sbtMode()
{
  const int sbtAllowed = checkAllowedSbt();
  if( sbtAllowed == 0 )
  {
    return;
  }
  const int w = g_cu.lw, h = g_cu.lh;
  if( decodeBin( CTX_SbtFlag + ( w * h <= 256 ? 1 : 0 ) ) == 0 )
  {
    return;
  }
  const bool verHalf = ( ( sbtAllowed >> SBT_VER_HALF ) & 1 ) != 0;
  const bool horHalf = ( ( sbtAllowed >> SBT_HOR_HALF ) & 1 ) != 0;
  const bool verQuad = ( ( sbtAllowed >> SBT_VER_QUAD ) & 1 ) != 0;
  const bool horQuad = ( ( sbtAllowed >> SBT_HOR_QUAD ) & 1 ) != 0;
  bool       quad    = false;
  if( ( horHalf || verHalf ) && ( horQuad || verQuad ) )
  {
    quad = decodeBin( CTX_SbtQuadFlag ) != 0;
  }
  bool hor = false;
  if( ( quad && verQuad && horQuad ) || ( !quad && verHalf && horHalf ) )
  {
    const int ctxIdx = w == h ? 0 : ( w < h ? 1 : 2 );
    hor              = decodeBin( CTX_SbtHorFlag + ctxIdx ) != 0;
  }
  else
  {
    hor = ( quad && horQuad ) || ( !quad && horHalf );
  }
  const int idx = hor ? ( quad ? SBT_HOR_QUAD : SBT_HOR_HALF ) : ( quad ? SBT_VER_QUAD : SBT_VER_HALF );
  g_cu.sbtInfo  = idx + ( g_cu.sbtInfo & 0xf0 );
  const int pos = decodeBin( CTX_SbtPosFlag ) != 0 ? 1 : 0;
  g_cu.sbtInfo  = ( pos << 4 ) + ( g_cu.sbtInfo & 0xcf );
}

// CU::getISPSplitDim
int ispSplitDim( int w, int h, bool horizontal )
{
  const int splitDim    = horizontal ? h : w;
  const int nonSplitDim = horizontal ? w : h;
  const int factor      = nonSplitDim < 16 ? 16 >> log2i( nonSplitDim ) : 1;
  return ( splitDim >> 2 ) < factor ? factor : ( splitDim >> 2 );
}

// a part of the transform tree: its TU (getArea: one channel's blocks in separate trees) and transform_unit
void transformLeaf( int4 luma, int4 chroma, bool chromaValid, int idx, int trDepth, inout CuCtx ctx )
{
  bool tuLuma   = true;
  bool tuChroma = chromaValid;
  if( g_dualTree || g_p.treeType != TREE_D )
  {
    if( g_p.chType == 0 )
    {
      tuChroma = false;
    }
    else
    {
      tuLuma = false;
    }
  }
  const int t = addTu( luma, tuLuma, chroma, tuChroma );
  if( t >= 0 )
  {
    transformUnit( t, idx, trDepth, chromaValid, ctx );
  }
}

// CABACReader::transform_tree: the CU's area, or its parts -- the max transform size's tiles (getMaxTuTiling), ISP's
// (getTUIntraSubPartitions: luma split, the chroma with the last part, none in the dual tree), SBT's (getSbtTuTiling)
// -- each a TU (one call of transformLeaf)
void transformTree( inout CuCtx ctx )
{
  const int4 a     = currArea();
  const bool cv    = g_chromaFormat != 0;
  const int4 c     = chromaOf( a );
  const bool isp   = g_cu.isp != 0 && g_p.chType == 0;
  const bool split = a.z > g_maxTbSize || a.w > g_maxTbSize || g_cu.sbtInfo != 0 || isp;
  const int  kind  = !split ? 0 : ( isp ? 2 : ( g_cu.sbtInfo != 0 ? 3 : 1 ) );
  // the tiles'
  const int numTilesH = max( 1, a.z >> g_log2MaxTb );
  const int numTilesV = max( 1, a.w >> g_log2MaxTb );
  const int numLog2H  = log2i( numTilesH );
  const int numLog2V  = log2i( numTilesV );
  // ISP's
  const bool horizontal = g_cu.isp == HOR_INTRA_SUBPARTITIONS;
  const int  dim        = isp ? ispSplitDim( a.z, a.w, horizontal ) : 1;
  // SBT's
  const int sbtIdx = g_cu.sbtInfo & 0xf;
  const int sbtPos = ( g_cu.sbtInfo >> 4 ) & 3;
  const int n      = kind == 0 ? 1 : ( kind == 1 ? numTilesH * numTilesV : ( kind == 2 ? ( horizontal ? a.w >> log2i( dim ) : a.z >> log2i( dim ) ) : 2 ) );
  for( int i = 0; i < n && !g_error; i++ )
  {
    int4 l   = a;
    int4 cc  = c;
    bool cvi = cv;
    if( kind == 1 )
    {
      const int zid = rsScanToZ( numLog2H, i );
      const int y   = zid >> numLog2H;
      const int x   = zid & ( ( 1 << numLog2H ) - 1 );
      l             = int4( a.x + ( a.z >> numLog2H ) * x, a.y + ( a.w >> numLog2V ) * y, a.z >> numLog2H, a.w >> numLog2V );
      cc            = int4( c.x + ( c.z >> numLog2H ) * x, c.y + ( c.w >> numLog2V ) * y, c.z >> numLog2H, c.w >> numLog2V );
    }
    else if( kind == 2 )
    {
      l   = horizontal ? int4( a.x, a.y + i * dim, a.z, dim ) : int4( a.x + i * dim, a.y, dim, a.w );
      cvi = cv && !g_dualTree && i == n - 1;
    }
    else if( kind == 3 )
    {
      int wf, hf, xf, yf;
      if( sbtIdx == SBT_HOR_QUAD )
      {
        wf = 4;
        xf = 0;
        hf = ( ( i == 0 && sbtPos == 0 ) || ( i == 1 && sbtPos == 1 ) ) ? 1 : 3;
        yf = i == 0 ? 0 : ( sbtPos == 0 ? 1 : 3 );
      }
      else if( sbtIdx == SBT_VER_QUAD )
      {
        wf = ( ( i == 0 && sbtPos == 0 ) || ( i == 1 && sbtPos == 1 ) ) ? 1 : 3;
        xf = i == 0 ? 0 : ( sbtPos == 0 ? 1 : 3 );
        hf = 4;
        yf = 0;
      }
      else if( sbtIdx == SBT_HOR_HALF )
      {
        wf = 4;
        xf = 0;
        hf = 2;
        yf = i == 0 ? 0 : 2;
      }
      else
      {
        wf = 2;
        xf = i == 0 ? 0 : 2;
        hf = 4;
        yf = 0;
      }
      l  = int4( a.x + ( ( a.z * xf ) >> 2 ), a.y + ( ( a.w * yf ) >> 2 ), ( a.z * wf ) >> 2, ( a.w * hf ) >> 2 );
      cc = int4( c.x + ( ( c.z * xf ) >> 2 ), c.y + ( ( c.w * yf ) >> 2 ), ( c.z * wf ) >> 2, ( c.w * hf ) >> 2 );
    }
    transformLeaf( l, cc, cvi, kind == 0 ? s_lvIdx[g_p.top] : i, kind == 0 ? 0 : 1, ctx );
  }
}

// CABACReader::cu_residual
void cuResidual( inout CuCtx ctx )
{
  if( !cuIsIntra() )
  {
    if( !cuMerge() )
    {
      g_cu.rootCbf = decodeBin( CTX_QtRootCbf ) != 0;
    }
    else
    {
      g_cu.rootCbf = true;
    }
    if( g_cu.rootCbf )
    {
      sbtMode();
    }
    if( !g_cu.rootCbf )
    {
      addEmptyTUs();
      return;
    }
  }
  else
  {
    g_cu.rootCbf = true;
  }
  if( cuIsInter() || cuIsIbc() )
  {
    adaptiveColorTransform();
  }
  ctx.violatesLfnstLuma          = false;
  ctx.violatesLfnstChroma        = false;
  ctx.lfnstLastScanPos           = false;
  ctx.violatesMtsCoeffConstraint = false;
  ctx.mtsLastScanPos             = false;

  transformTree( ctx );
  residualLfnstMode( ctx );
  mtsIdx( ctx );

  bool rootCbf = false;
  if( g_cu.lw > 0 )
  {
    rootCbf = rootCbf || ( g_cu.planeCbf & 1 ) != 0;
  }
  if( g_cu.cw > 0 )
  {
    rootCbf = rootCbf || ( g_cu.planeCbf & 6 ) != 0;
  }
  g_cu.rootCbf = rootCbf;
}

// --------------------------------------------------------------------------------------------------------------------
// transform_unit
// --------------------------------------------------------------------------------------------------------------------

// TU::checkTuNoResidual
bool checkTuNoResidual( int idx )
{
  if( ( g_cu.sbtInfo & 0xf ) == 0 )
  {
    return false;
  }
  const int pos = ( g_cu.sbtInfo >> 4 ) & 3;
  return ( pos == 0 && idx == 1 ) || ( pos == 1 && idx == 0 );
}

// CABACReader::cbf_comp
bool cbfComp( int comp, bool prevCbf, bool useIsp )
{
  const int set = comp == 0 ? CTX_QtCbf_0 : ( comp == 1 ? CTX_QtCbf_1 : CTX_QtCbf_2 );
  int       ctx;
  if( ( comp == 0 && g_cu.bdpcmL != 0 ) || ( comp != 0 && g_cu.bdpcmC != 0 ) )
  {
    ctx = comp == 2 ? 2 : 1;
  }
  else if( useIsp && comp == 0 )
  {
    ctx = 2 + ( prevCbf ? 1 : 0 );
  }
  else
  {
    ctx = comp == 2 && prevCbf ? 1 : 0;
  }
  return decodeBin( set + ctx ) != 0;
}

// CABACReader::cu_qp_delta
void cuQpDelta( int predQp )
{
  int qpY = predQp;
  int dqp = unaryMaxSymbol( CTX_DeltaQP, CTX_DeltaQP + 1, CU_DQP_TU_CMAX );
  if( dqp >= CU_DQP_TU_CMAX )
  {
    dqp += expGolombEqprob( 0 );
  }
  if( dqp > 0 )
  {
    if( decodeBinEP() != 0 )
    {
      dqp = -dqp;
    }
    qpY = ( ( predQp + dqp + ( MAX_QP + 1 ) + 2 * g_qpBdOffset ) % ( ( MAX_QP + 1 ) + g_qpBdOffset ) ) - g_qpBdOffset;
  }
  g_cu.qp = qpY;
}

// CABACReader::cu_chroma_qp_offset
void cuChromaQpOffset()
{
  const int length = picWord( kPicCqpListLen );
  int       qpAdj  = int( decodeBin( CTX_ChromaQpAdjFlag ) );
  if( qpAdj != 0 && length > 1 )
  {
    qpAdj += unaryMaxSymbol( CTX_ChromaQpAdjIdc, CTX_ChromaQpAdjIdc, length - 1 );
  }
  g_cu.chromaQpAdj = qpAdj;
  g_csChromaQpAdj  = qpAdj;
}

// CABACReader::joint_cb_cr
void jointCbCr( int t, int cbfMask )
{
  if( !tool( kToolJointCbCr ) )
  {
    return;
  }
  if( ( cuIsIntra() && cbfMask != 0 ) || cbfMask == 3 )
  {
    const int joint = decodeBin( CTX_JointCbCrFlag + cbfMask - 1 ) != 0 ? cbfMask : 0;
    setTuWord( t, 4, ( tuWord( t, 4 ) & ~( 3u << 3 ) ) | uint( joint ) << 3 );
  }
}

// CABACReader::transform_unit; areaCbValid: the partitioner's part has its chroma
void transformUnit( int t, int idx, int trDepth, bool areaCbValid, inout CuCtx ctx )
{
  const bool chromaCbfIsp = g_chromaFormat != 0 && areaCbValid && g_cu.isp != 0;
  const bool tuNoResidual = checkTuNoResidual( idx );
  bool       cbfCb = false, cbfCr = false;
  if( g_chromaFormat != 0 && areaCbValid && ( !cuIsSepTree() || g_p.chType == 1 ) && ( g_cu.isp == 0 || chromaCbfIsp ) )
  {
    if( !( g_cu.sbtInfo != 0 && tuNoResidual ) )
    {
      cbfCb = cbfComp( 1, false, false );
      cbfCr = cbfComp( 2, cbfCb, false );
    }
  }
  const bool sigChroma = g_chromaFormat != 0 && ( cbfCb || cbfCr );
  if( g_p.chType == 0 )
  {
    if( !cuIsIntra() && trDepth == 0 && !sigChroma )
    {
      setTuCbf( t, 0, true );
    }
    else if( g_cu.sbtInfo != 0 && tuNoResidual )
    {
      setTuCbf( t, 0, false );
    }
    else if( g_cu.sbtInfo != 0 && !sigChroma )
    {
      setTuCbf( t, 0, true );
    }
    else
    {
      bool cbfY = true;
      if( g_cu.isp != 0 )
      {
        const bool lumaCbfIsInferredAct = g_cu.act && g_cu.predMode == MODE_INTRA && trDepth == 0 && !sigChroma;
        bool       lastCbfIsInferred    = lumaCbfIsInferredAct;
        const int4 lb                   = tuBlock( t, 0 );
        const int  nTus = g_cu.isp == HOR_INTRA_SUBPARTITIONS ? g_cu.lh >> log2i( lb.w ) : g_cu.lw >> log2i( lb.z );
        if( idx == nTus - 1 )
        {
          bool rootCbfSoFar = false;
          for( int u = 0; u < t; u++ )
          {
            rootCbfSoFar = rootCbfSoFar || tuCbf( u, 0 );
          }
          if( !rootCbfSoFar )
          {
            lastCbfIsInferred = true;
          }
        }
        if( !lastCbfIsInferred )
        {
          // TU::getPrevTUCbf
          const bool prevCbf = t > 0 && tuLumaValid( t - 1 ) && tuCbf( t - 1, 0 );
          cbfY               = cbfComp( 0, prevCbf, true );
        }
      }
      else
      {
        cbfY = cbfComp( 0, false, false );
      }
      setTuCbf( t, 0, cbfY );
    }
  }
  if( g_chromaFormat != 0 && ( g_cu.isp == 0 || chromaCbfIsp ) )
  {
    setTuCbf( t, 1, cbfCb );
    setTuCbf( t, 2, cbfCr );
  }
  g_cu.planeCbf |= ( tuCbf( t, 0 ) ? 1 : 0 ) | ( tuCbf( t, 1 ) ? 2 : 0 ) | ( tuCbf( t, 2 ) ? 4 : 0 );

  const bool lumaOnly  = g_chromaFormat == 0 || !tuChromaValid( t );
  const bool cbfLuma   = tuCbf( t, 0 );
  const bool cbfChroma = lumaOnly ? false : ( cbfCb || cbfCr );
  if( g_cu.lw > 64 || g_cu.lh > 64 || cbfLuma || cbfChroma )
  {
    if( picWord( kPicUseDqp ) != 0 && !ctx.isDQPCoded )
    {
      if( !cuIsSepTree() || tuChType( t ) == 0 )
      {
        cuQpDelta( ctx.qp );
        ctx.qp         = g_cu.qp;
        ctx.isDQPCoded = true;
      }
    }
    if( !cuIsSepTree() || tuChType( t ) == 1 )
    {
      const int channelWidth  = !cuIsSepTree() ? g_cu.lw : g_cu.cw;
      const int channelHeight = !cuIsSepTree() ? g_cu.lh : g_cu.chh;
      if( sliceFlag( kSlChromaQpAdj ) && ( channelWidth > 64 || channelHeight > 64 || cbfChroma ) && !ctx.isChromaQpAdjCoded )
      {
        cuChromaQpOffset();
        ctx.isChromaQpAdjCoded = true;
      }
    }
    if( !lumaOnly )
    {
      jointCbCr( t, ( tuCbf( t, 1 ) ? 2 : 0 ) + ( tuCbf( t, 2 ) ? 1 : 0 ) );
      if( tuJoint( t ) != 0 )
      {
        g_cu.planeCbf |= 6;
      }
    }
    for( int comp = 0; comp <= 2 && !g_error; comp++ )
    {
      if( comp == 0 ? cbfLuma : ( !lumaOnly && tuCbf( t, comp ) ) )
      {
        residualCoding( t, comp, ctx );
      }
    }
  }
}

// --------------------------------------------------------------------------------------------------------------------
// LFNST's and MTS's indices
// --------------------------------------------------------------------------------------------------------------------

// CABACReader::residual_lfnst_mode
void residualLfnstMode( inout CuCtx ctx )
{
  if( !tool( kToolLfnst ) || !cuIsIntra() )
  {
    return;
  }
  const int chIdx = cuIsSepTree() && g_cu.chType == 1 ? 1 : 0;
  if( g_cu.isp != 0 )
  {
    // CU::canUseLfnstWithISP
    const bool horizontal = g_cu.isp == HOR_INTRA_SUBPARTITIONS;
    const int  tw         = horizontal ? g_cu.lw : ispSplitDim( g_cu.lw, g_cu.lh, false );
    const int  th         = horizontal ? ispSplitDim( g_cu.lw, g_cu.lh, true ) : g_cu.lh;
    if( !( tw >= 4 && th >= 4 ) )
    {
      return;
    }
  }
  if( g_cu.mip && !( g_cu.lw >= 16 && g_cu.lh >= 16 ) )
  {
    return;
  }
  if( g_cu.chType == 1 && min( g_cu.cw, g_cu.chh ) < 4 )
  {
    return;
  }
  const int lw = chIdx == 0 ? g_cu.lw : g_cu.cw << g_csx;
  const int lh = chIdx == 0 ? g_cu.lh : g_cu.chh << g_csy;
  if( lw > g_maxTbSize || lh > g_maxTbSize )
  {
    return;
  }
  const bool lumaFlag   = cuIsSepTree() ? g_cu.chType == 0 : true;
  const bool chromaFlag = cuIsSepTree() ? g_cu.chType == 1 : true;
  const bool nonZeroCoeffNonTsCorner8x8 = ( lumaFlag && ctx.violatesLfnstLuma ) || ( chromaFlag && ctx.violatesLfnstChroma );
  bool       isTrSkip = false;
  const int  numComp  = g_chromaFormat == 0 ? 1 : 3;
  for( int t = 0; t < g_cu.numTus; t++ )
  {
    for( int comp = 0; comp < numComp; comp++ )
    {
      const bool valid = comp == 0 ? tuLumaValid( t ) : tuChromaValid( t );
      if( valid && tuCbf( t, comp ) && tuMts( t, comp ) == MTS_SKIP )
      {
        isTrSkip = true;
      }
    }
  }
  if( nonZeroCoeffNonTsCorner8x8 || ( !ctx.lfnstLastScanPos && g_cu.isp == 0 ) || isTrSkip )
  {
    return;
  }
  const int cctx = cuIsSepTree() ? 1 : 0;
  int       idx  = int( decodeBin( CTX_LFNSTIdx + cctx ) );
  if( idx != 0 )
  {
    idx += int( decodeBin( CTX_LFNSTIdx + 2 ) );
  }
  g_cu.lfnst = idx;
}

// CABACReader::mts_idx
void mtsIdx( inout CuCtx ctx )
{
  int mts = tuMts( 0, 0 );
  // CU::isMTSAllowed
  const int tsMax   = 1 << g_log2MaxTs;
  bool      allowed = g_cu.chType == 0;
  allowed = allowed && ( cuIsIntra() ? tool( kToolMtsIntra ) : ( tool( kToolMtsInter ) && cuIsInter() ) );
  allowed = allowed && g_cu.lw <= 32 && g_cu.lh <= 32;
  allowed = allowed && g_cu.isp == 0;
  allowed = allowed && g_cu.sbtInfo == 0;
  allowed = allowed && !( g_cu.bdpcmL != 0 && g_cu.lw <= tsMax && g_cu.lh <= tsMax );
  if( allowed && !ctx.violatesMtsCoeffConstraint && ctx.mtsLastScanPos && g_cu.lfnst == 0 && mts != MTS_SKIP )
  {
    if( decodeBin( CTX_MTSIndex + 0 ) != 0 )
    {
      int ctxIdx = 1;
      mts        = MTS_DST7_DST7;
      for( int i = 0; i < 3; i++, ctxIdx++ )
      {
        const uint symbol = decodeBin( CTX_MTSIndex + ctxIdx );
        mts += int( symbol );
        if( symbol == 0 )
        {
          break;
        }
      }
    }
  }
  setTuMts( 0, 0, mts );
}

// --------------------------------------------------------------------------------------------------------------------
// the residual coding (CoeffCodingContext, residual_coding, residual_codingTS)
// --------------------------------------------------------------------------------------------------------------------


groupshared int  s_coef[4096];   // the block's levels: absolute (the regular coding) or signed (transform skip's)
groupshared uint s_tpl[1024];    // CtxTpl, a byte a position: the sum of absolute levels | the positions << 5
groupshared uint s_out[1024];    // the block's levels to write: position | level << 16
groupshared uint s_final[2048];  // the block's levels as written, 2 a word (int16)
// (s_coef, s_tpl and s_final zero between blocks: cleared where a block left them, after it)
groupshared int  s_sigPos[16];   // the subblock's significant positions, in decoding order
groupshared int  s_gt1Pos[16];

static int g_outCount;

struct Cctx
{
  int  comp, chType, width, height, log2W, log2H;
  int  log2CGW, log2CGH, log2CGSize, widthInGroups, heightInGroups;
  int  maxNumCoeff, maxLog2TrRange;
  bool signHiding;
  int  innerOff, cgOff, subBase;   // the groups' scan in s_scanTab, the group's, the current group's first position
  int  ctxLastX, ctxLastY, maxLastPosX, maxLastPosY, lastOffsetX, lastOffsetY, lastShiftX, lastShiftY;
  int  scanPosLast;
  int  subSetId, subSetPos, subSetPosX, subSetPosY, minSubPos, maxSubPos;
  int  sigGroupCtxId, sigGroupCtxIdTS;
  int  tmplCpSum1, tmplCpDiag;
  int  sigFlagCtx0, sigFlagCtx1, sigFlagCtx2, parFlagCtx, gtx0Ctx, gtx1Ctx;
  uint sigGroups0, sigGroups1;   // m_sigCoeffGroupFlag
  bool bdpcm;
  int  regBinLimit;
  bool ts;
  int  numCtxBins;
};

static Cctx g_cc;


// the position in the block of a scan position of the current group (vvdec's g_scanOrder[SCAN_GROUPED_4x4]: the
// groups in their diagonal scan, each group's positions in its own)
int blockPos( int scanPos )
{
  const int inner = int( uni( s_scanTab[g_cc.innerOff + ( scanPos & ( ( 1 << g_cc.log2CGSize ) - 1 ) )] ) & 0xffffu );
  return g_cc.subBase + ( inner >> g_cc.log2CGW ) * g_cc.width + ( inner & ( ( 1 << g_cc.log2CGW ) - 1 ) );
}

uint tplGet( int pos )
{
  return ( uni( s_tpl[pos >> 2] ) >> uint( ( pos & 3 ) * 8 ) ) & 0xffu;
}

void tplAdd( int pos, uint v )
{
  s_tpl[pos >> 2] += v << uint( ( pos & 3 ) * 8 );
}

bool sigGroup( int pos )
{
  return ( ( pos < 32 ? g_cc.sigGroups0 >> uint( pos ) : g_cc.sigGroups1 >> uint( pos - 32 ) ) & 1u ) != 0;
}

void setSigGroup()
{
  const int pos = g_cc.subSetPos;
  if( pos < 32 )
  {
    g_cc.sigGroups0 |= 1u << uint( pos );
  }
  else
  {
    g_cc.sigGroups1 |= 1u << uint( pos - 32 );
  }
}

// TU::getTbAreaAfterCoefZeroOut
int tbAreaAfterCoefZeroOut( int t, int comp, int w, int h )
{
  if( comp == 0 && ( tuMts( t, comp ) > MTS_SKIP || ( tool( kToolMts ) && g_cu.sbtInfo != 0 && w <= 32 && h <= 32 ) ) )
  {
    w = w == 32 ? 16 : w;
    h = h == 32 ? 16 : h;
  }
  return min( 32, w ) * min( 32, h );
}

// CoeffCodingContext's constructor, with residual_coding's clearing of the block's levels and templates
void initCctx( int t, int comp, bool signHide )
{
  const int4 b      = tuBlock( t, comp );
  g_cc.comp         = comp;
  g_cc.chType       = comp == 0 ? 0 : 1;
  g_cc.width        = b.z;
  g_cc.height       = b.w;
  g_cc.log2W        = log2i( b.z );
  g_cc.log2H        = log2i( b.w );
  const int sbb     = kLog2SbbSize[g_cc.log2W * 7 + g_cc.log2H];
  g_cc.log2CGW      = sbb & 15;
  g_cc.log2CGH      = sbb >> 4;
  g_cc.log2CGSize   = g_cc.log2CGW + g_cc.log2CGH;
  g_cc.widthInGroups  = min( 32, b.z ) >> g_cc.log2CGW;
  g_cc.heightInGroups = min( 32, b.w ) >> g_cc.log2CGH;
  g_cc.maxNumCoeff  = b.z * b.w;
  g_cc.signHiding   = signHide;
  const int range   = picWord( kPicTrRange );
  g_cc.maxLog2TrRange = g_cc.chType == 0 ? range & 0xff : range >> 8;
  g_cc.innerOff     = scanTabOffset( g_cc.log2CGW, g_cc.log2CGH );
  g_cc.cgOff        = scanTabOffset( log2i( g_cc.widthInGroups ), log2i( g_cc.heightInGroups ) );
  g_cc.subBase      = 0;
  g_cc.ctxLastX     = g_cc.chType == 0 ? CTX_LastX_0 : CTX_LastX_1;
  g_cc.ctxLastY     = g_cc.chType == 0 ? CTX_LastY_0 : CTX_LastY_1;
  g_cc.maxLastPosX  = kGroupIdx[min( 32, b.z ) - 1];
  g_cc.maxLastPosY  = kGroupIdx[min( 32, b.w ) - 1];
  g_cc.lastOffsetX  = g_cc.chType == 0 ? kPrefixCtx[g_cc.log2W] : 0;
  g_cc.lastOffsetY  = g_cc.chType == 0 ? kPrefixCtx[g_cc.log2H] : 0;
  g_cc.lastShiftX   = g_cc.chType == 1 ? clip3( 0, 2, b.z >> 3 ) : ( g_cc.log2W + 1 ) >> 2;
  g_cc.lastShiftY   = g_cc.chType == 1 ? clip3( 0, 2, b.w >> 3 ) : ( g_cc.log2H + 1 ) >> 2;
  g_cc.scanPosLast  = -1;
  g_cc.subSetId     = -1;
  g_cc.subSetPos    = -1;
  g_cc.subSetPosX   = -1;
  g_cc.subSetPosY   = -1;
  g_cc.minSubPos    = -1;
  g_cc.maxSubPos    = -1;
  g_cc.sigGroupCtxId   = -1;
  g_cc.sigGroupCtxIdTS = -1;
  g_cc.tmplCpSum1   = -1;
  g_cc.tmplCpDiag   = -1;
  g_cc.sigFlagCtx0  = g_cc.chType == 0 ? CTX_SigFlag_0 : CTX_SigFlag_1;
  g_cc.sigFlagCtx1  = g_cc.chType == 0 ? CTX_SigFlag_2 : CTX_SigFlag_3;
  g_cc.sigFlagCtx2  = g_cc.chType == 0 ? CTX_SigFlag_4 : CTX_SigFlag_5;
  g_cc.parFlagCtx   = g_cc.chType == 0 ? CTX_ParFlag_0 : CTX_ParFlag_1;
  g_cc.gtx0Ctx      = g_cc.chType == 0 ? CTX_GtxFlag_0 : CTX_GtxFlag_1;
  g_cc.gtx1Ctx      = g_cc.chType == 0 ? CTX_GtxFlag_2 : CTX_GtxFlag_3;
  g_cc.sigGroups0   = 0;
  g_cc.sigGroups1   = 0;
  g_cc.bdpcm        = ( comp == 0 ? g_cu.bdpcmL : g_cu.bdpcmC ) != 0;
  g_cc.regBinLimit  = ( tbAreaAfterCoefZeroOut( t, comp, b.z, b.w ) * 28 ) >> 4;
  g_cc.ts           = tuMts( t, comp ) == MTS_SKIP;
  g_cc.numCtxBins   = 0;
  g_outCount        = 0;
}

void clearResidualMemory()
{
  for( int i = g_lane; i < 4096; i += 32 )
  {
    s_coef[i] = 0;
  }
  for( int i = g_lane; i < 1024; i += 32 )
  {
    s_tpl[i] = 0;
  }
  for( int i = g_lane; i < 2048; i += 32 )
  {
    s_final[i] = 0;
  }
  GroupMemoryBarrierWithGroupSync();
}

// CoeffCodingContext::initSubblock
void initSubblock( int subSetId )
{
  g_cc.subSetId   = subSetId;
  g_cc.subSetPos  = int( uni( s_scanTab[g_cc.cgOff + subSetId] ) & 0xffffu );
  g_cc.subSetPosY = g_cc.subSetPos >> log2i( g_cc.widthInGroups );
  g_cc.subSetPosX = g_cc.subSetPos - g_cc.subSetPosY * g_cc.widthInGroups;
  g_cc.subBase    = ( g_cc.subSetPosY << g_cc.log2CGH ) * g_cc.width + ( g_cc.subSetPosX << g_cc.log2CGW );
  g_cc.minSubPos  = subSetId << g_cc.log2CGSize;
  g_cc.maxSubPos  = g_cc.minSubPos + ( 1 << g_cc.log2CGSize ) - 1;
  const bool lastHorGrp = g_cc.subSetPosX == g_cc.widthInGroups - 1;
  const bool lastVerGrp = g_cc.subSetPosY == g_cc.heightInGroups - 1;
  const uint sigRight   = !lastHorGrp && sigGroup( g_cc.subSetPos + 1 ) ? 1u : 0u;
  const uint sigLower   = !lastVerGrp && sigGroup( g_cc.subSetPos + g_cc.widthInGroups ) ? 1u : 0u;
  g_cc.sigGroupCtxId    = ( g_cc.chType == 0 ? CTX_SigCoeffGroup_0 : CTX_SigCoeffGroup_1 ) + int( sigRight | sigLower );
  if( g_cc.ts )
  {
    const int sigLeft  = g_cc.subSetPosX > 0 && sigGroup( g_cc.subSetPos - 1 ) ? 1 : 0;
    const int sigAbove = g_cc.subSetPosY > 0 && sigGroup( g_cc.subSetPos - g_cc.widthInGroups ) ? 1 : 0;
    g_cc.sigGroupCtxIdTS = CTX_TsSigCoeffGroup + sigLeft + sigAbove;
  }
}

int sigCtxIdAbs( int blkPos, int state )
{
  const int posY   = blkPos >> g_cc.log2W;
  const int posX   = blkPos & ( g_cc.width - 1 );
  const int diag   = posX + posY;
  const int tplVal = int( tplGet( blkPos ) );
  const int numPos = tplVal >> 5;
  const int sumAbs = tplVal & 31;
  int       ctxOfs = min( ( sumAbs + 1 ) >> 1, 3 ) + ( diag < 2 ? 4 : 0 );
  if( g_cc.chType == 0 )
  {
    ctxOfs += diag < 5 ? 4 : 0;
  }
  g_cc.tmplCpDiag = diag;
  g_cc.tmplCpSum1 = sumAbs - numPos;
  const int set   = max( 0, state - 1 );
  return ( set == 0 ? g_cc.sigFlagCtx0 : ( set == 1 ? g_cc.sigFlagCtx1 : g_cc.sigFlagCtx2 ) ) + ctxOfs;
}

void absVal1stPass( int blkPos, int absLevel1 )
{
  s_coef[blkPos] = absLevel1;
  const int  posY = blkPos >> g_cc.log2W;
  const int  posX = blkPos & ( g_cc.width - 1 );
  const uint v    = uint( 32 + absLevel1 );
  if( posY > 1 )
  {
    tplAdd( blkPos - 2 * g_cc.width, v );
  }
  if( posY > 0 && posX > 0 )
  {
    tplAdd( blkPos - g_cc.width - 1, v );
  }
  if( posY > 0 )
  {
    tplAdd( blkPos - g_cc.width, v );
  }
  if( posX > 1 )
  {
    tplAdd( blkPos - 2, v );
  }
  if( posX > 0 )
  {
    tplAdd( blkPos - 1, v );
  }
}

int ctxOffsetAbs()
{
  int offset = 0;
  if( g_cc.tmplCpDiag != -1 )
  {
    const int diag = g_cc.tmplCpDiag;
    offset         = min( g_cc.tmplCpSum1, 4 ) + 1;
    offset += diag == 0 ? ( g_cc.chType == 0 ? 15 : 5 ) : ( g_cc.chType == 0 ? ( diag < 3 ? 10 : ( diag < 10 ? 5 : 0 ) ) : 0 );
  }
  return offset;
}

int templateAbsSum( int blkPos, int baseLevel )
{
  const int posY = blkPos >> g_cc.log2W;
  const int posX = blkPos & ( g_cc.width - 1 );
  const int w    = g_cc.width;
  int       sum  = 0;
  if( posX + 2 < w )
  {
    sum += uni( s_coef[blkPos + 1] );
    sum += uni( s_coef[blkPos + 2] );
    if( posY + 1 < g_cc.height )
    {
      sum += uni( s_coef[blkPos + w + 1] );
    }
  }
  else if( posX + 1 < w )
  {
    sum += uni( s_coef[blkPos + 1] );
    if( posY + 1 < g_cc.height )
    {
      sum += uni( s_coef[blkPos + w + 1] );
    }
  }
  if( posY + 2 < g_cc.height )
  {
    sum += uni( s_coef[blkPos + w] );
    sum += uni( s_coef[blkPos + 2 * w] );
  }
  else if( posY + 1 < g_cc.height )
  {
    sum += uni( s_coef[blkPos + w] );
  }
  return max( min( sum - 5 * baseLevel, 31 ), 0 );
}

// CABACReader::last_sig_coeff
int lastSigCoeff( int comp )
{
  int posLastX    = 0;
  int posLastY    = 0;
  int maxLastPosX = g_cc.maxLastPosX;
  int maxLastPosY = g_cc.maxLastPosY;
  if( comp == 0 && tool( kToolMts ) && g_cu.sbtInfo != 0 && g_cc.width <= 32 && g_cc.height <= 32 )
  {
    maxLastPosX = g_cc.width == 32 ? kGroupIdx[15] : maxLastPosX;
    maxLastPosY = g_cc.height == 32 ? kGroupIdx[15] : maxLastPosY;
  }
  for( ; posLastX < maxLastPosX; posLastX++ )
  {
    if( decodeBin( g_cc.ctxLastX + g_cc.lastOffsetX + ( posLastX >> g_cc.lastShiftX ) ) == 0 )
    {
      break;
    }
  }
  for( ; posLastY < maxLastPosY; posLastY++ )
  {
    if( decodeBin( g_cc.ctxLastY + g_cc.lastOffsetY + ( posLastY >> g_cc.lastShiftY ) ) == 0 )
    {
      break;
    }
  }
  if( posLastX > 3 )
  {
    uint      temp  = 0;
    const int count = ( posLastX - 2 ) >> 1;
    for( int i = count - 1; i >= 0; i-- )
    {
      temp += decodeBinEP() << uint( i );
    }
    posLastX = kMinInGroup[posLastX] + int( temp );
  }
  if( posLastY > 3 )
  {
    uint      temp  = 0;
    const int count = ( posLastY - 2 ) >> 1;
    for( int i = count - 1; i >= 0; i-- )
    {
      temp += decodeBinEP() << uint( i );
    }
    posLastY = kMinInGroup[posLastY] + int( temp );
  }
  // the inverse scan: the group's index in the groups' scan, the position's in the group's
  const int cgRaster    = ( posLastY >> g_cc.log2CGH ) * g_cc.widthInGroups + ( posLastX >> g_cc.log2CGW );
  const int innerRaster = ( posLastY & ( ( 1 << g_cc.log2CGH ) - 1 ) ) * ( 1 << g_cc.log2CGW ) + ( posLastX & ( ( 1 << g_cc.log2CGW ) - 1 ) );
  const int scanPos     = int( uni( s_scanTab[g_cc.cgOff + cgRaster] ) >> 16 ) << g_cc.log2CGSize | int( uni( s_scanTab[g_cc.innerOff + innerRaster] ) >> 16 );
  return min( scanPos, g_cc.maxNumCoeff - 1 );
}

// CABACReader::residual_coding_subblock: the subblock's levels appended to s_out
int residualCodingSubblock( int stateTransTab, bool depQuant, inout int state )
{
  const int  minSubPos   = g_cc.minSubPos;
  const bool isLast      = ( g_cc.scanPosLast >> g_cc.log2CGSize ) == g_cc.subSetId;
  const int  firstSigPos = isLast ? g_cc.scanPosLast : g_cc.maxSubPos;
  int        nextSigPos  = firstSigPos;
  bool       sigGroupFlag = isLast || minSubPos == 0;
  if( !sigGroupFlag )
  {
    sigGroupFlag = decodeBin( g_cc.sigGroupCtxId ) != 0;
  }
  if( !sigGroupFlag )
  {
    return 0;
  }
  setSigGroup();

  const int inferSigPos = nextSigPos != g_cc.scanPosLast ? ( g_cc.subSetId != 0 ? minSubPos : -1 ) : nextSigPos;
  int       firstNZPos  = nextSigPos;
  int       lastNZPos   = -1;
  int       numNonZero  = 0;
  int       numGt1      = 0;
  int       remRegBins  = g_cc.regBinLimit;
  uint      gt2Mask     = 0;
  uint      stateVal    = 0;
  for( ; nextSigPos >= minSubPos && remRegBins >= 4; nextSigPos-- )
  {
    const int blkPos  = blockPos( nextSigPos );
    bool      sigFlag = numNonZero == 0 && nextSigPos == inferSigPos;
    if( !sigFlag )
    {
      sigFlag = decodeBin( sigCtxIdAbs( blkPos, state ) ) != 0;
      remRegBins--;
    }
    if( sigFlag )
    {
      const int ctxOff = ctxOffsetAbs();
      stateVal         = ( uint( state >> 1 ) & 1u ) | ( stateVal << 1 );
      s_sigPos[numNonZero] = blkPos;
      numNonZero++;
      firstNZPos = nextSigPos;
      lastNZPos  = max( lastNZPos, nextSigPos );
      const uint gt1Flag = decodeBin( g_cc.gtx1Ctx + ctxOff );
      remRegBins--;
      int absVal;
      if( gt1Flag != 0 )
      {
        const uint parFlag = decodeBin( g_cc.parFlagCtx + ctxOff );
        numGt1++;
        remRegBins--;
        const uint gt2Flag = decodeBin( g_cc.gtx0Ctx + ctxOff );
        gt2Mask |= gt2Flag << uint( numGt1 - 1 );
        remRegBins--;
        s_gt1Pos[numGt1 - 1] = blkPos;
        absVal               = 2 + int( parFlag ) + int( gt2Flag << 1 );
        state                = ( stateTransTab >> ( ( state << 2 ) + int( parFlag << 1 ) ) ) & 3;
      }
      else
      {
        absVal = 1;
        state  = ( stateTransTab >> ( ( state << 2 ) + 2 ) ) & 3;
      }
      absVal1stPass( blkPos, absVal );
    }
    else
    {
      state = ( stateTransTab >> ( state << 2 ) ) & 3;
    }
  }
  g_cc.regBinLimit = remRegBins;

  // the Rice codes of the levels past 3
  for( int k = 0; k < numGt1; k++, gt2Mask >>= 1 )
  {
    if( ( gt2Mask & 1u ) != 0 )
    {
      const int pos     = uni( s_gt1Pos[k] );
      const int sumAll  = templateAbsSum( pos, 4 );
      const int ricePar = kGoRiceParsCoeff[sumAll];
      const int rem     = int( decodeRemAbsEP( ricePar, COEF_REMAIN_BIN_REDUCTION, g_cc.maxLog2TrRange ) );
      s_coef[pos] += rem << 1;
    }
  }
  // the levels coded in bypass
  for( ; nextSigPos >= minSubPos; nextSigPos-- )
  {
    const int sub1   = ( state >> 1 ) & 1;
    const int blkPos = blockPos( nextSigPos );
    const int sumAll = templateAbsSum( blkPos, 0 );
    const int rice   = kGoRiceParsCoeff[sumAll];
    const int pos0   = ( state < 2 ? 1 : 2 ) << rice;
    const int rem    = int( decodeRemAbsEP( rice, COEF_REMAIN_BIN_REDUCTION, g_cc.maxLog2TrRange ) );
    const int tcoeff = rem == pos0 ? 0 : ( rem < pos0 ? rem + 1 : rem );
    state            = ( stateTransTab >> ( ( state << 2 ) + ( ( tcoeff & 1 ) << 1 ) ) ) & 3;
    if( tcoeff != 0 )
    {
      s_coef[blkPos] = tcoeff;
      stateVal       = uint( sub1 ) | ( stateVal << 1 );
      s_sigPos[numNonZero] = blkPos;
      numNonZero++;
      firstNZPos = nextSigPos;
      lastNZPos  = max( lastNZPos, nextSigPos );
    }
  }
  // the signs (the last one hidden: the levels' parity)
  const bool hide     = g_cc.signHiding && lastNZPos - firstNZPos >= SBH_THRESHOLD;
  const int  numSigns = hide ? numNonZero - 1 : numNonZero;
  uint       signs    = decodeBinsEP( numSigns );
  if( numNonZero > numSigns )
  {
    int sumAbs = 0;
    for( int i = 0; i < numNonZero; i++ )
    {
      sumAbs += uni( s_coef[uni( s_sigPos[i] )] );
    }
    signs = ( signs << 1 ) + uint( sumAbs & 1 );
  }
  // the levels (dependent quantization's: 2 q - parity), the first decoded the highest bit
  for( int j = 0; j < numNonZero; j++ )
  {
    const int  pos = uni( s_sigPos[j] );
    const uint bit = uint( numNonZero - 1 - j );
    const int  c   = uni( s_coef[pos] );
    const int  a   = depQuant ? c * 2 - int( ( stateVal >> bit ) & 1u ) : c;
    const int  v   = ( ( signs >> bit ) & 1u ) != 0 ? -a : a;
    s_out[g_outCount] = uint( pos ) | ( uint( v ) << 16 );
    g_outCount++;
  }
  return numNonZero;
}

// TS's contexts
int coefAt( int blkPos, int dx, int dy )
{
  return uni( s_coef[blkPos - dx - dy * g_cc.width] );
}

int sigCtxIdAbsTS( int blkPos )
{
  const int posY = blkPos >> g_cc.log2W;
  const int posX = blkPos & ( g_cc.width - 1 );
  int       numPos = 0;
  if( posX > 0 )
  {
    numPos += coefAt( blkPos, 1, 0 ) != 0 ? 1 : 0;
  }
  if( posY > 0 )
  {
    numPos += coefAt( blkPos, 0, 1 ) != 0 ? 1 : 0;
  }
  return CTX_TsSigFlag + numPos;
}

int signCtxIdAbsTS( int blkPos, bool bdpcm )
{
  const int posY      = blkPos >> g_cc.log2W;
  const int posX      = blkPos & ( g_cc.width - 1 );
  const int rightSign = posX > 0 ? coefAt( blkPos, 1, 0 ) : 0;
  const int belowSign = posY > 0 ? coefAt( blkPos, 0, 1 ) : 0;
  int       signCtx;
  if( ( rightSign == 0 && belowSign == 0 ) || ( rightSign * belowSign ) < 0 )
  {
    signCtx = 0;
  }
  else if( rightSign >= 0 && belowSign >= 0 )
  {
    signCtx = 1;
  }
  else
  {
    signCtx = 2;
  }
  if( bdpcm )
  {
    signCtx += 3;
  }
  return CTX_TsResidualSign + signCtx;
}

int lrg1CtxIdAbsTS( int blkPos, bool bdpcm )
{
  const int posY   = blkPos >> g_cc.log2W;
  const int posX   = blkPos & ( g_cc.width - 1 );
  int       numPos = 0;
  if( bdpcm )
  {
    numPos = 3;
  }
  else
  {
    if( posX > 0 )
    {
      numPos += coefAt( blkPos, 1, 0 ) != 0 ? 1 : 0;
    }
    if( posY > 0 )
    {
      numPos += coefAt( blkPos, 0, 1 ) != 0 ? 1 : 0;
    }
  }
  return CTX_TsLrg1Flag + numPos;
}

// CoeffCodingContext::decDeriveModCoeff
int decDeriveModCoeff( int rightPixel, int belowPixel, int absCoeff )
{
  if( absCoeff == 0 )
  {
    return 0;
  }
  const int pred1 = max( abs( belowPixel ), abs( rightPixel ) );
  if( absCoeff == 1 && pred1 > 0 )
  {
    return pred1;
  }
  return absCoeff - ( absCoeff <= pred1 ? 1 : 0 );
}

// CABACReader::residual_coding_subblockTS
void residualCodingSubblockTS( inout int maxX, inout int maxY )
{
  const int lastPos     = g_cc.maxSubPos;   // vvdec's minSubPos
  const int firstSigPos = g_cc.minSubPos;
  int       nextSigPos  = firstSigPos;
  uint      signPattern = 0;
  const bool isLastSubSet = ( ( g_cc.maxNumCoeff - 1 ) >> g_cc.log2CGSize ) == g_cc.subSetId;
  bool sigGroupFlag = isLastSubSet && g_cc.sigGroups0 == 0 && g_cc.sigGroups1 == 0;
  if( !sigGroupFlag )
  {
    sigGroupFlag = decodeBin( g_cc.sigGroupCtxIdTS ) != 0;
  }
  if( !sigGroupFlag )
  {
    return;
  }
  setSigGroup();

  const int inferSigPos     = lastPos;
  int       numNonZero      = 0;
  int       lastScanPosPass1 = -1;
  int       lastScanPosPass2 = -1;
  for( ; nextSigPos <= lastPos && g_cc.numCtxBins >= 4; nextSigPos++ )
  {
    const int blkPos  = blockPos( nextSigPos );
    bool      sigFlag = numNonZero == 0 && nextSigPos == inferSigPos;
    if( !sigFlag )
    {
      sigFlag = decodeBin( sigCtxIdAbsTS( blkPos ) ) != 0;
      g_cc.numCtxBins--;
    }
    if( sigFlag )
    {
      const uint sign = decodeBin( signCtxIdAbsTS( blkPos, g_cc.bdpcm ) );
      g_cc.numCtxBins--;
      signPattern += sign << uint( numNonZero );
      s_sigPos[numNonZero] = blkPos;
      numNonZero++;
      const uint gt1Flag = decodeBin( lrg1CtxIdAbsTS( blkPos, g_cc.bdpcm ) );
      g_cc.numCtxBins--;
      uint parFlag = 0;
      if( gt1Flag != 0 )
      {
        parFlag = decodeBin( CTX_TsParFlag );
        g_cc.numCtxBins--;
      }
      s_coef[blkPos] = ( sign != 0 ? -1 : 1 ) * int( 1 + parFlag + gt1Flag );
    }
    lastScanPosPass1 = nextSigPos;
  }
  // the greater-than flags
  for( int scanPos = firstSigPos; scanPos <= lastPos && g_cc.numCtxBins >= 4; scanPos++ )
  {
    const int pos    = blockPos( scanPos );
    int       tcoeff = uni( s_coef[pos] );
    int       cutoffVal = 2;
    for( int i = 0; i < 4; i++ )
    {
      if( tcoeff < 0 )
      {
        tcoeff = -tcoeff;
      }
      if( tcoeff >= cutoffVal )
      {
        const uint gt2Flag = decodeBin( CTX_TsGtxFlag + ( cutoffVal >> 1 ) );
        tcoeff += int( gt2Flag << 1 );
        g_cc.numCtxBins--;
      }
      cutoffVal += 2;
    }
    s_coef[pos]      = tcoeff;
    lastScanPosPass2 = scanPos;
  }
  // the Rice codes
  for( int scanPos = firstSigPos; scanPos <= lastPos; scanPos++ )
  {
    const int pos       = blockPos( scanPos );
    int       tcoeff    = uni( s_coef[pos] );
    const int cutoffVal = scanPos <= lastScanPosPass2 ? 10 : ( scanPos <= lastScanPosPass1 ? 2 : 0 );
    if( tcoeff < 0 )
    {
      tcoeff = -tcoeff;
    }
    if( tcoeff >= cutoffVal )
    {
      const int rem = int( decodeRemAbsEP( 1, COEF_REMAIN_BIN_REDUCTION, g_cc.maxLog2TrRange ) );
      tcoeff += scanPos <= lastScanPosPass1 ? rem << 1 : rem;
      if( tcoeff != 0 && scanPos > lastScanPosPass1 )
      {
        const uint sign = decodeBinEP();
        signPattern += sign << uint( numNonZero );
        s_sigPos[numNonZero] = pos;
        numNonZero++;
      }
    }
    if( !g_cc.bdpcm && cutoffVal != 0 )
    {
      if( tcoeff > 0 )
      {
        const int posY  = pos >> g_cc.log2W;
        const int posX  = pos & ( g_cc.width - 1 );
        const int right = posX > 0 ? coefAt( pos, 1, 0 ) : 0;
        const int below = posY > 0 ? coefAt( pos, 0, 1 ) : 0;
        tcoeff          = decDeriveModCoeff( right, below, tcoeff );
      }
    }
    s_coef[pos] = tcoeff;
  }
  // the levels
  for( int k = 0; k < numNonZero; k++ )
  {
    const int pos      = uni( s_sigPos[k] );
    const int absCoeff = uni( s_coef[pos] );
    maxX               = max( maxX, pos & ( g_cc.width - 1 ) );
    maxY               = max( maxY, pos >> g_cc.log2W );
    const int v        = ( signPattern & 1u ) != 0 ? -absCoeff : absCoeff;
    s_coef[pos]        = v;
    s_out[g_outCount]  = uint( pos ) | ( uint( v ) << 16 );
    g_outCount++;
    signPattern >>= 1;
  }
}

// the block's levels (s_out) to the planes of levels, over [0, w) x [0, h) of the block (the rest of it zero); then
// the shared memory the block used cleared
void writeLevels( int comp, int4 b, int w, int h )
{
  PROF_BEGIN( levelsStart );
  writeLevelsBody( comp, b, w, h );
  PROF_END( kProfLevels, levelsStart );
}

void writeLevelsBody( int comp, int4 b, int w, int h )
{
  const int stride = comp == 0 ? pc.stride0 : pc.stride12;
  const int pw     = comp == 0 ? g_width : g_width >> g_csx;
  const int ph     = comp == 0 ? g_height : g_height >> g_csy;
  const bool inside = b.x + w <= pw && b.y + h <= ph;
  if( !inside )
  {
    g_error = true;   // (a block off the planes)
  }
  // the levels into s_final at their positions (the threads write the same)
  for( int i = 0; i < g_outCount; i++ )
  {
    const uint e   = s_out[i];
    const uint pos = e & 0xffffu;
    const uint sh  = ( pos & 1u ) * 16u;
    s_final[pos >> 1] = ( s_final[pos >> 1] & ~( 0xffffu << sh ) ) | ( ( e >> 16 ) << sh );
  }
  GroupMemoryBarrierWithGroupSync();
  if( inside )
  {
    for( int u = g_lane; u < h << 5; u += 32 )
    {
      const int     x   = u & 31;
      const int     y   = u >> 5;
      if( x >= w )
      {
        continue;
      }
      const int     pos = ( y << g_cc.log2W ) + x;
      const int16_t v   = int16_t( int( s_final[pos >> 1] << ( 16u - ( uint( pos ) & 1u ) * 16u ) ) >> 16 );
      const int     at  = ( b.y + y ) * stride + b.x + x;
      if( comp == 0 )
      {
        pc.levels0[at] = v;
      }
      else if( comp == 1 )
      {
        pc.levels1[at] = v;
      }
      else
      {
        pc.levels2[at] = v;
      }
    }
  }
  GroupMemoryBarrierWithGroupSync();
  // cleared where the block left values: its levels, their templates' neighbours
  const int wb = g_cc.width;
  for( int i = 0; i < g_outCount; i++ )
  {
    const int pos  = int( s_out[i] & 0xffffu );
    const int posY = pos >> g_cc.log2W;
    const int posX = pos & ( wb - 1 );
    s_final[pos >> 1] = 0u;
    s_coef[pos]       = 0;
    s_tpl[pos >> 2]   = 0u;
    if( posY > 1 )
    {
      s_tpl[( pos - 2 * wb ) >> 2] = 0u;
    }
    if( posY > 0 && posX > 0 )
    {
      s_tpl[( pos - wb - 1 ) >> 2] = 0u;
    }
    if( posY > 0 )
    {
      s_tpl[( pos - wb ) >> 2] = 0u;
    }
    if( posX > 1 )
    {
      s_tpl[( pos - 2 ) >> 2] = 0u;
    }
    if( posX > 0 )
    {
      s_tpl[( pos - 1 ) >> 2] = 0u;
    }
  }
  GroupMemoryBarrierWithGroupSync();
}

// CABACReader::ts_flag, TU::isTSAllowed
void tsFlag( int t, int comp )
{
  int        ts      = ( ( g_cu.bdpcmL != 0 && comp == 0 ) || ( g_cu.bdpcmC != 0 && comp != 0 ) ) ? 1 : ( tuMts( t, comp ) == MTS_SKIP ? 1 : 0 );
  const int4 b       = tuBlock( t, comp );
  const int  tsMax   = 1 << g_log2MaxTs;
  bool       allowed = tool( kToolTs );
  allowed            = allowed && ( g_cu.isp == 0 || comp != 0 );
  allowed            = allowed && !( g_cu.bdpcmL != 0 && comp == 0 );
  allowed            = allowed && !( g_cu.bdpcmC != 0 && comp != 0 );
  allowed            = allowed && b.z <= tsMax && b.w <= tsMax;
  allowed            = allowed && g_cu.sbtInfo == 0;
  if( allowed )
  {
    ts = int( decodeBin( CTX_MTSIndex + ( comp == 0 ? 4 : 5 ) ) );
  }
  setTuMts( t, comp, ts != 0 ? MTS_SKIP : MTS_DCT2_DCT2 );
}

// CABACReader::residual_codingTS
void residualCodingTS( int t, int comp )
{
  initCctx( t, comp, false );
  g_cc.numCtxBins = ( g_cc.maxNumCoeff * 7 ) >> 2;
  int maxX = 0;
  int maxY = 0;
  for( int subSetId = 0; subSetId <= ( g_cc.maxNumCoeff - 1 ) >> g_cc.log2CGSize; subSetId++ )
  {
    initSubblock( subSetId );
    residualCodingSubblockTS( maxX, maxY );
  }
  if( g_cc.bdpcm )
  {
    setMaxScanPos( t, comp, g_cc.width, g_cc.height );
  }
  else
  {
    setMaxScanPos( t, comp, maxX, maxY );
  }
  const int4 b = tuBlock( t, comp );
  writeLevels( comp, b, b.z, b.w );
}

// CABACReader::residual_coding (timed)
void residualCoding( int t, int comp, inout CuCtx ctx )
{
  PROF_BEGIN( resStart );
  residualCodingBody( t, comp, ctx );
  PROF_END( kProfResidual, resStart );
  PROF_COUNT( kProfBlocks, 1 );
  PROF_COUNT( kProfLevelsN, g_outCount );
}

void residualCodingBody( int t, int comp, inout CuCtx ctx )
{
  if( comp == 2 && tuJoint( t ) == 3 )
  {
    return;
  }
  tsFlag( t, comp );
  if( tuMts( t, comp ) == MTS_SKIP && !sliceFlag( kSlTsResOff ) )
  {
    residualCodingTS( t, comp );
    return;
  }
  initCctx( t, comp, sliceFlag( kSlSignHiding ) );
  g_cc.scanPosLast = lastSigCoeff( comp );
  const bool notTs = tuMts( t, comp ) != MTS_SKIP;
  if( notTs && g_cc.height >= 4 && g_cc.width >= 4 )
  {
    const int maxLfnstPos = ( ( g_cc.height == 4 && g_cc.width == 4 ) || ( g_cc.height == 8 && g_cc.width == 8 ) ) ? 7 : 15;
    if( g_cc.scanPosLast > maxLfnstPos )
    {
      if( comp == 0 )
      {
        ctx.violatesLfnstLuma = true;
      }
      else
      {
        ctx.violatesLfnstChroma = true;
      }
    }
    if( g_cc.scanPosLast >= 1 )
    {
      ctx.lfnstLastScanPos = true;
    }
  }
  if( comp == 0 && notTs && g_cc.scanPosLast >= 1 )
  {
    ctx.mtsLastScanPos = true;
  }

  const int  stateTransTab  = sliceFlag( kSlDepQuant ) ? 32040 : 0;
  const bool depQuant       = sliceFlag( kSlDepQuant ) && notTs;
  int        state          = 0;
  int        maxX           = 0;
  int        maxY           = 0;
  const bool skipBlkPreCond = comp == 0 && tool( kToolMts ) && g_cu.sbtInfo != 0 && g_cc.height <= 32 && g_cc.width <= 32;
  for( int subSetId = g_cc.scanPosLast >> g_cc.log2CGSize; subSetId >= 0; subSetId-- )
  {
    initSubblock( subSetId );
    if( skipBlkPreCond )
    {
      if( ( g_cc.height == 32 && g_cc.subSetPosY >= ( 16 >> g_cc.log2CGH ) ) || ( g_cc.width == 32 && g_cc.subSetPosX >= ( 16 >> g_cc.log2CGW ) ) )
      {
        continue;
      }
    }
    const int numSig = residualCodingSubblock( stateTransTab, depQuant, state );
    if( numSig > 0 )
    {
      maxX = max( maxX, g_cc.subSetPosX );
      maxY = max( maxY, g_cc.subSetPosY );
    }
    if( comp == 0 && sigGroup( g_cc.subSetPos ) && ( g_cc.subSetPosY > 3 || g_cc.subSetPosX > 3 ) )
    {
      ctx.violatesMtsCoeffConstraint = true;
    }
  }
  if( g_cc.bdpcm )
  {
    maxX = g_cc.width;
    maxY = g_cc.height;
  }
  else
  {
    maxX = ( maxX + 1 ) << g_cc.log2CGW;
    maxY = ( maxY + 1 ) << g_cc.log2CGH;
  }
  writeLevels( comp, tuBlock( t, comp ), maxX, maxY );
  if( g_cc.scanPosLast == 0 )
  {
    setMaxScanPos( t, comp, 0, 0 );
  }
  else
  {
    setMaxScanPos( t, comp, maxX - 1, maxY - 1 );
  }
}
