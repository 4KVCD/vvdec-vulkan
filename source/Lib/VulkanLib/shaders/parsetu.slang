// The transform unit part of parse.slang, which includes this file: CABACReader::cu_residual and what it calls. That
// is the transform tree and its TUs (the CU, or its tiles of the largest transform size, ISP's or SBT's parts), each
// TU's coded block flags, delta QP, chroma QP offset and joint Cb-Cr mode, the residual coding (regular and transform
// skip) of each coded block, and the CU's LFNST and MTS indices. The TUs' records (GpuFrontEnd.h's TuRecord) stay in
// shared memory while their CU is parsed; the residual coding writes each coded block's levels into pc.levels, and
// where it put them into the TU's record.

// --------------------------------------------------------------------------------------------------------------------
// The current CU's TUs: their records stay in shared memory until the CU is done; writeTuRecords then stores them
// --------------------------------------------------------------------------------------------------------------------

static const int kMaxCuTus = 16;   // the most TUs a CU may have here (more give up the substream)
groupshared uint s_tu[kMaxCuTus * 16];

// Word w of the record of TU t (t: the TU's index in the current CU)
uint tuWord( int t, int w )
{
  return uni( s_tu[t * 16 + w] );
}

void setTuWord( int t, int w, uint v )
{
  s_tu[t * 16 + w] = v;
}

// The fields of a TU record (see GpuFrontEnd.h's TuRecord)
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

// The MTS index of component comp: luma's (3 bits), or a chroma component's transform skip flag, which is MTS_SKIP
// or MTS_DCT2_DCT2
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

// The channel type of the partitioner when TU t was added
int tuChType( int t )
{
  return int( ( tuWord( t, 4 ) >> 10 ) & 1u );
}

// TU t's block of a component: x, y, w, h in the component's samples
int4 tuBlock( int t, int comp )
{
  const uint p = comp == 0 ? tuWord( t, 0 ) : tuWord( t, 2 );
  const uint s = comp == 0 ? tuWord( t, 1 ) : tuWord( t, 3 );
  return int4( int( p & 0xffffu ), int( p >> 16 ), int( s & 0xffu ), int( ( s >> 8 ) & 0xffu ) );
}

// Sets where the component's levels are: their first one's index in pc.levels (record word 9 + comp) and the width
// of their rows (record word 12, a byte per component)
void setTuLevels( int t, int comp, uint at, int width )
{
  const uint shift = uint( comp ) * 8u;
  setTuWord( t, 9 + comp, at );
  setTuWord( t, 12, ( tuWord( t, 12 ) & ~( 0xffu << shift ) ) | ( uint( width ) & 0xffu ) << shift );
}

// Sets the component's maxScanPosX and maxScanPosY (record words 6 and 7, a byte per component)
void setMaxScanPos( int t, int comp, int x, int y )
{
  const uint shift = uint( comp ) * 8u;
  setTuWord( t, 6, ( tuWord( t, 6 ) & ~( 0xffu << shift ) ) | ( uint( x ) & 0xffu ) << shift );
  setTuWord( t, 7, ( tuWord( t, 7 ) & ~( 0xffu << shift ) ) | ( uint( y ) & 0xffu ) << shift );
}

// Adds a TU to the current CU (CodingStructure::addTU) with its luma block (in luma samples) and its chroma block (in
// chroma samples), each when valid. Returns the TU's index in the CU, or -1, giving up the substream, when the CTU
// has more TUs than a CTU can have (g_tuSlots) or the CU more than kMaxCuTus.
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
  // Words 5 to 15 start at zero: threads 5 to 15 clear one each
  if( g_lane >= 5 && g_lane < 16 )
  {
    s_tu[t * 16 + g_lane] = 0u;
  }
  setTuWord( t, 8, uint( g_cu.local ) );
  return t;
}

// Writes the CU's TU records as the staged records from at on (as far as there is room), each with its meta word: the
// CTU << kLocalBits | the TU's index among the CTU's TUs. The threads store the records together, one word each: the
// CU's records are 16 words a TU, one after another in s_tu and in pc.tus alike.
void writeTuRecords( uint at )
{
  const uint room   = uint( pc.tuCap ) > at ? uint( pc.tuCap ) - at : 0u;
  const int  stored = int( min( uint( g_cu.numTus ), room ) );   // the TUs there is room for
  for( int w = g_lane; w < stored * 16; w += 32 )
  {
    pc.tus[at * uint( kTuRecordWords ) + uint( w )] = s_tu[w];
  }
  if( g_lane < stored )
  {
    pc.tuMeta[at + uint( g_lane )] = uint( g_ctu ) << kLocalBits | uint( g_cu.firstTu + g_lane );
  }
}

// The chroma block, in chroma samples, of an area in luma samples
int4 chromaOf( int4 a )
{
  return int4( a.x >> g_csx, a.y >> g_csy, a.z >> g_csx, a.w >> g_csy );
}

// vvdec's g_rsScanToZ: the raster index of the i-th tile of the largest transform size, in decoding order. That order
// is z-order when the CU is 4 tiles wide, raster order otherwise.
int rsScanToZ( int numLog2H, int i )
{
  if( numLog2H == 2 )
  {
    const int t[16] = { 0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15 };
    return t[i];
  }
  return i;
}

// Adds the TUs of a CU without residuals, skipped or with rqt_root_cbf 0 (CodingStructure::addEmptyTUs): one TU for
// the CU, or one per tile of the largest transform size
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
// cu_residual and the transform tree
// --------------------------------------------------------------------------------------------------------------------

// The SBT types the CU allows (CU::checkAllowedSbt), a bit per type: 1 << SBT_VER_HALF, ...
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

// Parses cu_sbt_flag, cu_sbt_quad_flag, cu_sbt_horizontal_flag and cu_sbt_pos_flag (CABACReader::sbt_mode) into
// sbtInfo: the SBT type in bits 0..3, the position in bits 4 and 5
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

// The size of each ISP part across the split, in luma samples (CU::getISPSplitDim): a quarter of the CU's side, but
// at least enough for the part to hold 16 samples
int ispSplitDim( int w, int h, bool horizontal )
{
  const int splitDim    = horizontal ? h : w;
  const int nonSplitDim = horizontal ? w : h;
  const int factor      = nonSplitDim < 16 ? 16 >> log2i( nonSplitDim ) : 1;
  return ( splitDim >> 2 ) < factor ? factor : ( splitDim >> 2 );
}

// A leaf of the transform tree: adds its TU, which in a separate tree has only the current channel type's blocks (as
// vvdec's getArea gives them), and parses its transform_unit
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

// Parses the CU's transform tree (CABACReader::transform_tree): one TU for the CU, or one for each of its parts. The
// parts are the tiles of the largest transform size (getMaxTuTiling), ISP's parts (getTUIntraSubPartitions: the luma
// is split, the chroma goes with the last part, and in the dual tree there is no chroma) or SBT's two parts
// (getSbtTuTiling). The loop
// keeps transformLeaf to one call site, as every call is inlined.
// How transformTree splits the CU into transform units
static const int kTuWholeCu = 0;   // one TU: the CU
static const int kTuTiles   = 1;   // tiles of the largest transform size
static const int kTuIsp     = 2;   // ISP's sub-partitions
static const int kTuSbt     = 3;   // SBT's two parts

// SBT's part i (0 or 1) of the CU (getSbtTuTiling), in quarters of the CU: x, y, width, height
int4 sbtPartQuarters( int i, int sbtIdx, int sbtPos )
{
  const bool small = ( i == 0 && sbtPos == 0 ) || ( i == 1 && sbtPos == 1 );   // the part of 1 quarter, of a quad split
  if( sbtIdx == SBT_HOR_QUAD )
  {
    return int4( 0, i == 0 ? 0 : ( sbtPos == 0 ? 1 : 3 ), 4, small ? 1 : 3 );
  }
  if( sbtIdx == SBT_VER_QUAD )
  {
    return int4( i == 0 ? 0 : ( sbtPos == 0 ? 1 : 3 ), 0, small ? 1 : 3, 4 );
  }
  if( sbtIdx == SBT_HOR_HALF )
  {
    return int4( 0, i == 0 ? 0 : 2, 4, 2 );
  }
  return int4( i == 0 ? 0 : 2, 0, 2, 4 );
}

void transformTree( inout CuCtx ctx )
{
  const int4 a     = currArea();
  const bool cv    = g_chromaFormat != 0;
  const int4 c     = chromaOf( a );
  const bool isp   = g_cu.isp != 0 && g_p.chType == 0;
  const bool split = a.z > g_maxTbSize || a.w > g_maxTbSize || g_cu.sbtInfo != 0 || isp;
  const int  kind  = !split ? kTuWholeCu : ( isp ? kTuIsp : ( g_cu.sbtInfo != 0 ? kTuSbt : kTuTiles ) );
  // The tiles of the largest transform size
  const int numTilesH = max( 1, a.z >> g_log2MaxTb );
  const int numTilesV = max( 1, a.w >> g_log2MaxTb );
  const int numLog2H  = log2i( numTilesH );
  const int numLog2V  = log2i( numTilesV );
  // ISP's parts, dim luma samples across the split
  const bool horizontal = g_cu.isp == HOR_INTRA_SUBPARTITIONS;
  const int  dim        = isp ? ispSplitDim( a.z, a.w, horizontal ) : 1;
  // SBT's type and position
  const int sbtIdx = g_cu.sbtInfo & 0xf;
  const int sbtPos = ( g_cu.sbtInfo >> 4 ) & 3;
  const int n      = kind == kTuWholeCu ? 1
                     : ( kind == kTuTiles ? numTilesH * numTilesV : ( kind == kTuIsp ? ( horizontal ? a.w >> log2i( dim ) : a.z >> log2i( dim ) ) : 2 ) );
  for( int i = 0; i < n && !g_error; i++ )
  {
    int4 l   = a;
    int4 cc  = c;
    bool cvi = cv;
    if( kind == kTuTiles )
    {
      const int zid = rsScanToZ( numLog2H, i );
      const int y   = zid >> numLog2H;
      const int x   = zid & ( ( 1 << numLog2H ) - 1 );
      l             = int4( a.x + ( a.z >> numLog2H ) * x, a.y + ( a.w >> numLog2V ) * y, a.z >> numLog2H, a.w >> numLog2V );
      cc            = int4( c.x + ( c.z >> numLog2H ) * x, c.y + ( c.w >> numLog2V ) * y, c.z >> numLog2H, c.w >> numLog2V );
    }
    else if( kind == kTuIsp )
    {
      l   = horizontal ? int4( a.x, a.y + i * dim, a.z, dim ) : int4( a.x + i * dim, a.y, dim, a.w );
      cvi = cv && !g_dualTree && i == n - 1;
    }
    else if( kind == kTuSbt )
    {
      const int4 q = sbtPartQuarters( i, sbtIdx, sbtPos );
      l            = int4( a.x + ( ( a.z * q.x ) >> 2 ), a.y + ( ( a.w * q.y ) >> 2 ), ( a.z * q.z ) >> 2, ( a.w * q.w ) >> 2 );
      cc           = int4( c.x + ( ( c.z * q.x ) >> 2 ), c.y + ( ( c.w * q.y ) >> 2 ), ( c.z * q.z ) >> 2, ( c.w * q.w ) >> 2 );
    }
    transformLeaf( l, cc, cvi, kind == kTuWholeCu ? g_lv.idx : i, kind == kTuWholeCu ? 0 : 1, ctx );
  }
}

// Parses the CU's residuals (CABACReader::cu_residual): an inter CU's rqt_root_cbf and SBT, ACT for inter and IBC
// CUs, the transform tree, LFNST's and MTS's indices, and finally the CU's root CBF from its planes' CBFs
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

// With SBT, whether part idx of the CU is the one without residual (TU::checkTuNoResidual)
bool checkTuNoResidual( int idx )
{
  if( ( g_cu.sbtInfo & 0xf ) == 0 )
  {
    return false;
  }
  const int pos = ( g_cu.sbtInfo >> 4 ) & 3;
  return ( pos == 0 && idx == 1 ) || ( pos == 1 && idx == 0 );
}

// Parses a TU's coded block flag of component comp (CABACReader::cbf_comp). prevCbf is Cb's flag when parsing Cr's,
// and the previous ISP part's luma flag with useIsp; both choose the context.
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

// Parses cu_qp_delta_abs and cu_qp_delta_sign_flag (CABACReader::cu_qp_delta) and sets the CU's QP from the
// predicted QP predQp, wrapped into the valid range
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

// Parses cu_chroma_qp_offset_flag and cu_chroma_qp_offset_idx (CABACReader::cu_chroma_qp_offset): 0 none, else the
// entry of the PPS's chroma QP offset list plus one. CodingStructure::chromaQpAdj keeps it for the next CUs.
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

// Parses tu_joint_cbcr_residual_flag (CABACReader::joint_cb_cr); when it is set, TU t's joint Cb-Cr mode is its chroma
// CBF mask (Cb's flag << 1 | Cr's)
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

// Parses TU t (CABACReader::transform_unit): its coded block flags, then, when it has a coded block, the delta QP, the
// chroma QP offset, the joint Cb-Cr mode and the residual coding of each coded block. idx is the TU's part in the
// transform tree's split, trDepth its depth there (0 or 1); areaCbValid says whether the partitioner's part has a
// chroma block.
void transformUnit( int t, int idx, int trDepth, bool areaCbValid, inout CuCtx ctx )
{
  const bool chromaCbfIsp = g_chromaFormat != 0 && areaCbValid && g_cu.isp != 0;
  const bool tuNoResidual = checkTuNoResidual( idx );
  // The chroma CBFs
  bool       cbfCb = false, cbfCr = false;
  if( g_chromaFormat != 0 && areaCbValid && ( !cuIsSepTree() || g_p.chType == 1 ) && ( g_cu.isp == 0 || chromaCbfIsp ) )
  {
    if( !( g_cu.sbtInfo != 0 && tuNoResidual ) )
    {
      cbfCb = cbfComp( 1, false, false );
      cbfCr = cbfComp( 2, cbfCb, false );
    }
  }
  // The luma CBF. It is inferred for an inter CU's unsplit TU without chroma residual (1), for SBT's part without
  // residual (0) and its other part when that has no chroma residual (1), and for an ISP CU's last part when no part
  // before it has one (1); otherwise it is parsed.
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
          // the previous part's luma CBF (TU::getPrevTUCbf)
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

  // A TU with a coded block (or of a CU larger than 64): the delta QP and the chroma QP offset, once per quantization
  // group, the joint Cb-Cr mode, then the residual coding of each coded block
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

// Parses an intra CU's lfnst_idx (CABACReader::residual_lfnst_mode) when its size, its tools and what its residual
// coding found allow LFNST
void residualLfnstMode( inout CuCtx ctx )
{
  if( !tool( kToolLfnst ) || !cuIsIntra() )
  {
    return;
  }
  const int chIdx = cuIsSepTree() && g_cu.chType == 1 ? 1 : 0;
  if( g_cu.isp != 0 )
  {
    // ISP's parts must be at least 4x4 (CU::canUseLfnstWithISP)
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

// Parses mts_idx (CABACReader::mts_idx) into the luma MTS index of the CU's first TU, when MTS is allowed and the
// residual coding found the coefficients it needs
void mtsIdx( inout CuCtx ctx )
{
  int mts = tuMts( 0, 0 );
  // whether the CU allows MTS (CU::isMTSAllowed)
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
// The residual coding (CoeffCodingContext, residual_coding, residual_codingTS)
// --------------------------------------------------------------------------------------------------------------------


// The coded block being parsed. Positions in it are raster positions with the block's width as the row stride. A
// block codes at most its top-left 32 x 32 (the zero-out of 64-wide and 64-tall blocks), so its positions are below
// kCodedPositions: 32 rows of 64. (The shared memory here is sized to that: the parse's workgroups stay resident for
// long, and what they take the reconstruction's shaders cannot have.)
static const int kCodedPositions = 32 * 64;
// its levels, in 16 bits: absolute while their group is parsed (regular coding), signed once it is (and in transform
// skip blocks); and zeros two rows of 64 further, which the templates read below the last coded row. (A magnitude of
// 2^15 is stored as -2^15 and reads back right through abs(); a conforming stream has no level of +2^15.)
static const int kCoefWords = kCodedPositions + 2 * 64;
groupshared int16_t s_coef[kCoefWords];
// With dependent quantization, a bit per position: the level is 2 |q| less 1 there (the quantizer's state was 2 or 3)
static const int kDqWords = kCodedPositions / 32;
groupshared uint s_dqBits[kDqWords];
// (s_coef and s_dqBits are zero whenever a block starts: after each block, writeLevels clears what it set)

// The current coefficient group's templates (CoeffCodingContext's), a lane per position of the group (lane r: the
// group's raster position r): the template of a position is the five positions one and two to its right, one and two
// below it and below-right of it. g_tplSig holds the sum of the template's first-pass levels | the number of nonzero
// ones << 5 (CtxTpl: the significance flag's and the greater-than flags' contexts); g_tplAbs the sum of the template's
// absolute levels so far (templateAbsSum's: the Rice parameter). loadGroupTemplates works them out when the group
// starts; the group's levels then add to them as they are decoded (addToGroupTemplates).
static uint g_tplSig;
static uint g_tplAbs;

static int g_outCount;   // the block's nonzero levels

// Lane arrays (laneValue, setLane). Of the current block, its scans as s_scanTab holds them, word k in thread k: the
// scan inside a group (g_innerScan), and the scan of the groups (up to 64 words: g_groupScan0, then g_groupScan1);
// initCctx loads them. Of the current group: the block position | the raster position in the group << 16, and the
// absolute level, of its k-th nonzero level in decoding order (g_sigPos, g_sigAbs), and for its k-th level greater
// than 1, that level's index among the nonzero ones (g_gt1Sig).
static uint g_innerScan;
static uint g_groupScan0;
static uint g_groupScan1;
static int  g_sigPos;
static int  g_sigAbs;
static int  g_gt1Sig;

// vvdec's CoeffCodingContext: the coded block's sizes and scans, its context sets, and the current coefficient group
struct Cctx
{
  int  comp, chType, width, height, log2W, log2H;
  // The coefficient groups: their log2 size, and their number across and down the zero-out area (32 x 32 at most)
  int  log2CGW, log2CGH, log2CGSize, widthInGroups, heightInGroups;
  int  maxNumCoeff, maxLog2TrRange;
  bool signHiding;
  // Where the scan inside a group (innerOff) and the scan of the groups (cgOff) start in s_scanTab, and the block
  // position of the current group's top-left coefficient (subBase)
  int  innerOff, cgOff, subBase;
  // last_sig_coeff's contexts and limits
  int  ctxLastX, ctxLastY, maxLastPosX, maxLastPosY, lastOffsetX, lastOffsetY, lastShiftX, lastShiftY;
  int  scanPosLast;
  // The current group: its index in the groups' scan, its raster index among the groups and its column and row in
  // groups, its first and last scan positions
  int  subSetId, subSetPos, subSetPosX, subSetPosY, minSubPos, maxSubPos;
  int  sigGroupCtxId, sigGroupCtxIdTS;   // the contexts of the group's coded_sub_block_flag: regular, transform skip
  int  tmplCpSum1, tmplCpDiag;           // what sigCtxIdAbs found in the template, for ctxOffsetAbs
  // The first contexts of the significance flags' sets (one per dependent quantization state set), of the parity
  // flags, and of the greater-than flags (vvdec's m_gtxFlagCtxSet[0], used for the greater-than-3 flag, and [1], used
  // for the greater-than-1 flag)
  int  sigFlagCtx0, sigFlagCtx1, sigFlagCtx2, parFlagCtx, gtx0Ctx, gtx1Ctx;
  uint sigGroups0, sigGroups1;   // m_sigCoeffGroupFlag: the coded_sub_block_flags of the 64 groups, a bit each
  bool bdpcm;
  int  regBinLimit;              // the context-coded bins the regular coding may still use in the block
  bool ts;                       // transform skip
  int  numCtxBins;               // the context-coded bins the transform skip coding may still use
};

static Cctx g_cc;


// The raster position in the current group of scan position scanPos, which lies in the group. vvdec's
// g_scanOrder[SCAN_GROUPED_4x4] scans the groups diagonally and, within each group, its positions diagonally; the
// group's scan is g_innerScan.
int groupRaster( int scanPos )
{
  return int( laneValue( g_innerScan, uint( scanPos & ( ( 1 << g_cc.log2CGSize ) - 1 ) ) ) & 0xffffu );
}

// The block position of the current group's raster position inner
int blockPosOfRaster( int inner )
{
  return g_cc.subBase + ( inner >> g_cc.log2CGW ) * g_cc.width + ( inner & ( ( 1 << g_cc.log2CGW ) - 1 ) );
}

// The block position of scan position scanPos, which lies in the current group
int blockPos( int scanPos )
{
  return blockPosOfRaster( groupRaster( scanPos ) );
}

// The first-pass level (sig + gt1 + par + 2 gt3) that a level of magnitude v was decoded with: v up to 3, else 4 plus
// its parity (the rest came in the remainder)
int firstPassLevel( int v )
{
  return v <= 3 ? v : 4 + ( v & 1 );
}

// Works out the current group's templates (g_tplSig, g_tplAbs) from the levels decoded so far: those of the groups
// right of and below it (its own positions hold no level yet, and the positions outside the block count as zero).
// Each lane works out its position's template, the five positions in one round of reads.
void loadGroupTemplates()
{
  const int r     = g_lane;
  const int cgw   = 1 << g_cc.log2CGW;
  const int x     = ( g_cc.subSetPosX << g_cc.log2CGW ) + ( r & ( cgw - 1 ) );
  const int y     = ( g_cc.subSetPosY << g_cc.log2CGH ) + ( r >> g_cc.log2CGW );
  const int pos   = y * g_cc.width + x;
  const bool mine = r < ( 1 << g_cc.log2CGSize );
  int        sum1 = 0, count = 0, sumAbs = 0;
  [unroll]
  for( int k = 0; k < 5; k++ )
  {
    const int  dx     = k == 0 || k == 4 ? 1 : ( k == 1 ? 2 : 0 );
    const int  dy     = k == 2 || k == 4 ? 1 : ( k == 3 ? 2 : 0 );
    const bool inside = mine && x + dx < g_cc.width && y + dy < g_cc.height;
    const int  at     = inside ? pos + dy * g_cc.width + dx : 0;   // (a lane off the block reads position 0, unused)
    const int  v      = inside ? abs( int( s_coef[at] ) ) : 0;
    sum1 += firstPassLevel( v );
    count += v != 0 ? 1 : 0;
    sumAbs += v;
  }
  g_tplSig = uint( sum1 + ( count << 5 ) );
  g_tplAbs = uint( sumAbs );
}

// The lanes (the current group's raster positions) whose templates hold raster position q of the group: the
// positions one and two left of q, one and two above it, and above-left of it, inside the group
uint groupTemplateMask( int q )
{
  const int cgw  = 1 << g_cc.log2CGW;
  const int x    = q & ( cgw - 1 );
  const int y    = q >> g_cc.log2CGW;
  uint      mask = 0;
  mask |= x >= 1 ? 1u << uint( q - 1 ) : 0u;
  mask |= x >= 2 ? 1u << uint( q - 2 ) : 0u;
  mask |= y >= 1 ? 1u << uint( q - cgw ) : 0u;
  mask |= y >= 2 ? 1u << uint( q - 2 * cgw ) : 0u;
  mask |= x >= 1 && y >= 1 ? 1u << uint( q - cgw - 1 ) : 0u;
  return mask;
}

// A level decoded at the group's raster position q adds to the templates that hold q: sig to g_tplSig (the
// first-pass level and the count of nonzero ones, 32 + level; 0 after the first pass), abs to g_tplAbs
void addToGroupTemplates( int q, uint sig, uint abs )
{
  if( ( ( groupTemplateMask( q ) >> uint( g_lane ) ) & 1u ) != 0 )
  {
    g_tplSig += sig;
    g_tplAbs += abs;
  }
}

// The coded_sub_block_flag of the group at raster index pos among the groups
bool sigGroup( int pos )
{
  return ( ( pos < 32 ? g_cc.sigGroups0 >> uint( pos ) : g_cc.sigGroups1 >> uint( pos - 32 ) ) & 1u ) != 0;
}

// Sets the current group's coded_sub_block_flag (CoeffCodingContext::setSigGroup)
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

// The block's area left after the zero-out (TU::getTbAreaAfterCoefZeroOut): 32 x 32 at most, and sides of 32 count
// as 16 for luma with an MTS kernel, or with SBT when the SPS enables MTS
int tbAreaAfterCoefZeroOut( int t, int comp, int w, int h )
{
  if( comp == 0 && ( tuMts( t, comp ) > MTS_SKIP || ( tool( kToolMts ) && g_cu.sbtInfo != 0 && w <= 32 && h <= 32 ) ) )
  {
    w = w == 32 ? 16 : w;
    h = h == 32 ? 16 : h;
  }
  return min( 32, w ) * min( 32, h );
}

// Sets g_cc up for TU t's block of component comp, as CoeffCodingContext's constructor does, and zeroes g_outCount.
// (vvdec's residual_coding and the constructor also clear the block's levels and templates; here s_coef is already
// zero, as writeLevels clears it after each block, and the templates are the groups' lanes.)
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
  // The block's scans into the lane arrays (a table near the end of s_scanTab has no words past it: min keeps the
  // threads past the table inside s_scanTab, and their words go unused)
  g_innerScan       = s_scanTab[min( g_cc.innerOff + g_lane, 960 )];
  g_groupScan0      = s_scanTab[min( g_cc.cgOff + g_lane, 960 )];
  g_groupScan1      = s_scanTab[min( g_cc.cgOff + 32 + g_lane, 960 )];
}

// Word k of the current block's scan of the groups (s_scanTab's table from g_cc.cgOff on)
uint groupScanWord( int k )
{
  return k < 32 ? laneValue( g_groupScan0, uint( k ) ) : laneValue( g_groupScan1, uint( k - 32 ) );
}

// Zeroes s_coef and s_dqBits when the workgroup starts; writeLevels keeps them zero between blocks from then on
void clearResidualMemory()
{
  for( int i = g_lane; i < kCoefWords; i += 32 )
  {
    s_coef[i] = int16_t( 0 );
  }
  for( int i = g_lane; i < kDqWords; i += 32 )
  {
    s_dqBits[i] = 0;
  }
  GroupMemoryBarrierWithGroupSync();
}

// Makes group subSetId (its index in the groups' scan) the current group and works out the contexts of its
// coded_sub_block_flag from the flags of the groups next to it (CoeffCodingContext::initSubblock)
void initSubblock( int subSetId )
{
  g_cc.subSetId   = subSetId;
  g_cc.subSetPos  = int( groupScanWord( subSetId ) & 0xffffu );
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

// The context of sig_coeff_flag at the current group's raster position q (block position blkPos), from its template
// and the dependent quantization state (CoeffCodingContext::sigCtxIdAbs); keeps the template's values for
// ctxOffsetAbs
int sigCtxIdAbs( int q, int blkPos, int state )
{
  const int posY   = blkPos >> g_cc.log2W;
  const int posX   = blkPos & ( g_cc.width - 1 );
  const int diag   = posX + posY;
  const int tplVal = int( laneValue( g_tplSig, uint( q ) ) );
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

// The context offset of the greater-than and parity flags at the position sigCtxIdAbs last looked at
// (CoeffCodingContext::ctxOffsetAbs)
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

// The sum of the levels' magnitudes in the template of the current group's raster position q, less 5 * baseLevel,
// clipped to 0..31 (CoeffCodingContext::templateAbsSum); it picks the Rice parameter
int templateAbsSum( int q, int baseLevel )
{
  return max( min( int( laneValue( g_tplAbs, uint( q ) ) ) - 5 * baseLevel, 31 ), 0 );
}

// Parses the last significant position (CABACReader::last_sig_coeff: its x and y prefixes, then their suffixes) and
// returns its scan position
int lastSigCoeff( int comp )
{
  int posLastX    = 0;
  int posLastY    = 0;
  int maxLastPosX = g_cc.maxLastPosX;
  int maxLastPosY = g_cc.maxLastPosY;
  // (with SBT and MTS on, a luma side of 32 keeps only its first 16 positions)
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
  // the suffixes, ( prefix - 2 ) / 2 bypass bins each
  if( posLastX > 3 )
  {
    posLastX = kMinInGroup[posLastX] + int( decodeBinsEP( ( posLastX - 2 ) >> 1 ) );
  }
  if( posLastY > 3 )
  {
    posLastY = kMinInGroup[posLastY] + int( decodeBinsEP( ( posLastY - 2 ) >> 1 ) );
  }
  // The scan position from the inverse scans: the scan index of the position's group among the groups, times the
  // group size, plus the scan index of the position within its group
  const int cgRaster    = ( posLastY >> g_cc.log2CGH ) * g_cc.widthInGroups + ( posLastX >> g_cc.log2CGW );
  const int innerRaster = ( posLastY & ( ( 1 << g_cc.log2CGH ) - 1 ) ) * ( 1 << g_cc.log2CGW ) + ( posLastX & ( ( 1 << g_cc.log2CGW ) - 1 ) );
  const int scanPos     = int( groupScanWord( cgRaster ) >> 16 ) << g_cc.log2CGSize | int( laneValue( g_innerScan, uint( innerRaster ) ) >> 16 );
  return min( scanPos, g_cc.maxNumCoeff - 1 );
}

// Parses the current coefficient group of a regular (not transform skip) block and puts its nonzero levels, signed,
// into s_coef (CABACReader::residual_coding_subblock). state is dependent quantization's state, carried from group to
// group; stateTransTab is its transition table (0 without dependent quantization). Returns the number of nonzero
// levels.
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
  loadGroupTemplates();

  // First pass, while the budget of context-coded bins lasts: sig_coeff_flag, abs_level_gtx_flag[0] (greater than 1),
  // par_level_flag and abs_level_gtx_flag[1] (greater than 3)
  const int inferSigPos = nextSigPos != g_cc.scanPosLast ? ( g_cc.subSetId != 0 ? minSubPos : -1 ) : nextSigPos;
  int       firstNZPos  = nextSigPos;
  int       lastNZPos   = -1;
  int       numNonZero  = 0;
  int       numGt1      = 0;
  int       remRegBins  = g_cc.regBinLimit;
  uint      gt2Mask     = 0;
  uint      stateVal    = 0;
  int       sumAbsLevels = 0;   // the group's absolute levels added up, for sign data hiding
  for( ; nextSigPos >= minSubPos && remRegBins >= 4; nextSigPos-- )
  {
    const int q       = groupRaster( nextSigPos );
    const int blkPos  = blockPosOfRaster( q );
    bool      sigFlag = numNonZero == 0 && nextSigPos == inferSigPos;
    // The contexts of the significance flag and of the greater-than-1 flag, fetched together (the greater-than flags'
    // and the parity flag's come from the template sigCtxIdAbs keeps; an inferred significance keeps the last one's,
    // as vvdec's ctxOffsetAbs does)
    const int sigCtx = sigFlag ? 0 : sigCtxIdAbs( q, blkPos, state );
    const int ctxOff = ctxOffsetAbs();
    const int gt1Ctx = g_cc.gtx1Ctx + ctxOff;
    const uint2 sigModel = contextModel( sigCtx );
    const uint2 gt1Model = contextModel( gt1Ctx );
    if( !sigFlag )
    {
      sigFlag = decodeBinWith( sigCtx, sigModel ) != 0;
      remRegBins--;
    }
    if( sigFlag )
    {
      stateVal         = ( uint( state >> 1 ) & 1u ) | ( stateVal << 1 );
      g_sigPos         = setLane( g_sigPos, numNonZero, blkPos | q << 16 );
      numNonZero++;
      firstNZPos = nextSigPos;
      lastNZPos  = max( lastNZPos, nextSigPos );
      const uint gt1Flag = decodeBinWith( gt1Ctx, gt1Model );
      remRegBins--;
      int absVal;
      if( gt1Flag != 0 )
      {
        // (the parity flag's and the greater-than-3 flag's contexts, fetched together)
        const int   parCtx   = g_cc.parFlagCtx + ctxOff;
        const int   gt3Ctx   = g_cc.gtx0Ctx + ctxOff;
        const uint2 parModel = contextModel( parCtx );
        const uint2 gt3Model = contextModel( gt3Ctx );
        const uint parFlag = decodeBinWith( parCtx, parModel );
        numGt1++;
        remRegBins--;
        const uint gt2Flag = decodeBinWith( gt3Ctx, gt3Model );
        gt2Mask |= gt2Flag << uint( numGt1 - 1 );
        remRegBins--;
        g_gt1Sig             = setLane( g_gt1Sig, numGt1 - 1, numNonZero - 1 );
        absVal               = 2 + int( parFlag ) + int( gt2Flag << 1 );
        state                = ( stateTransTab >> ( ( state << 2 ) + int( parFlag << 1 ) ) ) & 3;
      }
      else
      {
        absVal = 1;
        state  = ( stateTransTab >> ( ( state << 2 ) + 2 ) ) & 3;
      }
      g_sigAbs = setLane( g_sigAbs, numNonZero - 1, absVal );
      sumAbsLevels += absVal;
      s_coef[blkPos] = int16_t( absVal );
      addToGroupTemplates( q, 32u + uint( absVal ), uint( absVal ) );
    }
    else
    {
      state = ( stateTransTab >> ( state << 2 ) ) & 3;
    }
  }
  g_cc.regBinLimit = remRegBins;

  // Second pass: abs_remainder, a Rice code, of the levels the first pass found greater than 3
  for( int k = 0; k < numGt1; k++, gt2Mask >>= 1 )
  {
    if( ( gt2Mask & 1u ) != 0 )
    {
      const int sig     = laneValue( g_gt1Sig, k );
      const int posQ    = laneValue( g_sigPos, sig );
      const int pos     = posQ & 0xffff;
      const int q       = posQ >> 16;
      const int sumAll  = templateAbsSum( q, 4 );
      const int ricePar = kGoRiceParsCoeff[sumAll];
      const int rem     = int( decodeRemAbsEP( ricePar, COEF_REMAIN_BIN_REDUCTION, g_cc.maxLog2TrRange ) );
      const int level   = laneValue( g_sigAbs, sig ) + ( rem << 1 );
      g_sigAbs          = setLane( g_sigAbs, sig, level );
      s_coef[pos]       = int16_t( level );
      addToGroupTemplates( q, 0u, uint( rem << 1 ) );
      sumAbsLevels += rem << 1;
    }
  }
  // The positions the first pass did not reach: their whole levels in bypass bins (dec_abs_level)
  for( ; nextSigPos >= minSubPos; nextSigPos-- )
  {
    const int sub1   = ( state >> 1 ) & 1;
    const int q      = groupRaster( nextSigPos );
    const int blkPos = blockPosOfRaster( q );
    const int sumAll = templateAbsSum( q, 0 );
    const int rice   = kGoRiceParsCoeff[sumAll];
    const int pos0   = ( state < 2 ? 1 : 2 ) << rice;
    const int rem    = int( decodeRemAbsEP( rice, COEF_REMAIN_BIN_REDUCTION, g_cc.maxLog2TrRange ) );
    const int tcoeff = rem == pos0 ? 0 : ( rem < pos0 ? rem + 1 : rem );
    state            = ( stateTransTab >> ( ( state << 2 ) + ( ( tcoeff & 1 ) << 1 ) ) ) & 3;
    if( tcoeff != 0 )
    {
      s_coef[blkPos] = int16_t( tcoeff );
      addToGroupTemplates( q, 0u, uint( tcoeff ) );
      stateVal       = uint( sub1 ) | ( stateVal << 1 );
      g_sigPos       = setLane( g_sigPos, numNonZero, blkPos | q << 16 );
      g_sigAbs       = setLane( g_sigAbs, numNonZero, tcoeff );
      sumAbsLevels += tcoeff;
      numNonZero++;
      firstNZPos = nextSigPos;
      lastNZPos  = max( lastNZPos, nextSigPos );
    }
  }
  // The signs. With sign data hiding, when the group's first and last nonzero positions are at least SBH_THRESHOLD
  // apart, the last level's sign is not coded: the parity of the sum of the absolute levels gives it.
  const bool hide     = g_cc.signHiding && lastNZPos - firstNZPos >= SBH_THRESHOLD;
  const int  numSigns = hide ? numNonZero - 1 : numNonZero;
  uint       signs    = decodeBinsEP( numSigns );
  if( numNonZero > numSigns )
  {
    signs = ( signs << 1 ) + uint( sumAbsLevels & 1 );
  }
  // The signs into s_coef, thread j taking the j-th level decoded. With dependent quantization vvdec stores a level q
  // as 2 * |q|, minus 1 when the quantizer's state at its position was 2 or 3 (the bits of stateVal): s_coef keeps
  // |q| for the later groups' templates, s_dqBits the bit, and writeLevels works the level out. The first level
  // decoded has the highest bit in signs and stateVal.
  if( g_lane < numNonZero )
  {
    const uint bit = uint( numNonZero - 1 - g_lane );
    const int  c   = g_sigAbs;
    const int  pos = g_sigPos & 0xffff;
    s_coef[pos]    = int16_t( ( ( signs >> bit ) & 1u ) != 0 ? -c : c );
    if( depQuant && ( ( stateVal >> bit ) & 1u ) != 0 )
    {
      InterlockedOr( s_dqBits[pos >> 5], 1u << uint( pos & 31 ) );
    }
  }
  g_outCount += numNonZero;
  return numNonZero;
}

// Transform skip's contexts and level mapping look at the levels left of and above a position, which vvdec names the
// right and below ones (rightPixel, belowPixel). coefAt is the level dx positions to the left and dy rows above.
int coefAt( int blkPos, int dx, int dy )
{
  return uni( int( s_coef[blkPos - dx - dy * g_cc.width] ) );
}

// The context of transform skip's sig_coeff_flag: how many of the two neighbours are nonzero
// (CoeffCodingContext::sigCtxIdAbsTS)
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

// The context of transform skip's coeff_sign_flag, from the neighbours' signs (CoeffCodingContext::signCtxIdAbsTS)
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

// The context of transform skip's greater-than-1 flag (CoeffCodingContext::lrg1CtxIdAbsTS)
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

// Transform skip's level mapping (CoeffCodingContext::decDeriveModCoeff). With pred1 the larger of the two
// neighbours' absolute levels, a coded absolute level of 1 stands for pred1, the coded levels 2 to pred1 for one less,
// and the others for themselves.
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

// Parses the current coefficient group of a transform skip block, in forward scan order, and leaves its nonzero
// levels, signed, in s_coef (CABACReader::residual_coding_subblockTS); maxX and maxY grow to the largest x and y of a
// nonzero level
void residualCodingSubblockTS( inout int maxX, inout int maxY )
{
  const int lastPos     = g_cc.maxSubPos;   // (vvdec's code names this minSubPos)
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

  // First pass, while the budget of context-coded bins lasts: sig_coeff_flag, coeff_sign_flag, abs_level_gtx_flag[0]
  // and par_level_flag
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
      g_sigPos     = setLane( g_sigPos, numNonZero, blkPos );
      numNonZero++;
      const uint gt1Flag = decodeBin( lrg1CtxIdAbsTS( blkPos, g_cc.bdpcm ) );
      g_cc.numCtxBins--;
      uint parFlag = 0;
      if( gt1Flag != 0 )
      {
        parFlag = decodeBin( CTX_TsParFlag );
        g_cc.numCtxBins--;
      }
      s_coef[blkPos] = int16_t( ( sign != 0 ? -1 : 1 ) * int( 1 + parFlag + gt1Flag ) );
    }
    lastScanPosPass1 = nextSigPos;
  }
  // Second pass, while the budget lasts: the greater-than flags abs_level_gtx_flag[1..4]
  for( int scanPos = firstSigPos; scanPos <= lastPos && g_cc.numCtxBins >= 4; scanPos++ )
  {
    const int pos    = blockPos( scanPos );
    int       tcoeff = uni( int( s_coef[pos] ) );
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
    s_coef[pos]      = int16_t( tcoeff );
    lastScanPosPass2 = scanPos;
  }
  // Third pass: abs_remainder (Rice parameter 1); the positions the first pass did not reach get their whole level
  // and their sign in bypass bins. Then the level mapping (decDeriveModCoeff), without BDPCM, of the positions the
  // first pass reached.
  for( int scanPos = firstSigPos; scanPos <= lastPos; scanPos++ )
  {
    const int pos       = blockPos( scanPos );
    int       tcoeff    = uni( int( s_coef[pos] ) );
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
        g_sigPos = setLane( g_sigPos, numNonZero, pos );
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
    s_coef[pos] = int16_t( tcoeff );
  }
  // The signed levels into s_coef (the contexts of the later groups read them), and the largest x and y
  for( int k = 0; k < numNonZero; k++ )
  {
    const int pos      = laneValue( g_sigPos, k );
    const int absCoeff = uni( int( s_coef[pos] ) );
    maxX               = max( maxX, pos & ( g_cc.width - 1 ) );
    maxY               = max( maxY, pos >> g_cc.log2W );
    const int v        = ( signPattern & 1u ) != 0 ? -absCoeff : absCoeff;
    s_coef[pos]        = int16_t( v );
    g_outCount++;
    signPattern >>= 1;
  }
}

// Writes the levels (s_coef; with dependent quantization, dq, 2 |q| less s_dqBits' bit) of TU t's block b of component
// comp over the rectangle [0, w) x [0, h), zeros included (w is at most 32), into pc.levels: the rectangle's rows one
// after the other, in the substream's chunk (past the room, pc.levelCap, the chunks are only counted, like the
// records: the host then parses the picture again with more).
// Every level outside the rectangle is zero and is not written: the transforms read no further than the rectangle.
// The TU's record keeps the place and the rows' width. Then clears the shared memory that the block used.
// (writeLevelsBody does the work; this wrapper times it for PROFILE.)
void writeLevels( int t, int comp, int4 b, int w, int h, bool dq )
{
  PROF_BEGIN( levelsStart );
  writeLevelsBody( t, comp, b, w, h, dq );
  PROF_END( kProfLevels, levelsStart );
}

void writeLevelsBody( int t, int comp, int4 b, int w, int h, bool dq )
{
  const int pw = comp == 0 ? g_width : g_width >> g_csx;
  const int ph = comp == 0 ? g_height : g_height >> g_csy;
  if( b.x + w > pw || b.y + h > ph )
  {
    g_error = true;   // (a block off the planes: give up the substream)
  }
  // The rectangle's place, a whole number of words, in the substream's chunk: a new one when it does not fit
  const uint size = uint( w * h + 1 ) & ~1u;
  if( g_levelsAt + size > g_levelsEnd )
  {
    uint chunk = 0;
    if( g_lane == 0 )
    {
      chunk = pc.counters[7].add( kLevelChunk );
    }
    g_levelsAt  = uni( chunk );
    g_levelsEnd = g_levelsAt + kLevelChunk;
  }
  const uint at   = g_levelsAt;
  g_levelsAt     += size;
  const bool room = at + size <= uint( pc.levelCap );
  setTuLevels( t, comp, at, w );
  // Each thread takes one column of the rectangle (its lane's): writes its levels when there is room, then zeroes
  // s_coef there (every level the block has lies inside the rectangle)
  for( int u = g_lane; u < h << 5; u += 32 )
  {
    const int x = u & 31;
    const int y = u >> 5;
    if( x >= w )
    {
      continue;
    }
    const int pos = ( y << g_cc.log2W ) + x;
    int       v   = int( s_coef[pos] );
    if( dq && v != 0 )
    {
      const int a = 2 * abs( v ) - int( ( s_dqBits[pos >> 5] >> uint( pos & 31 ) ) & 1u );
      v           = v < 0 ? -a : a;
    }
    if( room )
    {
      pc.levels[at + uint( y * w + x )] = int16_t( v );
    }
    s_coef[pos] = int16_t( 0 );
  }
  GroupMemoryBarrierWithGroupSync();
  if( dq )
  {
    // (the words of s_dqBits up to the rectangle's last position)
    const int words = ( ( ( h - 1 ) << g_cc.log2W ) + w + 31 ) >> 5;
    for( int i = g_lane; i < words; i += 32 )
    {
      s_dqBits[i] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
  }
}

// Parses transform_skip_flag of TU t's component comp, when TU::isTSAllowed allows it (CABACReader::ts_flag), and
// stores it as the component's MTS index; BDPCM implies transform skip
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

// The transform skip residual coding of TU t's block of comp (CABACReader::residual_codingTS): its groups in forward
// order; then the levels over the whole block, and maxScanPosX / maxScanPosY (the whole block with BDPCM)
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
  writeLevels( t, comp, b, b.z, b.w, false );
}

// CABACReader::residual_coding of TU t's block of comp; this wrapper times residualCodingBody for PROFILE
void residualCoding( int t, int comp, inout CuCtx ctx )
{
  PROF_BEGIN( resStart );
  residualCodingBody( t, comp, ctx );
  // The block is done with: zeroing its state ends its values' lives here, as codingUnitAtPart does for the CU's
  g_cc = (Cctx) 0;
  PROF_END( kProfResidual, resStart );
  PROF_COUNT( kProfBlocks, 1 );
  PROF_COUNT( kProfLevelsN, g_outCount );
}

// Parses transform_skip_flag, then the block's residual coding (transform skip's or the regular one). That covers the
// last position and what it means for LFNST and MTS, the groups from the last one back to the first, and the levels
// over the rectangle of the groups that have any (the whole block with BDPCM).
void residualCodingBody( int t, int comp, inout CuCtx ctx )
{
  if( comp == 2 && tuJoint( t ) == 3 )
  {
    return;   // (joint Cb-Cr mode 3 codes only Cb's residual)
  }
  tsFlag( t, comp );
  if( tuMts( t, comp ) == MTS_SKIP && !sliceFlag( kSlTsResOff ) )
  {
    residualCodingTS( t, comp );
    return;
  }
  initCctx( t, comp, sliceFlag( kSlSignHiding ) );
  g_cc.scanPosLast = lastSigCoeff( comp );
  // What the last position means for LFNST's and MTS's indices, which are parsed after the CU's residuals
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

  // The groups, from the last position's back to the first, with dependent quantization's state running through them
  // (32040 packs its state transitions, 2 bits per state and parity, as in vvdec's residual_coding)
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
      // (with SBT and MTS on, a luma side of 32 codes only the groups of its first 16 positions)
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
  // The levels over the rectangle of the groups that have any (the whole block with BDPCM), and maxScanPosX /
  // maxScanPosY, which vvdec sets to that rectangle's last column and row (0 when only the first position is coded)
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
  writeLevels( t, comp, tuBlock( t, comp ), maxX, maxY, depQuant );
  if( g_cc.scanPosLast == 0 )
  {
    setMaxScanPos( t, comp, 0, 0 );
  }
  else
  {
    setMaxScanPos( t, comp, maxX - 1, maxY - 1 );
  }
}
