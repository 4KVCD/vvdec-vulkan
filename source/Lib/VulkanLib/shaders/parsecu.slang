// parse.slang's coding units: CABACReader::coding_unit and what it calls for the prediction (skip, the prediction
// mode, BDPCM, the intra modes, the prediction unit's merge data and motion vector differences, AMVR, BCW), the CU's
// creation (CodingStructure::addCU) and its record (GpuFrontEnd.cpp's writeCuRecords).

struct Cu
{
  int  slot;
  int  local;                  // its index among its CTU's CUs
  int  lx, ly, lw, lh;         // its luma block (lw 0: none)
  int  cx, cy, cw, chh;        // its chroma block, in chroma samples (cw 0: none)
  int  chType, treeType, modeType;
  int  predMode;
  uint flags;                  // CuRecord's kMerge, kSkip, ...
  int  mergeIdx, mmvdIdx, geoSplitDir, geoIdx0, geoIdx1, imv, bcw, interDir;
  int  mvpIdx0, mvpIdx1, refIdx0, refIdx1;
  int  qp, chromaQpAdj;
  int2 mv[6];                  // [list * 3 + k]: the motion vector differences as the parse leaves them
  int  isp, lfnst, mipT, mrl, bdpcmL, bdpcmC, sbtInfo, smvd;
  bool mip, act, rootCbf;
  int  planeCbf;               // a bit a component
  int  intraDirL, intraDirC, qtDepth, depth;
  uint2 left, above;           // the partitioner's neighbours (their word pairs: zero none)
  int  dmvrOff;                // DMVR's refinements' place in the CTU's (0: none)
  int  firstTu, numTus;        // its TUs among the CTU's
};

static Cu g_cu;

bool cuIsIntra()
{
  return g_cu.predMode == MODE_INTRA;
}

bool cuIsInter()
{
  return g_cu.predMode == MODE_INTER;
}

bool cuIsIbc()
{
  return g_cu.predMode == MODE_IBC;
}

bool cuIsSepTree()
{
  return g_cu.treeType != TREE_D;
}

bool cuSkip()
{
  return ( g_cu.flags & kSkip ) != 0;
}

bool cuMerge()
{
  return ( g_cu.flags & kMerge ) != 0;
}

bool cuAffine()
{
  return ( g_cu.flags & kAffine ) != 0;
}

void setFlag( uint f, bool on )
{
  g_cu.flags = on ? ( g_cu.flags | f ) : ( g_cu.flags & ~f );
}

// --------------------------------------------------------------------------------------------------------------------
// the CU's creation and record
// --------------------------------------------------------------------------------------------------------------------

// CodingStructure::addCU and what coding_tree sets: the CU of the partitioner's current area
bool addCu()
{
  if( g_numCus >= g_cuSlots )
  {
    g_error = true;
    return false;
  }
  const int4 a = currArea();
  bool       lumaValid   = true;
  bool       chromaValid = g_chromaFormat != 0;
  int        treeType    = g_p.treeType;
  if( g_p.chType == 1 )
  {
    lumaValid = false;
    treeType  = TREE_C;
  }
  else if( g_dualTree || g_p.treeType == TREE_L )
  {
    chromaValid = false;
    treeType    = TREE_L;
  }
  Cu cu;
  cu.slot     = g_ctu * g_cuSlots + g_numCus;
  cu.local    = g_numCus;
  g_numCus++;
  cu.lx       = lumaValid ? a.x : 0;
  cu.ly       = lumaValid ? a.y : 0;
  cu.lw       = lumaValid ? a.z : 0;
  cu.lh       = lumaValid ? a.w : 0;
  cu.cx       = chromaValid ? a.x >> g_csx : 0;
  cu.cy       = chromaValid ? a.y >> g_csy : 0;
  cu.cw       = chromaValid ? a.z >> g_csx : 0;
  cu.chh      = chromaValid ? a.w >> g_csy : 0;
  cu.chType   = g_p.chType;
  cu.treeType = treeType;
  cu.modeType = g_p.modeType;
  cu.predMode = MODE_INTER;
  cu.flags    = 0;
  cu.mergeIdx = 0;
  cu.mmvdIdx  = 0;
  cu.geoSplitDir = 0;
  cu.geoIdx0  = 0;
  cu.geoIdx1  = 0;
  cu.imv      = 0;
  cu.bcw      = 0;
  cu.interDir = 0;
  cu.mvpIdx0  = 0;
  cu.mvpIdx1  = 0;
  cu.refIdx0  = -1;   // CodingUnit::minInit
  cu.refIdx1  = -1;
  cu.qp       = 0;
  cu.chromaQpAdj = 0;
  for( int i = 0; i < 6; i++ )
  {
    cu.mv[i] = int2( 0, 0 );
  }
  cu.isp      = 0;
  cu.lfnst    = 0;
  cu.mipT     = 0;
  cu.mrl      = 0;
  cu.bdpcmL   = 0;
  cu.bdpcmC   = 0;
  cu.sbtInfo  = 0;
  cu.smvd     = 0;
  cu.mip      = false;
  cu.act      = false;
  cu.rootCbf  = false;
  cu.planeCbf = 0;
  cu.intraDirL = DC_IDX;   // CodingUnit::minInit
  cu.intraDirC = 0;
  cu.qtDepth  = g_p.currQtDepth;
  cu.depth    = g_p.currDepth;
  cu.left     = uni( s_lvLeft[g_p.top] );
  cu.above    = uni( s_lvAbove[g_p.top] );
  cu.dmvrOff  = 0;
  if( lumaValid && a.w >= 8 && a.z >= 8 && a.z * a.w >= 128 )
  {
    cu.dmvrOff = g_dmvrOffset;
    g_dmvrOffset += max( 1, a.z >> 4 ) * max( 1, a.w >> 4 );
  }
  cu.firstTu = g_numTus;
  cu.numTus  = 0;
  g_cu       = cu;
  return true;
}

// the CU (done) in the CTU's maps: the luma tree's (with its luma block), the chroma tree's (with its chroma block)
void writeCuInfo()
{
  const Cu   cu   = g_cu;
  const uint base = uint( cu.local + 1 ) | uint( cu.predMode ) << 12 | ( ( cu.flags & kSkip ) != 0 ? 1u << 14 : 0u )
                    | ( ( cu.flags & kAffine ) != 0 ? 1u << 15 : 0u ) | ( cu.mip ? 1u << 16 : 0u ) | uint( cu.isp ) << 17
                    | uint( cu.qtDepth ) << 19 | uint( cu.depth ) << 22;
  const uint y    = ( uint( cu.intraDirL ) & 0xffu ) | ( uint( cu.qp ) & 0xffu ) << 8;
  const int  x0   = g_ctuX << g_log2Ctu;
  const int  y0   = g_ctuY << g_log2Ctu;
  for( int ch = 0; ch < 2; ch++ )
  {
    const bool valid = ch == 0 ? cu.lw > 0 : cu.cw > 0;
    if( !valid )
    {
      continue;
    }
    // the block in luma samples, its size in the channel type's
    const int  bx = ch == 0 ? cu.lx : cu.cx << g_csx;
    const int  by = ch == 0 ? cu.ly : cu.cy << g_csy;
    const int  bw = ch == 0 ? cu.lw : cu.cw << g_csx;
    const int  bh = ch == 0 ? cu.lh : cu.chh << g_csy;
    const uint2 info = uint2( base | uint( log2i( ch == 0 ? cu.lw : cu.cw ) ) << 26 | uint( log2i( ch == 0 ? cu.lh : cu.chh ) ) << 29, y );
    const int  w4  = max( bw >> 2, 1 );
    const int  h4  = max( bh >> 2, 1 );
    const int  lw4 = firstbithigh( uint( w4 ) );
    const int  at  = ch * 1024 + ( ( by - y0 ) >> 2 ) * 32 + ( ( bx - x0 ) >> 2 );
    for( int u = g_lane; u < w4 * h4; u += 32 )
    {
      s_map[at + ( u >> lw4 ) * 32 + ( u & ( w4 - 1 ) )] = info;
    }
  }
  GroupMemoryBarrierWithGroupSync();
}

void writeCuRecord()
{
  const Cu  cu  = g_cu;
  uint*     r   = pc.cus + uint( cu.slot ) * 32u;
  uint      flags = cu.flags | ( cu.chType == 1 ? kChroma : 0u );
  r[0]  = uint( cu.lx ) | uint( cu.ly ) << 16;
  r[1]  = uint( cu.lw ) | uint( cu.lh ) << 8 | uint( cu.predMode ) << 16 | flags << 18;
  r[2]  = uint( cu.mergeIdx ) | uint( cu.mmvdIdx & 63 ) << 3 | uint( cu.geoSplitDir & 63 ) << 9 | uint( cu.geoIdx0 ) << 15 | uint( cu.geoIdx1 ) << 18
          | uint( cu.imv ) << 21 | uint( cu.bcw ) << 23 | uint( cu.interDir ) << 26 | uint( cu.mvpIdx0 & 1 ) << 28 | uint( cu.mvpIdx1 & 1 ) << 29;
  r[3]  = ( uint( cu.refIdx0 ) & 0xffu ) | ( uint( cu.refIdx1 ) & 0xffu ) << 8 | ( uint( cu.qp ) & 0xffu ) << 16 | ( uint( cu.chromaQpAdj ) & 0xffu ) << 24;
  for( int i = 0; i < 6; i++ )
  {
    r[4 + 2 * i]     = uint( cu.mv[i].x );
    r[4 + 2 * i + 1] = uint( cu.mv[i].y );
  }
  r[16] = uint( cu.firstTu ) | uint( cu.numTus ) << 16;
  r[17] = uint( cu.chType ) | uint( cu.treeType ) << 1 | uint( cu.modeType ) << 3 | uint( cu.isp ) << 5 | uint( cu.lfnst ) << 7 | ( cu.mip ? 1u << 9 : 0u )
          | uint( cu.mipT ) << 10 | uint( cu.mrl ) << 11 | ( cu.act ? 1u << 13 : 0u ) | uint( cu.bdpcmL ) << 14 | uint( cu.bdpcmC ) << 16
          | ( cu.rootCbf ? 1u << 18 : 0u ) | uint( cu.sbtInfo ) << 19 | uint( cu.planeCbf & 7 ) << 27;
  r[18] = ( uint( cu.intraDirL ) & 0xffu ) | ( uint( cu.intraDirC ) & 0xffu ) << 8 | uint( cu.qtDepth ) << 16 | uint( cu.depth ) << 20;
  r[19] = cu.cw > 0 ? uint( cu.cx ) | uint( cu.cy ) << 16 : 0u;
  r[20] = ( cu.cw > 0 ? uint( cu.cw ) | uint( cu.chh ) << 8 | 1u << 16 : 0u ) | ( cu.lw > 0 ? 1u << 17 : 0u );
  r[21] = uint( cu.dmvrOff ) | ( infoValid( cu.left ) ? 1u << 16 : 0u ) | ( infoValid( cu.above ) ? 1u << 17 : 0u ) | ( cu.smvd != 0 ? 1u << 18 : 0u );
  for( int w = 22; w < 32; w++ )
  {
    r[w] = 0;
  }
}

// CU::predictQP
int predictQp( int prevQp )
{
  const int ch     = g_cu.chType;
  const int x      = ch == 0 ? g_cu.lx : g_cu.cx << g_csx;
  const int y      = ch == 0 ? g_cu.ly : g_cu.cy << g_csy;
  const uint2 above = cuAt( x, y - stepY( ch ), ch );   // (the CTU above's: zero out of the slice or tile)
  const uint2 left  = cuAt( x - stepX( ch ), y, ch );
  const int   mask  = g_ctuSize - 1;
  if( g_tileFirstCol && ( x & mask ) == 0 && ( y & mask ) == 0 && infoValid( above ) )
  {
    return infoQp( above );
  }
  const int a = ( y & mask ) != 0 ? infoQp( above ) : prevQp;
  const int b = ( x & mask ) != 0 ? infoQp( left ) : prevQp;
  return ( a + b + 1 ) >> 1;
}

// QpParam( tu, comp, false ).Qp( false ): a TU's chroma QP
int chromaQp( int comp, int jointCbCr )
{
  const int jCbCr  = jointCbCr == 3 ? 3 : comp;   // TU::getICTMode( tu, false ) == 2
  int       offset = picWord( kPicCqpOffsets + jCbCr - 1 ) + sliceWord( kSlChromaQpDelta + jCbCr - 1 );
  if( g_cu.chromaQpAdj > 0 )
  {
    offset += picWord( kPicCqpOffsets + 3 + ( g_cu.chromaQpAdj - 1 ) * 3 + jCbCr - 1 );
  }
  const int qpi    = clip3( -g_qpBdOffset, MAX_QP, g_cu.qp );
  const int baseQp = uni( s_cqp[( jCbCr - 1 ) * 128 + qpi + g_qpBdOffset] );
  return clip3( 0, MAX_QP + g_qpBdOffset, baseQp + offset + g_qpBdOffset );
}

// coding_tree's CU: its creation, QP, coding_unit, the TUs' chroma QPs, the records
void codingUnitAtPart( inout CuCtx ctx )
{
  if( !addCu() )
  {
    return;
  }
  if( ctx.qgStart )
  {
    ctx.qgStart = false;
    ctx.qp      = predictQp( ctx.qp );
  }
  int        lumaQpLocal = -1;
  const bool sepChroma   = picWord( kPicUseDqp ) != 0 && isSepTreeP() && g_cu.chType == 1;
  if( sepChroma )
  {
    const uint2 colLuma = cuAt( ( g_cu.cx + ( g_cu.cw >> 1 ) ) << g_csx, ( g_cu.cy + ( g_cu.chh >> 1 ) ) << g_csy, 0 );
    lumaQpLocal         = ctx.qp;
    if( infoValid( colLuma ) )
    {
      ctx.qp = infoQp( colLuma );
    }
  }
  g_cu.qp          = ctx.qp;
  g_cu.chromaQpAdj = g_csChromaQpAdj;

  PROF_BEGIN( cuStart );
  codingUnit( ctx );
  PROF_END( kProfCu, cuStart );
  PROF_COUNT( kProfCus, 1 );
  g_binCount += 32u;   // (a CU's work besides its bins: about 32 bins')
  PROF_COUNT( kProfTus, g_cu.numTus );

  if( sepChroma )
  {
    ctx.qp = lumaQpLocal;
  }
  if( g_chromaFormat != 0 )
  {
    for( int t = 0; t < g_cu.numTus; t++ )
    {
      if( tuChromaValid( t ) )
      {
        const int joint = tuJoint( t );
        setTuWord( t, 5, uint( chromaQp( 1, joint ) ) & 0xffu | ( uint( chromaQp( 2, joint ) ) & 0xffu ) << 8 );
      }
    }
  }
  PROF_BEGIN( recStart );
  writeCuRecord();
  writeTuRecords();
  writeCuInfo();
  PROF_END( kProfRecords, recStart );
}

// --------------------------------------------------------------------------------------------------------------------
// coding_unit
// --------------------------------------------------------------------------------------------------------------------

bool isConsIntra()
{
  return g_cu.modeType == MODE_TYPE_INTRA;
}

bool isConsInter()
{
  return g_cu.modeType == MODE_TYPE_INTER;
}

int ctxLeftAbove( uint flag )
{
  // (the flags of the word pairs: skip, affine)
  const bool left  = flag == kSkip ? infoSkip( g_cu.left ) : infoAffine( g_cu.left );
  const bool above = flag == kSkip ? infoSkip( g_cu.above ) : infoAffine( g_cu.above );
  return ( left ? 1 : 0 ) + ( above ? 1 : 0 );
}

int ctxIbcFlag()
{
  int ctx = ( infoValid( g_cu.left ) && infoPredMode( g_cu.left ) == MODE_IBC ) ? 1 : 0;
  ctx += ( infoValid( g_cu.above ) && infoPredMode( g_cu.above ) == MODE_IBC ) ? 1 : 0;
  return ctx;
}

// CABACReader::cu_skip_flag
void cuSkipFlag()
{
  const bool ibcFlag = tool( kToolIbc ) && g_cu.lw <= 64 && g_cu.lh <= 64;
  if( ( sliceIsIntra() || isConsIntra() ) && ibcFlag )
  {
    if( decodeBin( CTX_SkipFlag + ctxLeftAbove( kSkip ) ) != 0 )
    {
      setFlag( kSkip, true );
      g_cu.predMode = MODE_IBC;
    }
    return;
  }
  if( !ibcFlag && ( ( g_cu.lw == 4 && g_cu.lh == 4 ) || isConsIntra() ) )
  {
    return;
  }
  const bool skip = decodeBin( CTX_SkipFlag + ctxLeftAbove( kSkip ) ) != 0;
  if( skip && ibcFlag )
  {
    if( !isConsInter() )
    {
      if( g_cu.lw == 4 && g_cu.lh == 4 )
      {
        setFlag( kSkip, true );
        g_cu.predMode = MODE_IBC;
        return;
      }
      if( decodeBin( CTX_IBCFlag + ctxIbcFlag() ) != 0 )
      {
        setFlag( kSkip, true );
        g_cu.predMode = MODE_IBC;
      }
    }
  }
  if( skip )
  {
    setFlag( kSkip, true );
  }
}

// CABACReader::pred_mode
void predMode()
{
  if( isConsInter() )
  {
    return;
  }
  bool ibcAllowed = false;
  if( sliceIsIntra() || ( g_cu.lw == 4 && g_cu.lh == 4 ) || isConsIntra() )
  {
    ibcAllowed    = true;
    g_cu.predMode = MODE_INTRA;
  }
  else
  {
    int ctx = ( ( infoValid( g_cu.above ) && infoPredMode( g_cu.above ) == MODE_INTRA ) || ( infoValid( g_cu.left ) && infoPredMode( g_cu.left ) == MODE_INTRA ) ) ? 1 : 0;
    if( decodeBin( CTX_PredMode + ctx ) != 0 )
    {
      g_cu.predMode = MODE_INTRA;
    }
    else
    {
      ibcAllowed = true;
    }
  }
  ibcAllowed = ibcAllowed && g_cu.chType == 0 && tool( kToolIbc ) && g_cu.lw <= 64 && g_cu.lh <= 64;
  if( ibcAllowed )
  {
    if( decodeBin( CTX_IBCFlag + ctxIbcFlag() ) != 0 )
    {
      g_cu.predMode = MODE_IBC;
    }
  }
}

// CABACReader::adaptive_color_transform
void adaptiveColorTransform()
{
  if( !tool( kToolAct ) || cuIsSepTree() )
  {
    return;
  }
  g_cu.act = decodeBin( CTX_ACTFlag ) != 0;
}

// CABACReader::bdpcm_mode
void bdpcmMode( int comp )
{
  const int  tsMax   = 1 << g_log2MaxTs;
  const int  w       = comp == 0 ? g_cu.lw : g_cu.cw;
  const int  h       = comp == 0 ? g_cu.lh : g_cu.chh;
  const bool allowed = tool( kToolBdpcm ) && ( comp == 0 || !g_cu.act ) && w <= tsMax && h <= tsMax;
  if( !allowed )
  {
    return;
  }
  const int ctxId = comp == 0 ? 0 : 2;
  int       mode  = int( decodeBin( CTX_BDPCMMode + ctxId ) );
  if( mode != 0 )
  {
    mode += int( decodeBin( CTX_BDPCMMode + ctxId + 1 ) );
  }
  if( comp == 0 )
  {
    g_cu.bdpcmL = mode;
  }
  else
  {
    g_cu.bdpcmC = mode;
  }
}

// CABACReader::coding_unit with cu_pred_data (the skip's prediction unit and cu_pred_data's one call)
void codingUnit( inout CuCtx ctx )
{
  bool skipped = false;
  if( !sliceIsIntra() || tool( kToolIbc ) )
  {
    if( g_cu.lw > 0 )
    {
      cuSkipFlag();
    }
    if( cuSkip() )
    {
      g_cu.act = false;
      addEmptyTUs();
      skipped = true;
    }
    else
    {
      predMode();
    }
  }
  else
  {
    g_cu.predMode = MODE_INTRA;
  }
  if( !skipped && cuIsIntra() )
  {
    adaptiveColorTransform();
    if( g_cu.chType == 0 )
    {
      bdpcmMode( 0 );
      intraLumaPredMode();
    }
    if( ( g_cu.chType == 1 || !cuIsSepTree() ) && g_chromaFormat != 0 )
    {
      bdpcmMode( 1 );
      intraChromaPredMode();
    }
  }
  else if( !skipped && g_cu.lw == 0 )
  {
    g_cu.predMode = MODE_IBC;   // a dual tree's chroma CU
  }
  else
  {
    predictionUnit();
    if( !skipped && !cuMerge() )
    {
      if( cuAffine() )
      {
        affineAmvrMode();
      }
      else
      {
        amvrMode();
      }
      cuBcwFlag();
    }
  }
  if( !skipped )
  {
    cuResidual( ctx );
  }
}

// --------------------------------------------------------------------------------------------------------------------
// intra
// --------------------------------------------------------------------------------------------------------------------

// CABACReader::mip_flag, mip_pred_mode
void mipFlag()
{
  if( !tool( kToolMip ) )
  {
    return;
  }
  int ctxId = infoMip( g_cu.left ) ? 1 : 0;
  ctxId += infoMip( g_cu.above ) ? 1 : 0;
  ctxId = ( g_cu.lw > 2 * g_cu.lh || g_cu.lh > 2 * g_cu.lw ) ? 3 : ctxId;
  g_cu.mip = decodeBin( CTX_MipFlag + ctxId ) != 0;
}

void mipPredMode()
{
  g_cu.mipT          = int( decodeBinEP() );
  const int sizeId   = ( g_cu.lw == 4 && g_cu.lh == 4 ) ? 0 : ( ( g_cu.lw == 4 || g_cu.lh == 4 || ( g_cu.lw == 8 && g_cu.lh == 8 ) ) ? 1 : 2 );
  const int numModes = sizeId == 0 ? 16 : ( sizeId == 1 ? 8 : 6 );
  g_cu.intraDirL     = readTruncBinCode( numModes );
}

// CABACReader::extend_ref_line
void extendRefLine()
{
  if( g_cu.bdpcmL != 0 || !tool( kToolMrl ) )
  {
    return;
  }
  if( ( g_cu.ly & ( g_ctuSize - 1 ) ) == 0 )
  {
    return;
  }
  int mrl = decodeBin( CTX_MultiRefLineIdx + 0 ) != 0 ? 1 : 0;
  if( mrl != 0 )
  {
    mrl = decodeBin( CTX_MultiRefLineIdx + 1 ) != 0 ? 2 : 1;
  }
  g_cu.mrl = mrl;
}

// CU::canUseISPSplit
int canUseIspSplit( int w, int h )
{
  const bool notEnoughSamplesToSplit   = log2i( w ) + log2i( h ) <= 4;
  const bool cuSizeLargerThanMaxTrSize = w > g_maxTbSize || h > g_maxTbSize;
  const int  widthCanBeUsed            = ( !cuSizeLargerThanMaxTrSize && !notEnoughSamplesToSplit ) ? 4 : 2;
  const int  heightCanBeUsed           = ( !cuSizeLargerThanMaxTrSize && !notEnoughSamplesToSplit ) ? 0 : 2;
  return widthCanBeUsed >> heightCanBeUsed;
}

// CABACReader::isp_mode
void ispMode()
{
  if( g_cu.mrl != 0 || !tool( kToolIsp ) || g_cu.bdpcmL != 0 || g_cu.act )
  {
    return;
  }
  const int allowedSplits = canUseIspSplit( g_cu.lw, g_cu.lh );
  if( allowedSplits == NOT_INTRA_SUBPARTITIONS )
  {
    return;
  }
  if( decodeBin( CTX_ISPMode + 0 ) != 0 )
  {
    if( allowedSplits == HOR_INTRA_SUBPARTITIONS )
    {
      g_cu.isp = HOR_INTRA_SUBPARTITIONS;
    }
    else if( allowedSplits == VER_INTRA_SUBPARTITIONS )
    {
      g_cu.isp = VER_INTRA_SUBPARTITIONS;
    }
    else
    {
      g_cu.isp = 1 + int( decodeBin( CTX_ISPMode + 1 ) );
    }
  }
}

// PU::getIntraDirLuma of a neighbour
int intraDirLumaOf( uint2 info )
{
  return infoMip( info ) ? PLANAR_IDX : infoIntraDir( info );
}

// PU::getIntraMPMs
void getIntraMpms( out int mpm[6] )
{
  int leftIntraDir  = PLANAR_IDX;
  int aboveIntraDir = PLANAR_IDX;
  const int x = g_cu.lx, y = g_cu.ly, w = g_cu.lw, h = g_cu.lh;
  const uint2 cuLeft = cuAtRestricted( x - 1, y + h - 1, g_cu.local, 0 );
  if( infoValid( cuLeft ) && infoPredMode( cuLeft ) == MODE_INTRA )
  {
    leftIntraDir = intraDirLumaOf( cuLeft );
  }
  const uint2 cuAbove = cuAtRestricted( x + w - 1, y - 1, g_cu.local, 0 );
  if( infoValid( cuAbove ) && infoPredMode( cuAbove ) == MODE_INTRA && ( y & ( g_ctuSize - 1 ) ) != 0 )
  {
    aboveIntraDir = intraDirLumaOf( cuAbove );
  }
  const int offset = NUM_LUMA_MODE - 6;
  const int mod    = offset + 3;
  mpm[0] = PLANAR_IDX;
  mpm[1] = DC_IDX;
  mpm[2] = VER_IDX;
  mpm[3] = HOR_IDX;
  mpm[4] = VER_IDX - 4;
  mpm[5] = VER_IDX + 4;
  if( leftIntraDir == aboveIntraDir )
  {
    if( leftIntraDir > DC_IDX )
    {
      mpm[0] = PLANAR_IDX;
      mpm[1] = leftIntraDir;
      mpm[2] = ( ( leftIntraDir + offset ) % mod ) + 2;
      mpm[3] = ( ( leftIntraDir - 1 ) % mod ) + 2;
      mpm[4] = ( ( leftIntraDir + offset - 1 ) % mod ) + 2;
      mpm[5] = ( leftIntraDir % mod ) + 2;
    }
  }
  else
  {
    if( leftIntraDir > DC_IDX && aboveIntraDir > DC_IDX )
    {
      mpm[0] = PLANAR_IDX;
      mpm[1] = leftIntraDir;
      mpm[2] = aboveIntraDir;
      const int maxCand = max( mpm[1], mpm[2] );
      const int minCand = min( mpm[1], mpm[2] );
      if( maxCand - minCand == 1 )
      {
        mpm[3] = ( ( minCand + offset ) % mod ) + 2;
        mpm[4] = ( ( maxCand - 1 ) % mod ) + 2;
        mpm[5] = ( ( minCand + offset - 1 ) % mod ) + 2;
      }
      else if( maxCand - minCand >= 62 )
      {
        mpm[3] = ( ( minCand - 1 ) % mod ) + 2;
        mpm[4] = ( ( maxCand + offset ) % mod ) + 2;
        mpm[5] = ( minCand % mod ) + 2;
      }
      else if( maxCand - minCand == 2 )
      {
        mpm[3] = ( ( minCand - 1 ) % mod ) + 2;
        mpm[4] = ( ( minCand + offset ) % mod ) + 2;
        mpm[5] = ( ( maxCand - 1 ) % mod ) + 2;
      }
      else
      {
        mpm[3] = ( ( minCand + offset ) % mod ) + 2;
        mpm[4] = ( ( minCand - 1 ) % mod ) + 2;
        mpm[5] = ( ( maxCand + offset ) % mod ) + 2;
      }
    }
    else if( leftIntraDir + aboveIntraDir >= 2 )
    {
      const int maxCand = max( leftIntraDir, aboveIntraDir );
      mpm[0] = PLANAR_IDX;
      mpm[1] = maxCand;
      mpm[2] = ( ( maxCand + offset ) % mod ) + 2;
      mpm[3] = ( ( maxCand - 1 ) % mod ) + 2;
      mpm[4] = ( ( maxCand + offset - 1 ) % mod ) + 2;
      mpm[5] = ( maxCand % mod ) + 2;
    }
  }
}

// CABACReader::intra_luma_pred_mode
void intraLumaPredMode()
{
  if( g_cu.bdpcmL != 0 )
  {
    g_cu.intraDirL = g_cu.bdpcmL == 2 ? VER_IDX : HOR_IDX;
    return;
  }
  mipFlag();
  if( g_cu.mip )
  {
    mipPredMode();
    return;
  }
  extendRefLine();
  ispMode();
  const bool mpmFlag = g_cu.mrl != 0 ? true : decodeBin( CTX_IPredMode_0 ) != 0;
  int mpm[6];
  getIntraMpms( mpm );
  if( mpmFlag )
  {
    int ipredIdx = 0;
    if( g_cu.mrl == 0 )
    {
      ipredIdx = int( decodeBin( CTX_IntraLumaPlanarFlag + ( g_cu.isp == NOT_INTRA_SUBPARTITIONS ? 1 : 0 ) ) );
    }
    else
    {
      ipredIdx = 1;
    }
    if( ipredIdx != 0 )
    {
      while( ipredIdx < 5 && decodeBinEP() != 0 )
      {
        ipredIdx++;
      }
    }
    g_cu.intraDirL = mpm[ipredIdx];
  }
  else
  {
    int ipredMode = readTruncBinCode( NUM_LUMA_MODE - 6 );
    // the MPMs sorted
    for( int i = 1; i < 6; i++ )
    {
      for( int j = i; j > 0 && mpm[j - 1] > mpm[j]; j-- )
      {
        const int t = mpm[j];
        mpm[j]      = mpm[j - 1];
        mpm[j - 1]  = t;
      }
    }
    for( int i = 0; i < 6; i++ )
    {
      ipredMode += ipredMode >= mpm[i] ? 1 : 0;
    }
    g_cu.intraDirL = ipredMode;
  }
}

// the luma CU of the chroma block's centre (PU::getCoLocatedLumaPU of a separate tree's CU)
uint2 coLocatedLumaCu()
{
  return cuAt( ( g_cu.cx << g_csx ) + ( ( g_cu.cw << g_csx ) >> 1 ), ( g_cu.cy << g_csy ) + ( ( g_cu.chh << g_csy ) >> 1 ), 0 );
}

// CU::checkCCLMAllowed
bool checkCclmAllowed()
{
  if( !g_dualTree )
  {
    return true;
  }
  if( g_ctuSize <= 32 )
  {
    return true;
  }
  const int depthFor64x64Node = g_ctuSize == 128 ? 1 : 0;
  const int split1            = splitAtDepth( depthFor64x64Node );
  const int split2            = splitAtDepth( depthFor64x64Node + 1 );
  bool      allow             = false;
  if( split1 == CU_QUAD_SPLIT || ( split1 == CU_HORZ_SPLIT && split2 == CU_VERT_SPLIT ) )
  {
    allow = true;
  }
  else if( split1 == CU_DONT_SPLIT )
  {
    allow = true;
  }
  else if( split1 == CU_HORZ_SPLIT && split2 == CU_DONT_SPLIT )
  {
    allow = true;
  }
  if( allow )
  {
    const uint2 colLuma = cuAt( g_cu.cx << g_csx, g_cu.cy << g_csy, 0 );
    if( infoDepth( colLuma ) > depthFor64x64Node && infoQtDepth( colLuma ) == depthFor64x64Node )
    {
      allow = false;
    }
    else if( infoDepth( colLuma ) == depthFor64x64Node && infoIsp( colLuma ) != 0 )
    {
      allow = false;
    }
  }
  return allow;
}

// CABACReader::intra_chroma_pred_mode
void intraChromaPredMode()
{
  if( g_cu.bdpcmC != 0 )
  {
    g_cu.intraDirC = g_cu.bdpcmC == 2 ? VER_IDX : HOR_IDX;
    return;
  }
  if( g_cu.act )
  {
    g_cu.intraDirC = DM_CHROMA_IDX;
    return;
  }
  if( tool( kToolLmChroma ) && checkCclmAllowed() )
  {
    if( decodeBin( CTX_CclmModeFlag ) != 0 )
    {
      // intra_chroma_lmc_mode
      int symbol = int( decodeBin( CTX_CclmModeIdx ) );
      if( symbol != 0 )
      {
        symbol += int( decodeBinEP() );
      }
      g_cu.intraDirC = symbol == 0 ? LM_CHROMA_IDX : ( symbol == 1 ? MDLM_L_IDX : MDLM_T_IDX );
      return;
    }
  }
  if( decodeBin( CTX_IPredMode_1 ) == 0 )
  {
    g_cu.intraDirC = DM_CHROMA_IDX;
    return;
  }
  const int candId = int( decodeBinsEP( 2 ) );
  // PU::getIntraChromaCandModes: planar, vertical, horizontal, DC (the colocated luma mode's place: VDIA)
  int modeList[4] = { PLANAR_IDX, VER_IDX, HOR_IDX, DC_IDX };
  const bool dmChromaMip = !cuIsSepTree() && g_chromaFormat == 3 && g_cu.mip;
  if( !dmChromaMip )
  {
    int lumaMode;
    if( cuIsSepTree() )
    {
      lumaMode = intraDirLumaOf( coLocatedLumaCu() );
    }
    else
    {
      lumaMode = g_cu.mip ? PLANAR_IDX : g_cu.intraDirL;
    }
    for( int i = 0; i < 4; i++ )
    {
      if( lumaMode == modeList[i] )
      {
        modeList[i] = VDIA_IDX;
        break;
      }
    }
  }
  g_cu.intraDirC = modeList[candId];
}

// --------------------------------------------------------------------------------------------------------------------
// inter: the prediction unit
// --------------------------------------------------------------------------------------------------------------------

// CABACReader::mvd_coding
int2 mvdCoding()
{
  int horAbs = int( decodeBin( CTX_Mvd ) );
  int verAbs = int( decodeBin( CTX_Mvd ) );
  if( horAbs != 0 )
  {
    horAbs += int( decodeBin( CTX_Mvd + 1 ) );
  }
  if( verAbs != 0 )
  {
    verAbs += int( decodeBin( CTX_Mvd + 1 ) );
  }
  if( horAbs != 0 )
  {
    if( horAbs > 1 )
    {
      horAbs += int( decodeRemAbsEP( 1, 0, MV_BITS - 1 ) );
    }
    if( decodeBinEP() != 0 )
    {
      horAbs = -horAbs;
    }
  }
  if( verAbs != 0 )
  {
    if( verAbs > 1 )
    {
      verAbs += int( decodeRemAbsEP( 1, 0, MV_BITS - 1 ) );
    }
    if( decodeBinEP() != 0 )
    {
      verAbs = -verAbs;
    }
  }
  return int2( horAbs, verAbs );
}

int numRefIdx( int list )
{
  return ( sliceWord( kSlNumRef ) >> ( 8 * list ) ) & 0xff;
}

int symRefIdx( int list )
{
  return int( int8_t( ( sliceWord( kSlSymRef ) >> ( 8 * list ) ) & 0xff ) );
}

// CABACReader::ref_idx
int refIdxOf( int list )
{
  if( g_cu.smvd != 0 )
  {
    return symRefIdx( list );
  }
  const int numRef = numRefIdx( list );
  if( numRef <= 1 || decodeBin( CTX_RefPic + 0 ) == 0 )
  {
    return 0;
  }
  if( numRef <= 2 || decodeBin( CTX_RefPic + 1 ) == 0 )
  {
    return 1;
  }
  for( int idx = 3;; idx++ )
  {
    if( numRef <= idx || decodeBinEP() == 0 )
    {
      return idx - 1;
    }
  }
  return 0;
}

// CABACReader::inter_pred_idc
void interPredIdc()
{
  if( g_sliceType == P_SLICE )
  {
    g_cu.interDir = 1;
    return;
  }
  if( !( g_cu.lw + g_cu.lh <= 12 ) )   // PU::isBipredRestriction
  {
    const int ctxId = 7 - ( ( log2i( g_cu.lw ) + log2i( g_cu.lh ) + 1 ) >> 1 );
    if( decodeBin( CTX_InterDir + ctxId ) != 0 )
    {
      g_cu.interDir = 3;
      return;
    }
  }
  if( decodeBin( CTX_InterDir + 5 ) != 0 )
  {
    g_cu.interDir = 2;
    return;
  }
  g_cu.interDir = 1;
}

// CABACReader::affine_flag
void affineFlag()
{
  if( tool( kToolAffine ) && g_cu.lw >= 16 && g_cu.lh >= 16 )
  {
    setFlag( kAffine, decodeBin( CTX_AffineFlag + ctxLeftAbove( kAffine ) ) != 0 );
    if( cuAffine() && tool( kToolAffineType ) )
    {
      setFlag( kAffine6, decodeBin( CTX_AffineType ) != 0 );
    }
  }
}

// CABACReader::smvd_mode
void smvdMode()
{
  if( g_cu.interDir != 3 || cuAffine() || !tool( kToolSmvd ) || sliceFlag( kSlMvdL1Zero ) )
  {
    return;
  }
  if( !sliceFlag( kSlBiDirPred ) )
  {
    return;
  }
  g_cu.smvd = decodeBin( CTX_SmvdFlag ) != 0 ? 1 : 0;
}

// CABACReader::subblock_merge_flag
void subblockMergeFlag()
{
  if( !sliceIsIntra() && sliceWord( kSlMaxAffMerge ) > 0 && g_cu.lw >= 8 && g_cu.lh >= 8 )
  {
    setFlag( kAffine, decodeBin( CTX_SubblockMergeFlag + ctxLeftAbove( kAffine ) ) != 0 );
  }
}

// CABACReader::merge_idx
void mergeIdx()
{
  if( ( g_cu.flags & kGeo ) != 0 )
  {
    g_cu.geoSplitDir        = readTruncBinCode( 64 );
    const int maxNumGeoCand = picWord( kPicMaxGeoCand );
    const int numCandminus2 = maxNumGeoCand - 2;
    int       mergeCand0    = 0;
    int       mergeCand1    = 0;
    if( decodeBin( CTX_MergeIdx ) != 0 )
    {
      mergeCand0 += unaryMaxEqprob( numCandminus2 ) + 1;
    }
    if( numCandminus2 > 0 )
    {
      if( decodeBin( CTX_MergeIdx ) != 0 )
      {
        mergeCand1 += unaryMaxEqprob( numCandminus2 - 1 ) + 1;
      }
    }
    mergeCand1 += mergeCand1 >= mergeCand0 ? 1 : 0;
    g_cu.geoIdx0 = mergeCand0;
    g_cu.geoIdx1 = mergeCand1;
    return;
  }
  int numCandminus1;
  int ctxIdx = CTX_MergeIdx;
  if( cuIsIbc() )
  {
    numCandminus1 = picWord( kPicMaxIbcCand ) - 1;
  }
  else if( cuAffine() )
  {
    numCandminus1 = sliceWord( kSlMaxAffMerge ) - 1;
    ctxIdx        = CTX_AffMergeIdx;
  }
  else
  {
    numCandminus1 = picWord( kPicMaxMergeCand ) - 1;
  }
  int idx = 0;
  if( numCandminus1 > 0 && decodeBin( ctxIdx ) != 0 )
  {
    for( idx++; idx < numCandminus1 && decodeBinEP() != 0; idx++ )
    {
    }
  }
  g_cu.mergeIdx = idx;
}

// CABACReader::mmvd_merge_idx
void mmvdMergeIdx()
{
  int       var0 = 0, var1 = 0, var2 = 0;
  const int numCand            = picWord( kPicMaxMergeCand );
  const int numCandminus1Base = numCand > 1 ? 1 : 0;
  if( numCandminus1Base > 0 && decodeBin( CTX_MmvdMergeIdx ) != 0 )
  {
    for( var0++; var0 < numCandminus1Base && decodeBinEP() != 0; var0++ )
    {
    }
  }
  const int numCandminus1Step = 7;
  if( decodeBin( CTX_MmvdStepMvpIdx ) != 0 )
  {
    for( var1++; var1 < numCandminus1Step && decodeBinEP() != 0; var1++ )
    {
    }
  }
  if( decodeBinEP() != 0 )
  {
    var2 += 2;
  }
  if( decodeBinEP() != 0 )
  {
    var2 += 1;
  }
  g_cu.mmvdIdx = var0 * 32 + var1 * 4 + var2;
}

// CABACReader::merge_data
void mergeData()
{
  if( cuIsIbc() )
  {
    mergeIdx();
    return;
  }
  subblockMergeFlag();
  if( cuAffine() )
  {
    mergeIdx();
    return;
  }
  bool       regularMerge  = true;
  const bool ciipAvailable = tool( kToolCiip ) && !cuSkip() && g_cu.lw < 128 && g_cu.lh < 128 && g_cu.lw * g_cu.lh >= 64;
  const bool geoAvailable  = tool( kToolGeo ) && g_sliceType == B_SLICE && g_cu.lw >= 8 && g_cu.lh >= 8 && g_cu.lw <= 64 && g_cu.lh <= 64
                            && g_cu.lw < 8 * g_cu.lh && g_cu.lh < 8 * g_cu.lw;
  if( geoAvailable || ciipAvailable )
  {
    regularMerge = decodeBin( CTX_RegularMergeFlag + ( cuSkip() ? 0 : 1 ) ) != 0;
  }
  if( regularMerge )
  {
    if( tool( kToolMmvd ) )
    {
      setFlag( kMmvd, decodeBin( CTX_MmvdFlag ) != 0 );
    }
  }
  else
  {
    if( geoAvailable && ciipAvailable )
    {
      setFlag( kCiip, decodeBin( CTX_CiipFlag ) != 0 );
    }
    else if( ciipAvailable )
    {
      setFlag( kCiip, true );
    }
    if( ( g_cu.flags & kCiip ) != 0 )
    {
      g_cu.intraDirL = PLANAR_IDX;
      g_cu.intraDirC = DM_CHROMA_IDX;
    }
    else
    {
      setFlag( kGeo, true );
    }
  }
  if( ( g_cu.flags & kMmvd ) != 0 )
  {
    mmvdMergeIdx();
  }
  else
  {
    mergeIdx();
  }
}

// CABACReader::prediction_unit
void predictionUnit()
{
  if( cuSkip() )
  {
    setFlag( kMerge, true );
  }
  else
  {
    setFlag( kMerge, decodeBin( CTX_MergeFlag ) != 0 );
  }
  if( cuMerge() )
  {
    mergeData();
  }
  else
  {
    // IBC's AMVP (list 0, one difference) or inter's: each list's reference index, differences and predictor (one call
    // of mvdCoding)
    const bool ibc = cuIsIbc();
    if( ibc )
    {
      g_cu.interDir = 1;
      g_cu.refIdx0  = MAX_NUM_REF;
    }
    else
    {
      interPredIdc();
      affineFlag();
      smvdMode();
    }
    const int numMvd = ibc ? 1 : ( cuAffine() ? ( ( g_cu.flags & kAffine6 ) != 0 ? 3 : 2 ) : 1 );
    for( int list = 0; list < 2; list++ )
    {
      if( list == 0 ? g_cu.interDir == 2 : ( ibc || g_cu.interDir == 1 ) )
      {
        continue;
      }
      bool mvds = true;
      if( !ibc )
      {
        if( list == 0 )
        {
          g_cu.refIdx0 = refIdxOf( 0 );
        }
        else if( g_cu.smvd != 1 )
        {
          g_cu.refIdx1 = refIdxOf( 1 );
          mvds         = !( sliceFlag( kSlMvdL1Zero ) && g_cu.interDir == 3 );
        }
        else
        {
          mvds = false;
        }
      }
      for( int k = 0; mvds && k < numMvd; k++ )
      {
        const int2 mvd = mvdCoding();
        if( list == 0 )
        {
          g_cu.mv[k] = mvd;
        }
        else
        {
          g_cu.mv[3 + k] = mvd;
        }
      }
      const int mvp = ibc && picWord( kPicMaxIbcCand ) == 1 ? 0 : int( decodeBin( CTX_MVPIdx ) );
      if( list == 0 )
      {
        g_cu.mvpIdx0 = mvp;
      }
      else
      {
        g_cu.mvpIdx1 = mvp;
      }
    }
  }
  if( g_cu.smvd != 0 )
  {
    const int cur = g_cu.smvd - 1;   // the list coded
    if( cur == 0 )
    {
      g_cu.mv[3]   = -g_cu.mv[0];
      g_cu.refIdx1 = symRefIdx( 1 );
    }
    else
    {
      g_cu.mv[0]   = -g_cu.mv[3];
      g_cu.refIdx0 = symRefIdx( 0 );
    }
  }
}

// CU::hasSubCUNonZeroMVd
bool hasSubCuNonZeroMvd()
{
  bool nonZero = false;
  if( g_cu.interDir != 2 )
  {
    nonZero = nonZero || g_cu.mv[0].x != 0 || g_cu.mv[0].y != 0;
  }
  if( g_cu.interDir != 1 )
  {
    if( !sliceFlag( kSlMvdL1Zero ) || g_cu.interDir != 3 )
    {
      nonZero = nonZero || g_cu.mv[3].x != 0 || g_cu.mv[3].y != 0;
    }
  }
  return nonZero;
}

// CU::hasSubCUNonZeroAffineMVd
bool hasSubCuNonZeroAffineMvd()
{
  const int n       = ( g_cu.flags & kAffine6 ) != 0 ? 3 : 2;
  bool      nonZero = false;
  if( g_cu.interDir != 2 )
  {
    for( int i = 0; !nonZero && i < n; i++ )
    {
      nonZero = g_cu.mv[i].x != 0 || g_cu.mv[i].y != 0;
    }
  }
  if( !nonZero && g_cu.interDir != 1 )
  {
    if( !sliceFlag( kSlMvdL1Zero ) || g_cu.interDir != 3 )
    {
      for( int i = 0; !nonZero && i < n; i++ )
      {
        nonZero = g_cu.mv[3 + i].x != 0 || g_cu.mv[3 + i].y != 0;
      }
    }
  }
  return nonZero;
}

// CABACReader::amvr_mode
void amvrMode()
{
  if( !tool( kToolAmvr ) || !hasSubCuNonZeroMvd() )
  {
    return;
  }
  uint value = cuIsIbc() ? 1u : decodeBin( CTX_ImvFlag + 0 );
  if( value != 0 )
  {
    g_cu.imv = int( value );
    if( !cuIsIbc() )
    {
      value    = decodeBin( CTX_ImvFlag + 4 );
      g_cu.imv = value != 0 ? 1 : IMV_HPEL;
    }
    if( value != 0 )
    {
      value = decodeBin( CTX_ImvFlag + 1 );
      value++;
      g_cu.imv = int( value );
    }
  }
}

// CABACReader::affine_amvr_mode
void affineAmvrMode()
{
  if( !tool( kToolAffineAmvr ) || !hasSubCuNonZeroAffineMvd() )
  {
    return;
  }
  uint value = decodeBin( CTX_ImvFlag + 2 );
  if( value != 0 )
  {
    value = decodeBin( CTX_ImvFlag + 3 );
    value++;
  }
  g_cu.imv = int( value );
}

// CU::isBcwIdxCoded
bool isBcwIdxCoded()
{
  if( !tool( kToolBcw ) )
  {
    return false;
  }
  if( g_cu.predMode == MODE_IBC || g_cu.predMode == MODE_INTRA || g_sliceType == P_SLICE || g_cu.interDir != 3 )
  {
    return false;
  }
  if( g_cu.lw * g_cu.lh < 256 )
  {
    return false;
  }
  const uint wp = uint( sliceWord( kSlWp ) );
  // Slice::getWpScaling: a negative reference index takes index 0's
  const int r0 = max( g_cu.refIdx0, 0 );
  const int r1 = max( g_cu.refIdx1, 0 );
  if( ( ( wp >> uint( r0 ) ) & 1u ) != 0 || ( ( wp >> uint( 16 + r1 ) ) & 1u ) != 0 )
  {
    return false;
  }
  return true;
}

// CABACReader::cu_bcw_flag
void cuBcwFlag()
{
  if( !isBcwIdxCoded() )
  {
    return;
  }
  int        idx    = 0;
  const uint symbol = decodeBin( CTX_BcwIdx );
  const int  numBcw = sliceFlag( kSlCheckLDC ) ? 5 : 3;
  if( symbol == 1 )
  {
    const int prefixNumBits = numBcw - 2;
    idx                     = 1;
    for( int ui = 0; ui < prefixNumBits && decodeBinEP() != 0; ++ui )
    {
      idx++;
    }
  }
  const int bcwIdx = kBcwParsingOrder[idx];
  g_cu.bcw         = kBcwInternFwd[bcwIdx];   // CU::setBcwIdx: an AMVP bi-predicted CU's
}
