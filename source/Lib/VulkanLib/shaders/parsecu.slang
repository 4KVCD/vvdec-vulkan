// The coding unit part of parse.slang, which includes this file. For each leaf of the coding tree it creates the CU
// (CodingStructure::addCU) and predicts its QP, parses CABACReader::coding_unit with the prediction syntax it calls
// (skip, the prediction mode, BDPCM, the intra luma and chroma modes, the prediction unit's merge data and motion
// vector differences, AMVR, BCW), then stages the CU's record (laid out as GpuFrontEnd.cpp's writeCuRecords writes
// it) and its TUs' records, adds up the room the front end's job lists need for them, and writes the CU's entries in
// the maps of parsed CUs. The residual syntax is in parsetu.slang.

// The CU being parsed: the fields of its record (GpuFrontEnd.h's CuRecord), plus what the parse needs on the way
struct Cu
{
  int  local;                  // its index among its CTU's CUs
  int  lx, ly, lw, lh;         // its luma block, in luma samples (lw 0: none)
  int  cx, cy, cw, chh;        // its chroma block, in chroma samples (cw 0: none)
  int  chType, treeType, modeType;
  int  predMode;               // vvdec's PredMode: MODE_INTER, MODE_INTRA or MODE_IBC
  uint flags;                  // CuRecord's kMerge, kSkip, ...
  int  mergeIdx, mmvdIdx, geoSplitDir, geoIdx0, geoIdx1, imv, bcw, interDir;
  int  mvpIdx0, mvpIdx1, refIdx0, refIdx1;
  int  qp, chromaQpAdj;
  int2 mv[6];                  // [list * 3 + k]: the motion vector differences as parsed (k: affine control point)
  int  isp, lfnst, mipT, mrl, bdpcmL, bdpcmC, sbtInfo, smvd;
  bool mip, act, rootCbf;
  int  planeCbf;               // the coded block flags of its planes, a bit per component
  int  intraDirL, intraDirC, qtDepth, depth;
  uint2 left, above;           // the partitioner's neighbours, as map entries (zero: none)
  // Where its DMVR refinements start among the CTU's (luma CUs of at least 8x8 and 128 samples reserve room). Other
  // CUs keep 0, which is also where the CTU's first CU with room starts.
  int  dmvrOff;
  int  firstTu, numTus;        // its TUs among the CTU's
};

static Cu g_cu;

// The CU's prediction mode, tree and flags
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

// Sets or clears the record flag f
void setFlag( uint f, bool on )
{
  g_cu.flags = on ? ( g_cu.flags | f ) : ( g_cu.flags & ~f );
}

// --------------------------------------------------------------------------------------------------------------------
// The CU's creation and its record
// --------------------------------------------------------------------------------------------------------------------

// Starts the CU of the partitioner's current area, as CodingStructure::addCU and coding_tree set it up: its luma and
// chroma blocks (a chroma tree's CU has no luma block, a luma tree's CU no chroma block), its index in the CTU, its
// neighbours and its place among the CTU's DMVR refinements. Returns false, giving up the substream, when the CTU has
// more CUs than a CTU can have (g_cuSlots).
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
  cu.refIdx0  = -1;   // (as CodingUnit::minInit sets them)
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
  cu.intraDirL = DC_IDX;   // (as CodingUnit::minInit sets it)
  cu.intraDirC = 0;
  cu.qtDepth  = g_p.currQtDepth;
  cu.depth    = g_p.currDepth;
  cu.left     = g_lv.left;
  cu.above    = g_lv.above;
  cu.dmvrOff  = 0;
  // A luma CU of at least 8x8 and 128 samples reserves room for DMVR's refinements: one per 16x16 subblock (a side
  // shorter than 16 counts as one subblock)
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

// Enters the finished CU into the CTU's maps: into the luma tree's map over its luma block, and into the chroma tree's
// map over its chroma block
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
    // The block in luma samples; the entry holds its size in the channel type's samples
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

// Writes the CU's record (GpuFrontEnd.h's CuRecord) into the staged record at, word for word as writeCuRecords writes
// it from the CPU's parse
void writeCuRecord( uint at )
{
  const Cu  cu  = g_cu;
  uint*     r   = pc.cus + at * 32u;
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

// The QP predicted at the start of a quantization group (CU::predictQP): the mean of the QPs of the CUs above and to
// the left, each replaced by prevQp across the CTU's edge. At the top-left corner of a CTU in a tile's first column,
// it is the QP of the CU above, when that is in the same slice and tile.
int predictQp( int prevQp )
{
  const int ch     = g_cu.chType;
  const int x      = ch == 0 ? g_cu.lx : g_cu.cx << g_csx;
  const int y      = ch == 0 ? g_cu.ly : g_cu.cy << g_csy;
  const uint2 above = cuAt( x, y - stepY( ch ), ch );   // (zero when the CTU above is in another slice or tile)
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

// The chroma QP of component comp (1 Cb, 2 Cr) of a TU of the CU with joint Cb-Cr mode jointCbCr, as
// QpParam( tu, comp, false ).Qp( false ) works it out
int chromaQp( int comp, int jointCbCr )
{
  const int jCbCr  = jointCbCr == 3 ? 3 : comp;   // (mode 3 is TU::getICTMode( tu, false ) == 2: the joint QP)
  int       offset = picWord( kPicCqpOffsets + jCbCr - 1 ) + sliceWord( kSlChromaQpDelta + jCbCr - 1 );
  if( g_cu.chromaQpAdj > 0 )
  {
    offset += picWord( kPicCqpOffsets + 3 + ( g_cu.chromaQpAdj - 1 ) * 3 + jCbCr - 1 );
  }
  const int qpi    = clip3( -g_qpBdOffset, MAX_QP, g_cu.qp );
  const int baseQp = uni( s_cqp[( jCbCr - 1 ) * 128 + qpi + g_qpBdOffset] );
  return clip3( 0, MAX_QP + g_qpBdOffset, baseQp + offset + g_qpBdOffset );
}

// Parses the CU at a leaf of the coding tree, as coding_tree does there: creates the CU, predicts its QP when a
// quantization group starts, parses coding_unit, works out its TUs' chroma QPs, and writes its records and map entries
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
  // With delta QP, a chroma CU of a separate tree takes the QP of the luma CU at its centre; the luma QP comes back
  // after the CU
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
  g_binCount += 32u;   // (the CU's work besides its bins counts as 32 bins against the dispatch's budget)
  PROF_COUNT( kProfTus, g_cu.numTus );

  if( sepChroma )
  {
    ctx.qp = lumaQpLocal;
  }
  // The chroma QPs of the CU's TUs (TU record word 5)
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
  stageRecords();
  writeCuInfo();
  PROF_END( kProfRecords, recStart );
}

// --------------------------------------------------------------------------------------------------------------------
// The CU's records staged, and the room the front end's job lists need for them
// --------------------------------------------------------------------------------------------------------------------

// The bits of a meta word (pc.cuMeta, pc.tuMeta) below the CTU: the record's index among its CTU's CUs (or TUs)
static const uint kLocalBits = 12;

// Stages the CU's record and its TUs' records: each at the place the picture's counters give it (pc.counters[0] and
// [1]), with its meta word for parsecompact.slang's scatter. A record past the room (pc.cuCap, pc.tuCap) is only
// counted: the host then parses the picture again with more room. Then adds up the CU's share of the room the front
// end's job lists need.
void stageRecords()
{
  uint cuAt = 0;
  uint tuAt = 0;
  if( g_lane == 0 )
  {
    cuAt = pc.counters[0].add( 1u );
    tuAt = g_cu.numTus > 0 ? pc.counters[1].add( uint( g_cu.numTus ) ) : 0u;
  }
  cuAt = uni( cuAt );
  tuAt = uni( tuAt );
  if( cuAt < uint( pc.cuCap ) )
  {
    writeCuRecord( cuAt );
    pc.cuMeta[cuAt] = uint( g_ctu ) << kLocalBits | uint( g_cu.local );
  }
  writeTuRecords( tuAt );
  addCapacities();
}

// The pieces a block side of n samples splits into for the residual jobs: one per set bit (SBT's 3/4 blocks: two)
uint pieces( uint n )
{
  return countbits( n );
}

// The bands of rows an intra job of w x h samples is cut into (GpuRecon::recordIntra's: 256 samples a band)
uint bandsOf( uint w, uint h )
{
  const uint rows = max( 1u, 256u / max( w, 1u ) );
  return ( h + rows - 1u ) / rows;
}

// Adds the CU's share of the room the front end's job lists need to g_caps, as gpu::frontCapacities works it out from
// the CU's record and its TUs' records: residual jobs, inter tiles, intra jobs, intra bands, deblocking edges
void addCapacities()
{
  const Cu   cu = g_cu;
  const uint w  = uint( cu.lw );
  const uint h  = uint( cu.lh );
  const uint sx = uint( g_csx );
  const uint sy = uint( g_csy );
  if( cu.predMode == MODE_INTER && cu.lw > 0 )
  {
    // GPM's, the regular tiles, affine's pairs or SbTMVP's sub-PUs (the subblock merge CUs are affine ones)
    const uint tiles16 = ( ( w + 15u ) / 16u ) * ( ( h + 15u ) / 16u );
    const bool affine  = ( cu.flags & kAffine ) != 0;
    g_caps[1] += affine ? max( 2u * tiles16, max( w / 8u, 1u ) * max( h / 8u, 1u ) ) : tiles16;
    if( ( cu.flags & kCiip ) != 0 )
    {
      g_caps[2] += 3u;
      g_caps[3] += bandsOf( w, h ) + 2u * bandsOf( w >> sx, h >> sy );
    }
    if( affine )
    {
      g_caps[4] += 2u * ( ( w / 8u ) * ( h / 4u ) + ( h / 8u ) * ( w / 4u ) );   // the subblock grid's edges
    }
  }
  for( int t = 0; t < cu.numTus; t++ )
  {
    const uint w1           = tuWord( t, 1 );
    const uint w3           = tuWord( t, 3 );
    const bool luma         = ( w1 & 0x10000u ) != 0;
    const bool chroma       = ( w3 & 0x10000u ) != 0;
    const uint lw           = w1 & 0xffu;
    const uint lh           = ( w1 >> 8 ) & 0xffu;
    const uint cw           = w3 & 0xffu;
    const uint ch           = ( w3 >> 8 ) & 0xffu;
    const uint chromaPieces = chroma ? pieces( cw ) * pieces( ch ) : 0u;
    // a residual job, or the pieces of zeros, a component
    g_caps[0] += ( luma ? pieces( lw ) * pieces( lh ) : 0u ) + 2u * chromaPieces;
    if( cu.predMode != MODE_INTER || cu.rootCbf )
    {
      // an intra or IBC job a component, or LMCS's chroma pieces
      g_caps[2] += ( luma ? 1u : 0u ) + ( chroma ? 2u : 0u ) + 2u * chromaPieces;
      g_caps[3] += ( luma ? bandsOf( lw, lh ) : 0u ) + ( chroma ? 2u * bandsOf( cw, ch ) + 2u * chromaPieces * bandsOf( cw, ch ) : 0u );
    }
    // the edges along the TU's left and top sides, luma's and chroma's (on the luma grid)
    g_caps[4] += ( luma ? lw / 4u + lh / 4u : 0u ) + ( chroma ? ( cw << sx ) / 4u + ( ch << sy ) / 4u : 0u );
  }
}

// --------------------------------------------------------------------------------------------------------------------
// coding_unit
// --------------------------------------------------------------------------------------------------------------------

// Whether the CU's mode type allows only intra (inter) prediction (CU::isConsIntra, CU::isConsInter)
bool isConsIntra()
{
  return g_cu.modeType == MODE_TYPE_INTRA;
}

bool isConsInter()
{
  return g_cu.modeType == MODE_TYPE_INTER;
}

// The context offset of cu_skip_flag (flag kSkip) or of the affine flags (flag kAffine): how many of the CUs to the
// left and above have that flag
int ctxLeftAbove( uint flag )
{
  // (the map entries hold only these two flags)
  const bool left  = flag == kSkip ? infoSkip( g_cu.left ) : infoAffine( g_cu.left );
  const bool above = flag == kSkip ? infoSkip( g_cu.above ) : infoAffine( g_cu.above );
  return ( left ? 1 : 0 ) + ( above ? 1 : 0 );
}

// The context offset of pred_mode_ibc_flag: how many of the CUs to the left and above are IBC CUs
int ctxIbcFlag()
{
  int ctx = ( infoValid( g_cu.left ) && infoPredMode( g_cu.left ) == MODE_IBC ) ? 1 : 0;
  ctx += ( infoValid( g_cu.above ) && infoPredMode( g_cu.above ) == MODE_IBC ) ? 1 : 0;
  return ctx;
}

// Parses cu_skip_flag, and pred_mode_ibc_flag for a skipped CU (CABACReader::cu_skip_flag)
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

// Parses pred_mode_flag and pred_mode_ibc_flag, as far as the slice, the CU's size and its mode type leave them to
// the bitstream (CABACReader::pred_mode)
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

// Parses cu_act_enabled_flag (CABACReader::adaptive_color_transform)
void adaptiveColorTransform()
{
  if( !tool( kToolAct ) || cuIsSepTree() )
  {
    return;
  }
  g_cu.act = decodeBin( CTX_ACTFlag ) != 0;
}

// Parses the BDPCM flag and direction of the luma (comp 0) or the chroma (CABACReader::bdpcm_mode): 0 off,
// 1 horizontal, 2 vertical
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

// CABACReader::coding_unit, with cu_pred_data folded in. A skipped CU's prediction_unit and cu_pred_data's share the
// one call of predictionUnit below, as every call is inlined and a second call site would copy it.
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
    g_cu.predMode = MODE_IBC;   // (a chroma tree's CU that is not intra is IBC, as in cu_pred_data)
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
// Intra prediction data
// --------------------------------------------------------------------------------------------------------------------

// Parses intra_mip_flag (CABACReader::mip_flag). Its context counts the MIP CUs to the left and above, except for
// blocks more than twice as wide as high or as high as wide, which use context 3.
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

// Parses intra_mip_transposed_flag and intra_mip_mode, a truncated binary code of 16, 8 or 6 modes by the block's
// size (CABACReader::mip_pred_mode)
void mipPredMode()
{
  g_cu.mipT          = int( decodeBinEP() );
  const int sizeId   = ( g_cu.lw == 4 && g_cu.lh == 4 ) ? 0 : ( ( g_cu.lw == 4 || g_cu.lh == 4 || ( g_cu.lw == 8 && g_cu.lh == 8 ) ) ? 1 : 2 );
  const int numModes = sizeId == 0 ? 16 : ( sizeId == 1 ? 8 : 6 );
  g_cu.intraDirL     = readTruncBinCode( numModes );
}

// Parses the MRL index, 0, 1 or 2 (CABACReader::extend_ref_line); a CU on the CTU's top row has none
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

// The ISP splits a w x h CU allows (CU::canUseISPSplit): 4 when both directions are allowed, 0
// (NOT_INTRA_SUBPARTITIONS) when ISP is not, as the CU has too few samples or exceeds the largest transform
int canUseIspSplit( int w, int h )
{
  const bool notEnoughSamplesToSplit   = log2i( w ) + log2i( h ) <= 4;
  const bool cuSizeLargerThanMaxTrSize = w > g_maxTbSize || h > g_maxTbSize;
  const int  widthCanBeUsed            = ( !cuSizeLargerThanMaxTrSize && !notEnoughSamplesToSplit ) ? 4 : 2;
  const int  heightCanBeUsed           = ( !cuSizeLargerThanMaxTrSize && !notEnoughSamplesToSplit ) ? 0 : 2;
  return widthCanBeUsed >> heightCanBeUsed;
}

// Parses intra_subpartitions_mode_flag and intra_subpartitions_split_flag (CABACReader::isp_mode): ISP mode 1
// horizontal, 2 vertical
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

// The intra luma mode of a neighbouring CU, given its map entry (PU::getIntraDirLuma): planar for a MIP CU
int intraDirLumaOf( uint2 info )
{
  return infoMip( info ) ? PLANAR_IDX : infoIntraDir( info );
}

// The CU's 6 most probable intra modes (PU::getIntraMPMs), from the modes of the intra CUs to the left (beside the
// CU's bottom-left sample) and above (over its top-right sample, not across the CTU's top edge)
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

// Parses the CU's intra luma mode (CABACReader::intra_luma_pred_mode): BDPCM's direction fixes it; else MIP's flag
// and mode, or MRL, ISP and then an MPM (intra_luma_mpm_flag, intra_luma_not_planar_flag, the MPM index) or one of
// the remaining modes
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
    // Not an MPM: the coded value counts the modes that are not MPMs, so step over the MPMs in ascending order
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

// The luma CU at the centre of the CU's chroma block, as PU::getCoLocatedLumaPU finds it for a CU of a separate tree
uint2 coLocatedLumaCu()
{
  return cuAt( ( g_cu.cx << g_csx ) + ( ( g_cu.cw << g_csx ) >> 1 ), ( g_cu.cy << g_csy ) + ( ( g_cu.chh << g_csy ) >> 1 ), 0 );
}

// Whether the CU may use CCLM (CU::checkCCLMAllowed). Only the dual tree with CTUs of 64 or more restricts it, by how
// the CU's 64x64 area was split and by the luma CU at the CU's top-left corner.
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

// Parses the CU's intra chroma mode (CABACReader::intra_chroma_pred_mode): BDPCM's direction or ACT fixes it; else
// a CCLM mode, the DM mode (the luma's), or one of 4 listed modes
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
      // cclm_mode_idx (vvdec's intra_chroma_lmc_mode): LM, MDLM_L or MDLM_T
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
  // The listed modes (PU::getIntraChromaCandModes): planar, vertical, horizontal and DC, where the one equal to the
  // colocated luma mode is replaced by VDIA
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
// Inter and IBC: the prediction unit
// --------------------------------------------------------------------------------------------------------------------

// Parses one motion vector difference (CABACReader::mvd_coding)
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

// The slice's number of reference indices of a list
int numRefIdx( int list )
{
  return ( sliceWord( kSlNumRef ) >> ( 8 * list ) ) & 0xff;
}

// SMVD's reference index of a list (Slice::getSymRefIdx)
int symRefIdx( int list )
{
  return int( int8_t( ( sliceWord( kSlSymRef ) >> ( 8 * list ) ) & 0xff ) );
}

// Parses the reference index of a list (CABACReader::ref_idx); an SMVD CU uses SMVD's instead
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

// Parses the CU's inter direction (CABACReader::inter_pred_idc): 1 list 0, 2 list 1, 3 both
void interPredIdc()
{
  if( g_sliceType == P_SLICE )
  {
    g_cu.interDir = 1;
    return;
  }
  if( !( g_cu.lw + g_cu.lh <= 12 ) )   // (PU::isBipredRestriction: 8x4 and 4x8 CUs cannot predict from both lists)
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

// Parses inter_affine_flag and cu_affine_type_flag (CABACReader::affine_flag)
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

// Parses sym_mvd_flag (CABACReader::smvd_mode); g_cu.smvd becomes 1 when it is set
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

// Parses merge_subblock_flag (CABACReader::subblock_merge_flag), which makes the CU affine
void subblockMergeFlag()
{
  if( !sliceIsIntra() && sliceWord( kSlMaxAffMerge ) > 0 && g_cu.lw >= 8 && g_cu.lh >= 8 )
  {
    setFlag( kAffine, decodeBin( CTX_SubblockMergeFlag + ctxLeftAbove( kAffine ) ) != 0 );
  }
}

// Parses the merge index (CABACReader::merge_idx): a GPM CU's split direction and two merge candidates, or the merge
// index of an IBC, affine (subblock) or regular merge CU
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

// Parses MMVD's base candidate, distance and direction (CABACReader::mmvd_merge_idx), stored as
// mmvdIdx = base * 32 + distance * 4 + direction
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

// Parses a merge CU's merge data (CABACReader::merge_data): which merge mode it uses -- subblock (affine), regular,
// MMVD, CIIP or GPM -- and its indices
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

// Parses the prediction unit of an inter or IBC CU (CABACReader::prediction_unit): merge data, or the AMVP syntax
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
    // AMVP. IBC uses list 0 with one difference. Inter parses, for each list the CU uses, the reference index, the
    // differences (one, or one per affine control point) and the predictor index. The loops keep mvdCoding to one
    // call site, as every call is inlined.
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
  // SMVD: the other list mirrors the coded list's difference and takes SMVD's reference index
  if( g_cu.smvd != 0 )
  {
    const int cur = g_cu.smvd - 1;   // (the list whose difference was coded: smvdMode sets 1, so always list 0)
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

// Whether a coded motion vector difference of the CU is nonzero (CU::hasSubCUNonZeroMVd); AMVR is coded only then
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

// The same for an affine CU's control point differences (CU::hasSubCUNonZeroAffineMVd)
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

// Parses amvr_flag and amvr_precision_idx (CABACReader::amvr_mode) into imv: 0 quarter-sample, 1 full-sample,
// 2 four-sample, IMV_HPEL half-sample precision. An IBC CU codes no amvr_flag (it counts as 1), so its
// amvr_precision_idx picks full- or four-sample precision.
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

// Parses an affine CU's amvr_flag and amvr_precision_idx (CABACReader::affine_amvr_mode) into imv 0, 1 or 2
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

// Whether bcw_idx is coded (CU::isBcwIdxCoded): for bi-predicted inter CUs of at least 256 luma samples whose two
// references have no weighted prediction
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
  // (Slice::getWpScaling looks up a negative reference index as index 0)
  const int r0 = max( g_cu.refIdx0, 0 );
  const int r1 = max( g_cu.refIdx1, 0 );
  if( ( ( wp >> uint( r0 ) ) & 1u ) != 0 || ( ( wp >> uint( 16 + r1 ) ) & 1u ) != 0 )
  {
    return false;
  }
  return true;
}

// Parses bcw_idx (CABACReader::cu_bcw_flag): 5 weights when no reference follows the picture (getCheckLDC), else 3
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
  g_cu.bcw         = kBcwInternFwd[bcwIdx];   // (CU::setBcwIdx stores it for an AMVP bi-predicted CU, which this is)
}
