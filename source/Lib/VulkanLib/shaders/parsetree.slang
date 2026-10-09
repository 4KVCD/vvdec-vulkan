// parse.slang's coding tree: the maps of CUs and the neighbours, the partitioner (vvdec's Partitioner), the coding
// tree unit with SAO's and ALF's syntax, the coding tree (CABACReader::coding_tree, split_cu_mode, mode_constraint).

// --------------------------------------------------------------------------------------------------------------------
// the CUs parsed so far around the CTU: a word pair each 4x4 luma block -- of the CTU (s_map, per channel type: the
// luma tree's CUs, the chroma tree's; the single tree's in both), of the bottom row of the CTU above (s_above) and the
// right column of the CTU left (s_left), zero where none or out of reach (another slice or tile):
//   x: the CU's index in its CTU + 1 (12 bits) | prediction mode << 12 | skip << 14 | affine << 15 | MIP << 16 |
//      ISP << 17 | QT depth << 19 | depth << 22 | its block's (the channel type's) log2 width << 26 | log2 height << 29
//   y: its luma intra mode (byte) | its QP << 8 (byte)
// written when the CU is done (what is looked up is always another CU, done)
// --------------------------------------------------------------------------------------------------------------------

groupshared uint2 s_map[2048];   // [channel type][y >> 2][x >> 2] from the CTU's corner, 32 a row
groupshared uint2 s_above[64];   // [channel type][x >> 2]
groupshared uint2 s_left[64];    // [channel type][y >> 2]

static bool g_leftAvail;    // the CTU left (above) in the same slice and tile
static bool g_aboveAvail;
static bool g_tileFirstCol; // the CTU in its tile's first column

bool infoValid( uint2 i )
{
  return ( i.x & 0xfffu ) != 0;
}

int infoLocal( uint2 i )
{
  return int( i.x & 0xfffu ) - 1;
}

int infoPredMode( uint2 i )
{
  return int( ( i.x >> 12 ) & 3u );
}

bool infoSkip( uint2 i )
{
  return ( ( i.x >> 14 ) & 1u ) != 0;
}

bool infoAffine( uint2 i )
{
  return ( ( i.x >> 15 ) & 1u ) != 0;
}

bool infoMip( uint2 i )
{
  return ( ( i.x >> 16 ) & 1u ) != 0;
}

int infoIsp( uint2 i )
{
  return int( ( i.x >> 17 ) & 3u );
}

int infoQtDepth( uint2 i )
{
  return int( ( i.x >> 19 ) & 7u );
}

int infoDepth( uint2 i )
{
  return int( ( i.x >> 22 ) & 15u );
}

int infoLog2W( uint2 i )
{
  return int( ( i.x >> 26 ) & 7u );
}

int infoLog2H( uint2 i )
{
  return int( i.x >> 29 );
}

int infoIntraDir( uint2 i )
{
  return int( int8_t( i.y & 0xffu ) );
}

int infoQp( uint2 i )
{
  return int( int8_t( ( i.y >> 8 ) & 0xffu ) );
}

// whether two CTUs belong to the same slice and tile
bool sameSliceTile( int a, int b )
{
  return uni( pc.ctuInfo[a * 2] ) == uni( pc.ctuInfo[b * 2] ) && ( uni( pc.ctuInfo[a * 2 + 1] ) & 0xffffu ) == ( uni( pc.ctuInfo[b * 2 + 1] ) & 0xffffu );
}

// CodingStructure::getCU (and getCURestricted's, the CTUs out of reach zero): the CU of the channel type ch at the luma
// position (x, y) -- in the CTU, the row above it or the column left of it
uint2 cuAt( int x, int y, int ch )
{
  if( x < 0 || y < 0 || x >= g_width || y >= g_height )
  {
    return uint2( 0, 0 );
  }
  const int dx = x - ( g_ctuX << g_log2Ctu );
  const int dy = y - ( g_ctuY << g_log2Ctu );
  if( dx >= 0 && dy >= 0 && dx < g_ctuSize && dy < g_ctuSize )
  {
    return uni( s_map[ch * 1024 + ( dy >> 2 ) * 32 + ( dx >> 2 )] );
  }
  if( dy < 0 && dy >= -4 && dx >= 0 && dx < g_ctuSize )
  {
    return uni( s_above[ch * 32 + ( dx >> 2 )] );
  }
  if( dx < 0 && dx >= -4 && dy >= 0 && dy < g_ctuSize )
  {
    return uni( s_left[ch * 32 + ( dy >> 2 )] );
  }
  return uint2( 0, 0 );
}

// CodingStructure::getCURestricted( pos, curCu, chType ): in the CTU, not a CU after the current one (its index cur)
uint2 cuAtRestricted( int x, int y, int cur, int ch )
{
  const uint2 i  = cuAt( x, y, ch );
  const int   dx = x - ( g_ctuX << g_log2Ctu );
  const int   dy = y - ( g_ctuY << g_log2Ctu );
  if( dx >= 0 && dy >= 0 && infoValid( i ) && infoLocal( i ) > cur )
  {
    return uint2( 0, 0 );
  }
  return i;
}

// --------------------------------------------------------------------------------------------------------------------
// the partitioner (vvdec's Partitioner): its levels in shared memory, the rest in the thread's
// --------------------------------------------------------------------------------------------------------------------

static const int kMaxLevels = 24;

groupshared int s_lvSplit[kMaxLevels];
groupshared int s_lvIdx[kMaxLevels];
groupshared int s_lvNum[kMaxLevels];
groupshared uint2 s_lvAbove[kMaxLevels];
groupshared uint2 s_lvLeft[kMaxLevels];
groupshared int s_lvQg[kMaxLevels];    // qgEnable | qgChromaEnable << 1
groupshared int4 s_lvArea[kMaxLevels]; // the area split (luma samples: x, y, w, h)

struct Part
{
  int top;   // the current level
  int currDepth;
  int currQtDepth;
  int currTrDepth;
  int currSubdiv;
  int currMtDepth;
  int currImplicitBtDepth;
  int chType;
  int treeType;
  int modeType;
  int maxBTD;
  int maxBtSize;
  int minBtSize;
  int maxTtSize;
  int minTtSize;
  int minQtSize;
};

static Part g_p;

// the area of part idx of a split of a
int4 partArea( int split, int idx, int4 a )
{
  if( split == CU_QUAD_SPLIT )
  {
    return int4( a.x + ( idx & 1 ) * ( a.z >> 1 ), a.y + ( idx >> 1 ) * ( a.w >> 1 ), a.z >> 1, a.w >> 1 );
  }
  if( split == CU_HORZ_SPLIT )
  {
    return int4( a.x, a.y + idx * ( a.w >> 1 ), a.z, a.w >> 1 );
  }
  if( split == CU_VERT_SPLIT )
  {
    return int4( a.x + idx * ( a.z >> 1 ), a.y, a.z >> 1, a.w );
  }
  if( split == CU_TRIH_SPLIT )
  {
    const int q = a.w >> 2;
    return int4( a.x, a.y + ( idx == 0 ? 0 : ( idx == 1 ? q : 3 * q ) ), a.z, idx == 1 ? 2 * q : q );
  }
  if( split == CU_TRIV_SPLIT )
  {
    const int q = a.z >> 2;
    return int4( a.x + ( idx == 0 ? 0 : ( idx == 1 ? q : 3 * q ) ), a.y, idx == 1 ? 2 * q : q, a.w );
  }
  return a;   // CTU_LEVEL
}

int4 currArea()
{
  const int l = g_p.top;
  return partArea( uni( s_lvSplit[l] ), uni( s_lvIdx[l] ), int4( uni( s_lvArea[l].x ), uni( s_lvArea[l].y ), uni( s_lvArea[l].z ), uni( s_lvArea[l].w ) ) );
}

bool currQgEnable()
{
  return ( uni( s_lvQg[g_p.top] ) & 1 ) != 0;
}

bool currQgChromaEnable()
{
  return ( uni( s_lvQg[g_p.top] ) & 2 ) != 0;
}

bool isSepTreeP()
{
  return g_p.treeType != TREE_D || g_dualTree;
}

// a step in the channel's samples, in luma samples
int stepX( int ch )
{
  return ch == 0 ? 1 : 1 << g_csx;
}

int stepY( int ch )
{
  return ch == 0 ? 1 : 1 << g_csy;
}

// setNeighborCu
void setNeighborCu( int l )
{
  const int  ch = g_p.treeType == TREE_C ? 1 : g_p.chType;
  const int4 a  = partArea( uni( s_lvSplit[l] ), uni( s_lvIdx[l] ), int4( uni( s_lvArea[l].x ), uni( s_lvArea[l].y ), uni( s_lvArea[l].z ), uni( s_lvArea[l].w ) ) );
  if( g_lane == 0 )
  {
  }
  s_lvAbove[l] = cuAt( a.x, a.y - stepY( ch ), ch );
  s_lvLeft[l]  = cuAt( a.x - stepX( ch ), a.y, ch );
}

// Partitioner::initCtu
void initCtu( int chType )
{
  g_p.top                 = 0;
  g_p.currDepth           = 0;
  g_p.currTrDepth         = 0;
  g_p.currMtDepth         = 0;
  g_p.currQtDepth         = 0;
  g_p.currSubdiv          = 0;
  g_p.currImplicitBtDepth = 0;
  g_p.chType              = chType;
  g_p.treeType            = TREE_D;
  g_p.modeType            = MODE_TYPE_ALL;
  s_lvSplit[0]            = CTU_LEVEL;
  s_lvIdx[0]              = 0;
  s_lvNum[0]              = 1;
  s_lvQg[0]               = 3;
  s_lvArea[0]             = int4( g_ctuX << g_log2Ctu, g_ctuY << g_log2Ctu, g_ctuSize, g_ctuSize );
  setNeighborCu( 0 );
  // (GDR_ADJ) the luma's limits of intra or inter slices, the dual tree's chroma's
  const uint lim          = uint( sliceWord( chType == 1 && g_dualTree ? kSlPartChroma : kSlPartLuma ) );
  g_p.maxBTD              = int( lim & 0xffu );
  g_p.maxBtSize           = int( ( lim >> 8 ) & 0xffu );
  g_p.maxTtSize           = int( ( lim >> 16 ) & 0xffu );
  g_p.minQtSize           = int( lim >> 24 );
  g_p.minBtSize           = 1 << g_log2MinCb;
  g_p.minTtSize           = 1 << g_log2MinCb;
}

bool outsidePicture( int4 a )
{
  return a.x + a.z - 1 >= g_width || a.y + a.w - 1 >= g_height;
}

// Partitioner::splitCurrArea (the CU splits)
void splitCurrArea( int split )
{
  const int4 area           = currArea();
  const bool isImplicit     = outsidePicture( area );
  bool       qgEnable       = currQgEnable();
  bool       qgChromaEnable = currQgChromaEnable();
  const int  last           = g_p.top;
  const int  l              = last + 1;
  g_p.top                   = l;
  s_lvSplit[l]              = split;
  s_lvIdx[l]                = 0;
  s_lvNum[l]                = split == CU_QUAD_SPLIT ? 4 : ( split == CU_TRIH_SPLIT || split == CU_TRIV_SPLIT ? 3 : 2 );
  s_lvArea[l]               = area;
  if( split == CU_QUAD_SPLIT )
  {
    g_p.currTrDepth = 0;
    g_p.currMtDepth = 0;
    g_p.currQtDepth++;
    g_p.currSubdiv++;
  }
  else if( split == CU_HORZ_SPLIT || split == CU_VERT_SPLIT )
  {
    g_p.currTrDepth = 0;
    if( isImplicit )
    {
      g_p.currImplicitBtDepth++;
    }
    g_p.currMtDepth++;
  }
  else
  {
    g_p.currTrDepth = 0;
    g_p.currMtDepth++;
    g_p.currSubdiv++;
  }
  g_p.currDepth++;
  g_p.currSubdiv++;
  const uint subdiv = uint( sliceWord( kSlSubdiv ) );
  qgEnable          = qgEnable && g_p.currSubdiv <= int( subdiv & 0xffu );
  qgChromaEnable    = qgChromaEnable && g_p.currSubdiv <= int( ( subdiv >> 8 ) & 0xffu );
  s_lvQg[l]         = ( qgEnable ? 1 : 0 ) | ( qgChromaEnable ? 2 : 0 );
  s_lvAbove[l]      = s_lvAbove[last];
  s_lvLeft[l]       = s_lvLeft[last];
}

// Partitioner::nextPart (the CU splits)
bool nextPart()
{
  const int l   = g_p.top;
  const int idx = uni( s_lvIdx[l] ) + 1;
  s_lvIdx[l]    = idx;
  if( idx < uni( s_lvNum[l] ) )
  {
    setNeighborCu( l );
    const int split = uni( s_lvSplit[l] );
    if( split == CU_TRIH_SPLIT || split == CU_TRIV_SPLIT )
    {
      if( idx == 1 )
      {
        g_p.currSubdiv--;
      }
      else
      {
        g_p.currSubdiv++;
      }
    }
    return true;
  }
  return false;
}

// Partitioner::exitCurrSplit (the CU splits)
void exitCurrSplit()
{
  const int l         = g_p.top;
  const int currSplit = uni( s_lvSplit[l] );
  const int currIndex = uni( s_lvIdx[l] );
  g_p.top             = l - 1;
  const bool isImplicit = outsidePicture( currArea() );
  g_p.currDepth--;
  g_p.currSubdiv--;
  if( currSplit != CU_QUAD_SPLIT )
  {
    g_p.currMtDepth--;
    if( isImplicit )
    {
      g_p.currImplicitBtDepth--;
    }
    if( ( currSplit == CU_TRIH_SPLIT || currSplit == CU_TRIV_SPLIT ) && currIndex != 1 )
    {
      g_p.currSubdiv--;
    }
  }
  else
  {
    g_p.currQtDepth--;
    g_p.currSubdiv--;
  }
}

// Partitioner::canSplit: canNo | canQt << 1 | canBh << 2 | canBv << 3 | canTh << 4 | canTv << 5
int canSplit()
{
  bool       canNo = true, canQt = true, canBh = true, canTh = true, canBv = true, canTv = true;
  bool       canBtt = g_p.currMtDepth < ( g_p.maxBTD + g_p.currImplicitBtDepth );
  const int4 area   = currArea();
  const bool chroma = g_p.chType == 1;
  const int  cw     = area.z >> g_csx;   // the chroma partitioner's chroma block
  const int  chh    = area.w >> g_csy;
  const int  l      = g_p.top;

  if( g_dualTree && ( area.z > 64 || area.w > 64 ) )
  {
    return 2;   // QT only
  }
  if( g_p.treeType == TREE_C )
  {
    return 1;   // no split
  }
  const int  lastSplit = uni( s_lvSplit[l] );
  const bool isTrInPic = area.x + area.z <= g_width;
  const bool isBlInPic = area.y + area.w <= g_height;
  const bool isImplicit = !isBlInPic || !isTrInPic;

  if( lastSplit != CTU_LEVEL && lastSplit != CU_QUAD_SPLIT )
  {
    canQt = false;
  }
  if( area.z <= g_p.minQtSize )
  {
    canQt = false;
  }
  if( chroma && cw <= MIN_DUALTREE_CHROMA_WIDTH )
  {
    canQt = false;
  }
  if( isImplicit )
  {
    const bool isBtAllowed = area.z <= g_p.maxBtSize && area.w <= g_p.maxBtSize && area.z <= MAX_TU_SIZE_FOR_PROFILE
                             && area.w <= MAX_TU_SIZE_FOR_PROFILE && canBtt;
    canNo = false;
    canTh = false;
    canTv = false;
    canQt = canQt || !isBtAllowed;
    canBh = isBtAllowed && !isBlInPic && ( isTrInPic || !canQt );
    canBv = isBtAllowed && isBlInPic && !isTrInPic;
    canBv = canBv && ( !chroma || cw > 4 );
    canQt = canQt || ( !canBh && !canBv );
    return ( canNo ? 1 : 0 ) | ( canQt ? 2 : 0 ) | ( canBh ? 4 : 0 ) | ( canBv ? 8 : 0 ) | ( canTh ? 16 : 0 ) | ( canTv ? 32 : 0 );
  }

  canBtt = canBtt && ( area.z > g_p.minBtSize || area.w > g_p.minBtSize || area.z > g_p.minTtSize || area.w > g_p.minTtSize );
  canBtt = canBtt && ( ( area.z <= g_p.maxBtSize && area.w <= g_p.maxBtSize ) || ( area.z <= g_p.maxTtSize && area.w <= g_p.maxTtSize ) );
  if( !canBtt )
  {
    return ( canNo ? 1 : 0 ) | ( canQt ? 2 : 0 );
  }

  const bool allowModeBt = g_p.modeType != MODE_TYPE_INTER || area.z * area.w != 32;
  const bool allowModeTt = g_p.modeType != MODE_TYPE_INTER || area.z * area.w != 64;

  if( area.z > g_p.maxBtSize || area.w > g_p.maxBtSize || !allowModeBt )
  {
    canBh = false;
    canBv = false;
  }
  else
  {
    if( ( lastSplit == CU_TRIH_SPLIT || lastSplit == CU_TRIV_SPLIT ) && uni( s_lvIdx[l] ) == 1 )
    {
      const int parlSplit = lastSplit == CU_TRIH_SPLIT ? CU_HORZ_SPLIT : CU_VERT_SPLIT;
      canBh               = parlSplit != CU_HORZ_SPLIT;
      canBv               = parlSplit != CU_VERT_SPLIT;
    }
    canBh = canBh && ( area.w > g_p.minBtSize && area.w <= g_p.maxBtSize );
    canBh = canBh && ( area.z <= MAX_TU_SIZE_FOR_PROFILE || area.w > MAX_TU_SIZE_FOR_PROFILE );
    canBv = canBv && ( area.z > g_p.minBtSize && area.z <= g_p.maxBtSize );
    canBv = canBv && ( area.z > MAX_TU_SIZE_FOR_PROFILE || area.w <= MAX_TU_SIZE_FOR_PROFILE );
  }

  if( area.z > g_p.maxTtSize || area.w > g_p.maxTtSize || !allowModeTt || !( area.z <= MAX_TU_SIZE_FOR_PROFILE && area.w <= MAX_TU_SIZE_FOR_PROFILE ) )
  {
    canTh = false;
    canTv = false;
    if( !canBh && !canBv )
    {
      return ( canNo ? 1 : 0 ) | ( canQt ? 2 : 0 );
    }
  }
  else
  {
    canTh = canTh && !( area.w <= 2 * g_p.minTtSize );
    canTv = canTv && !( area.z <= 2 * g_p.minTtSize );
  }

  if( chroma )
  {
    canBh = canBh && ( cw * chh > MIN_DUALTREE_CHROMA_SIZE );
    canTh = canTh && ( cw * chh > MIN_DUALTREE_CHROMA_SIZE * 2 );
    canBv = canBv && ( cw * chh > MIN_DUALTREE_CHROMA_SIZE && cw > 4 );
    canTv = canTv && ( cw * chh > MIN_DUALTREE_CHROMA_SIZE * 2 && cw > 8 );
  }
  return ( canNo ? 1 : 0 ) | ( canQt ? 2 : 0 ) | ( canBh ? 4 : 0 ) | ( canBv ? 8 : 0 ) | ( canTh ? 16 : 0 ) | ( canTv ? 32 : 0 );
}

// CABACReader::split_cu_mode
int splitCuMode()
{
  const int  can   = canSplit();
  const bool canNo = ( can & 1 ) != 0, canQt = ( can & 2 ) != 0, canBh = ( can & 4 ) != 0, canBv = ( can & 8 ) != 0;
  const bool canTh = ( can & 16 ) != 0, canTv = ( can & 32 ) != 0;
  const int  numHor   = ( canBh ? 1 : 0 ) + ( canTh ? 1 : 0 );
  const int  numVer   = ( canBv ? 1 : 0 ) + ( canTv ? 1 : 0 );
  const int  numSplit = ( canQt ? 2 : 0 ) + numHor + numVer;
  bool       isSplit  = numSplit != 0;

  if( canNo && !isSplit )
  {
    return CU_DONT_SPLIT;
  }
  const int  l          = g_p.top;
  const uint2 cuLeft    = uni( s_lvLeft[l] );
  const uint2 cuAbove   = uni( s_lvAbove[l] );
  const int  ch         = g_p.chType;
  const int4 area       = currArea();
  const int  widthCurr  = ch == 0 ? area.z : area.z >> g_csx;
  const int  heightCurr = ch == 0 ? area.w : area.w >> g_csy;

  if( canNo && isSplit )
  {
    int ctxSplit = ( infoValid( cuLeft ) && ( 1 << infoLog2H( cuLeft ) ) < heightCurr ) ? 1 : 0;
    ctxSplit += ( infoValid( cuAbove ) && ( 1 << infoLog2W( cuAbove ) ) < widthCurr ) ? 1 : 0;
    const int ctxOffset[7] = { 0, 0, 0, 3, 3, 6, 6 };
    ctxSplit += ctxOffset[numSplit];
    isSplit = decodeBin( CTX_SplitFlag + ctxSplit ) != 0;
  }
  if( !isSplit )
  {
    return CU_DONT_SPLIT;
  }
  const bool canBtt = numHor != 0 || numVer != 0;
  bool       isQt   = canQt;
  if( isQt && canBtt )
  {
    int ctxQtSplit = ( infoValid( cuLeft ) && infoQtDepth( cuLeft ) > g_p.currQtDepth ) ? 1 : 0;
    ctxQtSplit += ( infoValid( cuAbove ) && infoQtDepth( cuAbove ) > g_p.currQtDepth ) ? 1 : 0;
    ctxQtSplit += g_p.currQtDepth < 2 ? 0 : 3;
    isQt = decodeBin( CTX_SplitQtFlag + ctxQtSplit ) != 0;
  }
  if( isQt )
  {
    return CU_QUAD_SPLIT;
  }
  const bool canHor = numHor != 0;
  bool       isVer  = numVer != 0;
  if( isVer && canHor )
  {
    int ctxBttHV = 0;
    if( numVer == numHor )
    {
      if( infoValid( cuLeft ) && infoValid( cuAbove ) )
      {
        const int wIdxAbove = infoLog2W( cuAbove );
        const int hIdxLeft  = infoLog2H( cuLeft );
        const int depAbove  = widthCurr >> wIdxAbove;
        const int depLeft   = heightCurr >> hIdxLeft;
        ctxBttHV            = depAbove == depLeft ? 0 : ( depAbove < depLeft ? 1 : 2 );
      }
    }
    else if( numVer < numHor )
    {
      ctxBttHV = 3;
    }
    else
    {
      ctxBttHV = 4;
    }
    isVer = decodeBin( CTX_SplitHvFlag + ctxBttHV ) != 0;
  }
  const bool can14 = isVer ? canTv : canTh;
  bool       is12  = isVer ? canBv : canBh;
  if( is12 && can14 )
  {
    const int ctxBtt12 = ( g_p.currMtDepth <= 1 ? 1 : 0 ) + ( isVer ? 2 : 0 );
    is12               = decodeBin( CTX_Split12Flag + ctxBtt12 ) != 0;
  }
  if( isVer && is12 )
  {
    return CU_VERT_SPLIT;
  }
  if( isVer )
  {
    return CU_TRIV_SPLIT;
  }
  if( is12 )
  {
    return CU_HORZ_SPLIT;
  }
  return CU_TRIH_SPLIT;
}

// signalModeCons: 0 inherit, 1 infer (intra), 2 signal
int signalModeCons( int split )
{
  if( g_dualTree || g_p.modeType != MODE_TYPE_ALL || g_chromaFormat == 3 || g_chromaFormat == 0 )
  {
    return 0;
  }
  const int4 area        = currArea();
  int        minLumaArea = area.z * area.w;
  if( split == CU_QUAD_SPLIT || split == CU_TRIH_SPLIT || split == CU_TRIV_SPLIT )
  {
    minLumaArea >>= 2;
  }
  else if( split == CU_VERT_SPLIT || split == CU_HORZ_SPLIT )
  {
    minLumaArea >>= 1;
  }
  const int  minChromaBlock = minLumaArea >> ( g_csx + g_csy );
  const int  cw             = area.z >> g_csx;
  const bool is2xNChroma    = ( cw == 4 && split == CU_VERT_SPLIT ) || ( cw == 8 && split == CU_TRIV_SPLIT );
  return minChromaBlock >= 16 && !is2xNChroma ? 0 : ( ( minLumaArea < 32 || sliceIsIntra() ) ? 1 : 2 );
}

// CABACReader::mode_constraint
int modeConstraint( int split )
{
  const int val = signalModeCons( split );
  if( val == 2 )
  {
    const int l       = g_p.top;
    const uint2 cuLeft  = uni( s_lvLeft[l] );
    const uint2 cuAbove = uni( s_lvAbove[l] );
    const int   ctxIdx  = ( ( infoValid( cuAbove ) && infoPredMode( cuAbove ) == MODE_INTRA ) || ( infoValid( cuLeft ) && infoPredMode( cuLeft ) == MODE_INTRA ) ) ? 1 : 0;
    return decodeBin( CTX_ModeConsFlag + ctxIdx ) != 0 ? MODE_TYPE_INTRA : MODE_TYPE_INTER;
  }
  if( val == 1 )
  {
    return MODE_TYPE_INTRA;
  }
  return g_p.modeType;
}

// the split of the CU's partition at a depth (CU::getSplitAtDepth, from the partitioner's levels)
int splitAtDepth( int depth )
{
  // the levels 1.. are the splits that led to the current area; depth d's is level d + 1
  if( depth >= g_p.currDepth )
  {
    return CU_DONT_SPLIT;
  }
  return uni( s_lvSplit[depth + 1] );
}

// --------------------------------------------------------------------------------------------------------------------
// the coding tree (CABACReader::coding_tree, without recursion: a stack of frames)
// --------------------------------------------------------------------------------------------------------------------

struct CuCtx
{
  bool qgStart;
  bool isDQPCoded;
  bool isChromaQpAdjCoded;
  int  qp;
  bool violatesLfnstLuma;
  bool violatesLfnstChroma;
  bool lfnstLastScanPos;
  bool violatesMtsCoeffConstraint;
  bool mtsLastScanPos;
};

CuCtx newCuCtx( int qp )
{
  CuCtx c;
  c.qgStart                    = false;
  c.isDQPCoded                 = false;
  c.isChromaQpAdjCoded         = false;
  c.qp                         = qp;
  c.violatesLfnstLuma          = false;
  c.violatesLfnstChroma        = false;
  c.lfnstLastScanPos           = false;
  c.violatesMtsCoeffConstraint = false;
  c.mtsLastScanPos             = false;
  return c;
}

static const int kMaxFrames = 32;
groupshared int s_frame[kMaxFrames];   // the frame's state (0: its parts, 1: its chroma) | chromaNotSplit << 1 | modeTypeParent << 2

bool inPicture( int4 a )
{
  return a.x < g_width && a.y < g_height;
}

void codingTree( inout CuCtx ctx )
{
  int  f     = 0;
  bool enter = true;
  while( f >= 0 && !g_error )
  {
    if( enter )
    {
      if( picWord( kPicUseDqp ) != 0 && currQgEnable() && g_p.chType == 0 )
      {
        ctx.qgStart    = true;
        ctx.isDQPCoded = false;
      }
      if( sliceFlag( kSlChromaQpAdj ) && currQgChromaEnable() )
      {
        ctx.isChromaQpAdjCoded = false;
        g_csChromaQpAdj        = 0;
      }
      const int split = splitCuMode();
      if( split == CU_DONT_SPLIT )
      {
        codingUnitAtPart( ctx );
        f--;
        enter = false;
        continue;
      }
      if( f + 1 >= kMaxFrames || g_p.top + 1 >= kMaxLevels )
      {
        g_error = true;
        break;
      }
      const int  modeTypeParent = g_p.modeType;
      g_p.modeType              = modeConstraint( split );
      const bool chromaNotSplit = modeTypeParent == MODE_TYPE_ALL && g_p.modeType == MODE_TYPE_INTRA;
      if( g_p.treeType == TREE_D )
      {
        g_p.treeType = chromaNotSplit ? TREE_L : TREE_D;
      }
      splitCurrArea( split );
      s_frame[f] = 0 | ( chromaNotSplit ? 2 : 0 ) | ( modeTypeParent << 2 );
      if( inPicture( currArea() ) )
      {
        f++;
        enter = true;
      }
      else
      {
        enter = false;
      }
      continue;
    }
    // frame f resumes: a part done (or outside the picture)
    const int fr = uni( s_frame[f] );
    if( ( fr & 1 ) == 0 )
    {
      if( nextPart() )
      {
        if( inPicture( currArea() ) )
        {
          f++;
          enter = true;
        }
        continue;
      }
      exitCurrSplit();
      if( ( fr & 2 ) != 0 )
      {
        // chromaNotSplit: the chroma of the whole area, a CU of the chroma tree
        g_p.chType   = 1;
        g_p.treeType = TREE_C;
        setNeighborCu( g_p.top );
        s_frame[f] = fr | 1;
        f++;
        enter = true;
        continue;
      }
      g_p.modeType = fr >> 2;
      f--;
      continue;
    }
    g_p.chType   = 0;
    g_p.treeType = TREE_D;
    g_p.modeType = fr >> 2;
    f--;
  }
}

// --------------------------------------------------------------------------------------------------------------------
// the coding tree unit: SAO (CABACReader::sao), ALF (readAlf), the coding tree(s)
// --------------------------------------------------------------------------------------------------------------------

uint ctuOutWord( int ctu, int w )
{
  return uni( pc.ctuOut[ctu * kCtuOutWords + w] );
}

void setCtuOut( int w, uint v )
{
  pc.ctuOut[g_ctu * kCtuOutWords + w] = v;
}

// the CTU's left or above neighbour in the same slice and tile
bool ctuLeftAvailable()
{
  return g_leftAvail;
}

bool ctuAboveAvailable()
{
  return g_aboveAvail;
}

// SAO's parameters of a component, out: modeIdc | typeIdc << 4 | typeAuxInfo << 8, then 4 offsets (bytes): the band
// offset's in band order, the edge offset's classes 0, 1, 3, 4
void sao()
{
  int mode[3]   = { SAO_MODE_OFF, SAO_MODE_OFF, SAO_MODE_OFF };
  int type[3]   = { 0, 0, 0 };
  int aux[3]    = { 0, 0, 0 };
  uint offs[3]  = { 0u, 0u, 0u };
  const bool lumaOn   = sliceFlag( kSlSaoLuma );
  const bool chromaOn = sliceFlag( kSlSaoChroma ) && g_chromaFormat != 0;
  if( lumaOn || chromaOn )
  {
    int mergeType = -1;
    if( ctuLeftAvailable() )
    {
      mergeType += int( decodeBin( CTX_SaoMergeFlag ) );
    }
    if( mergeType < 0 && ctuAboveAvailable() )
    {
      mergeType += int( decodeBin( CTX_SaoMergeFlag ) ) << 1;
    }
    if( mergeType >= 0 )
    {
      mode[0] = SAO_MODE_MERGE;
      type[0] = mergeType;
      if( chromaOn )
      {
        mode[1] = SAO_MODE_MERGE;
        mode[2] = SAO_MODE_MERGE;
        type[1] = mergeType;
        type[2] = mergeType;
      }
    }
    else
    {
      const int firstComp = lumaOn ? 0 : 1;
      const int lastComp  = chromaOn ? 2 : 0;
      for( int comp = firstComp; comp <= lastComp; comp++ )
      {
        if( comp != 2 )
        {
          if( decodeBin( CTX_SaoTypeIdx ) != 0 )
          {
            mode[comp] = SAO_MODE_NEW;
            type[comp] = decodeBinEP() != 0 ? SAO_TYPE_START_EO : SAO_TYPE_START_BO;
          }
        }
        else
        {
          mode[2] = mode[1];
          type[2] = type[1];
        }
        if( mode[comp] == SAO_MODE_OFF )
        {
          continue;
        }
        const int bitDepth      = g_qpBdOffset / 6 + 8;
        const int maxOffsetQVal = ( 1 << ( min( bitDepth, 10 ) - 5 ) ) - 1;
        int       offset[4];
        for( int k = 0; k < 4; k++ )
        {
          offset[k] = unaryMaxEqprob( maxOffsetQVal );
        }
        if( type[comp] == SAO_TYPE_START_BO )
        {
          for( int k = 0; k < 4; k++ )
          {
            if( offset[k] != 0 && decodeBinEP() != 0 )
            {
              offset[k] = -offset[k];
            }
          }
          aux[comp]  = int( decodeBinsEP( 5 ) );
          offs[comp] = ( uint( offset[0] ) & 0xffu ) | ( ( uint( offset[1] ) & 0xffu ) << 8 ) | ( ( uint( offset[2] ) & 0xffu ) << 16 )
                       | ( ( uint( offset[3] ) & 0xffu ) << 24 );
          continue;
        }
        aux[comp] = 0;
        if( comp != 2 )
        {
          type[comp] += int( decodeBinsEP( 2 ) );
        }
        else
        {
          type[2] = type[1];
        }
        offs[comp] = ( uint( offset[0] ) & 0xffu ) | ( ( uint( offset[1] ) & 0xffu ) << 8 ) | ( ( uint( -offset[2] ) & 0xffu ) << 16 )
                     | ( ( uint( -offset[3] ) & 0xffu ) << 24 );
      }
    }
  }
  for( int comp = 0; comp < 3; comp++ )
  {
    setCtuOut( 2 + 2 * comp, uint( mode[comp] ) | uint( type[comp] ) << 4 | uint( aux[comp] ) << 8 );
    setCtuOut( 3 + 2 * comp, offs[comp] );
  }
}

// readAlfCtuFilterIndex
int readAlfCtuFilterIndex()
{
  const int numAps = sliceWord( kSlAlf ) & 0xff;
  int       filtIndex = 0;
  const bool usePrevFilt = numAps > 0 && decodeBin( CTX_AlfUseTemporalFilt ) != 0;
  if( usePrevFilt )
  {
    if( numAps > 1 )
    {
      filtIndex = readTruncBinCode( numAps );
    }
    filtIndex += NUM_FIXED_FILTER_SETS;
  }
  else
  {
    filtIndex = readTruncBinCode( NUM_FIXED_FILTER_SETS );
  }
  return filtIndex;
}

// readAlf, out: enabled (3 bits) | filter index << 3 | Cb's alternative << 8 | Cr's << 12 | CC-ALF Cb's << 16 | Cr's << 20
void readAlf()
{
  const bool leftAvail  = ctuLeftAvailable();
  const bool aboveAvail = ctuAboveAvailable();
  const uint leftAlf    = leftAvail ? ctuOutWord( g_ctu - 1, 8 ) : 0u;
  const uint aboveAlf   = aboveAvail ? ctuOutWord( g_ctu - g_widthInCtus, 8 ) : 0u;
  uint       alf        = 0;
  const uint alfWord    = uint( sliceWord( kSlAlf ) );
  if( sliceFlag( kSlAlfY ) )
  {
    for( int comp = 0; comp < 3; comp++ )
    {
      const bool on = comp == 0 ? true : ( comp == 1 ? sliceFlag( kSlAlfCb ) : sliceFlag( kSlAlfCr ) );
      if( !on )
      {
        continue;
      }
      int ctx = int( ( leftAlf >> uint( comp ) ) & 1u ) + int( ( aboveAlf >> uint( comp ) ) & 1u );
      const uint enabled = decodeBin( CTX_ctbAlfFlag + comp * 3 + ctx );
      alf |= enabled << uint( comp );
      if( comp == 0 && enabled != 0 )
      {
        alf |= uint( readAlfCtuFilterIndex() ) << 3;
      }
      if( comp > 0 && enabled != 0 )
      {
        const int numAlts = int( ( alfWord >> 8 ) & 0xffu );
        int       decoded = 0;
        while( decoded < numAlts - 1 && decodeBin( CTX_ctbAlfAlternative + comp - 1 ) != 0 )
        {
          ++decoded;
        }
        alf |= uint( decoded ) << ( comp == 1 ? 8 : 12 );
      }
    }
  }
  for( int comp = 1; comp < ( g_chromaFormat == 0 ? 1 : 3 ); comp++ )
  {
    const bool on = comp == 1 ? sliceFlag( kSlCcAlfCb ) : sliceFlag( kSlCcAlfCr );
    if( !on )
    {
      continue;
    }
    const uint shift = comp == 1 ? 16u : 20u;
    int        ctxt  = ( ( leftAlf >> shift ) & 7u ) != 0 ? 1 : 0;
    ctxt += ( ( aboveAlf >> shift ) & 7u ) != 0 ? 1 : 0;
    ctxt += comp == 2 ? 3 : 0;
    int idcVal = int( decodeBin( CTX_CcAlfFilterControlFlag + ctxt ) );
    if( idcVal != 0 )
    {
      const int filterCount = int( ( alfWord >> ( comp == 1 ? 16u : 24u ) ) & 0xffu );
      while( idcVal != filterCount && decodeBinEP() != 0 )
      {
        idcVal++;
      }
    }
    alf |= uint( idcVal ) << shift;
  }
  setCtuOut( 8, alf );
}

// the chroma tree's or the luma tree's coding tree of a part of the CTU's quad split (dual tree, CTUs of 128: the
// partitioner as dt_implicit_qt_split leaves it for the part)
void partitionerAtQuadrant( int chType, int q )
{
  initCtu( chType );
  splitCurrArea( CU_QUAD_SPLIT );
  for( int i = 0; i < q; i++ )
  {
    nextPart();
  }
}

// CABACReader::coding_tree_unit
void codingTreeUnit( inout int prevQp[2] )
{
  g_numCus     = 0;
  g_numTus     = 0;
  g_dmvrOffset = 0;
  PROF_BEGIN( saoStart );
  sao();
  readAlf();
  PROF_END( kProfSao, saoStart );

  // the CTU's trees: the single tree; the dual tree's luma, then chroma; with CTUs of 128 (dt_implicit_qt_split) the
  // CTU's quad split, each part's luma tree then its chroma tree (one call of codingTree: inlined, each would copy it)
  CuCtx      ctxL  = newCuCtx( prevQp[0] );
  CuCtx      ctxC  = newCuCtx( prevQp[1] );
  const bool dual  = g_dualTree && g_chromaFormat != 0;
  const bool quads = dual && g_ctuSize > 64;
  if( quads )
  {
    initCtu( 0 );
    if( picWord( kPicUseDqp ) != 0 && currQgEnable() )
    {
      ctxL.qgStart    = true;
      ctxL.isDQPCoded = false;
      ctxC.qgStart    = true;
      ctxC.isDQPCoded = false;
    }
    if( sliceFlag( kSlChromaQpAdj ) && currQgChromaEnable() )
    {
      ctxL.isChromaQpAdjCoded = false;
      ctxC.isChromaQpAdjCoded = false;
      g_csChromaQpAdj         = 0;
    }
  }
  const int numTrees = !dual ? 1 : ( quads ? 8 : 2 );
  for( int i = 0; i < numTrees && !g_error; i++ )
  {
    const int ch = dual ? ( i & 1 ) : 0;
    if( quads )
    {
      partitionerAtQuadrant( ch, i >> 1 );
      if( !inPicture( currArea() ) )
      {
        continue;
      }
    }
    else
    {
      initCtu( ch );
    }
    CuCtx ctx = ch == 0 ? ctxL : ctxC;
    codingTree( ctx );
    if( ch == 0 )
    {
      ctxL = ctx;
    }
    else
    {
      ctxC = ctx;
    }
  }
  prevQp[0] = ctxL.qp;
  if( dual )
  {
    prevQp[1] = ctxC.qp;
  }
}
