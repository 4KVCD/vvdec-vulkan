// The coding tree part of parse.slang, which includes this file. It holds:
// - the maps of the CUs parsed so far, where the context derivations look up neighbouring CUs;
// - the partitioner (vvdec's Partitioner), which keeps the stack of splits from the CTU down to the current area;
// - the split syntax (CABACReader::split_cu_mode, mode_constraint) and the coding tree (CABACReader::coding_tree),
//   walked with an explicit stack of frames because shaders cannot recurse; each leaf of the tree is a CU, which
//   codingUnitAtPart (parsecu.slang) parses;
// - the coding tree unit (CABACReader::coding_tree_unit): SAO's and ALF's syntax, then the CTU's coding tree or, with
//   the dual tree, its luma and chroma trees.

// --------------------------------------------------------------------------------------------------------------------
// The maps of the CUs parsed so far around the CTU. Each holds a word pair per 4x4 luma block that describes the CU
// covering it: s_map covers the CTU itself, with one map per channel type (the luma tree's CUs and the chroma tree's;
// a single tree's CUs are in both); s_above covers the bottom row of 4x4 blocks of the CTU above, and s_left the
// right column of the CTU to the left. An entry is zero where no CU has been parsed, and in CTUs out of reach (another
// slice or tile). The word pair:
//   x: the CU's index in its CTU + 1 (12 bits) | prediction mode << 12 | skip << 14 | affine << 15 | MIP << 16 |
//      ISP mode << 17 | QT depth << 19 | depth << 22 | log2 width << 26 | log2 height << 29, the size being that of
//      the CU's block of the map's channel type, in that channel's samples
//   y: the CU's luma intra mode | its QP << 8 (signed bytes)
// A CU's entries are written when the CU is done, so a lookup always finds another CU, completely parsed.
// --------------------------------------------------------------------------------------------------------------------

// The maps are as large as the largest CTU of the shader's variant: MAX_CTU, 128 samples, or 64 for parse_ctu64,
// whose maps take a quarter of the shared memory (the parse's workgroups stay resident for long, and the shared
// memory they take the reconstruction's shaders cannot have). pc.bottomRows keeps 32 entries a channel type a CTU.
#ifndef MAX_CTU
#  define MAX_CTU 128
#endif
static const int kMapRow     = MAX_CTU / 4;         // a channel type's map: entries per row (one a 4x4 block)
static const int kMapChannel = kMapRow * kMapRow;   // entries per channel type
groupshared uint2 s_map[2 * kMapChannel];   // [channel type][y >> 2][x >> 2], from the CTU's top-left corner
groupshared uint2 s_above[2 * kMapRow];     // [channel type][x >> 2]: the bottom row of the CTU above
groupshared uint2 s_left[2 * kMapRow];      // [channel type][y >> 2]: the right column of the CTU to the left

// Where entry i of s_above (or s_left) is in pc.bottomRows' entries of CTU ctu
uint bottomRowsAt( int ctu, int i )
{
  return uint( ctu * 64 + ( i / kMapRow ) * 32 + i % kMapRow );
}

static bool g_leftAvail;    // the CTU to the left is in the same slice and tile
static bool g_aboveAvail;   // the CTU above is in the same slice and tile
static bool g_tileFirstCol; // the CTU is in its tile's first column

// The fields of a map entry
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

// Whether CTUs a and b belong to the same slice and the same tile
bool sameSliceTile( int a, int b )
{
  return uni( pc.ctuInfo[a * 2] ) == uni( pc.ctuInfo[b * 2] ) && ( uni( pc.ctuInfo[a * 2 + 1] ) & 0xffffu ) == ( uni( pc.ctuInfo[b * 2 + 1] ) & 0xffffu );
}

// The map entry of the CU of channel type ch at luma position (x, y), as CodingStructure::getCU finds it, for
// positions in the CTU, in the row of 4x4 blocks just above it or in the column just left of it. As with
// getCURestricted, CUs out of reach (outside the picture, or in another slice or tile) and other positions give zero.
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
    return uni( s_map[ch * kMapChannel + ( dy >> 2 ) * kMapRow + ( dx >> 2 )] );
  }
  if( dy < 0 && dy >= -4 && dx >= 0 && dx < g_ctuSize )
  {
    return uni( s_above[ch * kMapRow + ( dx >> 2 )] );
  }
  if( dx < 0 && dx >= -4 && dy >= 0 && dy < g_ctuSize )
  {
    return uni( s_left[ch * kMapRow + ( dy >> 2 )] );
  }
  return uint2( 0, 0 );
}

// CodingStructure::getCURestricted( pos, curCu, chType ): as cuAt, but within the CTU a CU that comes after the
// current CU (whose index is cur) gives zero
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
// The partitioner (vvdec's Partitioner). Its current level (g_lv) and the rest of its state (g_p) live in each thread,
// the same in all threads; the levels below the current one wait in shared memory (s_lv...).
// --------------------------------------------------------------------------------------------------------------------

static const int kMaxLevels = 24;   // the deepest stack of levels (deeper gives up the substream)

// Per level below the current one (as Level below): the split that made it (CTU_LEVEL at level 0, whose one part is
// the CTU), the index of its current part, its number of parts, the CUs above and left of its current part
// (setNeighborCu), its quantization group flags, and the area it splits
groupshared int s_lvSplit[kMaxLevels];
groupshared int s_lvIdx[kMaxLevels];
groupshared int s_lvNum[kMaxLevels];
groupshared uint2 s_lvAbove[kMaxLevels];
groupshared uint2 s_lvLeft[kMaxLevels];
groupshared int s_lvQg[kMaxLevels];    // qgEnable | qgChromaEnable << 1
groupshared int4 s_lvArea[kMaxLevels]; // the area split (luma samples: x, y, w, h)

// The partitioner's state besides its levels, with the names of vvdec's Partitioner
struct Part
{
  int top;   // the current level, the top of the stack
  int currDepth;
  int currQtDepth;
  int currSubdiv;
  int currMtDepth;
  int currImplicitBtDepth;
  int chType;
  int treeType;
  int modeType;
  // The partition limits, sizes in luma samples: maxBTD (the deepest multi-type tree), maxBtSize, maxTtSize and
  // minQtSize from kSlPartLuma or kSlPartChroma; minBtSize and minTtSize are the smallest coding block's size
  int maxBTD;
  int maxBtSize;
  int minBtSize;
  int maxTtSize;
  int minTtSize;
  int minQtSize;
};

static Part g_p;

// The partitioner's current level, the top of its stack, in registers (each read of shared memory is a round trip of
// tens of cycles for the one wave that parses): the levels below it wait in shared memory (s_lv...) while its parts
// are parsed (saveLevel when a part is split, loadLevel when its parts are done)
struct Level
{
  int   split;   // the split that made the level's parts (CTU_LEVEL at the CTU)
  int   idx;     // the current part
  int   num;     // the parts
  int   qg;      // qgEnable | qgChromaEnable << 1
  int4  area;    // the area split (luma samples: x, y, w, h)
  uint2 above;   // the CUs above and left of the current part (setNeighborCu)
  uint2 left;
};

static Level g_lv;

// Keeps the current level in shared memory as level l
void saveLevel( int l )
{
  s_lvSplit[l] = g_lv.split;
  s_lvIdx[l]   = g_lv.idx;
  s_lvNum[l]   = g_lv.num;
  s_lvQg[l]    = g_lv.qg;
  s_lvArea[l]  = g_lv.area;
  s_lvAbove[l] = g_lv.above;
  s_lvLeft[l]  = g_lv.left;
}

// Makes level l, kept in shared memory, the current level again
void loadLevel( int l )
{
  g_lv.split = uni( s_lvSplit[l] );
  g_lv.idx   = uni( s_lvIdx[l] );
  g_lv.num   = uni( s_lvNum[l] );
  g_lv.qg    = uni( s_lvQg[l] );
  g_lv.area  = int4( uni( s_lvArea[l].x ), uni( s_lvArea[l].y ), uni( s_lvArea[l].z ), uni( s_lvArea[l].w ) );
  g_lv.above = uni( s_lvAbove[l] );
  g_lv.left  = uni( s_lvLeft[l] );
}

// The area of part idx when area a is split with split; CTU_LEVEL's one part is a itself
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

// The area of the current part (Partitioner::currArea), in luma samples
int4 currArea()
{
  return partArea( g_lv.split, g_lv.idx, g_lv.area );
}

// Whether the current part may start a quantization group for the delta QP (Partitioner::currQgEnable) and for the
// chroma QP offsets (currQgChromaEnable)
bool currQgEnable()
{
  return ( g_lv.qg & 1 ) != 0;
}

bool currQgChromaEnable()
{
  return ( g_lv.qg & 2 ) != 0;
}

// Partitioner::isSepTree: the current part is in a separate luma or chroma tree (the dual tree, or a local dual tree)
bool isSepTreeP()
{
  return g_p.treeType != TREE_D || g_dualTree;
}

// The width (height) of one sample of channel type ch, in luma samples
int stepX( int ch )
{
  return ch == 0 ? 1 : 1 << g_csx;
}

int stepY( int ch )
{
  return ch == 0 ? 1 : 1 << g_csy;
}

// Records in the current level the CUs above and to the left of its current part, in the partitioner's channel type
// (the chroma tree's for TREE_C), as vvdec's setNeighborCu does
void setNeighborCu()
{
  const int  ch = g_p.treeType == TREE_C ? 1 : g_p.chType;
  const int4 a  = currArea();
  g_lv.above    = cuAt( a.x, a.y - stepY( ch ), ch );
  g_lv.left     = cuAt( a.x - stepX( ch ), a.y, ch );
}

// Sets the partitioner up for the CTU's tree of channel type chType (Partitioner::initCtu): level 0 is the CTU
void initCtu( int chType )
{
  g_p.top                 = 0;
  g_p.currDepth           = 0;
  g_p.currMtDepth         = 0;
  g_p.currQtDepth         = 0;
  g_p.currSubdiv          = 0;
  g_p.currImplicitBtDepth = 0;
  g_p.chType              = chType;
  g_p.treeType            = TREE_D;
  g_p.modeType            = MODE_TYPE_ALL;
  g_lv.split              = CTU_LEVEL;
  g_lv.idx                = 0;
  g_lv.num                = 1;
  g_lv.qg                 = 3;
  g_lv.area               = int4( g_ctuX << g_log2Ctu, g_ctuY << g_log2Ctu, g_ctuSize, g_ctuSize );
  setNeighborCu();
  // The partition limits, picked as vvdec does with GDR_ADJ: the luma (or single tree) limits of intra or inter
  // slices, or the dual tree chroma's
  const uint lim          = uint( sliceWord( chType == 1 && g_dualTree ? kSlPartChroma : kSlPartLuma ) );
  g_p.maxBTD              = int( lim & 0xffu );
  g_p.maxBtSize           = int( ( lim >> 8 ) & 0xffu );
  g_p.maxTtSize           = int( ( lim >> 16 ) & 0xffu );
  g_p.minQtSize           = int( lim >> 24 );
  g_p.minBtSize           = 1 << g_log2MinCb;
  g_p.minTtSize           = 1 << g_log2MinCb;
}

// Whether area a reaches past the picture's right or bottom edge (its split is then implicit)
bool outsidePicture( int4 a )
{
  return a.x + a.z - 1 >= g_width || a.y + a.w - 1 >= g_height;
}

// Splits the current part with split, pushing a level whose current part is the first one (Partitioner::splitCurrArea,
// for the CU splits). The new level starts with its parent's neighbours, as the first part shares its top-left corner.
void splitCurrArea( int split )
{
  const int4 area           = currArea();
  const bool isImplicit     = outsidePicture( area );
  bool       qgEnable       = currQgEnable();
  bool       qgChromaEnable = currQgChromaEnable();
  const int  last           = g_p.top;
  const int  l              = last + 1;
  saveLevel( last );   // (the current level waits below the new one; the new one starts with its neighbours)
  g_p.top                   = l;
  g_lv.split                = split;
  g_lv.idx                  = 0;
  g_lv.num                  = split == CU_QUAD_SPLIT ? 4 : ( split == CU_TRIH_SPLIT || split == CU_TRIV_SPLIT ? 3 : 2 );
  g_lv.area                 = area;
  if( split == CU_QUAD_SPLIT )
  {
    g_p.currMtDepth = 0;
    g_p.currQtDepth++;
    g_p.currSubdiv++;
  }
  else if( split == CU_HORZ_SPLIT || split == CU_VERT_SPLIT )
  {
    if( isImplicit )
    {
      g_p.currImplicitBtDepth++;
    }
    g_p.currMtDepth++;
  }
  else
  {
    g_p.currMtDepth++;
    g_p.currSubdiv++;
  }
  g_p.currDepth++;
  g_p.currSubdiv++;
  const uint subdiv = uint( sliceWord( kSlSubdiv ) );
  qgEnable          = qgEnable && g_p.currSubdiv <= int( subdiv & 0xffu );
  qgChromaEnable    = qgChromaEnable && g_p.currSubdiv <= int( ( subdiv >> 8 ) & 0xffu );
  g_lv.qg           = ( qgEnable ? 1 : 0 ) | ( qgChromaEnable ? 2 : 0 );
}

// Moves to the next part of the current level; returns false after its last part (Partitioner::nextPart, for the CU
// splits)
bool nextPart()
{
  const int idx = g_lv.idx + 1;
  g_lv.idx      = idx;
  if( idx < g_lv.num )
  {
    setNeighborCu();
    const int split = g_lv.split;
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

// Pops the current level, back to the part it split (Partitioner::exitCurrSplit, for the CU splits)
void exitCurrSplit()
{
  const int l         = g_p.top;
  const int currSplit = g_lv.split;
  const int currIndex = g_lv.idx;
  g_p.top             = l - 1;
  loadLevel( l - 1 );
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

// The splits the current part allows (Partitioner::canSplit), as canNo | canQt << 1 | canBh << 2 | canBv << 3 |
// canTh << 4 | canTv << 5
int canSplit()
{
  bool       canNo = true, canQt = true, canBh = true, canTh = true, canBv = true, canTv = true;
  bool       canBtt = g_p.currMtDepth < ( g_p.maxBTD + g_p.currImplicitBtDepth );
  const int4 area   = currArea();
  const bool chroma = g_p.chType == 1;
  const int  cw     = area.z >> g_csx;   // the part's chroma block, which the chroma tree's limits look at
  const int  chh    = area.w >> g_csy;

  if( g_dualTree && ( area.z > 64 || area.w > 64 ) )
  {
    return 2;   // QT only: in the dual tree, a block larger than 64 is split in four (no bin is coded)
  }
  if( g_p.treeType == TREE_C )
  {
    return 1;   // no split: a local dual tree's chroma CU covers its whole area
  }
  const int  lastSplit = g_lv.split;
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
    if( ( lastSplit == CU_TRIH_SPLIT || lastSplit == CU_TRIV_SPLIT ) && g_lv.idx == 1 )
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

// Parses how the current part is split (CABACReader::split_cu_mode): split_cu_flag, split_qt_flag,
// mtt_split_cu_vertical_flag and mtt_split_cu_binary_flag, each only when canSplit leaves a choice. The contexts come
// from the sizes and QT depths of the CUs left and above.
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
  const uint2 cuLeft    = g_lv.left;
  const uint2 cuAbove   = g_lv.above;
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

// vvdec's signalModeCons: whether the parts of split inherit the mode type (0, LDT_MODE_TYPE_INHERIT), are intra
// only (1, LDT_MODE_TYPE_INFER), or read it from mode_constraint_flag (2, LDT_MODE_TYPE_SIGNAL). Small chroma blocks
// of the single tree lead to a local dual tree this way.
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

// The mode type of the parts of split (CABACReader::mode_constraint)
int modeConstraint( int split )
{
  const int val = signalModeCons( split );
  if( val == 2 )
  {
    const uint2 cuLeft  = g_lv.left;
    const uint2 cuAbove = g_lv.above;
    const int   ctxIdx  = ( ( infoValid( cuAbove ) && infoPredMode( cuAbove ) == MODE_INTRA ) || ( infoValid( cuLeft ) && infoPredMode( cuLeft ) == MODE_INTRA ) ) ? 1 : 0;
    return decodeBin( CTX_ModeConsFlag + ctxIdx ) != 0 ? MODE_TYPE_INTRA : MODE_TYPE_INTER;
  }
  if( val == 1 )
  {
    return MODE_TYPE_INTRA;
  }
  return g_p.modeType;
}

// The split made at the given depth on the way down to the current CU, read from the partitioner's levels
// (CU::getSplitAtDepth); CU_DONT_SPLIT from the CU's own depth on
int splitAtDepth( int depth )
{
  // Levels 1 and up hold the splits that led to the current area: the split at depth d is level d + 1's
  if( depth >= g_p.currDepth )
  {
    return CU_DONT_SPLIT;
  }
  return depth + 1 == g_p.top ? g_lv.split : uni( s_lvSplit[depth + 1] );
}

// --------------------------------------------------------------------------------------------------------------------
// The coding tree (CABACReader::coding_tree). Shaders cannot recurse, so codingTree walks the tree with a stack of
// frames instead: frame f stands for the coding_tree call at depth f, and s_frame keeps what a call that split its
// area needs once its parts are done.
// --------------------------------------------------------------------------------------------------------------------

// vvdec's CUCtx: what the coding tree carries from CU to CU
struct CuCtx
{
  bool qgStart;              // a quantization group starts: the next CU predicts its QP
  bool isDQPCoded;           // the quantization group's cu_qp_delta has been parsed
  bool isChromaQpAdjCoded;   // the chroma quantization group's cu_chroma_qp_offset has been parsed
  int  qp;                   // the QP so far (at a quantization group's start, the previous QP that predicts the next)
  // What the CU's residual coding found, for LFNST's and MTS's indices (residual_lfnst_mode, mts_idx)
  bool violatesLfnstLuma;
  bool violatesLfnstChroma;
  bool lfnstLastScanPos;
  bool violatesMtsCoeffConstraint;
  bool mtsLastScanPos;
};

// A CuCtx that starts from QP qp
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

static const int kMaxFrames = 32;   // the deepest stack of frames (deeper gives up the substream)
// Per frame of a split area: bit 0 says what the frame does (0: it parses its parts, 1: it parses the chroma CU of its
// whole area), bit 1 is chromaNotSplit, and the bits from 2 hold modeTypeParent, the mode type to restore at its end
groupshared int s_frame[kMaxFrames];

// Whether area a starts inside the picture (coding_tree skips the parts that do not)
bool inPicture( int4 a )
{
  return a.x < g_width && a.y < g_height;
}

// Parses the coding tree of the partitioner's current area (CABACReader::coding_tree). Each pass of the loop either
// enters a part (enter: its split is parsed, and a part that is not split is a CU, which codingUnitAtPart parses) or
// resumes frame f after one of its parts.
void codingTree( inout CuCtx ctx )
{
  int  f     = 0;
  bool enter = true;
  while( f >= 0 && !g_error )
  {
    if( enter )
    {
      // A quantization group may start here, for the delta QP (luma) and for the chroma QP offsets
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
        // Not split: the part is a CU. Then back to the parent's frame.
        codingUnitAtPart( ctx );
        f--;
        enter = false;
        continue;
      }
      if( f + 1 >= kMaxFrames || g_p.top + 1 >= kMaxLevels )
      {
        g_error = true;   // (too deep for the stacks)
        break;
      }
      // Split: work out the parts' mode type. When they must be intra but the parent allowed every mode, the chroma
      // is not split (chromaNotSplit): the parts form a luma tree, and the chroma follows as one CU. Then enter the
      // first part, or skip it when it starts outside the picture.
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
    // Frame f resumes after one of its parts (or after skipping one outside the picture)
    const int fr = uni( s_frame[f] );
    if( ( fr & 1 ) == 0 )
    {
      if( nextPart() )
      {
        // the next part: enter it, unless it starts outside the picture
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
        // chromaNotSplit: after the luma parts, the chroma of the whole area follows as one CU of the chroma tree
        g_p.chType   = 1;
        g_p.treeType = TREE_C;
        setNeighborCu();
        s_frame[f] = fr | 1;
        f++;
        enter = true;
        continue;
      }
      // the frame ends: restore the parent's mode type and go back to the parent's frame
      g_p.modeType = fr >> 2;
      f--;
      continue;
    }
    // The chroma CU of a chromaNotSplit frame is done: back to the single tree, then to the parent's frame
    g_p.chType   = 0;
    g_p.treeType = TREE_D;
    g_p.modeType = fr >> 2;
    f--;
  }
}

// --------------------------------------------------------------------------------------------------------------------
// The coding tree unit: SAO (CABACReader::sao), ALF (readAlf), then the coding tree or trees
// --------------------------------------------------------------------------------------------------------------------

// Word w of a CTU's ctuOut
uint ctuOutWord( int ctu, int w )
{
  return uni( pc.ctuOut[ctu * kCtuOutWords + w] );
}

// Sets word w of the current CTU's ctuOut
void setCtuOut( int w, uint v )
{
  pc.ctuOut[g_ctu * kCtuOutWords + w] = v;
}

// Whether the CTU to the left (above) is in the same slice and tile, as SAO's merge flags and ALF's contexts need
bool ctuLeftAvailable()
{
  return g_leftAvail;
}

bool ctuAboveAvailable()
{
  return g_aboveAvail;
}

// Parses the CTU's SAO parameters (CABACReader::sao) and writes 2 ctuOut words per component, from word 2: modeIdc |
// typeIdc << 4 | typeAuxInfo << 8, then the 4 offsets as bytes -- a band offset's in band order from its first band
// (typeAuxInfo), an edge offset's of the classes 0, 1, 3 and 4. A merge stays SAO_MODE_MERGE with typeIdc 0 (left)
// or 1 (above), as vvdec's parse leaves it.
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

// Parses the CTU's luma filter set (CABACReader::readAlfCtuFilterIndex): one of the slice's APSs (alf_use_aps_flag,
// alf_luma_prev_filter_idx), numbered from NUM_FIXED_FILTER_SETS, or one of the fixed sets (alf_luma_fixed_filter_idx)
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

// Parses the CTU's ALF and CC-ALF syntax (CABACReader::readAlf) and writes ctuOut word 8: the enable flags of Y, Cb
// and Cr (bits 0..2) | the luma filter set << 3 | Cb's alternative filter << 8 | Cr's << 12 | Cb's CC-ALF filter << 16
// | Cr's << 20 (0: off). The contexts look at the same word of the CTUs to the left and above.
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

// Dual tree with CTUs of 128: sets the partitioner of channel type chType to quadrant q of the CTU's implicit quad
// split, as dt_implicit_qt_split leaves it before it parses that quadrant's coding tree
void partitionerAtQuadrant( int chType, int q )
{
  initCtu( chType );
  splitCurrArea( CU_QUAD_SPLIT );
  for( int i = 0; i < q; i++ )
  {
    nextPart();
  }
}

// Parses a CTU (CABACReader::coding_tree_unit): SAO, ALF, then its coding trees. prevQp holds the QPs carried from
// CTU to CTU: the luma (or single) tree's and the dual tree chroma's.
void codingTreeUnit( inout int prevQp[2] )
{
  g_numCus     = 0;
  g_numTus     = 0;
  g_dmvrOffset = 0;
  PROF_BEGIN( saoStart );
  sao();
  readAlf();
  PROF_END( kProfSao, saoStart );

  // The CTU's trees: one single tree; or the dual tree's luma tree, then its chroma tree; or, with CTUs of 128 in the
  // dual tree, the implicit quad split of dt_implicit_qt_split, where each quadrant's luma tree comes before its
  // chroma tree. All go through the one call of codingTree below: every call is inlined, so each call site would
  // copy the whole coding tree.
  CuCtx      ctxL  = newCuCtx( prevQp[0] );
  CuCtx      ctxC  = newCuCtx( prevQp[1] );
  const bool dual  = g_dualTree && g_chromaFormat != 0;
  const bool quads = dual && g_ctuSize > 64;
  if( quads )
  {
    // dt_implicit_qt_split: the CTU starts quantization groups for both trees
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
