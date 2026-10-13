// Motion derivation on the GPU (vvdec's DecCu::TaskDeriveCtuMotionInfo /
// xDeriveCUMV), from the CUs as the parse left them (GpuFrontEnd.h's records).
// Two variants:
//
// - CUMAP records which CU covers each 4x4 luma block of the picture (luma
//   tree), one workgroup per CU.
// - The main variant derives the CUs of each CTU row of a tile column (a
//   segment) in decoding order, one 32-thread workgroup per segment (the
//   history-based candidates run along a row, starting at each tile's left
//   edge). The segments of a tile run as a wavefront: a CTU starts once the
//   segment above is past the CTU above-right, as vvdec's MIDER tasks wait
//   (with WPP, past the CTU above). Tiles do not wait for each other: no CU is
//   the neighbour of a CU in another tile. A CU's motion goes to its motion
//   record (what vvdec keeps in the CodingUnit: control-point motion,
//   reference indices, BCW, ...) and to the motion of its 4x4 luma blocks
//   (vvdec's MotionInfo, CTU by CTU as vvdec keeps it), where its neighbours
//   look it up.
//
// It covers all of vvdec's motion tools: regular merge (spatial, temporal,
// history-based, pairwise and zero candidates), MMVD, CIIP (regular merge),
// AMVP with AMVR (SMVD's motion comes from the parse), BCW inheritance, IBC
// merge and AMVP with IBC's own history, the subblock merge (SbTMVP, the
// inherited and constructed affine candidates), affine AMVP, and GPM.

#include "common.slang"

// LITE=1: the variant for pictures whose SPS turns off GPM, affine and SbTMVP, without their code (GpuRecon picks
// it). The common paths' code is then a fraction of the shader's, and more of it stays in the instruction cache.
#ifndef LITE
#  define LITE 0
#endif

// The thread's lane in its row's 32-lane workgroup; loops over a CU's blocks take every 32nd block, from this lane.
static int g_lane = 0;

struct Params
{
  uint*         cus;        // GpuFrontEnd.h's CuRecord, 32 words each: CTU by CTU, each CTU's CUs in decoding order
  uint*         ctus;       // CtuRecord: 4 words each
  uint*         slices;     // the picture's slice table: the slice of each CTU, then the SliceRecords (64 words each)
  uint*         cuMap;      // the index of the CU at each 4x4 luma block, widthIn4 entries per row
  int*          cuMotion;   // out: each CU's motion record, 16 words (CuMotion)
  int*          motion;     // out: each 4x4 luma block's motion as vvdec's MotionInfo (5 words), num4x4 per CTU
  Atomic<uint>* progress;   // per CTU row: the CTUs done; then the rows taken (all zeroed before the dispatch)
  uint16_t*     subpics;    // subpicOn: per CTU, its subpicture's rectangle (if treated as a picture) or the picture's
  int           numCus;
  int           widthInCtus;
  int           heightInCtus;
  int           log2Ctu;
  int           width;      // luma samples
  int           height;
  int           widthIn4;
  int           plevel;     // log2 of the parallel merge level
  int           wpp;        // entropy coding sync is on (the CTU above-right is then out of reach)
  int           maxNumMergeCand;   // the SPS's
  int           subpicOn;
  int           sps;        // the SPS's flags: 1 SbTMVP, 2 affine, 4 the 6-parameter affine model
  uint*         cuMapC;     // the same as cuMap for the chroma tree (in a single tree, the luma tree's CUs)
  int           csx;        // chroma subsampling shifts
  int           csy;
  int           tileCols;   // the picture's tile columns (the segments of a CTU row)
  int           colOn;      // 1: the picture's colocated motion goes to colMotion
  // out: the picture's colocated motion, as vvdec's storeColMotion keeps it: per CTU, an entry of 5 words per 8x8
  // block, the motion of the block's top-left 4x4 block. TMVP in later pictures reads it; inter.slang writes DMVR's
  // refined motion into it afterwards.
  int*          colMotion;
};

[[vk::push_constant]] Params pc;

// --------------------------------------------------------------------------------------------------------------------
// The records (GpuFrontEnd.h)
// --------------------------------------------------------------------------------------------------------------------

static const uint kMerge   = 1;
static const uint kSkip    = 2;
static const uint kMmvd    = 4;
static const uint kAffine  = 8;
static const uint kGeo     = 16;
static const uint kCiip    = 32;
static const uint kChroma  = 64;   // a CU of the chroma tree
static const uint kAffine6 = 128;

static const int MODE_INTER = 0;
static const int MODE_INTRA = 1;
static const int MODE_IBC   = 2;

static const int MRG_TYPE_DEFAULT_N   = 0;
static const int MRG_TYPE_SUBPU_ATMVP = 1;
static const int MRG_TYPE_IBC         = 2;

// The kinds of motion a CU's motion record holds
static const int kNotInter    = 0;
static const int kTranslation = 1;
static const int kIbc         = 3;
static const int kAffineCu    = 4;   // affine (and the subblock merge's zero candidates)
static const int kSbtmvp      = 5;
static const int kGpm         = 6;

struct Cu
{
  int  x, y, w, h;
  int  predMode;
  uint flags;
  uint w2;
  uint w3;
  int  index;
};

// Shared-memory copy of words 0-3 of each CU record of the current CTU (the fields loadCu reads); cacheCtuCus fills
// it when a CTU starts. g_cachedFirst is the CTU's first CU and g_cachedCount the number of CUs cached.
static const int kCachedCus = 512;
groupshared uint s_cuRec[4 * kCachedCus];
static int g_cachedFirst = 0;
static int g_cachedCount = 0;

Cu loadCu( int index )
{
  uint4 r;
  if( uint( index - g_cachedFirst ) < uint( g_cachedCount ) )
  {
    const int at = ( index - g_cachedFirst ) * 4;
    r            = uint4( s_cuRec[at], s_cuRec[at + 1], s_cuRec[at + 2], s_cuRec[at + 3] );
  }
  else
  {
    uint* g = pc.cus + index * 32;
    r       = uint4( g[0], g[1], g[2], g[3] );
  }
  Cu cu;
  cu.x        = int( r.x & 0xffff );
  cu.y        = int( r.x >> 16 );
  cu.w        = int( r.y & 0xff );
  cu.h        = int( ( r.y >> 8 ) & 0xff );
  cu.predMode = int( ( r.y >> 16 ) & 3 );
  cu.flags    = r.y >> 18;
  cu.w2       = r.z;
  cu.w3       = r.w;
  cu.index    = index;
  return cu;
}

// Copies the CTU's CU records into shared memory; all threads of the workgroup call it.
void cacheCtuCus( int first, int count )
{
  g_cachedFirst = first;
  g_cachedCount = min( count, kCachedCus );
  const int n   = g_cachedCount * 4;
  for( int base = 0; base < n; base += 128 )
  {
    // (each thread issues its four loads together)
    uint v[4];
    for( int j = 0; j < 4; j++ )
    {
      const int i = base + j * 32 + g_lane;
      v[j]        = i < n ? pc.cus[( first + ( i >> 2 ) ) * 32 + ( i & 3 )] : 0u;
    }
    for( int j = 0; j < 4; j++ )
    {
      const int i = base + j * 32 + g_lane;
      if( i < n )
      {
        s_cuRec[i] = v[j];
      }
    }
  }
  GroupMemoryBarrierWithGroupSync();
}

int  mergeIdx( Cu cu ) { return int( cu.w2 & 7 ); }
int  mmvdIdx( Cu cu ) { return int( ( cu.w2 >> 3 ) & 63 ); }
int  geoSplitDir( Cu cu ) { return int( ( cu.w2 >> 9 ) & 63 ); }
int  geoIdx( Cu cu, int i ) { return int( ( cu.w2 >> ( 15 + 3 * i ) ) & 7 ); }
int  imvOf( Cu cu ) { return int( ( cu.w2 >> 21 ) & 3 ); }
int  bcwIdxOf( Cu cu ) { return int( ( cu.w2 >> 23 ) & 7 ); }
int  interDirOf( Cu cu ) { return int( ( cu.w2 >> 26 ) & 3 ); }
int  mvpIdx( Cu cu, int list ) { return int( ( cu.w2 >> ( 28 + list ) ) & 1 ); }
int  refIdxOf( Cu cu, int list ) { return int( int8_t( ( cu.w3 >> ( 8 * list ) ) & 0xff ) ); }
bool affine6Of( Cu cu ) { return ( cu.flags & kAffine6 ) != 0; }
int2 mvdOf( Cu cu, int list, int k )
{
  int* r = (int*) ( pc.cus + cu.index * 32 + 4 + list * 6 + k * 2 );
  return int2( r[0], r[1] );
}

// CtuRecord fields: the first CU, the CU count, the slice and the tile
uint ctuSliceIdx( int ctu ) { return pc.ctus[ctu * 4 + 1] >> 16; }
uint ctuTile( int ctu ) { return pc.ctus[ctu * 4 + 2] >> 16; }
bool ctuTileColStart( int ctu ) { return ( pc.ctus[ctu * 4 + 2] & 1 ) != 0; }
int  ctuOf( int x, int y ) { return ( y >> pc.log2Ctu ) * pc.widthInCtus + ( x >> pc.log2Ctu ); }

// A slice record (GpuFrontEnd.h's SliceRecord, 64 words), read through word( k ): Slice reads it from memory;
// CurSlice is the current CTU's, which every CU reads again and again, from shared memory (loadCurSlice); ColSlice is
// one of the colocated picture's, from shared memory when it is the one there (loadColSlices), else from memory.
interface ISlice
{
  uint word( int k );
};

static const int kSliceWords = 64;

struct Slice : ISlice
{
  uint* r;
  uint  word( int k ) { return r[k]; }
};

groupshared uint s_curSlice[kSliceWords];
static uint      g_curSliceIdx = 0xffffffffu;   // the slice whose record s_curSlice holds (its index in pc.slices)

struct CurSlice : ISlice
{
  uint word( int k ) { return s_curSlice[k]; }
};

groupshared uint s_colSlice[kSliceWords];
static uint      g_colSliceIdx = 0xffffffffu;   // the colocated picture's slice whose record s_colSlice holds

struct ColSlice : ISlice
{
  uint* r;        // the record in memory
  bool  cached;   // s_colSlice holds it
  uint  word( int k ) { return cached ? s_colSlice[k] : r[k]; }
};

// The colocated picture's slices of the current CTU and of the one right of it (loadColSlices)
static uint g_colCtuSlice[2];

Slice sliceOfCtu( int ctu )
{
  Slice s;
  s.r = pc.slices + pc.widthInCtus * pc.heightInCtus + ctuSliceIdx( ctu ) * kSliceWords;
  return s;
}

int   slicePoc<S : ISlice>( S s ) { return int( s.word( 0 ) ); }
int   numRefIdx<S : ISlice>( S s, int list ) { return int( ( s.word( 1 ) >> ( 8 * list ) ) & 0xff ); }
bool  isInterB<S : ISlice>( S s ) { return ( s.word( 1 ) & ( 1u << 16 ) ) != 0; }
bool  checkLDC<S : ISlice>( S s ) { return ( s.word( 1 ) & ( 1u << 17 ) ) != 0; }
bool  colFromL0<S : ISlice>( S s ) { return ( s.word( 1 ) & ( 1u << 18 ) ) != 0; }
bool  tmvp<S : ISlice>( S s ) { return ( s.word( 1 ) & ( 1u << 19 ) ) != 0; }
bool  mmvdFullPel<S : ISlice>( S s ) { return ( s.word( 1 ) & ( 1u << 20 ) ) != 0; }
bool  hasColPic<S : ISlice>( S s ) { return ( s.word( 1 ) & ( 1u << 21 ) ) != 0; }
bool  sliceIrap<S : ISlice>( S s ) { return ( s.word( 1 ) & ( 1u << 22 ) ) != 0; }
int   colRefIdx<S : ISlice>( S s ) { return int( ( s.word( 1 ) >> 24 ) & 15 ); }
int   maxIbcCand<S : ISlice>( S s ) { return int( s.word( 2 ) & 0xff ); }
int   maxGeoCand<S : ISlice>( S s ) { return int( ( s.word( 2 ) >> 8 ) & 0xff ); }
int   maxAffineCand<S : ISlice>( S s ) { return int( ( s.word( 2 ) >> 16 ) & 0xff ); }
uint  independentIdx<S : ISlice>( S s ) { return s.word( 3 ); }
bool  usedAsLongTerm<S : ISlice>( S s, int list, int refIdx ) { return ( ( s.word( 4 ) >> ( 16 * list + refIdx ) ) & 1 ) != 0; }
bool  rplLongTerm<S : ISlice>( S s, int list, int refIdx ) { return ( ( s.word( 5 ) >> ( 16 * list + refIdx ) ) & 1 ) != 0; }
int   refPoc<S : ISlice>( S s, int list, int refIdx ) { return int( s.word( 16 + 16 * list + refIdx ) ); }
// The colocated picture's motion (ColocatedMotionInfo, 5 words, num8x8 entries per CTU) and slice table
int*  colMotion<S : ISlice>( S s ) { return (int*) ( uint64_t( s.word( 6 ) ) | ( uint64_t( s.word( 7 ) ) << 32 ) ); }
uint* colSlices<S : ISlice>( S s ) { return (uint*) ( uint64_t( s.word( 8 ) ) | ( uint64_t( s.word( 9 ) ) << 32 ) ); }
// The POC of the colocated picture itself (slice.getRefPic( colList, colRefIdx )), from the reference lists
int   colPicPoc<S : ISlice>( S s ) { return refPoc( s, isInterB( s ) ? ( colFromL0( s ) ? 0 : 1 ) : 0, colRefIdx( s ) ); }

// --------------------------------------------------------------------------------------------------------------------
// Motion
// --------------------------------------------------------------------------------------------------------------------

// vvdec's MotionInfo, plus what its users keep alongside it (MvField lists, HPMVInfo)
struct Motion
{
  int2 mv[2];
  int  refIdx[2];   // < 0: the list is unused
  int  bcw;
  bool hpel;        // AMVR's half-sample filter (useAltHpelIf)
  int  mrgType;     // vvdec's MergeType
};

Motion noMotion()
{
  Motion m;
  m.mv[0]     = int2( 0, 0 );
  m.mv[1]     = int2( 0, 0 );
  m.refIdx[0] = -1;
  m.refIdx[1] = -1;
  m.bcw       = 0;
  m.hpel      = false;
  m.mrgType   = 0;
  return m;
}

int interDir( Motion m )
{
  return ( m.refIdx[0] >= 0 ? 1 : 0 ) + ( m.refIdx[1] >= 0 ? 2 : 0 );
}

// MotionInfo::operator==
bool sameMotion( Motion a, Motion b )
{
  if( a.refIdx[0] != b.refIdx[0] ) return false;
  if( a.refIdx[0] >= 0 && any( a.mv[0] != b.mv[0] ) ) return false;
  if( a.refIdx[1] != b.refIdx[1] ) return false;
  if( a.refIdx[1] >= 0 && any( a.mv[1] != b.mv[1] ) ) return false;
  return true;
}

// Mv::mvCliptoStorageBitDepth: wraps around to 18 bits
int wrapMv( int v )
{
  const int period = 1 << 18;
  int       r      = ( v + period ) & ( period - 1 );
  return r >= ( period >> 1 ) ? r - period : r;
}

int2 wrapMv2( int2 v ) { return int2( wrapMv( v.x ), wrapMv( v.y ) ); }

// Mv::clipToStorageBitDepth
int2 clipMv18( int2 v ) { return clamp( v, int2( -( 1 << 17 ) ), int2( ( 1 << 17 ) - 1 ) ); }

// Mv::changePrecision from src to dst (vvdec's MvPrecision: 4-pel 0, int 2, half 3, quarter 4, internal 6)
int2 changePrecision( int2 v, int src, int dst )
{
  const int shift = dst - src;
  if( shift >= 0 )
  {
    return v * ( 1 << shift );
  }
  const int rs     = -shift;
  const int offset = 1 << ( rs - 1 );
  return int2( v.x >= 0 ? ( v.x + offset - 1 ) >> rs : ( v.x + offset ) >> rs, v.y >= 0 ? ( v.y + offset - 1 ) >> rs : ( v.y + offset ) >> rs );
}

// Mv::roundToPrecision( src, dst ): converts to dst and back
int2 roundToPrecision( int2 v, int src, int dst )
{
  return changePrecision( changePrecision( v, src, dst ), dst, src );
}

// Mv::m_amvrPrecision
int amvrPrecision( int amvr )
{
  return amvr == 0 ? 4 : amvr == 1 ? 2 : amvr == 2 ? 0 : 3;
}

// Mv::roundToAmvrSignalPrecision( MV_PRECISION_INTERNAL, amvr )
int2 roundToAmvr( int2 v, int amvr )
{
  return roundToPrecision( v, 6, amvrPrecision( amvr ) );
}

// Mv::scaleMv
int2 scaleMv( int2 v, int scale )
{
  const int x = scale * v.x;
  const int y = scale * v.y;
  return int2( clamp( ( x + 128 - ( x >= 0 ? 1 : 0 ) ) >> 8, -131072, 131071 ), clamp( ( y + 128 - ( y >= 0 ? 1 : 0 ) ) >> 8, -131072, 131071 ) );
}

// roundAffineMv
int2 roundAffineMv( int2 v, int shift )
{
  const int offset = 1 << ( shift - 1 );
  return int2( ( v.x + offset - ( v.x >= 0 ? 1 : 0 ) ) >> shift, ( v.y + offset - ( v.y >= 0 ? 1 : 0 ) ) >> shift );
}

// xGetDistScaleFactor
int distScale( int currPoc, int currRefPoc, int colPoc, int colRefPoc )
{
  const int diffD = colPoc - colRefPoc;
  const int diffB = currPoc - currRefPoc;
  if( diffD == diffB )
  {
    return 4096;
  }
  const int tdb = clamp( diffB, -128, 127 );
  const int tdd = clamp( diffD, -128, 127 );
  const int x   = ( 0x4000 + abs( tdd / 2 ) ) / tdd;
  return clamp( ( tdb * x + 32 ) >> 6, -4096, 4095 );
}

// roundMvComp: rounds through vvdec's 6-bit mantissa, 4-bit exponent form
int roundMvComp( int val )
{
  const int sign  = val >> 31;
  const int scale = firstbithigh( uint( ( val ^ sign ) | 31 ) ) - 5;
  int       exponent, mantissa;
  if( scale >= 0 )
  {
    const int round = ( 1 << scale ) >> 1;
    const int n     = ( val + round ) >> scale;
    exponent        = scale + ( ( n ^ sign ) >> 5 );
    mantissa        = ( n & 31 ) | ( sign * ( 1 << 5 ) );
  }
  else
  {
    exponent = 0;
    mantissa = val;
  }
  const int f = exponent | ( mantissa * 16 );
  // convertMvFloatToFixed
  const int e = f & 15;
  const int m = f >> 4;
  return e == 0 ? m : ( m ^ 32 ) * ( 1 << ( e - 1 ) );
}

int log2Of( int v ) { return int( firstbithigh( uint( v ) ) ); }

// --------------------------------------------------------------------------------------------------------------------
// The motion of the 4x4 blocks, and the CUs' motion records
// --------------------------------------------------------------------------------------------------------------------

int* motionPtr( int x, int y )
{
  const int ctuMask  = ( 1 << pc.log2Ctu ) - 1;
  const int ctuSize4 = 1 << ( pc.log2Ctu - 2 );
  const int pos      = ( ( x & ctuMask ) >> 2 ) + ( ( ( y & ctuMask ) >> 2 ) * ctuSize4 );
  return pc.motion + ( ctuOf( x, y ) * ctuSize4 * ctuSize4 + pos ) * 5;
}

// The MotionInfo at (x, y)
Motion motionAt( int x, int y )
{
  int*       mi = motionPtr( x, y );
  Motion     m  = noMotion();
  m.mv[0]       = int2( mi[0], mi[1] );
  m.mv[1]       = int2( mi[2], mi[3] );
  const uint w  = uint( mi[4] );
  m.refIdx[0]   = int( int8_t( w & 0xff ) );
  m.refIdx[1]   = int( int8_t( ( w >> 8 ) & 0xff ) );
  return m;
}

// Stores the 4x4 block's MotionInfo, with kind in its padding (for the comparison with the CPU's motion).
void storeMotionAt( int x, int y, Motion m, int kind )
{
  int* mi = motionPtr( x, y );
  mi[0]   = m.mv[0].x;
  mi[1]   = m.mv[0].y;
  mi[2]   = m.mv[1].x;
  mi[3]   = m.mv[1].y;
  mi[4]   = int( ( uint( m.refIdx[0] ) & 0xff ) | ( ( uint( m.refIdx[1] ) & 0xff ) << 8 ) | ( uint( kind ) << 16 ) );
}

// A CU's motion record, as the CodingUnit keeps it after the derivation:
// mv[l][k] at word 6 l + 2 k, then the reference indices, the inter
// direction, BCW, AMVR's half-sample flag (imv == IMV_HPEL), the merge type,
// the kind of motion, and whether the affine model has 6 parameters.
struct CuMotion
{
  int2 mv[2][3];
  int  refIdx[2];
  int  interDir;
  int  bcw;
  bool hpel;
  int  mrgType;
  int  kind;
  bool affine6;
};

CuMotion noCuMotion()
{
  CuMotion c;
  for( int l = 0; l < 2; l++ )
  {
    for( int k = 0; k < 3; k++ )
    {
      c.mv[l][k] = int2( 0, 0 );
    }
    c.refIdx[l] = -1;
  }
  c.interDir = 0;
  c.bcw      = 0;
  c.hpel     = false;
  c.mrgType  = 0;
  c.kind     = kNotInter;
  c.affine6  = false;
  return c;
}

void storeCuMotion( int cuIdx, CuMotion c )
{
  int* r = pc.cuMotion + cuIdx * 16;
  for( int l = 0; l < 2; l++ )
  {
    for( int k = 0; k < 3; k++ )
    {
      r[6 * l + 2 * k]     = c.mv[l][k].x;
      r[6 * l + 2 * k + 1] = c.mv[l][k].y;
    }
  }
  r[12] = int( ( uint( c.refIdx[0] ) & 0xff ) | ( ( uint( c.refIdx[1] ) & 0xff ) << 8 ) | ( uint( c.interDir ) << 16 ) | ( uint( c.bcw ) << 18 )
               | ( c.hpel ? 1u << 21 : 0u ) | ( uint( c.mrgType ) << 22 ) | ( uint( c.kind ) << 24 ) | ( c.affine6 ? 1u << 29 : 0u ) );
}

CuMotion loadCuMotion( int cuIdx )
{
  int*     r = pc.cuMotion + cuIdx * 16;
  CuMotion c;
  for( int l = 0; l < 2; l++ )
  {
    for( int k = 0; k < 3; k++ )
    {
      c.mv[l][k] = int2( r[6 * l + 2 * k], r[6 * l + 2 * k + 1] );
    }
  }
  const uint w = uint( r[12] );
  c.refIdx[0]  = int( int8_t( w & 0xff ) );
  c.refIdx[1]  = int( int8_t( ( w >> 8 ) & 0xff ) );
  c.interDir   = int( ( w >> 16 ) & 3 );
  c.bcw        = int( ( w >> 18 ) & 7 );
  c.hpel       = ( w & ( 1u << 21 ) ) != 0;
  c.mrgType    = int( ( w >> 22 ) & 3 );
  c.kind       = int( ( w >> 24 ) & 15 );
  c.affine6    = ( w & ( 1u << 29 ) ) != 0;
  return c;
}

// --------------------------------------------------------------------------------------------------------------------
// Neighbours
// --------------------------------------------------------------------------------------------------------------------

struct Cur
{
  Cu       cu;
  int      ctu;
  CurSlice slice;
  uint  sliceIdx;   // the independent slice's index
  uint  tile;
  int4  rect;       // the subpicture if treated as a picture, else the picture (left, top, right, bottom)
};

int4 ctuRect( int ctu )
{
  if( pc.subpicOn != 0 )
  {
    uint16_t* r = pc.subpics + ctu * 4;
    return int4( int( r[0] ), int( r[1] ), int( r[2] ), int( r[3] ) );
  }
  return int4( 0, 0, pc.width - 1, pc.height - 1 );
}

// The CTUs left (0), above-left (1), above (2) and above-right (3) of the CU's CTU: g_nbAvail tells whether
// getCURestricted lets the CU use each (same slice and tile; above-right not with WPP), and g_nbSlice holds their
// independent slices. setNeighbourCtus sets both once per CTU.
static uint g_nbAvail;
static uint g_nbSlice[4];

// Which of those four CTUs is the one xdiff, ydiff CTUs from the CU's, or -1 for another
int nbCtu( int xdiff, int ydiff )
{
  return ydiff == 0 ? ( xdiff == -1 ? 0 : -1 ) : ydiff == -1 && abs( xdiff ) <= 1 ? ( xdiff + 2 ) : -1;
}

// CodingStructure::getCURestricted on luma for (x, y) next to the CU (in its CTU or one of the four around it): the
// CU there if cur may use it, else -1
int cuAt( Cur cur, int x, int y )
{
  if( x < 0 || y < 0 || x >= pc.width || y >= pc.height )
  {
    return -1;
  }
  const int ydiff = ( y >> pc.log2Ctu ) - ( cur.cu.y >> pc.log2Ctu );
  const int xdiff = ( x >> pc.log2Ctu ) - ( cur.cu.x >> pc.log2Ctu );
  const int idx   = int( pc.cuMap[( y >> 2 ) * pc.widthIn4 + ( x >> 2 )] );
  if( ydiff == 0 && xdiff == 0 )
  {
    return idx > cur.cu.index ? -1 : idx;
  }
  const int k = nbCtu( xdiff, ydiff );
  return k >= 0 && ( g_nbAvail & ( 1u << k ) ) != 0 ? idx : -1;
}

void setNeighbourCtus( Cur cur, int col, int row )
{
  // Thread k reads CTU k's record and slice, so the four CTUs cost a single wait for memory.
  const int  k     = g_lane & 3;
  const int  dx    = k == 0 ? -1 : k - 2;
  const int  dy    = k == 0 ? 0 : -1;
  const int  cx    = col + dx;
  const int  cy    = row + dy;
  const bool there = cx >= 0 && cx < pc.widthInCtus && cy >= 0 && !( dx > 1 - pc.wpp && dy < 0 );
  uint       slice = 0;
  bool       avail = false;
  if( there )
  {
    const int ctu = cy * pc.widthInCtus + cx;
    slice         = independentIdx( sliceOfCtu( ctu ) );
    avail         = slice == cur.sliceIdx && ctuTile( ctu ) == cur.tile;
  }
  const uint4 mask = WaveActiveBallot( avail );
  g_nbAvail        = mask.x & 15u;
  for( int j = 0; j < 4; j++ )
  {
    g_nbSlice[j] = WaveReadLaneAt( slice, j );
  }
}

// The independent slice of the CTU at (x, y), the CU's CTU or one around it (the same as sliceIdxOfCu of the CU there)
uint sliceIdxAt( Cur cur, int x, int y )
{
  const int k = nbCtu( ( x >> pc.log2Ctu ) - ( cur.cu.x >> pc.log2Ctu ), ( y >> pc.log2Ctu ) - ( cur.cu.y >> pc.log2Ctu ) );
  return k < 0 ? cur.sliceIdx : g_nbSlice[k];
}

// The independent slice of a CU (its CTU's)
uint sliceIdxOfCu( int cuIdx )
{
  Cu cu = loadCu( cuIdx );
  return independentIdx( sliceOfCtu( ctuOf( cu.x, cu.y ) ) );
}

// PU::isDiffMER
bool isDiffMer( int x1, int y1, int x2, int y2 )
{
  return ( x1 >> pc.plevel ) != ( x2 >> pc.plevel ) || ( y1 >> pc.plevel ) != ( y2 >> pc.plevel );
}

// --------------------------------------------------------------------------------------------------------------------
// The history tables (MotionHist), one pair per row: the regular one and IBC's
// --------------------------------------------------------------------------------------------------------------------

static const int kMaxHmvp = 5;   // MAX_NUM_HMVP_CANDS

groupshared int2 s_lutMv[2][2][kMaxHmvp];   // [ibc][list][entry]
groupshared int  s_lutRef[2][2][kMaxHmvp];
groupshared int  s_lutBcw[2][kMaxHmvp];
groupshared int  s_lutHpel[2][kMaxHmvp];
groupshared int  s_lutSize[2];

Motion lutEntry( int lut, int i )
{
  Motion m    = noMotion();
  m.mv[0]     = s_lutMv[lut][0][i];
  m.mv[1]     = s_lutMv[lut][1][i];
  m.refIdx[0] = s_lutRef[lut][0][i];
  m.refIdx[1] = s_lutRef[lut][1][i];
  m.bcw       = s_lutBcw[lut][i];
  m.hpel      = s_lutHpel[lut][i] != 0;
  return m;
}

void setLutEntry( int lut, int i, Motion m )
{
  s_lutMv[lut][0][i]  = m.mv[0];
  s_lutMv[lut][1][i]  = m.mv[1];
  s_lutRef[lut][0][i] = m.refIdx[0];
  s_lutRef[lut][1][i] = m.refIdx[1];
  s_lutBcw[lut][i]    = m.bcw;
  s_lutHpel[lut][i]   = m.hpel ? 1 : 0;
}

// MotionHist::addMiToLut (HPMVInfo's == compares the motion only)
void addToLut( int lut, Motion mi )
{
  // Thread i handles entry i: a ballot finds the first entry equal to mi, and the entries after it all move down
  // by one at once.
  const int  count  = s_lutSize[lut];
  const int  i      = g_lane;
  const bool alike  = i < count && sameMotion( lutEntry( lut, min( i, kMaxHmvp - 1 ) ), mi );
  const uint first  = WaveActiveBallot( alike ).x;
  const bool pruned = first != 0;
  const int  same   = pruned ? int( firstbitlow( first ) ) : 0;
  int        size   = count;
  if( pruned || count == kMaxHmvp )
  {
    const bool moves = i >= same && i + 1 < count;
    Motion     next  = noMotion();
    if( moves )
    {
      next = lutEntry( lut, i + 1 );
    }
    GroupMemoryBarrierWithGroupSync();   // (all read before any writes)
    if( moves )
    {
      setLutEntry( lut, i, next );
    }
    size--;
  }
  GroupMemoryBarrierWithGroupSync();
  setLutEntry( lut, size, mi );
  s_lutSize[lut] = size + 1;
  GroupMemoryBarrierWithGroupSync();
}

// --------------------------------------------------------------------------------------------------------------------
// The temporal candidates
// --------------------------------------------------------------------------------------------------------------------

// The colocated picture's motion at (x, y): its MotionInfo, and the slice of its CTU
Motion colMotionAt( Cur cur, int x, int y, out ColSlice colSlice )
{
  const int ctuCol = ctuOf( x, y );
  const int mask8  = ( 1 << ( pc.log2Ctu - 3 ) ) - 1;
  const int pos8   = ( ( x >> 3 ) & mask8 ) + ( ( ( y >> 3 ) & mask8 ) << ( pc.log2Ctu - 3 ) );
  int*      mi     = colMotion( cur.slice ) + ( ctuCol * ( 1 << ( 2 * ( pc.log2Ctu - 3 ) ) ) + pos8 ) * 5;
  Motion    m      = noMotion();
  m.mv[0]          = int2( mi[0], mi[1] );
  m.mv[1]          = int2( mi[2], mi[3] );
  const uint refs  = uint( mi[4] );
  m.refIdx[0]      = int( int8_t( refs & 0xff ) );
  m.refIdx[1]      = int( int8_t( ( refs >> 8 ) & 0xff ) );
  // (the colocated slices of the CTU and the one right of it are at hand: loadColSlices)
  uint*      colTable = colSlices( cur.slice );
  const uint colIdx   = ctuCol == cur.ctu ? g_colCtuSlice[0] : ctuCol == cur.ctu + 1 ? g_colCtuSlice[1] : colTable[ctuCol];
  colSlice.r          = colTable + pc.widthInCtus * pc.heightInCtus + colIdx * kSliceWords;
  colSlice.cached     = colIdx == g_colSliceIdx;
  return m;
}

// PU::getColocatedMVP
bool colocatedMvp( Cur cur, int list, int x, int y, int refIdx, bool sbFlag, out int2 mv )
{
  mv = int2( 0, 0 );
  if( cur.cu.predMode == MODE_IBC || !hasColPic( cur.slice ) )
  {
    return false;
  }
  if( x < cur.rect.x || y < cur.rect.y || x > cur.rect.z || y > cur.rect.w )
  {
    return false;   // outside the subpicture treated as a picture
  }
  int    colList = checkLDC( cur.slice ) ? list : ( colFromL0( cur.slice ) ? 1 : 0 );
  ColSlice colSlice;
  Motion mi = colMotionAt( cur, x, y, colSlice );
  if( mi.refIdx[0] < 0 && mi.refIdx[1] < 0 )
  {
    return false;
  }
  int colRef = mi.refIdx[colList];
  if( sbFlag && !checkLDC( cur.slice ) )
  {
    colList = list;
    colRef  = mi.refIdx[colList];
    if( colRef < 0 )
    {
      return false;
    }
  }
  else if( colRef < 0 )
  {
    colList = 1 - colList;
    colRef  = mi.refIdx[colList];
    if( colRef < 0 )
    {
      return false;
    }
  }
  const bool currLt = usedAsLongTerm( cur.slice, list, refIdx );
  const bool colLt  = usedAsLongTerm( colSlice, colList, colRef );
  if( currLt != colLt )
  {
    return false;
  }
  int2 colMv = mi.mv[colList];
  colMv      = int2( roundMvComp( colMv.x ), roundMvComp( colMv.y ) );
  if( currLt )
  {
    mv = clipMv18( colMv );
    return true;
  }
  const int scale = distScale( slicePoc( cur.slice ), refPoc( cur.slice, list, refIdx ), slicePoc( colSlice ), refPoc( colSlice, colList, colRef ) );
  mv              = scale == 4096 ? clipMv18( colMv ) : scaleMv( colMv, scale );
  return true;
}

// The temporal candidate's positions (getInterMergeCandidates, fillMvpCand, the affine constructed candidates'
// RB): C0 and whether it may be used, then C1
void colPositions( Cur cur, out bool c0Avail, out int2 c0, out int2 c1 )
{
  const int2 rb      = int2( cur.cu.x + cur.cu.w - 1 - 3, cur.cu.y + cur.cu.h - 1 - 3 );
  c1                 = int2( cur.cu.x + ( cur.cu.w >> 1 ), cur.cu.y + ( cur.cu.h >> 1 ) );
  c0                 = rb + int2( 4, 4 );
  const int ctuMask  = ( 1 << pc.log2Ctu ) - 1;
  // (minCUWidth 4): C0 must lie inside the picture, or inside the subpicture treated as a picture
  const bool boundary = rb.x + 4 <= cur.rect.z && rb.y + 4 <= cur.rect.w;
  c0Avail             = boundary && ( rb.y & ctuMask ) + 4 < ( 1 << pc.log2Ctu );
}

// --------------------------------------------------------------------------------------------------------------------
// Merge
// --------------------------------------------------------------------------------------------------------------------

static const int kMaxMerge = 6;

groupshared int2 s_candMv[2][kMaxMerge];
groupshared int  s_candRef[2][kMaxMerge];
groupshared int  s_candBcw[kMaxMerge];
groupshared int  s_candHpel[kMaxMerge];
groupshared int  s_candType[kMaxMerge];

Motion cand( int i )
{
  Motion m    = noMotion();
  m.mv[0]     = s_candMv[0][i];
  m.mv[1]     = s_candMv[1][i];
  m.refIdx[0] = s_candRef[0][i];
  m.refIdx[1] = s_candRef[1][i];
  m.bcw       = s_candBcw[i];
  m.hpel      = s_candHpel[i] != 0;
  m.mrgType   = s_candType[i];
  return m;
}

void setCand( int i, Motion m )
{
  s_candMv[0][i]  = m.mv[0];
  s_candMv[1][i]  = m.mv[1];
  s_candRef[0][i] = m.refIdx[0];
  s_candRef[1][i] = m.refIdx[1];
  s_candBcw[i]    = m.bcw;
  s_candHpel[i]   = m.hpel ? 1 : 0;
  s_candType[i]   = m.mrgType;
}

// --------------------------------------------------------------------------------------------------------------------
// The blocks next to the CU
// --------------------------------------------------------------------------------------------------------------------

// The 4x4 blocks next to the CU that the merge, AMVP and affine lists look at, by vvdec's names
static const int kA0            = 0;   // below the bottom-left one's left neighbour
static const int kA1            = 1;   // left of the bottom-left sample
static const int kB0            = 2;   // above-right
static const int kB1            = 3;   // above the top-right sample
static const int kB2            = 4;   // above-left
static const int kB3            = 5;   // above the top-left sample
static const int kA2            = 6;   // left of the top-left sample
static const int kNumNeighbours = 7;

int2 neighbourPos( Cu cu, int k )
{
  const int right  = cu.x + cu.w - 1;
  const int bottom = cu.y + cu.h - 1;
  return k == kA0   ? int2( cu.x - 1, bottom + 1 )
         : k == kA1 ? int2( cu.x - 1, bottom )
         : k == kB0 ? int2( right + 1, cu.y - 1 )
         : k == kB1 ? int2( right, cu.y - 1 )
         : k == kB2 ? int2( cu.x - 1, cu.y - 1 )
         : k == kB3 ? int2( cu.x, cu.y - 1 )
                    : int2( cu.x - 1, cu.y );
}

// What the lists read of a block next to the CU: the CU map's word there and the block's motion (motionAt's words)
struct NeighbourWords
{
  int  idx;
  int4 mv;
  uint w;
};

// A place outside the picture reads the nearest block's words, which neighbourCu then does not let anyone use.
NeighbourWords fetchNeighbour( int x, int y )
{
  const int      cx = clamp( x, 0, pc.width - 1 ), cy = clamp( y, 0, pc.height - 1 );
  int*           mi = motionPtr( cx, cy );
  NeighbourWords n;
  n.idx = int( pc.cuMap[( cy >> 2 ) * pc.widthIn4 + ( cx >> 2 )] );
  n.mv  = int4( mi[0], mi[1], mi[2], mi[3] );
  n.w   = uint( mi[4] );
  return n;
}

// Thread k (below kNumNeighbours) holds the words of the CU's neighbouring block k: fetchNeighbours reads all seven
// at once when the CU starts, so that the CU waits for memory once for all its lists' neighbours rather than once
// for each in turn; neighbourWords( k ) hands block k's to all the threads.
static NeighbourWords g_nbWords;

void fetchNeighbours( Cu cu )
{
  const int2 p = neighbourPos( cu, min( g_lane, kNumNeighbours - 1 ) );
  g_nbWords    = fetchNeighbour( p.x, p.y );
}

// Thread lane's v, which all the threads name alike: the compiler then keeps it in a scalar register
int laneInt( int v, int lane )
{
  return WaveReadLaneFirst( WaveReadLaneAt( v, lane ) );
}

NeighbourWords neighbourWords( int k )
{
  NeighbourWords n;
  n.idx = laneInt( g_nbWords.idx, k );
  n.mv  = int4( laneInt( g_nbWords.mv.x, k ), laneInt( g_nbWords.mv.y, k ), laneInt( g_nbWords.mv.z, k ), laneInt( g_nbWords.mv.w, k ) );
  n.w   = uint( laneInt( int( g_nbWords.w ), k ) );
  return n;
}

// cuAt for a block next to the CU at p, from its words: the CU there if cur may use it, else -1
int neighbourCu( Cur cur, int2 p, NeighbourWords n )
{
  if( p.x < 0 || p.y < 0 || p.x >= pc.width || p.y >= pc.height )
  {
    return -1;
  }
  const int xdiff = ( p.x >> pc.log2Ctu ) - ( cur.cu.x >> pc.log2Ctu );
  const int ydiff = ( p.y >> pc.log2Ctu ) - ( cur.cu.y >> pc.log2Ctu );
  if( xdiff == 0 && ydiff == 0 )
  {
    return n.idx > cur.cu.index ? -1 : n.idx;
  }
  const int k = nbCtu( xdiff, ydiff );
  return k >= 0 && ( g_nbAvail & ( 1u << k ) ) != 0 ? n.idx : -1;
}

// The kind of motion spanMotion and the others wrote into the block's word 4
int blockKind( NeighbourWords n )
{
  return int( ( n.w >> 16 ) & 0xff );
}

// Whether the block's CU is an inter CU (CU::isInter): intra CUs' blocks are kNotInter, IBC CUs' kIbc
bool blockInter( NeighbourWords n )
{
  const int kind = blockKind( n );
  return kind != kNotInter && kind != kIbc;
}

// The block's MotionInfo
Motion blockMotion( NeighbourWords n )
{
  Motion m    = noMotion();
  m.mv[0]     = n.mv.xy;
  m.mv[1]     = n.mv.zw;
  m.refIdx[0] = int( int8_t( n.w & 0xff ) );
  m.refIdx[1] = int( int8_t( ( n.w >> 8 ) & 0xff ) );
  return m;
}

// The motion of neighbouring block k if cur may use it and its CU is an inter CU; cuIdx: its CU, or -1 if cur may
// not use it
bool neighbourBlock( Cur cur, int k, out int cuIdx, out Motion m )
{
  const NeighbourWords n = neighbourWords( k );
  cuIdx                  = neighbourCu( cur, neighbourPos( cur.cu, k ), n );
  m                      = blockMotion( n );
  return cuIdx >= 0 && blockInter( n );
}

// BCW of the CU of neighbouring block k (an inter CU's): a translational CU's from the block (where spanMotion put
// it), other kinds' from the CU's motion record
int neighbourBcw( int k, int cuIdx )
{
  const NeighbourWords n = neighbourWords( k );
  return blockKind( n ) == kTranslation ? int( ( n.w >> 24 ) & 7 ) : loadCuMotion( cuIdx ).bcw;
}

// Neighbouring block k's motion if cur may use it and its CU is an inter CU, with what the CU keeps: BCW and AMVR's
// half-sample flag (a translational CU's from the block, other kinds' from the CU's motion record)
bool neighbourAt( Cur cur, int k, out int cuIdx, out Motion m )
{
  if( !neighbourBlock( cur, k, cuIdx, m ) )
  {
    m = noMotion();
    return false;
  }
  const NeighbourWords n = neighbourWords( k );
  if( blockKind( n ) == kTranslation )
  {
    m.bcw  = int( ( n.w >> 24 ) & 7 );
    m.hpel = ( ( n.w >> 27 ) & 1 ) != 0;
  }
  else
  {
    const CuMotion c = loadCuMotion( cuIdx );
    m.bcw            = c.bcw;
    m.hpel           = c.hpel;
  }
  return true;
}

// A spatial candidate from a neighbour's motion; the second list is used only in B slices.
Motion spatialCand( Motion n, bool interB )
{
  Motion c    = noMotion();
  c.mv[0]     = n.mv[0];
  c.refIdx[0] = n.refIdx[0];
  if( interB )
  {
    c.mv[1]     = n.mv[1];
    c.refIdx[1] = n.refIdx[1];
  }
  c.hpel = n.hpel;
  c.bcw  = interDir( n ) == 3 ? n.bcw : 0;   // (BCW_DEFAULT: 0)
  return c;
}

// xCheckSimilarMotion: compares candidate idx with the candidates before prevCnt that are not pruned yet.
bool similarMotion( int idx, int prevCnt, inout uint pruned )
{
  const Motion c = cand( idx );
  for( int i = 0; i < prevCnt; i++ )
  {
    if( ( pruned & ( 1u << i ) ) != 0 )
    {
      continue;
    }
    const Motion o = cand( i );
    if( interDir( o ) != interDir( c ) )
    {
      continue;
    }
    if( interDir( o ) == 3 )
    {
      if( o.refIdx[0] == c.refIdx[0] && o.refIdx[1] == c.refIdx[1] && all( o.mv[0] == c.mv[0] ) && all( o.mv[1] == c.mv[1] ) )
      {
        pruned |= 1u << i;
        return true;
      }
    }
    else
    {
      const int l = interDir( o ) - 1;
      if( o.refIdx[l] == c.refIdx[l] && all( o.mv[l] == c.mv[l] ) )
      {
        pruned |= 1u << i;
        return true;
      }
    }
  }
  return false;
}

// PU::getInterMergeCandidates up to candidate last (its fast exit; last < 0: all); returns the number of candidates.
int mergeCandidates( Cur cur, int last )
{
  const int  maxCand = pc.maxNumMergeCand;
  const bool interB  = isInterB( cur.slice );
  for( int i = 0; i < kMaxMerge; i++ )
  {
    setCand( i, noMotion() );
  }
  int cnt = 0;

  const int2 lt = int2( cur.cu.x, cur.cu.y );
  const int2 rt = int2( cur.cu.x + cur.cu.w - 1, cur.cu.y );
  const int2 lb = int2( cur.cu.x, cur.cu.y + cur.cu.h - 1 );

  // B1: above
  Motion     above, left;
  int        cuAbove;
  const bool availB1 = neighbourAt( cur, kB1, cuAbove, above ) && isDiffMer( lt.x, lt.y, rt.x, rt.y - 1 );
  if( availB1 )
  {
    setCand( cnt, spatialCand( above, interB ) );
    if( cnt == last ) return cnt + 1;
    cnt++;
  }
  if( cnt == maxCand ) return cnt;

  // A1: left
  int        cuLeft;
  const bool availA1 = neighbourAt( cur, kA1, cuLeft, left ) && isDiffMer( lt.x, lt.y, lb.x - 1, lb.y );
  if( availA1 )
  {
    if( !availB1 || sliceIdxAt( cur, rt.x, rt.y - 1 ) != sliceIdxAt( cur, lb.x - 1, lb.y ) || !sameMotion( above, left ) )
    {
      setCand( cnt, spatialCand( left, interB ) );
      if( cnt == last ) return cnt + 1;
      cnt++;
    }
  }
  if( cnt == maxCand ) return cnt;

  const int spatialCandPos = cnt;

  // B0: above right
  Motion     aboveRight;
  int        cuAboveRight;
  const bool availB0      = neighbourAt( cur, kB0, cuAboveRight, aboveRight ) && isDiffMer( lt.x, lt.y, rt.x + 1, rt.y - 1 );
  if( availB0 )
  {
    if( !availB1 || sliceIdxAt( cur, rt.x, rt.y - 1 ) != sliceIdxAt( cur, rt.x + 1, rt.y - 1 ) || !sameMotion( above, aboveRight ) )
    {
      setCand( cnt, spatialCand( aboveRight, interB ) );
      if( cnt == last ) return cnt + 1;
      cnt++;
    }
    if( cnt == maxCand ) return cnt;
  }

  // A0: below left
  Motion     belowLeft;
  int        cuBelowLeft;
  const bool availA0     = neighbourAt( cur, kA0, cuBelowLeft, belowLeft ) && isDiffMer( lt.x, lt.y, lb.x - 1, lb.y + 1 );
  if( availA0 )
  {
    if( !availA1 || sliceIdxAt( cur, lb.x - 1, lb.y + 1 ) != sliceIdxAt( cur, lb.x - 1, lb.y ) || !sameMotion( belowLeft, left ) )
    {
      setCand( cnt, spatialCand( belowLeft, interB ) );
      if( cnt == last ) return cnt + 1;
      cnt++;
    }
    if( cnt == maxCand ) return cnt;
  }

  // B2: above left
  if( cnt < 4 )
  {
    Motion     aboveLeft;
    int        cuAboveLeft;
    const bool availB2     = neighbourAt( cur, kB2, cuAboveLeft, aboveLeft ) && isDiffMer( lt.x, lt.y, lt.x - 1, lt.y - 1 );
    if( availB2 )
    {
      const uint sliceB2 = sliceIdxAt( cur, lt.x - 1, lt.y - 1 );
      if( ( !availA1 || sliceIdxAt( cur, lb.x - 1, lb.y ) != sliceB2 || !sameMotion( left, aboveLeft ) )
          && ( !availB1 || sliceIdxAt( cur, rt.x, rt.y - 1 ) != sliceB2 || !sameMotion( above, aboveLeft ) ) )
      {
        setCand( cnt, spatialCand( aboveLeft, interB ) );
        if( cnt == last ) return cnt + 1;
        cnt++;
      }
    }
    if( cnt == maxCand ) return cnt;
  }

  // The temporal candidate
  if( tmvp( cur.slice ) && cur.cu.w + cur.cu.h > 12 )
  {
    bool c0Avail;
    int2 c0, c1;
    colPositions( cur, c0Avail, c0, c1 );
    Motion c = noMotion();
    int2   mv;
    if( ( c0Avail && colocatedMvp( cur, 0, c0.x, c0.y, 0, false, mv ) ) || colocatedMvp( cur, 0, c1.x, c1.y, 0, false, mv ) )
    {
      c.mv[0]     = mv;
      c.refIdx[0] = 0;
    }
    if( interB && ( ( c0Avail && colocatedMvp( cur, 1, c0.x, c0.y, 0, false, mv ) ) || colocatedMvp( cur, 1, c1.x, c1.y, 0, false, mv ) ) )
    {
      c.mv[1]     = mv;
      c.refIdx[1] = 0;
    }
    if( interDir( c ) != 0 )
    {
      setCand( cnt, c );
      if( cnt == last ) return cnt + 1;
      cnt++;
    }
    if( cnt == maxCand ) return cnt;
  }

  // The history candidates (addMergeHMVPCand): newest first; the first two are pruned against B1 and A1.
  const int maxCandMin1 = maxCand - 1;
  if( cnt != maxCandMin1 )
  {
    uint      pruned  = 0;
    const int lutSize = s_lutSize[0];
    for( int i = 1; i <= lutSize; i++ )
    {
      Motion h    = lutEntry( 0, lutSize - i );
      Motion c    = noMotion();
      c.mv[0]     = h.mv[0];
      c.refIdx[0] = h.refIdx[0];
      if( interB )
      {
        c.mv[1]     = h.mv[1];
        c.refIdx[1] = h.refIdx[1];
      }
      c.hpel = h.hpel;
      setCand( cnt, c );
      if( i > 2 || !similarMotion( cnt, spatialCandPos, pruned ) )
      {
        c.bcw = interDir( c ) == 3 ? h.bcw : 0;
        setCand( cnt, c );
        if( cnt == last ) return cnt + 1;
        cnt++;
        if( cnt == maxCandMin1 )
        {
          break;
        }
      }
    }
    if( cnt < maxCandMin1 )
    {
      s_candHpel[cnt] = 0;
    }
  }

  // The pairwise average of the first two candidates
  if( cnt > 1 && cnt < maxCand )
  {
    Motion c0 = cand( 0 );
    Motion c1 = cand( 1 );
    Motion p  = noMotion();
    p.hpel    = c0.hpel == c1.hpel ? c0.hpel : false;
    for( int l = 0; l < ( interB ? 2 : 1 ); l++ )
    {
      const int ri = c0.refIdx[l];
      const int rj = c1.refIdx[l];
      if( ri < 0 && rj < 0 )
      {
        continue;
      }
      if( ri >= 0 && rj >= 0 )
      {
        p.mv[l]     = roundAffineMv( c0.mv[l] + c1.mv[l], 1 );
        p.refIdx[l] = ri;
      }
      else if( ri >= 0 )
      {
        p.mv[l]     = c0.mv[l];
        p.refIdx[l] = ri;
      }
      else
      {
        p.mv[l]     = c1.mv[l];
        p.refIdx[l] = rj;
      }
    }
    setCand( cnt, p );   // (when p is no candidate, its hpel stays in the slot, as in vvdec)
    if( interDir( p ) > 0 )
    {
      if( cnt == last ) return cnt + 1;   // (vvdec has no fast exit here; the list is the same either way)
      cnt++;
    }
    if( cnt == maxCand ) return cnt;
  }

  // Zero candidates
  const int numRef = interB ? min( numRefIdx( cur.slice, 0 ), numRefIdx( cur.slice, 1 ) ) : numRefIdx( cur.slice, 0 );
  int       r      = 0;
  int       refcnt = 0;
  while( cnt < maxCand )
  {
    Motion z    = noMotion();
    z.refIdx[0] = r;
    if( interB )
    {
      z.refIdx[1] = r;
    }
    if( !interB && refPoc( cur.slice, 0, r ) == slicePoc( cur.slice ) )
    {
      z.mrgType = MRG_TYPE_IBC;
    }
    setCand( cnt, z );
    cnt++;
    if( refcnt == numRef - 1 )
    {
      r = 0;
    }
    else
    {
      r++;
      refcnt++;
    }
  }
  return cnt;
}

// MergeCtx::setMmvdMergeCandiInfo, after getInterMMVDMergeCandidates: returns the motion of MMVD index idx
Motion mmvdMotion( Cur cur, int idx, int numCand )
{
  // The base candidates: the first two candidates of type DEFAULT_N
  Motion base[2];
  base[0]   = noMotion();
  base[1]   = noMotion();
  int bases = 0;
  for( int k = 0; k < numCand && bases < 2; k++ )
  {
    Motion c = cand( k );
    if( c.mrgType == MRG_TYPE_DEFAULT_N )
    {
      Motion b = noMotion();
      if( c.refIdx[0] >= 0 )
      {
        b.mv[0]     = c.mv[0];
        b.refIdx[0] = c.refIdx[0];
      }
      if( c.refIdx[1] >= 0 )
      {
        b.mv[1]     = c.mv[1];
        b.refIdx[1] = c.refIdx[1];
      }
      b.hpel      = c.hpel;
      base[bases] = b;
      bases++;
    }
  }
  const int baseIdx  = idx / 32;   // MMVD_MAX_REFINE_NUM: 8 steps x 4 directions
  const int t        = idx - baseIdx * 32;
  const int step     = t / 4;
  const int position = t - step * 4;
  int       offset   = ( 1 << step ) << 2;
  if( mmvdFullPel( cur.slice ) )
  {
    offset <<= 2;
  }
  const Motion b    = base[baseIdx];
  const int    ref0 = b.refIdx[0];
  const int    ref1 = b.refIdx[1];
  const int2   d    = position == 0 ? int2( offset, 0 ) : position == 1 ? int2( -offset, 0 ) : position == 2 ? int2( 0, offset ) : int2( 0, -offset );
  Motion       m    = noMotion();
  m.hpel            = b.hpel;
  if( ref0 >= 0 && ref1 >= 0 )
  {
    const int poc0    = refPoc( cur.slice, 0, ref0 );
    const int poc1    = refPoc( cur.slice, 1, ref1 );
    const int currPoc = slicePoc( cur.slice );
    int2      t0      = d;
    int2      t1      = d;
    if( poc0 - currPoc == poc1 - currPoc )
    {
      t1 = t0;
    }
    else if( abs( poc1 - currPoc ) > abs( poc0 - currPoc ) )
    {
      const int  scale = distScale( currPoc, poc0, currPoc, poc1 );
      const bool lt    = rplLongTerm( cur.slice, 0, ref0 ) || rplLongTerm( cur.slice, 1, ref1 );
      t1               = d;
      t0               = lt ? ( ( poc1 - currPoc ) * ( poc0 - currPoc ) > 0 ? t1 : -t1 ) : scaleMv( t1, scale );
    }
    else
    {
      const int  scale = distScale( currPoc, poc1, currPoc, poc0 );
      const bool lt    = rplLongTerm( cur.slice, 0, ref0 ) || rplLongTerm( cur.slice, 1, ref1 );
      t0               = d;
      t1               = lt ? ( ( poc1 - currPoc ) * ( poc0 - currPoc ) > 0 ? t0 : -t0 ) : scaleMv( t0, scale );
    }
    m.mv[0]     = b.mv[0] + t0;
    m.refIdx[0] = ref0;
    m.mv[1]     = b.mv[1] + t1;
    m.refIdx[1] = ref1;
  }
  else if( ref0 >= 0 )
  {
    m.mv[0]     = b.mv[0] + d;
    m.refIdx[0] = ref0;
  }
  else if( ref1 >= 0 )
  {
    m.mv[1]     = b.mv[1] + d;
    m.refIdx[1] = ref1;
  }
  return m;
}

// --------------------------------------------------------------------------------------------------------------------
// AMVP
// --------------------------------------------------------------------------------------------------------------------

// PU::addMVPCandUnscaled: neighbouring block k's motion, in either list, that points to the current reference picture
bool amvpSpatial( Cur cur, int list, int currRefPoc, int k, out int2 mv )
{
  mv = int2( 0, 0 );
  int    cuIdx;
  Motion n;
  if( !neighbourBlock( cur, k, cuIdx, n ) )
  {
    return false;
  }
  for( int s = 0; s < 2; s++ )
  {
    const int l = s == 0 ? list : 1 - list;
    const int r = n.refIdx[l];
    if( r >= 0 && currRefPoc == refPoc( cur.slice, l, r ) )
    {
      mv = n.mv[l];
      return true;
    }
  }
  return false;
}

// PU::fillMvpCand: returns candidate mvpIdx
int2 amvpCandidate( Cur cur, int list, int refIdx, int amvr, int mvpIdx )
{
  int2 cands[2];
  cands[0] = int2( 0, 0 );
  cands[1] = int2( 0, 0 );
  int num  = 0;
  if( refIdx < 0 )
  {
    return int2( 0, 0 );
  }
  const int  currRefPoc = refPoc( cur.slice, list, refIdx );
  int2       mv;
  if( amvpSpatial( cur, list, currRefPoc, kA0, mv ) || amvpSpatial( cur, list, currRefPoc, kA1, mv ) )
  {
    cands[num++] = mv;
  }
  if( amvpSpatial( cur, list, currRefPoc, kB0, mv ) || amvpSpatial( cur, list, currRefPoc, kB1, mv ) || amvpSpatial( cur, list, currRefPoc, kB2, mv ) )
  {
    cands[num++] = mv;
  }
  for( int i = 0; i < num; i++ )
  {
    cands[i] = roundToAmvr( cands[i], amvr );
  }
  if( num == 2 && all( cands[0] == cands[1] ) )
  {
    num = 1;
  }
  if( tmvp( cur.slice ) && num < 2 && cur.cu.w + cur.cu.h > 12 )
  {
    bool c0Avail;
    int2 c0, c1;
    colPositions( cur, c0Avail, c0, c1 );
    if( ( c0Avail && colocatedMvp( cur, list, c0.x, c0.y, refIdx, false, mv ) ) || colocatedMvp( cur, list, c1.x, c1.y, refIdx, false, mv ) )
    {
      cands[num++] = roundToAmvr( mv, amvr );
    }
  }
  // The history candidates (addAMVPHMVPCand): up to 4 entries, starting with the LUT's first entry
  if( num < 2 )
  {
    const int count = min( 4, s_lutSize[0] );
    for( int i = 0; i < count && num < 2; i++ )
    {
      const Motion h = lutEntry( 0, i );
      for( int s = 0; s < 2 && num < 2; s++ )
      {
        const int l = s == 0 ? list : 1 - list;
        const int r = h.refIdx[l];
        if( r >= 0 && currRefPoc == refPoc( cur.slice, l, r ) )
        {
          cands[num++] = roundToAmvr( h.mv[l], amvr );
        }
      }
    }
  }
  // Zeros for the rest (including the second slot when the spatial pair collapsed into one)
  for( int i = num; i < 2; i++ )
  {
    cands[i] = int2( 0, 0 );
  }
  return roundToAmvr( cands[mvpIdx], amvr );
}

// --------------------------------------------------------------------------------------------------------------------
// IBC
// --------------------------------------------------------------------------------------------------------------------

// An IBC neighbour's block vector (its MotionInfo, with miRefIdx[0] made valid)
bool ibcNeighbour( int cuIdx, int x, int y, out Motion m )
{
  m = noMotion();
  if( cuIdx < 0 || loadCu( cuIdx ).predMode != MODE_IBC )
  {
    return false;
  }
  m           = motionAt( x, y );
  m.refIdx[0] = 0;   // MI_NOT_VALID + 1
  return true;
}

// PU::getIBCMergeCandidates up to candidate last; the block vectors go to s_candMv[0].
int ibcMergeCandidates( Cur cur, int last )
{
  const int  maxCand = maxIbcCand( cur.slice );
  const bool gt4x4   = cur.cu.w * cur.cu.h > 16;
  for( int i = 0; i < kMaxMerge; i++ )
  {
    setCand( i, noMotion() );
  }
  int        cnt = 0;
  const int2 rt  = int2( cur.cu.x + cur.cu.w - 1, cur.cu.y );
  const int2 lb  = int2( cur.cu.x, cur.cu.y + cur.cu.h - 1 );

  Motion     left, above;
  const int  cuLeft  = cuAt( cur, lb.x - 1, lb.y );
  const bool availA1 = ibcNeighbour( cuLeft, lb.x - 1, lb.y, left );
  if( gt4x4 && availA1 )
  {
    Motion c    = noMotion();
    c.mv[0]     = left.mv[0];
    c.refIdx[0] = 0;
    setCand( cnt, c );
    if( cnt == last ) return cnt + 1;
    cnt++;
  }
  if( cnt == maxCand ) return cnt;

  const int  cuAbove = cuAt( cur, rt.x, rt.y - 1 );
  const bool availB1 = ibcNeighbour( cuAbove, rt.x, rt.y - 1, above );
  if( gt4x4 && availB1 )
  {
    if( !availA1 || sliceIdxOfCu( cuAbove ) != sliceIdxOfCu( cuLeft ) || !sameMotion( above, left ) )
    {
      Motion c    = noMotion();
      c.mv[0]     = above.mv[0];
      c.refIdx[0] = 0;
      setCand( cnt, c );
      if( cnt == last ) return cnt + 1;
      cnt++;
    }
  }
  if( cnt == maxCand ) return cnt;

  // The history candidates (addMergeHMVPCand for IBC): IBC has no pairwise candidate, so they may fill the list
  const int spatialCandPos = cnt;
  if( cnt != maxCand )
  {
    uint      pruned  = 0;
    const int lutSize = s_lutSize[1];
    for( int i = 1; i <= lutSize; i++ )
    {
      Motion h    = lutEntry( 1, lutSize - i );
      Motion c    = noMotion();
      c.mv[0]     = h.mv[0];
      c.refIdx[0] = h.refIdx[0];
      if( isInterB( cur.slice ) )
      {
        c.mv[1]     = h.mv[1];
        c.refIdx[1] = h.refIdx[1];
      }
      setCand( cnt, c );
      if( i > 1 || !gt4x4 || !similarMotion( cnt, spatialCandPos, pruned ) )
      {
        c.bcw = interDir( c ) == 3 ? h.bcw : 0;
        setCand( cnt, c );
        if( cnt == last ) return cnt + 1;
        cnt++;
        if( cnt == maxCand )
        {
          break;
        }
      }
    }
  }
  while( cnt < maxCand )
  {
    setCand( cnt, noMotion() );   // (0, 0) with NOT_VALID, i.e. interDir 1
    cnt++;
    if( last == cnt )
    {
      return cnt;
    }
  }
  return cnt;
}

// --------------------------------------------------------------------------------------------------------------------
// Affine
// --------------------------------------------------------------------------------------------------------------------

static const int kMaxAffine = 5;   // AFFINE_MRG_MAX_NUM_CANDS

groupshared int2 s_affMv[2][3][kMaxAffine];   // [list][control point][candidate]
groupshared int  s_affRef[2][kMaxAffine];
groupshared int  s_affDir[kMaxAffine];
groupshared int  s_affType[kMaxAffine];       // 1: the 6-parameter model
groupshared int  s_affMrgType[kMaxAffine];
groupshared int  s_affBcw[kMaxAffine];

void clearAffineCand( int i )
{
  for( int l = 0; l < 2; l++ )
  {
    for( int k = 0; k < 3; k++ )
    {
      s_affMv[l][k][i] = int2( 0, 0 );
    }
    s_affRef[l][i] = -1;
  }
  s_affDir[i]     = 0;
  s_affType[i]    = 0;
  s_affMrgType[i] = MRG_TYPE_DEFAULT_N;
  s_affBcw[i]     = 0;
}

// PU::xInheritedAffineMv: evaluates a neighbouring affine CU's model at the current CU's corners
void inheritedAffineMv( Cur cur, bool is6param, int nIdx, int list, out int2 mv[3] )
{
  mv[0]              = int2( 0, 0 );
  mv[1]              = int2( 0, 0 );
  mv[2]              = int2( 0, 0 );
  Cu        n        = loadCu( nIdx );
  CuMotion  nm       = loadCuMotion( nIdx );
  int       posNeiX  = n.x;
  int       posNeiY  = n.y;
  const int posCurX  = cur.cu.x;
  const int posCurY  = cur.cu.y;
  int2      mvLT     = nm.mv[list][0];
  int2      mvRT     = nm.mv[list][1];
  const int2 mvLB    = nm.mv[list][2];
  bool      topCtu   = false;
  if( ( posNeiY + n.h ) % ( 1 << pc.log2Ctu ) == 0 && posNeiY + n.h == posCurY )
  {
    // (the neighbour's bottom-left and bottom-right subblocks' motion)
    mvLT = motionAt( n.x, n.y + n.h - 1 ).mv[list];
    mvRT = motionAt( n.x + n.w - 1, n.y + n.h - 1 ).mv[list];
    posNeiY += n.h;
    topCtu = true;
  }
  const int shift = 7;   // MAX_CU_DEPTH
  const int dHorX = ( mvRT.x - mvLT.x ) * ( 1 << ( shift - log2Of( n.w ) ) );
  const int dHorY = ( mvRT.y - mvLT.y ) * ( 1 << ( shift - log2Of( n.w ) ) );
  int       dVerX, dVerY;
  if( nm.affine6 && !topCtu )
  {
    dVerX = ( mvLB.x - mvLT.x ) * ( 1 << ( shift - log2Of( n.h ) ) );
    dVerY = ( mvLB.y - mvLT.y ) * ( 1 << ( shift - log2Of( n.h ) ) );
  }
  else
  {
    dVerX = -dHorY;
    dVerY = dHorX;
  }
  const int scaleHor = mvLT.x * ( 1 << shift );
  const int scaleVer = mvLT.y * ( 1 << shift );
  int2      t;
  t     = int2( scaleHor + dHorX * ( posCurX - posNeiX ) + dVerX * ( posCurY - posNeiY ), scaleVer + dHorY * ( posCurX - posNeiX ) + dVerY * ( posCurY - posNeiY ) );
  mv[0] = clipMv18( roundAffineMv( t, shift ) );
  t     = int2( scaleHor + dHorX * ( posCurX + cur.cu.w - posNeiX ) + dVerX * ( posCurY - posNeiY ),
                scaleVer + dHorY * ( posCurX + cur.cu.w - posNeiX ) + dVerY * ( posCurY - posNeiY ) );
  mv[1] = clipMv18( roundAffineMv( t, shift ) );
  if( is6param )
  {
    t     = int2( scaleHor + dHorX * ( posCurX - posNeiX ) + dVerX * ( posCurY + cur.cu.h - posNeiY ),
                  scaleVer + dHorY * ( posCurX - posNeiX ) + dVerY * ( posCurY + cur.cu.h - posNeiY ) );
    mv[2] = clipMv18( roundAffineMv( t, shift ) );
  }
}

// The CU of neighbouring block k if it is usable for inheritance (an affine CU of merge type DEFAULT_N, whose blocks
// are kAffineCu: an affine CU with the SbTMVP candidate has kSbtmvp blocks), or -1
int affineNeighbour( Cur cur, int k )
{
  const int2           p   = neighbourPos( cur.cu, k );
  const NeighbourWords n   = neighbourWords( k );
  const int            idx = neighbourCu( cur, p, n );
  if( idx < 0 || blockKind( n ) != kAffineCu || !isDiffMer( cur.cu.x, cur.cu.y, p.x, p.y ) )
  {
    return -1;
  }
  return idx;
}

// PU::getAffineControlPointCand: builds a constructed candidate of the given model from the corners mi[] (where
// available), and adds it at numValid when it is valid.
void affineControlPointCand( Cur cur, Motion mi[4], bool avail[4], int4 verIdx, int bcw, int modelIdx, int verNum, inout int numValid )
{
  const int shift     = 7;
  const int shiftHtoW = shift + log2Of( cur.cu.w ) - log2Of( cur.cu.h );
  int       refIdx[2] = { -1, -1 };
  int       dir       = 0;
  if( verNum == 2 )
  {
    if( !avail[verIdx.x] || !avail[verIdx.y] )
    {
      return;
    }
    for( int l = 0; l < 2; l++ )
    {
      if( mi[verIdx.x].refIdx[l] >= 0 && mi[verIdx.y].refIdx[l] >= 0 && mi[verIdx.x].refIdx[l] == mi[verIdx.y].refIdx[l] )
      {
        dir |= l + 1;
        refIdx[l] = mi[verIdx.x].refIdx[l];
      }
    }
  }
  else
  {
    if( !avail[verIdx.x] || !avail[verIdx.y] || !avail[verIdx.z] )
    {
      return;
    }
    for( int l = 0; l < 2; l++ )
    {
      if( mi[verIdx.x].refIdx[l] >= 0 && mi[verIdx.y].refIdx[l] >= 0 && mi[verIdx.z].refIdx[l] >= 0 && mi[verIdx.x].refIdx[l] == mi[verIdx.y].refIdx[l]
          && mi[verIdx.x].refIdx[l] == mi[verIdx.z].refIdx[l] )
      {
        dir |= l + 1;
        refIdx[l] = mi[verIdx.x].refIdx[l];
      }
    }
  }
  if( dir == 0 )
  {
    return;
  }
  int2 cMv[2][4];
  for( int l = 0; l < 2; l++ )
  {
    for( int i = 0; i < 4; i++ )
    {
      cMv[l][i] = int2( 0, 0 );
    }
    if( ( dir & ( l + 1 ) ) == 0 )
    {
      continue;
    }
    for( int i = 0; i < verNum; i++ )
    {
      const int v = i == 0 ? verIdx.x : i == 1 ? verIdx.y : verIdx.z;
      cMv[l][v]   = mi[v].mv[l];
    }
    if( modelIdx == 1 )
    {
      cMv[l][2] = clipMv18( cMv[l][3] + cMv[l][0] - cMv[l][1] );
    }
    else if( modelIdx == 2 )
    {
      cMv[l][1] = clipMv18( cMv[l][3] + cMv[l][0] - cMv[l][2] );
    }
    else if( modelIdx == 3 )
    {
      cMv[l][0] = clipMv18( cMv[l][1] + cMv[l][2] - cMv[l][3] );
    }
    else if( modelIdx == 5 )
    {
      const int2 v = int2( cMv[l][0].x * ( 1 << shift ) + ( cMv[l][2].y - cMv[l][0].y ) * ( 1 << shiftHtoW ),
                           cMv[l][0].y * ( 1 << shift ) - ( cMv[l][2].x - cMv[l][0].x ) * ( 1 << shiftHtoW ) );
      cMv[l][1]    = clipMv18( roundAffineMv( v, shift ) );
    }
  }
  for( int k = 0; k < 3; k++ )
  {
    s_affMv[0][k][numValid] = cMv[0][k];
    s_affMv[1][k][numValid] = cMv[1][k];
  }
  s_affRef[0][numValid] = refIdx[0];
  s_affRef[1][numValid] = refIdx[1];
  s_affDir[numValid]    = dir;
  s_affType[numValid]   = verNum == 2 ? 0 : 1;
  s_affBcw[numValid]    = dir == 3 ? bcw : 0;
  numValid++;
}

// clipColPos
int2 clipColPos( Cur cur, int2 p )
{
  const int ctuSize = 1 << pc.log2Ctu;
  const int ctuX    = ( cur.cu.x >> pc.log2Ctu ) << pc.log2Ctu;
  const int ctuY    = ( cur.cu.y >> pc.log2Ctu ) << pc.log2Ctu;
  const int horMax  = min( cur.rect.z, ctuX + ctuSize + 3 );   // (the right edge of the subpicture, or of the picture)
  const int horMin  = max( 0, ctuX );
  const int verMax  = min( pc.height - 1, ctuY + ctuSize - 1 );
  const int verMin  = max( 0, ctuY );
  return int2( min( horMax, max( horMin, p.x ) ), min( verMax, max( verMin, p.y ) ) );
}

// PU::getInterMergeSubPuMvpCand: sets the SbTMVP candidate at index count (its centre motion) and, with spread,
// writes its 8x8 sub-PUs' motion into the CU's 4x4 blocks.
bool subPuMvpCand( Cur cur, int count, bool spread )
{
  const bool interB = isInterB( cur.slice );
  const int  colPoc = colPicPoc( cur.slice );
  int2       tmv    = int2( 0, 0 );
  if( count > 0 )
  {
    if( ( s_affDir[0] & 1 ) != 0 && refPoc( cur.slice, 0, s_affRef[0][0] ) == colPoc )
    {
      tmv = s_affMv[0][0][0];
    }
    else if( interB && ( s_affDir[0] & 2 ) != 0 && refPoc( cur.slice, 1, s_affRef[1][0] ) == colPoc )
    {
      tmv = s_affMv[1][0][0];
    }
  }
  const int2 t      = changePrecision( tmv, 6, 2 );   // to whole samples
  int2       center = int2( cur.cu.x + ( cur.cu.w >> 1 ) + t.x, cur.cu.y + ( cur.cu.h >> 1 ) + t.y );
  center            = clipColPos( cur, center ) & ~7;

  bool  found = false;
  ColSlice colSlice;
  if( interDir( colMotionAt( cur, center.x, center.y, colSlice ) ) != 0 )
  {
    s_affDir[count] = 0;
    for( int l = 0; l < ( interB ? 2 : 1 ); l++ )
    {
      int2 mv;
      if( colocatedMvp( cur, l, center.x, center.y, 0, true, mv ) )
      {
        s_affMv[l][0][count] = mv;
        s_affRef[l][count]   = 0;
        s_affDir[count] |= 1 << l;
        s_affBcw[count] = 0;
        found           = true;
      }
      else
      {
        s_affMv[l][0][count] = int2( 0, 0 );
        s_affRef[l][count]   = -1;
        s_affDir[count] &= ~( 1 << l );
      }
    }
  }
  if( !found || !spread )
  {
    return found;
  }

  // The 8x8 sub-PUs
  const bool biRestricted = cur.cu.w + cur.cu.h <= 12;
  const int  w8           = max( cur.cu.w >> 3, 1 );
  const int  lw8          = log2Of( w8 );
  for( int u = g_lane; u < w8 * max( cur.cu.h >> 3, 1 ); u += 32 )
  {
    {
      const int  x      = cur.cu.x + ( u & ( w8 - 1 ) ) * 8;
      const int  y      = cur.cu.y + ( u >> lw8 ) * 8;
      Motion     mi     = noMotion();
      const int2 colPos = clipColPos( cur, int2( x + 4 + t.x, y + 4 + t.y ) );
      bool       here   = false;
      if( interDir( colMotionAt( cur, colPos.x, colPos.y, colSlice ) ) != 0 )
      {
        for( int l = 0; l < ( !biRestricted && interB ? 2 : 1 ); l++ )
        {
          int2 mv;
          if( colocatedMvp( cur, l, colPos.x, colPos.y, 0, true, mv ) )
          {
            mi.refIdx[l] = 0;
            mi.mv[l]     = mv;
            here         = true;
          }
        }
      }
      if( !here )
      {
        mi.mv[0]     = s_affMv[0][0][count];
        mi.mv[1]     = s_affMv[1][0][count];
        mi.refIdx[0] = s_affRef[0][count];
        mi.refIdx[1] = s_affRef[1][count];
      }
      if( biRestricted && interDir( mi ) == 3 )
      {
        mi.mv[1]     = int2( 0, 0 );
        mi.refIdx[1] = -1;
      }
      storeMotionAt( x, y, mi, kSbtmvp );
      storeMotionAt( x + 4, y, mi, kSbtmvp );
      storeMotionAt( x, y + 4, mi, kSbtmvp );
      storeMotionAt( x + 4, y + 4, mi, kSbtmvp );
    }
  }
  return true;
}

// PU::getAffineMergeCand up to candidate last; the list goes to the s_aff* arrays.
void affineMergeCandidates( Cur cur, int last )
{
  const int  maxCand = maxAffineCand( cur.slice );
  const bool interB  = isInterB( cur.slice );
  for( int i = 0; i < kMaxAffine; i++ )
  {
    clearAffineCand( i );
  }
  int numValid = 0;

  // SbTMVP
  const bool subPu = ( pc.sps & 1 ) != 0 && !( slicePoc( cur.slice ) == refPoc( cur.slice, 0, 0 ) && sliceIrap( cur.slice ) );
  // (With TMVP there is a colocated picture, unless its tables are missing --
  // the comparison shows such cases -- and then its motion must not be read.)
  if( subPu && tmvp( cur.slice ) && hasColPic( cur.slice ) )
  {
    int        pos = 0;
    const int2 lb  = int2( cur.cu.x, cur.cu.y + cur.cu.h - 1 );
    Motion     left;
    int        cuLeft;
    if( neighbourBlock( cur, kA1, cuLeft, left ) && isDiffMer( cur.cu.x, cur.cu.y, lb.x - 1, lb.y ) )
    {
      s_affDir[pos]      = interDir( left );
      s_affMv[0][0][pos] = left.mv[0];
      s_affRef[0][pos]   = left.refIdx[0];
      if( interB )
      {
        s_affMv[1][0][pos] = left.mv[1];
        s_affRef[1][pos]   = left.refIdx[1];
      }
      pos++;
    }
    // (the sub-PUs' motion is written only for a CU that takes this candidate: merge index 0)
    if( subPuMvpCand( cur, pos, mergeIdx( cur.cu ) == 0 ) )
    {
      s_affMrgType[numValid] = MRG_TYPE_SUBPU_ATMVP;
      numValid++;
      if( numValid == last + 1 || numValid == maxCand )
      {
        return;
      }
    }
  }

  if( ( pc.sps & 2 ) != 0 )
  {
    // Inherited candidates: the first affine neighbour on the left (A0, A1) and the first one above (B0, B1, B2)
    int npu[2];
    npu[0]       = -1;
    npu[1]       = -1;
    int numNeigh = 0;
    int n        = affineNeighbour( cur, kA0 );
    if( n < 0 )
    {
      n = affineNeighbour( cur, kA1 );
    }
    if( n >= 0 )
    {
      npu[numNeigh++] = n;
    }
    n = affineNeighbour( cur, kB0 );
    if( n < 0 )
    {
      n = affineNeighbour( cur, kB1 );
    }
    if( n < 0 )
    {
      n = affineNeighbour( cur, kB2 );
    }
    if( n >= 0 )
    {
      npu[numNeigh++] = n;
    }
    for( int i = 0; i < numNeigh; i++ )
    {
      const int      nIdx = npu[i];
      const CuMotion nm   = loadCuMotion( nIdx );
      int2           c0[3], c1[3];
      c0[0] = int2( 0, 0 );
      c0[1] = int2( 0, 0 );
      c0[2] = int2( 0, 0 );
      c1[0] = int2( 0, 0 );
      c1[1] = int2( 0, 0 );
      c1[2] = int2( 0, 0 );
      if( nm.interDir != 2 )
      {
        inheritedAffineMv( cur, nm.affine6, nIdx, 0, c0 );
      }
      if( interB && nm.interDir != 1 )
      {
        inheritedAffineMv( cur, nm.affine6, nIdx, 1, c1 );
      }
      for( int k = 0; k < 3; k++ )
      {
        s_affMv[0][k][numValid] = c0[k];
        s_affMv[1][k][numValid] = c1[k];
      }
      s_affRef[0][numValid] = nm.refIdx[0];
      s_affRef[1][numValid] = nm.refIdx[1];
      s_affDir[numValid]    = nm.interDir;
      s_affType[numValid]   = nm.affine6 ? 1 : 0;
      s_affBcw[numValid]    = nm.bcw;
      if( numValid == last )
      {
        return;
      }
      numValid++;
      if( numValid == maxCand )
      {
        return;
      }
    }

    // Constructed candidates, from the corners LT (B2, B3, A2), RT (B1, B0), LB (A1, A0) and RB (the temporal one)
    Motion mi[4];
    bool   avail[4];
    int    neighBcw[2] = { 0, 0 };
    for( int i = 0; i < 4; i++ )
    {
      mi[i]    = noMotion();
      avail[i] = false;
    }
    // (the corners' blocks in the order they are tried: LT B2, B3, A2; RT B1, B0; LB A1, A0)
    const int kLT[3] = { kB2, kB3, kA2 };
    for( int i = 0; i < 3; i++ )
    {
      const int2 p = neighbourPos( cur.cu, kLT[i] );
      int        idx;
      Motion     m;
      if( neighbourBlock( cur, kLT[i], idx, m ) && isDiffMer( cur.cu.x, cur.cu.y, p.x, p.y ) )
      {
        avail[0]    = true;
        mi[0]       = m;
        neighBcw[0] = neighbourBcw( kLT[i], idx );
        break;
      }
    }
    const int kRT[2] = { kB1, kB0 };
    for( int i = 0; i < 2; i++ )
    {
      const int2 p = neighbourPos( cur.cu, kRT[i] );
      int        idx;
      Motion     m;
      if( neighbourBlock( cur, kRT[i], idx, m ) && isDiffMer( cur.cu.x, cur.cu.y, p.x, p.y ) )
      {
        avail[1]    = true;
        mi[1]       = m;
        neighBcw[1] = neighbourBcw( kRT[i], idx );
        break;
      }
    }
    const int kLB[2] = { kA1, kA0 };
    for( int i = 0; i < 2; i++ )
    {
      const int2 p = neighbourPos( cur.cu, kLB[i] );
      int        idx;
      Motion     m;
      if( neighbourBlock( cur, kLB[i], idx, m ) && isDiffMer( cur.cu.x, cur.cu.y, p.x, p.y ) )
      {
        avail[2] = true;
        mi[2]    = m;
        break;
      }
    }
    if( tmvp( cur.slice ) )
    {
      bool c0Avail;
      int2 c0, c1;
      colPositions( cur, c0Avail, c0, c1 );
      int2 mv;
      if( c0Avail && colocatedMvp( cur, 0, c0.x, c0.y, 0, false, mv ) )
      {
        mi[3].mv[0]     = mv;
        mi[3].refIdx[0] = 0;
        avail[3]        = true;
      }
      if( interB && c0Avail && colocatedMvp( cur, 1, c0.x, c0.y, 0, false, mv ) )
      {
        mi[3].mv[1]     = mv;
        mi[3].refIdx[1] = 0;
        avail[3]        = true;
      }
    }
    const int4 models[6] = { int4( 0, 1, 2, 0 ), int4( 0, 1, 3, 0 ), int4( 0, 2, 3, 0 ), int4( 1, 2, 3, 0 ), int4( 0, 1, 0, 0 ), int4( 0, 2, 0, 0 ) };
    const int  start     = ( pc.sps & 4 ) != 0 ? 0 : 4;
    for( int m = start; m < 6; m++ )
    {
      affineControlPointCand( cur, mi, avail, models[m], m == 3 ? neighBcw[1] : neighBcw[0], m, m < 4 ? 3 : 2, numValid );
      if( numValid != 0 && numValid - 1 == last )
      {
        return;
      }
      if( numValid == maxCand )
      {
        return;
      }
    }
  }

  // Zero candidates
  for( int cnt = numValid; cnt < maxCand; cnt++ )
  {
    for( int k = 0; k < 3; k++ )
    {
      s_affMv[0][k][cnt] = int2( 0, 0 );
      s_affMv[1][k][cnt] = int2( 0, 0 );
    }
    s_affRef[0][cnt] = 0;
    s_affDir[cnt]    = 1;
    if( interB )
    {
      s_affRef[1][cnt] = 0;
      s_affDir[cnt]    = 3;
    }
    s_affType[cnt] = 0;
  }
}

// InterPrediction::isSubblockVectorSpreadOverLimit
bool spreadOverLimit( int a, int b, int c, int d, int predType )
{
  const int s4        = 4 << 11;
  const int filterTap = 6;
  if( predType == 3 )
  {
    int w = max( max( 0, 4 * a + s4 ), max( 4 * c, 4 * a + 4 * c + s4 ) ) - min( min( 0, 4 * a + s4 ), min( 4 * c, 4 * a + 4 * c + s4 ) );
    int h = max( max( 0, 4 * b ), max( 4 * d + s4, 4 * b + 4 * d + s4 ) ) - min( min( 0, 4 * b ), min( 4 * d + s4, 4 * b + 4 * d + s4 ) );
    w     = ( w >> 11 ) + filterTap + 3;
    h     = ( h >> 11 ) + filterTap + 3;
    return w * h > ( filterTap + 9 ) * ( filterTap + 9 );
  }
  int w = max( 0, 4 * a + s4 ) - min( 0, 4 * a + s4 );
  int h = max( 0, 4 * b ) - min( 0, 4 * b );
  w     = ( w >> 11 ) + filterTap + 3;
  h     = ( h >> 11 ) + filterTap + 3;
  if( w * h > ( filterTap + 9 ) * ( filterTap + 5 ) )
  {
    return true;
  }
  w = max( 0, 4 * c ) - min( 0, 4 * c );
  h = max( 0, 4 * d + s4 ) - min( 0, 4 * d + s4 );
  w = ( w >> 11 ) + filterTap + 3;
  h = ( h >> 11 ) + filterTap + 3;
  return w * h > ( filterTap + 5 ) * ( filterTap + 9 );
}

// PU::setAllAffineMv: derives the subblocks' motion of list l from the corners and writes it into the 4x4 blocks
// (mv[l] only); the corners are kept in the CU's motion record.
void setAllAffineMv( Cur cur, inout CuMotion c, int2 lt, int2 rt, int2 lb, int l, bool clipCpmvs, int dir )
{
  const int w     = cur.cu.w;
  const int h     = cur.cu.h;
  const int shift = 7;
  if( clipCpmvs )
  {
    lt = wrapMv2( lt );
    rt = wrapMv2( rt );
    if( c.affine6 )
    {
      lb = wrapMv2( lb );
    }
  }
  const int dHorX = ( rt.x - lt.x ) * ( 1 << ( shift - log2Of( w ) ) );
  const int dHorY = ( rt.y - lt.y ) * ( 1 << ( shift - log2Of( w ) ) );
  int       dVerX, dVerY;
  if( c.affine6 )
  {
    dVerX = ( lb.x - lt.x ) * ( 1 << ( shift - log2Of( h ) ) );
    dVerY = ( lb.y - lt.y ) * ( 1 << ( shift - log2Of( h ) ) );
  }
  else
  {
    dVerX = -dHorY;
    dVerY = dHorX;
  }
  const int  scaleHor = lt.x * ( 1 << shift );
  const int  scaleVer = lt.y * ( 1 << shift );
  const bool spread   = spreadOverLimit( dHorX, dHorY, dVerX, dVerY, dir );
  int2       flb      = int2( 0, 0 );
  if( spread )
  {
    flb = clipMv18( roundAffineMv( int2( scaleHor + dHorX * ( w >> 1 ) + dVerX * ( h >> 1 ), scaleVer + dHorY * ( w >> 1 ) + dVerY * ( h >> 1 ) ), shift ) );
  }
  const int w4  = w >> 2;
  const int lw4 = log2Of( w4 );
  for( int u = g_lane; u < w4 * ( h >> 2 ); u += 32 )
  {
    const int x  = ( u & ( w4 - 1 ) ) * 4;
    const int y  = ( u >> lw4 ) * 4;
    int2      mv = flb;
    if( !spread )
    {
      mv = clipMv18( roundAffineMv( int2( scaleHor + dHorX * ( 2 + x ) + dVerX * ( 2 + y ), scaleVer + dHorY * ( 2 + x ) + dVerY * ( 2 + y ) ), shift ) );
    }
    int* mi       = motionPtr( cur.cu.x + x, cur.cu.y + y );
    mi[2 * l]     = mv.x;
    mi[2 * l + 1] = mv.y;
  }
  c.mv[l][0] = lt;
  c.mv[l][1] = rt;
  c.mv[l][2] = lb;
}

// PU::addAffineMVPCandUnscaled: the model of neighbouring block k's CU for the AMVP list, if it is an affine CU of
// merge type DEFAULT_N (whose blocks are kAffineCu: an affine CU with the SbTMVP candidate has kSbtmvp blocks)
bool affineAmvpInherited( Cur cur, int list, int refIdx, int k, int amvr, bool cur6, out int2 mv[3] )
{
  mv[0]                  = int2( 0, 0 );
  mv[1]                  = int2( 0, 0 );
  mv[2]                  = int2( 0, 0 );
  const NeighbourWords n = neighbourWords( k );
  const int            idx = neighbourCu( cur, neighbourPos( cur.cu, k ), n );
  if( idx < 0 || blockKind( n ) != kAffineCu )
  {
    return false;
  }
  const CuMotion nm       = loadCuMotion( idx );
  const Motion neib       = blockMotion( n );
  const int    currRefPoc = refPoc( cur.slice, list, refIdx );
  for( int s = 0; s < 2; s++ )
  {
    const int l = s == 0 ? list : 1 - list;
    if( ( nm.interDir & ( l + 1 ) ) == 0 || refPoc( cur.slice, l, neib.refIdx[l] ) != currRefPoc )
    {
      continue;
    }
    inheritedAffineMv( cur, cur6, idx, l, mv );
    for( int k = 0; k < ( cur6 ? 3 : 2 ); k++ )
    {
      if( amvr == 0 )
      {
        mv[k] = roundToPrecision( mv[k], 6, 4 );
      }
      else if( amvr == 2 )
      {
        mv[k] = roundToPrecision( mv[k], 6, 2 );
      }
    }
    return true;
  }
  return false;
}

// PU::fillAffineMvpCand: returns the corners of candidate mvpIdx
void affineAmvpCandidate( Cur cur, int list, int refIdx, int amvr, int mvpIdx, bool cur6, out int2 cand[3] )
{
  int2 lts[2], rts[2], lbs[2];
  for( int i = 0; i < 2; i++ )
  {
    lts[i] = int2( 0, 0 );
    rts[i] = int2( 0, 0 );
    lbs[i] = int2( 0, 0 );
  }
  int num = 0;
  cand[0] = int2( 0, 0 );
  cand[1] = int2( 0, 0 );
  cand[2] = int2( 0, 0 );
  if( refIdx < 0 )
  {
    return;
  }
  int2 o[3];
  // Inherited candidates: from the left (below-left, left) and from above (above-right, above, above-left)
  if( affineAmvpInherited( cur, list, refIdx, kA0, amvr, cur6, o ) || affineAmvpInherited( cur, list, refIdx, kA1, amvr, cur6, o ) )
  {
    lts[num] = o[0];
    rts[num] = o[1];
    lbs[num] = o[2];
    num++;
  }
  if( affineAmvpInherited( cur, list, refIdx, kB0, amvr, cur6, o ) || affineAmvpInherited( cur, list, refIdx, kB1, amvr, cur6, o )
      || affineAmvpInherited( cur, list, refIdx, kB2, amvr, cur6, o ) )
  {
    lts[num] = o[0];
    rts[num] = o[1];
    lbs[num] = o[2];
    num++;
  }
  if( num < 2 )
  {
    // Constructed candidates from the corners' motion (addMVPCandUnscaled)
    const int currRefPoc = refPoc( cur.slice, list, refIdx );
    int       pattern    = 0;
    int2      v[3];
    v[0] = int2( 0, 0 );
    v[1] = int2( 0, 0 );
    v[2] = int2( 0, 0 );
    int2 mv;
    if( amvpSpatial( cur, list, currRefPoc, kB2, mv ) || amvpSpatial( cur, list, currRefPoc, kB3, mv ) || amvpSpatial( cur, list, currRefPoc, kA2, mv ) )
    {
      v[0] = mv;
      pattern |= 1;
    }
    if( amvpSpatial( cur, list, currRefPoc, kB1, mv ) || amvpSpatial( cur, list, currRefPoc, kB0, mv ) )
    {
      v[1] = mv;
      pattern |= 2;
    }
    if( amvpSpatial( cur, list, currRefPoc, kA1, mv ) || amvpSpatial( cur, list, currRefPoc, kA0, mv ) )
    {
      v[2] = mv;
      pattern |= 4;
    }
    for( int k = 0; k < 3; k++ )
    {
      if( amvr == 0 )
      {
        v[k] = roundToPrecision( v[k], 6, 4 );
      }
      else if( amvr == 2 )
      {
        v[k] = roundToPrecision( v[k], 6, 2 );
      }
    }
    if( pattern == 7 || ( pattern == 3 && !cur6 ) )
    {
      lts[num] = v[0];
      rts[num] = v[1];
      lbs[num] = v[2];
      num++;
    }
    if( num < 2 )
    {
      for( int i = 2; i >= 0 && num < 2; i-- )
      {
        if( ( pattern & ( 1 << i ) ) != 0 )
        {
          lts[num] = v[i];
          rts[num] = v[i];
          lbs[num] = v[i];
          num++;
        }
      }
      if( num < 2 && tmvp( cur.slice ) )
      {
        bool c0Avail;
        int2 c0, c1;
        colPositions( cur, c0Avail, c0, c1 );
        if( ( c0Avail && colocatedMvp( cur, list, c0.x, c0.y, refIdx, false, mv ) ) || colocatedMvp( cur, list, c1.x, c1.y, refIdx, false, mv ) )
        {
          if( amvr == 0 )
          {
            mv = roundToPrecision( mv, 6, 4 );
          }
          else if( amvr == 2 )
          {
            mv = roundToPrecision( mv, 6, 2 );
          }
          lts[num] = mv;
          rts[num] = mv;
          lbs[num] = mv;
          num++;
        }
      }
      // (the zero candidates are already zero)
    }
  }
  cand[0] = lts[mvpIdx];
  cand[1] = rts[mvpIdx];
  cand[2] = lbs[mvpIdx];
  if( amvr != 1 )
  {
    for( int k = 0; k < 3; k++ )
    {
      cand[k] = changePrecision( cand[k], 6, 4 );
    }
  }
}

// --------------------------------------------------------------------------------------------------------------------
// GPM
// --------------------------------------------------------------------------------------------------------------------

// g_GeoParams: angle | distance << 8 of each split
static const uint16_t kGeoParams[64] = { 256, 768, 2, 258, 514, 770, 3, 259, 515, 771, 4, 260, 516, 772, 5, 261, 517, 773, 264, 776, 11, 267,
                                         523, 779, 12, 268, 524, 780, 13, 269, 525, 781, 14, 270, 526, 782, 272, 784, 274, 530, 786, 275, 531, 787,
                                         276, 532, 788, 277, 533, 789, 280, 792, 283, 539, 795, 284, 540, 796, 285, 541, 797, 286, 542, 798 };
static const int8_t   kDis[32]       = { 8, 8, 8, 8, 4, 4, 2, 1, 0, -1, -2, -4, -4, -8, -8, -8, -8, -8, -8, -8, -4, -4, -2, -1, 0, 1, 2, 4, 4, 8, 8, 8 };

// Candidate idx of PU::getGeoMergeCandidates (each candidate is a uni-directional motion), from the full merge list
Motion geoCandidate( int numMerge, int idx )
{
  int numValid = 0;
  for( int i = 0; i < numMerge; i++ )
  {
    const Motion c      = cand( i );
    const int    parity = i & 1;
    if( ( interDir( c ) & ( 1 + parity ) ) != 0 )
    {
      if( numValid == idx )
      {
        Motion m         = noMotion();
        m.mv[parity]     = c.mv[parity];
        m.refIdx[parity] = c.refIdx[parity];
        return m;
      }
      numValid++;
      if( numValid == 6 )   // GEO_MAX_NUM_UNI_CANDS
      {
        break;
      }
      continue;
    }
    if( ( interDir( c ) & ( 2 - parity ) ) != 0 )
    {
      if( numValid == idx )
      {
        Motion m             = noMotion();
        m.mv[1 - parity]     = c.mv[1 - parity];
        m.refIdx[1 - parity] = c.refIdx[1 - parity];
        return m;
      }
      numValid++;
      if( numValid == 6 )
      {
        break;
      }
    }
  }
  return noMotion();
}

// PU::spanGeoMotionInfo: sets the 4x4 blocks' motion from the split and the partitions' candidates
void spanGeo( Cur cur, int splitDir, Motion m0, Motion m1 )
{
  Motion    bi = noMotion();
  const int d0 = interDir( m0 );
  const int d1 = interDir( m1 );
  if( d0 == 1 && d1 == 2 )
  {
    bi.mv[0]     = m0.mv[0];
    bi.mv[1]     = m1.mv[1];
    bi.refIdx[0] = m0.refIdx[0];
    bi.refIdx[1] = m1.refIdx[1];
  }
  else if( d0 == 2 && d1 == 1 )
  {
    bi.mv[0]     = m1.mv[0];
    bi.mv[1]     = m0.mv[1];
    bi.refIdx[0] = m1.refIdx[0];
    bi.refIdx[1] = m0.refIdx[1];
  }
  else if( d0 == 1 && d1 == 1 )
  {
    bi.mv[0]     = m1.mv[0];
    bi.refIdx[0] = m1.refIdx[0];
  }
  else if( d0 == 2 && d1 == 2 )
  {
    bi.mv[1]     = m1.mv[1];
    bi.refIdx[1] = m1.refIdx[1];
  }
  const int  angle    = int( kGeoParams[splitDir] & 0xff );
  const int  distance = int( kGeoParams[splitDir] >> 8 );
  const bool flip     = angle >= 13 && angle <= 27;
  const int  distX    = angle;
  const int  distY    = ( distX + 8 ) % 32;
  const int  w        = cur.cu.w;
  const int  h        = cur.cu.h;
  int        offsetX  = ( -w ) >> 1;
  int        offsetY  = ( -h ) >> 1;
  if( distance > 0 )
  {
    if( angle % 16 == 8 || ( angle % 16 != 0 && h >= w ) )
    {
      offsetY += angle < 16 ? ( ( distance * h ) >> 3 ) : -( ( distance * h ) >> 3 );
    }
    else
    {
      offsetX += angle < 16 ? ( ( distance * w ) >> 3 ) : -( ( distance * w ) >> 3 );
    }
  }
  const int w4  = w >> 2;
  const int lw4 = log2Of( w4 );
  for( int u = g_lane; u < w4 * ( h >> 2 ); u += 32 )
  {
    const int x         = u & ( w4 - 1 );
    const int y         = u >> lw4;
    const int lookUpY   = ( ( ( 4 * y + offsetY ) * 2 ) + 5 ) * int( kDis[distY] );
    const int motionIdx = ( ( ( 4 * x + offsetX ) * 2 ) + 5 ) * int( kDis[distX] ) + lookUpY;
    const int mask      = abs( motionIdx ) < 32 ? 2 : ( motionIdx <= 0 ? ( flip ? 0 : 1 ) : ( flip ? 1 : 0 ) );
    storeMotionAt( cur.cu.x + 4 * x, cur.cu.y + 4 * y, mask == 2 ? bi : mask == 0 ? m0 : m1, kGpm );
  }
}

// --------------------------------------------------------------------------------------------------------------------
// A CU
// --------------------------------------------------------------------------------------------------------------------

// Fills the CU's 4x4 blocks with one motion.
void spanMotion( Cur cur, Motion m, int kind )
{
  // (the CU lies within one CTU and its sizes are powers of two, so its blocks are addressed from its first block)
  const int lw4     = log2Of( cur.cu.w ) - 2;
  const int log2C4  = pc.log2Ctu - 2;
  const int ctuMask = ( 1 << pc.log2Ctu ) - 1;
  int*      first   = pc.motion + ( ( ctuOf( cur.cu.x, cur.cu.y ) << ( 2 * log2C4 ) ) + ( ( cur.cu.y & ctuMask ) >> 2 << log2C4 )
                                  + ( ( cur.cu.x & ctuMask ) >> 2 ) ) * 5;
  const int w4      = 1 << lw4;
  // The packed word: the reference indices, the kind, then BCW and AMVR's half-sample flag (for neighbourAt)
  const int packed  = int( ( uint( m.refIdx[0] ) & 0xff ) | ( ( uint( m.refIdx[1] ) & 0xff ) << 8 ) | ( uint( kind ) << 16 )
                            | ( ( uint( m.bcw ) & 7 ) << 24 ) | ( m.hpel ? 1u << 27 : 0u ) );
  for( int u = g_lane; u < w4 * ( cur.cu.h >> 2 ); u += 32 )
  {
    int* mi = first + ( ( ( u >> lw4 ) << log2C4 ) + ( u & ( w4 - 1 ) ) ) * 5;
    mi[0]   = m.mv[0].x;
    mi[1]   = m.mv[0].y;
    mi[2]   = m.mv[1].x;
    mi[3]   = m.mv[1].y;
    mi[4]   = packed;
  }
}

// Whether the CTU's luma CUs (records first .. first + count - 1, cached by cacheCtuCus) are all intra; all threads
// call it.
bool ctuAllIntra( int first, int count )
{
  bool inter = false;
  for( int i = first + g_lane; i < first + count; i += 32 )
  {
    const Cu c = loadCu( i );
    inter      = inter || ( ( c.flags & kChroma ) == 0 && c.predMode != MODE_INTRA );
  }
  return !WaveActiveAnyTrue( inter );
}

// The motion of a CTU whose luma CUs are all intra, as storeCuMotion and spanMotion write an intra CU's, by all the
// threads at once: the CUs' records, and every 4x4 block of the CTU in the picture (its luma CUs cover them all).
void storeNoMotion( int ctu, int first, int count )
{
  for( int i = first + g_lane; i < first + count; i += 32 )
  {
    if( ( loadCu( i ).flags & kChroma ) == 0 )
    {
      storeCuMotion( i, noCuMotion() );
    }
  }
  const Motion m      = noMotion();
  const int    packed = int( ( uint( m.refIdx[0] ) & 0xff ) | ( ( uint( m.refIdx[1] ) & 0xff ) << 8 ) | ( uint( kNotInter ) << 16 ) );
  const int    log2C4 = pc.log2Ctu - 2;
  const int    x0     = ( ctu % pc.widthInCtus ) << pc.log2Ctu;
  const int    y0     = ( ctu / pc.widthInCtus ) << pc.log2Ctu;
  int*         base   = pc.motion + ( ctu << ( 2 * log2C4 ) ) * 5;
  for( int e = g_lane; e < 1 << ( 2 * log2C4 ); e += 32 )
  {
    const int x = x0 + ( ( e & ( ( 1 << log2C4 ) - 1 ) ) << 2 );
    const int y = y0 + ( ( e >> log2C4 ) << 2 );
    if( x < pc.width && y < pc.height )
    {
      int* mi = base + e * 5;
      mi[0]   = m.mv[0].x;
      mi[1]   = m.mv[0].y;
      mi[2]   = m.mv[1].x;
      mi[3]   = m.mv[1].y;
      mi[4]   = packed;
    }
  }
}

// MergeCtx::setMergeInfo's restriction (restrictBiPredMergeCandsOne): no bi-prediction for 8x4 and 4x8
void restrictBi( Cur cur, inout Motion m )
{
  if( cur.cu.w + cur.cu.h <= 12 && interDir( m ) == 3 )
  {
    m.refIdx[1] = -1;
    m.mv[1]     = int2( 0, 0 );
    m.bcw       = 0;
  }
}

// DecCu::xDeriveCUMV: derives the CU's motion and writes it to the CU's motion record and its 4x4 blocks
void deriveCu( Cur cur )
{
  Cu cu = cur.cu;
  fetchNeighbours( cu );   // (for the lists below)
  CuMotion c = noCuMotion();
  c.affine6   = affine6Of( cu );
  c.hpel      = imvOf( cu ) == 3;   // the parse's imv, unless the derivation sets it
  c.bcw       = bcwIdxOf( cu );
  c.interDir  = interDirOf( cu );
  for( int l = 0; l < 2; l++ )
  {
    for( int k = 0; k < 3; k++ )
    {
      c.mv[l][k] = mvdOf( cu, l, k );   // (what the CU keeps where the derivation does not set the motion)
    }
    c.refIdx[l] = refIdxOf( cu, l );
  }

  if( cu.predMode == MODE_IBC )
  {
    c.kind     = kIbc;
    c.interDir = 1;
    Motion m   = noMotion();
    if( ( cu.flags & kMerge ) != 0 )
    {
      // getIBCMergeCandidates + setMergeInfo: imv 0, BCW default
      const int idx = mergeIdx( cu );
      ibcMergeCandidates( cur, idx );
      const Motion b = cand( idx );
      c.mv[0][0]     = b.mv[0];
      c.mv[1][0]     = b.mv[1];
      c.refIdx[0]    = b.refIdx[0];
      c.refIdx[1]    = b.refIdx[1];
      c.hpel         = false;
      c.bcw          = 0;
      c.mrgType      = MRG_TYPE_IBC;
    }
    else
    {
      // fillIBCMvpCand: the first two merge candidates, rounded to whole or 4 samples
      ibcMergeCandidates( cur, 1 );
      const int  imv = imvOf( cu );
      const int2 p   = roundToPrecision( s_candMv[0][mvpIdx( cu, 0 )], 6, imv == 2 ? 0 : 2 );
      const int2 d   = imv != 0 ? changePrecision( mvdOf( cu, 0, 0 ), amvrPrecision( imv ), 6 ) : mvdOf( cu, 0, 0 ) * 16;
      c.mv[0][0]     = wrapMv2( p + d );
    }
    // spanMotionInfo: no reference indices for IBC
    m.mv[0] = c.mv[0][0];
    m.mv[1] = c.mv[1][0];
    storeCuMotion( cu.index, c );
    spanMotion( cur, m, kIbc );
    // IBC's history (not for blocks of 16 samples or fewer)
    if( cu.w * cu.h > 16 )
    {
      Motion h    = m;
      h.refIdx[0] = 0;   // MH_NOT_VALID + 1
      h.bcw       = 0;
      h.hpel      = c.hpel;
      addToLut( 1, h );
    }
    return;
  }

#if !LITE
  if( ( cu.flags & kMerge ) != 0 && ( cu.flags & kGeo ) != 0 )
  {
    // getGeoMergeCandidates (from the full merge list) + spanGeoMotionInfo
    c.kind        = kGpm;
    const int num = mergeCandidates( cur, -1 );
    Motion    m0  = geoCandidate( num, geoIdx( cu, 0 ) );
    Motion    m1  = geoCandidate( num, geoIdx( cu, 1 ) );
    c.mv[0][1]    = m0.mv[interDir( m0 ) == 1 ? 0 : 1];
    c.mv[1][1]    = m1.mv[interDir( m1 ) == 1 ? 0 : 1];
    storeCuMotion( cu.index, c );
    // The partitions' inter direction << 4 | reference index (CodingUnit's interDirrefIdxGeo0, 1)
    const uint g0 = uint( interDir( m0 ) ) << 4 | uint( m0.refIdx[interDir( m0 ) == 1 ? 0 : 1] );
    const uint g1 = uint( interDir( m1 ) ) << 4 | uint( m1.refIdx[interDir( m1 ) == 1 ? 0 : 1] );
    pc.cuMotion[cu.index * 16 + 13] = int( g0 | g1 << 8 );
    spanGeo( cur, geoSplitDir( cu ), m0, m1 );
    return;
  }

  if( ( cu.flags & kAffine ) != 0 && ( cu.flags & kMerge ) != 0 )
  {
    // The subblock merge list: SbTMVP, inherited and constructed affine, and zero candidates
    const int idx = mergeIdx( cu );
    affineMergeCandidates( cur, idx );
    c.interDir = s_affDir[idx];
    c.affine6  = s_affType[idx] != 0;
    c.bcw      = s_affBcw[idx];
    c.mrgType  = s_affMrgType[idx];
    if( c.mrgType == MRG_TYPE_SUBPU_ATMVP )
    {
      // (the sub-PUs' motion is already in the 4x4 blocks; the CU's reference
      // indices are those of the list's first entry, as in vvdec)
      c.kind      = kSbtmvp;
      c.refIdx[0] = s_affRef[0][idx];
      c.refIdx[1] = s_affRef[1][idx];
      storeCuMotion( cu.index, c );
      return;
    }
    c.kind = kAffineCu;
    for( int l = 0; l < 2; l++ )
    {
      if( numRefIdx( cur.slice, l ) > 0 )
      {
        c.refIdx[l] = s_affRef[l][idx];
        setAllAffineMv( cur, c, s_affMv[l][0][idx], s_affMv[l][1][idx], s_affMv[l][2][idx], l, false, c.interDir );
      }
    }
  }
  else if( ( cu.flags & kAffine ) != 0 )
  {
    // Affine AMVP
    c.kind        = kAffineCu;
    const int imv = imvOf( cu );
    const int sh  = imv == 2 ? 2 : 0;
    for( int l = 0; l < 2; l++ )
    {
      if( numRefIdx( cur.slice, l ) > 0 && ( c.interDir & ( 1 << l ) ) != 0 )
      {
        int2 p[3];
        affineAmvpCandidate( cur, l, c.refIdx[l], imv, mvpIdx( cu, l ), c.affine6, p );
        const int2 d0 = mvdOf( cu, l, 0 ) * ( 1 << sh );
        const int2 d1 = mvdOf( cu, l, 1 ) * ( 1 << sh );
        int2       lt = p[0] + d0;
        int2       rt = p[1] + d1 + d0;
        if( imv != 1 )
        {
          lt = changePrecision( lt, 4, 6 );
          rt = changePrecision( rt, 4, 6 );
        }
        int2 lb = int2( 0, 0 );
        if( c.affine6 )
        {
          const int2 d2 = mvdOf( cu, l, 2 ) * ( 1 << sh );
          lb            = p[2] + d2 + d0;
          if( imv != 1 )
          {
            lb = changePrecision( lb, 4, 6 );
          }
        }
        setAllAffineMv( cur, c, lt, rt, lb, l, true, c.interDir );
      }
    }
  }

  if( c.kind == kAffineCu )
  {
    // spanMotionInfo for affine: set the lists' reference indices, and zero the motion of the unused lists
    const int w4  = cu.w >> 2;
    const int lw4 = log2Of( w4 );
    for( int u = g_lane; u < w4 * ( cu.h >> 2 ); u += 32 )
    {
      {
        const int x  = ( u & ( w4 - 1 ) ) * 4;
        const int y  = ( u >> lw4 ) * 4;
        int*      mi = motionPtr( cu.x + x, cu.y + y );
        for( int l = 0; l < 2; l++ )
        {
          if( c.refIdx[l] < 0 )
          {
            mi[2 * l]     = 0;
            mi[2 * l + 1] = 0;
          }
        }
        mi[4] = int( ( uint( c.refIdx[0] ) & 0xff ) | ( ( uint( c.refIdx[1] ) & 0xff ) << 8 ) | ( uint( kAffineCu ) << 16 ) );
      }
    }
    storeCuMotion( cu.index, c );
    return;
  }
#endif

  // Translational motion
  c.kind   = kTranslation;
  Motion m = noMotion();
  if( ( cu.flags & kMerge ) != 0 )
  {
    if( ( cu.flags & kMmvd ) != 0 )
    {
      const int idx  = mmvdIdx( cu );
      const int base = idx / 32;
      const int num  = mergeCandidates( cur, base + 1 );
      m              = mmvdMotion( cur, idx, num );   // (hpel comes from the base candidate)
      // The end of setMmvdMergeCandiInfo: BCW from the merge list's entry at
      // the base index, the motion clipped, no bi-prediction for 8x4 and 4x8
      const Motion b = cand( base );
      m.bcw          = interDir( b ) == 3 ? b.bcw : 0;
      for( int l = 0; l < 2; l++ )
      {
        if( m.refIdx[l] >= 0 )
        {
          m.mv[l] = clipMv18( m.mv[l] );
        }
      }
      restrictBi( cur, m );
    }
    else
    {
      // (CIIP CUs as well)
      const int idx = mergeIdx( cu );
      mergeCandidates( cur, idx );
      m     = cand( idx );
      m.bcw = interDir( m ) == 3 ? m.bcw : 0;
      restrictBi( cur, m );
    }
    c.mrgType = m.mrgType;
    c.hpel    = m.hpel;
    c.bcw     = m.bcw;
  }
  else
  {
    // AMVP (with or without applyImv, the same here); with SMVD, the second list's motion comes from the parse
    const int amvr = imvOf( cu );
    m.hpel         = amvr == 3;
    m.bcw          = bcwIdxOf( cu );
    for( int l = 0; l < 2; l++ )
    {
      m.mv[l]     = mvdOf( cu, l, 0 );
      m.refIdx[l] = refIdxOf( cu, l );
      if( ( c.interDir & ( 1 << l ) ) != 0 && numRefIdx( cur.slice, l ) > 0 )
      {
        const int2 d = changePrecision( mvdOf( cu, l, 0 ), amvrPrecision( amvr ), 6 );
        m.mv[l]      = wrapMv2( amvpCandidate( cur, l, m.refIdx[l], amvr, mvpIdx( cu, l ) ) + d );
      }
    }
  }
  c.mv[0][0]  = m.mv[0];
  c.mv[1][0]  = m.mv[1];
  c.refIdx[0] = m.refIdx[0];
  c.refIdx[1] = m.refIdx[1];
  c.interDir  = interDir( m );
  storeCuMotion( cu.index, c );
  spanMotion( cur, m, kTranslation );

  // Update the history (not for affine or GPM CUs; subject to the parallel merge level)
  const bool on = ( ( cu.x + cu.w ) >> pc.plevel ) > ( cu.x >> pc.plevel ) && ( ( cu.y + cu.h ) >> pc.plevel ) > ( cu.y >> pc.plevel );
  if( on )
  {
    Motion h = m;
    h.bcw    = interDir( m ) == 3 ? m.bcw : 0;
    h.hpel   = c.hpel;
    addToLut( 0, h );
  }
}

// --------------------------------------------------------------------------------------------------------------------
// The entry points
// --------------------------------------------------------------------------------------------------------------------

#if CUMAP
// The CU map: one workgroup per CU writes the CU's index at each of its 4x4 blocks (luma tree: cuMap; chroma tree:
// cuMapC).
[shader( "compute" )]
[numthreads( 64, 1, 1 )]
void main( uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID )
{
  const int index = int( group.y * 32768 + group.x );
  if( index >= pc.numCus )
  {
    return;
  }
  Cu         cu    = loadCu( index );
  uint*      r     = pc.cus + index * 32;
  const uint valid = r[20];
  if( ( valid & ( 1u << 17 ) ) != 0 )
  {
    const int w4  = cu.w >> 2;
    const int lw4 = log2Of( w4 );
    const int n   = w4 * ( cu.h >> 2 );
    for( int i = int( local.x ); i < n; i += 64 )
    {
      const int y = ( cu.y >> 2 ) + ( i >> lw4 );
      const int x = ( cu.x >> 2 ) + ( i & ( w4 - 1 ) );
      pc.cuMap[y * pc.widthIn4 + x] = uint( index );
    }
  }
  if( ( valid & ( 1u << 16 ) ) != 0 )
  {
    // the chroma block, in luma samples
    const int cx = int( r[19] & 0xffff ) << pc.csx;
    const int cy = int( r[19] >> 16 ) << pc.csy;
    const int cw = int( r[20] & 0xff ) << pc.csx;
    const int ch = int( ( r[20] >> 8 ) & 0xff ) << pc.csy;
    const int w4  = max( cw >> 2, 1 );
    const int lw4 = log2Of( w4 );
    const int n   = w4 * max( ch >> 2, 1 );
    for( int i = int( local.x ); i < n; i += 64 )
    {
      const int y = ( cy >> 2 ) + ( i >> lw4 );
      const int x = ( cx >> 2 ) + ( i & ( w4 - 1 ) );
      pc.cuMapC[y * pc.widthIn4 + x] = uint( index );
    }
  }
}
#else
// The segment this workgroup derives. The segments are taken in order from a counter, so that a segment waits only
// for segments whose workgroups already run.
groupshared int s_segment;

// Tile column tc's CTU columns [x, y): the tile columns begin at the CTU columns whose records carry the flag of a
// tile column's first CTU (row 0's records; every CTU row has the same tile columns).
int2 tileColumnSpan( int tc )
{
  int2 span = int2( 0, pc.widthInCtus );
  int  seen = 0;
  for( int base = 0; base < pc.widthInCtus; base += 32 )
  {
    const int c    = base + g_lane;
    uint      mask = WaveActiveBallot( c < pc.widthInCtus && ctuTileColStart( c ) ).x;
    while( mask != 0 )
    {
      const int first = base + int( firstbitlow( mask ) );
      if( seen == tc )
      {
        span.x = first;
      }
      else if( seen == tc + 1 )
      {
        span.y = first;
      }
      seen++;
      mask &= mask - 1;
    }
  }
  return span;
}

// Puts the record of the CTU's slice in shared memory (s_curSlice), unless it is there already; all threads call it.
// After the CTU before has finished with the record (the CTU's end has barriers).
void loadCurSlice( int ctu )
{
  const uint idx = ctuSliceIdx( ctu );
  if( idx != g_curSliceIdx )
  {
    uint* r                 = pc.slices + pc.widthInCtus * pc.heightInCtus + idx * kSliceWords;
    s_curSlice[g_lane]      = r[g_lane];
    s_curSlice[g_lane + 32] = r[g_lane + 32];
    g_curSliceIdx           = idx;
    g_colSliceIdx           = 0xffffffffu;   // (s_colSlice holds a slice of the old slice's colocated picture)
    GroupMemoryBarrierWithGroupSync();
  }
}

// For a CTU whose slice uses TMVP and has a colocated picture (after loadCurSlice): g_colCtuSlice gets the colocated
// picture's slices of the CTU and of the one right of it (the temporal candidates of the CTU's CUs lie in those two),
// and s_colSlice the record of the first, unless it holds it already; all threads call it.
void loadColSlices( int ctu, int col )
{
  uint* colTable   = colSlices( CurSlice() );
  g_colCtuSlice[0] = colTable[ctu];
  g_colCtuSlice[1] = col + 1 < pc.widthInCtus ? colTable[ctu + 1] : g_colCtuSlice[0];
  if( g_colCtuSlice[0] != g_colSliceIdx )
  {
    uint* r                 = colTable + pc.widthInCtus * pc.heightInCtus + g_colCtuSlice[0] * kSliceWords;
    s_colSlice[g_lane]      = r[g_lane];
    s_colSlice[g_lane + 32] = r[g_lane + 32];
    g_colSliceIdx           = g_colCtuSlice[0];
    GroupMemoryBarrierWithGroupSync();
  }
}

// The most 8x8 blocks of a CTU each thread copies: a CTU of 128 x 128 has 256, 8 for each of the 32 threads
static const int kColPerThread = 8;

// Copies the colocated motion of the CTU's 8x8 blocks (their top-left 4x4 block's motion) into pc.colMotion. Each
// thread reads all its blocks' motion before it writes any: the compiler cannot move a read of pc.motion past a write
// to pc.colMotion (they might overlap), so reading and writing word by word would wait for memory 40 times.
void copyColMotion( int ctu )
{
  const int across8  = 1 << ( pc.log2Ctu - 3 );
  const int num8x8   = across8 * across8;
  const int ctuSize4 = 1 << ( pc.log2Ctu - 2 );
  uint4     head[kColPerThread];   // words 0-3 of each block's motion
  uint      tail[kColPerThread];   // word 4
  [unroll]
  for( int j = 0; j < kColPerThread; j++ )
  {
    const int e = g_lane + 32 * j;
    head[j]     = uint4( 0, 0, 0, 0 );
    tail[j]     = 0;
    if( e < num8x8 )
    {
      const int x4  = ( e & ( across8 - 1 ) ) * 2;
      const int y4  = ( e >> ( pc.log2Ctu - 3 ) ) * 2;
      uint*     src = (uint*) ( pc.motion + ( ctu * ctuSize4 * ctuSize4 + y4 * ctuSize4 + x4 ) * 5 );
      head[j]       = uint4( src[0], src[1], src[2], src[3] );
      tail[j]       = src[4];
    }
  }
  [unroll]
  for( int j = 0; j < kColPerThread; j++ )
  {
    const int e = g_lane + 32 * j;
    if( e < num8x8 )
    {
      uint* dst = (uint*) ( pc.colMotion + ( ctu * num8x8 + e ) * 5 );
      dst[0]    = head[j].x;
      dst[1]    = head[j].y;
      dst[2]    = head[j].z;
      dst[3]    = head[j].w;
      dst[4]    = tail[j];
    }
  }
}

[shader( "compute" )]
[numthreads( 32, 1, 1 )]
void main( uint3 local : SV_GroupThreadID )
{
  g_lane = int( local.x );
  // The segments are taken in order (row by row, each row's tile columns from left to right), so that the segment
  // above a segment was taken before it.
  const int segments = pc.heightInCtus * pc.tileCols;
  if( g_lane == 0 )
  {
    s_segment = int( pc.progress[segments].add( 1u ) );
  }
  GroupMemoryBarrierWithGroupSync();
  const int segment = s_segment;
  if( segment >= segments )
  {
    return;
  }
  const int  row       = segment / pc.tileCols;
  const int2 span      = tileColumnSpan( segment % pc.tileCols );
  const bool waitAbove = row > 0 && ctuTile( row * pc.widthInCtus + span.x ) == ctuTile( ( row - 1 ) * pc.widthInCtus + span.x );
  for( int col = span.x; col < span.y; col++ )
  {
    const int ctu   = row * pc.widthInCtus + col;
    const int first = int( pc.ctus[ctu * 4] );
    const int count = int( pc.ctus[ctu * 4 + 1] & 0xffff );
    cacheCtuCus( first, count );
    // A CTU whose luma CUs are all intra (an intra slice's, unless IBC) reads no neighbour's motion: it does not wait
    // for the segment above, and its CUs' motion, none, is written by all the threads at once rather than CU by CU.
    const bool allIntra = ctuAllIntra( first, count );
    // Wait until the segment above, in the same tile, is past the CTU above-right (or done, at the tile's right
    // edge). With WPP the CTU above-right is out of the CUs' reach (cuAt): past the CTU above is enough.
    if( waitAbove && !allIntra )
    {
      if( g_lane == 0 )
      {
        const uint need = uint( min( col + 2 - pc.wpp, span.y ) - span.x );
        while( pc.progress[segment - pc.tileCols].load( MemoryOrder.Relaxed ) < need )
        {
        }
      }
      AllMemoryBarrierWithGroupSync();
    }
    if( ctuTileColStart( ctu ) )
    {
      s_lutSize[0] = 0;
      s_lutSize[1] = 0;
    }
    if( allIntra )
    {
      storeNoMotion( ctu, first, count );
    }
    else
    {
      Cur cur;
      cur.ctu = ctu;
      loadCurSlice( ctu );
      cur.sliceIdx = independentIdx( cur.slice );
      // (only with TMVP: the colocated picture has the picture's size then, and the CUs read it only then; without,
      // hasColPic may name a reference picture of another size, whose slice table has other CTUs)
      if( tmvp( cur.slice ) && hasColPic( cur.slice ) )
      {
        loadColSlices( ctu, col );
      }
      cur.tile     = ctuTile( ctu );
      cur.rect     = ctuRect( ctu );
      setNeighbourCtus( cur, col, row );
      for( int i = first; i < first + count; i++ )
      {
        cur.cu = loadCu( i );
        if( ( cur.cu.flags & kChroma ) != 0 )
        {
          continue;
        }
        if( cur.cu.predMode == MODE_INTRA )
        {
          storeCuMotion( i, noCuMotion() );
          spanMotion( cur, noMotion(), kNotInter );
        }
        else
        {
          deriveCu( cur );
        }
        // (each thread's writes of the CU's blocks become visible to all before the next CU looks them up)
        workgroupBarrierAll();
      }
    }
    AllMemoryBarrierWithGroupSync();   // (makes the CTU's motion visible to the rows below)
    if( g_lane == 0 )
    {
      pc.progress[segment].store( uint( col + 1 - span.x ), MemoryOrder.Release );
    }
    // The CTU's colocated motion, from its now final motion. After the progress, as the rows below do not read it,
    // and in this pass rather than one of its own: the front end's first part waits for every pass in it.
    if( pc.colOn != 0 )
    {
      copyColMotion( ctu );
    }
  }
}
#endif
