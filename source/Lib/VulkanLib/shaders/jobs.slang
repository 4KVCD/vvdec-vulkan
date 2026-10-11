// Builds the jobs of a picture's reconstruction on the GPU (the CPU's version is
// vvdec's DecCu::TaskGpuCtu: xGpuTrafoJobs, xGpuIntraJobs, xGpuIbcJobs,
// xGpuInterJobs, xGpuCiipJobs), from the parsed CUs and TUs (GpuFrontEnd.h's
// records) and the motion the GPU derived (shaders/mider.slang). It writes the
// residual jobs (GpuJobs.h's TrafoJob), the inter jobs (InterJob) and the intra
// jobs (IntraJob), each list in the CPU's order: CTU by CTU, and within a CTU,
// the CUs in decoding order.
//
// The passes (pc.pass):
// 0 COUNT: one thread per CU counts the CU's jobs in each list and stores its
//          offset within its block of 256 CUs; each block stores its totals.
// 1 SCAN:  one workgroup turns the blocks' totals into offsets and writes the
//          overall totals.
// 2 WRITE: one thread per CU writes the CU's jobs (each intra job's wave is
//          left at 0).
//          In both, the jobs of an SbTMVP CU's sub-PUs are made by the whole
//          workgroup together (sbtmvpJobs).
// 3 WAVES: sets each intra job's wave, in decoding order: one above the
//          highest wave of the samples the job reads, where inter's samples
//          are wave 0 (GpuJobs.h's WaveMap). One 32-thread workgroup per CTU
//          row; the rows run as a wavefront, as vvdec's CTU tasks run them.
//
// There are five lists: 0 the residual jobs, 1 inter's regular jobs, 2 the
// intra jobs, 3 inter's BDOF jobs, 4 inter's affine pairs. Inter's three lists
// lie one after another in the inter buffer, as GpuRecon::recordInter lays
// them out for the inter, inter_bdof and inter_affine passes.

#include "common.slang"
#include "scan.slang"

// COUNT_ONLY=1: the variant for pass 0 (jobs_count). With the writes off at compile time, the compiler drops the work
// that only fills in the jobs' words, which COUNT has no use for: what is left decides how many jobs each list gets.
#ifndef COUNT_ONLY
#  define COUNT_ONLY 0
#endif

struct Params
{
  uint* cus;         // CuRecord, 32 words each
  uint* tus;         // TuRecord, 16 words each
  uint* ctus;        // CtuRecord, 4 words each
  uint* slices;      // mider.slang's slice table
  uint* jobSlices;   // 256 words per slice (GpuFrontEnd.h's JobSlice)
  uint* cuMapL;      // the index of the luma-tree CU at each 4x4 luma block
  uint* cuMapC;      // the same for the chroma tree (in a single tree, the luma tree's CUs)
  int*  cuMotion;    // mider.slang's motion records per CU (16 words each)
  int*  motion;      // mider.slang's MotionInfo per 4x4 block (5 words)
  uint* tables;      // the constant tables (GpuFrontEnd.h's JobTables)
  uint* scaling;     // per list (6), log2 width (7), log2 height (7): the dequantization matrix's offset + 1 (0: none)
  uint* counts;      // per CU: the offsets of its jobs within its block, one word per list (5 words)
  uint* blocks;      // per block: its job totals, which SCAN replaces with offsets (5 words); then the totals (5 words)
  uint* trafo;       // out: TrafoJob, 8 words each
  uint* inter;       // out: InterJob, 12 words each
  uint* intra;       // out: IntraJob, 12 words each
  int   numCus;
  int   numBlocks;
  int   widthInCtus;
  int   heightInCtus;
  int   log2Ctu;
  int   width;       // luma samples
  int   height;
  int   widthIn4;
  int   chromaFormat;   // 0 4:0:0, 1 4:2:0, 2 4:2:2, 3 4:4:4
  int   csx;            // chroma subsampling shifts (0, 1)
  int   csy;
  int   bitDepth;
  int   qpBdOffset;
  int   tsMinQp;        // transform skip's minimum QP: 4 + 6 * (the SPS's internal minus input bit depth)
  uint  sps;            // kSps* flags
  uint  ph;             // kPh* flags (from the picture header and the PPS, and what runs on the GPU)
  int   pass;
  int   capTrafo;       // room in each output, in jobs
  int   capInter;
  int   capIntra;
  int   num8x8;         // room for DMVR's refinements per CTU (PreCalcValues::num8x8CtuBlks)
  uint  waveTag;        // WAVES: the picture's tag in the wave map (see mapWave)
  uint* waveMap;        // WAVES: per component, the wave of each area of 4x4 luma samples (GpuJobs.h's WaveMap)
  Atomic<uint>* progress;   // WAVES: per CTU row, the CTUs done; then the rows taken (zeroed before the dispatch)
  uint* args;        // SCAN: the front end's indirect arguments (GpuRecon's m_frontArgs): residual and inter passes
};

[[vk::push_constant]] Params pc;

// The SPS's flags
static const uint kSpsMts          = 1;
static const uint kSpsImplicitMts  = 2;
static const uint kSpsIntraMts     = 4;
static const uint kSpsInterMts     = 8;
static const uint kSpsLfnst        = 16;
static const uint kSpsNoSlLfnst    = 32;    // getDisableScalingMatrixForLfnstBlks
static const uint kSpsNoSlAct      = 64;    // getScalingMatrixForAlternativeColourSpaceDisabledFlag
static const uint kSpsSlActSpace   = 128;   // getScalingMatrixDesignatedColourSpaceFlag
static const uint kSpsBdof         = 256;
static const uint kSpsDmvr         = 512;
static const uint kSpsProf         = 1024;
static const uint kSpsBcw          = 2048;
static const uint kSpsWpp          = 4096;
// The picture's flags (picture header and PPS)
static const uint kPhNoDmvr        = 1;
static const uint kPhNoBdof        = 2;
static const uint kPhNoProf        = 4;
static const uint kPhJointSign     = 8;
static const uint kPhLmcs          = 16;
static const uint kPhLmcsChroma    = 32;
static const uint kPpsWp           = 64;
static const uint kPpsWpBi         = 128;
static const uint kPhTrafo         = 256;   // the GPU makes the residuals
static const uint kPhInter         = 512;   // the GPU makes inter's prediction

// CuRecord's flags
static const uint kMerge   = 1;
static const uint kSkip    = 2;
static const uint kMmvd    = 4;
static const uint kAffine  = 8;
static const uint kGeo     = 16;
static const uint kCiip    = 32;
static const uint kAffine6 = 128;

static const int MODE_INTER = 0;
static const int MODE_INTRA = 1;
static const int MODE_IBC   = 2;

static const int MRG_TYPE_DEFAULT_N   = 0;
static const int MRG_TYPE_SUBPU_ATMVP = 1;

static const int PLANAR_IDX    = 0;
static const int DC_IDX        = 1;
static const int HOR_IDX       = 18;
static const int DIA_IDX       = 34;
static const int VER_IDX       = 50;
static const int VDIA_IDX      = 66;
static const int NUM_LUMA_MODE = 67;
static const int LM_CHROMA_IDX = 67;
static const int MDLM_L_IDX    = 68;
static const int MDLM_T_IDX    = 69;
static const int DM_CHROMA_IDX = 70;

static const int DCT2 = 0;
static const int DCT8 = 1;
static const int DST7 = 2;

// GpuJobs.h's job flags and kinds
static const uint kTrafoTransformSkip = 1;
static const uint kTrafoBdpcmHor      = 2;
static const uint kTrafoBdpcmVer      = 4;
static const uint kTrafoZero          = 8;

static const int  kIntraAngular = 0;
static const int  kIntraBdpcm   = 1;
static const int  kIntraCclm    = 2;
static const int  kIntraIbc     = 3;
static const int  kIntraMip     = 4;
static const int  kIntraLmcs    = 5;
static const int  kIntraCiip    = 6;
static const uint kIntraResidual      = 1;
static const uint kIntraFiltered      = 2;
static const uint kIntraMipTransposed = 4;
static const uint kIntraIsp           = 8;

static const uint kInterL0        = 1;
static const uint kInterL1        = 2;
static const uint kInterDmvr      = 4;
static const uint kInterHalfPel   = 8;
static const uint kInterResidualY = 16;
static const uint kInterLumaMap   = 128;
static const uint kInterDeferCb   = 256;
static const uint kInterDeferY    = 1024;
static const uint kInterBcw       = 2048;
static const uint kInterBdof      = 4096;
static const uint kInterAffine    = 8192;
static const uint kInterGeo       = 16384;
static const uint kInterWp        = 32768;

// Offsets into the constant tables (GpuFrontEnd.h's JobTables)
static const int T_LFNST  = 0;      // g_lfnstLut
static const int T_C422   = 128;    // g_chroma422IntraAngleMappingTable
static const int T_GEO    = 256;    // g_GeoParams: angle | distance << 8
static const int T_MASK   = 320;    // g_angle2mask
static const int T_MIRROR = 352;    // g_angle2mirror
static const int T_WOFF   = 384;    // g_weightOffset[split][h][w]: x | y << 16
static const int T_BCW1   = 1408;   // getBcwWeight( g_BcwInternBcw[idx], REF_PIC_LIST_1 )

static const int kInvQuantScales[2][6] = { { 40, 45, 51, 57, 64, 72 }, { 57, 64, 72, 80, 90, 102 } };
static const int kIctModes[2][4]       = { { 0, 3, 1, 2 }, { 0, -3, -1, -2 } };
static const int kDeltaQpAct[4]        = { -5, 1, 3, 1 };
static const int kIntraFilter[2][8]    = { { 24, 24, 24, 14, 2, 0, 0, 0 }, { 40, 40, 40, 28, 4, 0, 0, 0 } };
static const int kAngTable[32]         = { 0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 26, 29, 32, 35, 39, 45, 51, 57, 64, 73, 86, 102, 128, 171, 256, 341, 512, 1024 };
static const int kModeShift[6]         = { 0, 6, 10, 12, 14, 15 };

int log2Of( int v ) { return int( firstbithigh( uint( v ) ) ); }   // floor( log2 ), as getLog2

// --------------------------------------------------------------------------------------------------------------------
// The records
// --------------------------------------------------------------------------------------------------------------------

struct Cu
{
  int  index;
  int  ctu;
  int4 blk[2];       // per channel: x, y, width, height (width 0: none)
  int  predMode;
  uint flags;
  int  chType;       // 1: a CU of the chroma tree
  int  treeType;     // 0: TREE_D
  int  isp;
  int  lfnst;
  bool mip;
  bool mipT;
  int  mrl;
  bool act;
  int  bdpcmL;
  int  bdpcmC;
  bool rootCbf;
  int  sbtInfo;
  int  planeCbf;
  int  intraDir[2];
  int  qp;
  int  firstTu;      // global TU index
  int  numTus;
  int  dmvrOff;      // global index where DMVR's refinements of the CU's subblocks start (CodingUnit::mvdL0SubPuOff)
  bool leftAvail;    // whether the left and above neighbours exist, as the parse found them (CodingUnit::left, above)
  bool aboveAvail;
  bool smvd;
  int  geoSplit;
};

int ctuOfLuma( int x, int y ) { return ( y >> pc.log2Ctu ) * pc.widthInCtus + ( x >> pc.log2Ctu ); }

Cu loadCu( int index )
{
  uint* r = pc.cus + index * 32;
  Cu    cu;
  cu.index         = index;
  cu.predMode      = int( ( r[1] >> 16 ) & 3 );
  cu.flags         = r[1] >> 18;
  cu.qp            = int( int8_t( ( r[3] >> 16 ) & 0xff ) );
  cu.geoSplit      = int( ( r[2] >> 9 ) & 63 );
  const uint w17   = r[17];
  cu.chType        = int( w17 & 1 );
  cu.treeType      = int( ( w17 >> 1 ) & 3 );
  cu.isp           = int( ( w17 >> 5 ) & 3 );
  cu.lfnst         = int( ( w17 >> 7 ) & 3 );
  cu.mip           = ( w17 & ( 1u << 9 ) ) != 0;
  cu.mipT          = ( w17 & ( 1u << 10 ) ) != 0;
  cu.mrl           = int( ( w17 >> 11 ) & 3 );
  cu.act           = ( w17 & ( 1u << 13 ) ) != 0;
  cu.bdpcmL        = int( ( w17 >> 14 ) & 3 );
  cu.bdpcmC        = int( ( w17 >> 16 ) & 3 );
  cu.rootCbf       = ( w17 & ( 1u << 18 ) ) != 0;
  cu.sbtInfo       = int( ( w17 >> 19 ) & 0xff );
  cu.planeCbf      = int( ( w17 >> 27 ) & 7 );
  cu.intraDir[0]   = int( r[18] & 0xff );
  cu.intraDir[1]   = int( ( r[18] >> 8 ) & 0xff );
  const uint valid = r[20];
  const bool luma  = ( valid & ( 1u << 17 ) ) != 0;
  cu.blk[0]        = luma ? int4( int( r[0] & 0xffff ), int( r[0] >> 16 ), int( r[1] & 0xff ), int( ( r[1] >> 8 ) & 0xff ) ) : int4( 0 );
  cu.blk[1]        = ( valid & ( 1u << 16 ) ) != 0 ? int4( int( r[19] & 0xffff ), int( r[19] >> 16 ), int( r[20] & 0xff ), int( ( r[20] >> 8 ) & 0xff ) )
                                                   : int4( 0 );
  const int lx     = luma ? int( r[0] & 0xffff ) : int( r[19] & 0xffff ) << pc.csx;
  const int ly     = luma ? int( r[0] >> 16 ) : int( r[19] >> 16 ) << pc.csy;
  cu.ctu           = ctuOfLuma( lx, ly );
  cu.firstTu       = int( pc.ctus[cu.ctu * 4 + 3] ) + int( r[16] & 0xffff );
  cu.numTus        = int( r[16] >> 16 );
  const uint w21   = r[21];
  cu.dmvrOff       = cu.ctu * pc.num8x8 + int( w21 & 0xffff );
  cu.leftAvail     = ( w21 & ( 1u << 16 ) ) != 0;
  cu.aboveAvail    = ( w21 & ( 1u << 17 ) ) != 0;
  cu.smvd          = ( w21 & ( 1u << 18 ) ) != 0;
  return cu;
}

bool isIntra( Cu cu ) { return cu.predMode == MODE_INTRA; }
bool isSepTree( Cu cu ) { return cu.treeType != 0; }
bool valid( int4 b ) { return b.z != 0 && b.w != 0; }

// The motion the GPU derived for a CU (mider.slang's CuMotion)
struct Motion
{
  int2 mv[2][3];
  int  refIdx[2];
  int  interDir;
  int  bcw;
  bool hpel;
  int  mrgType;
  bool affine6;
  int  geo[2];   // GPM: each partition's inter direction << 4 | reference index
};

Motion loadMotion( int index )
{
  int*   r = pc.cuMotion + index * 16;
  Motion m;
  for( int l = 0; l < 2; l++ )
  {
    for( int k = 0; k < 3; k++ )
    {
      m.mv[l][k] = int2( r[6 * l + 2 * k], r[6 * l + 2 * k + 1] );
    }
  }
  const uint w = uint( r[12] );
  m.refIdx[0]  = int( int8_t( w & 0xff ) );
  m.refIdx[1]  = int( int8_t( ( w >> 8 ) & 0xff ) );
  m.interDir   = int( ( w >> 16 ) & 3 );
  m.bcw        = int( ( w >> 18 ) & 7 );
  m.hpel       = ( w & ( 1u << 21 ) ) != 0;
  m.mrgType    = int( ( w >> 22 ) & 3 );
  m.affine6    = ( w & ( 1u << 29 ) ) != 0;
  const uint g = uint( r[13] );
  m.geo[0]     = int( g & 0xff );
  m.geo[1]     = int( ( g >> 8 ) & 0xff );
  return m;
}

struct Tu
{
  int  index;   // global TU index
  int4 blk[2];  // per channel (Cr's block equals Cb's)
  int  cbf;
  int  jointCbCr;
  int  mts[3];
  int  chType;
  int  chromaQp[2];
  int  maxX[3];
  int  maxY[3];
  uint levelsAt[3];    // with the GPU's parse: where each component's levels are in the picture's levels
  uint levelsRow[3];   // and the width of their rows (0: the CPU's parse, whose levels are in the planes)
};

Tu loadTu( int index )
{
  uint* r = pc.tus + index * 16;
  Tu    tu;
  tu.index       = index;
  tu.blk[0]      = ( r[1] & ( 1u << 16 ) ) != 0 ? int4( int( r[0] & 0xffff ), int( r[0] >> 16 ), int( r[1] & 0xff ), int( ( r[1] >> 8 ) & 0xff ) ) : int4( 0 );
  tu.blk[1]      = ( r[3] & ( 1u << 16 ) ) != 0 ? int4( int( r[2] & 0xffff ), int( r[2] >> 16 ), int( r[3] & 0xff ), int( ( r[3] >> 8 ) & 0xff ) ) : int4( 0 );
  tu.cbf         = int( r[4] & 7 );
  tu.jointCbCr   = int( ( r[4] >> 3 ) & 3 );
  tu.mts[0]      = int( ( r[4] >> 5 ) & 7 );
  tu.mts[1]      = int( ( r[4] >> 8 ) & 1 );
  tu.mts[2]      = int( ( r[4] >> 9 ) & 1 );
  tu.chType      = int( ( r[4] >> 10 ) & 1 );
  tu.chromaQp[0] = int( int8_t( r[5] & 0xff ) );
  tu.chromaQp[1] = int( int8_t( ( r[5] >> 8 ) & 0xff ) );
  for( int c = 0; c < 3; c++ )
  {
    tu.maxX[c]      = int( ( r[6] >> ( 8 * c ) ) & 0xff );
    tu.maxY[c]      = int( ( r[7] >> ( 8 * c ) ) & 0xff );
    tu.levelsAt[c]  = r[9 + c];
    tu.levelsRow[c] = ( r[12] >> ( 8 * c ) ) & 0xff;
  }
  return tu;
}

int  chOf( int comp ) { return comp == 0 ? 0 : 1; }
bool cbfOf( Tu tu, int comp ) { return ( ( tu.cbf >> comp ) & 1 ) != 0; }

// A channel's subsampling shifts: 0 for luma, the format's for chroma.
int scaleX( int ch ) { return ch == 0 ? 0 : pc.csx; }
int scaleY( int ch ) { return ch == 0 ? 0 : pc.csy; }

// Slice records: mider.slang's slice table and the jobs' JobSlice (GpuFrontEnd.h)
uint* sliceOfCtu( int ctu ) { return pc.slices + pc.widthInCtus * pc.heightInCtus + pc.slices[ctu] * 64; }
uint* jobSliceOfCtu( int ctu ) { return pc.jobSlices + pc.slices[ctu] * 256; }
int   slicePoc( uint* s ) { return int( s[0] ); }
bool  isInterB( uint* s ) { return ( s[1] & ( 1u << 16 ) ) != 0; }
uint  independentIdx( uint* s ) { return s[3]; }
bool  usedAsLongTerm( uint* s, int list, int refIdx ) { return ( s[4] & ( 1u << ( 16 * list + refIdx ) ) ) != 0; }
int   refPoc( uint* s, int list, int refIdx ) { return int( s[16 + 16 * list + refIdx] ); }
uint  ctuTile( int ctu ) { return pc.ctus[ctu * 4 + 2] >> 16; }
int   sliceType( uint* j ) { return int( j[0] & 3 ); }   // B 0, P 1, I 2
bool  sliceLmcs( uint* j ) { return ( j[0] & 4 ) != 0; }
bool  sliceDepQuant( uint* j ) { return ( j[0] & 8 ) != 0; }
bool  sliceScalingList( uint* j ) { return ( j[0] & 16 ) != 0; }
// Per reference: its slot in the reference table, whether it is scaled (RPR) or the picture's size, and its weights
uint  refInfo( uint* j, int list, int refIdx ) { return j[32 + 16 * list + max( refIdx, 0 )]; }
int   refSlot( uint* j, int list, int refIdx ) { return int( refInfo( j, list, refIdx ) & 0xff ); }
bool  refScaled( uint* j, int list, int refIdx ) { return ( refInfo( j, list, refIdx ) & ( 1u << 8 ) ) != 0; }
bool  refSameSize( uint* j, int list, int refIdx ) { return ( refInfo( j, list, refIdx ) & ( 1u << 9 ) ) != 0; }
bool  wpPresent( uint* j, int list, int refIdx, int c ) { return ( refInfo( j, list, refIdx ) & ( 1u << ( 10 + c ) ) ) != 0; }
uint  wpParam( uint* j, int list, int refIdx, int c ) { return j[64 + ( 16 * list + max( refIdx, 0 ) ) * 3 + c]; }

// The CU covering pos (in the channel's samples); -1 outside the picture.
int cuAt( int ch, int2 pos )
{
  const int lx = pos.x << scaleX( ch );
  const int ly = pos.y << scaleY( ch );
  if( pos.x < 0 || pos.y < 0 || lx >= pc.width || ly >= pc.height )
  {
    return -1;
  }
  const int at = ( ly >> 2 ) * pc.widthIn4 + ( lx >> 2 );
  return int( ch == 0 ? pc.cuMapL[at] : pc.cuMapC[at] );
}

// CodingStructure::getCURestricted( pos, cur, chType ): the CU at pos (in the channel's samples) if cur may use it,
// else -1. Usable CUs lie in cur's CTU and precede cur in decoding order, or lie in a CTU to the left of or above
// cur's (also the one above-right, without WPP) in the same slice and tile.
int cuRestricted( int2 pos, Cu cur, int ch )
{
  const int at = cuAt( ch, pos );
  if( at < 0 )
  {
    return -1;
  }
  const int sx    = scaleX( ch );
  const int sy    = scaleY( ch );
  const int ydiff = ( ( pos.y << sy ) >> pc.log2Ctu ) - ( ( cur.blk[ch].y << sy ) >> pc.log2Ctu );
  const int xdiff = ( ( pos.x << sx ) >> pc.log2Ctu ) - ( ( cur.blk[ch].x << sx ) >> pc.log2Ctu );
  if( xdiff == 0 && ydiff == 0 )
  {
    return at <= cur.index ? at : -1;
  }
  const int wpp = ( pc.sps & kSpsWpp ) != 0 ? 1 : 0;
  if( ydiff > 0 || xdiff > 1 - wpp || ( ydiff == 0 && xdiff > 0 ) )
  {
    return -1;
  }
  const int ctu = ctuOfLuma( pos.x << sx, pos.y << sy );
  return independentIdx( sliceOfCtu( ctu ) ) == independentIdx( sliceOfCtu( cur.ctu ) ) && ctuTile( ctu ) == ctuTile( cur.ctu ) ? at : -1;
}

// The CTU of the CU with the given index.
int ctuOfCu( int index )
{
  uint* r = pc.cus + index * 32;
  return ( r[20] & ( 1u << 17 ) ) != 0 ? ctuOfLuma( int( r[0] & 0xffff ), int( r[0] >> 16 ) )
                                       : ctuOfLuma( int( r[19] & 0xffff ) << pc.csx, int( r[19] >> 16 ) << pc.csy );
}

// The CU's TU at pos, as getTU finds it: the first TU whose block (in channel ch) ends past pos, otherwise the last.
int tuOf( int cuIndex, int2 pos, int ch )
{
  uint*     r     = pc.cus + cuIndex * 32;
  const int ctu   = ctuOfCu( cuIndex );
  const int first = int( pc.ctus[ctu * 4 + 3] ) + int( r[16] & 0xffff );
  const int n     = int( r[16] >> 16 );
  for( int i = 0; i < n - 1; i++ )
  {
    const Tu tu = loadTu( first + i );
    if( tu.blk[ch].x + tu.blk[ch].z > pos.x && tu.blk[ch].y + tu.blk[ch].w > pos.y )
    {
      return first + i;
    }
  }
  return first + n - 1;
}

// --------------------------------------------------------------------------------------------------------------------
// Job output: COUNT only counts the jobs; WRITE also writes them at the CU's offsets
// --------------------------------------------------------------------------------------------------------------------

static const int kLists = 5;

struct Out
{
  bool write;
  int  n[5];      // the CU's jobs so far, per list
  int  base[5];   // where the CU's jobs start, per list
};

static const int kTrafoWords = 10;   // a transform job's words (GpuJobs.h's TrafoJob)

void emitTrafo( inout Out o, uint w[12] )
{
  if( o.write && o.base[0] + o.n[0] < pc.capTrafo )
  {
    uint* p = pc.trafo + ( o.base[0] + o.n[0] ) * kTrafoWords;
    for( int i = 0; i < kTrafoWords; i++ )
    {
      p[i] = w[i];
    }
  }
  o.n[0]++;
}

// The thread's CU, when it is an SbTMVP CU: interJobs leaves the jobs of its sub-PUs' motion compensation to
// sbtmvpJobs, with their flags and the CU's BCW index.
static bool g_sb      = false;
static uint g_sbFlags = 0;
static int  g_sbBcw   = 0;

// Writes an inter job at index at of the inter buffer, if there is room.
void writeInter( int at, uint w[12] )
{
  if( at < pc.capInter )
  {
    uint* p = pc.inter + at * 12;
    for( int i = 0; i < 12; i++ )
    {
      p[i] = w[i];
    }
  }
}

// Adds an inter job to its list: BDOF jobs (3), affine pairs (4), or the regular jobs (1).
void emitInter( inout Out o, uint w[12] )
{
  const uint flags = w[1] >> 16;
  const int  list  = ( flags & kInterAffine ) != 0 ? 4 : ( flags & kInterBdof ) != 0 ? 3 : 1;
  if( o.write )
  {
    writeInter( o.base[list] + o.n[list], w );
  }
  o.n[list]++;
}

// Adds the second job of an affine pair (InterJob::setAffine's next), right after the first.
void emitAffineNext( inout Out o, uint w[12] )
{
  if( o.write )
  {
    writeInter( o.base[4] + o.n[4], w );
  }
  o.n[4]++;
}

void emitIntra( inout Out o, uint w[12] )
{
  if( o.write && o.base[2] + o.n[2] < pc.capIntra )
  {
    uint* p = pc.intra + ( o.base[2] + o.n[2] ) * 12;
    for( int i = 0; i < 12; i++ )
    {
      p[i] = w[i];
    }
  }
  o.n[2]++;
}

void clearJob( out uint w[12] )
{
  for( int i = 0; i < 12; i++ )
  {
    w[i] = 0;
  }
}

// TrafoJob::set
void setTrafo( inout uint w[12], int x, int y, int log2W, int log2H, int comp, int typeHor, int typeVer, uint flags, int ict, int lastX, int lastY,
               int scale, int rightShift, int inputMaximum )
{
  w[0] = uint( x ) | uint( y ) << 16;
  w[1] = uint( log2W ) | uint( log2H ) << 4 | uint( comp ) << 8 | uint( typeHor ) << 12 | uint( typeVer ) << 14 | flags << 16 | uint( ict + 4 ) << 24;
  w[2] = uint( lastX ) | uint( lastY ) << 8;
  w[3] = uint( scale );
  w[4] = uint( rightShift );
  w[5] = uint( inputMaximum );
  w[6] = 0;
  w[7] = 0;
}

// IntraJob::set
void setIntra( inout uint w[12], int x, int y, int width, int height, int comp, int kind, int mode, uint flags, int bdpcmDir, int refLine )
{
  clearJob( w );
  w[0] = uint( x ) | uint( y ) << 16;
  w[1] = uint( width ) | uint( height ) << 8 | uint( comp ) << 16 | uint( kind ) << 24;
  w[2] = uint( mode ) | flags << 8 | uint( bdpcmDir ) << 16 | uint( refLine ) << 24;
}

// InterJob::set
void setInter( inout uint w[12], int x, int y, int width, int height, uint flags, int2 mv0, int2 mv1, int cuX, int cuY, int ref0, int ref1,
               int dmvrIndex )
{
  w[0]  = uint( x ) | uint( y ) << 16;
  w[1]  = uint( width ) | uint( height ) << 8 | flags << 16;
  w[2]  = uint( mv0.x );
  w[3]  = uint( mv0.y );
  w[4]  = uint( mv1.x );
  w[5]  = uint( mv1.y );
  w[6]  = uint( cuX ) | uint( cuY ) << 16;
  w[7]  = uint( ref0 ) | uint( ref1 ) << 8;
  w[8]  = uint( dmvrIndex );
  w[9]  = 0;
  w[10] = 0;
  w[11] = 0;
}

// --------------------------------------------------------------------------------------------------------------------
// Intra modes (UnitTools)
// --------------------------------------------------------------------------------------------------------------------

// PU::getCoLocatedLumaPU: the luma CU at the centre of the CU's block (the CU itself in a single tree)
int coLocatedLuma( Cu cu )
{
  if( !isSepTree( cu ) )
  {
    return cu.index;
  }
  const int4 b  = cu.blk[cu.chType];
  const int  sx = scaleX( cu.chType );
  const int  sy = scaleY( cu.chType );
  return cuAt( 0, int2( ( b.x << sx ) + ( ( b.z << sx ) >> 1 ), ( b.y << sy ) + ( ( b.w << sy ) >> 1 ) ) );
}

// PU::getIntraDirLuma of a CU (planar for a MIP CU)
int intraDirLuma( int cuIndex )
{
  uint* r = pc.cus + cuIndex * 32;
  return ( r[17] & ( 1u << 9 ) ) != 0 ? PLANAR_IDX : int( r[18] & 0xff );
}

bool cuMipOf( int cuIndex ) { return ( pc.cus[cuIndex * 32 + 17] & ( 1u << 9 ) ) != 0; }

// PU::getFinalIntraMode
int finalIntraMode( Cu cu, int ch )
{
  int mode = cu.intraDir[ch];
  if( mode == DM_CHROMA_IDX && ch == 1 )
  {
    mode = intraDirLuma( coLocatedLuma( cu ) );
  }
  if( pc.chromaFormat == 2 && ch == 1 && mode < NUM_LUMA_MODE )
  {
    mode = int( pc.tables[T_C422 + mode] );
  }
  return mode;
}

bool isLmcMode( int mode ) { return mode >= LM_CHROMA_IDX && mode <= MDLM_T_IDX; }

// CU::isMIP
bool isMip( Cu cu, int ch )
{
  if( ch == 0 )
  {
    return cu.mip;
  }
  return !isSepTree( cu ) && pc.chromaFormat == 3 && cuMipOf( coLocatedLuma( cu ) ) && cu.intraDir[1] == DM_CHROMA_IDX;
}

// PU::getWideAngIntraMode (the variant LFNST uses)
int wideAngIntraMode( Cu cu, Tu tu, int mode, int comp )
{
  if( mode < 2 )
  {
    return mode;
  }
  const int4 area  = cu.isp != 0 && comp == 0 ? cu.blk[0] : tu.blk[chOf( comp )];
  const int  delta = abs( log2Of( area.z ) - log2Of( area.w ) );
  int        pred  = mode;
  if( area.z > area.w && mode < 2 + kModeShift[delta] )
  {
    pred += VDIA_IDX - 1;
  }
  else if( area.w > area.z && pred > VDIA_IDX - kModeShift[delta] )
  {
    pred -= VDIA_IDX + 1;
  }
  return pred;
}

// IntraPrediction::getWideAngle (the variant the prediction uses)
int wideAngle( int width, int height, int mode )
{
  if( mode > DC_IDX && mode <= VDIA_IDX )
  {
    const int delta = abs( log2Of( width ) - log2Of( height ) );
    if( width > height && mode < 2 + kModeShift[delta] )
    {
      mode += VDIA_IDX - 1;
    }
    else if( height > width && mode > VDIA_IDX - kModeShift[delta] )
    {
      mode -= VDIA_IDX - 1;
    }
  }
  return mode;
}

// IntraPrediction::useFilteredIntraRefSamples, for a block of width x height
bool useFilteredRef( Cu cu, int comp, int width, int height )
{
  const int ch = chOf( comp );
  if( cu.mrl != 0 || cu.bdpcmL != 0 )
  {
    return false;
  }
  const int dirMode = finalIntraMode( cu, ch );
  if( dirMode == DC_IDX )
  {
    return false;
  }
  if( dirMode == PLANAR_IDX )
  {
    return width * height > 32;
  }
  const int predMode = wideAngle( width, height, dirMode );
  const int diff     = min( abs( predMode - HOR_IDX ), abs( predMode - VER_IDX ) );
  const int log2Size = ( log2Of( width ) + log2Of( height ) ) >> 1;
  const int angMode  = predMode >= DIA_IDX ? predMode - VER_IDX : -( predMode - HOR_IDX );
  return diff > kIntraFilter[ch][log2Size] && ( kAngTable[abs( angMode )] & 0x1f ) == 0;
}

// --------------------------------------------------------------------------------------------------------------------
// The residual jobs (DecCu::xGpuTrafoJobs)
// --------------------------------------------------------------------------------------------------------------------

// TrQuant::getTrTypes
void trTypes( Cu cu, Tu tu, int comp, out int typeHor, out int typeVer )
{
  typeHor             = DCT2;
  typeVer             = DCT2;
  const bool intra    = isIntra( cu );
  const bool luma     = comp == 0;
  const bool implicit = intra && luma && ( pc.sps & kSpsImplicitMts ) != 0 && cu.lfnst == 0 && !cu.mip;
  const bool isp      = intra && luma && cu.isp != 0;
  if( isp && cu.lfnst != 0 )
  {
    return;
  }
  const int lw = tu.blk[0].z;
  const int lh = tu.blk[0].w;
  if( ( pc.sps & kSpsMts ) == 0 )
  {
    return;
  }
  if( implicit || isp )
  {
    if( lw >= 4 && lw <= 16 )
    {
      typeHor = DST7;
    }
    if( lh >= 4 && lh <= 16 )
    {
      typeVer = DST7;
    }
    return;
  }
  const bool inter    = cu.predMode == MODE_INTER && luma;
  const bool explicit = intra ? ( pc.sps & kSpsIntraMts ) != 0 && luma : ( pc.sps & kSpsInterMts ) != 0 && inter;
  if( inter && cu.sbtInfo != 0 )
  {
    const int idx = cu.sbtInfo & 0xf;
    const int pos = ( cu.sbtInfo >> 4 ) & 3;
    if( idx == 1 || idx == 3 )   // SBT_VER_HALF, SBT_VER_QUAD
    {
      if( lh > 32 )
      {
        typeHor = DCT2;
        typeVer = DCT2;
      }
      else if( pos == 0 )
      {
        typeHor = DCT8;
        typeVer = DST7;
      }
      else
      {
        typeHor = DST7;
        typeVer = DST7;
      }
    }
    else
    {
      if( lw > 32 )
      {
        typeHor = DCT2;
        typeVer = DCT2;
      }
      else if( pos == 0 )
      {
        typeHor = DST7;
        typeVer = DCT8;
      }
      else
      {
        typeHor = DST7;
        typeVer = DST7;
      }
    }
    return;
  }
  if( explicit && tu.mts[comp] > 1 )
  {
    typeHor = ( ( tu.mts[comp] - 2 ) & 1 ) != 0 ? DCT8 : DST7;
    typeVer = ( ( tu.mts[comp] - 2 ) >> 1 ) != 0 ? DCT8 : DST7;
  }
}

void trafoJobs( Cu cu, inout Out o )
{
  uint*      jslice   = jobSliceOfCtu( cu.ctu );
  const bool sign     = ( pc.ph & kPhJointSign ) != 0;
  const int  numComp  = pc.chromaFormat == 0 ? 1 : 3;
  for( int t = 0; t < cu.numTus; t++ )
  {
    const Tu tu = loadTu( cu.firstTu + t );
    for( int c = 0; c < numComp; c++ )
    {
      if( !valid( tu.blk[chOf( c )] ) )
      {
        continue;
      }
      int comp = c;
      int ict  = 0;
      if( tu.jointCbCr != 0 && c > 0 )
      {
        if( c != 1 )
        {
          continue;   // handled by Cb's job
        }
        comp = ( tu.jointCbCr >> 1 ) != 0 ? 1 : 2;
        ict  = kIctModes[sign ? 1 : 0][tu.jointCbCr];
      }
      else if( !cbfOf( tu, c ) )
      {
        continue;
      }
      const int4 block = tu.blk[chOf( comp )];
      const bool ts    = tu.mts[comp] == 1;   // MTS_SKIP

      // QpParam: the CU's QP (luma) or the parse's chroma QPs (QpParam without ACT's offset), then ACT's offset
      int baseQp = comp == 0 ? cu.qp + pc.qpBdOffset : tu.chromaQp[comp - 1];
      if( cu.act )
      {
        const bool jointQp = comp != 0 && kIctModes[0][tu.jointCbCr] == 2;
        baseQp             = clip3( 0, 63 + pc.qpBdOffset, baseQp + kDeltaQpAct[jointQp ? 3 : comp] );
      }
      const int qp = ts ? max( baseQp, pc.tsMinQp ) : baseQp;

      // Whether the explicit scaling list applies, as Quant::dequant decides it
      const bool lfnstApplied = cu.lfnst > 0 && ( isSepTree( cu ) || comp == 0 );
      const bool scalingList  = sliceScalingList( jslice ) && !ts && !( lfnstApplied && ( pc.sps & kSpsNoSlLfnst ) != 0 )
                               && !( ( pc.sps & kSpsNoSlAct ) != 0 && ( ( pc.sps & kSpsSlActSpace ) != 0 ) == cu.act );

      // Quant::dequantParam
      const int  log2W     = log2Of( block.z );
      const int  log2H     = log2Of( block.w );
      const bool bdpcm     = ( cu.bdpcmL != 0 && comp == 0 ) || ( cu.bdpcmC != 0 && comp != 0 );
      const int  maxX      = bdpcm ? block.z - 1 : tu.maxX[comp];
      const int  maxY      = bdpcm ? block.w - 1 : tu.maxY[comp];
      const bool sqrtAdj   = !ts && ( ( log2W + log2H ) & 1 ) == 1;
      const int  trShift   = 15 - pc.bitDepth - ( ( log2W + log2H ) >> 1 ) + ( sqrtAdj ? -1 : 0 );
      const bool depQuant  = sliceDepQuant( jslice ) && !ts;
      const int  qpPer     = depQuant ? ( qp + 1 ) / 6 : qp / 6;
      const int  qpRem     = depQuant ? qp + 1 - 6 * qpPer : qp - 6 * qpPer;
      const int  rightShift = 6 + ( depQuant ? 1 : 0 ) - ( ( ts ? 0 : trShift ) + qpPer ) + ( scalingList ? 4 : 0 );
      const int  scale     = kInvQuantScales[sqrtAdj ? 1 : 0][qpRem];
      const int  target    = min( 15 + 1, 32 + rightShift - 7 );
      const int  inputMax  = ( 1 << ( target - 1 ) ) - 1;

      int  typeHor = DCT2, typeVer = DCT2;
      uint flags   = 0;
      if( ts )
      {
        flags |= kTrafoTransformSkip;
      }
      else
      {
        trTypes( cu, tu, comp, typeHor, typeVer );
      }
      if( bdpcm )
      {
        flags |= ( comp == 0 ? cu.bdpcmL : cu.bdpcmC ) == 1 ? kTrafoBdpcmHor : kTrafoBdpcmVer;
      }

      uint w[12];
      clearJob( w );
      setTrafo( w, block.x, block.y, log2W, log2H, comp, typeHor, typeVer, flags, ict, maxX, maxY, scale, rightShift, inputMax );
      w[8] = tu.levelsAt[comp];   // the levels' place (the GPU's parse)
      w[9] = tu.levelsRow[comp];
      // LFNST: the set and the orientation follow from the block's intra mode (TrQuant::xInvLfnst)
      if( cu.lfnst != 0 && ( pc.sps & kSpsLfnst ) != 0 && !ts && ( isSepTree( cu ) || comp == 0 ) )
      {
        const int ch   = chOf( comp );
        int       mode = isMip( cu, ch ) ? PLANAR_IDX : isLmcMode( cu.intraDir[ch] ) ? intraDirLuma( coLocatedLuma( cu ) ) : finalIntraMode( cu, ch );
        const int wide = wideAngIntraMode( cu, tu, mode, comp );
        mode           = wide < 0 ? wide + 14 + NUM_LUMA_MODE : wide >= NUM_LUMA_MODE ? wide + 14 : wide;
        const bool tr  = ( mode >= NUM_LUMA_MODE && mode >= NUM_LUMA_MODE + 14 ) || ( mode < NUM_LUMA_MODE && mode > DIA_IDX );
        w[6]           = uint( cu.lfnst ) | pc.tables[T_LFNST + mode] << 2 | uint( tr ) << 4 | uint( block.z >= 8 && block.w >= 8 ) << 5
               | uint( ( block.z == 4 && block.w == 4 ) || ( block.z == 8 && block.w == 8 ) ) << 6;
      }
      if( scalingList )
      {
        const int list = ( cu.predMode == MODE_INTRA ? 0 : 3 ) + comp;
        w[7]           = pc.scaling[( list * 7 + log2W ) * 7 + log2H];
      }
      emitTrafo( o, w );
    }
  }
}

// --------------------------------------------------------------------------------------------------------------------
// The intra neighbourhood (IntraPrediction)
// --------------------------------------------------------------------------------------------------------------------

// isAboveAvailable: counts the units in the row above posLT that are available, stopping at the first unit whose CU
// cur may not use or whose TU is not decoded yet.
int aboveAvailable( Cu cu, int tuIndex, int ch, int2 posLT, int numUnits, int unitWidth )
{
  const int maxDx  = numUnits * unitWidth;
  int2      refPos = int2( posLT.x, posLT.y - 1 );
  int       dx     = 0;
  while( dx < maxDx )
  {
    const int a = cuRestricted( refPos, cu, ch );
    if( a < 0 )
    {
      break;
    }
    const int t = tuOf( a, refPos, ch );
    if( ctuOfCu( a ) == cu.ctu && t >= tuIndex )
    {
      break;
    }
    const int4 b    = loadTu( t ).blk[ch];
    const int  diff = b.z - refPos.x + b.x;
    dx += diff;
    refPos.x += diff;
  }
  return min( dx / unitWidth, numUnits );
}

// isLeftAvailable: the same for the column left of posLT.
int leftAvailable( Cu cu, int tuIndex, int ch, int2 posLT, int numUnits, int unitHeight )
{
  const int maxDy  = numUnits * unitHeight;
  int2      refPos = int2( posLT.x - 1, posLT.y );
  int       dy     = 0;
  while( dy < maxDy )
  {
    const int a = cuRestricted( refPos, cu, ch );
    if( a < 0 )
    {
      break;
    }
    const int t = tuOf( a, refPos, ch );
    if( ctuOfCu( a ) == cu.ctu && t >= tuIndex )
    {
      break;
    }
    const int4 b    = loadTu( t ).blk[ch];
    const int  diff = b.w - refPos.y + b.y;
    dy += diff;
    refPos.y += diff;
  }
  return min( dy / unitHeight, numUnits );
}

// xDeriveNeighborSize (getNeighborUnits) for the block area of component comp of TU tuIndex. A CU's components after
// the first one of its channel reuse the units last derived for that CU (vvdec's m_lastCUidx).
struct Units
{
  int  cu;   // the CU the units were derived for
  int3 u;    // top-left, above, left
};

int3 neighborUnits( Cu cu, int tuIndex, int4 area, int comp, inout Units cache )
{
  const int firstComp = cu.chType == 0 ? 0 : 1;
  if( cache.cu == cu.index && comp != firstComp )
  {
    return cache.u;
  }
  const int ch               = chOf( comp );
  const int csx              = scaleX( ch );
  const int csy              = scaleY( ch );
  const int unitWidth        = 4 >> csx;
  const int unitHeight       = 4 >> csy;
  const int totalAboveUnits  = ( 2 * area.z + unitWidth - 1 ) / unitWidth;
  const int totalLeftUnits   = ( 2 * area.w + unitHeight - 1 ) / unitHeight;
  const int numAboveUnits    = area.z / unitWidth;
  const int numLeftUnits     = area.w / unitHeight;
  const int ctuMask          = ( 1 << pc.log2Ctu ) - 1;
  const bool sameCtu         = ( area.x & ( ctuMask >> csx ) ) != 0 && ( area.y & ( ctuMask >> csy ) ) != 0;
  int3       u;
  u.x = sameCtu ? 1 : ( cuRestricted( int2( area.x - 1, area.y - 1 ), cu, ch ) >= 0 ? 1 : 0 );
  if( cu.aboveAvail || area.y > cu.blk[ch].y )
  {
    u.y = numAboveUnits + aboveAvailable( cu, tuIndex, ch, int2( area.x + area.z, area.y ), totalAboveUnits - numAboveUnits, unitWidth );
  }
  else
  {
    u.y = 0;
  }
  if( cu.leftAvail || area.x > cu.blk[ch].x )
  {
    u.z = numLeftUnits + leftAvailable( cu, tuIndex, ch, int2( area.x, area.y + area.w ), totalLeftUnits - numLeftUnits, unitHeight );
  }
  else
  {
    u.z = 0;
  }
  cache.cu = cu.index;
  cache.u  = u;
  return u;
}

// IntraPrediction::getLmAvailability: which of CCLM's templates are available
void cclmAvailability( Cu cu, int4 area, inout uint w[12] )
{
  const int  mode    = cu.intraDir[1];
  const int  cWidth  = area.z;
  const int  cHeight = area.w;
  bool       lumaAbove, lumaLeft, above = false, left = false;
  int        actualTop = 0, actualLeft = 0;
  {
    // xGetLumaRecPixels
    const bool chromaTree       = cu.chType == 1;
    const int  aw               = chromaTree ? area.z : area.z << pc.csx;
    const int  ah               = chromaTree ? area.w : area.w << pc.csy;
    const int  unitWidth        = chromaTree ? 4 >> pc.csx : 4;
    const int  unitHeight       = chromaTree ? 4 >> pc.csy : 4;
    const int  tuWidthInUnits   = aw / unitWidth;
    const int  tuHeightInUnits  = ah / unitHeight;
    const int  chromaUnitWidth  = 4 >> pc.csx;
    const int  chromaUnitHeight = 4 >> pc.csy;
    const int  totalAboveUnits  = mode == MDLM_T_IDX ? ( 2 * cWidth + chromaUnitWidth - 1 ) / chromaUnitWidth : tuWidthInUnits;
    const int  totalLeftUnits   = mode == MDLM_L_IDX ? ( 2 * cHeight + chromaUnitHeight - 1 ) / chromaUnitHeight : tuHeightInUnits;
    const int  leftUnits        = cu.leftAvail || area.x > cu.blk[1].x ? totalLeftUnits : 0;
    const int  aboveUnits       = cu.aboveAvail || area.y > cu.blk[1].y ? totalAboveUnits : 0;
    lumaLeft                    = leftUnits >= tuHeightInUnits;
    lumaAbove                   = aboveUnits >= tuWidthInUnits;
  }
  {
    // xGetLMParameters (the unit height follows the horizontal subsampling, as it does there)
    const int unitWidth       = 4 >> pc.csx;
    const int unitHeight      = 4 >> pc.csx;
    const int tuWidthInUnits  = cWidth / unitWidth;
    const int tuHeightInUnits = cHeight / unitHeight;
    const int totalAboveUnits = ( 2 * cWidth + unitWidth - 1 ) / unitWidth;
    const int totalLeftUnits  = ( 2 * cHeight + unitHeight - 1 ) / unitHeight;
    int       aboveRightUnits = totalAboveUnits - tuWidthInUnits;
    int       leftBelowUnits  = totalLeftUnits - tuHeightInUnits;
    const int tuIndex         = tuOf( cu.index, area.xy, 1 );
    if( mode == MDLM_T_IDX )
    {
      int avai = 0;
      if( cu.aboveAvail || area.y > cu.blk[1].y )
      {
        avai            = tuWidthInUnits;
        aboveRightUnits = aboveRightUnits > cHeight / unitWidth ? cHeight / unitWidth : aboveRightUnits;
        avai += aboveAvailable( cu, tuIndex, 1, int2( area.x + cWidth, area.y ), aboveRightUnits, unitWidth );
      }
      above     = avai >= tuWidthInUnits;
      actualTop = unitWidth * avai;
    }
    else if( mode == MDLM_L_IDX )
    {
      int avai = 0;
      if( cu.leftAvail || area.x > cu.blk[1].x )
      {
        avai           = tuHeightInUnits;
        leftBelowUnits = leftBelowUnits > cWidth / unitHeight ? cWidth / unitHeight : leftBelowUnits;
        avai += leftAvailable( cu, tuIndex, 1, int2( area.x, area.y + cHeight ), leftBelowUnits, unitHeight );
      }
      left       = avai >= tuHeightInUnits;
      actualLeft = unitHeight * avai;
    }
    else if( mode == LM_CHROMA_IDX )
    {
      above      = cu.aboveAvail || area.y > cu.blk[1].y;
      left       = cu.leftAvail || area.x > cu.blk[1].x;
      actualTop  = cWidth;
      actualLeft = cHeight;
    }
  }
  w[4] = uint( lumaAbove ) << 16 | uint( lumaLeft ) << 17 | uint( above ) << 18 | uint( left ) << 19;
  w[5] = uint( actualTop ) | uint( actualLeft ) << 16;
}

// gpuLmcsChroma: LMCS's chroma scale comes from the luma left of and above the CU at the top-left of the VPDU that
// holds lumaPos.
void setLmcsScale( int2 lumaPos, inout uint w[12] )
{
  const int  ctuSize = 1 << pc.log2Ctu;
  const int  count   = min( 64, ctuSize );
  const int  mask    = ctuSize == 128 ? ~63 : ~( ctuSize - 1 );
  const int2 vpdu    = int2( lumaPos.x & mask, lumaPos.y & mask );
  const Cu   first   = loadCu( cuAt( 0, vpdu ) );
  const int2 at      = first.blk[0].xy;
  const bool above   = cuRestricted( int2( at.x, at.y - 1 ), first, 0 ) >= 0;
  const bool left    = cuRestricted( int2( at.x - 1, at.y ), first, 0 ) >= 0;
  w[8]               = uint( at.x ) | uint( at.y ) << 16;
  w[9]               = 1u | uint( left ) << 1 | uint( above ) << 2 | uint( log2Of( count ) ) << 4;
}

// --------------------------------------------------------------------------------------------------------------------
// The intra jobs (DecCu::xGpuIntraJobs, xGpuIbcJobs)
// --------------------------------------------------------------------------------------------------------------------

void intraJobs( Cu cu, inout Out o, inout Units cache )
{
  uint*     jslice  = jobSliceOfCtu( cu.ctu );
  const int numComp = pc.chromaFormat == 0 ? 1 : 3;
  for( int t = 0; t < cu.numTus; t++ )
  {
    const Tu tu = loadTu( cu.firstTu + t );
    for( int comp = 0; comp < numComp; comp++ )
    {
      const int  ch   = chOf( comp );
      const int4 area = tu.blk[ch];
      if( !valid( area ) )
      {
        continue;
      }
      const int mode  = finalIntraMode( cu, ch );
      const int bdpcm = comp == 0 ? cu.bdpcmL : cu.bdpcmC;
      int       kind  = kIntraAngular;
      uint      flags = 0;
      int       jobMode = mode;
      if( bdpcm != 0 )
      {
        kind = kIntraBdpcm;
      }
      else if( isMip( cu, ch ) )
      {
        // MIP: the luma CU's mode, or for 4:4:4 chroma in DM mode the co-located luma CU's
        kind          = kIntraMip;
        const int lum = comp == 0 ? cu.index : coLocatedLuma( cu );
        jobMode       = int( pc.cus[lum * 32 + 18] & 0xff );
        flags |= ( pc.cus[lum * 32 + 17] & ( 1u << 10 ) ) != 0 ? kIntraMipTransposed : 0;
      }
      else if( comp != 0 && isLmcMode( mode ) )
      {
        kind = kIntraCclm;
      }
      const bool isp = comp == 0 && cu.isp != 0;
      const int3 units = isp ? neighborUnits( cu, cu.firstTu, cu.blk[0], comp, cache ) : neighborUnits( cu, tu.index, area, comp, cache );
      if( cbfOf( tu, comp ) || ( comp != 0 && tu.jointCbCr != 0 ) )
      {
        flags |= kIntraResidual;
      }
      if( kind == kIntraAngular && comp == 0 && !isp && useFilteredRef( cu, comp, area.z, area.w ) )
      {
        flags |= kIntraFiltered;
      }
      const int refLine = comp == 0 ? cu.mrl : 0;
      uint      w[12];
      setIntra( w, area.x, area.y, area.z, area.w, comp, kind, kind == kIntraAngular || kind == kIntraMip ? jobMode : 0,
                flags | ( isp ? kIntraIsp : 0 ), bdpcm, refLine );
      w[3] = uint( units.x ) | uint( units.y ) << 8 | uint( units.z ) << 16;
      if( isp )
      {
        // The predicted block: a vertical split's sub-partitions 1 or 2 samples wide share one block 4 samples wide.
        const int4 cuArea = cu.blk[0];
        int4       pred   = area;
        if( cu.isp == 2 && ( ( cuArea.z == 8 && cuArea.w > 4 ) || cuArea.z == 4 ) )
        {
          pred.x = cuArea.x + ( ( area.x - cuArea.x ) & ~3 );
          pred.z = 4;
        }
        const bool leftAvail  = cuRestricted( int2( pred.x - 1, pred.y ), cu, 0 ) >= 0;
        const bool aboveAvail = cuRestricted( int2( pred.x, pred.y - 1 ), cu, 0 ) >= 0;
        w[4]                  = uint( cuArea.x ) | uint( cuArea.y ) << 16;
        w[5]                  = uint( cuArea.z ) | uint( cuArea.w ) << 8 | uint( cu.isp ) << 16 | uint( leftAvail ) << 18 | uint( aboveAvail ) << 19;
      }
      // LMCS: the chroma residual is scaled according to the luma around the VPDU of the block's luma
      if( comp != 0 && sliceLmcs( jslice ) && ( pc.ph & kPhLmcsChroma ) != 0 && area.z * area.w > 4 && ( cbfOf( tu, comp ) || tu.jointCbCr != 0 ) )
      {
        const int2 lumaPos = valid( tu.blk[0] ) ? tu.blk[0].xy : int2( tu.blk[tu.chType].x << scaleX( tu.chType ), tu.blk[tu.chType].y << scaleY( tu.chType ) );
        setLmcsScale( lumaPos, w );
      }
      if( kind == kIntraCclm )
      {
        cclmAvailability( cu, area, w );
      }
      emitIntra( o, w );
    }
  }
}

void ibcJobs( Cu cu, Motion m, inout Out o )
{
  uint*      jslice  = jobSliceOfCtu( cu.ctu );
  const int  numComp = pc.chromaFormat == 0 ? 1 : 3;
  // The block vector in whole samples (Mv::changePrecision from the internal precision)
  int2 bv;
  bv.x = m.mv[0][0].x >= 0 ? ( m.mv[0][0].x + 7 ) >> 4 : ( m.mv[0][0].x + 8 ) >> 4;
  bv.y = m.mv[0][0].y >= 0 ? ( m.mv[0][0].y + 7 ) >> 4 : ( m.mv[0][0].y + 8 ) >> 4;
  for( int t = 0; t < cu.numTus; t++ )
  {
    const Tu tu = loadTu( cu.firstTu + t );
    for( int comp = 0; comp < numComp; comp++ )
    {
      const int  ch   = chOf( comp );
      const int4 area = tu.blk[ch];
      // Only the CU's own components (a skipped CU's TUs also cover the chroma of a luma-tree CU)
      if( !valid( area ) || !valid( cu.blk[ch] ) )
      {
        continue;
      }
      const int bvx = comp == 0 ? bv.x : bv.x >> scaleX( 1 );
      const int bvy = comp == 0 ? bv.y : bv.y >> scaleY( 1 );
      uint      flags = 0;
      if( cbfOf( tu, comp ) || ( comp != 0 && tu.jointCbCr != 0 ) )
      {
        flags |= kIntraResidual;
      }
      uint w[12];
      setIntra( w, area.x, area.y, area.z, area.w, comp, kIntraIbc, 0, flags, 0, 0 );
      w[6] = uint( uint16_t( int16_t( bvx ) ) ) | uint( uint16_t( int16_t( bvy ) ) ) << 16;
      if( comp != 0 && cu.rootCbf && ( pc.ph & kPhLmcs ) != 0 && ( pc.ph & kPhLmcsChroma ) != 0 && sliceLmcs( jslice ) && area.z * area.w > 4
          && ( cbfOf( tu, comp ) || tu.jointCbCr != 0 ) )
      {
        setLmcsScale( tu.blk[0].xy, w );
      }
      emitIntra( o, w );
    }
  }
}

// --------------------------------------------------------------------------------------------------------------------
// The inter jobs (DecCu::xGpuInterJobs, xGpuCiipJobs)
// --------------------------------------------------------------------------------------------------------------------

// PU::isBiPredFromDifferentDirEqDistPoc
bool biDiffDirEqDist( uint* slice, Motion m )
{
  if( m.refIdx[0] < 0 || m.refIdx[1] < 0 )
  {
    return false;
  }
  if( usedAsLongTerm( slice, 0, m.refIdx[0] ) || usedAsLongTerm( slice, 1, m.refIdx[1] ) )
  {
    return false;
  }
  const int poc = slicePoc( slice );
  return poc - refPoc( slice, 0, m.refIdx[0] ) == refPoc( slice, 1, m.refIdx[1] ) - poc;
}

bool anyWp( uint* jslice, int refIdx0, int refIdx1 )
{
  bool any = false;
  for( int c = 0; c < 3; c++ )
  {
    any = any || wpPresent( jslice, 0, refIdx0, c ) || wpPresent( jslice, 1, refIdx1, c );
  }
  return any;
}

// PU::checkDMVRCondition
bool dmvrCondition( Cu cu, Motion m, uint* slice, uint* jslice )
{
  if( ( pc.sps & kSpsDmvr ) == 0 || ( pc.ph & kPhNoDmvr ) != 0 )
  {
    return false;
  }
  const int4 b = cu.blk[0];
  return ( cu.flags & kMerge ) != 0 && m.mrgType == MRG_TYPE_DEFAULT_N && ( cu.flags & kCiip ) == 0 && ( cu.flags & kAffine ) == 0
         && ( cu.flags & kMmvd ) == 0 && biDiffDirEqDist( slice, m ) && b.w >= 8 && b.z >= 8 && b.z * b.w >= 128 && m.bcw == 0
         && !anyWp( jslice, m.refIdx[0], m.refIdx[1] ) && ( m.refIdx[0] < 0 || refSameSize( jslice, 0, m.refIdx[0] ) )
         && ( m.refIdx[1] < 0 || refSameSize( jslice, 1, m.refIdx[1] ) );
}

// gpuBdof: the bioApplied condition of motionCompensation
bool bdofCondition( Cu cu, Motion m, uint* slice, uint* jslice )
{
  if( ( pc.sps & kSpsBdof ) == 0 || ( pc.ph & kPhNoBdof ) != 0 || ( cu.flags & kAffine ) != 0 || ( cu.flags & kCiip ) != 0 || cu.smvd
      || ( ( pc.sps & kSpsBcw ) != 0 && m.bcw != 0 ) || m.refIdx[0] < 0 || m.refIdx[1] < 0 )
  {
    return false;
  }
  const int  st = sliceType( jslice );
  const bool wp = anyWp( jslice, m.refIdx[0], m.refIdx[1] ) && st == 0;
  const int4 b  = cu.blk[0];
  if( wp || ( ( pc.ph & kPpsWp ) != 0 && st == 1 ) || !biDiffDirEqDist( slice, m ) || b.w < 8 || b.z < 8 || b.z * b.w < 128 )
  {
    return false;
  }
  return !refScaled( jslice, 0, m.refIdx[0] ) && !refScaled( jslice, 1, m.refIdx[1] );
}

// gpuWp: sets the job's weighted-prediction parameters (flags holds the job's lists and its DMVR and BDOF bits; bcw is
// the CU's BCW index). GPM's jobs, which gpuWp leaves without, do not come here.
void wpJob( int bcw, uint* jslice, int refIdx0, int refIdx1, uint flags, inout uint w[12] )
{
  const int st = sliceType( jslice );
  if( ( flags & ( kInterDmvr | kInterBdof ) ) != 0 || !( ( ( pc.ph & kPpsWpBi ) != 0 && st == 0 && bcw == 0 ) || ( ( pc.ph & kPpsWp ) != 0 && st == 1 ) ) )
  {
    return;
  }
  int       weight[2][3] = { { 0, 0, 0 }, { 0, 0, 0 } };
  int       offset[2][3] = { { 0, 0, 0 }, { 0, 0, 0 } };
  int       log2Denom[3] = { 0, 0, 0 };
  int       n            = 0;
  const int numComp      = pc.chromaFormat == 0 ? 1 : 3;
  for( int list = 0; list < 2; list++ )
  {
    if( ( flags & ( list == 0 ? kInterL0 : kInterL1 ) ) == 0 )
    {
      continue;
    }
    const int refIdx = list == 0 ? refIdx0 : refIdx1;
    for( int c = 0; c < numComp; c++ )
    {
      const uint p = wpParam( jslice, list, refIdx, c );
      weight[n][c] = int( int16_t( p & 0xffff ) );
      offset[n][c] = int( int8_t( ( p >> 16 ) & 0xff ) );
      log2Denom[c] = int( ( p >> 24 ) & 15 );
    }
    n++;
  }
  w[1] |= kInterWp << 16;
  for( int c = 0; c < 3; c++ )
  {
    w[8 + c] = ( uint( weight[0][c] ) & 1023 ) | ( uint( weight[1][c] ) & 1023 ) << 10 | uint( log2Denom[c] ) << 20 | ( uint( offset[0][c] ) & 255 ) << 24;
  }
  w[11] = ( uint( offset[1][0] ) & 255 ) | ( uint( offset[1][1] ) & 255 ) << 8 | ( uint( offset[1][2] ) & 255 ) << 16 | uint( n == 2 ) << 24;
}

// The lists of a block that motionCompensation predicts with one motion, and their references' slots. A bi-prediction
// from one picture with one vector predicts from list 0 alone, unless DMVR refines it or the weighted bi-prediction
// weighs the lists.
uint mcLists( uint* slice, uint* jslice, int refIdx0, int refIdx1, int2 mv0, int2 mv1, bool dmvr, out int ref0, out int ref1 )
{
  uint lists = 0;
  ref0       = 0;
  ref1       = 0;
  if( refIdx0 >= 0 )
  {
    lists |= kInterL0;
    ref0 = refSlot( jslice, 0, refIdx0 );
  }
  if( refIdx1 >= 0 )
  {
    lists |= kInterL1;
    ref1 = refSlot( jslice, 1, refIdx1 );
  }
  if( !dmvr && refIdx0 >= 0 && refIdx1 >= 0 && isInterB( slice ) && ( pc.ph & kPpsWpBi ) == 0 && refPoc( slice, 0, refIdx0 ) == refPoc( slice, 1, refIdx1 )
      && all( mv0 == mv1 ) )
  {
    lists &= ~kInterL1;
  }
  return lists;
}

// The job of a tile of such a block: the tile at (x, y), the block at (bx, by) and bw wide (the reach of the clip of
// the motion). bflags holds the block's flags, bcw the CU's BCW index; ciip: a CIIP CU's block, which has no BCW.
void mcTile( uint bflags, int bcw, bool ciip, uint* jslice, int x, int y, int tileW, int tileH, int bx, int by, int bw, int2 mv0, int2 mv1,
             int ref0, int ref1, int refIdx0, int refIdx1, int dmvrIndex, inout uint w[12] )
{
  setInter( w, x, y, tileW, tileH, bflags, mv0, mv1, bx, by, ref0, ref1, dmvrIndex );
  w[7] |= uint( bw ) << 20;   // setClipWidth
  if( ( bflags & kInterL1 ) != 0 && ( bflags & kInterL0 ) != 0 && bcw != 0 && !ciip )
  {
    w[1] |= kInterBcw << 16;
    w[7] |= ( pc.tables[T_BCW1 + bcw] + 2 ) << 16;
  }
  wpJob( bcw, jslice, refIdx0, refIdx1, bflags, w );
}

// Emits the jobs of a translational CU's block, which motionCompensation predicts with one motion, in tiles of up to
// 16 x 16.
void mcBlock( Cu cu, Motion m, uint* slice, uint* jslice, uint flags, int bx, int by, int bw, int bh, int refIdx0, int refIdx1, int2 mv0, int2 mv1,
              inout Out o )
{
  const bool dmvr = dmvrCondition( cu, m, slice, jslice ) && !( refIdx0 >= 0 && refScaled( jslice, 0, refIdx0 ) )
                    && !( refIdx1 >= 0 && refScaled( jslice, 1, refIdx1 ) );
  int        ref0, ref1;
  uint       bflags = flags | mcLists( slice, jslice, refIdx0, refIdx1, mv0, mv1, dmvr, ref0, ref1 );
  if( bdofCondition( cu, m, slice, jslice ) )
  {
    bflags |= kInterBdof;
  }
  if( dmvr )
  {
    bflags |= kInterDmvr;
  }
  const int tileW = min( bw, 16 );
  const int tileH = min( bh, 16 );
  int       num   = 0;
  for( int y = by; y < by + bh; y += tileH )
  {
    for( int x = bx; x < bx + bw; x += tileW )
    {
      uint w[12];
      mcTile( bflags, m.bcw, ( cu.flags & kCiip ) != 0, jslice, x, y, tileW, tileH, bx, by, bw, mv0, mv1, ref0, ref1, refIdx0, refIdx1,
              dmvr ? cu.dmvrOff + num : 0, w );
      emitInter( o, w );
      num++;
    }
  }
}

// The motion of the 4x4 block at a luma position (mider.slang's MotionInfo): reference indices and vectors.
void motionAt( int2 pos, out int ref[2], out int2 mv[2] )
{
  const int ctuMask  = ( 1 << pc.log2Ctu ) - 1;
  const int ctuSize4 = 1 << ( pc.log2Ctu - 2 );
  const int p        = ( ( pos.x & ctuMask ) >> 2 ) + ( ( ( pos.y & ctuMask ) >> 2 ) * ctuSize4 );
  int*      mi       = pc.motion + ( ctuOfLuma( pos.x, pos.y ) * ctuSize4 * ctuSize4 + p ) * 5;
  mv[0]              = int2( mi[0], mi[1] );
  mv[1]              = int2( mi[2], mi[3] );
  const uint w       = uint( mi[4] );
  ref[0]             = int( int8_t( w & 0xff ) );
  ref[1]             = int( int8_t( ( w >> 8 ) & 0xff ) );
}

// xGpuCiipJobs
void ciipJobs( Cu cu, inout Out o, inout Units cache, bool lmcsChromaOn )
{
  const int4 y4       = cu.blk[0];
  const int  leftIdx  = cuRestricted( int2( y4.x - 1, y4.y + y4.w - 1 ), cu, 0 );
  const int  aboveIdx = cuRestricted( int2( y4.x + y4.z - 1, y4.y - 1 ), cu, 0 );
  const bool leftIn   = leftIdx >= 0 && int( ( pc.cus[leftIdx * 32 + 1] >> 16 ) & 3 ) == MODE_INTRA;
  const bool aboveIn  = aboveIdx >= 0 && int( ( pc.cus[aboveIdx * 32 + 1] >> 16 ) & 3 ) == MODE_INTRA;
  const int  wIntra   = 3 - ( leftIn ? 0 : 1 ) - ( aboveIn ? 0 : 1 );
  const bool blendCh  = pc.chromaFormat != 0 && cu.blk[1].z > 2;
  const int  numComp  = pc.chromaFormat == 0 ? 1 : 3;
  const Tu   firstTu  = loadTu( cu.firstTu );
  for( int comp = 0; comp < numComp; comp++ )
  {
    const int4 area  = cu.blk[chOf( comp )];
    const int3 units = neighborUnits( cu, cu.firstTu, area, comp, cache );
    uint       flags = 0;
    if( ( ( cu.planeCbf >> comp ) & 1 ) != 0 )
    {
      flags |= kIntraResidual;
    }
    if( comp == 0 && useFilteredRef( cu, 0, y4.z, y4.w ) )
    {
      flags |= kIntraFiltered;
    }
    uint w[12];
    setIntra( w, area.x, area.y, area.z, area.w, comp, kIntraCiip, PLANAR_IDX, flags, 0, 0 );
    w[3] = uint( units.x ) | uint( units.y ) << 8 | uint( units.z ) << 16;
    w[6] = uint( comp == 0 || blendCh ? wIntra : 0 );
    if( comp != 0 && lmcsChromaOn && firstTu.blk[1].z * firstTu.blk[1].w > 4 )
    {
      bool residual = false;
      for( int t = 0; t < cu.numTus; t++ )
      {
        const Tu tu = loadTu( cu.firstTu + t );
        residual    = residual || cbfOf( tu, comp ) || tu.jointCbCr != 0;
      }
      if( residual )
      {
        setLmcsScale( firstTu.blk[0].xy, w );
      }
    }
    emitIntra( o, w );
  }
}

// InterPrediction::isSubblockVectorSpreadOverLimit
bool spreadOverLimit( int a, int b, int c, int d, int predType )
{
  const int s4 = 4 << 11;
  if( predType == 3 )
  {
    int bw = max( max( 0, 4 * a + s4 ), max( 4 * c, 4 * a + 4 * c + s4 ) ) - min( min( 0, 4 * a + s4 ), min( 4 * c, 4 * a + 4 * c + s4 ) );
    int bh = max( max( 0, 4 * b ), max( 4 * d + s4, 4 * b + 4 * d + s4 ) ) - min( min( 0, 4 * b ), min( 4 * d + s4, 4 * b + 4 * d + s4 ) );
    bw     = ( bw >> 11 ) + 9;
    bh     = ( bh >> 11 ) + 9;
    return bw * bh > 15 * 15;
  }
  int bw = max( 0, 4 * a + s4 ) - min( 0, 4 * a + s4 );
  int bh = max( 0, 4 * b ) - min( 0, 4 * b );
  bw     = ( bw >> 11 ) + 9;
  bh     = ( bh >> 11 ) + 9;
  if( bw * bh > 15 * 11 )
  {
    return true;
  }
  bw = max( 0, 4 * c ) - min( 0, 4 * c );
  bh = max( 0, 4 * d + s4 ) - min( 0, 4 * d + s4 );
  bw = ( bw >> 11 ) + 9;
  bh = ( bh >> 11 ) + 9;
  return bw * bh > 15 * 11;
}

void interJobs( Cu cu, Motion m, inout Out o, inout Units cache )
{
  uint*      slice      = sliceOfCtu( cu.ctu );
  uint*      jslice     = jobSliceOfCtu( cu.ctu );
  const int4 y4         = cu.blk[0];
  const int  numComp    = pc.chromaFormat == 0 ? 1 : 3;
  uint       flags      = m.hpel ? kInterHalfPel : 0;
  const bool lmcs       = sliceLmcs( jslice ) && ( pc.ph & kPhLmcs ) != 0;
  const bool lmcsChroma = lmcs && ( pc.ph & kPhLmcsChroma ) != 0 && cu.rootCbf;
  if( lmcs )
  {
    flags |= kInterLumaMap;
  }

  // The residual: the GPU adds it to each component that has one. Inside such a component, the transform blocks
  // without a residual are cleared by zero jobs in the residual list.
  if( cu.rootCbf )
  {
    for( int c = 0; c < numComp; c++ )
    {
      if( ( ( cu.planeCbf >> c ) & 1 ) == 0 )
      {
        continue;
      }
      if( ( cu.flags & kCiip ) == 0 )
      {
        flags |= lmcsChroma && c != 0 ? kInterDeferCb << ( c - 1 ) : kInterResidualY << c;
      }
      for( int t = 0; t < cu.numTus; t++ )
      {
        const Tu   tu    = loadTu( cu.firstTu + t );
        const int4 block = tu.blk[chOf( c )];
        if( valid( block ) && !cbfOf( tu, c ) && !( c != 0 && tu.jointCbCr != 0 ) && ( pc.ph & kPhTrafo ) != 0 )
        {
          // in blocks whose sizes are powers of two (an SBT transform unit without residual can be 3/4 of the CU)
          for( int y = 0; y < block.w; y += 1 << log2Of( block.w - y ) )
          {
            for( int x = 0; x < block.z; x += 1 << log2Of( block.z - x ) )
            {
              uint w[12];
              clearJob( w );
              setTrafo( w, block.x + x, block.y + y, log2Of( block.z - x ), log2Of( block.w - y ), c, 0, 0, kTrafoZero, 0, 0, 0, 0, 0, 0 );
              emitTrafo( o, w );
            }
          }
        }
      }
    }
  }

  // CIIP: keep all of inter's prediction aside, for the blend with planar's
  if( ( cu.flags & kCiip ) != 0 )
  {
    flags |= kInterDeferY | kInterDeferCb | kInterDeferCb << 1;
  }

  if( ( cu.flags & kGeo ) != 0 )
  {
    // motionCompensationGeo: each partition is a uni-prediction; the two are blended with the mask of the split's angle
    const int  dir0  = m.geo[0] >> 4;
    const int  dir1  = m.geo[1] >> 4;
    const int  ref0  = refSlot( jslice, dir0 - 1, m.geo[0] & 15 );
    const int  ref1  = refSlot( jslice, dir1 - 1, m.geo[1] & 15 );
    const int  split = cu.geoSplit;
    const int  angle = int( pc.tables[T_GEO + split] & 0xff );
    const int  wIdx  = log2Of( y4.z ) - 3;
    const int  hIdx  = log2Of( y4.w ) - 3;
    const uint woff  = pc.tables[T_WOFF + ( split * 4 + hIdx ) * 4 + wIdx];
    for( int y = y4.y; y < y4.y + y4.w; y += 16 )
    {
      for( int x = y4.x; x < y4.x + y4.z; x += 16 )
      {
        uint w[12];
        setInter( w, x, y, min( 16, y4.z ), min( 16, y4.w ), flags | kInterL0 | kInterL1, m.mv[0][1], m.mv[1][1], y4.x, y4.y, ref0, ref1, 0 );
        w[7] |= uint( y4.z ) << 20;
        w[1] |= kInterGeo << 16;
        // (as InterJob::setGeo packs them)
        w[9] = pc.tables[T_MASK + angle] | pc.tables[T_MIRROR + angle] << 8 | uint( int( int16_t( woff & 0xffff ) ) ) << 16
               | uint( int( int16_t( woff >> 16 ) ) ) << 24;
        emitInter( o, w );
      }
    }
  }
  else if( ( cu.flags & kAffine ) != 0 && m.mrgType != MRG_TYPE_SUBPU_ATMVP )
  {
    // xPredAffineBlk: 4x4 subblocks, each with its own motion, refined by PROF
    uint aflags = flags;
    int  ref0 = 0, ref1 = 0;
    if( m.refIdx[0] >= 0 )
    {
      aflags |= kInterL0;
      ref0 = refSlot( jslice, 0, m.refIdx[0] );
    }
    if( m.refIdx[1] >= 0 )
    {
      aflags |= kInterL1;
      ref1 = refSlot( jslice, 1, m.refIdx[1] );
    }
    if( m.refIdx[0] >= 0 && m.refIdx[1] >= 0 && isInterB( slice ) && ( pc.ph & kPpsWpBi ) == 0
        && refPoc( slice, 0, m.refIdx[0] ) == refPoc( slice, 1, m.refIdx[1] ) && all( m.mv[0][0] == m.mv[1][0] ) && all( m.mv[0][1] == m.mv[1][1] )
        && ( !m.affine6 || all( m.mv[0][2] == m.mv[1][2] ) ) )
    {
      aflags &= ~kInterL1;
    }
    const bool bcw = ( aflags & kInterL0 ) != 0 && ( aflags & kInterL1 ) != 0 && m.bcw != 0;

    bool prof[2]      = { false, false };
    bool spread[2]    = { false, false };
    int  params[2][6] = { { 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0 } };
    for( int list = 0; list < 2; list++ )
    {
      if( m.refIdx[list] < 0 )
      {
        continue;
      }
      const int2 lt    = m.mv[list][0];
      const int2 rt    = m.mv[list][1];
      const int2 lb    = m.mv[list][2];
      const int  dHorX = ( rt.x - lt.x ) * ( 1 << ( 7 - log2Of( y4.z ) ) );
      const int  dHorY = ( rt.y - lt.y ) * ( 1 << ( 7 - log2Of( y4.z ) ) );
      const int  dVerX = m.affine6 ? ( lb.x - lt.x ) * ( 1 << ( 7 - log2Of( y4.w ) ) ) : -dHorY;
      const int  dVerY = m.affine6 ? ( lb.y - lt.y ) * ( 1 << ( 7 - log2Of( y4.w ) ) ) : dHorX;
      spread[list]     = spreadOverLimit( dHorX, dHorY, dVerX, dVerY, m.interDir );
      prof[list]       = ( pc.sps & kSpsProf ) != 0 && ( pc.ph & kPhNoProf ) == 0
                   && !( ( m.affine6 && all( lt == rt ) && all( lt == lb ) ) || ( !m.affine6 && all( lt == rt ) ) ) && !spread[list]
                   && !refScaled( jslice, list, m.refIdx[list] );
      params[list][0] = lt.x * ( 1 << 7 );
      params[list][1] = lt.y * ( 1 << 7 );
      params[list][2] = dHorX;
      params[list][3] = dHorY;
      params[list][4] = dVerX;
      params[list][5] = dVerY;
    }
    for( int y = y4.y; y < y4.y + y4.w; y += 16 )
    {
      for( int x = y4.x; x < y4.x + y4.z; x += 16 )
      {
        uint w[12];
        setInter( w, x, y, min( 16, y4.z ), min( 16, y4.w ), aflags | kInterAffine, int2( 0 ), int2( 0 ), y4.x, y4.y, ref0, ref1, 0 );
        if( bcw )
        {
          w[1] |= kInterBcw << 16;
          w[7] |= ( pc.tables[T_BCW1 + m.bcw] + 2 ) << 16;
        }
        wpJob( m.bcw, jslice, m.refIdx[0], m.refIdx[1], aflags, w );
        uint next[12];
        clearJob( next );
        // InterJob::setAffine
        w[2]    = uint( params[0][0] );
        w[3]    = uint( params[0][1] );
        w[4]    = uint( params[1][0] );
        w[5]    = uint( params[1][1] );
        next[0] = uint( y4.z ) | uint( y4.w ) << 8 | uint( prof[0] ) << 16 | uint( prof[1] ) << 17 | uint( spread[0] ) << 18 | uint( spread[1] ) << 19;
        for( int k = 0; k < 4; k++ )
        {
          next[1 + k] = uint( params[0][2 + k] );
          next[5 + k] = uint( params[1][2 + k] );
        }
        emitInter( o, w );
        emitAffineNext( o, next );
      }
    }
  }
  else if( m.mrgType == MRG_TYPE_SUBPU_ATMVP )
  {
    // SbTMVP: the workgroup makes the jobs of the sub-PUs' motion compensation together (sbtmvpJobs)
    g_sb      = true;
    g_sbFlags = flags;
    g_sbBcw   = m.bcw;
  }
  else
  {
    mcBlock( cu, m, slice, jslice, flags, y4.x, y4.y, y4.z, y4.w, m.refIdx[0], m.refIdx[1], m.mv[0][0], m.mv[1][0], o );
  }

  if( ( cu.flags & kCiip ) != 0 )
  {
    ciipJobs( cu, o, cache, lmcsChroma );
  }
  // The chroma kept aside: each transform block's prediction plus its residual, scaled where finishLMCSAndReco does
  else if( ( flags & ( kInterDeferCb | kInterDeferCb << 1 ) ) != 0 )
  {
    for( int t = 0; t < cu.numTus; t++ )
    {
      const Tu tu = loadTu( cu.firstTu + t );
      for( int c = 1; c <= 2; c++ )
      {
        const int4 block = tu.blk[1];
        if( ( flags & ( kInterDeferCb << ( c - 1 ) ) ) == 0 || !valid( block ) )
        {
          continue;
        }
        const bool scaled = ( cbfOf( tu, c ) || tu.jointCbCr != 0 ) && block.z * block.w > 4;
        for( int y = 0; y < block.w; y += 1 << log2Of( block.w - y ) )
        {
          for( int x = 0; x < block.z; x += 1 << log2Of( block.z - x ) )
          {
            const int bw = 1 << log2Of( block.z - x );
            const int bh = 1 << log2Of( block.w - y );
            uint      w[12];
            setIntra( w, block.x + x, block.y + y, bw, bh, c, kIntraLmcs, 0, kIntraResidual, 0, 0 );
            if( scaled )
            {
              setLmcsScale( tu.blk[0].xy, w );
            }
            emitIntra( o, w );
          }
        }
      }
    }
  }
}

// DecCu::TaskGpuCtu for one CU: emits all of the CU's jobs
void cuJobs( int index, inout Out o )
{
  const Cu cu    = loadCu( index );
  Units    cache = { -1, int3( 0 ) };
  if( ( pc.ph & kPhTrafo ) != 0 && cu.rootCbf )
  {
    trafoJobs( cu, o );
  }
  if( cu.predMode == MODE_INTRA )
  {
    intraJobs( cu, o, cache );
  }
  else if( cu.predMode == MODE_IBC )
  {
    ibcJobs( cu, loadMotion( index ), o );
  }
  else if( ( pc.ph & kPhInter ) != 0 )
  {
    interJobs( cu, loadMotion( index ), o, cache );
  }
}

// --------------------------------------------------------------------------------------------------------------------
// The waves of the intra jobs (GpuJobs.h's WaveMap, as DecCu's xGpu*Jobs use it)
// --------------------------------------------------------------------------------------------------------------------

int waveShiftX( int c ) { return 2 - ( c == 0 ? 0 : pc.csx ); }
int waveShiftY( int c ) { return 2 - ( c == 0 ? 0 : pc.csy ); }
int waveWidth( int c ) { return ( ( pc.width >> ( c == 0 ? 0 : pc.csx ) ) + ( 1 << waveShiftX( c ) ) - 1 ) >> waveShiftX( c ); }
int waveHeight( int c ) { return ( ( pc.height >> ( c == 0 ? 0 : pc.csy ) ) + ( 1 << waveShiftY( c ) ) - 1 ) >> waveShiftY( c ); }
// A wave-map entry: the wave | the tag of the picture that wrote it << 16. An entry of another picture is wave 0,
// as the map was zeroed for each picture before (GpuRecon zeroes it only when the tags wrap around).
int mapWave( uint v )
{
  return ( v >> 16 ) == pc.waveTag ? int( v & 0xffffu ) : 0;
}

uint* waveRow( int c, int j )
{
  const int planeSize = waveWidth( 0 ) * waveHeight( 0 );
  return pc.waveMap + c * planeSize + j * waveWidth( c );
}

// WaveMap::maxIn computed by the 32 threads of a row's workgroup (lane is the calling thread's), each thread taking
// every 32nd unit; every thread gets the highest wave in the rectangle.
int waveMaxInP( int c, int x, int y, int width, int height, int lane )
{
  if( x + width <= 0 || y + height <= 0 )
  {
    return 0;
  }
  const int x0   = max( x, 0 ) >> waveShiftX( c );
  const int y0   = max( y, 0 ) >> waveShiftY( c );
  const int x1   = min( ( x + width - 1 ) >> waveShiftX( c ), waveWidth( c ) - 1 );
  const int y1   = min( ( y + height - 1 ) >> waveShiftY( c ), waveHeight( c ) - 1 );
  const int cols = x1 - x0 + 1;
  const int n    = cols > 0 && y1 >= y0 ? cols * ( y1 - y0 + 1 ) : 0;
  int         wave = 0;
  const float rcp  = 1.0f / float( max( cols, 1 ) );   // (exactly u / cols, since u and cols are small)
  for( int u = lane; u < n; u += 32 )
  {
    const int j = y0 + int( ( float( u ) + 0.5f ) * rcp );
    wave        = max( wave, mapWave( waveRow( c, j )[x0 + u - ( j - y0 ) * cols] ) );
  }
  return WaveActiveMax( wave );
}

// WaveMap::set by the 32 threads of a row's workgroup, each thread taking every 32nd unit.
void waveSetP( int c, int x, int y, int width, int height, int wave, int lane )
{
  const int x0   = x >> waveShiftX( c );
  const int y0   = y >> waveShiftY( c );
  const int x1   = min( ( x + width - 1 ) >> waveShiftX( c ), waveWidth( c ) - 1 );
  const int y1   = min( ( y + height - 1 ) >> waveShiftY( c ), waveHeight( c ) - 1 );
  const int cols = x1 - x0 + 1;
  const int n    = cols > 0 && y1 >= y0 ? cols * ( y1 - y0 + 1 ) : 0;
  const float rcp = 1.0f / float( max( cols, 1 ) );   // (exactly u / cols, since u and cols are small)
  for( int u = lane; u < n; u += 32 )
  {
    const int j                                 = y0 + int( ( float( u ) + 0.5f ) * rcp );
    waveRow( c, j )[x0 + u - ( j - y0 ) * cols] = uint( wave ) | pc.waveTag << 16;
  }
}

// Sets a job's wave to one above the highest wave of the samples it reads (the row's 32 threads work together).
void jobWaveP( uint* job, int lane )
{
  const int  x     = int( job[0] & 0xffff );
  const int  y     = int( job[0] >> 16 );
  const int  w     = int( job[1] & 0xff );
  const int  h     = int( ( job[1] >> 8 ) & 0xff );
  const int  c     = int( ( job[1] >> 16 ) & 0xff );
  const int  kind  = int( job[1] >> 24 );
  const uint flags = ( job[2] >> 8 ) & 0xff;
  int        wave  = 0;
  if( kind == kIntraIbc )
  {
    const int bvx = int( int16_t( job[6] & 0xffff ) );
    const int bvy = int( int16_t( job[6] >> 16 ) );
    wave          = waveMaxInP( c, x + bvx, y + bvy, w, h, lane );
  }
  else if( kind != kIntraLmcs )
  {
    if( ( flags & kIntraIsp ) != 0 )
    {
      const int cx    = int( job[4] & 0xffff );
      const int cy    = int( job[4] >> 16 );
      const int cw    = int( job[5] & 0xff );
      const int ch    = int( ( job[5] >> 8 ) & 0xff );
      const int split = int( ( job[5] >> 16 ) & 3 );
      int       px = x, pw = w;
      if( split == 2 && ( ( cw == 8 && ch > 4 ) || cw == 4 ) )
      {
        px = cx + ( ( x - cx ) & ~3 );
        pw = 4;
      }
      const bool first   = px == cx && y == cy;
      const int  x0      = first ? cx : px;
      const int  y0      = first ? cy : y;
      const int  topLen  = first ? 2 * cw : cw + pw;
      const int  leftLen = first ? 2 * ch : ch + h;
      wave               = max( waveMaxInP( 0, x0 - 1, y0 - 1, topLen + 1, 1, lane ), waveMaxInP( 0, x0 - 1, y0, 1, leftLen, lane ) );
    }
    else
    {
      const int refLine = int( job[2] >> 24 );
      wave = max( waveMaxInP( c, x - 1 - refLine, y - 1 - refLine, 2 * w + 1 + refLine, 1, lane ),
                  waveMaxInP( c, x - 1 - refLine, y - refLine, 1, 2 * h + refLine, lane ) );
    }
  }
  if( ( job[9] & 1 ) != 0 )
  {
    const int ax    = int( job[8] & 0xffff );
    const int ay    = int( job[8] >> 16 );
    const int count = 1 << ( ( job[9] >> 4 ) & 15 );
    if( ( job[9] & 2 ) != 0 )
    {
      wave = max( wave, waveMaxInP( 0, ax - 1, ay, 1, count, lane ) );
    }
    if( ( job[9] & 4 ) != 0 )
    {
      wave = max( wave, waveMaxInP( 0, ax, ay - 1, count, 1, lane ) );
    }
  }
  if( kind == kIntraCclm )
  {
    const int lx    = x << pc.csx, ly = y << pc.csy, lw = w << pc.csx, lh = h << pc.csy;
    const int right = ( ( lx >> pc.log2Ctu ) + 2 ) << pc.log2Ctu;
    wave            = max( wave, waveMaxInP( 0, lx - 4, ly - 4, min( 2 * lw + 8, right - ( lx - 4 ) ), 2 * lh + 8, lane ) );
  }
  if( lane == 0 )
  {
    job[7] = uint( wave + 1 );
  }
  waveSetP( c, x, y, w, h, wave + 1, lane );
}

// --------------------------------------------------------------------------------------------------------------------
// The passes
// --------------------------------------------------------------------------------------------------------------------

#if defined( WAVES )
groupshared int  s_row;
groupshared int2 s_jobRange[32];   // the intra jobs of the 32 CTUs being walked: [first, end)

// The intra jobs of CTU ctu: from the first job of its first CU up to the first job of the next CTU's first CU
int2 intraJobRange( int ctu )
{
  const int first = int( pc.ctus[ctu * 4] );
  const int count = int( pc.ctus[ctu * 4 + 1] & 0xffff );
  if( count == 0 )
  {
    return int2( 0, 0 );
  }
  const int begin = int( pc.blocks[( first >> 8 ) * kLists + 2] + pc.counts[first * kLists + 2] );
  const int next  = first + count;
  const int end   = next < pc.numCus ? int( pc.blocks[( next >> 8 ) * kLists + 2] + pc.counts[next * kLists + 2] )
                                     : int( pc.blocks[pc.numBlocks * kLists + 2] );
  return int2( begin, min( end, pc.capIntra ) );
}

// WAVES: one workgroup of 32 threads per CTU row; the rows are taken in order (from a counter) and run as the
// wavefront of vvdec's CTU tasks: a CTU with intra jobs waits until the row above is past the CTU above-right (its
// jobs read samples up to there). A CTU without intra jobs reads and writes no wave, so it waits for nothing.
[shader( "compute" )]
[numthreads( 32, 1, 1 )]
void main( uint3 local : SV_GroupThreadID )
{
  const int lane = int( local.x );
  if( lane == 0 )
  {
    s_row = int( pc.progress[pc.heightInCtus].add( 1u ) );
  }
  GroupMemoryBarrierWithGroupSync();
  const int row = s_row;
  if( row >= pc.heightInCtus )
  {
    return;
  }
  for( int col = 0; col < pc.widthInCtus; col++ )
  {
    // every 32 CTUs, each thread reads one CTU's range of jobs
    if( ( col & 31 ) == 0 )
    {
      GroupMemoryBarrierWithGroupSync();   // (the ranges before read)
      s_jobRange[lane] = col + lane < pc.widthInCtus ? intraJobRange( row * pc.widthInCtus + col + lane ) : int2( 0, 0 );
      GroupMemoryBarrierWithGroupSync();
    }
    const int2 range = s_jobRange[col & 31];
    if( range.x < range.y )
    {
      if( row > 0 )
      {
        if( lane == 0 )
        {
          const uint need = uint( min( col + 2, pc.widthInCtus ) );
          while( pc.progress[row - 1].load( MemoryOrder.Relaxed ) < need )
          {
          }
        }
        AllMemoryBarrierWithGroupSync();
      }
      for( int j = range.x; j < range.y; j++ )
      {
        jobWaveP( pc.intra + j * 12, lane );
        // (the job's wave-map units are written before the next job reads them)
        workgroupBarrierAll();
      }
      AllMemoryBarrierWithGroupSync();   // (the CTU's waves visible to the rows below)
    }
    if( lane == 0 )
    {
      pc.progress[row].store( uint( col + 1 ), MemoryOrder.Release );
    }
  }
}
#else
// SbTMVP's motion compensation (xSubPuMC). An SbTMVP CU's 8x8 sub-PUs are joined along the CU's longer side where their
// motion is the same, and each joined run is predicted in tiles of 16 samples along that side (the last tile 8, where
// the run's length is odd). A thread that made a large CU's jobs alone held up its wave for long: up to 256 sub-PUs,
// each read and compared after the one before. So the workgroup makes them together, an SbTMVP CU at a time: a thread
// per sub-PU reads the sub-PU's motion, compares it with the motion of the sub-PU before it, and makes the job that
// starts at its sub-PU, if one does.
groupshared int  s_sbOwner[256];   // the threads whose CUs are SbTMVP CUs
groupshared int4 s_sbCu[2];        // the CU at hand: its luma block; its CTU, its jobs' flags, its BCW index, its first job
groupshared int4 s_sbMv[256];      // per sub-PU, line by line along the runs: its vectors (list 0's x, y, list 1's x, y)
groupshared uint s_sbRef[256];     // and its reference indices (list 0's | list 1's << 8, 8 bits each)
groupshared uint s_sbJoined[16];   // per line: bit k, the line's sub-PU k has the motion of sub-PU k - 1

// MotionInfo::operator== on two sub-PUs of s_sbMv and s_sbRef: the vectors of the lists they use
bool sameSubPuMotion( int a, int b )
{
  const uint r = s_sbRef[a];
  if( r != s_sbRef[b] )
  {
    return false;
  }
  const int4 ma = s_sbMv[a];
  const int4 mb = s_sbMv[b];
  return ( ( r & 0xff ) == 0xff || all( ma.xy == mb.xy ) ) && ( ( r >> 8 ) == 0xff || all( ma.zw == mb.zw ) );
}

// The jobs of the SbTMVP CUs of the workgroup's threads (g_sb), counted into their threads' Out and, in WRITE, written.
// All the threads call it, in uniform control flow.
void sbtmvpJobs( int tid, int index, inout Out o )
{
  uint       numSb;
  const uint at = workgroupPrefixSum( g_sb ? 1u : 0u, tid, numSb );
  if( g_sb )
  {
    s_sbOwner[at] = tid;
  }
  for( uint i = 0; i < numSb; i++ )
  {
    GroupMemoryBarrierWithGroupSync();   // (s_sbOwner written; the shared state of the CU before read)
    const int owner = s_sbOwner[i];
    if( tid == owner )
    {
      const Cu cu = loadCu( index );
      s_sbCu[0]   = cu.blk[0];
      s_sbCu[1]   = int4( cu.ctu, int( g_sbFlags ), g_sbBcw, o.base[1] + o.n[1] );
    }
    if( tid < 16 )
    {
      s_sbJoined[tid] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    const int4 y4     = s_sbCu[0];
    const int4 info   = s_sbCu[1];
    const bool verMC  = y4.w > y4.z;                     // the runs go down the CU's columns
    const int  n1     = ( verMC ? y4.w : y4.z ) >> 3;   // the sub-PUs of a line
    const int  k      = tid & ( n1 - 1 );               // the thread's sub-PU: its place in its line, and its line
    const int  line   = tid >> log2Of( n1 );
    const bool mine   = tid < ( y4.z >> 3 ) * ( y4.w >> 3 );
    const int  x      = y4.x + 8 * ( verMC ? line : k );
    const int  y      = y4.y + 8 * ( verMC ? k : line );
    uint*      slice  = sliceOfCtu( info.x );
    uint*      jslice = jobSliceOfCtu( info.x );
    if( mine )
    {
      int  ref[2];
      int2 mv[2];
      motionAt( int2( x, y ), ref, mv );
      s_sbMv[tid]  = int4( mv[0], mv[1] );
      s_sbRef[tid] = ( uint( ref[0] ) & 0xff ) | ( uint( ref[1] ) & 0xff ) << 8;
    }
    GroupMemoryBarrierWithGroupSync();
    // (with a scaled reference picture no sub-PUs are joined)
    const bool scaled = refScaled( jslice, 0, 0 ) || ( sliceType( jslice ) == 0 && refScaled( jslice, 1, 0 ) );
    if( mine && k > 0 && !scaled && sameSubPuMotion( tid, tid - 1 ) )
    {
      InterlockedOr( s_sbJoined[line], 1u << k );
    }
    GroupMemoryBarrierWithGroupSync();
    // The sub-PU's run: from the line's sub-PU s up to e (exclusive). Its jobs start at every second sub-PU from s.
    const uint joined = mine ? s_sbJoined[line] : 0u;
    const uint upTo   = ( 2u << k ) - 1;
    const int  s      = int( firstbithigh( ~joined & upTo ) );
    const uint after  = ~joined & ( ( 1u << n1 ) - 1 ) & ~upTo;
    const int  e      = after != 0 ? int( firstbitlow( after ) ) : n1;
    const bool starts = mine && ( ( k - s ) & 1 ) == 0;
    uint       numJobs;
    const uint job = workgroupPrefixSum( starts ? 1u : 0u, tid, numJobs );
    if( tid == owner )
    {
      o.n[1] += int( numJobs );
    }
    if( o.write && starts )
    {
      // The tile of two sub-PUs, in the block of the run's even length; or the run's last sub-PU alone, in its own
      // block, where the length is odd. The run's motion is its first sub-PU's.
      const bool alone   = ( ( e - s ) & 1 ) != 0 && k == e - 1;
      const int  from    = alone ? k : s;   // the block's first sub-PU, and its length
      const int  length  = alone ? 1 : ( e - s ) & ~1;
      const int  first   = line * n1 + s;
      const int4 mv      = s_sbMv[first];
      const uint r       = s_sbRef[first];
      const int  refIdx0 = int( int8_t( r & 0xff ) );
      const int  refIdx1 = int( int8_t( r >> 8 ) );
      int        ref0, ref1;
      const uint bflags  = uint( info.y ) | mcLists( slice, jslice, refIdx0, refIdx1, mv.xy, mv.zw, false, ref0, ref1 );
      uint       w[12];
      if( !verMC )
      {
        mcTile( bflags, info.z, false, jslice, x, y, alone ? 8 : 16, 8, y4.x + 8 * from, y, 8 * length, mv.xy, mv.zw, ref0, ref1, refIdx0, refIdx1, 0, w );
      }
      else
      {
        mcTile( bflags, info.z, false, jslice, x, y, 8, alone ? 8 : 16, x, y4.y + 8 * from, 8, mv.xy, mv.zw, ref0, ref1, refIdx0, refIdx1, 0, w );
      }
      writeInter( info.w + int( job ), w );
    }
  }
}

[shader( "compute" )]
[numthreads( 256, 1, 1 )]
void main( uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID )
{
  const int tid = int( local.x );
  if( !COUNT_ONLY && pc.pass == 1 )
  {
    // SCAN: turn the blocks' totals into offsets; each thread takes a run of blocks.
    const int per = ( pc.numBlocks + 255 ) / 256;
    const int b0  = min( tid * per, pc.numBlocks );
    const int b1  = min( b0 + per, pc.numBlocks );
    int       sum[5] = { 0, 0, 0, 0, 0 };
    for( int b = b0; b < b1; b++ )
    {
      for( int k = 0; k < kLists; k++ )
      {
        sum[k] += int( pc.blocks[b * kLists + k] );
      }
    }
    int at[5];
    for( int k = 0; k < kLists; k++ )
    {
      uint total;
      at[k] = int( workgroupPrefixSum( uint( sum[k] ), tid, total ) );
      if( tid == 0 )
      {
        pc.blocks[pc.numBlocks * kLists + k] = total;
      }
    }
    for( int b = b0; b < b1; b++ )
    {
      for( int k = 0; k < kLists; k++ )
      {
        const int v               = int( pc.blocks[b * kLists + k] );
        pc.blocks[b * kLists + k] = uint( at[k] );
        at[k] += v;
      }
    }
    // The indirect dispatches: one workgroup per job (per pair, for affine), in rows of 32768 workgroups.
    if( tid == 0 )
    {
      // (inter's lists lie one after another in its buffer; first[] holds their first jobs)
      const uint regular  = pc.blocks[pc.numBlocks * kLists + 1];
      const uint bdof     = pc.blocks[pc.numBlocks * kLists + 3];
      const uint total[4] = { pc.blocks[pc.numBlocks * kLists + 0], regular, bdof, pc.blocks[pc.numBlocks * kLists + 4] / 2 };
      const uint first[4] = { 0, 0, regular, regular + bdof };
      for( int d = 0; d < 4; d++ )
      {
        pc.args[3 * d]          = min( total[d], 32768u );
        pc.args[3 * d + 1]      = ( total[d] + 32767u ) / 32768u;
        pc.args[3 * d + 2]      = 1;
        pc.args[32 + 2 * d]     = total[d];
        pc.args[32 + 2 * d + 1] = first[d];
      }
    }
    return;
  }

  const int index = int( group.x ) * 256 + tid;
  Out       o;
  o.write = !COUNT_ONLY && pc.pass == 2;
  for( int k = 0; k < kLists; k++ )
  {
    o.n[k]    = 0;
    o.base[k] = o.write && index < pc.numCus ? int( pc.blocks[group.x * kLists + k] + pc.counts[index * kLists + k] ) : 0;
  }
  if( o.write )
  {
    // inter's lists lie one after another
    o.base[3] += int( pc.blocks[pc.numBlocks * kLists + 1] );
    o.base[4] += int( pc.blocks[pc.numBlocks * kLists + 1] + pc.blocks[pc.numBlocks * kLists + 3] );
  }
  if( index < pc.numCus )
  {
    cuJobs( index, o );
  }
  sbtmvpJobs( tid, index, o );
  if( COUNT_ONLY || pc.pass == 0 )
  {
    // COUNT: store the CU's offsets within its block, and the block's totals.
    for( int k = 0; k < kLists; k++ )
    {
      uint       total;
      const uint before = workgroupPrefixSum( index < pc.numCus ? uint( o.n[k] ) : 0u, tid, total );
      if( tid == 0 )
      {
        pc.blocks[group.x * kLists + k] = total;
      }
      if( index < pc.numCus )
      {
        pc.counts[index * kLists + k] = before;
      }
    }
  }
}
#endif
