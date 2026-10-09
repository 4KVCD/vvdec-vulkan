// The parse of the slice data (CABAC) on the GPU: vvdec's DecSlice::parseSlice and CABACReader, from each substream's
// bytes to what the parse leaves for the reconstruction -- the CUs' and TUs' records (GpuFrontEnd.h's CuRecord and
// TuRecord, as writeCuRecords writes them from the CPU's parse), the residuals' levels in the planes of levels (as
// CABACReader::residual_coding leaves them in the picture: each coded block's levels over the rectangle its TU's
// maxScanPosX / maxScanPosY give, with dependent quantization's 2 q - parity), each CTU's SAO and ALF parameters as
// the parse leaves them (merges unresolved).
//
// A workgroup of 32 threads takes a substream -- a slice's, a tile's or, with WPP, a row of CTUs, as the entry points
// split the slice data -- the workgroup of the substream's index. All 32 threads take every step of the parse (the
// same data, the same results: on a SIMD unit what one thread computes, all compute at the cost of one); the loops
// over a block's 4x4 units or levels give each thread every 32nd. The arithmetic decoder and the context models are
// vvdec's (BinDecoder, BinProbModel), the contexts in shared memory, their initial states and rates from the host (each
// slice's, vvdec's own Ctx::init). With WPP a row begins with the contexts the row above had after its first CTU, and
// each CTU needs the CTU above it parsed.
//
// A dispatch parses each substream up to a budget of bins (pc.binBudget, a CU counted as 32 more), then saves where it
// is at the CTU's end (kSt...) and the next dispatch resumes it; a CTU whose CTU above, another substream's, is not
// parsed after a short wait pauses its substream till the next dispatch too. So no dispatch runs long (Windows resets a
// GPU busy for 2 s on one), whatever the slice data; the host records enough of them for the picture.
//
// The records go to each CTU's slots (as many as a CTU can have CUs and TUs), the neighbours looked up through maps of
// the CUs (the CU's slot + 1 at each 4x4 luma block, a map per channel type); parsecompact.slang then gathers the
// records in raster order.

#include "common.slang"
#include "parsectx.slang"

#ifndef PROFILE
#  define PROFILE 0
#endif

struct Params
{
  uint*         bits;      // the slice data of the picture's slices, one after the other (bytes; 256 more readable)
  uint*         streams;   // per substream: kStreamWords words
  uint*         ctuList;   // the substreams' CTUs in their order, 2 words each: the CTU's address, kCtu... flags | slot << 8
  uint*         ctuInfo;   // per CTU, 2 words: its slice; its tile | the first column of its tile << 16
  uint*         slices;    // per slice: kSliceWords words, then its contexts' initial states and rates (CTX_COUNT each)
  int*          pic;       // the picture's parameters (kPic...)
  uint*         scanTab;   // vvdec's diagonal scans (g_scanOrder's ungrouped) of the blocks 1..16 x 1..16: kScanTab...
  uint*         cus;       // out: the CU records, kPicCuSlots slots a CTU
  uint*         tus;       // out: the TU records, kPicTuSlots slots a CTU
  uint*         ctuOut;    // out: per CTU kCtuOutWords: its CUs, its TUs, its SAO parameters, its ALF parameters
  uint2*        bottomRows;// per CTU its CUs' word pairs (s_map's) of its bottom row of 4x4 blocks, luma's then chroma's
  int16_t*      levels0;   // out: the levels, laid out as the planes
  int16_t*      levels1;
  int16_t*      levels2;
  Atomic<uint>* sync;      // [0] the substreams' runs (a dispatch's each), [1] errors, then per CTU: parsed (all zeroed before)
  uint*         wppCtx;    // WPP: the contexts after a row's first CTU (CTX_COUNT words a slot)
  int           numStreams;
  int           stride0;   // the planes of levels' strides
  int           stride12;
  Atomic<uint>* prof;      // PROFILE: the parts' cycles and counts (kProf...), summed
  uint*         state;     // per substream kStateWords: where its parse is (zeroed before the first dispatch)
  int           binBudget; // the bins a dispatch parses of a substream (then it pauses after the CTU)
};

// a substream's state between dispatches
static const int kStateWords   = 1024;
static const int kStK          = 0;     // its next CTU (of pc.ctuList's)
static const int kStDone       = 1;     // parsed (or given up)
static const int kStRange      = 2;     // the arithmetic decoder's
static const int kStValue      = 3;
static const int kStBitsNeeded = 4;
static const int kStPos        = 5;
static const int kStQp0        = 6;     // the QPs before (luma's, the dual tree chroma's)
static const int kStQp1        = 7;
static const int kStCqpAdj     = 8;     // CodingStructure::chromaQpAdj
static const int kStCtx        = 16;    // the contexts' states (their rates: the slice's)
static const int kStLeft       = 400;   // the column left of the next CTU (s_left, 64 word pairs)

static uint g_binCount;   // the bins decoded in this dispatch (and 32 a CU)

// PROFILE: the parts timed (cycles of the subgroup's clock) and the counts
static const int kProfCtu      = 0;    // the CTUs, whole
static const int kProfCu       = 1;    // coding_unit (its prediction data, its residuals)
static const int kProfResidual = 2;    // residual_coding (its levels' writing too)
static const int kProfLevels   = 3;    // the levels' writing
static const int kProfRecords  = 4;    // the CU's and TUs' records, its word pairs in the maps
static const int kProfSao      = 5;    // SAO's and ALF's syntax
static const int kProfDecode   = 6;    // in decodeBin
static const int kProfDecodeEP = 7;    // in decodeBinEP
static const int kProfBins     = 8;    // counts: the context-coded bins
static const int kProfBinsEP   = 9;    // the bypass bins
static const int kProfCus      = 10;
static const int kProfTus      = 11;
static const int kProfBlocks   = 12;   // the coded blocks (residual_coding's)
static const int kProfLevelsN  = 13;   // the levels (non-zero)
static const int kProfWords    = 16;

#if PROFILE
static uint g_prof[kProfWords];
uint profClock()
{
  return clock2x32ARB().x;
}
void profAdd( int k, uint v )
{
  g_prof[k] += v;
}
#  define PROF_BEGIN( name ) const uint name = profClock()
#  define PROF_END( k, name ) profAdd( k, ( profClock() - name ) & 0xfffffu )   // (the subgroup clock: 20 bits of cycles on RDNA)
#  define PROF_COUNT( k, v ) profAdd( k, uint( v ) )
#else
#  define PROF_BEGIN( name )
#  define PROF_END( k, name )
#  define PROF_COUNT( k, v )
#endif

[[vk::push_constant]] Params pc;

// --------------------------------------------------------------------------------------------------------------------
// the host's tables
// --------------------------------------------------------------------------------------------------------------------

// the picture's parameters (pc.pic)
static const int kPicWidthInCtus  = 0;
static const int kPicHeightInCtus = 1;
static const int kPicLog2Ctu      = 2;
static const int kPicWidth        = 3;    // luma samples
static const int kPicHeight       = 4;
static const int kPicChromaFormat = 5;    // vvdec's ChromaFormat: 0 4:0:0, 1 4:2:0, 2 4:2:2, 3 4:4:4
static const int kPicQpBdOffset   = 6;
static const int kPicLog2MinCb    = 7;
static const int kPicMaxTbSize    = 8;
static const int kPicWpp          = 9;    // entropy coding sync
static const int kPicLog2MaxTs    = 10;   // the largest transform skip block's log2 size
static const int kPicTools        = 11;   // the SPS's tools: kTool...
static const int kPicMaxMergeCand = 12;
static const int kPicMaxGeoCand   = 13;
static const int kPicMaxIbcCand   = 14;
static const int kPicUseDqp       = 15;   // the PPS's cu_qp_delta_enabled_flag
static const int kPicCqpListLen   = 16;   // the PPS's chroma QP offset list's length
static const int kPicTrRange      = 17;   // the SPS's maxLog2TrDynamicRange: luma | chroma << 8
static const int kPicCuSlots      = 18;   // a CTU's slots of CU records
static const int kPicTuSlots      = 19;   // and of TU records
static const int kPicTileCols     = 20;
static const int kPicCqpOffsets   = 24;   // the PPS's chroma QP offsets (Cb, Cr, joint Cb-Cr), then its list's entries 1..6
static const int kPicWords        = 48;   // the words before the chroma QP mappings
static const int kPicChromaQpMap  = 160;  // the SPS's chroma QP mappings (Cb, Cr, joint Cb-Cr), 128 each, at qp + qpBdOffset

// the SPS's tools (kPicTools)
static const int kToolSbt        = 1 << 0;
static const int kToolAffine     = 1 << 1;
static const int kToolAffineType = 1 << 2;    // the 6-parameter model
static const int kToolAmvr       = 1 << 3;
static const int kToolAffineAmvr = 1 << 4;
static const int kToolBcw        = 1 << 5;
static const int kToolCiip       = 1 << 6;
static const int kToolGeo        = 1 << 7;
static const int kToolMmvd       = 1 << 8;
static const int kToolSmvd       = 1 << 9;
static const int kToolIsp        = 1 << 10;
static const int kToolMrl        = 1 << 11;
static const int kToolMip        = 1 << 12;
static const int kToolLfnst      = 1 << 13;
static const int kToolMtsIntra   = 1 << 14;
static const int kToolMtsInter   = 1 << 15;
static const int kToolLmChroma   = 1 << 16;
static const int kToolIbc        = 1 << 17;
static const int kToolAct        = 1 << 18;
static const int kToolBdpcm      = 1 << 19;
static const int kToolTs         = 1 << 20;
static const int kToolJointCbCr  = 1 << 21;
static const int kToolMts        = 1 << 22;   // sps_mts_enabled_flag (SPS::getUseMTS)

// a slice's parameters (pc.slices)
static const int kSliceWords     = 16;
static const int kSliceStride    = kSliceWords + 2 * CTX_COUNT;
static const int kSlType         = 0;    // vvdec's SliceType: 0 B, 1 P, 2 I
static const int kSlQp           = 1;
static const int kSlFlags        = 2;    // kSl...
static const int kSlNumRef       = 3;    // L0 | L1 << 8
static const int kSlSymRef       = 4;    // SMVD's reference indices: L0 | L1 << 8
static const int kSlMaxAffMerge  = 5;    // the picture header's MaxNumAffineMergeCand
static const int kSlAlf          = 6;    // ALF's APSs | chroma's alternatives << 8 | CC-ALF's Cb filters << 16 | Cr's << 24
static const int kSlSubdiv       = 7;    // cu_qp_delta_subdiv | cu_chroma_qp_offset_subdiv << 8
static const int kSlChromaQpDelta = 8;   // Cb, Cr, joint Cb-Cr (3 words)
static const int kSlWp           = 11;   // weighted prediction's reference indices: L0's bits | L1's << 16
static const int kSlPartLuma     = 12;   // maxBTD | maxBtSize << 8 | maxTtSize << 16 | minQtSize << 24
static const int kSlPartChroma   = 13;   // the dual tree chroma's

static const int kSlSaoLuma      = 1 << 0;
static const int kSlSaoChroma    = 1 << 1;
static const int kSlAlfY         = 1 << 2;
static const int kSlAlfCb        = 1 << 3;
static const int kSlAlfCr        = 1 << 4;
static const int kSlCcAlfCb      = 1 << 5;
static const int kSlCcAlfCr      = 1 << 6;
static const int kSlDepQuant     = 1 << 7;
static const int kSlSignHiding   = 1 << 8;
static const int kSlTsResOff     = 1 << 9;   // slice_ts_residual_coding_disabled_flag
static const int kSlChromaQpAdj  = 1 << 10;
static const int kSlMvdL1Zero    = 1 << 11;
static const int kSlBiDirPred    = 1 << 12;
static const int kSlCheckLDC     = 1 << 13;
static const int kSlDualTree     = 1 << 14;   // an intra slice with the SPS's dual tree

// a substream (pc.streams)
static const int kStreamWords    = 8;
static const int kStOffset       = 0;    // its first byte in pc.bits
static const int kStSize         = 1;
static const int kStFirstCtu     = 2;    // its first CTU in pc.ctuList
static const int kStCtus         = 3;
static const int kStSlice        = 4;

// the CTUs of pc.ctuList: flags
static const uint kCtuRestart    = 1;    // the contexts initialised (and the arithmetic decoder restarted, unless the substream's first)
static const uint kCtuSyncLoad   = 2;    // WPP: then the row above's contexts (slot - tile columns)
static const uint kCtuSyncSave   = 4;    // WPP: the contexts after the CTU saved (slot)
static const uint kCtuTerminate  = 8;    // a substream's end within the substream (no entry points): end_of_subset_one_bit
static const uint kCtuWaitAbove  = 16;   // the CTU above in the same slice and tile, from another substream: wait for it

// a CTU's out (pc.ctuOut)
static const int kCtuOutWords    = 16;   // its CUs, its TUs, SAO (3 x 2 words), ALF (1 word)

// the CU records' flags (GpuFrontEnd.h's CuRecord)
static const uint kMerge   = 1;
static const uint kSkip    = 2;
static const uint kMmvd    = 4;
static const uint kAffine  = 8;
static const uint kGeo     = 16;
static const uint kCiip    = 32;
static const uint kChroma  = 64;
static const uint kAffine6 = 128;

// --------------------------------------------------------------------------------------------------------------------
// vvdec's constants
// --------------------------------------------------------------------------------------------------------------------

static const int MODE_INTER = 0;
static const int MODE_INTRA = 1;
static const int MODE_IBC   = 2;

static const int TREE_D = 0;
static const int TREE_L = 1;
static const int TREE_C = 2;

static const int MODE_TYPE_ALL   = 0;
static const int MODE_TYPE_INTER = 1;
static const int MODE_TYPE_INTRA = 2;

static const int B_SLICE = 0;
static const int P_SLICE = 1;
static const int I_SLICE = 2;

static const int CTU_LEVEL     = 0;
static const int CU_QUAD_SPLIT = 1;
static const int CU_HORZ_SPLIT = 2;
static const int CU_VERT_SPLIT = 3;
static const int CU_TRIH_SPLIT = 4;
static const int CU_TRIV_SPLIT = 5;
static const int CU_DONT_SPLIT = 2000;

static const int PLANAR_IDX    = 0;
static const int DC_IDX        = 1;
static const int HOR_IDX       = 18;
static const int VER_IDX       = 50;
static const int VDIA_IDX      = 66;
static const int NUM_LUMA_MODE = 67;
static const int LM_CHROMA_IDX = 67;
static const int MDLM_L_IDX    = 68;
static const int MDLM_T_IDX    = 69;
static const int DM_CHROMA_IDX = 70;

static const int MTS_DCT2_DCT2 = 0;
static const int MTS_SKIP      = 1;
static const int MTS_DST7_DST7 = 2;

static const int NOT_INTRA_SUBPARTITIONS = 0;
static const int HOR_INTRA_SUBPARTITIONS = 1;
static const int VER_INTRA_SUBPARTITIONS = 2;

static const int SBT_VER_HALF = 1;
static const int SBT_HOR_HALF = 2;
static const int SBT_VER_QUAD = 3;
static const int SBT_HOR_QUAD = 4;

static const int IMV_HPEL = 3;

static const int MAX_NUM_REF = 16;

static const int SAO_MODE_OFF   = 0;
static const int SAO_MODE_NEW   = 1;
static const int SAO_MODE_MERGE = 2;
static const int SAO_TYPE_START_EO = 0;
static const int SAO_TYPE_START_BO = 4;

static const int MAX_QP = 63;
static const int NUM_FIXED_FILTER_SETS = 16;
static const int MAX_TU_SIZE_FOR_PROFILE = 64;
static const int MIN_DUALTREE_CHROMA_WIDTH = 4;
static const int MIN_DUALTREE_CHROMA_SIZE  = 16;
static const int MV_BITS = 18;
static const int COEF_REMAIN_BIN_REDUCTION = 5;
static const int SBH_THRESHOLD = 4;
static const int CU_DQP_TU_CMAX = 5;

static const uint kRenormTable[32] = { 6, 5, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
static const int  kGroupIdx[64] = { 0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
                                    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11 };
static const int  kMinInGroup[14] = { 0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96 };
static const int  kGoRiceParsCoeff[32] = { 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3 };
static const int  kPrefixCtx[8] = { 0, 0, 0, 3, 6, 10, 15, 21 };
// g_log2SbbSize[log2 w][log2 h]: the coefficient groups' log2 width | log2 height << 4
static const int  kLog2SbbSize[49] = {
  0x00, 0x10, 0x20, 0x30, 0x40, 0x40, 0x40,
  0x01, 0x11, 0x11, 0x31, 0x31, 0x31, 0x31,
  0x02, 0x11, 0x22, 0x22, 0x22, 0x22, 0x22,
  0x03, 0x13, 0x22, 0x22, 0x22, 0x22, 0x22,
  0x04, 0x13, 0x22, 0x22, 0x22, 0x22, 0x22,
  0x04, 0x13, 0x22, 0x22, 0x22, 0x22, 0x22,
  0x04, 0x13, 0x22, 0x22, 0x22, 0x22, 0x22 };
static const int  kBcwParsingOrder[5] = { 2, 3, 1, 4, 0 };
static const int  kBcwInternFwd[5]    = { 1, 2, 0, 3, 4 };

// --------------------------------------------------------------------------------------------------------------------
// the workgroup's state
// --------------------------------------------------------------------------------------------------------------------

static int g_lane = 0;   // the thread's lane: the loops over a block's units give it every 32nd

// the picture's
static int  g_widthInCtus, g_heightInCtus, g_log2Ctu, g_ctuSize, g_width, g_height, g_widthIn4;
static int  g_chromaFormat, g_csx, g_csy, g_qpBdOffset, g_log2MinCb, g_maxTbSize, g_wpp, g_log2MaxTs, g_tools;
static int  g_cuSlots, g_tuSlots, g_tileCols, g_log2MaxTb;
// the slice's
static int  g_slice, g_sliceType, g_sliceQp, g_sliceFlags;
static bool g_dualTree;
// the CTU's
static int  g_ctu, g_ctuX, g_ctuY;
static int  g_numCus, g_numTus;    // its CUs and TUs so far
static int  g_dmvrOffset;          // DMVR's refinements' cache: the CTU's so far (CtuData::dmvrMvCacheOffset)
static int  g_csChromaQpAdj;       // CodingStructure::chromaQpAdj
static bool g_error;

bool tool( int t )
{
  return ( g_tools & t ) != 0;
}

bool sliceFlag( int f )
{
  return ( g_sliceFlags & f ) != 0;
}

groupshared int  s_pic[kPicWords];            // the picture's words
groupshared int  s_cqp[3 * 128];              // its chroma QP mappings
groupshared uint s_slice[kSliceWords];        // the slice's words
groupshared uint s_scanTab[961];              // the scans: scan position | its inverse (scan index of the raster position) << 16

// a value all threads hold alike, so marked for the compiler (scalar registers, scalar branches)
uint uni( uint v )
{
  return WaveReadLaneFirst( v );
}

int uni( int v )
{
  return WaveReadLaneFirst( v );
}

uint2 uni( uint2 v )
{
  return WaveReadLaneFirst( v );
}

int picWord( int w )
{
  return uni( s_pic[w] );
}

int sliceWord( int w )
{
  return int( uni( s_slice[w] ) );
}

// where the scan of a block 1 << log2W wide and 1 << log2H high begins in s_scanTab (log2 sizes 0..4)
int scanTabOffset( int log2W, int log2H )
{
  return ( ( 1 << log2W ) - 1 ) * 31 + ( 1 << log2W ) * ( ( 1 << log2H ) - 1 );
}

bool sliceIsIntra()
{
  return g_sliceType == I_SLICE;
}

int log2i( int v )
{
  return firstbithigh( uint( v ) );
}

// --------------------------------------------------------------------------------------------------------------------
// the arithmetic decoder (vvdec's BinDecoder) and the context models (BinProbModel)
// --------------------------------------------------------------------------------------------------------------------

groupshared uint2 s_ctx[CTX_COUNT];   // per context: its state (m_state[0] | m_state[1] << 16), its rates (m_rate0 | m_rate1 << 8)

static uint g_range;
static uint g_value;
static int  g_bitsNeeded;
static uint g_pos;       // the next byte in pc.bits
static uint g_end;       // the substream's end: bytes past it read as 0
static uint g_win;       // the window of 128 bytes of the slice data the threads hold, 4 each
static uint g_winBase;   // its first byte (a multiple of 128)

uint readByte()
{
  const uint p = g_pos;
  g_pos++;
  if( p >= g_end )
  {
    return 0u;
  }
  if( p - g_winBase >= 128u )
  {
    g_winBase = p & ~127u;
    g_win     = pc.bits[( g_winBase >> 2 ) + uint( g_lane )];
  }
  const uint word = WaveReadLaneAt( g_win, int( ( p - g_winBase ) >> 2 ) );
  return ( word >> ( ( p & 3u ) * 8u ) ) & 0xffu;
}

// BinDecoder::start
void startDecoder()
{
  g_winBase    = 0x80000000u;   // (far from any byte: the first read loads the window)
  g_range      = 510;
  g_value      = readByte() << 8;
  g_value     += readByte();
  g_bitsNeeded = -8;
}

uint decodeBin( int ctxId )
{
  PROF_COUNT( kProfBins, 1 );
  g_binCount++;
#if PROFILE
  PROF_BEGIN( binStart );
  const uint bin = decodeBinBody( ctxId );
  PROF_END( kProfDecode, binStart );
  return bin;
}

uint decodeBinBody( int ctxId )
{
#endif
  const uint2 cr   = uni( s_ctx[ctxId] );
  uint       s0    = cr.x & 0xffffu;
  uint       s1    = cr.x >> 16;
  const uint q     = ( s0 + s1 ) >> 8;
  uint       bin   = q >> 7;
  const uint lps   = ( ( ( ( q ^ ( bin != 0 ? 0xffu : 0u ) ) >> 2 ) * ( g_range >> 5 ) ) >> 1 ) + 4;
  uint       range = g_range - lps;
  const uint sr    = range << 7;
  if( g_value >= sr )
  {
    const uint numBits = 8u - uint( firstbithigh( lps ) );   // (m_RenormTable_32[lps >> 3]: lps from 4 to 255)
    g_value            = ( g_value - sr ) << numBits;
    range              = lps << numBits;
    bin ^= 1u;
    g_bitsNeeded += int( numBits );
  }
  else if( range < 256 )
  {
    g_value <<= 1;
    range <<= 1;
    g_bitsNeeded++;
  }
  g_range = range;
  if( g_bitsNeeded >= 0 )
  {
    g_value += readByte() << uint( g_bitsNeeded );
    g_bitsNeeded -= 8;
  }
  // BinProbModel::update
  const uint rate0 = cr.y & 0xffu;
  const uint rate1 = cr.y >> 8;
  const int  d0    = int( bin != 0 ? 0x7fffu : ( 0xffffu >> ( 16u - rate0 ) ) ) - int( s0 );
  const int  d1    = int( bin != 0 ? 0x7fffu : ( 0xffffu >> ( 16u - rate1 ) ) ) - int( s1 );
  s0               = uint( int( s0 ) + ( d0 >> rate0 ) * 32 ) & 0xffffu;
  s1               = uint( int( s1 ) + ( d1 >> rate1 ) * 2 ) & 0xffffu;
  s_ctx[ctxId].x   = s0 | ( s1 << 16 );
  return bin;
}

uint decodeBinEP()
{
  PROF_COUNT( kProfBinsEP, 1 );
  g_binCount++;
#if PROFILE
  PROF_BEGIN( epStart );
  const uint bin = decodeBinEPBody();
  PROF_END( kProfDecodeEP, epStart );
  return bin;
}

uint decodeBinEPBody()
{
#endif
  uint value = g_value << 1;
  if( ++g_bitsNeeded >= 0 )
  {
    value += readByte();
    g_bitsNeeded = -8;
  }
  uint       bin = 0;
  const uint sr  = g_range << 7;
  if( value >= sr )
  {
    value -= sr;
    bin = 1;
  }
  g_value = value;
  return bin;
}

// (as vvdec's decodeBinsEP and decodeAlignedBinsEP: the same bins, one at a time)
uint decodeBinsEP( int numBins )
{
  uint bins = 0;
  for( int i = 0; i < numBins; i++ )
  {
    bins = ( bins << 1 ) | decodeBinEP();
  }
  return bins;
}

uint decodeRemAbsEP( int goRicePar, int cutoff, int maxLog2TrDynamicRange )
{
  int        prefix    = 0;
  const int  maxPrefix = 32 - maxLog2TrDynamicRange;
  uint       codeWord  = 0;
  do
  {
    prefix++;
    codeWord = decodeBinEP();
  } while( codeWord != 0 && prefix < maxPrefix );
  prefix -= 1 - int( codeWord );
  int  length = goRicePar;
  uint offset;
  if( prefix < cutoff )
  {
    offset = uint( prefix ) << uint( goRicePar );
  }
  else
  {
    offset = ( ( ( 1u << uint( prefix - cutoff ) ) + uint( cutoff ) - 1u ) << uint( goRicePar ) );
    length += ( prefix == 32 - maxLog2TrDynamicRange ? maxLog2TrDynamicRange - goRicePar : prefix - cutoff );
  }
  return offset + decodeBinsEP( length );
}

uint decodeBinTrm()
{
  g_range -= 2;
  const uint sr = g_range << 7;
  if( g_value >= sr )
  {
    return 1;
  }
  if( g_range < 256 )
  {
    g_range += g_range;
    g_value += g_value;
    if( ++g_bitsNeeded == 0 )
    {
      g_value += readByte();
      g_bitsNeeded = -8;
    }
  }
  return 0;
}

// CABACReader::unary_max_symbol
int unaryMaxSymbol( int ctxId0, int ctxIdN, int maxSymbol )
{
  int onesRead = 0;
  while( onesRead < maxSymbol && decodeBin( onesRead == 0 ? ctxId0 : ctxIdN ) == 1 )
  {
    ++onesRead;
  }
  return onesRead;
}

// CABACReader::unary_max_eqprob
int unaryMaxEqprob( int maxSymbol )
{
  for( int k = 0; k < maxSymbol; k++ )
  {
    if( decodeBinEP() == 0 )
    {
      return k;
    }
  }
  return maxSymbol;
}

// CABACReader::exp_golomb_eqprob (count 0 in VVC)
int expGolombEqprob( int count )
{
  uint symbol = 0;
  uint bit    = 1;
  while( ( bit & ~( uint( count ) >> 5 ) ) != 0 )
  {
    bit = decodeBinEP();
    symbol += bit << uint( count );
    count++;
  }
  if( --count != 0 )
  {
    symbol += decodeBinsEP( count );
  }
  return int( symbol );
}

// g_tbMax: the threshold of truncated binary codes up to 256 symbols
int tbMax( int maxSymbol )
{
  return maxSymbol <= 1 ? 0 : firstbithigh( uint( maxSymbol ) );
}

// CABACReader::xReadTruncBinCode
int readTruncBinCode( int maxSymbol )
{
  int thresh;
  if( maxSymbol > 256 )
  {
    int threshVal = 1 << 8;
    thresh        = 8;
    while( threshVal <= maxSymbol )
    {
      thresh++;
      threshVal <<= 1;
    }
    thresh--;
  }
  else
  {
    thresh = tbMax( maxSymbol );
  }
  const int val    = 1 << thresh;
  const int b      = maxSymbol - val;
  int       symbol = int( decodeBinsEP( thresh ) );
  if( symbol >= val - b )
  {
    const int altSymbol = int( decodeBinEP() );
    symbol <<= 1;
    symbol += altSymbol;
    symbol -= ( val - b );
  }
  return symbol;
}

// the slice's initial contexts (BinDecoder::reset's Ctx::init, the host's)
void initContexts()
{
  const uint base = uint( g_slice * kSliceStride + kSliceWords );
  for( int i = g_lane; i < CTX_COUNT; i += 32 )
  {
    s_ctx[i] = uint2( pc.slices[base + uint( i )], pc.slices[base + uint( CTX_COUNT + i )] );
  }
  GroupMemoryBarrierWithGroupSync();
}

void loadContexts( int slot )
{
  for( int i = g_lane; i < CTX_COUNT; i += 32 )
  {
    s_ctx[i].x = pc.wppCtx[uint( slot * CTX_COUNT + i )];
  }
  GroupMemoryBarrierWithGroupSync();
}

void saveContexts( int slot )
{
  GroupMemoryBarrierWithGroupSync();
  for( int i = g_lane; i < CTX_COUNT; i += 32 )
  {
    pc.wppCtx[uint( slot * CTX_COUNT + i )] = s_ctx[i].x;
  }
}

#include "parsetree.slang"
#include "parsecu.slang"
#include "parsetu.slang"

// --------------------------------------------------------------------------------------------------------------------
// the substreams
// --------------------------------------------------------------------------------------------------------------------

groupshared int s_parsed;

// whether the CTU is parsed (its records, maps and contexts visible): another substream's, waited for a little while
// (it may be parsing it in this dispatch)
bool ctuParsed( int ctu )
{
  if( g_lane == 0 )
  {
    int spins = 0;
    while( pc.sync[2 + ctu].load( MemoryOrder.Relaxed ) == 0 && spins < ( 1 << 14 ) )
    {
      spins++;
    }
    s_parsed = spins < ( 1 << 14 ) ? 1 : 0;
  }
  AllMemoryBarrierWithGroupSync();
  return uni( s_parsed ) != 0;
}

void loadPicture()
{
  for( int i = g_lane; i < kPicWords; i += 32 )
  {
    s_pic[i] = pc.pic[i];
  }
  for( int i = g_lane; i < 3 * 128; i += 32 )
  {
    s_cqp[i] = pc.pic[kPicChromaQpMap + i];
  }
  for( int i = g_lane; i < 961; i += 32 )
  {
    s_scanTab[i] = pc.scanTab[i];
  }
  GroupMemoryBarrierWithGroupSync();
  g_widthInCtus  = picWord( kPicWidthInCtus );
  g_heightInCtus = picWord( kPicHeightInCtus );
  g_log2Ctu      = picWord( kPicLog2Ctu );
  g_ctuSize      = 1 << g_log2Ctu;
  g_width        = picWord( kPicWidth );
  g_height       = picWord( kPicHeight );
  g_widthIn4     = ( g_width + 3 ) >> 2;
  g_chromaFormat = picWord( kPicChromaFormat );
  g_csx          = g_chromaFormat == 1 || g_chromaFormat == 2 ? 1 : 0;
  g_csy          = g_chromaFormat == 1 ? 1 : 0;
  g_qpBdOffset   = picWord( kPicQpBdOffset );
  g_log2MinCb    = picWord( kPicLog2MinCb );
  g_maxTbSize    = picWord( kPicMaxTbSize );
  g_log2MaxTb    = firstbithigh( uint( g_maxTbSize ) );
  g_wpp          = picWord( kPicWpp );
  g_log2MaxTs    = picWord( kPicLog2MaxTs );
  g_tools        = picWord( kPicTools );
  g_cuSlots      = picWord( kPicCuSlots );
  g_tuSlots      = picWord( kPicTuSlots );
  g_tileCols     = picWord( kPicTileCols );
}

// the CTU's start: its neighbours' availability, its map cleared, the row above it (from the CTU above, when in the
// same slice and tile), the column left of it (the CTU before, the same then)
void beginCtu()
{
  const uint info = uni( pc.ctuInfo[g_ctu * 2 + 1] );
  g_tileFirstCol  = ( info & 0x10000u ) != 0;
  g_leftAvail     = g_ctuX > 0 && sameSliceTile( g_ctu - 1, g_ctu );
  g_aboveAvail    = g_ctuY > 0 && sameSliceTile( g_ctu - g_widthInCtus, g_ctu );
  if( g_aboveAvail )
  {
    AllMemoryBarrierWithGroupSync();   // (the rows the CTU above left, this workgroup's or one before)
  }
  for( int i = g_lane; i < 2048; i += 32 )
  {
    s_map[i] = uint2( 0, 0 );
  }
  for( int i = g_lane; i < 64; i += 32 )
  {
    s_above[i] = g_aboveAvail ? pc.bottomRows[( g_ctu - g_widthInCtus ) * 64 + i] : uint2( 0, 0 );
    if( !g_leftAvail )
    {
      s_left[i] = uint2( 0, 0 );
    }
  }
  GroupMemoryBarrierWithGroupSync();
}

// the CTU's end: its right column the next CTU's left, its bottom row for the CTU below
void endCtu()
{
  const int last = ( g_ctuSize >> 2 ) - 1;
  for( int i = g_lane; i < 64; i += 32 )
  {
    const int ch = i >> 5;
    const int j  = i & 31;
    pc.bottomRows[g_ctu * 64 + i] = s_map[ch * 1024 + last * 32 + j];
    s_left[i]                     = s_map[ch * 1024 + j * 32 + last];
  }
  GroupMemoryBarrierWithGroupSync();
}

// where the substream's parse is, for the next dispatch: before CTU k, or done
void saveState( uint* state, int k, int prevQp[2], bool done )
{
  GroupMemoryBarrierWithGroupSync();
  for( int i = g_lane; i < CTX_COUNT; i += 32 )
  {
    state[kStCtx + i] = s_ctx[i].x;
  }
  for( int i = g_lane; i < 64; i += 32 )
  {
    state[kStLeft + 2 * i]     = s_left[i].x;
    state[kStLeft + 2 * i + 1] = s_left[i].y;
  }
  if( g_lane == 0 )
  {
    state[kStK]          = uint( k );
    state[kStDone]       = done ? 1u : 0u;
    state[kStRange]      = g_range;
    state[kStValue]      = g_value;
    state[kStBitsNeeded] = uint( g_bitsNeeded );
    state[kStPos]        = g_pos;
    state[kStQp0]        = uint( prevQp[0] );
    state[kStQp1]        = uint( prevQp[1] );
    state[kStCqpAdj]     = uint( g_csChromaQpAdj );
  }
}

[shader( "compute" )]
[numthreads( 32, 1, 1 )]
void main( uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID )
{
  g_lane           = int( local.x );
  const int stream = int( group.x );
  if( stream >= pc.numStreams )
  {
    return;
  }
  uint* state = pc.state + stream * kStateWords;
  if( uni( state[kStDone] ) != 0 )
  {
    return;   // (parsed in a dispatch before)
  }
  if( g_lane == 0 )
  {
    pc.sync[0].add( 1u );
  }
  loadPicture();
  g_error = false;
#if PROFILE
  for( int i = 0; i < kProfWords; i++ )
  {
    g_prof[i] = 0;
  }
#endif
  clearResidualMemory();

  uint* st          = pc.streams + stream * kStreamWords;
  g_pos             = uni( st[kStOffset] );
  g_end             = g_pos + uni( st[kStSize] );
  const int first   = int( uni( st[kStFirstCtu] ) );
  const int count   = int( uni( st[kStCtus] ) );
  g_slice           = int( uni( st[kStSlice] ) );
  for( int i = g_lane; i < kSliceWords; i += 32 )
  {
    s_slice[i] = pc.slices[g_slice * kSliceStride + i];
  }
  GroupMemoryBarrierWithGroupSync();
  g_sliceType       = sliceWord( kSlType );
  g_sliceQp         = sliceWord( kSlQp );
  g_sliceFlags      = sliceWord( kSlFlags );
  g_dualTree        = sliceFlag( kSlDualTree );
  g_csChromaQpAdj   = 0;

  int prevQp[2] = { g_sliceQp, g_sliceQp };
  int k         = int( uni( state[kStK] ) );
  if( k > 0 )
  {
    // resumed: the slice's contexts (their rates) with the states the dispatch before left, its decoder, its QPs, the
    // column left of the CTU
    initContexts();
    for( int i = g_lane; i < CTX_COUNT; i += 32 )
    {
      s_ctx[i].x = state[kStCtx + i];
    }
    for( int i = g_lane; i < 64; i += 32 )
    {
      s_left[i] = uint2( state[kStLeft + 2 * i], state[kStLeft + 2 * i + 1] );
    }
    GroupMemoryBarrierWithGroupSync();
    g_range         = uni( state[kStRange] );
    g_value         = uni( state[kStValue] );
    g_bitsNeeded    = int( uni( state[kStBitsNeeded] ) );
    g_pos           = uni( state[kStPos] );
    g_winBase       = 0x80000000u;   // (far from any byte: the next read loads the window)
    prevQp[0]       = int( uni( state[kStQp0] ) );
    prevQp[1]       = int( uni( state[kStQp1] ) );
    g_csChromaQpAdj = int( uni( state[kStCqpAdj] ) );
  }
  g_binCount  = 0;
  bool paused = false;
  while( k < count && !g_error )
  {
    g_ctu             = int( uni( pc.ctuList[( first + k ) * 2] ) );
    const uint flags  = uni( pc.ctuList[( first + k ) * 2 + 1] ) & 0xffu;
    const int  slot   = int( uni( pc.ctuList[( first + k ) * 2 + 1] ) >> 8 );
    g_ctuX            = g_ctu % g_widthInCtus;
    g_ctuY            = g_ctu / g_widthInCtus;
    // the CTU above another substream's and not parsed: the substream pauses before the CTU (nothing of it changed)
    if( ( flags & ( kCtuSyncLoad | kCtuWaitAbove ) ) != 0 && !ctuParsed( g_ctu - g_widthInCtus ) )
    {
      paused = true;
      break;
    }
    if( k == 0 || ( flags & kCtuRestart ) != 0 )
    {
      // the substream's (or a tile's, a CTU row's) start: the slice's contexts, the arithmetic decoder (re)started
      initContexts();
      if( ( flags & kCtuSyncLoad ) != 0 )
      {
        loadContexts( slot - g_tileCols );
      }
      startDecoder();
      prevQp[0] = g_sliceQp;
      prevQp[1] = g_sliceQp;
    }

    PROF_BEGIN( ctuStart );
    beginCtu();
    codingTreeUnit( prevQp );
    endCtu();
    PROF_END( kProfCtu, ctuStart );

    if( ( flags & kCtuSyncSave ) != 0 )
    {
      saveContexts( slot );
    }
    if( ( flags & kCtuTerminate ) != 0 )
    {
      decodeBinTrm();   // end_of_subset_one_bit (1): the next CTU restarts the decoder at the next byte
    }
    // the CTU's out, then the CTU parsed (its records and maps visible to the substreams below)
    pc.ctuOut[g_ctu * kCtuOutWords + 0] = uint( g_numCus );
    pc.ctuOut[g_ctu * kCtuOutWords + 1] = uint( g_numTus );
    AllMemoryBarrierWithGroupSync();
    if( g_lane == 0 )
    {
      pc.sync[2 + g_ctu].store( 1u, MemoryOrder.Release );
    }
    k++;
    // the dispatch's share of the work parsed: the substream pauses after the CTU
    if( g_binCount >= uint( pc.binBudget ) && k < count )
    {
      paused = true;
      break;
    }
  }
#if PROFILE
  if( g_lane == 0 )
  {
    for( int i = 0; i < kProfWords; i++ )
    {
      if( g_prof[i] != 0 )
      {
        pc.prof[i].add( g_prof[i] );
      }
    }
  }
#endif
  if( g_error )
  {
    // given up: the rest of the substream's CTUs marked parsed all the same, for the rows that wait for them
    AllMemoryBarrierWithGroupSync();
    if( g_lane == 0 )
    {
      pc.sync[1].add( 1u );
      for( int j = k; j < count; j++ )
      {
        pc.sync[2 + int( pc.ctuList[( first + j ) * 2] )].store( 1u, MemoryOrder.Release );
      }
    }
  }
  saveState( state, k, prevQp, !paused || g_error );
}
