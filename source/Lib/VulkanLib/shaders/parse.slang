// The parse of the slice data (CABAC) on the GPU: vvdec's DecSlice::parseSlice and CABACReader ported to a compute
// shader. From each substream's bytes it produces what the CPU's parse leaves for the reconstruction:
// - the records of the CUs and TUs (GpuFrontEnd.h's CuRecord and TuRecord, laid out as writeCuRecords writes them
//   from the CPU's parse);
// - the coefficient levels, in planes laid out like the picture's planes, as CABACReader::residual_coding leaves them
//   in the picture: each coded block's levels over the rectangle that its TU's maxScanPosX / maxScanPosY describe.
//   With dependent quantization, a level q is stored as 2 * |q| - 1 when the quantizer's state was 2 or 3 and as
//   2 * |q| otherwise, with q's sign, as vvdec stores it;
// - each CTU's SAO and ALF parameters as the parse leaves them (SAO merges are not resolved).
//
// This file holds the layout of the host's tables, vvdec's constants, the arithmetic decoder and the loop over a
// substream's CTUs. It includes the rest of the parse: parsetree.slang (the coding tree, SAO and ALF), parsecu.slang
// (the coding units) and parsetu.slang (the transform units and the residual coding). parsectx.slang numbers the
// context models as vvdec does.
//
// One workgroup of 32 threads parses one substream: workgroup i takes substream i. A substream is a slice, a tile or,
// with WPP, a row of CTUs in a tile, as the entry points split the slice data. All 32 threads run every step of the
// parse on the same data and get the same results: on a SIMD unit, a value that one thread computes costs as much as
// when all 32 compute it. Only the loops over a block's 4x4 units or levels share out the work: each thread takes
// every 32nd item. The arithmetic decoder and the context models are vvdec's (BinDecoder, BinProbModel). The contexts
// live in shared memory; their initial states and rates come from the host, which computes them for each slice with
// vvdec's own Ctx::init. With WPP, a CTU row starts with the contexts the row above had after its first CTU, and each
// CTU waits until the CTU above it is parsed.
//
// The parse can stop and resume. A dispatch parses each substream until it has decoded pc.binBudget bins (each CU
// counts as 32 bins more), then stops at the end of that CTU, saves where it is (the kSt... words of pc.state), and
// the next dispatch resumes from there. A substream also pauses until the next dispatch when its next CTU needs the
// CTU above, which another substream parses, and that CTU is not parsed after a short wait. So no dispatch runs long,
// whatever the slice data (Windows resets the GPU when one dispatch keeps it busy for 2 s). The host (ParseService,
// or GpuRecon::runParse for the check) submits the dispatches, a few per submission, until every substream is parsed
// or none advances any more.
//
// Each CU's record and its TUs' records go wherever the picture's counters give them room (the staged records), each
// with a meta word that tells its CTU and its index among the CTU's CUs (or TUs); parsecompact.slang then copies them
// to their places in raster order of the CTUs. Neighbouring CUs are looked up in maps that hold, for each 4x4 luma
// block, a word pair describing the CU there, including its index in the CTU plus one (one map per channel type; see
// parsetree.slang). The parse also adds up the room the front end's job lists need (gpu::frontCapacities).

#include "common.slang"
#include "records.slang"
#include "parsectx.slang"

#ifndef PROFILE
#  define PROFILE 0
#endif

// READ_INVOCATION (1 unless the build says 0): laneValue reads a thread's value with OpSubgroupReadInvocationKHR, which
// needs the device's VK_EXT_shader_subgroup_ballot; parse_shuffle is built without it, for devices that lack it
#ifndef READ_INVOCATION
#  define READ_INVOCATION 1
#endif

struct Params
{
  // The slice data of the picture's slices, one after another, as bytes packed 4 per word. 256 zero bytes follow it,
  // so that loading a 128-byte window (readByte) never reads past the buffer.
  uint*         bits;
  uint*         streams;   // the substreams, kStreamWords words each
  // The CTUs of all substreams, each substream's in decoding order, 2 words per CTU: its raster-scan address; its
  // kCtu... flags | its WPP context slot << 8 (CTU row * tile columns + tile column)
  uint*         ctuList;
  // Per CTU in raster order, 2 words: the index of its slice; the index of its tile | 1 << 16 if the CTU is in its
  // tile's first column
  uint*         ctuInfo;
  // Per slice, kSliceWords parameter words, then its contexts' initial states and their rates (CTX_COUNT words each)
  uint*         slices;
  int*          pic;       // the picture's parameters (kPic...)
  // vvdec's diagonal scans (g_scanOrder[SCAN_UNGROUPED]) of the blocks 1, 2, 4, 8 and 16 samples wide and high, 961
  // words (GpuParse.h's kParseScanTab); loadPicture copies them into s_scanTab
  uint*         scanTab;
  uint*         cus;       // out: the staged CU records (32 words each), pc.cuCap of them at most
  uint*         tus;       // out: the staged TU records (16 words each), pc.tuCap of them at most
  uint*         ctuOut;    // out: kCtuOutWords words per CTU: its numbers of CUs and TUs, SAO's and ALF's parameters
  // Per CTU, the map entries (word pairs, see mapGet) of its bottom row of 4x4 luma blocks: 32 of the luma tree, then
  // 32 of the chroma tree. The CTU below reads them.
  uint2*        bottomRows;
  // out: the coefficient levels, each coded block's rectangle (writeLevels) with its rows one after the other, in the
  // substream's chunk of them (pc.counters[7] gives the chunks out); pc.levelCap levels at most
  int16_t*      levels;
  // [0] counts the substreams' runs (each workgroup that parses a substream in a dispatch adds 1), [1] counts the
  // substreams that gave up, and [2 + CTU] turns 1 once that CTU is parsed. The host zeroes it before the first
  // dispatch.
  Atomic<uint>* sync;
  uint*         wppCtx;    // WPP: the context states after the first CTU of a CTU row, CTX_COUNT words per slot
  int           numStreams;   // the number of substreams (the workgroups past it return at once)
  int           levelCap;  // the room for levels
  Atomic<uint>* prof;      // PROFILE: the cycles and counts of kProf..., summed over the workgroups
  uint*         state;     // per substream, kStateWords words: where its parse stands (the host zeroes it first)
  int           binBudget; // the bins a dispatch parses of a substream before it pauses at the end of a CTU
  // out: per staged CU record (and per staged TU record), its meta word: its CTU << kLocalBits | its index among the
  // CTU's CUs (or TUs)
  uint*         cuMeta;
  uint*         tuMeta;
  // [0] the CU records staged so far, [1] the TU records (counted on past the room: the host then parses the picture
  // again with more), [2..6] the room the front end's job lists need, added up over the CUs (gpu::frontCapacities
  // without its margins): residual jobs, inter tiles, intra jobs, intra bands, deblocking edges; [7] the levels
  // given out in chunks (counted on past the room too). Zeroed first.
  Atomic<uint>* counters;
  int           cuCap;     // the room for staged CU records
  int           tuCap;     // and for TU records
};

// The words of a substream's state between dispatches (pc.state)
static const int kStateWords   = 1024;  // the words per substream
static const int kStK          = 0;     // the next CTU to parse, as an index among the substream's CTUs in pc.ctuList
static const int kStDone       = 1;     // 1 once the substream is parsed (or has given up)
// The arithmetic decoder's state: g_range, g_value, g_bitsNeeded and g_pos (the next byte to read)
static const int kStRange      = 2;
static const int kStValue      = 3;
static const int kStBitsNeeded = 4;
static const int kStPos        = 5;
// The QPs carried from CTU to CTU (prevQp): the luma tree's (or the single tree's), then the dual tree chroma's
static const int kStQp0        = 6;
static const int kStQp1        = 7;
static const int kStCqpAdj     = 8;     // CodingStructure::chromaQpAdj
static const int kStLevelsAt   = 9;     // the substream's chunk of pc.levels: its next free level, its end
static const int kStLevelsEnd  = 10;
static const int kStCtx        = 16;    // the contexts' states, CTX_COUNT words (their rates are the slice's)
static const int kStLeft       = 400;   // s_left, the map entries of the column left of the next CTU (64 word pairs)

static uint g_binCount;   // the bins decoded in this dispatch, plus 32 for each CU parsed (compared with pc.binBudget)
// The substream's chunk of pc.levels, where writeLevels puts the blocks' levels: its next free level and its end. A
// chunk is kLevelChunk levels, at least a whole block's (32 x 32), taken from pc.counters[7] when the next block does
// not fit in what is left (the rest of the chunk stays unused).
static const uint kLevelChunk = 4096;
static uint       g_levelsAt;
static uint       g_levelsEnd;
// The room the front end's job lists need for the CUs parsed in this dispatch (pc.counters[2..6]'s order), added to
// pc.counters at the dispatch's end
static uint g_caps[5];

// PROFILE: the parts of the parse that are timed (in cycles of the subgroup's clock), then the things counted
static const int kProfCtu      = 0;    // whole CTUs
static const int kProfCu       = 1;    // coding_unit (its prediction data and residuals)
static const int kProfResidual = 2;    // residual_coding, including the writing of its levels
static const int kProfLevels   = 3;    // the writing of the levels
static const int kProfRecords  = 4;    // the CU's and its TUs' records, and the CU's entries in the maps
static const int kProfSao      = 5;    // SAO's and ALF's syntax
static const int kProfDecode   = 6;    // the time in decodeBin
static const int kProfDecodeEP = 7;    // the time in decodeBinEP
static const int kProfBins     = 8;    // counts: the context-coded bins
static const int kProfBinsEP   = 9;    // the bypass bins
static const int kProfCus      = 10;   // the CUs
static const int kProfTus      = 11;   // the TUs
static const int kProfBlocks   = 12;   // the coded blocks (residual_coding calls)
static const int kProfLevelsN  = 13;   // the nonzero levels
static const int kProfDispatch = 14;   // whole dispatches (each substream's, from its start to its state saved)
static const int kProfStart    = 15;   // their starts: the picture, the slice, the contexts, the state restored
static const int kProfWords    = 16;   // the size of the table of counters

// PROF_BEGIN starts timing a part, PROF_END adds the part's time to counter k, and PROF_COUNT adds v to counter k.
// The long parts take the device's real-time clock (units of 10 ns on AMD); decodeBin and decodeBinEP take the
// subgroup's cycle clock (PROF_BEGIN_CYCLES, PROF_END_CYCLES), which on RDNA counts only 20 bits of cycles (~0.4 ms).
// The workgroup sums its counters in g_prof and adds them to pc.prof at the end. Without PROFILE the macros vanish.
#if PROFILE
static uint g_prof[kProfWords];
uint profClock()
{
  return getRealtimeClock().x;
}
uint profCycles()
{
  return clock2x32ARB().x;
}
void profAdd( int k, uint v )
{
  g_prof[k] += v;
}
#  define PROF_BEGIN( name ) const uint name = profClock()
#  define PROF_END( k, name ) profAdd( k, profClock() - name )
#  define PROF_BEGIN_CYCLES( name ) const uint name = profCycles()
#  define PROF_END_CYCLES( k, name ) profAdd( k, ( profCycles() - name ) & 0xfffffu )
#  define PROF_COUNT( k, v ) profAdd( k, uint( v ) )
#else
#  define PROF_BEGIN( name )
#  define PROF_END( k, name )
#  define PROF_BEGIN_CYCLES( name )
#  define PROF_END_CYCLES( k, name )
#  define PROF_COUNT( k, v )
#endif

[[vk::push_constant]] Params pc;

// --------------------------------------------------------------------------------------------------------------------
// The tables the host writes (GpuParse.cpp's buildParseInput, which uses the same layout)
// --------------------------------------------------------------------------------------------------------------------

// The picture's parameters (pc.pic)
static const int kPicWidthInCtus  = 0;
static const int kPicHeightInCtus = 1;
static const int kPicLog2Ctu      = 2;    // log2 of the CTU size, in luma samples
static const int kPicWidth        = 3;    // the picture's size, in luma samples
static const int kPicHeight       = 4;
static const int kPicChromaFormat = 5;    // vvdec's ChromaFormat: 0 4:0:0, 1 4:2:0, 2 4:2:2, 3 4:4:4
static const int kPicQpBdOffset   = 6;    // the SPS's QpBdOffset: 6 * (bit depth - 8)
static const int kPicLog2MinCb    = 7;    // log2 of the smallest coding block, in luma samples
static const int kPicMaxTbSize    = 8;    // the largest transform block, in luma samples
static const int kPicWpp          = 9;    // 1 with entropy coding sync (WPP)
static const int kPicLog2MaxTs    = 10;   // log2 of the largest transform skip block
static const int kPicTools        = 11;   // the SPS's tools: kTool...
static const int kPicMaxMergeCand = 12;   // the SPS's numbers of merge candidates: regular, GPM, IBC
static const int kPicMaxGeoCand   = 13;
static const int kPicMaxIbcCand   = 14;
static const int kPicUseDqp       = 15;   // the PPS's cu_qp_delta_enabled_flag
static const int kPicCqpListLen   = 16;   // the length of the PPS's chroma QP offset list
static const int kPicTrRange      = 17;   // the SPS's maxLog2TrDynamicRange: luma | chroma << 8
static const int kPicCuSlots      = 18;   // CU record slots per CTU
static const int kPicTuSlots      = 19;   // TU record slots per CTU
static const int kPicTileCols     = 20;   // the number of tile columns
// The PPS's chroma QP offsets (Cb, Cr, joint Cb-Cr), then the entries 1..6 of its chroma QP offset list, 3 words each
static const int kPicCqpOffsets   = 24;
static const int kPicWords        = 48;   // the words loadPicture copies into s_pic: all the parameters above
// The SPS's chroma QP mapping tables (Cb, Cr, joint Cb-Cr), 128 words each, indexed by qp + QpBdOffset
static const int kPicChromaQpMap  = 160;

// The SPS's tools, one bit each (kPicTools)
static const int kToolSbt        = 1 << 0;    // subblock transform
static const int kToolAffine     = 1 << 1;
static const int kToolAffineType = 1 << 2;    // the 6-parameter affine model
static const int kToolAmvr       = 1 << 3;
static const int kToolAffineAmvr = 1 << 4;
static const int kToolBcw        = 1 << 5;
static const int kToolCiip       = 1 << 6;
static const int kToolGeo        = 1 << 7;    // GPM
static const int kToolMmvd       = 1 << 8;
static const int kToolSmvd       = 1 << 9;
static const int kToolIsp        = 1 << 10;
static const int kToolMrl        = 1 << 11;   // multiple reference lines
static const int kToolMip        = 1 << 12;
static const int kToolLfnst      = 1 << 13;
static const int kToolMtsIntra   = 1 << 14;   // explicit MTS in intra CUs
static const int kToolMtsInter   = 1 << 15;   // explicit MTS in inter CUs
static const int kToolLmChroma   = 1 << 16;   // CCLM
static const int kToolIbc        = 1 << 17;
static const int kToolAct        = 1 << 18;   // the adaptive colour transform
static const int kToolBdpcm      = 1 << 19;
static const int kToolTs         = 1 << 20;   // transform skip
static const int kToolJointCbCr  = 1 << 21;
static const int kToolMts        = 1 << 22;   // sps_mts_enabled_flag (SPS::getUseMTS)

// A slice's parameters (pc.slices)
static const int kSliceWords     = 16;   // the parameter words; the slice's initial contexts follow them
static const int kSliceStride    = kSliceWords + 2 * CTX_COUNT;   // the words per slice
static const int kSlType         = 0;    // vvdec's SliceType: 0 B, 1 P, 2 I
static const int kSlQp           = 1;    // the slice QP
static const int kSlFlags        = 2;    // the kSl... flags below
static const int kSlNumRef       = 3;    // the number of reference indices: L0 | L1 << 8
static const int kSlSymRef       = 4;    // SMVD's reference indices: L0 | L1 << 8 (signed bytes)
static const int kSlMaxAffMerge  = 5;    // the picture header's MaxNumAffineMergeCand
// ALF: the number of the slice's luma APSs | the number of chroma alternative filters << 8 | the number of CC-ALF
// filters for Cb << 16 | for Cr << 24
static const int kSlAlf          = 6;
static const int kSlSubdiv       = 7;    // cu_qp_delta_subdiv | cu_chroma_qp_offset_subdiv << 8
static const int kSlChromaQpDelta = 8;   // the slice's chroma QP deltas of Cb, Cr and joint Cb-Cr (3 words)
// Weighted prediction: a bit per reference index whose weights are present, list 0's from bit 0, list 1's from bit 16
static const int kSlWp           = 11;
// The partition limits (Partitioner::initCtu's): maxBTD | maxBtSize << 8 | maxTtSize << 16 | minQtSize << 24;
// kSlPartLuma holds the luma tree's (or the single tree's), kSlPartChroma the dual tree chroma's
static const int kSlPartLuma     = 12;
static const int kSlPartChroma   = 13;

// The slice's flags (kSlFlags)
static const int kSlSaoLuma      = 1 << 0;
static const int kSlSaoChroma    = 1 << 1;
static const int kSlAlfY         = 1 << 2;
static const int kSlAlfCb        = 1 << 3;
static const int kSlAlfCr        = 1 << 4;
static const int kSlCcAlfCb      = 1 << 5;
static const int kSlCcAlfCr      = 1 << 6;
static const int kSlDepQuant     = 1 << 7;   // dependent quantization
static const int kSlSignHiding   = 1 << 8;   // sign data hiding
static const int kSlTsResOff     = 1 << 9;   // slice_ts_residual_coding_disabled_flag
static const int kSlChromaQpAdj  = 1 << 10;  // cu_chroma_qp_offset_enabled_flag
static const int kSlMvdL1Zero    = 1 << 11;  // the picture header's mvd_l1_zero_flag
static const int kSlBiDirPred    = 1 << 12;  // Slice::getBiDirPred: the slice has the references SMVD needs
static const int kSlCheckLDC     = 1 << 13;  // Slice::getCheckLDC: no reference follows the picture (BCW has 5 weights)
static const int kSlDualTree     = 1 << 14;   // an intra slice with the SPS's dual tree

// A substream's words (pc.streams)
static const int kStreamWords    = 8;
static const int kStOffset       = 0;    // its first byte in pc.bits
static const int kStSize         = 1;    // its size in bytes
static const int kStFirstCtu     = 2;    // its first CTU in pc.ctuList
static const int kStCtus         = 3;    // its number of CTUs
static const int kStSlice        = 4;    // the index of its slice

// The flags of a CTU in pc.ctuList
// kCtuRestart: the contexts are initialised again and the arithmetic decoder restarts at this CTU, the first of a
// tile or, with WPP, of a CTU row in its tile. The host does not set it on a slice's first CTU; a substream's first
// CTU always starts this way.
static const uint kCtuRestart    = 1;
// kCtuSyncLoad: WPP, with the restart: load the contexts that the row above saved, from slot - tile columns
static const uint kCtuSyncLoad   = 2;
static const uint kCtuSyncSave   = 4;    // WPP: save the contexts after this CTU (a row's first), in its slot
// kCtuTerminate: this CTU ends a tile or a WPP row, but the slice data goes on in the same substream (the slice has
// no entry points): decode end_of_subset_one_bit after it
static const uint kCtuTerminate  = 8;
// kCtuWaitAbove: WPP, and the CTU above is in the same slice and tile: wait until it is parsed (another substream
// parses it when the slice has entry points)
static const uint kCtuWaitAbove  = 16;

// The words per CTU in pc.ctuOut: 0 its number of CUs, 1 its number of TUs, 2..7 SAO's parameters (2 words per
// component), 8 ALF's; the rest are unused
static const int kCtuOutWords    = 16;

// The CU records' flags (GpuFrontEnd.h's CuRecord::Flags)
static const uint kMerge   = 1;
static const uint kSkip    = 2;
static const uint kMmvd    = 4;
static const uint kAffine  = 8;
static const uint kGeo     = 16;
static const uint kCiip    = 32;
static const uint kChroma  = 64;    // the CU belongs to the chroma tree
static const uint kAffine6 = 128;   // the 6-parameter affine model

// --------------------------------------------------------------------------------------------------------------------
// vvdec's constants and tables, under vvdec's names
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

// g_uiGroupIdx: the group (the value of last_sig_coeff_x/y_prefix) of each last position
static const int  kGroupIdx[64] = { 0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
                                    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11 };
// g_uiMinInGroup: the first position of each group
static const int  kMinInGroup[14] = { 0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96 };
// g_auiGoRiceParsCoeff: the Rice parameter for each template sum (templateAbsSum)
static const int  kGoRiceParsCoeff[32] = { 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3 };
// CoeffCodingContext's prefix_ctx: the first context of a luma block's last position prefix, by the log2 size
static const int  kPrefixCtx[8] = { 0, 0, 0, 3, 6, 10, 15, 21 };
// g_log2SbbSize: the coefficient groups' log2 width | log2 height << 4, at [log2 w * 7 + log2 h] for the blocks 1 to
// 64 samples wide and high
static const int  kLog2SbbSize[49] = {
  0x00, 0x10, 0x20, 0x30, 0x40, 0x40, 0x40,
  0x01, 0x11, 0x11, 0x31, 0x31, 0x31, 0x31,
  0x02, 0x11, 0x22, 0x22, 0x22, 0x22, 0x22,
  0x03, 0x13, 0x22, 0x22, 0x22, 0x22, 0x22,
  0x04, 0x13, 0x22, 0x22, 0x22, 0x22, 0x22,
  0x04, 0x13, 0x22, 0x22, 0x22, 0x22, 0x22,
  0x04, 0x13, 0x22, 0x22, 0x22, 0x22, 0x22 };
// g_BcwParsingOrder and g_BcwInternFwd: from the parsed BCW index to the index vvdec stores
static const int  kBcwParsingOrder[5] = { 2, 3, 1, 4, 0 };
static const int  kBcwInternFwd[5]    = { 1, 2, 0, 3, 4 };

// --------------------------------------------------------------------------------------------------------------------
// The workgroup's state
// --------------------------------------------------------------------------------------------------------------------

static int g_lane = 0;   // the thread's lane (0..31); loops over a block's units take every 32nd unit from it

// The picture's parameters (loadPicture); sizes in luma samples, g_csx and g_csy the chroma subsampling shifts
static int  g_widthInCtus, g_log2Ctu, g_ctuSize, g_width, g_height;
static int  g_chromaFormat, g_csx, g_csy, g_qpBdOffset, g_log2MinCb, g_maxTbSize, g_log2MaxTs, g_tools;
static int  g_cuSlots, g_tuSlots, g_tileCols, g_log2MaxTb;
// The substream's slice: its index, type, QP and flags (kSl...), and whether it uses the dual tree
static int  g_slice, g_sliceType, g_sliceQp, g_sliceFlags;
static bool g_dualTree;
// The current CTU: its raster-scan address and its position in CTUs
static int  g_ctu, g_ctuX, g_ctuY;
static int  g_numCus, g_numTus;    // the CUs and TUs the CTU has so far (its next free slots)
static int  g_dmvrOffset;          // the DMVR refinements the CTU's CUs reserved so far (CtuData::dmvrMvCacheOffset)
static int  g_csChromaQpAdj;       // CodingStructure::chromaQpAdj
static bool g_error;               // the substream gave up: slots or a stack were full, or a block was off the planes

// LITE=1: the variant for pictures that leave off the tools below (GpuParse's ParseInput::lite): their syntax is
// compiled out, which takes a large part of the code away and keeps more of the rest in the instruction cache.
#ifndef LITE
#  define LITE 0
#endif
static const int kToolsLeftOut = LITE != 0 ? ( kToolSbt | kToolAffine | kToolAffineType | kToolAffineAmvr | kToolBcw | kToolCiip | kToolGeo
                                               | kToolMmvd | kToolSmvd | kToolIsp | kToolMrl | kToolMip | kToolLfnst | kToolMtsIntra
                                               | kToolMtsInter | kToolAct | kToolJointCbCr )
                                           : 0;
static const int kSliceFlagsLeftOut = LITE != 0 ? kSlDepQuant : 0;

// Whether the SPS enables tool t (kTool...)
bool tool( int t )
{
  return ( t & kToolsLeftOut ) == 0 && ( g_tools & t ) != 0;
}

// Whether the slice has flag f (kSl...)
bool sliceFlag( int f )
{
  return ( f & kSliceFlagsLeftOut ) == 0 && ( g_sliceFlags & f ) != 0;
}

groupshared int  s_pic[kPicWords];            // the picture's parameters (the first kPicWords words of pc.pic)
groupshared int  s_cqp[3 * 128];              // the picture's chroma QP mapping tables (kPicChromaQpMap)
groupshared uint s_slice[kSliceWords];        // the slice's parameter words
// The scan table (pc.scanTab): from where scanTabOffset says for each block size, word k holds the raster position of
// scan position k | the scan position of raster position k << 16
groupshared uint s_scanTab[961];

// Marks a value that all threads hold alike as uniform, so that the compiler can keep it in a scalar register and
// branch on it with scalar branches
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

// The value v has in thread `lane`, which all threads name alike. The compiler must know that the value is the same in
// all threads: the arithmetic decoder works out its registers, the bins and all the bins decide from the bytes this
// reads, and these then run as scalar code. OpSubgroupReadInvocationKHR does that in one v_readlane on AMD. Vulkan
// 1.1's way to read a thread chosen at run time, WaveReadLaneAt, is a shuffle: on AMD a round trip through the LDS
// unit, whose result the compiler takes as different in each thread, so uni() has to follow it.
uint laneValue( uint v, uint lane )
{
#if READ_INVOCATION
  const uint l = uni( lane );
  return spirv_asm
  {
    OpExtension "SPV_KHR_shader_ballot";
    OpCapability SubgroupBallotKHR;
    result:$$uint = OpSubgroupReadInvocationKHR $v $l
  };
#else
  return uni( WaveReadLaneAt( v, int( lane ) ) );
#endif
}

int laneValue( int v, int lane )
{
  return int( laneValue( uint( v ), uint( lane ) ) );
}

// A lane array holds up to 32 values in one register, value k in thread k: laneValue( a, k ) reads value k (in one
// v_readlane, where shared memory takes a round trip), and setLane( a, k, v ) gives the array with value k set to v
int setLane( int a, int k, int v )
{
  return g_lane == k ? v : a;
}

// Word w of the picture's parameters
int picWord( int w )
{
  return uni( s_pic[w] );
}

// Word w of the slice's parameters
int sliceWord( int w )
{
  return int( uni( s_slice[w] ) );
}

// Where the scan of a block 1 << log2W wide and 1 << log2H high starts in s_scanTab (log2 sizes 0..4). The blocks
// follow one another by width, and within a width by height; GpuParse.cpp lays the table out the same way.
int scanTabOffset( int log2W, int log2H )
{
  return ( ( 1 << log2W ) - 1 ) * 31 + ( 1 << log2W ) * ( ( 1 << log2H ) - 1 );
}

bool sliceIsIntra()
{
  return g_sliceType == I_SLICE;
}

// floor( log2( v ) ) for v > 0
int log2i( int v )
{
  return firstbithigh( uint( v ) );
}

// --------------------------------------------------------------------------------------------------------------------
// The arithmetic decoder (vvdec's BinDecoder) and the context models (BinProbModel)
// --------------------------------------------------------------------------------------------------------------------

// Per context model: its two probability states (m_state[0] | m_state[1] << 16) and its two adaptation rates
// (m_rate0 | m_rate1 << 8)
groupshared uint2 s_ctx[CTX_COUNT];

// The arithmetic decoder's registers, as in BinDecoder (m_Range, m_Value, m_bitsNeeded)
static uint g_range;
static uint g_value;
static int  g_bitsNeeded;
static uint g_pos;       // the next byte to read in pc.bits
static uint g_end;       // the end of the substream's bytes: bytes past it read as 0
static uint g_win;       // a window of 128 bytes of the slice data, spread over the threads: each holds 4 bytes
static uint g_winBase;   // the window's first byte (a multiple of 128)

// Reads the next byte of the substream. The 32 threads load a 128-byte window together, one word each, and all take
// the byte from the thread that holds it (laneValue), so the slice data is read from memory once per 128 bytes.
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
  const uint word = laneValue( g_win, ( p - g_winBase ) >> 2 );
  return ( word >> ( ( p & 3u ) * 8u ) ) & 0xffu;
}

// Starts the arithmetic decoder at the next byte (BinDecoder::start)
void startDecoder()
{
  g_winBase    = 0x80000000u;   // (far from every byte, so the first read loads the window)
  g_range      = 510;
  g_value      = readByte() << 8;
  g_value     += readByte();
  g_bitsNeeded = -8;
}

// Context model ctxId's two probability states and adaptation rates (s_ctx's word pair). Fetching the models of the
// next bins together, before the first of them is decoded, waits for shared memory once.
uint2 contextModel( int ctxId )
{
  return uni( s_ctx[ctxId] );
}

// Decodes a bin with context model ctxId, whose states and rates contextModel fetched as cr, and updates the model
// (BinDecoder::decodeBin). With PROFILE, decodeBinWith times decodeBinBody; without it, the #if folds the body into
// decodeBinWith. decodeBinEP below is built the same way.
uint decodeBinWith( int ctxId, uint2 cr )
{
  PROF_COUNT( kProfBins, 1 );
  g_binCount++;
#if PROFILE
  PROF_BEGIN_CYCLES( binStart );
  const uint bin = decodeBinBody( ctxId, cr );
  PROF_END_CYCLES( kProfDecode, binStart );
  return bin;
}

uint decodeBinBody( int ctxId, uint2 cr )
{
#endif
  uint       s0    = cr.x & 0xffffu;
  uint       s1    = cr.x >> 16;
  const uint q     = ( s0 + s1 ) >> 8;
  uint       bin   = q >> 7;
  const uint lps   = ( ( ( ( q ^ ( bin != 0 ? 0xffu : 0u ) ) >> 2 ) * ( g_range >> 5 ) ) >> 1 ) + 4;
  const uint range = g_range - lps;
  const uint sr    = range << 7;
  if( g_value < sr )
  {
    // The most probable symbol: renormalized by a bit when range is below 256 (it is 128 to 510). (A shift of 0 or 1
    // rather than an if: the compiler's code for an if here took a detour through vector registers.)
    const uint numBits = ( range >> 8 ) ^ 1u;
    g_value <<= numBits;
    g_range = range << numBits;
    g_bitsNeeded += int( numBits );
  }
  else
  {
    const uint numBits = 8u - uint( firstbithigh( lps ) );   // (= m_RenormTable_32[lps >> 3], as lps is 4 to 255)
    g_value            = ( g_value - sr ) << numBits;
    g_range            = lps << numBits;
    bin ^= 1u;
    g_bitsNeeded += int( numBits );
  }
  if( g_bitsNeeded >= 0 )
  {
    g_value += readByte() << uint( g_bitsNeeded );
    g_bitsNeeded -= 8;
  }
  // Update the context model's two states towards the bin (BinProbModel::update)
  const uint rate0 = cr.y & 0xffu;
  const uint rate1 = cr.y >> 8;
  const int  d0    = int( bin != 0 ? 0x7fffu : ( 0xffffu >> ( 16u - rate0 ) ) ) - int( s0 );
  const int  d1    = int( bin != 0 ? 0x7fffu : ( 0xffffu >> ( 16u - rate1 ) ) ) - int( s1 );
  s0               = uint( int( s0 ) + ( d0 >> rate0 ) * 32 ) & 0xffffu;
  s1               = uint( int( s1 ) + ( d1 >> rate1 ) * 2 ) & 0xffffu;
  s_ctx[ctxId].x   = s0 | ( s1 << 16 );
  return bin;
}

// Decodes a bin with context model ctxId (decodeBinWith, the model fetched first)
uint decodeBin( int ctxId )
{
  return decodeBinWith( ctxId, contextModel( ctxId ) );
}

// Decodes a bypass bin, one of equal probabilities (BinDecoder::decodeBinEP)
uint decodeBinEP()
{
  PROF_COUNT( kProfBinsEP, 1 );
  g_binCount++;
#if PROFILE
  PROF_BEGIN_CYCLES( epStart );
  const uint bin = decodeBinEPBody();
  PROF_END_CYCLES( kProfDecodeEP, epStart );
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

// Decodes numBins bypass bins, the first into the highest bit (BinDecoder::decodeBinsEP; its decodeAlignedBinsEP gives
// the same bins): while more than 8 are left, a byte is read and 8 bins come out of it; then the rest, after at most
// one more byte read. Each bin compares the value with the range scaled to the bin's position, and a bin of 1 takes
// that off the value.
uint decodeBinsEP( int numBins )
{
  PROF_COUNT( kProfBinsEP, numBins );
  g_binCount += uint( numBins );
  uint bins = 0;
  int  left = numBins;
  while( left > 8 )
  {
    g_value          = ( g_value << 8 ) + ( readByte() << uint( 8 + g_bitsNeeded ) );
    uint scaledRange = g_range << 15;
    for( int i = 0; i < 8; i++ )
    {
      scaledRange >>= 1;
      const bool one = g_value >= scaledRange;
      bins           = ( bins << 1 ) | ( one ? 1u : 0u );
      g_value       -= one ? scaledRange : 0u;
    }
    left -= 8;
  }
  g_bitsNeeded += left;
  g_value <<= uint( left );
  if( g_bitsNeeded >= 0 )
  {
    g_value += readByte() << uint( g_bitsNeeded );
    g_bitsNeeded -= 8;
  }
  uint scaledRange = g_range << uint( left + 7 );
  for( int i = 0; i < left; i++ )
  {
    scaledRange >>= 1;
    const bool one = g_value >= scaledRange;
    bins           = ( bins << 1 ) | ( one ? 1u : 0u );
    g_value       -= one ? scaledRange : 0u;
  }
  return bins;
}

// Decodes a remainder coded in bypass bins (BinDecoder::decodeRemAbsEP): a unary prefix, then a Rice code with the
// parameter goRicePar while the prefix is below cutoff, an Exp-Golomb style suffix beyond it
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

// Decodes a terminating bin (BinDecoder::decodeBinTrm)
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

// Decodes a truncated unary code of up to maxSymbol, its first bin with context ctxId0 and the others with ctxIdN
// (CABACReader::unary_max_symbol)
int unaryMaxSymbol( int ctxId0, int ctxIdN, int maxSymbol )
{
  int onesRead = 0;
  while( onesRead < maxSymbol && decodeBin( onesRead == 0 ? ctxId0 : ctxIdN ) == 1 )
  {
    ++onesRead;
  }
  return onesRead;
}

// Decodes a truncated unary code of up to maxSymbol in bypass bins (CABACReader::unary_max_eqprob)
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

// Decodes an Exp-Golomb code of order count in bypass bins (CABACReader::exp_golomb_eqprob). The parse uses it only
// with count 0, for cu_qp_delta_abs.
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

// vvdec's g_tbMax table, for truncated binary codes of up to 256 symbols: floor( log2( maxSymbol ) ), the length of
// the short codewords
int tbMax( int maxSymbol )
{
  return maxSymbol <= 1 ? 0 : firstbithigh( uint( maxSymbol ) );
}

// Decodes a truncated binary code of maxSymbol symbols in bypass bins (CABACReader::xReadTruncBinCode)
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

// Loads the slice's initial context states and rates into s_ctx. The host computed them with vvdec's Ctx::init, as
// BinDecoder::reset does.
void initContexts()
{
  const uint base = uint( g_slice * kSliceStride + kSliceWords );
  for( int i = g_lane; i < CTX_COUNT; i += 32 )
  {
    s_ctx[i] = uint2( pc.slices[base + uint( i )], pc.slices[base + uint( CTX_COUNT + i )] );
  }
  GroupMemoryBarrierWithGroupSync();
}

// WPP: replaces the context states with those saved in slot (the rates stay the slice's). Another substream's
// workgroup saved them, so they are read as common.slang's coherentLoad reads such words.
void loadContexts( int slot )
{
  for( int i = g_lane; i < CTX_COUNT; i += 32 )
  {
    const uint at = uint( slot * CTX_COUNT + i );
    s_ctx[i].x    = kCoherentReads ? coherentLoad( (uint64_t)pc.wppCtx + uint64_t( at ) * 4 ) : pc.wppCtx[at];
  }
  GroupMemoryBarrierWithGroupSync();
}

// WPP: saves the context states in slot, for the next CTU row
void saveContexts( int slot )
{
  GroupMemoryBarrierWithGroupSync();
  for( int i = g_lane; i < CTX_COUNT; i += 32 )
  {
    pc.wppCtx[uint( slot * CTX_COUNT + i )] = s_ctx[i].x;
  }
}

// The syntax: the coding tree, SAO and ALF; the coding units; the transform units and the residual coding
#include "parsetree.slang"
#include "parsecu.slang"
#include "parsetu.slang"

// --------------------------------------------------------------------------------------------------------------------
// The substreams
// --------------------------------------------------------------------------------------------------------------------

groupshared int s_parsed;   // ctuParsed's answer, which thread 0 shares with the others

// How many times ctuParsed polls: about as long as a CTU takes to parse. A wave that waits longer keeps issuing loads
// and instructions that the reconstruction's passes on the other queues need; the substream resumes in the next
// dispatch instead.
static const int kCtuParsedPolls = 1 << 7;

// Whether another substream's CTU is parsed, so that its records, map entries, ctuOut words and saved contexts can be
// read. Thread 0 polls the CTU's flag for a short while, as the other substream may be parsing the CTU in this very
// dispatch.
bool ctuParsed( int ctu )
{
  if( g_lane == 0 )
  {
    int spins = 0;
    while( pollCounter( pc.sync + 2 + ctu ) == 0 && spins < kCtuParsedPolls )
    {
      spins++;
    }
    s_parsed = spins < kCtuParsedPolls ? 1 : 0;
  }
  AllMemoryBarrierWithGroupSync();
  return uni( s_parsed ) != 0;
}

// Copies the picture's parameters, its chroma QP mapping tables and the scan table into shared memory, and reads the
// parameters the parse uses most into the g_... variables
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
  g_log2Ctu      = picWord( kPicLog2Ctu );
  g_ctuSize      = 1 << g_log2Ctu;
  g_width        = picWord( kPicWidth );
  g_height       = picWord( kPicHeight );
  g_chromaFormat = picWord( kPicChromaFormat );
  g_csx          = g_chromaFormat == 1 || g_chromaFormat == 2 ? 1 : 0;
  g_csy          = g_chromaFormat == 1 ? 1 : 0;
  g_qpBdOffset   = picWord( kPicQpBdOffset );
  g_log2MinCb    = picWord( kPicLog2MinCb );
  g_maxTbSize    = picWord( kPicMaxTbSize );
  g_log2MaxTb    = firstbithigh( uint( g_maxTbSize ) );
  g_log2MaxTs    = picWord( kPicLog2MaxTs );
  g_tools        = picWord( kPicTools );
  g_cuSlots      = picWord( kPicCuSlots );
  g_tuSlots      = picWord( kPicTuSlots );
  g_tileCols     = picWord( kPicTileCols );
}

// Prepares the neighbours for the CTU. It works out whether the CTUs to the left and above are in the same slice and
// tile, fills s_above from the bottom rows the CTU above left (zeros when it is out of reach), and clears s_left
// unless the CTU to the left is in reach: s_left then still holds that CTU's right column.
void beginCtu()
{
  const uint info = uni( pc.ctuInfo[g_ctu * 2 + 1] );
  g_tileFirstCol  = ( info & 0x10000u ) != 0;
  g_leftAvail     = g_ctuX > 0 && sameSliceTile( g_ctu - 1, g_ctu );
  g_aboveAvail    = g_ctuY > 0 && sameSliceTile( g_ctu - g_widthInCtus, g_ctu );
  if( g_aboveAvail )
  {
    // (makes visible the bottom rows of the CTU above, which this workgroup, another substream's workgroup or a
    // workgroup of an earlier dispatch wrote)
    AllMemoryBarrierWithGroupSync();
  }
  for( int i = g_lane; i < 2 * kMapRow; i += 32 )
  {
    // (with kCoherentReads, each word as coherentLoad reads it: another substream's workgroup may have written them)
    const uint64_t at = (uint64_t)pc.bottomRows + uint64_t( g_aboveAvail ? bottomRowsAt( g_ctu - g_widthInCtus, i ) : 0u ) * 8;
    s_above[i]        = !g_aboveAvail ? uint2( 0, 0 )
                        : kCoherentReads ? uint2( coherentLoad( at ), coherentLoad( at + 4 ) )
                                         : pc.bottomRows[bottomRowsAt( g_ctu - g_widthInCtus, i )];
    if( !g_leftAvail )
    {
      s_left[i] = uint2( 0, 0 );
    }
  }
  GroupMemoryBarrierWithGroupSync();
}

// Ends the CTU: stores its bottom row (the last CUs parsed over its columns) in pc.bottomRows for the CTU below.
// s_left keeps its right column for the next CTU.
void endCtu()
{
  for( int i = g_lane; i < 2 * kMapRow; i += 32 )
  {
    pc.bottomRows[bottomRowsAt( g_ctu, i )] = s_above[i];
  }
}

// Saves where the substream's parse stands, for the next dispatch: its next CTU k (or that it is done), the context
// states, the arithmetic decoder, the QPs carried to the next CTU, CodingStructure::chromaQpAdj and s_left
void saveState( uint* state, int k, int prevQp[2], bool done )
{
  GroupMemoryBarrierWithGroupSync();
  for( int i = g_lane; i < CTX_COUNT; i += 32 )
  {
    state[kStCtx + i] = s_ctx[i].x;
  }
  for( int i = g_lane; i < 2 * kMapRow; i += 32 )
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
    state[kStLevelsAt]   = g_levelsAt;
    state[kStLevelsEnd]  = g_levelsEnd;
  }
}

// Workgroup i parses substream i: from where the previous dispatch left it, CTU by CTU, until the substream ends, its
// share of bins for this dispatch is spent, or its next CTU has to wait for the CTU above.
// Resumes the substream a previous dispatch paused: the slice's contexts for their rates, then the contexts' states,
// the arithmetic decoder, the QPs and the column left of the next CTU as that dispatch saved them in state
void resumeSubstream( uint* state, inout int prevQp[2] )
{
  initContexts();
  for( int i = g_lane; i < CTX_COUNT; i += 32 )
  {
    s_ctx[i].x = state[kStCtx + i];
  }
  for( int i = g_lane; i < 2 * kMapRow; i += 32 )
  {
    s_left[i] = uint2( state[kStLeft + 2 * i], state[kStLeft + 2 * i + 1] );
  }
  GroupMemoryBarrierWithGroupSync();
  g_range         = uni( state[kStRange] );
  g_value         = uni( state[kStValue] );
  g_bitsNeeded    = int( uni( state[kStBitsNeeded] ) );
  g_pos           = uni( state[kStPos] );
  g_winBase       = 0x80000000u;   // (far from every byte, so the next read loads the window)
  prevQp[0]       = int( uni( state[kStQp0] ) );
  prevQp[1]       = int( uni( state[kStQp1] ) );
  g_csChromaQpAdj = int( uni( state[kStCqpAdj] ) );
}

// Gives the substream up after an error: counts the error, and marks its remaining CTUs (k to count - 1 of its list
// from first) parsed all the same, so that the rows that wait for them do not wait in vain
void giveUpSubstream( int first, int k, int count )
{
  AllMemoryBarrierWithGroupSync();
  if( g_lane == 0 )
  {
    pc.sync[1].add( 1u );
    for( int j = k; j < count; j++ )
    {
      pc.sync[2 + int( pc.ctuList[( first + j ) * 2] )].add( 1u, MemoryOrder.Release );
    }
  }
}

// Adds the room the job lists need for this dispatch's CUs (g_caps) to the picture's counters
void addJobListRoom()
{
  if( g_lane == 0 )
  {
    for( int i = 0; i < 5; i++ )
    {
      if( g_caps[i] != 0 )
      {
        pc.counters[2 + i].add( g_caps[i] );
      }
    }
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
    return;   // (an earlier dispatch finished the substream)
  }
  if( g_lane == 0 )
  {
    pc.sync[0].add( 1u );   // (counts this run of the substream)
  }
  PROF_BEGIN( dispatchStart );
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
    resumeSubstream( state, prevQp );
  }
  // the substream's chunk of levels (none at its start: the host zeroes the state)
  g_levelsAt  = uni( state[kStLevelsAt] );
  g_levelsEnd = uni( state[kStLevelsEnd] );
  PROF_END( kProfStart, dispatchStart );
  g_binCount  = 0;
  for( int i = 0; i < 5; i++ )
  {
    g_caps[i] = 0;
  }
  bool paused = false;
  while( k < count && !g_error )
  {
    g_ctu             = int( uni( pc.ctuList[( first + k ) * 2] ) );
    const uint flags  = uni( pc.ctuList[( first + k ) * 2 + 1] ) & 0xffu;
    const int  slot   = int( uni( pc.ctuList[( first + k ) * 2 + 1] ) >> 8 );
    g_ctuX            = g_ctu % g_widthInCtus;
    g_ctuY            = g_ctu / g_widthInCtus;
    // When the CTU needs the CTU above and that one is not parsed yet (another substream parses it), pause before
    // this CTU. Nothing of it has been parsed, so the next dispatch starts it afresh.
    if( ( flags & ( kCtuSyncLoad | kCtuWaitAbove ) ) != 0 && !ctuParsed( g_ctu - g_widthInCtus ) )
    {
      paused = true;
      break;
    }
    if( k == 0 || ( flags & kCtuRestart ) != 0 )
    {
      // The start of the substream, of a tile or of a WPP row: the slice's initial contexts (or, with WPP, the ones
      // the row above saved), the arithmetic decoder started at the next byte, the QPs reset to the slice QP
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
      decodeBinTrm();   // end_of_subset_one_bit (1); the next CTU restarts the decoder at the next byte
    }
    // Write the CTU's numbers of CUs and TUs, make everything the CTU wrote visible, then mark it parsed for the
    // substreams that wait for it
    pc.ctuOut[g_ctu * kCtuOutWords + 0] = uint( g_numCus );
    pc.ctuOut[g_ctu * kCtuOutWords + 1] = uint( g_numTus );
    AllMemoryBarrierWithGroupSync();
    if( g_lane == 0 )
    {
      // (an atomic add, as mider.slang publishes its progress: on NVIDIA's GPUs a release store did not publish the
      // writes before it)
      pc.sync[2 + g_ctu].add( 1u, MemoryOrder.Release );
    }
    k++;
    // This dispatch's share of bins is spent: pause after the CTU
    if( g_binCount >= uint( pc.binBudget ) && k < count )
    {
      paused = true;
      break;
    }
  }
  PROF_END( kProfDispatch, dispatchStart );
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
    giveUpSubstream( first, k, count );
  }
  addJobListRoom();
  saveState( state, k, prevQp, !paused || g_error );
}
