// The intra jobs that jobs.slang writes and intrasort.slang and intra.slang read: GpuJobs.h's IntraJob, 12 words.
// Word 1 holds the block's width and height in samples, its component and its kind; word 2 its mode, its flags, its
// BDPCM direction and its reference line.

static const int kIntraJobWords = 12;

// Word 7 holds the job's wave, which jobs.slang sets (IntraJob::setWave). intrasort.slang cuts the jobs into bands of
// rows; in a band's copy of the job, the word holds the band's rows instead (first | count << 16, IntraJob::setRows).
static const int kWaveWord = 7;

// The kinds of intra jobs (IntraJob::Kind)
static const int kIntraAngular = 0;   // planar, DC and the angular modes
static const int kIntraBdpcm   = 1;
static const int kIntraCclm    = 2;
static const int kIntraIbc     = 3;
static const int kIntraMip     = 4;   // matrix-based intra prediction (the job's mode is the MIP mode)
static const int kIntraLmcs    = 5;   // an inter CU's chroma under LMCS: its prediction, kept aside, plus its scaled residual
static const int kIntraCiip    = 6;   // a CIIP CU's component: the inter prediction, kept aside, blended with planar's

// The flags of intra jobs (IntraJob::Flags)
static const uint kIntraResidual      = 1;   // the picture holds the block's residual
static const uint kIntraFiltered      = 2;   // predict from the [1 2 1]-filtered reference samples
static const uint kIntraMipTransposed = 4;
static const uint kIntraIsp           = 8;   // a luma sub-partition of an ISP CU
