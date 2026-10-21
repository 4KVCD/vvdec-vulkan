// The records that the parse leaves for the GPU front end (GpuFrontEnd.h documents each record's words): a CU's
// CuRecord, a TU's TuRecord, a CTU's CtuRecord, the picture's slice table and jobs.slang's per-slice JobSlice. Each
// record is a fixed number of 32-bit words, stored one after another.

static const int kCuRecordWords    = 32;    // CuRecord
static const int kTuRecordWords    = 16;    // TuRecord
static const int kCtuRecordWords   = 4;     // CtuRecord
static const int kSliceRecordWords = 64;    // a slice's record in the slice table
static const int kJobSliceWords    = 256;   // JobSlice

// The CTU record's words: the index of the CTU's first CU (the CUs are stored CTU by CTU, each CTU's in decoding
// order); its number of CUs and its slice's index; whether it starts a row of its tile and its tile's index; the index
// of its first TU.
uint ctuFirstCu( uint* ctus, int ctu )
{
  return ctus[ctu * kCtuRecordWords];
}

uint ctuNumCus( uint* ctus, int ctu )
{
  return ctus[ctu * kCtuRecordWords + 1] & 0xffff;
}

uint ctuSliceIndex( uint* ctus, int ctu )
{
  return ctus[ctu * kCtuRecordWords + 1] >> 16;
}

bool ctuStartsTileRow( uint* ctus, int ctu )
{
  return ( ctus[ctu * kCtuRecordWords + 2] & 1 ) != 0;
}

uint ctuTileIndex( uint* ctus, int ctu )
{
  return ctus[ctu * kCtuRecordWords + 2] >> 16;
}

uint ctuFirstTu( uint* ctus, int ctu )
{
  return ctus[ctu * kCtuRecordWords + 3];
}

// The slice table holds one word per CTU, the index of the CTU's slice, followed by the slices' records. Returns the
// record of the slice of CTU ctu, in a picture of numCtus CTUs.
uint* sliceRecordOfCtu( uint* slices, int numCtus, int ctu )
{
  return slices + numCtus + slices[ctu] * kSliceRecordWords;
}

// A slice record's words (writeSliceTable): its POC; whether it is a B slice and whether deblocking is off; the index of
// its independent slice (the dependent slices of one share it); which references are long-term; their POCs.
int slicePoc( uint* s )
{
  return int( s[0] );
}

bool isInterB( uint* s )
{
  return ( s[1] & ( 1u << 16 ) ) != 0;
}

bool deblockingOff( uint* s )
{
  return ( s[1] & ( 1u << 23 ) ) != 0;
}

uint independentIdx( uint* s )
{
  return s[3];
}

bool usedAsLongTerm( uint* s, int list, int refIdx )
{
  return ( s[4] & ( 1u << ( 16 * list + refIdx ) ) ) != 0;
}

int refPoc( uint* s, int list, int refIdx )
{
  return int( s[16 + 16 * list + refIdx] );
}
