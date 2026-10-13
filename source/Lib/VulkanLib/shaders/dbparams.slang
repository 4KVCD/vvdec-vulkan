// Derives the deblocking parameters on the GPU, as vvdec's
// LoopFilter::calcFilterStrengths does, from the parsed CUs and TUs
// (GpuFrontEnd.h's records) and the motion the GPU derived
// (shaders/mider.slang). The output is one LoopFilterParam per 4x4 luma block
// and direction, which holds the edge's boundary strengths, QPs, filter
// lengths and flags. The parameters are laid out as vvdec keeps them: CTU by
// CTU, first the vertical edges, then the horizontal ones.
//
// A CU writes only the parameters of the edges of its own blocks (the Q side
// of each edge), reading its neighbours' records. There are three passes:
// first the CUs of the luma tree (or of the single tree), one thread per CU;
// then, in the variant built with SUBBLOCKS=1, one thread per 4x4 block, the
// edges inside the luma tree's CUs whose boundary strengths vvdec works out
// block by block (SbTMVP or affine motion, ISP's subpartitions); then the CUs
// of the chroma tree, one thread per CU, which add their bits to the same
// blocks' parameters after the luma tree has written them, as vvdec processes
// a region's luma tree before its chroma tree. A CU's thread would otherwise
// walk all of its 4x4 blocks alone (a 128x128 affine CU has about a thousand
// boundary strengths to work out), and the pass would wait for the largest
// of those CUs.

#include "common.slang"

#ifndef SUBBLOCKS
#define SUBBLOCKS 0
#endif

struct Params
{
  uint* cus;        // CuRecord, 32 words each
  uint* tus;        // TuRecord, 16 words each
  uint* ctus;       // CtuRecord, 4 words each
  uint* slices;     // mider.slang's slice table
  uint* cuMapL;     // the index of the luma-tree CU at each 4x4 luma block
  uint* cuMapC;     // the same for the chroma tree (in a single tree, the luma tree's CUs)
  int*  cuMotion;   // mider.slang's motion records per CU (16 words each)
  int*  motion;     // mider.slang's MotionInfo per 4x4 block
  uint* lfp;        // out: LoopFilterParam, 2 words per 4x4 block, per CTU and direction
  uint* ctuInfo;    // per CTU: the index of its subpicture | (loop filter across that subpicture) << 16
  int   numCus;
  int   widthInCtus;
  int   heightInCtus;
  int   log2Ctu;
  int   width;      // luma samples
  int   height;
  int   widthIn4;
  int   csx;        // chroma subsampling shifts (0, 1)
  int   csy;
  int   chroma;     // nonzero unless the format is 4:0:0
  int   qpBdOffset; // the SPS's QpBDOffset
  int   pass;       // 0: the CUs of the luma tree (or single tree), 1: the CUs of the chroma tree (unused with SUBBLOCKS)
  int   flags;      // bits: 1 loop filter across slices, 2 across tiles (PPS); 4 subpicture info present (SPS)
  int   vbPosHor[4];   // virtual boundaries: their count, then up to 3 positions in luma samples
  int   vbPosVer[4];
  uint  lfpTag;      // the picture's tag in the parameters (see loadLfp)
};

[[vk::push_constant]] Params pc;

static const uint kMerge  = 1;
static const uint kAffine = 8;
static const uint kGeo    = 16;
static const uint kCiip   = 32;

static const int MODE_INTER = 0;
static const int MODE_INTRA = 1;
static const int MODE_IBC   = 2;

static const int EDGE_VER = 0;
static const int EDGE_HOR = 1;

static const int MRG_TYPE_SUBPU_ATMVP = 1;

// --------------------------------------------------------------------------------------------------------------------
// The records
// --------------------------------------------------------------------------------------------------------------------

struct Cu
{
  int  index;
  int  chType;     // 1: a CU of the chroma tree
  int  treeType;   // 0: TREE_D
  int  predMode;
  uint flags;
  int  isp;
  int  bdpcmL;
  int  bdpcmC;
  int  qp;
  int4 blk[2];     // per channel: x, y, width, height (width 0: none)
  int  firstTu;
  int  numTus;
  int  ctu;
};

int ctuOfLuma( int x, int y ) { return ( y >> pc.log2Ctu ) * pc.widthInCtus + ( x >> pc.log2Ctu ); }

Cu loadCu( int index )
{
  uint* r = pc.cus + index * 32;
  Cu    cu;
  cu.index    = index;
  cu.predMode = int( ( r[1] >> 16 ) & 3 );
  cu.flags    = r[1] >> 18;
  cu.qp       = int( int8_t( ( r[3] >> 16 ) & 0xff ) );
  cu.chType   = int( r[17] & 1 );
  cu.treeType = int( ( r[17] >> 1 ) & 3 );
  cu.isp      = int( ( r[17] >> 5 ) & 3 );
  cu.bdpcmL   = int( ( r[17] >> 14 ) & 3 );
  cu.bdpcmC   = int( ( r[17] >> 16 ) & 3 );
  const uint valid = r[20];
  cu.blk[0]   = ( valid & ( 1u << 17 ) ) != 0 ? int4( int( r[0] & 0xffff ), int( r[0] >> 16 ), int( r[1] & 0xff ), int( ( r[1] >> 8 ) & 0xff ) ) : int4( 0 );
  cu.blk[1]   = ( valid & ( 1u << 16 ) ) != 0 ? int4( int( r[19] & 0xffff ), int( r[19] >> 16 ), int( r[20] & 0xff ), int( ( r[20] >> 8 ) & 0xff ) ) : int4( 0 );
  // The CU's CTU, from its luma position; a chroma-tree CU has no luma block, so its chroma position is scaled.
  const bool lumaValid = ( valid & ( 1u << 17 ) ) != 0;
  const int  lx        = lumaValid ? int( r[0] & 0xffff ) : int( r[19] & 0xffff ) << pc.csx;
  const int  ly        = lumaValid ? int( r[0] >> 16 ) : int( r[19] >> 16 ) << pc.csy;
  cu.ctu       = ctuOfLuma( lx, ly );
  cu.firstTu   = int( pc.ctus[cu.ctu * 4 + 3] ) + int( r[16] & 0xffff );
  cu.numTus    = int( r[16] >> 16 );
  return cu;
}

int  mergeTypeOf( Cu cu ) { return int( ( uint( pc.cuMotion[cu.index * 16 + 12] ) >> 22 ) & 3 ); }
bool isIntra( Cu cu ) { return cu.predMode == MODE_INTRA; }
bool isIbc( Cu cu ) { return cu.predMode == MODE_IBC; }
bool affineOf( Cu cu ) { return ( cu.flags & kAffine ) != 0; }
bool geoOf( Cu cu ) { return ( cu.flags & kGeo ) != 0; }
bool ciipOf( Cu cu ) { return ( cu.flags & kCiip ) != 0; }
bool mergeOf( Cu cu ) { return ( cu.flags & kMerge ) != 0; }
bool isSepTree( Cu cu ) { return cu.treeType != 0; }

// Whether the CU's motion differs from one 8x8 sub-block to the next (SbTMVP or affine).
bool subBlockMotion( Cu cu ) { return ( mergeOf( cu ) && mergeTypeOf( cu ) == MRG_TYPE_SUBPU_ATMVP ) || affineOf( cu ); }

// Whether vvdec works out the CU's boundary strengths block by block after marking its edges (refineBs): with
// sub-block motion, and for ISP's subpartitions.
bool refinedInside( Cu cu ) { return subBlockMotion( cu ) || cu.isp != 0; }

struct Tu
{
  int4 blk[2];
  int  cbf;
  int  jointCbCr;
  int  chType;
  int  chromaQp[2];
};

Tu loadTu( int index )
{
  uint* r = pc.tus + index * 16;
  Tu    tu;
  tu.blk[0]      = ( r[1] & ( 1u << 16 ) ) != 0 ? int4( int( r[0] & 0xffff ), int( r[0] >> 16 ), int( r[1] & 0xff ), int( ( r[1] >> 8 ) & 0xff ) ) : int4( 0 );
  tu.blk[1]      = ( r[3] & ( 1u << 16 ) ) != 0 ? int4( int( r[2] & 0xffff ), int( r[2] >> 16 ), int( r[3] & 0xff ), int( ( r[3] >> 8 ) & 0xff ) ) : int4( 0 );
  tu.cbf         = int( r[4] & 7 );
  tu.jointCbCr   = int( ( r[4] >> 3 ) & 3 );
  tu.chType      = int( ( r[4] >> 10 ) & 1 );
  tu.chromaQp[0] = int( int8_t( r[5] & 0xff ) );
  tu.chromaQp[1] = int( int8_t( ( r[5] >> 8 ) & 0xff ) );
  return tu;
}

bool valid( int4 b ) { return b.z != 0 && b.w != 0; }

// A channel's subsampling shifts: 0 for luma, the format's for chroma.
int scaleX( int ch ) { return ch == 0 ? 0 : pc.csx; }
int scaleY( int ch ) { return ch == 0 ? 0 : pc.csy; }

// The CU covering pos (in the channel's samples), as CodingStructure::getCU finds it; -1 outside the picture.
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

// The CU's TU at pos, as LoopFilter's getTU finds it: the first TU whose block (in channel ch) ends past pos,
// otherwise the last TU (a CU with one TU returns it directly).
Tu tuAt( Cu cu, int2 pos, int ch )
{
  if( cu.numTus == 1 )
  {
    return loadTu( cu.firstTu );
  }
  for( int i = 0; i < cu.numTus - 1; i++ )
  {
    Tu tu = loadTu( cu.firstTu + i );
    if( tu.blk[ch].x + tu.blk[ch].z > pos.x && tu.blk[ch].y + tu.blk[ch].w > pos.y )
    {
      return tu;
    }
  }
  return loadTu( cu.firstTu + cu.numTus - 1 );
}

Tu lastTu( Cu cu ) { return loadTu( cu.firstTu + cu.numTus - 1 ); }

// Slice records (mider.slang's slice table)
uint* sliceOfCtu( int ctu ) { return pc.slices + pc.widthInCtus * pc.heightInCtus + pc.slices[ctu] * 64; }
int   slicePoc( uint* s ) { return int( s[0] ); }
bool  isInterB( uint* s ) { return ( s[1] & ( 1u << 16 ) ) != 0; }
bool  deblockingOff( uint* s ) { return ( s[1] & ( 1u << 23 ) ) != 0; }
uint  independentIdx( uint* s ) { return s[3]; }
int   refPoc( uint* s, int list, int refIdx ) { return int( s[16 + 16 * list + refIdx] ); }
uint  ctuTile( int ctu ) { return pc.ctus[ctu * 4 + 2] >> 16; }

// --------------------------------------------------------------------------------------------------------------------
// The parameters
// --------------------------------------------------------------------------------------------------------------------

// LoopFilterParam: qp[3] (int8), bs, sideMaxFiltLength, flags
struct Lfp
{
  int qp[3];
  int bs;
  int sideMaxFiltLength;
  int flags;
};

// The LoopFilterParam entry of the 4x4 luma block at (gx, gy), in 4x4-block units, for the edges of direction dir.
uint* lfpPtr( int dir, int gx, int gy )
{
  const int sh   = pc.log2Ctu - 2;
  const int mask = ( 1 << sh ) - 1;
  const int ctu  = ( gy >> sh ) * pc.widthInCtus + ( gx >> sh );
  return pc.lfp + ( ( ( 2 * ctu + dir ) << ( 2 * sh ) ) + ( gx & mask ) + ( ( gy & mask ) << sh ) ) * 2;
}

// Converts a position in the channel's samples to its 4x4 luma block (the unit of CodingStructure::inCtuPos).
int2 lfpPos( int ch, int2 pos )
{
  return int2( ( pos.x << scaleX( ch ) ) >> 2, ( pos.y << scaleY( ch ) ) >> 2 );
}

// An entry's second word carries the tag of the picture that wrote it in its upper 16 bits: an entry of another
// picture reads as zero, as the map was zeroed for each picture before (GpuRecon zeroes it only when the tags wrap
// around).
Lfp loadLfp( uint* p )
{
  Lfp        l;
  const uint b0       = p[1];
  const bool ours     = ( b0 >> 16 ) == pc.lfpTag;
  const uint a        = ours ? p[0] : 0u;
  const uint b        = ours ? b0 : 0u;
  l.qp[0]             = int( int8_t( a & 0xff ) );
  l.qp[1]             = int( int8_t( ( a >> 8 ) & 0xff ) );
  l.qp[2]             = int( int8_t( ( a >> 16 ) & 0xff ) );
  l.bs                = int( a >> 24 );
  l.sideMaxFiltLength = int( b & 0xff );
  l.flags             = int( ( b >> 8 ) & 0xff );
  return l;
}

void storeLfp( uint* p, Lfp l )
{
  p[0] = ( uint( l.qp[0] ) & 0xff ) | ( ( uint( l.qp[1] ) & 0xff ) << 8 ) | ( ( uint( l.qp[2] ) & 0xff ) << 16 ) | ( uint( l.bs & 0xff ) << 24 );
  p[1] = uint( l.sideMaxFiltLength & 0xff ) | ( uint( l.flags & 0xff ) << 8 ) | ( pc.lfpTag << 16 );
}

int  bsSet( int v, int comp ) { return v << ( comp << 1 ); }
int  bsGet( int bs, int comp ) { return ( bs >> ( comp << 1 ) ) & 3; }
bool filterEdge( Lfp l, int ch ) { return ( ( l.flags >> ch ) & 1 ) != 0; }
void setFilterEdge( inout Lfp l, int ch, bool f ) { l.flags = ( l.flags & ~( 1 << ch ) ) | ( ( f ? 1 : 0 ) << ch ); }
void setFilterCMFL( inout Lfp l, bool f ) { l.flags = ( l.flags & ~( 1 << 5 ) ) | ( ( f ? 1 : 0 ) << 5 ); }

// The motion of the 4x4 block at a luma position (mider.slang's MotionInfo).
void motionAt( int2 pos, out int2 mv[2], out int ref[2] )
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

int parl( int dir, int2 p ) { return dir == EDGE_HOR ? p.x : p.y; }
int perp( int dir, int2 p ) { return dir == EDGE_HOR ? p.y : p.x; }
int parlSize( int dir, int4 b ) { return dir == EDGE_HOR ? b.z : b.w; }
int perpSize( int dir, int4 b ) { return dir == EDGE_HOR ? b.w : b.z; }

// --------------------------------------------------------------------------------------------------------------------
// LoopFilter::xGetBoundaryStrengthSingle
// --------------------------------------------------------------------------------------------------------------------

static const int kNoPic = int( 0x80000000 );

void boundaryStrength( int dir, inout Lfp lfp, Cu cuQ, int2 localPos, Cu cuP )
{
  uint*      sliceQ    = sliceOfCtu( cuQ.ctu );
  const int  chType    = cuQ.chType;
  const int4 cuBlk     = cuQ.blk[chType];
  const int2 posQ      = localPos;
  const int2 posP      = int2( posQ.x - ( dir == EDGE_VER ? 1 : 0 ), posQ.y - dir );
  const Tu   tuQ       = tuAt( cuQ, posQ, chType );
  const Tu   tuP       = tuAt( cuP, posP, chType );
  const bool hasLuma   = valid( cuQ.blk[0] );
  const bool hasChroma = pc.chroma != 0 && valid( cuQ.blk[1] );

  bool cuPcIsIntra = false;
  int  chrmBS      = 2;
  if( hasLuma )
  {
    lfp.qp[0] = ( cuQ.qp + cuP.qp + 1 ) >> 1;
  }
  if( hasChroma )
  {
    const int  qpBdOffset2 = pc.qpBdOffset << 1;
    const bool isPQDiffCh  = chType == 0 && cuP.treeType != 0;
    const Tu   tuQc        = cuQ.isp != 0 ? lastTu( cuQ ) : tuQ;
    Cu         cuPc        = cuP;
    Tu         tuPc        = cuP.isp != 0 ? lastTu( cuP ) : tuP;
    if( isPQDiffCh )
    {
      const int2 posPc = int2( posP.x >> pc.csx, posP.y >> pc.csy );
      cuPc             = loadCu( cuAt( 1, posPc ) );
      tuPc             = tuAt( cuPc, posPc, 1 );
    }
    cuPcIsIntra = isIntra( cuPc );
    lfp.qp[1]   = ( tuPc.chromaQp[0] + tuQc.chromaQp[0] - qpBdOffset2 + 1 ) >> 1;
    lfp.qp[2]   = ( tuPc.chromaQp[1] + tuQc.chromaQp[1] - qpBdOffset2 + 1 ) >> 1;
    if( cuPcIsIntra )
    {
      chrmBS = ( cuPc.predMode == MODE_INTRA && cuPc.bdpcmC != 0 ) && ( cuQ.predMode == MODE_INTRA && cuQ.bdpcmC != 0 ) ? 0 : 2;
    }
  }

  const int bsMask = ( hasLuma ? bsSet( 3, 0 ) : 0 ) | bsSet( 3, 3 ) | ( hasChroma ? bsSet( 3, 1 ) : 0 ) | ( hasChroma ? bsSet( 3, 2 ) : 0 );

  // Intra: bs 2 (0 when both sides use BDPCM).
  if( cuP.predMode == MODE_INTRA || cuQ.predMode == MODE_INTRA )
  {
    const int edgeIdx = ( perp( dir, posQ ) - perp( dir, cuBlk.xy ) ) / 4;
    const int bsY     = cuP.bdpcmL != 0 && cuQ.bdpcmL != 0 ? 0 : 2;
    if( cuQ.isp != 0 && edgeIdx != 0 )
    {
      lfp.bs |= bsSet( bsY, 0 ) & bsMask;
    }
    else
    {
      lfp.bs |= ( bsSet( bsY, 0 ) + bsSet( chrmBS, 1 ) + bsSet( chrmBS, 2 ) ) & bsMask;
    }
    return;
  }
  else if( cuPcIsIntra )
  {
    lfp.bs |= bsSet( chrmBS, 1 ) + bsSet( chrmBS, 2 );
  }

  if( ( lfp.bs & bsMask ) != 0 && ( ciipOf( cuP ) || ciipOf( cuQ ) ) )
  {
    lfp.bs |= ( bsSet( 2, 0 ) + bsSet( 2, 1 ) + bsSet( 2, 2 ) ) & bsMask;
    return;
  }

  int tmpBs = 0;
  if( ( lfp.bs & bsMask ) != 0 )
  {
    int cbfSum = tuQ.cbf | tuP.cbf;
    tmpBs += bsSet( cbfSum & 1, 0 );
    if( !( cuP.predMode != MODE_INTRA && cuQ.predMode != MODE_INTRA && cuPcIsIntra ) )
    {
      const int jointChr = tuQ.jointCbCr != 0 || tuP.jointCbCr != 0 ? 1 : 0;
      cbfSum >>= 1;
      tmpBs += bsSet( ( cbfSum & 1 ) | jointChr, 1 );
      cbfSum >>= 1;
      tmpBs += bsSet( ( cbfSum & 1 ) | jointChr, 2 );
    }
  }
  if( bsGet( tmpBs, 0 ) == 1 )
  {
    lfp.bs |= tmpBs & bsMask;
    return;
  }
  if( ciipOf( cuP ) || ciipOf( cuQ ) )
  {
    lfp.bs |= 1 & bsMask;
    return;
  }
  if( !hasLuma )
  {
    lfp.bs |= tmpBs & bsMask;
    return;
  }
  if( bsGet( lfp.bs, 3 ) != 0 && bsGet( lfp.bs, 3 ) != 3 )
  {
    lfp.bs |= tmpBs & bsMask;
    return;
  }
  if( hasChroma )
  {
    lfp.bs |= tmpBs & bsMask;
  }
  if( cuP.predMode != cuQ.predMode && hasLuma )
  {
    lfp.bs |= 1 & bsMask;
    return;
  }

  // Compare the motion (the CU's channel is luma here).
  int2      mvQ[2], mvP[2];
  int       refQ[2], refP[2];
  motionAt( posQ, mvQ, refQ );
  motionAt( posP, mvP, refP );
  uint*     sliceP     = sliceOfCtu( cuP.ctu );
  const int nThreshold = 8;   // ( 1 << MV_FRACTIONAL_BITS_INTERNAL ) >> 1

  if( isInterB( sliceQ ) || isInterB( sliceP ) )
  {
    // The reference pictures, as POCs (for IBC, the current picture).
    const int piRefP0 = isIbc( cuP ) ? slicePoc( sliceP ) : refP[0] >= 0 ? refPoc( sliceP, 0, refP[0] ) : kNoPic;
    const int piRefP1 = isIbc( cuP ) ? kNoPic : refP[1] >= 0 ? refPoc( sliceP, 1, refP[1] ) : kNoPic;
    const int piRefQ0 = isIbc( cuQ ) ? slicePoc( sliceQ ) : refQ[0] >= 0 ? refPoc( sliceQ, 0, refQ[0] ) : kNoPic;
    const int piRefQ1 = isIbc( cuQ ) ? kNoPic : refQ[1] >= 0 ? refPoc( sliceQ, 1, refQ[1] ) : kNoPic;
    int       uiBs    = 0;
    if( ( piRefP0 == piRefQ0 && piRefP1 == piRefQ1 ) || ( piRefP0 == piRefQ1 && piRefP1 == piRefQ0 ) )
    {
      int2 p[2], q[2];
      p[0] = piRefP0 != kNoPic ? mvP[0] : int2( 0 );
      p[1] = piRefP1 != kNoPic ? mvP[1] : int2( 0 );
      q[0] = piRefQ0 != kNoPic ? mvQ[0] : int2( 0 );
      q[1] = piRefQ1 != kNoPic ? mvQ[1] : int2( 0 );
      if( piRefP0 != piRefP1 )
      {
        if( piRefP0 == piRefQ0 )
        {
          uiBs = any( abs( q[0] - p[0] ) >= nThreshold ) || any( abs( q[1] - p[1] ) >= nThreshold ) ? 1 : 0;
        }
        else
        {
          uiBs = any( abs( q[1] - p[0] ) >= nThreshold ) || any( abs( q[0] - p[1] ) >= nThreshold ) ? 1 : 0;
        }
      }
      else
      {
        uiBs = ( any( abs( q[0] - p[0] ) >= nThreshold ) || any( abs( q[1] - p[1] ) >= nThreshold ) )
                   && ( any( abs( q[1] - p[0] ) >= nThreshold ) || any( abs( q[0] - p[1] ) >= nThreshold ) )
                 ? 1 : 0;
      }
    }
    else
    {
      uiBs = 1;
    }
    lfp.bs |= ( uiBs + tmpBs ) & bsMask;
    return;
  }

  // P slices
  const int piRefP0 = isIbc( cuP ) ? slicePoc( sliceP ) : refPoc( sliceP, 0, refP[0] );
  const int piRefQ0 = isIbc( cuQ ) ? slicePoc( sliceQ ) : refPoc( sliceQ, 0, refQ[0] );
  if( piRefP0 != piRefQ0 )
  {
    lfp.bs |= ( tmpBs + 1 ) & bsMask;
    return;
  }
  lfp.bs |= ( any( abs( mvQ[0] - mvP[0] ) >= nThreshold ) ? ( tmpBs + 1 ) : tmpBs ) & bsMask;
}

// --------------------------------------------------------------------------------------------------------------------
// LoopFilter::xSetMaxFilterLengthPQFromTransformSizes
// --------------------------------------------------------------------------------------------------------------------

// Luma's maximum filter lengths on the P and Q sides of a transform edge, as sideMaxFiltLength holds them (P << 4 | Q,
// plus 128): 1 next to a block of 4 samples or fewer, else 7 next to a block of 32 or more (5 for an affine P side)
// and 3 otherwise
int lumaMaxFilterLengths( int sizeP, int sizeQ, Cu cuP )
{
  int maxFLPQ;
  if( sizeP <= 4 || sizeQ <= 4 )
  {
    maxFLPQ = 17;
  }
  else
  {
    maxFLPQ = sizeP >= 32 ? ( affineOf( cuP ) ? 5 : 7 ) : 3;
    maxFLPQ <<= 4;
    maxFLPQ += sizeQ >= 32 ? 7 : 3;
  }
  return maxFLPQ + 128;
}

// Marks a transform edge in the boundary strength: bsSet( 3, 3 ) where the edge already has a strength or is the
// CU's edge too (sameCuTu), else bsSet( 1, 3 )
void markTransformEdge( inout Lfp lfp, bool sameCuTu )
{
  if( lfp.bs != 0 || sameCuTu )
  {
    lfp.bs |= bsSet( 3, 3 );
  }
  else
  {
    lfp.bs |= bsSet( 1, 3 );
  }
}

// How far along the edge, from posQ (d into the Q side's transform block tb), the P side's transform block tuP
// reaches: the run whose parameters are the same
int sharedEdgeRun( int dir, Tu tuP, int ch, int2 posQ, int4 tb, int d )
{
  const int distance = parl( dir, tuP.blk[ch].xy ) + parlSize( dir, tuP.blk[ch] ) - parl( dir, posQ );
  return min( distance, parlSize( dir, tb ) - d );
}

void filterLengthsFromTransforms( int dir, Cu cu, Tu currTu, bool bValue, bool deriveBdStrngt, bool pqSameCtu )
{
  int start = 0, end = 1;
  if( isSepTree( cu ) )
  {
    if( cu.chType == 0 )
    {
      end = 0;
    }
    else
    {
      start = 1;
    }
  }
  if( start != end && ( pc.chroma == 0 || !valid( currTu.blk[1] ) ) )
  {
    end = 0;
  }
  const int  csx  = start != end ? pc.csx : 0;
  const int  csy  = start != end ? pc.csy : 0;
  const int4 area = currTu.blk[end];

  for( int ch = start; ch <= end; ch++ )
  {
    const bool vld = valid( currTu.blk[ch] );
    if( !vld || perp( dir, currTu.blk[ch].xy ) == 0 )
    {
      continue;
    }
    const int4 tb       = currTu.blk[ch];
    int2       lp       = lfpPos( ch, tb.xy );   // the LoopFilterParam entry, stepped along the edge
    const int  inc      = dir == EDGE_HOR ? 4 >> scaleX( ch ) : 4 >> scaleY( ch );
    const int  incFst   = dir == EDGE_HOR ? 4 >> scaleX( start ) : 4 >> scaleY( start );
    const bool sameCuTu = perp( dir, tb.xy ) == perp( dir, cu.blk[ch].xy );
    const int2 step     = dir == EDGE_HOR ? int2( 1, 0 ) : int2( 0, 1 );

    if( ch == end && deriveBdStrngt )
    {
      if( start != end )
      {
        for( int d = 0, dFst = 0; d < parlSize( dir, tb ); )
        {
          const int2 posQ     = int2( tb.x + dir * d, tb.y + ( 1 - dir ) * d );
          const int2 posP     = int2( posQ.x - ( 1 - dir ), posQ.y - dir );
          const int  sizeQ    = perpSize( dir, tb );
          const Cu   cuP      = loadCu( cuAt( ch, posP ) );
          const int4 fb       = currTu.blk[start];
          const int2 posPfst  = int2( fb.x + ( dir == EDGE_HOR ? dFst : -1 ), fb.y + ( dir == EDGE_HOR ? -1 : dFst ) );
          const Cu   cuPfstCh = loadCu( cuAt( start, posPfst ) );
          const Tu   tuP      = tuAt( cuP, posP, ch );
          const int  sizeP    = perpSize( dir, tuP.blk[ch] );
          uint*      ptr      = lfpPtr( dir, lp.x, lp.y );
          Lfp        lfp      = loadLfp( ptr );
          setFilterCMFL( lfp, sizeQ >= 8 && sizeP >= 8 );
          if( bValue )
          {
            boundaryStrength( dir, lfp, cu, int2( ( area.x + dir * d ) << csx, ( area.y + ( 1 - dir ) * d ) << csy ), cuPfstCh );
          }
          lfp.bs &= ~bsSet( 3, 3 );
          storeLfp( ptr, lfp );
          if( !isIntra( cu ) && !isIntra( cuP ) && cuP.index == cuPfstCh.index && !geoOf( cu ) && !geoOf( cuP ) )
          {
            const int distance = sharedEdgeRun( dir, tuP, ch, posQ, tb, d );
            if( distance > inc && !affineOf( cuP ) && !( mergeOf( cuP ) && mergeTypeOf( cuP ) == MRG_TYPE_SUBPU_ATMVP ) )
            {
              for( int j = inc; j < distance; j += inc, d += inc, dFst += incFst )
              {
                lp += step;
                storeLfp( lfpPtr( dir, lp.x, lp.y ), lfp );
              }
            }
          }
          lp += step;
          d += inc;
          dFst += incFst;
        }
      }
      else
      {
        for( int d = 0, dFst = 0; d < parlSize( dir, tb ); d += inc, dFst += incFst )
        {
          const int2 posQ     = int2( tb.x + dir * d, tb.y + ( 1 - dir ) * d );
          const int2 posP     = int2( posQ.x - ( 1 - dir ), posQ.y - dir );
          const int  sizeQ    = perpSize( dir, tb );
          const Cu   cuP      = loadCu( cuAt( ch, posP ) );
          const Cu   cuPfstCh = cuP;   // (start == end)
          const Tu   tuP      = tuAt( cuP, posP, ch );
          const int  sizeP    = perpSize( dir, tuP.blk[ch] );
          uint*      ptr      = lfpPtr( dir, lp.x, lp.y );
          Lfp        lfp      = loadLfp( ptr );
          setFilterEdge( lfp, cu.chType, bValue );
          if( bValue )
          {
            markTransformEdge( lfp, sameCuTu );
          }
          else
          {
            lfp.bs |= bsSet( 1, 3 );
          }
          if( ch == 0 )
          {
            lfp.sideMaxFiltLength = lumaMaxFilterLengths( sizeP, sizeQ, cuP );
          }
          else
          {
            setFilterCMFL( lfp, sizeQ >= 8 && sizeP >= 8 );
          }
          if( bValue )
          {
            boundaryStrength( dir, lfp, cu, int2( ( area.x + dir * d ) << csx, ( area.y + ( 1 - dir ) * d ) << csy ), cuPfstCh );
          }
          lfp.bs &= ~bsSet( 3, 3 );
          storeLfp( ptr, lfp );
          lp += step;
        }
      }
    }
    else
    {
      for( int d = 0; d < parlSize( dir, tb ); )
      {
        const int2 posQ     = int2( tb.x + dir * d, tb.y + ( 1 - dir ) * d );
        const int2 posP     = int2( posQ.x - ( 1 - dir ), posQ.y - dir );
        const int  sizeQ    = perpSize( dir, tb );
        const Cu   cuP      = loadCu( cuAt( ch, posP ) );
        const Tu   tuP      = tuAt( cuP, posP, ch );
        const int  sizeP    = perpSize( dir, tuP.blk[ch] );
        const int  distance = sharedEdgeRun( dir, tuP, ch, posQ, tb, d );
        if( ch == 0 )
        {
          uint* ptr = lfpPtr( dir, lp.x, lp.y );
          Lfp   lfp = loadLfp( ptr );
          setFilterEdge( lfp, cu.chType, bValue );
          if( bValue )
          {
            markTransformEdge( lfp, sameCuTu );
            lfp.sideMaxFiltLength = lumaMaxFilterLengths( sizeP, sizeQ, cuP );
          }
          storeLfp( ptr, lfp );
          if( distance > inc )
          {
            for( int j = inc; j < distance; j += inc, d += inc )
            {
              lp += step;
              storeLfp( lfpPtr( dir, lp.x, lp.y ), lfp );
            }
          }
          d += inc;
          lp += step;
        }
        else
        {
          for( int j = 0; j < distance; j += inc, d += inc )
          {
            uint* ptr = lfpPtr( dir, lp.x, lp.y );
            Lfp   lfp = loadLfp( ptr );
            if( ch == start )
            {
              setFilterEdge( lfp, cu.chType, bValue );
              if( bValue )
              {
                markTransformEdge( lfp, sameCuTu );
              }
            }
            setFilterCMFL( lfp, sizeQ >= 8 && sizeP >= 8 );
            storeLfp( ptr, lfp );
            lp += step;
          }
        }
      }
    }
  }
}

// --------------------------------------------------------------------------------------------------------------------
// LoopFilter::xSetEdgeFilterInsidePu, xSetMaxFilterLengthPQForCodingSubBlocks and the boundary strengths of a CU
// whose motion or prediction differs inside it, one 4x4 block at a time
// --------------------------------------------------------------------------------------------------------------------

// Whether the 4x4 block g's edge in direction dir starts a transform block: calcFilterStrengths marks those edges with
// bit 7 of sideMaxFiltLength.
bool transformEdge( int dir, int2 g )
{
  return ( loadLfp( lfpPtr( dir, g.x, g.y ) ).sideMaxFiltLength & 128 ) != 0;
}

// Whether a vertical (dir EDGE_VER) or horizontal edge at the luma position pos lies on a virtual boundary.
bool onVirtualBoundary( int dir, int2 pos )
{
  const int count = dir == EDGE_VER ? pc.vbPosVer[0] : pc.vbPosHor[0];
  for( int i = 0; i < count; i++ )
  {
    if( ( dir == EDGE_VER ? pc.vbPosVer[1 + i] : pc.vbPosHor[1 + i] ) == perp( dir, pos ) )
    {
      return true;
    }
  }
  return false;
}

// Whether a transform edge or the CU's far edge lies within `blocks` 4x4 blocks of the sub-block edge at the 4x4 block
// g, perpVal luma samples from the CU's left or top edge (perpVal >= 8). The CU's far edge is tested first, so that
// the marks read always lie inside the CU.
bool transformEdgeNear( int dir, Cu cu, int2 g, int perpVal, int blocks )
{
  if( perpVal + 4 * blocks >= perpSize( dir, cu.blk[0] ) )
  {
    return true;
  }
  const int2 off = dir == EDGE_HOR ? int2( 0, blocks ) : int2( blocks, 0 );
  return transformEdge( dir, g - off ) || transformEdge( dir, g + off );
}

// xSetMaxFilterLengthPQForCodingSubBlocks for the 4x4 block g of a CU with sub-block motion, whose edge in direction
// dir lies on the CU's 8x8 grid, perpVal luma samples from the CU's left or top edge: the edge's new
// sideMaxFiltLength. It reads only the transform edges' marks, inside the CU, which the per-CU pass set.
int subBlockFilterLength( int dir, Cu cu, int2 g, int perpVal, int sideMaxFiltLength )
{
  int maxFLP, maxFLQ;
  int te = 0;
  if( ( sideMaxFiltLength & 128 ) != 0 )
  {
    te     = 128;
    maxFLQ = min( sideMaxFiltLength & 7, 5 );
    maxFLP = ( sideMaxFiltLength >> 4 ) & 7;
    if( perpVal > 0 )
    {
      maxFLP = min( maxFLP, 5 );
    }
  }
  else if( perpVal > 0 && transformEdgeNear( dir, cu, g, perpVal, 1 ) )
  {
    maxFLP = maxFLQ = 1;
  }
  else if( perpVal > 0 && ( perpVal == 8 || transformEdgeNear( dir, cu, g, perpVal, 2 ) ) )
  {
    maxFLP = maxFLQ = 2;
  }
  else
  {
    maxFLP = maxFLQ = 3;
  }
  return ( maxFLP << 4 ) + maxFLQ + te;
}

// The edges of the 4x4 block g, inside a luma-tree CU whose boundary strengths vvdec refines (refinedInside), after the
// per-CU pass has set the CU's transform edges. With sub-block motion, an edge on the CU's 8x8 grid is marked for
// filtering (xSetEdgeFilterInsidePu, except on a virtual boundary) and gets its filter lengths; then every marked
// edge gets its boundary strength. vvdec runs these steps over the whole CU, one step after the other; each block's
// steps read and write only the block's own parameters, plus the transform marks next to it, which no step changes.
void subBlockEdges( Cu cu, int2 g )
{
  const int2 pos   = g * 4;                // the block's luma position
  const int2 local = pos - cu.blk[0].xy;   // its place in the CU
  const bool subPu = subBlockMotion( cu );
  for( int dir = EDGE_VER; dir <= EDGE_HOR; dir++ )
  {
    uint*     ptr     = lfpPtr( dir, g.x, g.y );
    Lfp       lfp     = loadLfp( ptr );
    const int perpVal = perp( dir, local );
    if( subPu && ( perpVal & 7 ) == 0 )
    {
      if( perpVal > 0 )
      {
        const bool edge = !onVirtualBoundary( dir, pos );
        setFilterEdge( lfp, cu.chType, edge );
        if( edge && lfp.bs != 0 )
        {
          lfp.bs |= bsSet( 3, 3 );
        }
      }
      lfp.sideMaxFiltLength = subBlockFilterLength( dir, cu, g, perpVal, lfp.sideMaxFiltLength );
    }
    // (an edge on the picture's left or top boundary is never marked: it has no P side)
    const int indexP = perpVal != 0 ? cu.index : cuAt( cu.chType, pos - ( dir == EDGE_VER ? int2( 1, 0 ) : int2( 0, 1 ) ) );
    if( filterEdge( lfp, cu.chType ) && indexP >= 0 )
    {
      boundaryStrength( dir, lfp, cu, pos, indexP == cu.index ? cu : loadCu( indexP ) );
    }
    lfp.bs &= ~bsSet( 3, 3 );
    storeLfp( ptr, lfp );
  }
}

// --------------------------------------------------------------------------------------------------------------------
// LoopFilter::calcFilterStrengths
// --------------------------------------------------------------------------------------------------------------------

// Whether the loop filter may cross from cu to other under the slice, tile and subpicture rules (CU::isAvailable).
bool lfAvailable( Cu cu, Cu other )
{
  uint*      s0            = sliceOfCtu( cu.ctu );
  uint*      s1            = sliceOfCtu( other.ctu );
  const bool acrossSlices  = ( pc.flags & 1 ) != 0;
  const bool acrossTiles   = ( pc.flags & 2 ) != 0;
  const uint sub0          = pc.ctuInfo[cu.ctu];
  const uint sub1          = pc.ctuInfo[other.ctu];
  const bool acrossSubPics = ( pc.flags & 4 ) == 0 || ( ( sub0 >> 16 ) != 0 && ( sub1 >> 16 ) != 0 );
  return ( acrossSlices || independentIdx( s0 ) == independentIdx( s1 ) ) && ( acrossTiles || ctuTile( cu.ctu ) == ctuTile( other.ctu ) )
         && ( acrossSubPics || ( sub0 & 0xffff ) == ( sub1 & 0xffff ) );
}

// The positions of the virtual boundaries (luma samples) that cross area a, as isCrossedByVirtualBoundaries finds them.
void virtualBoundaries( int4 a, out int numHor, out int numVer, out int3 hor, out int3 ver )
{
  numHor = 0;
  numVer = 0;
  hor    = int3( 0 );
  ver    = int3( 0 );
  for( int i = 0; i < pc.vbPosHor[0]; i++ )
  {
    const int p = pc.vbPosHor[1 + i];
    if( a.y <= p && p <= a.y + a.w )
    {
      if( numHor == 0 ) hor.x = p; else if( numHor == 1 ) hor.y = p; else hor.z = p;
      numHor++;
    }
  }
  for( int i = 0; i < pc.vbPosVer[0]; i++ )
  {
    const int p = pc.vbPosVer[1 + i];
    if( a.x <= p && p <= a.x + a.z )
    {
      if( numVer == 0 ) ver.x = p; else if( numVer == 1 ) ver.y = p; else ver.z = p;
      numVer++;
    }
  }
}

// Turns off the edges at pos that lie on a virtual boundary (xDeriveEdgefilterParam).
void edgeFilterParam( int2 pos, int numVer, int numHor, int3 ver, int3 hor, inout bool verEdge, inout bool horEdge )
{
  for( int i = 0; i < numVer; i++ )
  {
    if( ( i == 0 ? ver.x : i == 1 ? ver.y : ver.z ) == pos.x )
    {
      verEdge = false;
      break;
    }
  }
  for( int i = 0; i < numHor; i++ )
  {
    if( ( i == 0 ? hor.x : i == 1 ? hor.y : hor.z ) == pos.y )
    {
      horEdge = false;
      break;
    }
  }
}

// The threads per CU of the per-CU passes: one per direction and every kTuThreads-th TU of the CU
static const int kTuThreads    = 4;
static const int kThreadsPerCu = 2 * kTuThreads;

// The parameters of the edges in direction dir of CU cu's TUs first, first + kTuThreads, ...
void calcFilterStrengths( Cu cu, int dir, int first )
{
  const int4 area = cu.blk[cu.chType];
  const int  sx   = scaleX( cu.chType );
  const int  sy   = scaleY( cu.chType );

  int  numHor, numVer;
  int3 hor, ver;
  virtualBoundaries( int4( area.x << sx, area.y << sy, area.z << sx, area.w << sy ), numHor, numVer, hor, ver );
  const bool crossed = numHor > 0 || numVer > 0;

  // Whether the edge at the CU's left (EDGE_VER) or top is filtered (xGetLoopfilterParam).
  bool cuEdge = false;
  if( dir == EDGE_VER && area.x > 0 )
  {
    cuEdge = lfAvailable( cu, loadCu( cuAt( cu.chType, int2( area.x - 1, area.y ) ) ) );
  }
  if( dir == EDGE_HOR && area.y > 0 )
  {
    cuEdge = lfAvailable( cu, loadCu( cuAt( cu.chType, int2( area.x, area.y - 1 ) ) ) );
  }

  // A CU with sub-block motion or ISP gets only its transform edges marked here: the edges inside it and all its
  // boundary strengths come from subBlockEdges, a thread per 4x4 block.
  const bool refineBs       = refinedInside( cu );
  // Masks that round to the LF_PARAM_MAP's unit, a 4x4 luma block, in the channel's samples.
  const int  maskBlkX       = ~( ( 1 << ( 2 - sx ) ) - 1 );
  const int  maskBlkY       = ~( ( 1 << ( 2 - sy ) ) - 1 );
  const int  ctuMask        = ( 1 << pc.log2Ctu ) - 1;
  const bool pqCuSameCtuHor = ( area.y & ( ctuMask >> sy ) ) > 0;
  const bool pqCuSameCtuVer = ( area.x & ( ctuMask >> sx ) ) > 0;

  for( int t = first; t < cu.numTus; t += kTuThreads )
  {
    const Tu   tu            = loadTu( cu.firstTu + t );
    const int4 areaTu        = tu.blk[tu.chType];
    bool       verEdgeFilter = ( areaTu.x & maskBlkX ) == area.x ? cuEdge : true;
    bool       horEdgeFilter = ( areaTu.y & maskBlkY ) == area.y ? cuEdge : true;
    const bool pqSameCtuHor  = ( areaTu.y & maskBlkY ) == area.y ? pqCuSameCtuHor : true;
    const bool pqSameCtuVer  = ( areaTu.x & maskBlkX ) == area.x ? pqCuSameCtuVer : true;
    if( crossed )
    {
      edgeFilterParam( int2( ( areaTu.x & maskBlkX ) << sx, ( areaTu.y & maskBlkY ) << sy ), numVer, numHor, ver, hor, verEdgeFilter, horEdgeFilter );
    }
    if( dir == EDGE_VER )
    {
      filterLengthsFromTransforms( EDGE_VER, cu, tu, verEdgeFilter, !refineBs, pqSameCtuVer );
    }
    else
    {
      filterLengthsFromTransforms( EDGE_HOR, cu, tu, horEdgeFilter, !refineBs, pqSameCtuHor );
    }
  }
}

#if SUBBLOCKS
// One thread per 4x4 luma block: the edges inside a luma-tree CU with sub-block motion or ISP (a chroma-tree CU has
// neither); blocks of other CUs, and of slices with deblocking off, are skipped.
[shader( "compute" )]
[numthreads( 64, 1, 1 )]
void main( uint3 id : SV_DispatchThreadID )
{
  const int index     = int( id.y * 65536 + id.x );
  const int heightIn4 = ( pc.height + 3 ) >> 2;
  if( index >= pc.widthIn4 * heightIn4 )
  {
    return;
  }
  // (mider_cumap writes every 4x4 block's CU; the checks keep a wrong entry from sending the reads outside the
  // picture's records)
  const uint cuIndex = pc.cuMapL[index];
  if( cuIndex >= uint( pc.numCus ) )
  {
    return;
  }
  const Cu   cu = loadCu( int( cuIndex ) );
  const int2 g  = int2( index % pc.widthIn4, index / pc.widthIn4 );
  if( cu.chType != 0 || !refinedInside( cu ) || any( g * 4 < cu.blk[0].xy ) || any( g * 4 >= cu.blk[0].xy + cu.blk[0].zw )
      || deblockingOff( sliceOfCtu( cu.ctu ) ) )
  {
    return;
  }
  subBlockEdges( cu, g );
}
#else
// kThreadsPerCu threads per CU of the pass's tree (a direction and every kTuThreads-th TU each); CUs in slices with
// deblocking off are skipped.
[shader( "compute" )]
[numthreads( 64, 1, 1 )]
void main( uint3 id : SV_DispatchThreadID )
{
  const int thread = int( id.y * 65536 + id.x );
  const int index  = thread / kThreadsPerCu;
  const int sub    = thread % kThreadsPerCu;
  if( index >= pc.numCus )
  {
    return;
  }
  Cu cu = loadCu( index );
  if( cu.chType != pc.pass || deblockingOff( sliceOfCtu( cu.ctu ) ) || sub / 2 >= cu.numTus )
  {
    return;
  }
  calcFilterStrengths( cu, sub & 1, sub >> 1 );
}
#endif
