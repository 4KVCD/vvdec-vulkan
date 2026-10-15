/* -----------------------------------------------------------------------------
The copyright in this software is being made available under the Clear BSD
License, included below. No patent rights, trademark rights and/or 
other Intellectual Property Rights other than the copyrights concerning 
the Software are granted under this license.

The Clear BSD License

Copyright (c) 2018-2026, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. & The VVdeC Authors.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted (subject to the limitations in the disclaimer below) provided that
the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

     * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

     * Neither the name of the copyright holder nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.


------------------------------------------------------------------------------------------- */

#pragma once

#include "MD5.h"

#include <streambuf>
#include <array>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#if ( defined( _MSC_VER ) && defined( _M_X64 ) ) || defined( __SSE4_2__ )   // (MinGW defines _M_X64 too)
#  include <nmmintrin.h>
#  define VVDECAPP_CRC32C_INSTRUCTION 1
#endif

namespace vvdecoderapp
{

const int MD5_DIGEST_LENGTH = 128/8;

static inline std::string MD5ToHexString(unsigned char hash[MD5_DIGEST_LENGTH])
{
  static const char* hex = "0123456789abcdef";
  std::string result;

  for(unsigned pos=0; pos < MD5_DIGEST_LENGTH; ++pos)
  {
    result += hex[hash[pos] >> 4];
    result += hex[hash[pos] & 0xf];
  }

  return result;
}

// VVDEC_FASTHASH=1: the output's CRC-32C instead of its MD5 (a test aid: fast enough to check a decode at full speed;
// the digest is the CRC's 4 bytes, then 12 zero bytes)
static inline bool fastHashRequested()
{
#ifdef _MSC_VER
#  pragma warning( push )
#  pragma warning( disable : 4996 )
#endif
  return getenv( "VVDEC_FASTHASH" ) != nullptr && atoi( getenv( "VVDEC_FASTHASH" ) ) != 0;
#ifdef _MSC_VER
#  pragma warning( pop )
#endif
}

static inline uint32_t crc32c( uint32_t crc, const unsigned char* p, size_t n )
{
#if VVDECAPP_CRC32C_INSTRUCTION
  uint64_t c = crc;
  for( ; n >= 8; p += 8, n -= 8 )
  {
    uint64_t v;
    memcpy( &v, p, 8 );
    c = _mm_crc32_u64( c, v );
  }
  crc = uint32_t( c );
  for( ; n > 0; p++, n-- )
  {
    crc = _mm_crc32_u8( crc, *p );
  }
#else
  for( ; n > 0; p++, n-- )
  {
    crc ^= *p;
    for( int k = 0; k < 8; k++ )
    {
      crc = ( crc >> 1 ) ^ ( 0x82f63b78u & ( 0u - ( crc & 1u ) ) );
    }
  }
#endif
  return crc;
}

class MD5StreamBuf : public std::streambuf
{
public:
  explicit MD5StreamBuf( size_t buf_size = 64 )
    : m_buffer( buf_size )
  {
    std::streambuf::setp( &m_buffer.front(), &m_buffer.back() + 1 );
  }

  int_type overflow(int_type ch) override
  {
    update( (unsigned char*)m_buffer.data(), m_buffer.size() );

    // reset start & end-pointers
    std::streambuf::setp( &m_buffer.front(), &m_buffer.back() + 1 );
    return std::streambuf::sputc(ch);
  }

  int sync() override
  {
    const auto fillLen = std::streambuf::pptr() - std::streambuf::pbase();
    if( fillLen )
      update( (unsigned char*)std::streambuf::pbase(), (size_t)fillLen );
    return 0;
  }

  void finalize(unsigned char digest[16])
  {
    sync();
    if( m_fast )
    {
      const uint32_t crc = ~m_crc;
      memset( digest, 0, MD5_DIGEST_LENGTH );
      memcpy( digest, &crc, sizeof( crc ) );
      return;
    }
    m_md5_ctx.finalize( digest);
  }

  std::string finalizeHex()
  {
    unsigned char digest[MD5_DIGEST_LENGTH];
    finalize(digest);
    return MD5ToHexString(digest);
  }

private:
  void update( const unsigned char* data, size_t size )
  {
    if( m_fast )
    {
      m_crc = crc32c( m_crc, data, size );
      return;
    }
    m_md5_ctx.update( (unsigned char*)data, (unsigned int)size );
  }

  vvdec::libmd5::MD5     m_md5_ctx;
  std::vector<char_type> m_buffer;
  const bool             m_fast = fastHashRequested();
  uint32_t               m_crc  = 0xffffffffu;
};

}   // namespace vvdecoderapp
