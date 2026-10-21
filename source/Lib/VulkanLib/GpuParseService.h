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

/** \file     GpuParseService.h
    \brief    The GPU's parse of a picture's slice data (ParseSlot), and of the pictures ahead of their
              reconstruction (ParseService, VVDEC_GPU_PARSE=2)

    A ParseSlot holds one picture's parse on the GPU: its inputs, where each substream stopped between dispatches,
    and what the parse leaves. parse.slang writes each CU's record, and its TUs' records, wherever the picture's
    counters give them room (the staged records), with the CTU and the place in the CTU of each (its meta word);
    parsecompact.slang's first pass then works out each CTU's first CU and TU in raster order, and its scatter passes
    copy the records to those places, in the buffers the front end reads. The parse also adds up the room the front
    end's job lists need (gpu::frontCapacities).

    The ParseService parses the pictures as soon as all their slices are in (DecLibParser), many pictures at a time:
    its thread records a few dispatches of every picture it is parsing into one submission on the parse queue, so
    that their substreams run side by side, and waits for it. When a picture's parse is done, a small submission
    on a queue of its own (Context::submitParseCopy) copies its records into buffers of their own, of their exact size
    (a ParsedPicture: parsecompact.slang's scatter), while the next step runs; once the copy is done, the picture goes
    to its reconstruction (Picture::gpuParsed) and its slot is free for the next picture. The reconstruction gives
    the ParsedPicture back once the picture's GPU work is done (release), and its buffers are reused. So a picture
    parsed ahead of its reconstruction holds only its records.
*/

#pragma once

#include "GpuArena.h"
#include "GpuContext.h"
#include "GpuFrontEnd.h"
#include "GpuParse.h"

#include "CommonLib/CommonDef.h"

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace vvdec
{

class CodingStructure;
struct Picture;

namespace gpu
{

// The buffers a parse stages its records in, before the compaction puts them in raster order (GPU memory): the CU and
// TU records and their meta words. The ParseService takes them from its arena for each picture, with room for the
// picture's kind.
struct StagedRecords
{
  Buffer cus, tus, cuMeta, tuMeta;
};

// After an overflow (ParseSlot::Progress::Overflow), a parse starts again with room for a quarter more records and
// levels than it counted, plus these to spare.
static constexpr size_t kOverflowRecordsSpare = 64;
static constexpr size_t kOverflowLevelsSpare  = 4096;
inline size_t roomAfterOverflow( size_t counted, size_t spare )
{
  return counted * 5 / 4 + spare;
}

// One picture's parse on the GPU
class ParseSlot
{
public:
  explicit ParseSlot( Context* ctx ) : m_ctx( ctx ) {}
  ~ParseSlot();
  ParseSlot( const ParseSlot& )            = delete;
  ParseSlot& operator=( const ParseSlot& ) = delete;

  // Sets the slot up for the picture of cs: its inputs uploaded, its records staged in staged and its levels written
  // into levels (the slot holds both until takeStaged and takeLevels), every substream at its start. Returns false
  // when a slice's data was not captured.
  bool prepare( const CodingStructure& cs, StagedRecords staged, Buffer levels );
  // With VVDEC_GPU_PROFILE and the shader clock: the parse's cycle counts are added up at prof (kParseProfileWords)
  void setProfile( VkDeviceAddress prof ) { m_prof = prof; }
  // Records into cmd the clearing of the parse's counters (in GPU memory) when the parse starts from the beginning
  // (after prepare or restart). The steps after it need a barrier.
  void recordClear( VkCommandBuffer cmd );
  // Records into cmd one dispatch of the parse: each substream is parsed from where it stopped, up to binBudget bins
  // (a CU counts as 32 bins). The budget is kStepBins, or more while ParseService has few substreams in parse and every
  // picture in parse has a large substream.
  static constexpr int32_t kStepBins = 1 << 12;
  void recordStep( VkCommandBuffer cmd, int32_t binBudget = kStepBins ) const;
  // Records into cmd the compaction's first pass, which does its work once every CTU is parsed: it writes each CTU's
  // record (its first CU and TU in raster order) and the totals. serial is the step's number in ParseService (from 1).
  // The totals go to the half of the totals buffer chosen by serial's parity and carry serial, so that one step's
  // totals stay intact while the next step runs.
  void recordCompaction( VkCommandBuffer cmd, uint32_t serial = 0 ) const;

  // Where the parse is, after a submission of its steps and compaction
  enum class Progress
  {
    Parsing,    // substreams left to parse
    Done,       // all parsed and compacted
    Overflow,   // all parsed, but with more records than the room: restart() and parse again
    Failed,     // a substream given up on (a broken bitstream), or none advanced for kParseMaxStalls submissions
  };
  // serial: the step whose totals to read (its compaction's serial); 0: the only ones (a step at a time)
  Progress progress( uint32_t serial = 0 );
  // After Overflow: the records staged in staged and the levels written into levels (the slot's buffers taken back
  // first: takeStaged, takeLevels), and every substream back at its start
  void   restart( StagedRecords staged, Buffer levels );
  // What the parse counted, also past the room: the CU and TU records, the levels
  size_t cusCounted() const;
  size_t tusCounted() const;
  size_t levelsCounted() const;
  // The buffers of the staged records and of the levels, given back (empty ones when the slot holds none)
  StagedRecords   takeStaged();
  Buffer          takeLevels();
  // Makes staged hold at least cus and tus records (for a caller that keeps its own buffers: GpuRecon's check)
  static void     growStaged( Context* ctx, StagedRecords& staged, size_t cus, size_t tus );
  VkDeviceAddress levels() const { return m_levels.addr; }

  // --- What the parse left, once Done ---
  size_t                 numCus() const { return m_numCus; }
  size_t                 numTus() const { return m_numTus; }
  size_t                 numLevels() const { return m_numLevels; }
  const FrontCapacities& capacities() const { return m_caps; }
  // CTU ctu's SAO and ALF parameters, as the parse wrote them (gpu::captureSaoAlf's layout: kParseSaoAlfWords words)
  const uint32_t*        saoAlf( int ctu ) const;
  // Records into cmd the copies of the records to their places in raster order: the CU and TU records into cus and
  // tus (numCus() and numTus() of them), the CTU records into ctus. Without a barrier after them.
  void recordScatter( VkCommandBuffer cmd, VkDeviceAddress cus, VkDeviceAddress tus, VkDeviceAddress ctus ) const;
  // Records into cmd the copy of the levels (numLevels() of them) into levels, in the same places
  void recordLevelsCopy( VkCommandBuffer cmd, const Buffer& levels ) const;
  size_t   numCtus() const { return m_numCtus; }

  // --- For the statistics and the check (VVDEC_GPU_PARSE=1) ---
  const ParseInput& input() const { return m_in; }
  uint32_t          runs() const;   // the substreams' runs: one per substream and dispatch it parsed in
  uint32_t          errors() const;   // the substreams given up on, or left unparsed
  const uint32_t*   ctuRecords() const { return reinterpret_cast<const uint32_t*>( m_ctus.mapped ); }   // CtuRecord a CTU
  const uint32_t*   ctuOut() const { return reinterpret_cast<const uint32_t*>( m_ctuOut.mapped ); }     // kParseCtuOutWords a CTU
  // The numbers of CUs and TUs (the CTUs' counts), then the parse's counters: the staged CU and TU records, the job
  // lists' room (5 words) and the levels. These are the totals of the step that progress() read last.
  const uint32_t*   totals() const;

  Picture* picture  = nullptr;   // the picture parsed (ParseService)
  bool     failed   = false;     // its parse gave up (GpuRecon's check, VVDEC_GPU_PARSE=1)
  uint64_t lastStep = 0;         // the parse timeline's value of the last step that steps it (ParseService)

private:
  // Takes the staged records' buffers, and the room they give
  void setStaged( StagedRecords staged );
  // Clears what the parse counts and where each substream is, for a parse from the start
  void clear();

  Context*        m_ctx = nullptr;
  ParseInput      m_in;
  size_t          m_numCtus = 0;
  bool            m_ctu64   = false;   // CTUs of 64 samples at most (prepare): the parse's _ctu64 variants
  VkDeviceAddress m_prof = 0;
  // the inputs (upload memory)
  Buffer m_bits, m_streams, m_ctuList, m_ctuInfo, m_slices, m_pic, m_scans;
  // what the parse writes: the staged records and their meta words (the caller's buffers: prepare to takeStaged),
  // each CTU's bottom rows (the CTU below reads them), WPP's saved contexts, the counters (the staged records, the
  // job lists' room) in GPU memory; the substreams' runs, the errors and each CTU's readiness (sync), each substream's
  // state, each CTU's counts and SAO / ALF parameters (ctuOut) in host memory
  Buffer m_cus, m_tus, m_cuMeta, m_tuMeta, m_bottomRows, m_wpp, m_counters, m_sync, m_state, m_ctuOut;
  // the compaction's first pass: the CTU records, the totals (host memory)
  Buffer m_ctus, m_totals;
  // the levels (GPU memory, ParseService's: the slot holds it from prepare to takeLevels)
  Buffer m_levels;
  size_t m_levelCap = 0, m_numLevels = 0;   // the room for levels, the levels once Done
  size_t          m_cuCap = 0, m_tuCap = 0;   // the staged records' room
  size_t          m_numCus = 0, m_numTus = 0;
  FrontCapacities m_caps;
  uint64_t        m_lastProgress = 0;   // the substreams' next CTUs added up, after the submission before
  uint32_t        m_stalls       = 0;   // submissions in a row without progress
  bool            m_fresh        = false;   // the counters to clear (recordClear)
  uint32_t        m_half         = 0;   // the half of m_totals that progress() read last
};

// What the GPU's parse left of a picture once it is done (ParseService): its CU, TU and CTU records in raster order
// (GpuFrontEnd.h's layout) and its coefficient levels (each TU record says where its levels are), in GPU memory of
// their own and of their exact size; its CTUs' SAO and ALF parameters; the room the front end's job lists need. The
// picture holds it until its GPU work is done (ParseService::release).
class ParsedPicture
{
public:
  size_t                 numCus() const { return m_numCus; }
  size_t                 numTus() const { return m_numTus; }
  const FrontCapacities& capacities() const { return m_caps; }
  // CTU ctu's SAO and ALF parameters, as the parse wrote them (gpu::captureSaoAlf's layout: kParseSaoAlfWords words)
  const uint32_t*        saoAlf( int ctu ) const { return m_saoAlf.data() + size_t( ctu ) * kSaoAlfWords; }
  VkDeviceAddress        cus() const { return m_cus.addr; }
  VkDeviceAddress        tus() const { return m_tus.addr; }
  VkDeviceAddress        ctus() const { return m_ctus.addr; }
  VkDeviceAddress        levels() const { return m_levels.addr; }
  // the parse's copies' timeline value once the records are in place (Context::submitParseCopy): the front end waits
  // for it
  uint64_t               parseValue() const { return m_parseValue; }

  Picture* picture = nullptr;   // the picture parsed

private:
  friend class ParseService;
  static constexpr size_t kSaoAlfWords = kParseSaoAlfWords;
  size_t                  m_numCus = 0, m_numTus = 0, m_numLevels = 0;
  FrontCapacities         m_caps;
  std::vector<uint32_t>   m_saoAlf;
  Buffer                  m_cus, m_tus, m_ctus, m_levels;
  uint64_t                m_parseValue = 0;
};

// Sets each CTU's SAO and ALF parameters in its CTU data (CtuData::saoParam, alfParam) from the GPU's parse of the
// picture, as the CPU's parse leaves them
void applyParsedSaoAlf( CodingStructure& cs, const ParsedPicture& parsed );

// How many pictures the ParseService has in parse and ahead of their reconstruction (GpuParseService.cpp's
// kIntegratedLimits and kDiscreteLimits say why)
struct ParseLimits
{
  size_t minParsing, maxParsing, streamsInParse, maxAhead;
};

// The GPU's parse of the pictures ahead of their reconstruction (VVDEC_GPU_PARSE=2), on the parse queue
class ParseService
{
public:
  // wakeUp: called after each picture handed over (the decoder's thread pool may sleep while its tasks wait for it)
  ParseService( Context* ctx, std::function<void()> wakeUp );
  // Stops the thread once its submission is done. The pictures waiting stay unparsed (the decoder is shutting down).
  ~ParseService();
  ParseService( const ParseService& )            = delete;
  ParseService& operator=( const ParseService& ) = delete;

  // Parses the picture, whose slices are all in, as soon as it can: then pic->gpuParseResult holds the result (null
  // when the parse gave up) and pic->gpuParsed unlocks.
  void enqueue( Picture* pic );
  // Gives a picture's parse result back once the picture's GPU work, which reads its records, is done: its buffers
  // are reused for the pictures parsed next
  void release( ParsedPicture* parsed );
  // What the service did so far
  ParseStats stats() const;

private:
  void           run();                                    // the thread
  // (with m_mutex held:) whether the thread has anything to do; the parse results held (handed over or being copied,
  // not given back yet); the waiting pictures to start now, taken from m_waiting
  bool                  hasWork() const;
  size_t                heldResults() const;
  std::vector<Picture*> takeStartingPictures();
  void           failAll();                                // after a GPU failure: every picture's parse fails
  ParseSlot*     takeSlot();                               // a slot no picture uses, or a new one
  ParsedPicture* takeParsed();                             // a parse result no picture holds, or a new one
  StagedRecords  takeStaged( size_t cus, size_t tus );     // room for a parse's staged records, from the arena
  void           giveRecords( ParseSlot* slot );           // the slot's records and levels back to the arena
  void           giveSlot( ParseSlot* slot );              // the same, and the slot free
  void           startPicture( Picture* pic );             // its slot set up, among the pictures parsed
  // A picture's records and levels, counted by its parse (with its slice data's bytes), or the room for them
  struct Counts
  {
    size_t cus = 0, tus = 0, levels = 0, bytes = 0;
    void   raise( const Counts& counts );   // each to the larger of the two
  };
  Counts         startRoom( const Picture& pic ) const;    // the room its parse starts with
  // A step of the pictures in parse in flight: its commands, its fence, the parse timeline's value once it is done,
  // its number (its compaction marks the totals with it) and the pictures it steps. One or two are kept in the parse
  // queue (stepsAhead): with two, the queue has the next step while the thread takes the results of the one before.
  struct StepFrame
  {
    VkCommandPool           pool   = VK_NULL_HANDLE;
    VkCommandBuffer         cmd    = VK_NULL_HANDLE;
    VkFence                 fence  = VK_NULL_HANDLE;
    uint64_t                value  = 0;
    uint32_t                serial = 0;
    std::vector<ParseSlot*> slots;
  };
  // One round of the service: the steps kept in the parse queue, the parsed pictures' records copied, the oldest step's
  // results taken (finishPictures), the copied pictures handed over, the slots whose parse is over freed or restarted
  void           runRound();
  size_t         stepsAhead() const;                       // the steps to keep in the parse queue: 1 or 2
  void           submitSteps();                            // a step of the pictures in parse into a free frame
  void           finishPictures( const StepFrame& frame ); // after the step: the pictures done to be copied, others kept
  void           submitCopies();                           // the copy's submission (after the steps that step its slots)
  void           endSlots();                               // the slots no step in flight steps: restarted or freed
  void           handOverCopied();                         // the copied pictures handed over (their copy done)
  void           waitAll();                                // the steps and the copy in flight done (stopping)
  void           handOver( Picture* pic, ParsedPicture* parsed );   // (parsed null: the parse gave up)

  Context*                m_ctx;
  std::function<void()>   m_wakeUp;
  std::thread             m_thread;
  mutable std::mutex      m_mutex;   // guards what the other threads touch: m_waiting, m_released, m_stop, m_stats
  std::condition_variable m_wake;
  bool                    m_stop   = false;
  bool                    m_broken = false;   // the GPU failed: every picture fails
  std::vector<Picture*>   m_waiting;                              // enqueued, not started yet
  std::vector<ParsedPicture*> m_released;                         // given back
  std::vector<std::unique_ptr<ParseSlot>>      m_slots;           // all of them (the thread's)
  std::vector<ParseSlot*> m_free;                                 // reusable (the thread's)
  std::vector<ParseSlot*> m_parsing;                              // the pictures being parsed (the thread's)
  // the pictures parsed, whose records the next round copies from their slots into their results; then the copy in
  // flight, its pictures handed over once it is done (the thread's)
  std::vector<std::pair<ParseSlot*, ParsedPicture*>> m_copying, m_copied;
  // the slots whose parse is over (Overflow, Failed) while a step in flight still steps them: restarted or freed once
  // that step is done (the thread's)
  std::vector<ParseSlot*> m_restarting, m_freeing;
  std::vector<std::unique_ptr<ParsedPicture>>        m_results;   // all of them (the thread's)
  std::vector<ParsedPicture*>                        m_freeResults;   // reusable (the thread's)
  // GPU memory for the records and levels, staged and in the results; a tick for each picture started. A chunk with
  // no records for kIdlePictures pictures is freed: memory the stream no longer needs goes back (the records of
  // larger pictures than now), while a chunk in use now and then stays.
  static constexpr uint64_t kIdlePictures = 64;
  RangeArena                m_records;
  // how many pictures may be in parse and ahead, for the GPU's kind (GpuParseService.cpp's ParseLimits)
  const ParseLimits         m_limits;
  // the most records and levels a picture had so far (startRoom): of the intra pictures, of the others, of the others
  // in each temporal layer
  Counts                  m_mostIntra, m_mostInter, m_mostLayer[MAX_TLAYER];
  // The steps in flight, a ring of two frames: m_inFlight of them (0 to 2: stepsAhead), the oldest at m_firstFrame
  StepFrame               m_frames[2];
  size_t                  m_firstFrame = 0;
  size_t                  m_inFlight   = 0;
  StepFrame&              oldestFrame() { return m_frames[m_firstFrame]; }
  StepFrame&              nextFreeFrame() { return m_frames[( m_firstFrame + m_inFlight ) % 2]; }
  void                    retireOldestFrame()   // after its step is done
  {
    m_firstFrame = ( m_firstFrame + 1 ) % 2;
    m_inFlight--;
  }
  uint32_t                m_serial     = 0;   // the last step's number
  uint64_t                m_doneValue  = 0;   // the parse timeline's value of the last step taken
  VkCommandPool           m_copyPool  = VK_NULL_HANDLE;   // the copy's submission
  VkCommandBuffer         m_copyCmd   = VK_NULL_HANDLE;
  VkFence                 m_copyFence = VK_NULL_HANDLE;
  uint64_t                m_copyValue = 0;   // the parse copies' timeline value of the copy in flight
  uint64_t                m_copyAfter = 0;   // the parse timeline's value the copy in flight waits for
  VkDeviceAddress         m_prof  = 0;   // VVDEC_GPU_PROFILE with the shader clock: the parse's cycle counts
  Buffer                  m_profBuf;
  ParseStats              m_stats;
};

}   // namespace gpu
}   // namespace vvdec
