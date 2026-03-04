#pragma once

#include "rocksdb/rocksdb_namespace.h"
#include "rocksdb/db.h"

#include <cmath>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <array>
#include <random>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <atomic>
#include <cassert>
#include <shared_mutex>
#include <thread>
#include <iomanip>
#include "rocksdb/cache.h"
#include "rocksdb/advanced_cache.h"
#include "rocksdb/compaction_filter.h"
#include "rocksdb/options.h"
#include "rocksdb/statistics.h"
#include <memory>
#include <queue>

#define OWN_CACHE_IMPL false
#define LRU_KEY_TYPE size_t
namespace ROCKSDB_NAMESPACE {

struct ReadHitLevel {
  std::vector<int> level_hits;  // number of hits per level
  int total_hits = 0;           // total number of hits

  void record_hit(size_t level) {
    if (level >= level_hits.size()) {
      level_hits.resize(level + 1, 0);
    }
    level_hits[level]++;
    total_hits++;
  }

  void report() const {
    std::cout << "Read Hit Levels Report:" << std::endl;
    for (size_t i = 0; i < level_hits.size(); i++) {
      double fraction = (total_hits > 0) ? (static_cast<double>(level_hits[i]) / total_hits) : 0.0;
      std::cout << "  Level " << i << ": " << level_hits[i]
                << " hits (" << (fraction * 100.0) << "%)" << std::endl;
    }
    std::cout << "  Total Hits: " << total_hits << std::endl;
  }
};
extern ReadHitLevel global_read_hit_level;

extern std::shared_ptr<Cache> global_frequent_write_key_cache;
extern std::shared_ptr<Cache> global_frequent_read_key_cache;
extern rocksdb::Cache::CacheItemHelper* global_cache_item_helper;


class RotateCMS {
  using CELL_STORAGE = uint16_t;   // can hold up to 16-bit effective counters
public:
  RotateCMS(size_t width, size_t depth, uint32_t counter_bits = 10)
      : width_(width),
        depth_(depth),
        mask_(is_power_of_two(width) ? (width - 1) : 0),
        active_(0),
        rotate_requested_(false),
        bits_(counter_bits),
        max_(compute_max(counter_bits)) {
    if (bits_ == 0 || bits_ > 16) {
      throw std::invalid_argument("RotateCMS: counter_bits must be in [1,16] for uint16_t storage");
    }

    for (int i = 0; i < 2; ++i) {
      tables_[i].n = width_ * depth_;
      tables_[i].counters = std::make_unique<std::atomic<CELL_STORAGE>[]>(tables_[i].n);
      for (size_t j = 0; j < tables_[i].n; ++j) {
        tables_[i].counters[j].store(0, std::memory_order_relaxed);
      }
      tables_[i].total.store(0, std::memory_order_relaxed);
    }
  }

  // update(): increments ACTIVE table by 1 for this key (saturating at max_)
  void update(std::string_view key) {
    for (;;) {
      const int a = active_.load(std::memory_order_acquire);
      const uint64_t h1 = hash64(key, kSeed1);
      const uint64_t h2 = hash64(key, kSeed2) | 1ULL;

      bool saturated = false;
      for (size_t r = 0; r < depth_; ++r) {
        const size_t idx = index_for_row(r, h1, h2);
        if (!inc_saturating(tables_[a].counters[idx], max_)) {
          saturated = true;
          break;
        }
      }

      // Count this update even if some counters saturated?
      // Two options:
      // (1) Count only if all increments succeeded (as before): keeps "total" aligned with true increments.
      // (2) Always count: makes total reflect event volume even under saturation.
      //
      // For your current design (overflow->rotate), keep behavior (1):
      if (!saturated) {
        tables_[a].total.fetch_add(1, std::memory_order_relaxed);
        return;
      }

      // A counter hit max_ -> rotate and retry
      request_rotate();
    }
  }

  // get(): reads INACTIVE table only; returns est/total_in_inactive
  double get(std::string_view key) const {
    const int a = active_.load(std::memory_order_acquire);
    const int b = a ^ 1;

    const uint64_t total = tables_[b].total.load(std::memory_order_relaxed);
    if (total == 0) return 0.0; // no data yet -> treat as 0-frequency

    const uint64_t h1 = hash64(key, kSeed1);
    const uint64_t h2 = hash64(key, kSeed2) | 1ULL;

    // est stored in a wider type to avoid issues when bits_ close to 16.
    uint32_t est = static_cast<uint32_t>(max_);
    for (size_t r = 0; r < depth_; ++r) {
      const size_t idx = index_for_row(r, h1, h2);
      const uint32_t v =
          static_cast<uint32_t>(tables_[b].counters[idx].load(std::memory_order_relaxed));
      est = std::min(est, v);
    }

    return static_cast<double>(est) / static_cast<double>(total);
  }

  uint64_t inactive_total_updates() const {
    const int a = active_.load(std::memory_order_acquire);
    const int b = a ^ 1;
    return tables_[b].total.load(std::memory_order_relaxed);
  }

  void report() const {
    std::cout << "RotateCMS Report:\n";
    std::cout << "  Active Table: " << active_.load(std::memory_order_acquire) << "\n";
    std::cout << "  Rotate Count: " << rotate_count_ << "\n";
    std::cout << "  Counter bits: " << bits_ << " (max=" << max_ << ")\n";
    for (int i = 0; i < 2; ++i) {
      std::cout << "  Table " << i << ": total updates = "
                << tables_[i].total.load(std::memory_order_relaxed) << "\n";
    }
  }

private:
  struct Table {
    size_t n = 0;
    std::unique_ptr<std::atomic<CELL_STORAGE>[]> counters;
    std::atomic<uint64_t> total{0};
  };

  void request_rotate() {
    bool expected = false;
    if (!rotate_requested_.compare_exchange_strong(expected, true,
                                                  std::memory_order_acq_rel,
                                                  std::memory_order_relaxed)) {
      std::this_thread::yield();
      return;
    }

    std::unique_lock<std::mutex> lk(rotate_mu_, std::try_to_lock);
    if (!lk.owns_lock()) {
      rotate_requested_.store(false, std::memory_order_release);
      std::this_thread::yield();
      return;
    }

    rotate_impl();
    rotate_requested_.store(false, std::memory_order_release);
  }

  void rotate_impl() {
    const int a = active_.load(std::memory_order_acquire);
    const int b = a ^ 1;

    clear_table(b);
    active_.store(b, std::memory_order_release);
    rotate_count_++;
  }

  void clear_table(int which) {
    auto& tab = tables_[which];
    for (size_t i = 0; i < tab.n; ++i) {
      tab.counters[i].store(0, std::memory_order_relaxed);
    }
    tab.total.store(0, std::memory_order_relaxed);
  }

  static uint32_t compute_max(uint32_t bits) {
    // bits in [1,16] here. For bits==16, (1u<<16) is OK in 32-bit.
    return (bits == 32) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
  }

  // returns true if incremented; false if already at max_val
  static bool inc_saturating(std::atomic<CELL_STORAGE>& c, uint32_t max_val) {
    for (;;) {
      CELL_STORAGE old = c.load(std::memory_order_relaxed);
      if (old >= static_cast<CELL_STORAGE>(max_val)) return false;
      const CELL_STORAGE next = static_cast<CELL_STORAGE>(old + 1);
      if (c.compare_exchange_weak(old, next,
                                  std::memory_order_relaxed,
                                  std::memory_order_relaxed)) {
        return true;
      }
    }
  }

  // ---- hashing + indexing ----
  static constexpr uint64_t kSeed1 = 0x9e3779b97f4a7c15ULL;
  static constexpr uint64_t kSeed2 = 0xbf58476d1ce4e5b9ULL;

  static uint64_t hash64(std::string_view s, uint64_t seed) {
    uint64_t h = 1469598103934665603ULL ^ seed;
    for (unsigned char c : s) { h ^= (uint64_t)c; h *= 1099511628211ULL; }
    h ^= (h >> 30); h *= 0xbf58476d1ce4e5b9ULL;
    h ^= (h >> 27); h *= 0x94d049bb133111ebULL;
    h ^= (h >> 31);
    return h;
  }

  static bool is_power_of_two(size_t x) {
    return x && ((x & (x - 1)) == 0);
  }

  inline size_t index_for_row(size_t r, uint64_t h1, uint64_t h2) const {
    const uint64_t x = h1 + (uint64_t)r * h2;
    const size_t col = (mask_ != 0) ? (size_t)(x & mask_) : (size_t)(x % width_);
    return r * width_ + col;
  }

private:
  const size_t width_;
  const size_t depth_;
  const size_t mask_;

  Table tables_[2];

  std::atomic<int> active_;
  std::mutex rotate_mu_;
  std::atomic<bool> rotate_requested_;
  uint32_t rotate_count_ = 0;

  const uint32_t bits_;   // effective counter bits requested by user
  const uint32_t max_;    // saturating max value = (1<<bits_)-1
};
extern RotateCMS* global_read_frequency_tracker;
extern RotateCMS* global_write_frequency_tracker;

class FreqBucketer {
public:
  static constexpr int kNumThresholds = 5;
  static constexpr int kNumBuckets = kNumThresholds + 1;

  explicit FreqBucketer(const std::array<double, kNumThresholds>& thresholds)
      : th_(thresholds) {}

  inline int Bucket(double f) const {
    if (!(f > 0.0)) return 0; // handles 0, negative, NaN
    for (int i = 0; i < kNumThresholds; ++i) {
      if (f < th_[i]) return i;
    }
    return kNumThresholds;
  }

private:
  std::array<double, kNumThresholds> th_;
};
class KvSeparationPolicy {
public:
  static constexpr int R = FreqBucketer::kNumBuckets;
  static constexpr int W = FreqBucketer::kNumBuckets;

  inline KvSeparationPolicy(const RotateCMS* read_tracker,
                            const RotateCMS* write_tracker,
                            const std::array<double, FreqBucketer::kNumThresholds>& read_thresholds,
                            const std::array<double, FreqBucketer::kNumThresholds>& write_thresholds)
      : read_tracker_(read_tracker),
        write_tracker_(write_tracker),
        read_bucketer_(read_thresholds),
        write_bucketer_(write_thresholds) {
    InitStaticTable();
    InitCounters();

    // Exploration defaults: OFF
    epsilon_ppm_.store(0, std::memory_order_relaxed);         // 0 = no exploration
    min_bucket_hits_to_explore_.store(1000, std::memory_order_relaxed); // avoid noise early
  }

  // epsilon in parts-per-million (ppm): 10_000 ppm = 1%, 50_000 ppm = 5%
  inline void SetExploration(uint32_t epsilon_ppm, uint64_t min_bucket_hits = 1000) {
    epsilon_ppm_.store(epsilon_ppm, std::memory_order_relaxed);
    min_bucket_hits_to_explore_.store(min_bucket_hits, std::memory_order_relaxed);
  }

  inline uint32_t GetExplorationPPM() const {
    return epsilon_ppm_.load(std::memory_order_relaxed);
  }

  // Main API: call from BlobDB placement logic.
  inline bool ShouldSeparate(std::string_view user_key) const {
    const double rf = read_tracker_->get(user_key);
    const double wf = write_tracker_->get(user_key);

    if (rf < 0.0 || wf < 0.0) {
      return true;
    }

    const int rb = read_bucketer_.Bucket(rf);
    const int wb = write_bucketer_.Bucket(wf);

    bucket_hits_[rb][wb].fetch_add(1, std::memory_order_relaxed);

    bool sep = (decision_[rb][wb].load(std::memory_order_relaxed) != 0);
    // ---------- Step 3: epsilon exploration ----------
    const uint32_t eps = epsilon_ppm_.load(std::memory_order_relaxed);
    if (eps != 0) {
      const uint64_t hits = bucket_hits_[rb][wb].load(std::memory_order_relaxed);
      const uint64_t min_hits = min_bucket_hits_to_explore_.load(std::memory_order_relaxed);

      // Only explore buckets that already have enough traffic (reduces variance).
      if (hits >= min_hits) {
        if (RandomBelow1e6() < eps) {
          // Flip decision for this call.
          sep = !sep;

          explore_hits_[rb][wb].fetch_add(1, std::memory_order_relaxed);
          if (sep) explore_flip_sep_[rb][wb].fetch_add(1, std::memory_order_relaxed);
        }
      }
    }
    // -----------------------------------------------
    if (sep) {
      separate_hits_[rb][wb].fetch_add(1, std::memory_order_relaxed);
    }
    return sep;
  }

  inline void SetDecision(int read_bucket, int write_bucket, bool separate) {
    decision_[read_bucket][write_bucket].store(separate ? 1u : 0u, std::memory_order_relaxed);
  }

  inline bool GetDecision(int read_bucket, int write_bucket) const {
    return decision_[read_bucket][write_bucket].load(std::memory_order_relaxed) != 0;
  }

  inline void Report(bool reset_after = false) const {
    std::cout << "==== KV Separation Policy Report ====\n";

    uint64_t total_hits = 0;
    uint64_t total_sep = 0;
    uint64_t total_explore = 0;

    const uint32_t eps = epsilon_ppm_.load(std::memory_order_relaxed);
    const uint64_t min_hits = min_bucket_hits_to_explore_.load(std::memory_order_relaxed);
    std::cout << "Exploration: epsilon=" << eps << " ppm"
              << " (=" << (eps / 10000.0) << "%)"
              << ", min_bucket_hits=" << min_hits << "\n\n";

    // Header
    std::cout << std::setw(6) << "R\\W";
    for (int w = 0; w < W; ++w) {
      std::cout << std::setw(22) << ("W" + std::to_string(w));
    }
    std::cout << "\n";

    for (int r = 0; r < R; ++r) {
      std::cout << std::setw(6) << ("R" + std::to_string(r));

      for (int w = 0; w < W; ++w) {
        const uint64_t hits = bucket_hits_[r][w].load(std::memory_order_relaxed);
        const uint64_t sep_hits = separate_hits_[r][w].load(std::memory_order_relaxed);
        const bool decision = decision_[r][w].load(std::memory_order_relaxed) != 0;

        const uint64_t exp = explore_hits_[r][w].load(std::memory_order_relaxed);

        total_hits += hits;
        total_sep += sep_hits;
        total_explore += exp;

        const double sep_ratio = (hits > 0) ? double(sep_hits) / double(hits) : 0.0;
        const double exp_ratio = (hits > 0) ? double(exp) / double(hits) : 0.0;

        std::ostringstream cell;
        // format: hits|S/I|sepRatio|expRatio
        cell << hits << "|"
             << (decision ? "S" : "I") << "|"
             << std::fixed << std::setprecision(3)
             << sep_ratio << "|"
             << std::fixed << std::setprecision(3)
             << exp_ratio;

        std::cout << std::setw(22) << cell.str();
      }
      std::cout << "\n";
    }

    std::cout << "\nTotal hits      : " << total_hits << "\n";
    std::cout << "Total separated : " << total_sep << "\n";
    std::cout << "Global sep ratio: "
              << (total_hits ? double(total_sep) / double(total_hits) : 0.0) << "\n";
    std::cout << "Total explored  : " << total_explore << "\n";
    std::cout << "Global exp ratio: "
              << (total_hits ? double(total_explore) / double(total_hits) : 0.0) << "\n";

    std::cout << "\n Read Tracker:\n";
    read_tracker_->report();
    std::cout << "\n Write Tracker:\n";
    write_tracker_->report();
    std::cout << "=====================================\n";

    if (reset_after) ResetCounters();
  }

  inline void ResetCounters() const {
    for (int r = 0; r < R; ++r) {
      for (int w = 0; w < W; ++w) {
        bucket_hits_[r][w].store(0, std::memory_order_relaxed);
        separate_hits_[r][w].store(0, std::memory_order_relaxed);
        explore_hits_[r][w].store(0, std::memory_order_relaxed);
        explore_flip_sep_[r][w].store(0, std::memory_order_relaxed);
      }
    }
  }

private:
  inline void InitStaticTable() {
    // Default: separate everywhere
    for (int r = 0; r < R; ++r) {
      for (int w = 0; w < W; ++w) {
        decision_[r][w].store(1u, std::memory_order_relaxed); // 1 = SEPARATE
      }
    }

    // Override: High R + Low W => INLINE
    const int r_high_min = R - 2;
    const int w_low_max  = 1;
    for (int r = r_high_min; r < R; ++r) {
      for (int w = 0; w <= w_low_max; ++w) {
        decision_[r][w].store(0u, std::memory_order_relaxed);
      }
    }

    // Extra protection: extremely read-hot => always INLINE
    const int r_very_high = R - 1;
    for (int w = 0; w < W - 1; ++w) {
      decision_[r_very_high][w].store(0u, std::memory_order_relaxed);
    }
  }

  inline void InitCounters() const {
    for (int r = 0; r < R; ++r) {
      for (int w = 0; w < W; ++w) {
        bucket_hits_[r][w].store(0, std::memory_order_relaxed);
        separate_hits_[r][w].store(0, std::memory_order_relaxed);
        explore_hits_[r][w].store(0, std::memory_order_relaxed);
        explore_flip_sep_[r][w].store(0, std::memory_order_relaxed);
      }
    }
  }

  // Fast thread-local PRNG: returns [0, 1e6).
  // (Used only for exploration decision. Very low overhead.)
  static inline uint32_t RandomBelow1e6() {
    thread_local uint64_t x = 0x9e3779b97f4a7c15ULL ^
                              (uint64_t)(uintptr_t)&x ^
                              (uint64_t)std::hash<std::thread::id>{}(std::this_thread::get_id());

    // xorshift64*
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    uint64_t r = x * 2685821657736338717ULL;

    return static_cast<uint32_t>(r % 1000000ULL);
  }

private:
  const RotateCMS* read_tracker_;
  const RotateCMS* write_tracker_;

  FreqBucketer read_bucketer_;
  FreqBucketer write_bucketer_;

  std::atomic<uint8_t> decision_[R][W];

  mutable std::atomic<uint64_t> bucket_hits_[R][W];
  mutable std::atomic<uint64_t> separate_hits_[R][W];

  // Step 3 counters:
  mutable std::atomic<uint64_t> explore_hits_[R][W];      // how many times we flipped in this bucket
  mutable std::atomic<uint64_t> explore_flip_sep_[R][W];  // among flips, how many resulted in SEPARATE

  // Exploration config:
  // epsilon in ppm, stored atomically to allow runtime change
  std::atomic<uint32_t> epsilon_ppm_;
  std::atomic<uint64_t> min_bucket_hits_to_explore_;
};
extern KvSeparationPolicy* global_kv_sep_policy;

struct LearningStatistics {
  uint64_t n_writes = 0;
  double t_writes_us = 0.0;
  uint64_t n_reads = 0;
  double t_reads_us = 0.0;
  uint64_t cur_compaction_read_bytes = 0;
  uint64_t last_compaction_read_bytes = 0;
  void record_write(double latency_us) {
    n_writes++;
    t_writes_us += latency_us;
  }
  void record_read(double latency_us) {
    n_reads++;
    t_reads_us += latency_us;
  }
  void record_compaction_read_bytes(DB* db) {
    cur_compaction_read_bytes = db->GetOptions().statistics->getTickerCount(
        rocksdb::COMPACT_READ_BYTES);
  }
  void reset() {
    n_writes = 0;
    t_writes_us = 0.0;
    n_reads = 0;
    t_reads_us = 0.0;
    last_compaction_read_bytes = cur_compaction_read_bytes;
    cur_compaction_read_bytes = 0;
  }
  double get_throughput() const {
    double total_time_s = (t_writes_us + t_reads_us) / 1e6;
    if (total_time_s == 0.0) {
      return 0.0;
    }
    return (n_writes + n_reads) / total_time_s;
  }
  uint64_t get_compaction_read_bytes() const {
    return cur_compaction_read_bytes - last_compaction_read_bytes;
  }
  void report() const {
    std::cout << "==== Interval Statistics Report ====\n";
    std::cout << "Writes: " << n_writes << ", total time (s): " << (t_writes_us / 1e6)
              << ", latency: " << (t_writes_us / n_writes) << " us\n";
    std::cout << "Reads: " << n_reads << ", total time (s): " << (t_reads_us / 1e6)
              << ", latency: " << (t_reads_us / n_reads) << " us\n";
    std::cout << "Compaction read bytes: " << get_compaction_read_bytes() << "\n";
    std::cout << "Overall throughput (ops/s): " << get_throughput() << "\n";
    std::cout << "=====================================\n";
  }
};
extern LearningStatistics global_learning_stats;
}  // namespace ROCKSDB_NAMESPACE