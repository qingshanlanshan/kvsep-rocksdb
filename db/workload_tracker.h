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

struct KVSepParams {
  double theta0 = 0.0;
  double theta1 = 0.0;
  double theta2 = 0.0;
};
extern KVSepParams global_kvsep_params;


class ParamOptimizer {
public:
  using Vec3 = std::array<double, 3>;

  struct Config {
    // bounds
    Vec3 lo{0.0, -10.0, -10.0};
    Vec3 hi{10.0, 10.0, 10.0};

    // quantization per dimension (0 disables)
    Vec3 quant_step{0.0, 0.0, 0.0};

    // max change of CENTER theta per round (0 disables)
    Vec3 max_center_delta_per_round{0.0, 0.0, 0.0};

    // schedules (one-point SPSA / bandit gradient)
    double a0 = 0.05;
    double c0 = 0.02;
    double A  = 50.0;
    double alpha = 0.602;
    double gamma = 0.101;

    // baseline EMA
    double baseline_beta = 0.2;

    // probability to explore (perturb). (1.0 = always perturb)
    // If <1, sometimes return center to keep baseline honest.
    double explore_prob = 0.8;
  };

  explicit ParamOptimizer()
      : theta_({5, -1, 1}),
        cfg_(),
        rng_(std::random_device{}()),
        coin_(0.5),
        uni_(0.0, 1.0) {
    project(theta_);
  }

  // Get params to run for the NEXT round.
  // You should apply returned params to your system and run until you can measure one latency.
  Vec3 propose_next() {
    // Move to next iteration index (ties ck/ak to proposals)
    ++k_;

    const double ck = c_k();

    exploring_ = (uni_(rng_) < cfg_.explore_prob);
    for (int i = 0; i < 3; ++i) {
      delta_[i] = exploring_ ? (coin_(rng_) ? 1.0 : -1.0) : 0.0;
    }

    last_proposed_ = theta_;
    if (exploring_) {
      for (int i = 0; i < 3; ++i) last_proposed_[i] += ck * delta_[i];
    }

    project(last_proposed_);
    have_pending_observation_ = true;
    return last_proposed_;
  }

  // Provide the latency for the MOST RECENT params returned by propose_next().
  // One call per round.
  void observe_latency(double latency) {
    if (!have_pending_observation_) {
      // No matching propose_next() yet; ignore safely.
      return;
    }
    have_pending_observation_ = false;

    updateBaseline(latency);

    // If we didn't explore (returned center), we don't have direction info; no update.
    if (!exploring_) return;

    const double err = latency - baseline_; // positive => worse than baseline
    const double ak = a_k();
    const double ck = c_k();

    Vec3 next_center = theta_;
    for (int i = 0; i < 3; ++i) {
      // one-point update: theta <- theta - ak * err * delta / ck
      next_center[i] -= ak * err * (delta_[i] / ck);
    }

    capDelta(next_center, theta_, cfg_.max_center_delta_per_round);
    project(next_center);
    theta_ = next_center;
  }

  // Optional getters
  Vec3 center_theta() const { return theta_; }
  double baseline() const { return baseline_; }
  uint64_t iter() const { return k_; }

private:
  Vec3 theta_;
  Config cfg_;

  uint64_t k_ = 0;

  // pending proposal state
  bool have_pending_observation_ = false;
  bool exploring_ = false;
  Vec3 delta_{0.0, 0.0, 0.0};
  Vec3 last_proposed_{0.0, 0.0, 0.0};

  // baseline
  bool baseline_init_ = false;
  double baseline_ = 0.0;

  // rng
  std::mt19937_64 rng_;
  std::bernoulli_distribution coin_;
  std::uniform_real_distribution<double> uni_;

  static double clamp(double x, double l, double h) {
    return std::min(h, std::max(l, x));
  }

  void project(Vec3& v) const {
    for (int i = 0; i < 3; ++i) {
      v[i] = clamp(v[i], cfg_.lo[i], cfg_.hi[i]);
      const double qs = cfg_.quant_step[i];
      if (qs > 0.0) {
        v[i] = std::round(v[i] / qs) * qs;
        v[i] = clamp(v[i], cfg_.lo[i], cfg_.hi[i]);
      }
    }
  }

  void updateBaseline(double L) {
    if (!baseline_init_) {
      baseline_ = L;
      baseline_init_ = true;
    } else {
      const double b = cfg_.baseline_beta;
      baseline_ = (1.0 - b) * baseline_ + b * L;
    }
  }

  double a_k() const {
    return cfg_.a0 / std::pow(cfg_.A + static_cast<double>(k_), cfg_.alpha);
  }
  double c_k() const {
    return cfg_.c0 / std::pow(static_cast<double>(k_), cfg_.gamma);
  }

  static void capDelta(Vec3& next, const Vec3& cur, const Vec3& max_delta) {
    for (int i = 0; i < 3; ++i) {
      if (max_delta[i] <= 0.0) continue;
      next[i] = clamp(next[i], cur[i] - max_delta[i], cur[i] + max_delta[i]);
    }
  }
};
extern ParamOptimizer global_kvsep_param_optimizer;


template <class Key = std::string,
          class Hash = std::hash<Key>,
          class Eq = std::equal_to<Key>,
          size_t kNumShards = 64>
class HotKeyTracker {
 public:
  struct HotKeyStats {
    double score = 0.0;       // stored score at last_write (NOT globally decayed)
    uint64_t last_write = 0;  // global write index when last updated (write)
    int level = -1;           // user-maintained metadata
  };

  struct Snapshot {
    uint64_t write_id = 0;
    double lambda = 0.0;
    // Optional decay LUT; if present, decay(delta) uses LUT for delta <= max.
    const std::vector<double>* decay_lut = nullptr;
    uint32_t decay_lut_max = 0;
  };

  // half_life_writes: after this many writes without updates to a key, its score halves.
  // max_delta_for_lut: if > 0, precompute decay table for [0..max_delta_for_lut].
  explicit HotKeyTracker(double half_life_writes = 1000.0,
                         uint32_t max_delta_for_lut = 200000)
      : global_write_id_(0) {
    set_half_life_writes(half_life_writes);
    if (max_delta_for_lut > 0) {
      build_decay_lut(max_delta_for_lut);
    }
  }

  // Call this on every insert/update for key k.
  // This MUTATES per-key state (score and last_write).
  void update(const Key& k) {
    const uint64_t current = global_write_id_.fetch_add(1, std::memory_order_relaxed) + 1;

    auto& shard = shards_[shard_index(k)];
    std::lock_guard<std::mutex> lock(shard.mu);

    auto& s = shard.map[k];  // creates if missing
    if (s.last_write != 0) {
      const uint64_t delta = current - s.last_write;
      if (delta > 0) s.score *= decay_factor(delta);
    } else {
      s.score = 0.0;
    }

    s.score += 1.0;
    s.last_write = current;
  }

  // Get current hotness of key k (decayed up to current write id),
  // but DOES NOT modify internal stats (read-only).
  // Returns {hotness_score, level}. Missing key => {0.0, 0}.
  std::pair<double, int> hotness(const Key& k) const {
    uint64_t now = global_write_id_.load(std::memory_order_relaxed);

    const auto& shard = shards_[shard_index(k)];
    std::lock_guard<std::mutex> lock(shard.mu);

    auto it = shard.map.find(k);
    if (it == shard.map.end()) return {0.0, 0};

    const HotKeyStats& s = it->second;
    if (s.last_write == 0) return {0.0, s.level};

    const uint64_t delta = now - s.last_write;
    if (delta == 0) return {s.score, s.level};

    return {s.score * decay_factor(delta), s.level};
  }

  // Record level metadata for a key (if present).
  void record_level(const Key& k, int level) {
    auto& shard = shards_[shard_index(k)];
    std::lock_guard<std::mutex> lock(shard.mu);
    auto it = shard.map.find(k);
    if (it != shard.map.end()) it->second.level = level;
  }

  // Writes elapsed since the key was last written.
  // If key is missing, return a large value (treated as cold).
  uint64_t since_last_write(const Key& k) const {
    uint64_t now = global_write_id_.load(std::memory_order_relaxed);

    const auto& shard = shards_[shard_index(k)];
    std::lock_guard<std::mutex> lock(shard.mu);

    auto it = shard.map.find(k);
    if (it == shard.map.end()) return now;  // very cold
    const HotKeyStats& s = it->second;
    if (s.last_write == 0) return now;
    return now - s.last_write;
  }

  // Optional: garbage-collect very cold keys.
  // Decay is computed against current write id, but we DO NOT update the stored stats.
  // (We simply remove if the implied decayed score is below threshold.)
  void gc(double min_score = 0.01) {
    const uint64_t now = global_write_id_.load(std::memory_order_relaxed);

    for (auto& shard : shards_) {
      std::lock_guard<std::mutex> lock(shard.mu);
      for (auto it = shard.map.begin(); it != shard.map.end();) {
        const HotKeyStats& s = it->second;
        double cur = 0.0;
        if (s.last_write != 0) {
          const uint64_t delta = now - s.last_write;
          cur = (delta == 0) ? s.score : (s.score * decay_factor(delta));
        }
        if (cur < min_score) {
          it = shard.map.erase(it);
        } else {
          ++it;
        }
      }
    }
  }

  // Snapshot that compaction threads can use to compute decay without touching internals.
  // You can call snapshot() once per compaction and then do lookups with get_raw_stats().
  Snapshot snapshot() const {
    Snapshot s;
    s.write_id = global_write_id_.load(std::memory_order_relaxed);
    s.lambda = lambda_.load(std::memory_order_relaxed);
    if (!decay_lut_.empty()) {
      s.decay_lut = &decay_lut_;
      s.decay_lut_max = static_cast<uint32_t>(decay_lut_.size() - 1);
    }
    return s;
  }

  // Read-only raw stats (no decay). Returns false if missing.
  bool get_raw_stats(const Key& k, HotKeyStats* out) const {
    const auto& shard = shards_[shard_index(k)];
    std::lock_guard<std::mutex> lock(shard.mu);
    auto it = shard.map.find(k);
    if (it == shard.map.end()) return false;
    *out = it->second;
    return true;
  }

  // Configure half-life (thread-safe). Rebuild LUT if enabled.
  void set_half_life_writes(double half_life_writes) {
    if (half_life_writes <= 0.0) half_life_writes = 1.0;
    const double lambda = std::log(2.0) / half_life_writes;
    lambda_.store(lambda, std::memory_order_relaxed);

    // If LUT enabled, rebuild with same size.
    if (!decay_lut_.empty()) {
      build_decay_lut(static_cast<uint32_t>(decay_lut_.size() - 1));
    }
  }

  // Optional: if you want to know how many keys you track (approx, lock-heavy).
  size_t size() const {
    size_t total = 0;
    for (const auto& shard : shards_) {
      std::lock_guard<std::mutex> lock(shard.mu);
      total += shard.map.size();
    }
    return total;
  }

  // Report stub (your old report measured timing; that was expensive).
  // Prefer external profiling (perf) instead of per-call chrono.
  void report() const {
    // Keep it simple to avoid overhead creep.
    // You can add counters with relaxed atomics if you really need.
  }

 private:
  struct Shard {
    mutable std::mutex mu;
    std::unordered_map<Key, HotKeyStats, Hash, Eq> map;
  };

  static_assert((kNumShards & (kNumShards - 1)) == 0,
                "kNumShards must be power of two for fast masking.");

  size_t shard_index(const Key& k) const {
    size_t h = hasher_(k);
    return h & (kNumShards - 1);
  }

  // Decay factor for integer delta writes: exp(-lambda * delta),
  // optionally using LUT when available.
  double decay_factor(uint64_t delta) const {
    if (!decay_lut_.empty()) {
      const uint32_t max = static_cast<uint32_t>(decay_lut_.size() - 1);
      if (delta <= max) return decay_lut_[static_cast<size_t>(delta)];
      return 0.0;
    }
    const double lambda = lambda_.load(std::memory_order_relaxed);
    return std::exp(-lambda * static_cast<double>(delta));
  }

  void build_decay_lut(uint32_t max_delta) {
    decay_lut_.assign(static_cast<size_t>(max_delta) + 1, 0.0);
    const double lambda = lambda_.load(std::memory_order_relaxed);
    decay_lut_[0] = 1.0;
    for (uint32_t d = 1; d <= max_delta; ++d) {
      decay_lut_[d] = std::exp(-lambda * static_cast<double>(d));
    }
  }

  std::array<Shard, kNumShards> shards_;
  Hash hasher_{};

  std::atomic<uint64_t> global_write_id_;
  std::atomic<double> lambda_;

  // Optional LUT to avoid exp() on frequent lookups.
  std::vector<double> decay_lut_;
};
extern HotKeyTracker<>* global_read_hotness_tracker;
extern HotKeyTracker<>* global_write_hotness_tracker;

struct LearningStats {
  struct Workload {
    size_t n_write = 0;
    size_t n_read = 0;
    size_t n_scan = 0;
    size_t value_size = 0;   // sum of write value sizes
    size_t scan_length = 0;  // sum of scan lengths
  } workload;

  float total_op_time = 0.0f;      // sum of op latency (whatever unit you use)
  uint64_t op_count = 0;           // number of ops recorded
  size_t window_size = (size_t)1e5;
  bool compaction_triggered = false;

  // Store the compaction output level for the *most recent* compaction trigger.
  // You should set this when compaction is triggered.
  int last_output_level = 0;

  mutable std::mutex mutex_;

  // -------- record APIs (same style as your original) --------
  void mark_compaction(int output_level) {
    std::lock_guard<std::mutex> lock(mutex_);
    compaction_triggered = true;
    last_output_level = output_level;
  }

  void record_read(float op_time) {
    std::lock_guard<std::mutex> lock(mutex_);
    total_op_time += op_time;
    op_count++;
    workload.n_read++;
  }

  void record_write(float op_time, size_t value_sz) {
    std::lock_guard<std::mutex> lock(mutex_);
    total_op_time += op_time;
    op_count++;
    workload.n_write++;
    workload.value_size += value_sz;
  }

  void record_scan(float op_time, size_t length) {
    std::lock_guard<std::mutex> lock(mutex_);
    total_op_time += op_time;
    op_count++;
    workload.n_scan++;
    workload.scan_length += length;
  }

  // -------- RL-friendly snapshot --------
  struct Snapshot {
    // workload fractions
    float p_write = 0.0f;
    float p_read = 0.0f;
    float p_scan = 0.0f;

    // averages (raw)
    float avg_value_size = 0.0f;
    float avg_scan_len = 0.0f;

    // normalized versions (0..1) using constants you can tune
    float avg_value_norm = 0.0f;
    float avg_scan_norm = 0.0f;

    // performance signal
    float avg_latency = 0.0f;

    // context
    int output_level = 0;
  };

  // Returns true and fills `out` when:
  //   - compaction_triggered == true, AND
  //   - you have at least some ops, AND
  //   - (optionally) op_count >= window_size
  //
  // Also resets the window when it returns true.
  bool take_snapshot_if_ready(Snapshot* out) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!compaction_triggered) return false;
    if (op_count == 0) {  // nothing to learn from
      compaction_triggered = false;
      return false;
    }

    // Optional: require enough samples before updating
    if (op_count < window_size) {
      // You can choose to return false here if you want strict windows.
      // For now we allow update immediately after compaction trigger.
      // return false;
    }

    const float nW = static_cast<float>(workload.n_write);
    const float nR = static_cast<float>(workload.n_read);
    const float nS = static_cast<float>(workload.n_scan);
    const float tot = std::max(1.0f, nW + nR + nS);

    Snapshot s;
    s.p_write = nW / tot;
    s.p_read  = nR / tot;
    s.p_scan  = nS / tot;

    s.avg_latency = total_op_time / static_cast<float>(op_count);

    s.avg_value_size = (workload.n_write > 0)
        ? static_cast<float>(workload.value_size) /
              static_cast<float>(workload.n_write)
        : 0.0f;

    s.avg_scan_len = (workload.n_scan > 0)
        ? static_cast<float>(workload.scan_length) /
              static_cast<float>(workload.n_scan)
        : 0.0f;

    // Normalization constants (tune for your system)
    constexpr float kValueNormDenom = 8192.0f;  // 8KB
    constexpr float kScanNormDenom  = 1024.0f;  // 1K keys/entries

    s.avg_value_norm = std::min(1.0f, s.avg_value_size / kValueNormDenom);
    s.avg_scan_norm  = std::min(1.0f, s.avg_scan_len / kScanNormDenom);

    s.output_level = last_output_level;

    // Output
    *out = s;

    // Reset window
    workload = {};
    total_op_time = 0.0f;
    op_count = 0;
    compaction_triggered = false;

    return true;
  }
};

extern LearningStats global_rl_stats;


class KvSepThresholdModel {
public:
  // Threshold actions (same buckets as before)
  static constexpr uint32_t kNumActions = 8;
  static constexpr uint32_t kMaxLevel = 8;

  // Discretization bins (tune later; keep as-is for now)
  static constexpr uint32_t kWorkloadBuckets = 16;
  static constexpr uint32_t kHotBins = 32;

  // policy size = workload_bucket * level * read_bin * write_bin
  static constexpr uint32_t kPolicySize =
      kWorkloadBuckets * kMaxLevel * kHotBins * kHotBins;

  KvSepThresholdModel() {
    // default policy: action 3 (4096) everywhere
    policy_.fill(3);
    workload_bucket_.store(0, std::memory_order_relaxed);
  }

  // Update called periodically (your "every several operations" path)
  // For now: only computes workload bucket; policy stays default.
  // (In the next step, we'll learn/populate policy_, but you can test speed now.)
  void update_from_snapshot(const struct LearningStats::Snapshot& snap) {
    workload_bucket_.store(make_workload_bucket_(snap), std::memory_order_relaxed);
  }

  // HOT PATH: per key during compaction
  // O(1) work: bins + index + array load
  uint32_t get_threshold_bytes(double read_hotness,
                               double write_hotness,
                               int output_level) const {
    uint32_t wb = workload_bucket_.load(std::memory_order_relaxed);

    uint32_t L = (output_level < 0) ? 0u : (output_level >= (int)kMaxLevel ? (kMaxLevel - 1) : (uint32_t)output_level);
    uint32_t rb = hot_bin_(read_hotness);
    uint32_t wb2 = hot_bin_(write_hotness);

    uint32_t idx = (((wb * kMaxLevel + L) * kHotBins + rb) * kHotBins + wb2);
    uint8_t a = policy_[idx];   // 0..7
    return buckets_[a];
  }

private:
  // Normal member buckets (no static definition hassles)
  const std::array<uint32_t, kNumActions> buckets_ = {
      0u, 256u, 1024u, 4096u, 8192u, 16384u, 32768u, 65536u
  };

  // Precomputed action table
  std::array<uint8_t, kPolicySize> policy_;

  // Current workload bucket (updated periodically)
  std::atomic<uint32_t> workload_bucket_{0};

private:
  // Very cheap hotness binning.
  // IMPORTANT: This assumes your hotness score is usually within ~[0, 128].
  // If your scores are larger, it just saturates.
  static uint32_t hot_bin_(double h) {
    constexpr double h_max = 128.0;
    if (h <= 0.0) return 0;
    if (h >= h_max) return kHotBins - 1;
    return static_cast<uint32_t>(h * (kHotBins - 1) / h_max);
  }

  // Map workload snapshot -> [0, 15]
  // For now: 4 bits = dominant op type (read/write/scan) + scan-length high/low
  static uint32_t make_workload_bucket_(const struct LearningStats::Snapshot& s) {
    // dominant op type (2 bits)
    uint32_t dom = 0; // 0=read, 1=write, 2=scan
    if (s.p_write >= s.p_read && s.p_write >= s.p_scan) dom = 1;
    else if (s.p_scan >= s.p_read && s.p_scan >= s.p_write) dom = 2;

    // scan length bit (1 bit)
    uint32_t long_scan = (s.avg_scan_norm > 0.5f) ? 1u : 0u;

    // value size bit (1 bit) (optional signal)
    uint32_t large_val = (s.avg_value_norm > 0.5f) ? 1u : 0u;

    // 2 + 1 + 1 = 4 bits => 0..15
    return (dom << 2) | (long_scan << 1) | large_val;
  }
};
extern KvSepThresholdModel global_kvsep_model;

class LRU {
public:
    explicit LRU(size_t capacity) : capacity_(capacity), head_(nullptr), tail_(nullptr) {}

    ~LRU() {
        Node* current = head_;
        while (current) {
            Node* next = current->next;
            delete current;
            current = next;
        }
    }

    std::string access(const std::string& key) {
      auto it = map_.find(key);
      if (it != map_.end()) {
        Node* node = it->second;
        remove(node);
        insert_at_front(node);
      } else {
        Node* new_node = new Node(key);
        insert_at_front(new_node);
        map_[key] = new_node;

        if (map_.size() > capacity_) {
          // Evict least recently used
          assert(tail_ != nullptr);
          auto result = tail_->key;
          map_.erase(tail_->key);
          auto node = tail_;
          remove(node);
          delete node;
          return result;
        }
      }
      return "";
    }

    bool contains(const std::string& key) const {
        return map_.find(key) != map_.end();
    }

    std::string peek_tail() const {
        if (tail_) {
            return tail_->key;
        }
        return "";
    }

    bool is_full() const {
        return map_.size() >= capacity_;
    }
private:
    struct Node {
        std::string key;
        Node* prev;
        Node* next;
        Node(const std::string& k) : key(k), prev(nullptr), next(nullptr) {}
    };

    size_t capacity_;
    std::unordered_map<std::string, Node*> map_;
    Node* head_;
    Node* tail_;

    void remove(Node* node) {
        if (node->prev) {
            node->prev->next = node->next;
        } else {
            head_ = node->next;
        }
        if (node->next) {
            node->next->prev = node->prev;
        } else {
            tail_ = node->prev;
        }
    }

    void insert_at_front(Node* node) {
        node->next = head_;
        node->prev = nullptr;
        if (head_) {
            head_->prev = node;
        }
        head_ = node;
        if (!tail_) {
            tail_ = head_;
        }
    }
};

class SLRU {
public:
    SLRU(size_t capacity, float protected_fraction = 0.8f)
        : protected_capacity_(static_cast<size_t>(capacity * protected_fraction)),
          probationary_capacity_(capacity - protected_capacity_),
          protected_(protected_capacity_),
          probationary_(probationary_capacity_) {}
    std::string access(const std::string& key) {
        if (protected_.contains(key)) {
            protected_.access(key);
            return "";
        } else if (probationary_.contains(key)) {
            // probationary_.access(key);
            // Promote to protected
            auto victim = protected_.access(key);
            if (!victim.empty()) {
                return probationary_.access(victim);
            }
            return "";
        } else {
            // New entry goes to probationary
            return probationary_.access(key);
        }
    }
    bool contains(const std::string& key) const {
        return protected_.contains(key) || probationary_.contains(key);
    }

    std::string peek_tail() {
        return probationary_.peek_tail();
    }

    bool is_full() const {
        return protected_.is_full() && probationary_.is_full();
    }

private:
    size_t protected_capacity_;
    size_t probationary_capacity_;
    LRU protected_;
    LRU probationary_;
};

#if true
class TinyLFU {
public:
  TinyLFU(size_t capacity, float window_fraction = 0.01f)
      : window_cache_(static_cast<size_t>(capacity * window_fraction)),
        main_cache_(capacity - static_cast<size_t>(capacity * window_fraction)) {}
  std::string access(const std::string& key) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    return access_no_lock_(key);
  }

  void access(const std::vector<std::string>& keys) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    for (const auto& key : keys) {
      access_no_lock_(key);
    }
  }

  bool contains(const std::string& key) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return main_cache_.contains(key) || window_cache_.contains(key);
  }

private:
  LRU window_cache_;
  SLRU main_cache_;
  std::unordered_map<std::string, int> frequency_table_;
  mutable std::shared_mutex mutex_;

  std::string access_no_lock_(const std::string& key) {
    if (main_cache_.contains(key)) {
      main_cache_.access(key);
      return "";
    } else if (window_cache_.contains(key)) {
      frequency_table_[key]++;
      window_cache_.access(key);
      if (frequency_table_[key] >= 8) {
        reset_frequency();
      }
      return "";
    } else {
      auto window_victim = window_cache_.access(key);
      frequency_table_[key] = 1;

      if (window_victim.empty()) return "";

      if (!main_cache_.is_full()) {
        return main_cache_.access(window_victim);
      }

      auto main_victim = main_cache_.peek_tail();
      if (frequency_table_[window_victim] > frequency_table_[main_victim]) {
        return main_cache_.access(window_victim);
      }
      return window_victim;
    }
  }

  void reset_frequency() {
    for (auto it = frequency_table_.begin(); it != frequency_table_.end(); ) {
      it->second /= 2;  // halve frequency
      if (it->second == 0) {
        it = frequency_table_.erase(it);
      } else {
        ++it;
      }
    }
  }
};
#else
class TinyLFU { 
public:
  TinyLFU(size_t capacity, float window_fraction = 0.01f)
      : window_cache_(static_cast<size_t>(capacity * window_fraction)),
        main_cache_(capacity -
                    static_cast<size_t>(capacity * window_fraction)) {}
  std::string access(const std::string& key) {
    // std::lock_guard<std::mutex> lock(mutex_);
    if (main_cache_.contains(key)) {
      main_cache_.access(key);
      return "";
    } else if (window_cache_.contains(key)) {
      window_cache_.access(key);
      frequency_table_[key]++;
      return "";
    } else {
      auto window_victim = window_cache_.access(key);
      frequency_table_[key] = 1;
      if (window_victim.empty()) return "";
      if (!main_cache_.is_full()) {
        return main_cache_.access(window_victim);
      }
      auto main_victim = main_cache_.peek_tail();
      if (frequency_table_[window_victim] > frequency_table_[main_victim]) {
        return main_cache_.access(window_victim);
      }
      return window_victim;
    }
    if (frequency_table_[key] > 8) {
      reset_frequency();
    }
  }
  bool contains(const std::string& key) const {
    // std::lock_guard<std::mutex> lock(mutex_);
    return main_cache_.contains(key) || window_cache_.contains(key);
  }

 private:
  LRU window_cache_;
  SLRU main_cache_;
  std::unordered_map<std::string, int> frequency_table_;
  mutable std::mutex mutex_;
  void reset_frequency() {
    for (auto it = frequency_table_.begin(); it != frequency_table_.end();) {
      it->second /= 2;  // halve frequency
      if (it->second == 0) {
        it = frequency_table_.erase(it);
      } else {
        ++it;
      }
    }
  }
};
#endif
extern TinyLFU* global_frequent_write_key_cache;
extern TinyLFU* global_frequent_read_key_cache;

struct FetchBlobCompaction {
void compact_file_and_fetch_blob_value(rocksdb::DB* db, std::vector<std::string>& files, std::vector<std::string>& user_keys, int level) {
  std::lock_guard<std::mutex> lock(mutex_);
  user_keys_.clear();
  for (const auto& key : user_keys) {
    user_keys_.push_back(key);
  }
  db->CompactFiles(rocksdb::CompactionOptions(), files, level);
}
private:
  std::mutex mutex_;
  std::vector<std::string> user_keys_;
};

}  // namespace ROCKSDB_NAMESPACE