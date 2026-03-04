#include "db/workload_tracker.h"

namespace ROCKSDB_NAMESPACE {
ReadHitLevel global_read_hit_level;

// disables hot key tracker and RL model if not nullptr
std::shared_ptr<Cache> global_frequent_write_key_cache = nullptr;
std::shared_ptr<Cache> global_frequent_read_key_cache = nullptr;
rocksdb::Cache::CacheItemHelper* global_cache_item_helper = nullptr;

RotateCMS* global_read_frequency_tracker = nullptr;
RotateCMS* global_write_frequency_tracker = nullptr;
KvSeparationPolicy* global_kv_sep_policy = nullptr;
LearningStatistics global_learning_stats;
}  // namespace ROCKSDB_NAMESPACE