#include "db/workload_tracker.h"

namespace ROCKSDB_NAMESPACE {

HotKeyTracker<>* global_read_hotness_tracker = nullptr;
HotKeyTracker<>* global_write_hotness_tracker = nullptr;
LearningStats global_rl_stats;
KVSepParams global_kvsep_params;
ParamOptimizer global_kvsep_param_optimizer;
KvSepThresholdModel global_kvsep_model;
ReadHitLevel global_read_hit_level;

// disables hot key tracker and RL model if not nullptr
TinyLFU* global_frequent_write_key_cache = nullptr;
TinyLFU* global_frequent_read_key_cache = nullptr;

}  // namespace ROCKSDB_NAMESPACE