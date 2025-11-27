#include <iostream>
#include <numeric>
#include <gflags/gflags.h>
#include <random>

#include "rocksdb/db.h"
#include "rocksdb/filter_policy.h"
#include "rocksdb/options.h"
#include "rocksdb/monkey_filter.h"
#include "util/string_util.h"
#include "rocksdb/monkey_filter.h"
#include "rocksdb/statistics.h"

DEFINE_string(level_capacities, "4194304,41943040,419430400,4194304000,41943040000", "Comma-separated list of level capacities");
DEFINE_string(run_numbers, "1,1,1,1,1", "Comma-separated list of run numbers");
DEFINE_int32(bpk, 10, "Bits per key for filter");
DEFINE_int32(kvsize, 1024, "Size of key-value pair");
DEFINE_string(compaction_style, "moose", "Compaction style: moose or default");
DEFINE_int32(prepare_entries, 20000000, "Number of entries to prepare");
DEFINE_int32(test_entries, 10000000, "Number of entries to test");
DEFINE_string(workload, "prepare", "prepare or test");
DEFINE_string(path, "/tmp/db", "dbpath");
DEFINE_int32(blob_starting_level, -1, "blob file starting level");
DEFINE_int32(blob_ending_level, -1, "blob file ending level");
DEFINE_string(workload_config,"0.5,0.5,0","Comma-separated list of workload config: write_ratio, read_ratio, range_ratio");

inline std::string ItoaWithPadding(const uint64_t key, uint64_t size) {
  std::string key_str = std::to_string(key);
  std::string padding_str(size - key_str.size(), '0');
  key_str = padding_str + key_str;
  return key_str;
}

class KeyGenerator {
 public:
  KeyGenerator(uint64_t start, uint64_t end, uint64_t key_size,
               uint64_t value_size, bool shuffle = true) {
    start_ = start;
    end_ = end;
    idx_ = 0;
    key_size_ = key_size;
    value_size_ = value_size;
    shuffle_ = shuffle;
  }
  std::string Key() const { return ItoaWithPadding(keys_[idx_], key_size_); }
  std::string Value() const {
    return ItoaWithPadding(keys_[idx_], value_size_);
  }
  bool Next() {
    idx_++;
    return idx_ < keys_.size();
  }
  void SeekToFirst() {
    for (uint64_t i = start_; i < end_; i++) {
      keys_.push_back(i);
    }
    if (shuffle_) {
      auto rng = std::default_random_engine{};
      std::shuffle(std::begin(keys_), std::end(keys_), rng);
    }
    idx_ = 0;
  }

 private:
  uint64_t idx_;
  std::vector<uint64_t> keys_;
  uint64_t start_;
  uint64_t end_;
  uint64_t key_size_;
  uint64_t value_size_;
  bool shuffle_;
};

void PrepareDB(rocksdb::DB* db) {
  rocksdb::WriteOptions write_options;
  rocksdb::ReadOptions read_options;
  KeyGenerator key_gen(0, FLAGS_prepare_entries, 24, FLAGS_kvsize - 24);
  key_gen.SeekToFirst();
  int idx = 0;
  while (key_gen.Next()) {
    auto status = db->Put(write_options, key_gen.Key(), key_gen.Value());
    if (!status.ok()) {
      std::cerr << "Failed to put key " << key_gen.Key() << " value "
                << key_gen.Value() << ", because: " << status.ToString() << std::endl;
      exit(1);
    }
    idx ++;
    if (idx % 100000 == 0) {
      std::cout << "prepared: " << idx << " entries" << std::endl;
    }
  }
}

struct WorkloadConfig {
  double write_ratio;
  double read_ratio;
  double range_ratio;
  WorkloadConfig(const std::string& config_str) {
    auto splits = rocksdb::StringSplit(config_str, '_');
    if (splits.size() == 1) {
      splits = rocksdb::StringSplit(config_str, ',');
    }
    if (splits.size() != 3) {
      std::cerr << "Invalid workload config: " << config_str << std::endl;
      exit(1);
    }
    write_ratio = std::stod(splits[0]);
    read_ratio = std::stod(splits[1]);
    range_ratio = std::stod(splits[2]);
    double total = write_ratio + read_ratio + range_ratio;
    write_ratio /= total;
    read_ratio /= total;
    range_ratio /= total;
  }
};

void RunWorkload(rocksdb::DB* db, const WorkloadConfig& config) {
  rocksdb::WriteOptions write_options;
  rocksdb::ReadOptions read_options;

  std::cout<<"workload config: "
           <<" write_ratio="<<config.write_ratio
           <<", read_ratio="<<config.read_ratio
           <<", range_ratio="<<config.range_ratio
           <<std::endl;
  KeyGenerator key_gen(0, FLAGS_test_entries, 24, FLAGS_kvsize - 24);
  key_gen.SeekToFirst();
  int idx = 0;
  while (key_gen.Next()) {
    double p = static_cast<double>(rand()) / RAND_MAX;    
    if (p < config.read_ratio) {
      // get result
      std::string value;
      auto status = db->Get(read_options, key_gen.Key(), &value);
      if (!status.ok()) {
        std::cerr << "Failed to get key " << key_gen.Key() << std::endl;
        exit(1);
      }
    }
    else if (p < config.write_ratio + config.read_ratio) {
      // put
      auto status = db->Put(write_options, key_gen.Key(), key_gen.Value());
      if (!status.ok()) {
        std::cerr << "Failed to put key " << key_gen.Key()
                  << " value " << key_gen.Value() << std::endl;
        exit(1);
      }
    }
    else {
      // range read
      rocksdb::Iterator* it = db->NewIterator(read_options);
      auto key = key_gen.Key();
      it->Seek(key);
      if (!it->Valid()) {
        std::cerr << "Failed to seek to key " << key << std::endl;
        exit(1);
      }
      for (int i = 0; i < 16 && it->Valid(); i++) {
        it->Next();
      }
      delete it;
    }
    idx ++;
    if (idx % 100000 == 0) {
      std::cout << "conducted " << idx << " operations" << std::endl;
    }
  }
}

void set_blob_options(rocksdb::Options& options) {
  if (FLAGS_blob_starting_level >= 0 && (FLAGS_blob_ending_level >= FLAGS_blob_starting_level || FLAGS_blob_ending_level < 0)) {
    options.enable_blob_files = true;
    // options.enable_blob_garbage_collection = true;
    options.blob_file_starting_level = FLAGS_blob_starting_level;
    options.blob_file_ending_level = FLAGS_blob_ending_level;
  } 
}

rocksdb::Options get_default_options() {
  rocksdb::Options options;
  options.create_if_missing = true;
  options.write_buffer_size = 2 << 20;
  options.level_compaction_dynamic_level_bytes = false;
  options.max_bytes_for_level_base = options.max_bytes_for_level_base * options.max_bytes_for_level_multiplier;
  auto table_options = options.table_factory->GetOptions<rocksdb::BlockBasedTableOptions>();
  table_options->filter_policy.reset(rocksdb::NewBloomFilterPolicy(FLAGS_bpk));
  return options;
}

rocksdb::Options get_moose_options() {
  rocksdb::Options options;
  options.compaction_style = rocksdb::kCompactionStyleMoose;
  options.create_if_missing = true;
  options.write_buffer_size = 2 << 20;
  options.level_compaction_dynamic_level_bytes = false;

  std::vector<std::string> split_st =
      rocksdb::StringSplit(FLAGS_level_capacities, ',');
  std::vector<uint64_t> level_capacities;
  for (auto& s : split_st) {
    level_capacities.push_back(std::stoull(s));
  }

  split_st = rocksdb::StringSplit(FLAGS_run_numbers, '_');
  if (split_st.size() != level_capacities.size()) {
    split_st = rocksdb::StringSplit(FLAGS_run_numbers, ',');
  }

  std::vector<int> run_numbers;
  for (auto& s : split_st) {
    run_numbers.push_back(std::stoi(s));
  }
  std::vector<uint64_t> physical_level_capacities;
  for (int i = 0; i < (int)run_numbers.size(); i++) {
    uint64_t run_size = level_capacities[i] / run_numbers[i];
    for (int j = 0; j < (int)run_numbers[i]; j++) {
      physical_level_capacities.push_back(run_size);
    }
  }
  options.level_capacities = physical_level_capacities;
  options.run_numbers = run_numbers;
  options.num_levels = std::accumulate(run_numbers.begin(), run_numbers.end(), 0);

  uint64_t entry_num = std::accumulate(options.level_capacities.begin() + 1, options.level_capacities.end(), 0UL) / FLAGS_kvsize;
  uint64_t filter_memory = entry_num * FLAGS_bpk / 8;

  // auto tmp = rocksdb::MonkeyBpks(entry_num, filter_memory, options.level_capacities, FLAGS_kvsize);
  std::vector<double> bpks = {(double)FLAGS_bpk};
  // std::copy(tmp.begin(), tmp.end(), std::back_inserter(bpks));
  for (int i = 1; i < (int)options.level_capacities.size(); i++) {
    bpks.push_back(FLAGS_bpk);
  }
  // display options
  std::cout << "level capacities: " << std::endl;
  for (auto lvl_cap : options.level_capacities) {
    std::cout << "  " << lvl_cap << std::endl;
  }
  std::cout << "run numbers: " << std::endl;
  for (auto rn : options.run_numbers) {
    std::cout << "  " << rn << std::endl;
  }
  std::cout << "bpks: " << std::endl;
  for (auto bpk : bpks) {
    std::cout << "  " << bpk << std::endl;
  }
  
  auto table_options = options.table_factory->GetOptions<rocksdb::BlockBasedTableOptions>();
  table_options->filter_policy.reset(rocksdb::NewMonkeyFilterPolicy(bpks));

  return options;
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  rocksdb::DB* db;

  rocksdb::Options options;
  if (FLAGS_compaction_style == "default") {
    options = get_default_options();
  } else if (FLAGS_compaction_style == "moose") {
    options = get_moose_options();
  } else {
    std::cerr << "Unknown compaction style: " << FLAGS_compaction_style << std::endl;
    return 1;
  }
  set_blob_options(options);
  if (FLAGS_workload == "test") {
    options.use_direct_io_for_flush_and_compaction = true;
    options.use_direct_reads = true;
  }
  options.level0_slowdown_writes_trigger = 4;
  options.level0_stop_writes_trigger = 8;
  options.statistics = rocksdb::CreateDBStatistics();
  auto status = rocksdb::DB::Open(options, FLAGS_path, &db);
  if (!status.ok()) {
    std::cerr << "Failed to open db: " << status.ToString() << std::endl;
    return 1;
  }
  auto start_timer = std::chrono::high_resolution_clock::now();
  if (FLAGS_workload == "prepare") {
    PrepareDB(db);
  } else if (FLAGS_workload == "test") {
    RunWorkload(db, WorkloadConfig(FLAGS_workload_config));
  }
  auto end_timer = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_timer - start_timer).count();
  int num_operations = (FLAGS_workload == "prepare") ? FLAGS_prepare_entries : FLAGS_test_entries;
  std::cout << "Total time (ms): " << duration << std::endl;
  std::cout << "Operations: " << num_operations << std::endl;
  std::cout << "Throughput (ops/sec): " << (num_operations * 1000.0) / duration << std::endl;
  std::string stat;
  db->GetProperty("rocksdb.stats", &stat);
  std::cout << stat << std::endl;
  std::cout << "statistics: " << options.statistics->ToString() << std::endl;

  db->Close();
  delete db;
  return 0;
}