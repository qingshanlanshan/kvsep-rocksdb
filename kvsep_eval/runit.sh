cur_dir=$(pwd)
# build
build_dir=../build
cd $build_dir
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
# cmake -DCMAKE_BUILD_TYPE=Debug ..
make ldb -j
make test_kvsep -j
exe=./tools/test_kvsep

echo build done

put_mode="update"
run_numbers="1_1_1_1_1"
key_zipf_dist=true
value_uniform_dist=true
kv_size=1024
min_blob_size=128
write_frequency_cache=true
read_frequency_cache=false

run_test() {
    workload_config=$1
    workload_name=$2

    # Lhat=0
    # blob_starting_level=-1
    # blob_ending_level=-1
    # db_path=$cur_dir/db/kvsep_db_5_${run_numbers}_${Lhat}_zipf_uniform
    # echo running on db: $db_path
    # # $exe -workload="prepare" -path=$db_path -kvsize=$kv_size -run_numbers=$run_numbers -blob_starting_level=$blob_starting_level -blob_ending_level=$blob_ending_level -put_mode=$put_mode -workload_config=$workload_config -key_zipf_dist=$key_zipf_dist -value_uniform_dist=$value_uniform_dist > $cur_dir/log/prepare.log
    # $exe -workload="test" -path=$db_path -kvsize=$kv_size -run_numbers=$run_numbers -blob_starting_level=$blob_starting_level -blob_ending_level=$blob_ending_level -put_mode=$put_mode -workload_config=$workload_config -key_zipf_dist=$key_zipf_dist -value_uniform_dist=$value_uniform_dist > $cur_dir/log/test_${workload_config}_5_${run_numbers}_${Lhat}_zipf_uniform_${workload_name}.log


    # for Lhat in 1 2 3 4 5; do
    #     blob_starting_level=0
    #     blob_ending_level=$((Lhat-1))
    #     db_path=$cur_dir/db/kvsep_db_5_${run_numbers}_${Lhat}_zipf_uniform
    #     echo running on db: $db_path
    #     # $exe -workload="prepare" -path=$db_path -kvsize=$kv_size -run_numbers=$run_numbers -blob_starting_level=$blob_starting_level -blob_ending_level=$blob_ending_level -put_mode=$put_mode -workload_config=$workload_config -key_zipf_dist=$key_zipf_dist -value_uniform_dist=$value_uniform_dist > $cur_dir/log/prepare.log
    #     $exe -workload="test" -path=$db_path -kvsize=$kv_size -run_numbers=$run_numbers -blob_starting_level=$blob_starting_level -blob_ending_level=$blob_ending_level -put_mode=$put_mode -workload_config=$workload_config -key_zipf_dist=$key_zipf_dist -value_uniform_dist=$value_uniform_dist > $cur_dir/log/test_${workload_config}_5_${run_numbers}_${Lhat}_zipf_uniform_${workload_name}.log
    # done

    # # test with min blob size
    # for Lhat in 4; do
    #     blob_starting_level=0
    #     blob_ending_level=$((Lhat-1))
    #     db_path=$cur_dir/db/kvsep_db_5_${run_numbers}_${Lhat}_zipf_uniform_${min_blob_size}B
    #     echo running on db: $db_path
    #     # $exe -workload="prepare" -min_blob_size=$min_blob_size -path=$db_path -kvsize=$kv_size -run_numbers=$run_numbers -blob_starting_level=$blob_starting_level -blob_ending_level=$blob_ending_level -put_mode=$put_mode -workload_config=$workload_config -key_zipf_dist=$key_zipf_dist -value_uniform_dist=$value_uniform_dist > $cur_dir/log/prepare.log
    #     $exe -workload="test" -min_blob_size=$min_blob_size -path=$db_path -kvsize=$kv_size -run_numbers=$run_numbers -blob_starting_level=$blob_starting_level -blob_ending_level=$blob_ending_level -put_mode=$put_mode -workload_config=$workload_config -key_zipf_dist=$key_zipf_dist -value_uniform_dist=$value_uniform_dist > $cur_dir/log/test_${workload_config}_5_${run_numbers}_${Lhat}_zipf_uniform_${workload_name}_${min_blob_size}B.log
    # done

    blob_starting_level=0
    blob_ending_level=4
    db_path=$cur_dir/db/kvsep_db_5_1_1_1_1_1_zipf_uniform_test
    log_path=$cur_dir/log/test_${workload_config}_5_${run_numbers}_zipf_uniform_${workload_name}_test_${write_frequency_cache}_${read_frequency_cache}.log
    echo running on db: $db_path
    # rm -rf $db_path
    # $exe -workload="prepare" \
    #     -path=${db_path} \
    #     -kvsize=$kv_size \
    #     -run_numbers=$run_numbers \
    #     -blob_starting_level=$blob_starting_level \
    #     -blob_ending_level=$blob_ending_level \
    #     -put_mode=$put_mode \
    #     -workload_config=$workload_config \
    #     -key_zipf_dist=$key_zipf_dist \
    #     -value_uniform_dist=$value_uniform_dist \
    #     -hotness_tracking=false \
    #     -write_frequency_cache=true \
    #     -read_frequency_cache=false \
    #     > $cur_dir/log/prepare.log
    $exe -workload="test" \
        -path=${db_path} \
        -kvsize=$kv_size \
        -run_numbers=$run_numbers \
        -blob_starting_level=$blob_starting_level \
        -blob_ending_level=$blob_ending_level \
        -put_mode=$put_mode \
        -workload_config=$workload_config \
        -key_zipf_dist=$key_zipf_dist \
        -value_uniform_dist=$value_uniform_dist \
        -hotness_tracking=false \
        -write_frequency_cache=$write_frequency_cache \
        -read_frequency_cache=$read_frequency_cache \
        > $log_path
    echo log saved to $log_path
}

# run_test "0_100_0" "read"
# run_test "100_0_0" "write"
run_test "50_50_0" "read_write"
# run_test "33_33_33" "balanced"
# run_test "0_0_100" "scan"
