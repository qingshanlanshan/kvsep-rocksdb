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
hotness_tracking="cache"
direct_read=false

run_test() {
    workload_config=$1
    workload_name=$2

    # blob_starting_level=-1
    # blob_ending_level=-1
    # db_path=$cur_dir/db/kvsep_db_5_rocksdb_zipf_uniform
    # log_path=$cur_dir/log/test_${workload_config}_5_rocksdb_zipf_uniform_${workload_name}.log
    # echo running on db: $db_path
    # $exe -workload="prepare" -path=$db_path -kvsize=$kv_size -run_numbers=$run_numbers -blob_starting_level=$blob_starting_level -blob_ending_level=$blob_ending_level -put_mode=$put_mode -workload_config=$workload_config -key_zipf_dist=$key_zipf_dist -value_uniform_dist=$value_uniform_dist > $cur_dir/log/prepare.log
    # $exe -workload="test" -path=$db_path -kvsize=$kv_size -run_numbers=$run_numbers -blob_starting_level=$blob_starting_level -blob_ending_level=$blob_ending_level -put_mode=$put_mode -workload_config=$workload_config -key_zipf_dist=$key_zipf_dist -value_uniform_dist=$value_uniform_dist > $log_path

    # blob_starting_level=0
    # blob_ending_level=-1
    # db_path=$cur_dir/db/kvsep_db_5_blobdb_zipf_uniform
    # log_path=$cur_dir/log/test_${workload_config}_5_blobdb_zipf_uniform_${workload_name}.log
    # echo running on db: $db_path
    # # $exe -workload="prepare" -path=$db_path -kvsize=$kv_size -run_numbers=$run_numbers -blob_starting_level=$blob_starting_level -blob_ending_level=$blob_ending_level -put_mode=$put_mode -workload_config=$workload_config -key_zipf_dist=$key_zipf_dist -value_uniform_dist=$value_uniform_dist > $cur_dir/log/prepare.log
    # $exe -workload="test" -path=$db_path -kvsize=$kv_size -run_numbers=$run_numbers -blob_starting_level=$blob_starting_level -blob_ending_level=$blob_ending_level -put_mode=$put_mode -workload_config=$workload_config -key_zipf_dist=$key_zipf_dist -value_uniform_dist=$value_uniform_dist > $log_path


    # for blob_ending_level in 0 1 2 3; do
    #     blob_starting_level=0
    #     db_path=$cur_dir/db/kvsep_db_5_${blob_starting_level}_${blob_ending_level}_zipf_uniform
    #     log_path=$cur_dir/log/test_${workload_config}_5_${blob_starting_level}_${blob_ending_level}_zipf_uniform_${workload_name}$( [ "$direct_read" = "true" ] && echo "_direct_read" ).log
    #     echo running on db: $db_path
    #     # $exe -workload="prepare" -path=$db_path -kvsize=$kv_size -run_numbers=$run_numbers -blob_starting_level=$blob_starting_level -blob_ending_level=$blob_ending_level -put_mode=$put_mode -workload_config=$workload_config -key_zipf_dist=$key_zipf_dist -value_uniform_dist=$value_uniform_dist > $cur_dir/log/prepare.log
    #     $exe -workload="test" -path=$db_path -kvsize=$kv_size -run_numbers=$run_numbers -blob_starting_level=$blob_starting_level -blob_ending_level=$blob_ending_level -put_mode=$put_mode -workload_config=$workload_config -key_zipf_dist=$key_zipf_dist -value_uniform_dist=$value_uniform_dist -direct_read=$direct_read > $log_path
    # done

    # for blob_starting_level in 1 2 3 4; do
    #     blob_ending_level=4
    #     db_path=$cur_dir/db/kvsep_db_5_${blob_starting_level}_${blob_ending_level}_zipf_uniform
    #     log_path=$cur_dir/log/test_${workload_config}_5_${blob_starting_level}_${blob_ending_level}_zipf_uniform_${workload_name}$( [ "$direct_read" = "true" ] && echo "_direct_read" ).log
    #     # $exe -workload="prepare" -path=$db_path -kvsize=$kv_size -run_numbers=$run_numbers -blob_starting_level=$blob_starting_level -blob_ending_level=$blob_ending_level -put_mode=$put_mode -workload_config=$workload_config -key_zipf_dist=$key_zipf_dist -value_uniform_dist=$value_uniform_dist > $cur_dir/log/prepare.log
    #     $exe -workload="test" -path=$db_path -kvsize=$kv_size -run_numbers=$run_numbers -blob_starting_level=$blob_starting_level -blob_ending_level=$blob_ending_level -put_mode=$put_mode -workload_config=$workload_config -key_zipf_dist=$key_zipf_dist -value_uniform_dist=$value_uniform_dist -direct_read=$direct_read > $log_path
    # done

    # test with min blob size
    # blob_starting_level=0
    # blob_ending_level=-1
    # db_path=$cur_dir/db/kvsep_db_5_${blob_starting_level}_${blob_ending_level}_zipf_uniform_${min_blob_size}B
    # echo running on db: $db_path
    # $exe -workload="prepare" -min_blob_size=$min_blob_size -path=$db_path -kvsize=$kv_size -run_numbers=$run_numbers -blob_starting_level=$blob_starting_level -blob_ending_level=$blob_ending_level -put_mode=$put_mode -workload_config=$workload_config -key_zipf_dist=$key_zipf_dist -value_uniform_dist=$value_uniform_dist > $cur_dir/log/prepare.log
    # $exe -workload="test" -min_blob_size=$min_blob_size -path=$db_path -kvsize=$kv_size -run_numbers=$run_numbers -blob_starting_level=$blob_starting_level -blob_ending_level=$blob_ending_level -put_mode=$put_mode -workload_config=$workload_config -key_zipf_dist=$key_zipf_dist -value_uniform_dist=$value_uniform_dist > $cur_dir/log/test_${workload_config}_5_${run_numbers}_5_zipf_uniform_${workload_name}_${min_blob_size}B.log

    blob_starting_level=0
    blob_ending_level=4
    db_path=$cur_dir/db/kvsep_db_5_test_zipf_uniform
    log_path=$cur_dir/log/test_${workload_config}_5_test_zipf_uniform_${workload_name}_${hotness_tracking}$( [ "$direct_read" = "true" ] && echo "_direct_read" ).log
    echo running on db: $db_path
    rm -rf $db_path
    cp -r ${db_path}_backup $db_path
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
    #     -hotness_tracking=$hotness_tracking \
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
        -hotness_tracking=$hotness_tracking \
        -direct_read=$direct_read \
        > $log_path
    echo log saved to $log_path
}

# run_test "0_100_0" "read"
# run_test "100_0_0" "write"
run_test "50_50_0" "read_write"
# run_test "33_33_33" "balanced"
# run_test "0_0_100" "scan"
