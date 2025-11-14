cur_dir=$(pwd)
# build
build_dir=../build
cd $build_dir
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make ldb -j
make test_kvsep -j
exe=./tools/test_kvsep

workload_config="0.5,0.5,0"

# run_numbers="1_1_1_1_1"
# Lhat=0
# blob_starting_level=-1
# blob_ending_level=$((Lhat-1))
# db_path=$cur_dir/db/kvsep_db_5_${run_numbers}_${Lhat}
# # $exe -workload="prepare" -path=$db_path -run_numbers=$run_numbers -blob_starting_level=$blob_starting_level -blob_ending_level=$blob_ending_level -workload_config=$workload_config > $cur_dir/log/prepare.log
# $exe -workload="test" -path=$db_path -run_numbers=$run_numbers -blob_starting_level=$blob_starting_level -blob_ending_level=$blob_ending_level -workload_config=$workload_config > $cur_dir/log/test_50_50_0_5_${run_numbers}_$Lhat.log

# run_numbers="1_1_1_1_1"
# Lhat=1
# blob_starting_level=0
# blob_ending_level=$((Lhat-1))
# db_path=$cur_dir/db/kvsep_db_5_${run_numbers}_${Lhat}
# $exe -workload="prepare" -path=$db_path -run_numbers=$run_numbers -blob_starting_level=$blob_starting_level -blob_ending_level=$blob_ending_level -workload_config=$workload_config > $cur_dir/log/prepare.log
# $exe -workload="test" -path=$db_path -run_numbers=$run_numbers -blob_starting_level=$blob_starting_level -blob_ending_level=$blob_ending_level -workload_config=$workload_config > $cur_dir/log/test_50_50_0_5_${run_numbers}_$Lhat.log

# run_numbers="1_1_1_1_1"
# Lhat=2
# blob_starting_level=0
# blob_ending_level=$((Lhat-1))
# db_path=$cur_dir/db/kvsep_db_5_${run_numbers}_${Lhat}
# $exe -workload="prepare" -path=$db_path -run_numbers=$run_numbers -blob_starting_level=$blob_starting_level -blob_ending_level=$blob_ending_level -workload_config=$workload_config > $cur_dir/log/prepare.log
# $exe -workload="test" -path=$db_path -run_numbers=$run_numbers -blob_starting_level=$blob_starting_level -blob_ending_level=$blob_ending_level -workload_config=$workload_config > $cur_dir/log/test_50_50_0_5_${run_numbers}_$Lhat.log

# run_numbers="1_1_1_1_1"
# Lhat=3
# blob_starting_level=0
# blob_ending_level=$((Lhat-1))
# db_path=$cur_dir/db/kvsep_db_5_${run_numbers}_${Lhat}
# $exe -workload="prepare" -path=$db_path -run_numbers=$run_numbers -blob_starting_level=$blob_starting_level -blob_ending_level=$blob_ending_level -workload_config=$workload_config > $cur_dir/log/prepare.log
# $exe -workload="test" -path=$db_path -run_numbers=$run_numbers -blob_starting_level=$blob_starting_level -blob_ending_level=$blob_ending_level -workload_config=$workload_config > $cur_dir/log/test_50_50_0_5_${run_numbers}_$Lhat.log

run_numbers="1_1_1_1_1"
Lhat=4
blob_starting_level=0
blob_ending_level=-1
db_path=$cur_dir/db/kvsep_db_5_${run_numbers}_${Lhat}
$exe -workload="prepare" -path=$db_path -run_numbers=$run_numbers -blob_starting_level=$blob_starting_level -blob_ending_level=$blob_ending_level -workload_config=$workload_config > $cur_dir/log/prepare.log
$exe -workload="test" -path=$db_path -run_numbers=$run_numbers -blob_starting_level=$blob_starting_level -blob_ending_level=$blob_ending_level -workload_config=$workload_config > $cur_dir/log/test_50_50_0_5_${run_numbers}_$Lhat.log
