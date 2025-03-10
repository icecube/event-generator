

# ---------------------------------------
# Test:
#       - reconstruction via tiny 1-cascade model
#       - covariance matrix via tiny 1-cascade model
# Note: the tests here use the processing framework `ic3_processing` from
https://github.com/mhuen/ic3-processing
# and the config located in this directory:
./test_data_cfg.yaml
# ---------------------------------------

# Define directory for output, which we will directly place into the
# test_data directory of egenerator
export egenerator_dir=/INSERT/PATH/TO/EVENT-GENERATOR/DIRECTORY
export output_dir=${egenerator_dir}/tests_manual/test_data/egenerator_test_01

# Create jobs [adjust the python env in config prior to job creation]
ic3_create_job_files ${egenerator_dir}/tests_manual/test_data_cfg.yaml -d ${output_dir}

# Run scripts on NPX GPU/CPU
${output_dir}/processing/NuGen/NuE/low_energy/IC86_2013_holeice_30_v4/l5/jobs/1/job_DNN_l5_00000002.sh
${output_dir}/processing/NuGen/NuE/medium_energy/IC86_2013_holeice_30_v4/l5/jobs/1/job_DNN_l5_00000001.sh

# run test
python ${egenerator_dir}/tests_manual/test.py
