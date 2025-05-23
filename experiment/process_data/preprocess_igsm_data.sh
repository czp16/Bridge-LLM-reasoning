raw_dataset_dir='data/iGSM'
parquet_dataset_dir='data/parquet_files/iGSM' # where we will save parquent data

if [ ! -d $parquet_dataset_dir ]; then
  mkdir -p $parquet_dataset_dir;
fi

# preprocess SFT data
for filename in "SFT_2K_CoT_op_15-20_no_detailed_reflect_0.0_analysis_0" \
                "SFT_2K_CoT_op_15-20_detailed_reflect_0.1_analysis_2" \
                "SFT_2K_4Aaug_CoT_op_15-20_no_detailed_reflect_0.0_analysis_0" \
                "SFT_2K_4Qaug_CoT_op_15-20_no_detailed_reflect_0.0_analysis_0"; do
    SFT_data_path=${raw_dataset_dir}/${filename}
    python simple_verl/scripts/data_preprocess/igsm_sft.py \
        --train_dataset_path=${SFT_data_path} \
        --save_dir=${parquet_dataset_dir}
done

# preprocess RL data
RL_train_path=${raw_dataset_dir}/RL_train_10K_no_CoT_op_15-20
RL_val_path=${raw_dataset_dir}/RL_val_500_no_CoT_op_21-25

python simple_verl/scripts/data_preprocess/igsm.py \
    --train_dataset_path=${RL_train_path} \
    --val_dataset_path=${RL_val_path} \
    --save_dir=${parquet_dataset_dir}
