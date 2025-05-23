save_dir="data/iGSM"

if [ ! -d $save_dir ]; then
  mkdir -p $save_dir;
fi

# SFT data generation, use CoT to generate data with reflection and analysis
data_type="SFT"
dataset_size=2000
min_op=15
max_op=20

for cot_type in "short" "long" "short+a_aug" "short+q_aug"; do
    if [ $cot_type == "long" ]; then
        detailed_computation="--detailed_computation"
        reflect_p=0.1
        analysis_lvl=2
    elif [ $cot_type == "short" ]; then
        detailed_computation=""
        reflect_p=0.0
        analysis_lvl=0
    elif [ $cot_type == "short+q_aug" ]; then
        detailed_computation="--question_aug=4"
        reflect_p=0.0
        analysis_lvl=0
    elif [ $cot_type == "short+a_aug" ]; then
        detailed_computation="--answer_aug=4"
        reflect_p=0.0
        analysis_lvl=0
    else
        detailed_computation=""
        reflect_p=0.0
        analysis_lvl=0
    fi

    python iGSM-reasoning/scripts/generate_dataset.py \
        --use_cot \
        ${detailed_computation} \
        --max_operations=${max_op} \
        --min_operations=${min_op} \
        --dataset_size=${dataset_size} \
        --reflection_prob=${reflect_p} \
        --analysis_level=${analysis_lvl} \
        --data_type=${data_type} \
        --save_dir=${save_dir}
done

max_solution=1000 
dataset_size=10000
# RL train data generation, not use CoT
data_type="RL_train"
python iGSM-reasoning/scripts/generate_dataset.py \
    --max_operations=${max_op} \
    --min_operations=${min_op} \
    --data_type=${data_type} \
    --save_dir=${save_dir} \
    --dataset_size=${dataset_size} \
    --max_solution=${max_solution}

# RL validation data generation, not use CoT
data_type="RL_val"
min_op=21
max_op=25
python iGSM-reasoning/scripts/generate_dataset.py \
    --max_operations=${max_op} \
    --min_operations=${min_op} \
    --data_type=${data_type} \
    --save_dir=${save_dir} \
    --max_solution=${max_solution}

# data_type="test"
# dataset_size=500
# min_op=25
# max_op=25

# python iGSM-reasoning/scripts/generate_dataset.py \
#     --force \
#     --max_operations=${max_op} \
#     --min_operations=${min_op} \
#     --dataset_size=${dataset_size} \
#     --data_type=${data_type} \
#     --save_dir=${save_dir}
#     --max_solution=${max_solution}