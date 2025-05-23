# BRIDGE
## Installation
1. We use Anaconda or Miniconda to manage python environment.
2. Create conda env,
    ```
    cd Bridge-LLM-reasoning
    conda create -n bridge python=3.10
    conda activate bridge
    ```
3. Install PyTorch according to your platform and cuda version, we use pytorch 2.6.0 with CUDA 12.4 here:
    ```
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    ```
4. Install flash attention and vllm (we use 0.8.1),
    ```
    pip install flash-attn --no-build-isolation
    pip install vllm==0.8.1
    ```
5. Install iGSM-reasoning,
    ```
    cd iGSM-reasoning
    pip install -e .
    ```
    See [iGSM-reasoning/README.md](iGSM-reasoning/README.md) for detailed introduction of iGSM-reasoning.

6. Install simple VeRL (it is mainly derived from [VeRL](https://github.com/volcengine/verl), our main modifications are to simplify some codes):
    ```
    cd ../simple_verl
    pip install -e .
    ```

7. Login wandb and huggingface:
    ```
    wandb login
    huggingface-cli login
    ```

## Prepare iGSM dataset
We will prepare the dataset for both SFT and RL training. 
1. Go to the first level directory `Bridge-LLM-reasoning/`,
    ```
    cd ..
    ```
2. Generate datasets for SFT and RL: run
    ```
    experiment/process_data/data_generation.sh
    ```
    It will make dir `data/iGSM` and save the dataset in the dir.
3. Preprocess them and Convert them to `.parquet` file: run
    ```
    experiment/process_data/preprocess_igsm_data.sh
    ```
    The data preprocessing includes adding system prompt, applying SFT query and answer templates and other miscs.


## Run SFT and RL experiments
We suppose the experiment is run on a 2xA100(80G) server. Run experiments (SFT + RL) by
```
experiment/run_qwen2.5-1.5B-igsm.sh
```
You can use other scripts to run with other models.

- Run SFT.
    Run `experiment/run_qwen2.5-1.5B-igsm-sft.sh` to start SFT. The model will be saved to [model/sft](model/sft) dir. Remember to modify training data path if you use another one.

- Then you can run `experiment/run_qwen2.5-1.5B-igsm.sh` to start GRPO training. But you need to modify
    - training / test data path if you use another one.
    - sft model path.

  You can also modify
    - batch_size like `xxx_batch_size_per_gpu` according to the memory usage.
    - increase `gpu_memory_utilization` if on A100.
    - set `enforce_eager` and `free_cache_engine` to be false if memory allows. It will accelerate vllm inference by ~20%.
    - remove `export VLLM_ATTENTION_BACKEND=XFORMERS` at the beginning of the script if it works, then we will use engine V1 for inference.

## How to get the evaluation generation
Currently, we manually run vllm inference from the checkpoints, will consider update it later. 
See [simple_verl/scripts/evaluation/readme.md](simple_verl/scripts/evaluation/readme.md) for instructions.