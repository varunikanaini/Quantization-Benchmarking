## 🚀 DeepSeek-R1-Distill-Qwen-1.5B — Full Setup, Quantization & Benchmarking Guide

Welcome! 👋 This guide helps you set up, quantize, serve, and benchmark the **DeepSeek-R1-Distill-Qwen-1.5B** model using `llama.cpp`, `vLLM`, and ROCm tools.

> 🧠 Whether you're experimenting, benchmarking, or deploying in production — follow this step-by-step!

---

## 🔐 SSH Access

Login to your cloud machines to get started:

```bash
ssh ubuntu@193.143.78.158
# Inside:
ssh gpu-22

ssh ubuntu@193.143.78.200
ssh gpu-60
````

---

## 📦 Download the Model

Use Hugging Face CLI to download the base model:

```bash
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir ../DeepSeek-R1-Distill-Qwen-1.5B
```

---

## 🏗️ Clone & Build `llama.cpp`

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build
cmake --build build --config Release
```
> 🔧 To ADD BUILD for AMD GPU's :

```bash
#hip build 
cd /var/lib/ubuntu/vn2/llama.cpp
rm -rf build
mkdir build && cd build

cmake .. \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx942 \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build . --config Release -- -j16
```

---

## 🔁 Convert HF to GGUF Format

Make sure the model is downloaded, then convert it:

```bash
cd llama.cpp
python convert_hf_to_gguf.py ../DeepSeek-R1-Distill-Qwen-1.5B --outfile DeepSeek-R1-Distill-Qwen-1.5B-f16.gguf
python convert_hf_to_gguf.py ../gsm8k_v8_pass1 --outfile Distilled_Qwen1.5B-f16_gsm8k_v8_pass1_qat.gguf
```

---

## ⚙️ Add Custom Quantization Logic

Manually copy:

* `quant_config.py`
* `quant_map.py`

into your working directory.

---

## 🔧 Run Layerwise Dynamic Quantization

```bash
chmod +x run_custom_quant_TEST.sh
./run_custom_quant_TEST.sh
```

---

## 🧩 Serve Quantized Model (llama.cpp)

> 🤡 Use Tmux: Inside one shell, serve the model. In other shell, Benchmark the model using benchmarking scripts.

```bash
HIP_VISIBLE_DEVICES=3 ./llama.cpp/build/bin/llama-server   -m ./DYNAMIC_QUANTIZATION/gguf/DeepSeek-R1-Distill-Qwen-1.5B-gsm8k_v8_qat.gguf   -c 32678   --port 1596   -t 8   -ngl 28
```

>  To apply batching, run the below code [Adjust np based on your preference].

 ```bash
  HIP_VISIBLE_DEVICES=3 ./llama.cpp/build/bin/llama-server   -m ./DYNAMIC_QUANTIZATION/gguf/DeepSeek-R1-Distill-Qwen-1.5B-gsm8k_v8_qat.gguf   -c 32678   --port 1596   -t 8   -ngl 28 -np 8 -cb
 ```

📂 GGUF files:

> ✨ Includes Other gguf files for benchmarking

* `DeepSeek-R1-Distill-Qwen-1.5B-mixed-from-json2.gguf`
* `DeepSeek-R1-Distill-Qwen-1.5B-gsm8k_v8_qat.gguf`

---

## 🐳 Serve OG Model with Docker + vLLM

> 🤡 Use Tmux: Inside one shell, serve the model. In other shell, Benchmark the model using benchmarking scripts.

```bash
docker run -it \
  -e HIP_VISIBLE_DEVICES=1 \
  --network=host \
  --group-add=video \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  -v /shareddata/dheyo/varunika/benchmarking/DeepSeek-R1-Distill-Qwen-1.5B:/app/model \
  rocm/vllm \
  bash -c 'python -m vllm.entrypoints.openai.api_server --model /app/model --port 1400 --served-model-name deepseek_qwen_distill'
```

---

## 🐍 Serve Unsloth GGUF Models

### 📥 Download Models:

```bash
huggingface-cli download unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF DeepSeek-R1-Distill-Qwen-1.5B-Q2_K_L.gguf --local-dir ../DeepSeek-R1-Distill-Qwen-1.5B
huggingface-cli download unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf --local-dir ../DeepSeek-R1-Distill-Qwen-1.5B
huggingface-cli download unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf --local-dir ../DeepSeek-R1-Distill-Qwen-1.5B
```

### ▶️ Serve Commands:

> 🤡 Use Tmux: Inside one shell, serve the model. In other shell, Benchmark the model using benchmarking scripts.

```bash
HIP_VISIBLE_DEVICES=1 ./llama.cpp/build/bin/llama-server \
  -m ./DeepSeek-R1-Distill-Qwen-1.5B/DeepSeek-R1-Distill-Qwen-1.5B-Q2_K_L.gguf \
  -c 2048 \
  --port 1700 \
  -t 8 \
  -ngl 28
```

```bash
HIP_VISIBLE_DEVICES=1 ./llama.cpp/build/bin/llama-server \
  -m ../DeepSeek-R1-Distill-Qwen-1.5B/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf \
  -c 2048 \
  --port 1703 \
  -t 8 \
  -ngl 28
```

```bash
HIP_VISIBLE_DEVICES=1 ./llama.cpp/build/bin/llama-server \
  -m ../DeepSeek-R1-Distill-Qwen-1.5B/DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf \
  -c 32678 \
  --port 1805 \
  -t 8 \
  -ngl 28
```

---

## 📊 Run Benchmarks

```bash
source bench_env/bin/activate
pip install datasets pandas requests
python3 bench.py
```

---

## 💾 Download Result Files

### 🖥️ On Server:

```bash
mv ~/vn2/mmlu_* .
```

### 📤 From Server to Local Machine:

```bash
scp ubuntu@193.143.78.158:/shareddata/dheyo/abhilash/vn/mmlu_* .
```

---

## 🧪 TO Run LiveCodeBench (LCB)

```bash
export HIP_VISIBLE_DEVICES=7
VLLM_USE_TRITON_FLASH_ATTN=0 python -m lcb_runner.runner.main \
  --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
  --scenario codeexecution \
  --evaluate \
  --release_version release_v1
```

## ⚗️ TO RUN Lm_Evaluation_Harness 🙈

> Let's redirect to the working directory and environment, in case setting up from start, follow https://github.com/EleutherAI/lm-evaluation-harness/blob/main/README.md
```bash
cd /shareddata/dheyo/varunika/benchmarking/llmeval/DYNAMIC_QUANTIZATION/lm-evaluation-harness
source lm_eval/bin/activate
```
> 👽️ Once redirected, run the command!
> 🚧 Change the path and tokenizer based on the model you wish to serve!!
```bash
HIP_VISIBLE_DEVICES=3 lm_eval \
    --model hf \
    --model_args pretrained=/shareddata/dheyo/varunika/benchmarking/llmeval/DYNAMIC_QUANTIZATION/gguf/,gguf_file=DeepSeek-R1-Distill-Qwen-1.5B-gsm8k_v8_qat.gguf,tokenizer=/shareddata/dheyo/varunika/benchmarking/llmeval/gsm8k_v8 \
    --tasks gsm8k \
    --device cuda:0 \
    --batch_size 1
```

## QUANTIZATION AWARE TRAINING 

> Let's redirect to the working directory and environment, in case setting up from start, follow https://github.com/dheyoai/torchtune/tree/dheyo_fp4
```bash
cd /shareddata/dheyo/varunika/QAT/torchtune/recipes/configs/distilled_qwen2_5/1.5B_qat_full.yaml
```

>  This config assumes that you've run the following command before launching:
```bash
tune download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --output-dir /shareddata/dheyo/varunika/QAT/torchtune/models/DeepSeek-R1-Distill-Qwen-1.5B
```

> To launch on 2 devices, run the following command from root:
```bash
tune run --nproc_per_node 2 qat_distributed --config recipes/configs/distilled_qwen2_5/1.5B_qat_full.yaml
```

> You can add specific overrides through the command line. For example, to override the checkpointer directory while launching training:
```bash
tune run --nproc_per_node 2 full_finetune_distributed --config qwen2_5/1.5B_full checkpointer.checkpoint_dir=<YOUR_CHECKPOINT_DIR>
```

> Keep changing the no of epochs, lr, batch size

# To add support for a new dataset:
> change 1:
```bash
dataset:
  _component_: torchtune.datasets.new_dataset_name <---- change here
  packed: False  # True increases speed
  split: train[:95%]
seed: null
shuffle: True

# Validation
run_val_every_n_steps: 100 # Change to an integer to enable validation every N steps
dataset_val:
  _component_: torchtune.datasets.new_dataset_name <---- change here
  split: train[95%:]
batch_size_val: ${batch_size}
```

---

> 🎉 You’re all set to benchmark and deploy DeepSeek models efficiently!
> 💬 For issues or contributions, feel free to open a discussion or PR.

```
