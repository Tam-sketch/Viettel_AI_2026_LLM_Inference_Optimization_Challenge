# 🚀 Viettel AI Race 2026: LFM2.5 vLLM Serving Optimization

<div align="center">
  <img src="https://img.shields.io/badge/vLLM-0.6.x-blue?style=for-the-badge&logo=vllm" alt="vLLM Version" />
  <img src="https://img.shields.io/badge/Model-LFM2.5--1.2B-orange?style=for-the-badge" alt="Model" />
  <img src="https://img.shields.io/badge/Hardware-NVIDIA_H200_MIG_7-76B900?style=for-the-badge&logo=nvidia" alt="Hardware" />
  <img src="https://img.shields.io/badge/ERS_Score-61.26-success?style=for-the-badge" alt="ERS Score" />
</div>

## 📑 Table of Contents
- [Introduction](#-introduction)
- [Environment & Constraints](#-environment--constraints)
- [Benchmark Results (V10 Milestone)](#-benchmark-results-v10-milestone)
- [Optimization Strategies (The Secret Sauce)](#-optimization-strategies-the-secret-sauce)
- [Deployment Configuration (Quick Start)](#-deployment-configuration-quick-start)
- [Setup Guide](#️-setup-guide)
- [Identified Pitfalls](#️-identified-pitfalls)
- [Author](#-author)

---

## 📌 Introduction

This repository contains the configuration and optimization strategies for serving the **LFM2.5-1.2B-Instruct** model using a custom **vLLM** engine, developed for the **Viettel AI Race 2026**. 

The core objective is to maximize the **ERS (Efficiency Ranking Score)** by striking a perfect balance between an ultra-low **TTFT (Time To First Token)** and high concurrency handling, without triggering OOM (Out Of Memory) or Timeout errors.

## 🖥️ Environment & Constraints

The system is designed to survive in a highly "asymmetrical" hardware environment where the GPU is exceptionally powerful, but the CPU acts as a severe bottleneck:
- **GPU:** NVIDIA H200 (MIG 7 Profile).
- **CPU:** Strictly limited to **3 Cores** (Causes severe bottlenecks for I/O and scheduling threads).
- **Base Image:** `24521569/lfm-optimized:v1` (A custom image integrating FlashInfer 0.6.11 and CUDA 13.0.2).
- **Benchmark Characteristics:** The scoring system fires concurrent multi-turn chat requests following a Poisson distribution at a **single worker**. The most heavily penalized metric is the `failed_count` (Timeouts).

## 📊 Benchmark Results (V10 Milestone)

After dozens of iterations and rigorous testing, the **V10** configuration was identified as the optimal physical boundary of the system, achieving record-breaking TTFT speeds:

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **ERS Score** | **61.26** | Safely cleared the baseline, placed in the high-tier zone. |
| **TTFT P50** | `43ms` | Ultra-fast initialization thanks to GDN Prefill backend. |
| **TTFT P95** | `70ms` | Ensures 95% of requests respond almost instantly. |
| **Success Rate** | `413 / 420` | Dropped 7 requests due to a slight Decode Stall (CPU limits). |
| **TBT Median** | `4ms` | Highly stable Time Between Tokens. |

## 💡 Optimization Strategies (The Secret Sauce)

The V10 configuration is not a coincidence but the result of deeply exploiting vLLM's advanced features tailored to the competition's rules:

1. **The Multi-turn Weapon: `--enable-prefix-caching`**
   Since the scoring rules send full conversation histories to a single worker, Prefix Caching prevents the system from recomputing K-V tensors for past context, dropping the TTFT for subsequent turns to near zero.

2. **Bandwidth Overclocking: The FP8 Trade-off**
   - Utilizing `--kv-cache-dtype=fp8_e4m3` combined with `--quantization=fp8`.
   - **Pros:** Compresses memory bandwidth to the absolute minimum, allowing the H200 GPU to digest prompts at a blistering 70ms.
   - **Cons:** On-the-fly weights casting heavily strains the 3-core CPU, causing a slight Decode Stall (dropping 7 requests). This was a deliberate trade-off to maintain record-breaking TTFT.

3. **Safety Bounds:**
   Setting `--max-num-seqs=14`, `--max-num-batched-tokens=4096`, and `--gpu-memory-utilization=0.95` creates an unbreakable safety net. Even during a Poisson burst, the VRAM is strictly protected from fragmentation and OOM crashes.

## 🚀 Deployment Configuration (Quick Start)

Deploy immediately using the optimized `docker-compose.yml`:

```yaml
services:
  model:
    image: 24521569/lfm-optimized:v1
    entrypoint:
      - python3
      - -m
      - vllm.entrypoints.openai.api_server
    command:
      # [1] Core Configuration
      - --model=/model
      - --served-model-name=LFM2.5-1.2B-Instruct
      - --host=0.0.0.0
      - --port=8000
      - --tensor-parallel-size=1
      
      # [2] VRAM Management for H200 (MIG 7)
      - --max-model-len=5120
      - --gpu-memory-utilization=0.95
      
      # [3] Queue Safety Bounds (Protects the 3-Core CPU)
      - --max-num-seqs=14
      - --max-num-batched-tokens=4096
      
      # [4] Memory Overclocking (Online Quantization)
      - --quantization=fp8
      - --kv-cache-dtype=fp8_e4m3
      - --dtype=bfloat16
      
      # [5] Prefill & Prefix Caching Optimization
      - --enable-prefix-caching
      - --gdn-prefill-backend=flashinfer
      
      # [6] CPU Overhead Elimination
      - --disable-log-stats
      - --no-enable-log-requests
      - --trust-remote-code

    ports:
      - "8000:8000"
    shm_size: "2g"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
Here is the complete, professional English version of the project `README.md`. You can copy and paste this directly into your GitHub repository.

---

```markdown
# 🚀 Viettel AI Race 2026: LFM2.5 vLLM Serving Optimization

<div align="center">
  <img src="https://img.shields.io/badge/vLLM-0.6.x-blue?style=for-the-badge&logo=vllm" alt="vLLM Version" />
  <img src="https://img.shields.io/badge/Model-LFM2.5--1.2B-orange?style=for-the-badge" alt="Model" />
  <img src="https://img.shields.io/badge/Hardware-NVIDIA_H200_MIG_7-76B900?style=for-the-badge&logo=nvidia" alt="Hardware" />
  <img src="https://img.shields.io/badge/ERS_Score-61.26-success?style=for-the-badge" alt="ERS Score" />
</div>

## 📑 Table of Contents
- [Introduction](#-introduction)
- [Environment & Constraints](#-environment--constraints)
- [Benchmark Results (V10 Milestone)](#-benchmark-results-v10-milestone)
- [Optimization Strategies (The Secret Sauce)](#-optimization-strategies-the-secret-sauce)
- [Deployment Configuration (Quick Start)](#-deployment-configuration-quick-start)
- [Setup Guide](#️-setup-guide)
- [Identified Pitfalls](#️-identified-pitfalls)
- [Author](#-author)

---

## 📌 Introduction

This repository contains the configuration and optimization strategies for serving the **LFM2.5-1.2B-Instruct** model using a custom **vLLM** engine, developed for the **Viettel AI Race 2026**. 

The core objective is to maximize the **ERS (Efficiency Ranking Score)** by striking a perfect balance between an ultra-low **TTFT (Time To First Token)** and high concurrency handling, without triggering OOM (Out Of Memory) or Timeout errors.

## 🖥️ Environment & Constraints

The system is designed to survive in a highly "asymmetrical" hardware environment where the GPU is exceptionally powerful, but the CPU acts as a severe bottleneck:
- **GPU:** NVIDIA H200 (MIG 7 Profile).
- **CPU:** Strictly limited to **3 Cores** (Causes severe bottlenecks for I/O and scheduling threads).
- **Base Image:** `24521569/lfm-optimized:v1` (A custom image integrating FlashInfer 0.6.11 and CUDA 13.0.2).
- **Benchmark Characteristics:** The scoring system fires concurrent multi-turn chat requests following a Poisson distribution at a **single worker**. The most heavily penalized metric is the `failed_count` (Timeouts).

## 📊 Benchmark Results (V10 Milestone)

After dozens of iterations and rigorous testing, the **V10** configuration was identified as the optimal physical boundary of the system, achieving record-breaking TTFT speeds:

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **ERS Score** | **61.26** | Safely cleared the baseline, placed in the high-tier zone. |
| **TTFT P50** | `43ms` | Ultra-fast initialization thanks to GDN Prefill backend. |
| **TTFT P95** | `70ms` | Ensures 95% of requests respond almost instantly. |
| **Success Rate** | `413 / 420` | Dropped 7 requests due to a slight Decode Stall (CPU limits). |
| **TBT Median** | `4ms` | Highly stable Time Between Tokens. |

## 💡 Optimization Strategies (The Secret Sauce)

The V10 configuration is not a coincidence but the result of deeply exploiting vLLM's advanced features tailored to the competition's rules:

1. **The Multi-turn Weapon: `--enable-prefix-caching`**
   Since the scoring rules send full conversation histories to a single worker, Prefix Caching prevents the system from recomputing K-V tensors for past context, dropping the TTFT for subsequent turns to near zero.

2. **Bandwidth Overclocking: The FP8 Trade-off**
   - Utilizing `--kv-cache-dtype=fp8_e4m3` combined with `--quantization=fp8`.
   - **Pros:** Compresses memory bandwidth to the absolute minimum, allowing the H200 GPU to digest prompts at a blistering 70ms.
   - **Cons:** On-the-fly weights casting heavily strains the 3-core CPU, causing a slight Decode Stall (dropping 7 requests). This was a deliberate trade-off to maintain record-breaking TTFT.

3. **Safety Bounds:**
   Setting `--max-num-seqs=14`, `--max-num-batched-tokens=4096`, and `--gpu-memory-utilization=0.95` creates an unbreakable safety net. Even during a Poisson burst, the VRAM is strictly protected from fragmentation and OOM crashes.

## 🚀 Deployment Configuration (Quick Start)

Deploy immediately using the optimized `docker-compose.yml`:

```yaml
services:
  model:
    image: 24521569/lfm-optimized:v1
    entrypoint:
      - python3
      - -m
      - vllm.entrypoints.openai.api_server
    command:
      # [1] Core Configuration
      - --model=/model
      - --served-model-name=LFM2.5-1.2B-Instruct
      - --host=0.0.0.0
      - --port=8000
      - --tensor-parallel-size=1
      
      # [2] VRAM Management for H200 (MIG 7)
      - --max-model-len=5120
      - --gpu-memory-utilization=0.95
      
      # [3] Queue Safety Bounds (Protects the 3-Core CPU)
      - --max-num-seqs=14
      - --max-num-batched-tokens=4096
      
      # [4] Memory Overclocking (Online Quantization)
      - --quantization=fp8
      - --kv-cache-dtype=fp8_e4m3
      - --dtype=bfloat16
      
      # [5] Prefill & Prefix Caching Optimization
      - --enable-prefix-caching
      - --gdn-prefill-backend=flashinfer
      
      # [6] CPU Overhead Elimination
      - --disable-log-stats
      - --no-enable-log-requests
      - --trust-remote-code

    ports:
      - "8000:8000"
    shm_size: "2g"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

```

## ⚙️ Setup Guide

To run this configuration in a local environment (or personal server) before submitting it to the organizers, follow these steps:

### 1. Prerequisites

Ensure your host machine has the following components installed:

* **OS:** Linux (Ubuntu 20.04/22.04 recommended).
* **NVIDIA Drivers & CUDA:** Compatible with CUDA 13.0+ (Driver version >= 535.x).
* **Docker & NVIDIA Container Toolkit:** Mandatory for GPU passthrough.
```bash
# Install NVIDIA Container Toolkit (If not already installed)
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

```



### 2. Prepare the Model Weights

In the official system, weights are pre-mounted at `/model`. For local testing, you need to download the model and mount it manually.

```bash
# Create working directory
mkdir -p vllm-lfm-serving && cd vllm-lfm-serving

# Clone the model from HuggingFace (Requires git-lfs)
git clone [https://huggingface.co/viettel/LFM2.5-1.2B-Instruct](https://huggingface.co/viettel/LFM2.5-1.2B-Instruct) ./model_weights

```

### 3. Configure and Launch

Create the `docker-compose.yml` file with the content from the [Quick Start](https://www.google.com/search?q=%23-deployment-configuration-quick-start) section.
*(Note: Add the `volumes` block to mount the local model weights into the container).*

```yaml
    # ... (Keep existing configs)
    ports:
      - "8000:8000"
    shm_size: "2g"
    volumes:
      - ./model_weights:/model   # Mount local weights to /model inside the container
    deploy:
      resources:
        # ...

```

Launch the vLLM server in detached mode:

```bash
docker compose up -d

```

### 4. Verification

Check the logs to ensure the model has successfully loaded and the KV Cache is initialized:

```bash
docker compose logs -f model

```

*Success Indicator: You will see `Uvicorn running on http://0.0.0.0:8000` alongside the VRAM allocation profiling block.*

Send a test cURL request to verify the actual response speed:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LFM2.5-1.2B-Instruct",
    "messages": [
      {"role": "user", "content": "Hello, how are you today?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": true
  }'

```

## ⚠️ Identified Pitfalls

During our reverse engineering of the custom image and analysis of `arg_utils.py`, the following flags were proven detrimental to this specific environment and are **STRICTLY PROHIBITED**:

* ❌ `--swap-space`: Removed from the updated vLLM source code; using this triggers an immediate `Exit Code 2` crash.
* ❌ `--performance-mode=throughput`: Silently doubles `max_num_seqs` (14 $\rightarrow$ 28) at the C++ level. This overwhelms the 3-core CPU, leading to >80 timeout requests.
* ❌ `--kv-cache-dtype=fp8_e5m2`: Causes severe kernel conflicts with LFM2.5 during the Decode phase, plummeting `tokens_per_sec` to near zero (Decode Stall). Always prioritize `e4m3`.

---

## 👨‍💻 Author

* **Phan Khánh Tâm (Tâm)**
* Student, Computer Engineering @ University of Information Technology (UIT), VNU-HCM
* Competitor @ Viettel AI Race 2026

```

```
