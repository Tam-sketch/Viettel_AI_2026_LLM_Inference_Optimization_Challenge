# 🚀 Viettel AI Race 2026: LFM2.5 vLLM Serving Optimization

<div align="center">
  <img src="https://img.shields.io/badge/vLLM-0.6.x-blue?style=for-the-badge&logo=vllm" alt="vLLM Version" />
  <img src="https://img.shields.io/badge/Model-LFM2.5--1.2B-orange?style=for-the-badge" alt="Model" />
  <img src="https://img.shields.io/badge/Hardware-NVIDIA_H200_MIG_7-76B900?style=for-the-badge&logo=nvidia" alt="Hardware" />
  <img src="https://img.shields.io/badge/ERS_Score-61.26-success?style=for-the-badge" alt="ERS Score" />
</div>

## 📑 Mục lục
- [Giới thiệu](#-giới-thiệu)
- [Môi trường & Ràng buộc](#-môi-trường--ràng-buộc)
- [Kết quả Benchmark (Mốc V10)](#-kết-quả-benchmark-mốc-v10)
- [Chiến lược Tối ưu hóa (The Secret Sauce)](#-chiến-lược-tối-ưu-hóa-the-secret-sauce)
- [Cấu hình Triển khai (Quick Start)](#-cấu-hình-triển-khai-quick-start)
- [Các "Cạm bẫy" Đã Phân Tích (Pitfalls)](#-các-cạm-bẫy-đã-phân-tích-pitfalls)
- [Tác giả](#-tác-giả)

---

## 📌 Giới thiệu

Dự án này là kho lưu trữ cấu hình và chiến lược tối ưu hóa engine **vLLM** phục vụ cho mô hình **LFM2.5-1.2B-Instruct**, thuộc khuôn khổ cuộc thi **Viettel AI Race 2026**. 

Mục tiêu cốt lõi của dự án là tối đa hóa điểm số **ERS (Efficiency Ranking Score)** bằng cách cân bằng giữa **TTFT (Time To First Token)** siêu thấp và khả năng xử lý đồng thời cường độ cao (High Concurrency) mà không gây OOM (Out Of Memory) hoặc Timeout.

## 🖥️ Môi trường & Ràng buộc

Hệ thống được thiết kế để sống sót trong một môi trường phần cứng "bất đối xứng", nơi GPU rất mạnh nhưng CPU lại là nút thắt cổ chai lớn nhất:
- **GPU:** NVIDIA H200 (Profile MIG 7).
- **CPU:** Bị giới hạn cứng ở **3 Cores** (Gây nghẽn nghiêm trọng cho các luồng xử lý I/O và Scheduling).
- **Base Image:** `24521569/lfm-optimized:v1` (Custom image tích hợp FlashInfer 0.6.11, CUDA 13.0.2).
- **Đặc thù Benchmark:** Hệ thống chấm điểm bắn các luồng request mô phỏng hội thoại đa lượt (multi-turn chat) theo phân phối Poisson vào **1 worker duy nhất**. Yếu tố trừ điểm nặng nhất là `failed_count` (Timeout).

## 📊 Kết quả Benchmark (Mốc V10)

Qua nhiều vòng thử nghiệm, cấu hình **V10** được xác định là ranh giới cân bằng vật lý tối ưu nhất của hệ thống, đạt kỷ lục về TTFT:

| Metric | Giá trị | Ghi chú |
| :--- | :--- | :--- |
| **ERS Score** | **61.26** | Vượt rào cấu hình gốc, nằm trong top an toàn. |
| **TTFT P50** | `43ms` | Độ trễ khởi tạo siêu thanh nhờ GDN Prefill. |
| **TTFT P95** | `70ms` | Đảm bảo 95% request phản hồi ngay lập tức. |
| **Success Rate** | `413 / 420` | Đánh rơi 7 request do Decode Stall nhẹ (Giới hạn CPU). |
| **TBT Median** | `4ms` | Time Between Tokens cực kỳ ổn định. |

## 💡 Chiến lược Tối ưu hóa (The Secret Sauce)

Cấu hình V10 không phải là sự ngẫu nhiên mà là kết quả của việc khai thác triệt để các tính năng nâng cao của vLLM:

1. **Vũ khí Multi-turn: `--enable-prefix-caching`**
   Vì luật chấm điểm gửi lại toàn bộ lịch sử hội thoại cho 1 worker duy nhất, Prefix Caching giúp hệ thống không phải tính toán lại K-V tensors cho các token cũ, kéo TTFT ở các turn sau xuống gần bằng 0.

2. **Ép xung Băng thông: Cú Trade-off FP8**
   - Sử dụng `--kv-cache-dtype=fp8_e4m3` kết hợp `--quantization=fp8`.
   - **Được:** Ép băng thông bộ nhớ xuống tối thiểu, giúp GPU H200 nhai nuốt prompt ở tốc độ 70ms.
   - **Mất:** Chuyển đổi định dạng tạ (On-the-fly casting) tạo sức ép lên CPU 3-core, gây hiện tượng Decode Stall nhẹ (Làm rơi 7 request). Đây là sự đánh đổi có chủ đích để giữ TTFT ở mức kỷ lục.

3. **Lồng An toàn (Safety Bounds):**
   Thiết lập `--max-num-seqs=14`, `--max-num-batched-tokens=4096`, và `--gpu-memory-utilization=0.95`. Dù request có ập đến dồn dập (Poisson burst), VRAM luôn được bảo vệ khỏi hiện tượng phân mảnh (fragmentation) và tràn bộ nhớ (OOM).

## 🚀 Cấu hình Triển khai (Quick Start)

Triển khai ngay lập tức thông qua `docker-compose.yml`:

```yaml
services:
  model:
    image: 24521569/lfm-optimized:v1
    entrypoint:
      - python3
      - -m
      - vllm.entrypoints.openai.api_server
    command:
      # [1] Cấu hình Cơ bản
      - --model=/model
      - --served-model-name=LFM2.5-1.2B-Instruct
      - --host=0.0.0.0
      - --port=8000
      - --tensor-parallel-size=1
      
      # [2] Quản lý VRAM H200 (MIG 7)
      - --max-model-len=5120
      - --gpu-memory-utilization=0.95
      
      # [3] Ranh giới Hàng đợi (Bảo vệ CPU 3-Core)
      - --max-num-seqs=14
      - --max-num-batched-tokens=4096
      
      # [4] Ép xung Bộ nhớ (Online Quantization)
      - --quantization=fp8
      - --kv-cache-dtype=fp8_e4m3
      - --dtype=bfloat16
      
      # [5] Tối ưu Prefill & Prefix Caching
      - --enable-prefix-caching
      - --gdn-prefill-backend=flashinfer
      
      # [6] Triệt tiêu Overhead
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
