
# 🚀 Viettel AI Race 2026: Tối Ưu Hóa vLLM Serving Cho Mô Hình LFM2.5 (https://www.google.com/url?sa=E&source=gmail&q=https://competition.viettel.vn/contests/llm-2026)
<div align="center">
  <img src="https://img.shields.io/badge/vLLM-0.6.x-blue?style=for-the-badge&logo=vllm" alt="vLLM Version" />
  <img src="https://img.shields.io/badge/Model-LFM2.5--1.2B-orange?style=for-the-badge" alt="Model" />
  <img src="https://img.shields.io/badge/Hardware-NVIDIA_H200_MIG_7-76B900?style=for-the-badge&logo=nvidia" alt="Hardware" />
  <img src="https://img.shields.io/badge/ERS_Score-61.26-success?style=for-the-badge" alt="ERS Score" />
  <img src="https://img.shields.io/badge/Final_Rank-170%2F281-blueviolet?style=for-the-badge" alt="Final Rank" />
</div>

## 📑 Mục Lục
- [Tổng Quan Dự Án](#-tổng-quan-dự-án)
- [Môi Trường & Ràng Buộc Phần Cứng](#-môi-trường--ràng-buộc-phần-cứng)
- [So Sánh Docker Image Gốc vs Tự Build](#-so-sánh-docker-image-gốc-vs-tự-build)
- [Kết Quả Chung Cuộc & Benchmark](#-kết-quả-chung-cuộc--benchmark)
- [Chiến Lược Tối Ưu Hóa](#-chiến-lược-tối-ưu-hóa)
- [Cấu Hình Triển Khai](#-cấu-hình-triển-khai)
- [Hướng Dẫn Cài Đặt & Chạy Thử](#-hướng-dẫn-cài-đặt--chạy-thử)
- [Các Cạm Bẫy Cần Tránh](#-các-cạm-bẫy-cần-tránh)
- [Tác Giả](#-tác-giả)

---

## 📌 Tổng Quan Dự Án

Kho lưu trữ này chứa toàn bộ cấu hình, tài liệu kỹ thuật và chiến lược tối ưu hóa engine **vLLM** phục vụ mô hình **LFM2.5-1.2B-Instruct** trong khuôn khổ cuộc thi **Viettel AI Race 2026**.

Mục tiêu trọng tâm là tối đa hóa điểm số **ERS (Efficiency Ranking Score)** thông qua việc ép thấp thời gian tạo token đầu tiên (**TTFT**) và duy trì khả năng phục vụ đồng thời ở cường độ cao mà không gây tràn bộ nhớ (OOM) hay nghẽn luồng xử lý (Timeout).

## 🖥️ Môi Trường & Ràng Buộc Phần Cứng

Hệ thống hoạt động trong môi trường phần cứng có sự bất đối xứng lớn:
* **GPU:** NVIDIA H200 (Cấu hình phân vùng MIG 7 cố định).
* **CPU:** Giới hạn nghiêm ngặt ở **3 Cores** (Gây nghẽn nghiêm trọng cho các tác vụ lập lịch, tiền xử lý và I/O).
* **Cơ Chế Đánh Giá:** Máy chấm gửi các luồng request hội thoại đa lượt (multi-turn chat) theo phân phối Poisson tới **1 worker duy nhất**. Điểm số bị phạt nặng nhất bởi chỉ số `failed_count` (các request quá thời gian chờ).

## 🐳 So Sánh Docker Image Gốc vs Tự Build

Để vượt qua giới hạn phần cứng, Docker Image đã được xây dựng lại (custom build) thay vì sử dụng image mặc định từ BTC:

| Tiêu chí | Image Gốc BTC (`vllm/vllm-openai:v0.22.1`) | Image Tự Build (`24521569/lfm-optimized:v1`) |
| :--- | :--- | :--- |
| **Bản chất Image** | Image chuẩn phân phối từ vLLM registry / BTC cấp ban đầu. | Image tùy biến (custom build) được tối ưu sâu và đóng gói riêng. |
| **Nền tảng Runtime** | Môi trường CUDA / PyTorch cơ bản theo bản phát hành cũ. | Cập nhật **CUDA 13.0.2**, **Python 3.12**, tối ưu riêng cho kiến trúc NVIDIA H200. |
| **Nhánh vLLM** | Mã nguồn vLLM bản cũ (v0.22.x / v0.4.x đời đầu). | Nhánh mã nguồn vLLM hiện đại (**v0.6.x+**), tương thích cấu trúc mới. |
| **Backend Attention** | Hỗ trợ FlashAttention / PagedAttention mặc định. | Tích hợp sẵn **FlashInfer `0.6.11.post2`** và kernel tùy biến cho LFM. |
| **Cơ chế GDN Prefill** | Không hỗ trợ cờ `--gdn-prefill-backend`. | Hỗ trợ và nhận diện cờ `--gdn-prefill-backend=flashinfer`. |
| **Quản lý VRAM & Swap** | Còn sử dụng cơ chế `--swap-space`. | Loại bỏ hoàn toàn `--swap-space` trong `arg_utils.py`, chuyển sang quản lý thuần VRAM. |
| **Lượng tử hóa KV Cache**| Hỗ trợ cơ bản. | Tối ưu hóa sâu cho định dạng **`fp8_e4m3`** trên phần cứng Hopper (H200). |
| **Điểm ERS Thực Tế** | **~49.00** (Nghẽn queue, fail 88/420 requests). | **61.26** (TTFT P95 = 70ms, hoàn thành 413/420 requests). |

## 📊 Kết Quả Chung Cuộc & Benchmark

Dự án khép lại chiến dịch với thứ hạng **170/281** chung cuộc. Cấu hình **V10** là mốc cân bằng tối ưu nhất được xác lập trong quá trình thử nghiệm thực tế:

| Chỉ Số | Kết Quả |
| :--- | :--- |
| **Thứ Hạng Chung Cuộc** | **170 / 281** |
| **Điểm ERS Đạt Được** | **61.26** |
| **TTFT P50** | `43ms` |
| **TTFT P95** | `70ms` |
| **Tỷ Lệ Thành Công** | `413 / 420` |
| **TBT Median** | `4ms` |

## 💡 Chiến Lược Tối Ưu Hóa

Cấu hình V10 được xây dựng dựa trên việc khai thác sâu các đặc tính vận hành của vLLM:

1. **Khai Thác Prefix Caching Cho Hội Thoại Đa Lượt:**
   * Vì máy chấm gửi lại toàn bộ lịch sử hội thoại cho cùng một worker, cờ `--enable-prefix-caching` giúp tái sử dụng KV cache của các lượt trước. TTFT ở các turn sau được giảm xuống gần bằng 0.

2. **Ép Băng Thông Bằng FP8 (Online Quantization):**
   * Sử dụng kết hợp `--kv-cache-dtype=fp8_e4m3` và `--quantization=fp8`.
   * **Ưu điểm:** Giảm dung lượng chiếm dụng bộ nhớ đệm, cho phép H200 prefill prompt ở tốc độ 70ms.
   * **Đánh đổi:** Việc ép kiểu tạ mô hình online tạo thêm áp lực tính toán lên 3 CPU cores, gây ra hiện tượng chậm nhẹ ở pha Decode (đánh rơi 7 request). Đây là sự đánh đổi có tính toán để duy trì điểm TTFT ở mức cao nhất.

3. **Thiết Lập Khung An Toàn (Safety Bounds):**
   * Giới hạn `--max-num-seqs=14`, `--max-num-batched-tokens=4096` cùng `--gpu-memory-utilization=0.95`.
   * Đảm bảo khi lưu lượng request tăng đột biến (Poisson burst), VRAM không bị phân mảnh và loại bỏ hoàn toàn nguy cơ sập container do OOM.

## 🚀 Cấu Hình Triển Khai

File cấu hình chính thức `docker-compose.yml` đạt mốc 61.26 ERS:

```yaml
services:
  model:
    image: 24521569/lfm-optimized:v1
    entrypoint:
      - python3
      - -m
      - vllm.entrypoints.openai.api_server
    command:
      # [1] Tham số dịch vụ cơ bản
      - --model=/model
      - --served-model-name=LFM2.5-1.2B-Instruct
      - --host=0.0.0.0
      - --port=8000
      - --tensor-parallel-size=1
      
      # [2] Quản lý bộ nhớ VRAM H200 (MIG 7)
      - --max-model-len=5120
      - --gpu-memory-utilization=0.95
      
      # [3] Kiểm soát hàng đợi an toàn cho 3 CPU Cores
      - --max-num-seqs=14
      - --max-num-batched-tokens=4096
      
      # [4] Ép xung bộ nhớ đệm và lượng tử hóa
      - --quantization=fp8
      - --kv-cache-dtype=fp8_e4m3
      - --dtype=bfloat16
      
      # [5] Tối ưu hóa Prefill & Tái sử dụng Cache
      - --enable-prefix-caching
      - --gdn-prefill-backend=flashinfer
      
      # [6] Triệt tiêu Overhead CPU từ Logging
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

## ⚙️ Hướng Dẫn Cài Đặt & Chạy Thử

Để thiết lập và kiểm tra môi trường trên máy chủ local trước khi nộp bài:

### 1. Chuẩn Bị Môi Trường

* **Hệ điều hành:** Linux (Khuyến nghị Ubuntu 20.04/22.04 LTS).
* **NVIDIA Driver & CUDA:** Tương thích CUDA 13.0+ (Driver >= 535.x).
* **Docker & NVIDIA Container Toolkit:** Đảm bảo Docker có quyền truy cập GPU.
```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

```



### 2. Chuẩn Bị Tệp Trọng Số (Model Weights)

Tải mô hình **LFM2.5-1.2B-Instruct** từ HuggingFace về thư mục cục bộ để mount vào container:

```bash
mkdir -p vllm-lfm-serving && cd vllm-lfm-serving
git clone [https://huggingface.co/viettel/LFM2.5-1.2B-Instruct](https://huggingface.co/viettel/LFM2.5-1.2B-Instruct) ./model_weights

```

### 3. Khởi Chạy Container

Thêm khối `volumes` vào `docker-compose.yml` để mount thư mục trọng số mô hình:

```yaml
    ports:
      - "8000:8000"
    shm_size: "2g"
    volumes:
      - ./model_weights:/model   # Mount trọng số local vào /model trong container
    deploy:
      resources:
        # ...

```

Chạy container ở chế độ chạy ngầm:

```bash
docker compose up -d

```

### 4. Kiểm Tra Vận Hành

Xem log khởi tạo mô hình và KV cache:

```bash
docker compose logs -f model

```

Gửi request kiểm tra độ trễ phản hồi qua API:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LFM2.5-1.2B-Instruct",
    "messages": [
      {"role": "user", "content": "Xin chào, hãy giới thiệu ngắn gọn về bạn."}
    ],
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": true
  }'

```

## ⚠️ Các Cạm Bẫy Cần Tránh

Quá trình dịch ngược mã nguồn `arg_utils.py` và phân tích log thử nghiệm đã chỉ ra các cờ cấu hình gây lỗi nghiêm trọng:

* ❌ `--swap-space`: Đã bị xóa hoàn toàn khỏi mã nguồn vLLM bản này; khai báo cờ sẽ khiến container sập ngay lập tức (`Exit Code 2`).
* ❌ `--performance-mode=throughput`: Tự động nhân đôi `max_num_seqs` (từ 14 lên 28) ở tầng C++, vượt quá khả năng xử lý của 3 CPU cores và làm timeout hơn 80 requests.
* ❌ `--kv-cache-dtype=fp8_e5m2`: Gây xung đột kernel tính toán với LFM2.5 ở pha Decode, kéo `tokens_per_sec` về gần bằng 0. Bắt buộc dùng `fp8_e4m3`.

---

## 👨‍💻 Tác Giả

* **Phan Khánh Tâm**
* Khoa Kỹ thuật Máy tính — Trường Đại học Công nghệ Thông tin, ĐHQG-HCM (UIT)
* Thành tích: Hạng **170 / 281** tại **Viettel AI Race 2026** (ERS: 61.26)

```

```
