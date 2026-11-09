# Spatial KIS — Hệ Thống Tìm Kiếm Video Theo Không Gian (Known-Item Search)

Hệ thống truy vấn video cục bộ độc lập (*Local-first Video Retrieval System*), tích hợp mô hình thị giác - ngôn ngữ **OpenCLIP ViT-B-16**, cơ chế chia lưới không gian **Spatial Grid 3×3**, cơ sở dữ liệu vector **Milvus**, cùng mô hình dịch ngôn ngữ offline **Opus-MT (Vi $\rightarrow$ En)**.

---

## 🌟 Tính Năng Nổi Bật

1. **Phân Vùng Không Gian Chính Xác (Spatial Regions)**:
   - Mỗi keyframe được trích xuất thành **10 vector đặc trưng**:
     - `region_id = 0`: Toàn bộ khung hình (*Global Frame*).
     - `region_id = 1..9`: Lưới 3×3 (*Row-major Local Crops* với độ trùm mép 12% để tránh mất vật thể ở biên).
   - Hỗ trợ các vùng tìm kiếm định danh: `all` (toàn ảnh), `left` (cột 1, 4, 7), `right` (cột 3, 6, 9), `center` (ô 5), `top` (hàng 1, 2, 3), `bottom` (hàng 7, 8, 9) và 4 góc riêng biệt (`top_left`, `top_right`, `bottom_left`, `bottom_right`).
2. **Truy Vấn Đa Điều Kiện (Multi-condition Conjunction)**:
   - Cho phép kết hợp nhiều mô tả vật thể khác nhau với các vị trí không gian tương ứng (Logic AND giữa các điều kiện, Logic OR giữa các ô trong cùng một vùng không gian).
   - Tùy chỉnh trọng số (*weight*) cho từng điều kiện tìm kiếm.
3. **Làm Giàu Truy Vấn Ngôn Ngữ Ngoại Tuyến (Offline Vi-En Expansion)**:
   - Tích hợp mô hình `Helsinki-NLP/opus-mt-vi-en` chạy trực tiếp trên GPU/CPU nội bộ, không phụ thuộc internet và không lo bị giới hạn API (rate-limit).
   - Hiển thị trực tiếp ô **✨ Query sau khi làm giàu (EN)** để người dùng kiểm tra và chủ động điều chỉnh từ khóa tiếng Anh trước khi tìm kiếm.
4. **Trình Phát Video Trực Tiếp Tại Frame Khớp**:
   - Xem video phát trực tiếp ngay tại mốc thời gian (*timestamp*) của keyframe tìm được.
   - Tự động ngắt phát video và âm thanh ngay khi đóng cửa sổ xem.
5. **Cơ Sở Dữ Liệu Vector Quy Mô Lớn**:
   - Sử dụng **Milvus Standalone v2.5.16** quản lý hơn 6.34 triệu vector trên toàn bộ 726 video (thuộc tập dữ liệu L01–L30), hỗ trợ chỉ mục `HNSW_PQ` cho tốc độ truy vấn mili-giây.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```
spatial-kis/
├── artifacts/
│   └── current/                 # Chứa dữ liệu index hiện hành (manifest, embeddings, images)
│       ├── manifest.jsonl       # Danh mục các keyframe đã trích xuất
│       ├── active_index.json    # Cấu hình index Milvus đang hoạt động
│       ├── images/              # Ảnh thumbnail keyframe định dạng WebP
│       └── session_history.jsonl# Lịch sử truy vấn & phản hồi người dùng
├── config/
│   ├── preprocess.json          # Cấu hình phân đoạn cảnh, ngưỡng độ nét và trích frame
│   └── index_profiles.json      # Cấu hình profile index Milvus (HNSW_PQ / GPU_CAGRA)
├── docker-compose.yml           # Khởi chạy Milvus, MinIO, etcd
├── milvus.yaml                  # Cấu hình bộ nhớ, cache và network cho Milvus
├── pyproject.toml               # Khai báo gói và dependency
├── src/
│   └── spatial_kis/
│       ├── api.py               # FastAPI backend & serving static Web UI
│       ├── expander.py          # Dịch thuật & làm giàu câu truy vấn bằng Opus-MT
│       ├── pipeline.py          # Pipeline trích xuất keyframe, crop lưới 3x3 và encode CLIP
│       ├── ranking.py           # Thuật toán xếp hạng, kết hợp điều kiện và đa dạng hóa shot
│       ├── regions.py           # Định nghĩa và ánh xạ tọa độ 10 vùng không gian
│       ├── import_milvus.py     # Nạp vector embeddings vào Milvus Collection
│       └── verify_artifact.py   # Kiểm tra tính toàn vẹn của artifact
├── tests/                       # Unit tests kiểm thử ranking và regions
└── web/                         # Giao diện người dùng thuần túy (Vanilla HTML/CSS/JS)
    ├── index.html               # Trang giao diện tìm kiếm
    ├── app.js                   # Xử lý tương tác, gọi API và điều khiển video player
    └── style.css                # Giao diện Dark-theme hiện đại
```

---

## 🚀 Hướng Dẫn Cài Đặt & Khởi Chạy

### 1. Chuẩn Bị Môi Trường
Sử dụng môi trường conda `retrieval_baseline` đã được cài đặt sẵn các gói cần thiết:
```powershell
conda activate retrieval_baseline
```

### 2. Khởi Động Vector Database (Milvus)
Di chuyển vào thư mục `spatial-kis` và khởi động cụm Docker containers:
```powershell
cd D:\Courses\HK7\CS336\spatial-kis
docker compose up -d
```
> Kiểm tra trạng thái bằng lệnh `docker ps`. Cần có 3 containers: `spatial-kis-milvus`, `spatial-kis-minio`, `spatial-kis-etcd` ở trạng thái `Up`.

### 3. Khởi Chạy Giao Diện Web & API Server
Chạy máy chủ FastAPI bằng lệnh sau:

**Trên PowerShell**:
```powershell
$env:PYTHONPATH = "D:\Courses\HK7\CS336\spatial-kis\src"
$env:SPATIAL_KIS_ARTIFACT = "D:\Courses\HK7\CS336\spatial-kis\artifacts\current"
$env:SPATIAL_KIS_VIDEO_ROOT = "D:\Courses\HK7\CS336\videos"
$env:MILVUS_URI = "http://localhost:19530"

& "C:\Users\phung\miniconda3\envs\retrieval_baseline\python.exe" -m uvicorn spatial_kis.api:app --host 0.0.0.0 --port 8010
```

**Trên CMD (Command Prompt)**:
```cmd
set PYTHONPATH=D:\Courses\HK7\CS336\spatial-kis\src
set SPATIAL_KIS_ARTIFACT=D:\Courses\HK7\CS336\spatial-kis\artifacts\current
set SPATIAL_KIS_VIDEO_ROOT=D:\Courses\HK7\CS336\videos
set MILVUS_URI=http://localhost:19530

C:\Users\phung\miniconda3\envs\retrieval_baseline\python.exe -m uvicorn spatial_kis.api:app --host 0.0.0.0 --port 8010
```

Sau khi máy chủ khởi động thành công, mở trình duyệt web và truy cập:
👉 **`http://localhost:8010`**

---

## 🛑 Dừng Hệ Thống Khi Không Sử Dụng

1. **Dừng API Server**: Nhấn `Ctrl + C` tại cửa sổ dòng lệnh đang chạy uvicorn.
2. **Dừng Docker Milvus**:
   ```powershell
   cd D:\Courses\HK7\CS336\spatial-kis
   docker compose stop
   ```
   *(Lệnh `docker compose stop` sẽ lưu giữ nguyên vẹn dữ liệu vector trong thư mục `volumes/` mà không bị mất mát).*

---

## 📖 Hướng Dẫn Sử Dụng Giao Diện

1. **Nhập mô tả & Chọn vị trí không gian**:
   - Nhập từ khóa hoặc câu mô tả vào ô chính (hỗ trợ cả tiếng Việt lẫn tiếng Anh).
   - Chọn vị trí không gian tương ứng trong dropdown (ví dụ: `left` nếu đối tượng ở bên trái, `right` nếu ở bên phải, `center` nếu ở chính giữa).
   - Điều chỉnh trọng số (Weight) nếu điều kiện đó quan trọng hơn các điều kiện khác.
2. **Làm giàu truy vấn (Dịch sang tiếng Anh)**:
   - Bấm nút **🌐 Dịch / Làm giàu** để mô hình AI dịch câu tiếng Việt sang tiếng Anh và hiển thị kết quả vào ô **✨ Query sau khi làm giàu (EN)**.
   - Bạn có thể chỉnh sửa trực tiếp nội dung trong ô này nếu muốn tinh chỉnh từ vựng trước khi bấm tìm kiếm.
3. **Thêm điều kiện**:
   - Bấm **+ Thêm điều kiện** để mô tả thêm đối tượng thứ hai, thứ ba trong cùng một khung hình (hệ thống sẽ tìm các frame thỏa mãn đồng thời tất cả các điều kiện).
4. **Xem kết quả & Video**:
   - Bấm **🔍 Bắt đầu Tìm kiếm**.
   - Kết quả hiển thị kèm ảnh thumbnail, mã video, mốc thời gian, điểm số tương đồng (*Score*) và tên ô không gian khớp nhất.
   - Bấm **▶ Xem video** để mở trình phát video tua đến chính xác khoảnh khắc của keyframe đó.
   - Bấm **✕ Đóng** hoặc nhấn phím `Esc` để dừng video và tắt âm thanh ngay lập tức.
   - Bấm **✓** (Chính xác) hoặc **×** (Không liên quan) để lưu đánh giá vào lịch sử phiên làm việc.

---

## 🧪 Kiểm Thử Hệ Thống (Unit Tests)

Chạy kiểm thử cho bộ định tuyến không gian và thuật toán xếp hạng:
```powershell
$env:PYTHONPATH = "D:\Courses\HK7\CS336\spatial-kis\src"
& "C:\Users\phung\miniconda3\envs\retrieval_baseline\python.exe" -m unittest spatial-kis/tests/test_regions.py spatial-kis/tests/test_ranking.py
```
Tất cả các kiểm thử đều vượt qua (100% Passed).
