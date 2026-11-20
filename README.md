# estate-predict

Dự báo giá nhà bằng Machine Learning.

## Cấu trúc thư mục

```
estate-predict/
├── data/
│   ├── raw/                    # Dữ liệu gốc
│   │   └── vietnam_housing_dirty.csv
│   └── processed/              # Dữ liệu sau xử lý
├── notebooks/
│   └── 01_cleaning.ipynb       # Làm sạch, tiền xử lý, mô hình hóa, đánh giá
├── models/
│   ├── learnt.pkl              # Mô hình tốt nhất (kèm pipeline và khoảng dự báo)
│   └── all_models.pkl          # Tất cả mô hình đã huấn luyện
├── app/
│   └── streamlit_app.py        # Ứng dụng trình bày và định giá
├── reports/
│   ├── cleaning_log.csv        # Số bản ghi sau từng bước
│   ├── best_params.csv         # Tham số tốt nhất (grid search)
│   ├── cv_results.csv          # Kết quả cross-validation
│   ├── metrics.csv             # Kết quả trên tập kiểm thử
│   └── figures/                # Biểu đồ đánh giá
├── requirements.txt            # Danh sách thư viện Python
├── .gitignore
└── README.md
```

| Thư mục / File | Vai trò |
|---|---|
| `data/raw/` | Dữ liệu gốc, giữ nguyên không chỉnh sửa. File chính là `vietnam_housing_dirty.csv`. |
| `data/processed/` | Dữ liệu đã làm sạch, do `notebooks/01_cleaning.ipynb` tạo ra. Không đưa lên Git. |
| `notebooks/01_cleaning.ipynb` | Toàn bộ quy trình: làm sạch (mục 1–4), tiền xử lý (mục 5), mô hình hóa (mục 6), đánh giá (mục 7). Tạo `data/processed/`, `models/`, `reports/`. |
| `models/learnt.pkl` | Mô hình tốt nhất (Stacking), gồm pipeline tiền xử lý và tham số khoảng dự báo 95%. |
| `app/streamlit_app.py` | Ứng dụng Streamlit: ý tưởng, dẫn dắt, phản biện, minh chứng, định giá trực tuyến. |
| `reports/` | Nhật ký làm sạch, tham số tốt nhất, kết quả cross-validation và kiểm thử, biểu đồ. |
| `requirements.txt` | Danh sách thư viện cần cài. |
| `.gitignore` | Các tệp Git bỏ qua: môi trường ảo, cache, `plan.md` và các tệp do notebook sinh ra (`data/processed/`, `models/`, `reports/`). |

Luồng xử lý: `data/raw` → `01_cleaning.ipynb` → `data/processed`, `models/learnt.pkl`, `reports/` → `app/streamlit_app.py`.

## Dữ liệu

`data/raw/vietnam_housing_dirty.csv`: 3.500 tin rao bán nhà ở Hà Nội và TP.HCM, mô phỏng dữ liệu crawl thô **chưa làm sạch**.

- Các cột về nhà và giá được trích từ tập dữ liệu công khai tại [AndrewBui-AI-CS/House_price_prediction](https://github.com/AndrewBui-AI-CS/House_price_prediction) (tệp `src/clean data/full.csv`, phiên bản tháng 01/2022). Theo mô tả của nhóm tác giả, dữ liệu được thu thập từ alonhadat.com.vn; thời điểm thu thập và giấy phép sử dụng không được công bố.
- Dữ liệu được cố ý thêm 36 loại lỗi để luyện làm sạch: số dạng chữ, sai đơn vị, ngoại lai, Unicode, sai chính tả, sai cột, trùng lặp, dòng rác...
- Các cột `url`, `title`, `address`, `posted_date`, `contact`, `scraped_at` **do máy sinh ra** (url, số điện thoại là giả), chỉ để giống dữ liệu crawl.
- Đây là **giá rao bán** (không phải giá giao dịch thực), thu thập trước tháng 01/2022.

Các cột chính:

| Cột | Ý nghĩa |
|---|---|
| `order` | Số thứ tự |
| `url`, `title`, `address`, `posted_date` | Đường dẫn, tiêu đề, địa chỉ, ngày đăng tin |
| `contact`, `scraped_at` | Liên hệ (đã che số), thời điểm thu thập |
| `city` | Thành phố (Hà Nội / Hồ Chí Minh) |
| `district` | Quận/huyện |
| `house_type` | Loại nhà: `Nhà mặt tiền` hoặc `Nhà trong hẻm` |
| `area_m2` | Diện tích đất (m²) |
| `width_m`, `length_m` | Chiều ngang, chiều dài (m), có thể trống |
| `road_width_m` | Độ rộng đường/hẻm trước nhà (m) |
| `floors` | Số tầng |
| `bedrooms` | Số phòng ngủ |
| `has_dining_room`, `has_kitchen`, `has_terrace`, `has_parking` | Có phòng ăn / bếp / sân thượng / chỗ đỗ ô tô |
| `price_billion_vnd` | **Giá rao bán (tỷ VNĐ)**, biến mục tiêu |

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Sử dụng

Thực thi toàn bộ `notebooks/01_cleaning.ipynb` (kernel `.venv`, *Run All*, khoảng 3 phút), sau đó chạy ứng dụng:

```bash
streamlit run app/streamlit_app.py
```

Dữ liệu gốc nằm tại `data/raw/vietnam_housing_dirty.csv`.
