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
│   └── eda.ipynb               # Phân tích khám phá dữ liệu
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Làm sạch và biến đổi dữ liệu
│   ├── train.py                # Huấn luyện mô hình
│   ├── evaluate.py             # Đánh giá mô hình
│   └── predict.py              # Dự đoán giá nhà
├── models/                     # Lưu các mô hình đã huấn luyện
├── reports/
│   └── figures/                # Biểu đồ và kết quả đánh giá
├── app/
│   └── streamlit_app.py        # Giao diện dự báo giá nhà
├── requirements.txt            # Danh sách thư viện Python
├── .gitignore
└── README.md
```

| Thư mục / File | Vai trò |
|---|---|
| `data/raw/` | Dữ liệu gốc, giữ nguyên không chỉnh sửa. File chính là `vietnam_housing_dirty.csv`. |
| `data/processed/` | Dữ liệu đã làm sạch, do `src/preprocessing.py` tạo ra. Không đưa lên Git. |
| `notebooks/eda.ipynb` | Khám phá dữ liệu: phân phối, giá trị thiếu, tương quan giữa các biến. |
| `src/preprocessing.py` | Đọc dữ liệu gốc, xử lý giá trị thiếu, mã hóa biến phân loại, chuẩn hóa, ghi ra `data/processed/`. |
| `src/train.py` | Huấn luyện mô hình trên dữ liệu đã xử lý và lưu mô hình vào `models/`. |
| `src/evaluate.py` | Tính các chỉ số đánh giá (MAE, RMSE, R²) và lưu biểu đồ vào `reports/figures/`. |
| `src/predict.py` | Nạp mô hình đã lưu và dự đoán giá cho dữ liệu mới. |
| `models/` | Các file mô hình đã huấn luyện (ví dụ `.joblib`). Không đưa lên Git. |
| `reports/figures/` | Biểu đồ EDA và kết quả đánh giá mô hình. |
| `app/streamlit_app.py` | Giao diện web để người dùng nhập thông tin nhà và xem giá dự đoán. |
| `requirements.txt` | Danh sách thư viện cần cài. |
| `.gitignore` | Các file và thư mục Git bỏ qua (môi trường ảo, cache, dữ liệu xử lý, mô hình). |

Luồng xử lý: `data/raw` → `preprocessing.py` → `data/processed` → `train.py` → `models` → `evaluate.py` / `predict.py` / `streamlit_app.py`.

## Dữ liệu

`data/raw/vietnam_housing_dirty.csv`: 3.500 tin rao bán nhà ở Hà Nội và TP.HCM, mô phỏng dữ liệu crawl thô **chưa làm sạch**.

- Các cột về nhà và giá lấy từ dữ liệu thật do nhóm sinh viên ĐH Bách khoa Hà Nội crawl từ [alonhadat.com.vn](https://alonhadat.com.vn/), công bố tại [AndrewBui-AI-CS/House_price_prediction](https://github.com/AndrewBui-AI-CS/House_price_prediction) (file `src/clean data/full.csv`, commit 01/2022). Repo gốc không ghi giấy phép sử dụng.
- Dữ liệu được cố ý thêm 36 loại lỗi để luyện làm sạch: số dạng chữ, sai đơn vị, ngoại lai, Unicode, sai chính tả, sai cột, trùng lặp, dòng rác...
- Các cột `url`, `title`, `address`, `posted_date`, `contact`, `scraped_at` **do máy sinh ra** (url, số điện thoại là giả), chỉ để giống dữ liệu crawl.
- Đây là **giá rao bán** (không phải giá giao dịch thực), thu thập khoảng năm 2021.

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

```bash
python src/preprocessing.py
python src/train.py
python src/evaluate.py
python src/predict.py
streamlit run app/streamlit_app.py
```

Dữ liệu gốc nằm tại `data/raw/vietnam_housing_dirty.csv`.
