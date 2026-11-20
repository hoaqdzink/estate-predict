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
│   ├── 01_cleaning.ipynb       # Làm sạch dữ liệu
│   ├── 02_eda.ipynb            # Phân tích khám phá, tiền xử lý
│   ├── 03_modeling.ipynb       # Huấn luyện mô hình (chưa tạo)
│   └── 04_evaluation.ipynb     # Đánh giá mô hình (chưa tạo)
├── reports/
│   └── cleaning_log.csv        # Số dòng sau từng bước làm sạch
├── requirements.txt            # Danh sách thư viện Python
├── .gitignore
└── README.md
```

| Thư mục / File | Vai trò |
|---|---|
| `data/raw/` | Dữ liệu gốc, giữ nguyên không chỉnh sửa. File chính là `vietnam_housing_dirty.csv`. |
| `data/processed/` | Dữ liệu đã làm sạch, do `notebooks/01_cleaning.ipynb` tạo ra. Không đưa lên Git. |
| `notebooks/01_cleaning.ipynb` | Làm sạch dữ liệu thô, ghi ra `data/processed/clean.csv`. |
| `notebooks/02_eda.ipynb` | Khám phá dữ liệu, biến đổi phân phối giá, vị trí, ngoại lai, tạo đặc trưng. |
| `notebooks/03_modeling.ipynb` | Huấn luyện mô hình và lưu vào `models/`. |
| `notebooks/04_evaluation.ipynb` | Tính RMSE, MAE, R², MAPE và lưu biểu đồ vào `reports/figures/`. |
| `reports/` | Báo cáo làm sạch; biểu đồ và kết quả đánh giá sẽ lưu tại đây. |
| `requirements.txt` | Danh sách thư viện cần cài. |
| `.gitignore` | Các file và thư mục Git bỏ qua (môi trường ảo, cache, dữ liệu xử lý). |

Luồng xử lý: `data/raw` → `01_cleaning` → `data/processed` → `02_eda` → `03_modeling` → `04_evaluation` → ứng dụng (Bước 4).

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

Mở các notebook trong `notebooks/` theo thứ tự 01 → 04, chọn kernel `.venv`, rồi bấm **Run All**.

Dữ liệu gốc nằm tại `data/raw/vietnam_housing_dirty.csv`.
