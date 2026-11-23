# Dự báo giá bất động sản đô thị

**Đề tài 06:** Urban Real Estate Valuation & Hedonic Pricing. Bài toán hồi quy ước lượng giá rao bán nhà ở tại Hà Nội và TP.HCM dựa trên diện tích, vị trí, tiện ích xung quanh và đặc điểm căn nhà.

Dự án gồm hai phần:
- **Notebook** `notebooks/01_cleaning.ipynb`: làm sạch dữ liệu, tiền xử lý, huấn luyện và đánh giá mô hình.
- **Ứng dụng Streamlit** `app/streamlit_app.py`: trình bày kết quả và định giá trực tuyến kèm khoảng dự báo 95%.

**Kết quả** (mô hình Stacking, tập kiểm thử độc lập 564 bản ghi): RMSE ≈ 3,21 tỷ, MAE ≈ 1,98 tỷ, R² ≈ 0,76, MAPE ≈ 24,8%.

---

## 1. Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|---|---|
| Python | 3.9 trở lên (đã kiểm thử với Python 3.9.6) |
| Hệ điều hành | macOS, Linux hoặc Windows |
| Dung lượng | khoảng 1 GB cho môi trường ảo và thư viện |
| Kết nối mạng | chỉ cần khi cài thư viện; notebook và ứng dụng chạy không cần mạng |

**Riêng macOS:** LightGBM cần thư viện OpenMP. Cài bằng Homebrew trước khi chạy:

```bash
brew install libomp
```

Nếu thiếu thư viện này, khi import LightGBM sẽ gặp lỗi `Library not loaded: @rpath/libomp.dylib`.

---

## 2. Cài đặt

### Bước 1. Lấy mã nguồn

```bash
git clone <địa chỉ repo> estate-predict
cd estate-predict
```

Hoặc giải nén tệp dự án rồi mở Terminal tại thư mục `estate-predict`.

### Bước 2. Tạo môi trường ảo

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Khi môi trường ảo được kích hoạt, đầu dòng lệnh hiện `(.venv)`.

### Bước 3. Cài thư viện

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Bước 4. Kiểm tra cài đặt

```bash
python -c "import pandas, sklearn, lightgbm, shap, streamlit; print('Cài đặt thành công')"
```

---

## 3. Chạy notebook

Notebook đọc dữ liệu gốc và tạo ra toàn bộ kết quả: dữ liệu đã xử lý, mô hình, bảng đánh giá và biểu đồ. **Cần chạy notebook ít nhất một lần trước khi mở ứng dụng.**

**Cách 1: dùng trình soạn thảo có hỗ trợ Jupyter** (VS Code, Kiro, JupyterLab)
1. Mở `notebooks/01_cleaning.ipynb`.
2. Chọn kernel Python của môi trường ảo `.venv`.
3. Chọn *Run All*. Thời gian chạy khoảng 3 phút.
4. Ô cuối cùng hiện danh sách khóa của `learnt.pkl` và kết quả trên tập kiểm thử là hoàn tất.

**Cách 2: chạy từ dòng lệnh**

```bash
jupyter nbconvert --to notebook --execute notebooks/01_cleaning.ipynb --inplace --ExecutePreprocessor.timeout=1800
```

**Các tệp được tạo ra:**

| Tệp | Nội dung |
|---|---|
| `data/processed/clean.csv` | Dữ liệu sau làm sạch (2.945 bản ghi) |
| `data/processed/features.csv` | Dữ liệu cho mô hình, sau tiền xử lý (2.819 bản ghi) |
| `models/learnt.pkl` | Mô hình tốt nhất, gồm pipeline tiền xử lý và tham số khoảng dự báo 95% |
| `models/all_models.pkl` | Tất cả mô hình đã huấn luyện |
| `reports/cleaning_log.csv` | Số bản ghi sau từng bước xử lý |
| `reports/best_params.csv` | Tham số tốt nhất của từng mô hình (grid search) |
| `reports/cv_results.csv` | Kết quả cross-validation trên tập huấn luyện |
| `reports/metrics.csv` | Kết quả trên tập kiểm thử |
| `reports/figures/*.png` | Biểu đồ đánh giá |

Notebook cho cùng một kết quả ở mọi lần chạy. Các tệp trên được ghi đè mỗi lần chạy; dữ liệu gốc không bị thay đổi.

---

## 4. Chạy ứng dụng Streamlit

```bash
streamlit run app/streamlit_app.py
```

Trình duyệt tự mở tại `http://localhost:8501`. Dừng ứng dụng bằng `Ctrl + C` trong Terminal.

Ứng dụng gồm 6 trang:

| Trang | Nội dung |
|---|---|
| 1. Ý tưởng | Bài toán, lý thuyết định giá hedonic, quy trình thực hiện |
| 2. Dẫn dắt | Dữ liệu thô, làm sạch, mất cân bằng dữ liệu, phân phối giá, vị trí và tiện ích xung quanh |
| 3. Phản biện | Các lựa chọn đã cân nhắc (Box-Cox/Log1p, IQR/Mahalanobis, trích chọn thuộc tính, Stacking...) và hạn chế |
| 4. Minh chứng | So sánh mô hình, sai số theo phân khúc, giải thích SHAP, khoảng dự báo 95% |
| 5. Mô hình huấn luyện | Cấu trúc mô hình Stacking, trọng số mô hình meta, nội dung `learnt.pkl`, code mẫu nạp mô hình để dự báo, minh họa từng bước tính trên một căn thật |
| 6. Định giá trực tuyến | Nhập thông số căn nhà, nhận giá dự báo, khoảng dự báo 95%, giải thích cách mô hình tính ra giá (dự báo của từng mô hình con, giá trị SHAP của từng thuộc tính) và các căn tương tự |

---

## 5. Xử lý lỗi thường gặp

| Hiện tượng | Nguyên nhân và cách xử lý |
|---|---|
| Ứng dụng báo *"Thiếu các tệp kết quả"* | Chưa chạy notebook. Thực hiện mục 3 rồi mở lại ứng dụng. |
| `Library not loaded: @rpath/libomp.dylib` (macOS) | Thiếu OpenMP cho LightGBM. Chạy `brew install libomp`. |
| `ModuleNotFoundError` | Môi trường ảo chưa được kích hoạt, hoặc notebook dùng sai kernel. Kích hoạt `.venv` (mục 2, bước 2) và chọn kernel `.venv` trong notebook. |
| `command not found: streamlit` hoặc `jupyter` | Môi trường ảo chưa được kích hoạt. Hoặc chạy trực tiếp `.venv/bin/streamlit run app/streamlit_app.py`. |
| Cảnh báo `NotOpenSSLWarning` (macOS) | Do bản Python có sẵn của macOS dùng LibreSSL. Không ảnh hưởng tới kết quả. |
| Notebook báo lỗi ở một ô giữa chừng | Các ô phải chạy tuần tự. Chọn *Restart Kernel* rồi *Run All*. |

---

## 6. Cấu trúc thư mục

```
estate-predict/
├── data/
│   ├── raw/
│   │   └── vietnam_housing_dirty.csv   # Dữ liệu gốc (3.510 dòng, chưa làm sạch)
│   └── processed/                      # Do notebook tạo ra
├── notebooks/
│   └── 01_cleaning.ipynb               # Toàn bộ quy trình phân tích và huấn luyện
├── models/
│   ├── learnt.pkl                      # Mô hình tốt nhất (có trong Git)
│   └── all_models.pkl                  # Do notebook tạo ra
├── app/
│   └── streamlit_app.py                # Ứng dụng trình bày và định giá
├── reports/                            # Do notebook tạo ra
├── requirements.txt
├── .gitignore
└── README.md
```

Nội dung notebook:

| Mục | Nội dung |
|---|---|
| 1–4 | Khai báo thư viện, nạp dữ liệu, làm sạch dữ liệu (19 bước), lưu kết quả |
| 5 | Tiền xử lý: mất cân bằng dữ liệu, phân phối giá, Box-Cox/Log1p, tọa độ, khoảng cách tới trung tâm, tiện ích xung quanh, ngoại lai IQR/Mahalanobis, tạo đặc trưng |
| 6 | Mô hình hóa: chia dữ liệu phân tầng, pipeline, trích chọn thuộc tính, grid search cross-validation, Stacking, lưu `learnt.pkl` |
| 7 | Đánh giá: RMSE, MAE, R², MAPE trên tập kiểm thử, sai số theo phân khúc, SHAP, khoảng dự báo 95%, kết luận |

**Tệp được Git bỏ qua** (`.gitignore`): môi trường ảo, cache, `plan.md`, và các tệp do notebook sinh ra trong `data/processed/`, `models/` (trừ `learnt.pkl`), `reports/`.

---

## 7. Dữ liệu

`data/raw/vietnam_housing_dirty.csv`: 3.500 tin rao bán nhà ở Hà Nội và TP.HCM, mô phỏng dữ liệu thu thập thô **chưa làm sạch**.

- Các cột về nhà và giá được trích từ tập dữ liệu công khai tại [AndrewBui-AI-CS/House_price_prediction](https://github.com/AndrewBui-AI-CS/House_price_prediction) (tệp `src/clean data/full.csv`, phiên bản tháng 01/2022). Theo mô tả của nhóm tác giả, dữ liệu được thu thập từ alonhadat.com.vn; thời điểm thu thập và giấy phép sử dụng không được công bố.
- Dữ liệu được cố ý thêm 36 loại lỗi để thực hành làm sạch: số dạng chữ, sai đơn vị, ngoại lai, lỗi Unicode, sai chính tả, sai cột, trùng lặp, dòng rác...
- Các cột `url`, `title`, `address`, `posted_date`, `contact`, `scraped_at` được sinh nhân tạo (url và số điện thoại là giả lập).
- Tọa độ quận/huyện và số tiện ích xung quanh lấy từ OpenStreetMap (bản trích xuất Việt Nam ngày 19/11/2026), được lưu cố định trong notebook.

| Cột | Ý nghĩa |
|---|---|
| `order` | Số thứ tự |
| `url`, `title`, `address`, `posted_date` | Đường dẫn, tiêu đề, địa chỉ, ngày đăng tin |
| `contact`, `scraped_at` | Liên hệ (đã che số), thời điểm thu thập |
| `city`, `district` | Thành phố, quận/huyện |
| `house_type` | Loại nhà: `Nhà mặt tiền` hoặc `Nhà trong hẻm` |
| `area_m2` | Diện tích đất (m²) |
| `width_m`, `length_m` | Chiều ngang, chiều dài (m) |
| `road_width_m` | Độ rộng đường/hẻm trước nhà (m) |
| `floors`, `bedrooms` | Số tầng, số phòng ngủ |
| `has_dining_room`, `has_kitchen`, `has_terrace`, `has_parking` | Có phòng ăn, nhà bếp, sân thượng, chỗ đỗ ô tô |
| `price_billion_vnd` | **Giá rao bán (tỷ VNĐ)**, biến mục tiêu |

**Hạn chế:** giá rao bán, không phải giá giao dịch thực tế; dữ liệu thu thập trước 01/2022; không có thông tin tình trạng pháp lý; vị trí chỉ ở cấp quận/huyện.
