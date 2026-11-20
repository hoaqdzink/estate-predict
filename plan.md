# Kế hoạch Đề tài 06: Dự báo giá bất động sản đô thị

Dữ liệu: `data/raw/vietnam_housing_dirty.csv` (3.500 tin, Hà Nội và TP.HCM)
Biến mục tiêu: `price_billion_vnd` (tỷ VNĐ)

---

## Chuẩn bị

- [x] Tạo môi trường ảo `.venv` và cài `requirements.txt`
- [x] Thêm thư viện vào `requirements.txt`: `scipy`, `lightgbm`, `mapie`, `geopy`, `shap`
- [x] Khởi tạo Git (`git init`) và commit lần đầu

---

## Bước 0: Làm sạch dữ liệu (`notebooks/01_cleaning.ipynb`)

Đầu vào: `data/raw/vietnam_housing_dirty.csv`. Đầu ra: `data/processed/clean.csv`.

### 0.1 Lỗi ở mức file
- [x] Đọc file với `dtype=str` để giữ nguyên giá trị gốc
- [x] Bỏ các dòng tiêu đề bị lặp giữa file
- [x] Bỏ các dòng rỗng (toàn dấu phẩy)
- [x] Phát hiện các dòng lệch cột, sửa lại hoặc bỏ

### 0.2 Chuẩn hóa chữ
- [x] Chuẩn hóa Unicode về NFC
- [x] Bỏ khoảng trắng thừa, ký tự xuống dòng, tab
- [x] Bỏ thẻ HTML (`<b>`, `&nbsp;`), emoji, cụm quảng cáo (`CHÍNH CHỦ`, `!!!`)
- [x] Sửa lỗi font (mojibake)
- [x] Chuẩn hóa `city` về 2 giá trị: `Hà Nội`, `Hồ Chí Minh`
- [x] Chuẩn hóa `district`: bỏ tiền tố `Quận/Q./Huyện`, thêm dấu, sửa chính tả, đưa về một tên chuẩn
- [x] Sửa các dòng bị đổi chỗ `city` và `district`
- [x] Phát hiện quận không thuộc thành phố (dùng `address` hoặc `title` để kiểm tra)
- [x] Chuẩn hóa `house_type` về 2 giá trị: `Nhà mặt tiền`, `Nhà trong hẻm`
- [x] Chuẩn hóa 4 cột có/không về `True/False`

### 0.3 Chuyển chữ thành số
- [x] Gom các ký hiệu thiếu (`N/A`, `-`, `?`, `không rõ`, `null`...) về NaN
- [x] Viết hàm đọc giá: `tỷ`, `triệu`, `5 tỷ 200 triệu`, `8ty7`, `5.200.000.000`, `5.20e+09`, `$`, số full-width, khoảng giá, `triệu/m²`
- [x] Phát hiện giá ghi theo đơn vị triệu (ví dụ `6500`) và đổi về tỷ
- [x] Viết hàm đọc diện tích: `m2`, `m²`, `ha`, `4x12`, `50-55`, `~50`, `mét vuông`, số có số 0 ở đầu
- [x] Viết hàm đọc kích thước: dấu phẩy thập phân, đơn vị `m`, `cm`, `nở hậu`, `12/13`
- [x] Sửa các ô bị Excel đổi thành ngày (`5-Thg4` thành `4.5`)
- [x] Viết hàm đọc số tầng: `lầu`, `tầng`, `1 trệt 2 lầu`, `lửng`, `tum`, chữ (`hai`)
- [x] Viết hàm đọc số phòng ngủ: `PN`, `phòng`, `3-4`, `3+`
- [x] Viết hàm đọc độ rộng đường: `hẻm xe hơi`, `ba gác`
- [x] Chuẩn hóa `posted_date` về dạng ngày (xử lý `Hôm nay`, `Hôm qua`, ngày kiểu Mỹ, số ngày Excel, ngày không hợp lệ)

### 0.4 Giá trị vô lý
- [x] Số âm, 0 tầng, quá nhiều tầng, quá nhiều phòng: đổi thành NaN
- [x] Giá `0`, `-1`, `999`, `Liên hệ`, `Thỏa thuận`: đổi thành NaN
- [x] Diện tích gõ thừa số 0: phát hiện bằng so sánh với `width × length` hoặc với `title` (còn 8 căn không có thông tin đối chiếu, để Bước 1.4 xử lý)
- [x] Đảo lại các dòng có chiều ngang lớn hơn chiều dài
- [x] Kiểm tra giá trong `title` có khớp với cột giá không

### 0.5 Trùng lặp
- [x] Bỏ dòng trùng y hệt (trừ cột `order`)
- [x] Bỏ tin trùng theo `url` (sau khi bỏ `?utm...`, `#...`)
- [x] Bỏ tin đăng lại: cùng quận, diện tích, số tầng, số phòng và giá lệch dưới 5%

### 0.6 Giá trị thiếu
- [x] Bỏ các dòng không có giá
- [ ] Điền thiếu cho biến số bằng trung vị theo quận (chuyển sang pipeline ở Bước 2, để tránh rò rỉ dữ liệu test)
- [ ] Điền thiếu cho biến phân loại bằng giá trị phổ biến nhất hoặc nhãn `Không rõ` (chuyển sang Bước 2)

### 0.7 Kết quả
- [x] Bảng số dòng trước và sau từng bước làm sạch
- [x] Lưu `data/processed/clean.csv`

---

## Bước 1: EDA và tiền xử lý (`notebooks/02_eda.ipynb`)

### 1.1 Khám phá dữ liệu
- [ ] Thống kê mô tả các cột
- [ ] Biểu đồ phân phối giá, diện tích, giá/m²
- [ ] So sánh giá theo thành phố, quận, loại nhà (boxplot)
- [ ] Ma trận tương quan giữa các biến số

### 1.2 Biến đổi phân phối giá
- [ ] Tính độ lệch (skewness) của giá
- [ ] Vẽ histogram và Q-Q plot trước khi biến đổi
- [ ] Thử biến đổi log1p
- [ ] Thử biến đổi Box-Cox (`scipy.stats.boxcox`), ghi lại `lambda`
- [ ] So sánh và chọn cách biến đổi

### 1.3 Dữ liệu vị trí
- [ ] Lập danh sách các quận trong dữ liệu
- [ ] Geocode tâm quận bằng Nominatim (`geopy`), lưu vào `data/processed/district_coords.csv`
- [ ] Tính khoảng cách tới trung tâm (Hồ Hoàn Kiếm cho Hà Nội, chợ Bến Thành cho TP.HCM)
- [ ] (Không bắt buộc) Đếm tiện ích theo quận từ OpenStreetMap (trường học, bệnh viện, chợ)

### 1.4 Ngoại lai
- [ ] Loại ngoại lai bằng IQR trên giá/m², tính riêng từng thành phố
- [ ] Loại ngoại lai bằng khoảng cách Mahalanobis (ngưỡng χ², p = 0,975)
- [ ] So sánh số dòng bị loại theo hai cách và chọn cách dùng

### 1.5 Đặc trưng
- [ ] Tạo tỷ lệ ngang/dài
- [ ] Mã hóa `district` (one-hot hoặc target encoding)
- [ ] Mã hóa `city`, `house_type`
- [ ] Không đưa giá/m² vào mô hình (rò rỉ thông tin giá)
- [ ] Lưu `data/processed/features.csv`

---

## Bước 2: Mô hình hóa (`notebooks/03_modeling.ipynb`)

- [ ] Chia train/test 80/20, cố định `random_state`
- [ ] Dựng pipeline `ColumnTransformer` (scaler cho biến số, encoder cho biến phân loại)
- [ ] Mô hình cơ sở: Linear Regression
- [ ] Ridge Regression: dò `alpha` bằng CV 5 fold
- [ ] Lasso: dò `alpha` bằng CV 5 fold
- [ ] ElasticNet: dò `alpha` và `l1_ratio`
- [ ] Gradient Boosting Regressor: dò tham số
- [ ] LightGBM: dò tham số
- [ ] Stacking Ensemble (các mô hình trên, mô hình meta là Ridge)
- [ ] Lưu các mô hình vào `models/*.joblib`

---

## Bước 3: Đánh giá (`notebooks/04_evaluation.ipynb`)

- [ ] Đổi giá dự đoán về đơn vị tỷ VNĐ trước khi tính chỉ số
- [ ] Tính RMSE, MAE, R², MAPE trên tập test
- [ ] Xuất bảng so sánh ra `reports/metrics.csv`
- [ ] Biểu đồ giá thật so với giá dự đoán
- [ ] Biểu đồ phần dư
- [ ] Độ quan trọng của đặc trưng (LightGBM hoặc SHAP)
- [ ] Phân tích sai số theo thành phố và loại nhà
- [ ] Lưu biểu đồ vào `reports/figures/`
- [ ] Chọn mô hình tốt nhất

---

## Bước 4: Ứng dụng (`src/predict.py`, `app/streamlit_app.py`)

- [ ] Viết hàm `predict()` nhận thông số căn nhà và trả về giá
- [ ] Tính khoảng dự báo 95% (chọn một: LightGBM quantile hoặc conformal prediction bằng MAPIE)
- [ ] Form nhập: thành phố, quận, loại nhà, diện tích, kích thước, số tầng, số phòng, tiện ích
- [ ] Hiển thị giá dự báo kèm khoảng 95%
- [ ] Kiểm tra dữ liệu nhập (diện tích dương, quận đúng thành phố...)
- [ ] Chạy thử `streamlit run app/streamlit_app.py`

---

## Báo cáo

- [ ] Mô tả dữ liệu và nguồn
- [ ] Quy trình làm sạch và bảng số dòng trước/sau
- [ ] Kết quả EDA và lựa chọn biến đổi
- [ ] Bảng so sánh mô hình
- [ ] Giải thích khoảng dự báo 95% (khác với khoảng tin cậy)
- [ ] Hạn chế: không có tình trạng pháp lý, vị trí chỉ ở cấp quận, giá rao bán chứ không phải giá giao dịch, dữ liệu năm 2021
- [ ] Ảnh chụp ứng dụng
- [ ] Cập nhật `README.md`
