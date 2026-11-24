# Dự báo giá bất động sản đô thị

**Môn học:** Khai phá dữ liệu (Data Mining). **Đề tài 06:** Urban Real Estate Valuation & Hedonic Pricing.

Áp dụng quy trình khai phá tri thức (KDD: lựa chọn → tiền xử lý → biến đổi → khai phá → diễn giải/đánh giá) với **tác vụ hồi quy** trên dữ liệu tin rao bán nhà ở Hà Nội và TP.HCM, nhằm tìm ra các yếu tố quyết định giá (diện tích, vị trí, tiện ích xung quanh, đặc điểm căn nhà) và dự báo giá chuyển nhượng.

Dự án gồm ba phần, dùng chung một mô hình (`models/learnt.pkl`):

| Phần | Tệp | Dùng để |
|---|---|---|
| **Notebook** | `notebooks/01_cleaning.ipynb` | Toàn bộ quy trình KDD: làm sạch, tiền xử lý, khai phá (huấn luyện mô hình hồi quy), đánh giá. Tạo ra mọi tệp kết quả |
| **Ứng dụng Streamlit** | `app/streamlit_app.py` | Trình bày đề tài theo mạch Ý tưởng → Dẫn dắt → Phản biện → Minh chứng (12 trang) |
| **Web Django** | `web/` | Trang định giá nhà và API JSON |

**Kết quả** (mô hình Stacking, tập kiểm thử độc lập 564 bản ghi): RMSE ≈ 3,21 tỷ, MAE ≈ 1,98 tỷ, R² ≈ 0,76, MAPE ≈ 24,8%; khoảng dự báo 95% có độ phủ 94,1%.

### Chạy nhanh

```bash
python3.14 -m venv .venv && source .venv/bin/activate   # tạo và bật môi trường ảo (Python 3.14)
pip install -r requirements-dev.txt                      # cài thư viện
jupyter nbconvert --to notebook --execute notebooks/01_cleaning.ipynb --inplace --ExecutePreprocessor.timeout=1800
streamlit run app/streamlit_app.py                       # http://localhost:8501
python web/manage.py runserver                           # http://127.0.0.1:8000 (mở Terminal khác)
```

Chi tiết từng bước ở các mục dưới.

---

## 1. Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|---|---|
| Python | **3.14** (đã kiểm thử với Python 3.14.10, cùng bản với Streamlit Community Cloud) |
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

Mô hình được lưu bằng các phiên bản thư viện cố định trong `app/requirements.txt`, nên cần đúng **Python 3.14** để các bản này cài được dạng đóng gói sẵn.

Nếu máy chưa có Python 3.14, có thể cài nhanh bằng [uv](https://docs.astral.sh/uv/): `pip install uv`, rồi `uv python install 3.14`.

macOS / Linux:

```bash
python3.14 -m venv .venv          # hoặc: uv venv --python 3.14 .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
```

Khi môi trường ảo được kích hoạt, đầu dòng lệnh hiện `(.venv)`.

### Bước 3. Cài thư viện

```bash
pip install --upgrade pip
pip install -r requirements-dev.txt
```

### Bước 4. Kiểm tra cài đặt

```bash
python -c "import pandas, sklearn, lightgbm, shap, streamlit, django; print('Cài đặt thành công')"
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
| `reports/artifacts.pkl` | Kết quả trung gian cho ứng dụng (điểm từng fold, lưới grid search, dự báo kiểm thử, phần dư ngoài fold, SHAP) |

Notebook cho cùng một kết quả ở mọi lần chạy. Các tệp trên được ghi đè mỗi lần chạy; dữ liệu gốc không bị thay đổi.

---

## 4. Chạy ứng dụng Streamlit

**Cách 1: dòng lệnh**

```bash
source .venv/bin/activate
streamlit run app/streamlit_app.py
```

**Cách 2: trong Kiro / VS Code:** bấm `Cmd + Shift + D` (Windows: `Ctrl + Shift + D`), chọn cấu hình **"Mở ứng dụng Streamlit"**, rồi bấm ▶.

Trình duyệt tự mở tại `http://localhost:8501` (nếu không, mở trình duyệt và gõ địa chỉ này). Lần đầu chạy, Streamlit có thể hỏi email trong Terminal: bấm `Enter` để bỏ qua. Dừng ứng dụng bằng `Ctrl + C` trong Terminal, hoặc nút ■ (Stop) nếu chạy bằng cách 2. Khi sửa mã nguồn trong lúc ứng dụng đang chạy, chỉ cần tải lại trang trình duyệt.

Ứng dụng gồm 12 trang, chia nhóm theo các giai đoạn KDD ở thanh bên trái. Mỗi trang trình bày theo mạch **Ý tưởng → Dẫn dắt → Phản biện → Minh chứng**, mỗi phần nằm trong một khung màu riêng.

| Nhóm | Trang | Nội dung |
|---|---|---|
| Giới thiệu | 0. Tổng quan | Thông tin đề tài, kết quả chính, quy trình KDD, bảng đối chiếu yêu cầu |
| Chuẩn bị dữ liệu (KDD 1–3) | 1. Dữ liệu & mất cân bằng | Dữ liệu thô, tổng quan 36 loại lỗi, mô tả thuộc tính, phân tích mất cân bằng |
| | 1b. Ví dụ giải tay | Tính tay trên 12 căn: hồi quy tuyến tính (phương trình chuẩn), dự báo, R², RMSE, Ridge, IQR, Haversine; đối chiếu với scikit-learn |
| | 2. Tiền xử lý | Làm sạch (trước/sau), phân phối giá và Box-Cox/Log1p, vị trí và tiện ích xung quanh, ngoại lai IQR/Mahalanobis |
| Khai phá dữ liệu (KDD 4) | 3. Chọn mẫu & Grid Search | Chia phân tầng, cross-validation 5 fold, Lasso, độ quan trọng hoán vị, lưới tham số của từng mô hình |
| | 4. Stacking & learnt.pkl | Cấu trúc mô hình, trọng số meta, nội dung `learnt.pkl`, code nạp mô hình, minh họa từng bước tính |
| | 5. Khoảng dự báo (bổ sung) | Cross-conformal, đường hiệu chỉnh độ phủ, so sánh thang log với sai số tuyệt đối |
| Diễn giải & đánh giá (KDD 5) | 6. So sánh mô hình | Bảng CV và kiểm thử, độ ổn định giữa các fold, so sánh cặp mô hình bằng bootstrap |
| | 7. Đánh giá & độ tin cậy | Khoảng tin cậy bootstrap của RMSE, MAE, R², MAPE; phần dư; sai số theo nhóm; khi nào nên tin dự báo |
| | 8. Giải thích kết quả (SHAP) | Mức đóng góp và chiều hướng ảnh hưởng của thuộc tính, quan hệ phi tuyến, giải thích từng căn |
| Tri thức & ứng dụng | 9. Định giá trực tuyến | Nhập thông số căn nhà → giá, khoảng dự báo, cách mô hình tính ra giá, căn tương tự |
| | 10. Tri thức & kết luận | 8 quy luật về giá khai phá được, kết quả, các lựa chọn đã phản biện, hạn chế, hướng phát triển |

**Cấu hình phân tích** (mục ⚙️ ở thanh bên trái): thành phố, ranh giới phân khúc giá, hệ số IQR, mức Mahalanobis, mức khoảng dự báo, số lần bootstrap, hạt giống ngẫu nhiên. Các bảng và biểu đồ liên quan được tính lại ngay; mô hình đã huấn luyện không thay đổi. Nút **Khôi phục mặc định** đưa cấu hình về ban đầu.

Giao diện (màu, phông chữ) cấu hình trong `.streamlit/config.toml` và `app/core/ui.py`.

---

### Triển khai lên Streamlit Community Cloud

Repo đã có sẵn những gì Streamlit Cloud cần: `app/requirements.txt` (thư viện cố định phiên bản trùng với lúc huấn luyện), `packages.txt` (thư viện hệ thống cho LightGBM), `.streamlit/config.toml` (giao diện), và các tệp dữ liệu ứng dụng đọc (`models/learnt.pkl`, `data/processed/clean.csv`, `data/processed/features.csv`, `reports/artifacts.pkl`, `reports/cleaning_log.csv`).

1. Đăng nhập [share.streamlit.io](https://share.streamlit.io) bằng tài khoản GitHub.
2. Chọn **Create app** → **Deploy a public app from GitHub**.
3. Điền: Repository `hoaqdzink/estate-predict`, Branch `main`, Main file path `app/streamlit_app.py`.
4. Mở **Advanced settings**, giữ **Python 3.14** (bản mặc định). Thư viện trong `app/requirements.txt` được cố định cho Python 3.14; chọn bản Python khác sẽ phải biên dịch thư viện từ mã nguồn và rất lâu.
5. Bấm **Deploy**. Lần đầu cài thư viện mất vài phút.

Sau khi chạy lại notebook, cần commit lại 5 tệp dữ liệu trên thì bản triển khai mới cập nhật kết quả.

---

## 5. Chạy web Django

Ngoài ứng dụng Streamlit (dùng để trình bày), dự án có một web Django dùng chung mô hình `models/learnt.pkl`, gồm trang định giá và API JSON. Web không dùng cơ sở dữ liệu.

```bash
python web/manage.py runserver
```

Mở `http://127.0.0.1:8000`. Trong Kiro/VS Code có thể chọn cấu hình **"Mở web Django"** ở tab Run and Debug. Streamlit và Django chạy độc lập, có thể mở cùng lúc ở hai Terminal. Dừng web bằng `Ctrl + C`.

| Địa chỉ | Nội dung |
|---|---|
| `/` | Form định giá: giá dự báo, khoảng dự báo 95%, SHAP, dự báo của 5 mô hình con, tiện ích xung quanh, căn tương tự. Có thể truyền thông số trên đường dẫn để chia sẻ kết quả, ví dụ `/?city=Hà Nội&district=Cầu Giấy&house_type=Nhà mặt tiền&area_m2=60&road_width_m=6&floors=5` |
| `/mo-hinh/` | Kết quả so sánh mô hình và trọng số mô hình meta |
| `/api/` | Tài liệu API |
| `GET /api/districts/?city=Hà Nội` | Danh sách quận/huyện của một thành phố |
| `POST /api/predict/` | Dự báo giá từ JSON; thêm `?explain=0` để bỏ phần SHAP |

Ví dụ gọi API:

```bash
curl -X POST http://127.0.0.1:8000/api/predict/ -H "Content-Type: application/json" \
  -d '{"city": "Hà Nội", "district": "Cầu Giấy", "house_type": "Nhà trong hẻm", "area_m2": 50, "road_width_m": 3, "floors": 5}'
```

Kiểm thử: `python web/manage.py test valuation` (10 kiểm thử: trang, kiểm tra dữ liệu nhập, API, dự báo khớp mô hình, SHAP cộng đúng bằng dự báo).

Khi triển khai thật, đặt `DJANGO_DEBUG=0`, `DJANGO_SECRET_KEY` và `DJANGO_ALLOWED_HOSTS`; khóa mặc định chỉ dùng khi phát triển.

**Triển khai lên Vercel:** import repo trên Vercel, giữ Root Directory là thư mục gốc. Vercel tự nhận Django qua `web/manage.py`, cài thư viện từ `requirements.txt` và dùng Python theo `.python-version` (3.14). Ở **Settings → Environment Variables** đặt `DJANGO_DEBUG=0` và `DJANGO_SECRET_KEY` (chuỗi ngẫu nhiên dài). Tên miền `*.vercel.app` được cho phép sẵn; tên miền riêng thì thêm vào `DJANGO_ALLOWED_HOSTS`. Vercel chỉ chạy được web Django; ứng dụng Streamlit vẫn triển khai trên Streamlit Community Cloud.

---

## 6. Xử lý lỗi thường gặp

| Hiện tượng | Nguyên nhân và cách xử lý |
|---|---|
| Ứng dụng báo *"Thiếu các tệp kết quả"* | Chưa chạy notebook, hoặc notebook là bản cũ chưa tạo `reports/artifacts.pkl`. Thực hiện mục 3 rồi mở lại ứng dụng. |
| `Library not loaded: @rpath/libomp.dylib` (macOS) | Thiếu OpenMP cho LightGBM. Chạy `brew install libomp`. |
| `ModuleNotFoundError` | Môi trường ảo chưa được kích hoạt, hoặc notebook dùng sai kernel. Kích hoạt `.venv` (mục 2, bước 2) và chọn kernel `.venv` trong notebook. |
| `command not found: streamlit`, `jupyter` hoặc `ModuleNotFoundError: django` | Môi trường ảo chưa được kích hoạt. Hoặc chạy trực tiếp `.venv/bin/streamlit run app/streamlit_app.py`. |
| Lỗi khi nạp `learnt.pkl` (`ModuleNotFoundError`, `AttributeError` trong sklearn) | Phiên bản thư viện khác với lúc huấn luyện. Cài đúng `requirements-dev.txt` trên Python 3.14, hoặc chạy lại notebook để huấn luyện lại bằng thư viện hiện có. |
| Build trên Streamlit Cloud rất lâu, log có dòng `Building numpy`, `Building scipy`... | App được tạo với bản Python khác 3.14. Xóa app và tạo lại, giữ Python 3.14 ở Advanced settings. |
| Notebook báo lỗi ở một ô giữa chừng | Các ô phải chạy tuần tự. Chọn *Restart Kernel* rồi *Run All*. |
| `Address already in use` / cổng 8501 hoặc 8000 đang bận | Ứng dụng đã chạy ở Terminal khác. Mở thẳng địa chỉ trên trình duyệt, hoặc dừng tiến trình cũ, hoặc chạy cổng khác: `streamlit run app/streamlit_app.py --server.port 8502`, `python web/manage.py runserver 8001`. |
| Trang Streamlit trắng hoặc chỉ hiện khung xám | Trang đang tính toán lần đầu (nạp mô hình, SHAP). Chờ vài giây hoặc tải lại trang. |

---

## 7. Cấu trúc thư mục

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
├── .streamlit/
│   └── config.toml                     # Màu và phông chữ của Streamlit
├── .vscode/                            # Cấu hình chạy trong Kiro / VS Code
├── app/
│   ├── streamlit_app.py                # Tệp khởi động: điều hướng, cấu hình phân tích
│   ├── core/                           # Hàm dùng chung: data.py (dữ liệu, cấu hình), model.py (dự báo, SHAP), ui.py (giao diện)
│   └── views/                          # 12 trang của ứng dụng
├── web/                                # Web Django: trang định giá và API
│   ├── manage.py
│   ├── estate_web/                     # Cấu hình Django
│   └── valuation/                      # Dịch vụ dự báo, form, giao diện, API, kiểm thử
├── reports/                            # Do notebook tạo ra
├── requirements.txt                    # Thư viện cho web Django (Vercel)
├── requirements-dev.txt                # Toàn bộ thư viện để chạy dự án trên máy
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
| 8 | Lưu kết quả trung gian cho ứng dụng (`reports/artifacts.pkl`) |

**Tệp được Git bỏ qua** (`.gitignore`): môi trường ảo, cache, `plan.md`, và các tệp do notebook sinh ra, trừ 5 tệp ứng dụng Streamlit cần (`models/learnt.pkl`, `data/processed/clean.csv`, `data/processed/features.csv`, `reports/artifacts.pkl`, `reports/cleaning_log.csv`).

---

## 8. Dữ liệu

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
