# %% Thư viện và đường dẫn
"""Làm sạch dữ liệu nhà đất (Bước 0).

Đầu vào : data/raw/vietnam_housing_dirty.csv
Đầu ra  : data/processed/clean.csv
          reports/cleaning_log.csv  (số dòng sau từng bước)

Chạy: python -m src.preprocessing
"""
from __future__ import annotations

import datetime as dt
import difflib
import html
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "vietnam_housing_dirty.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "clean.csv"
LOG_PATH = ROOT / "reports" / "cleaning_log.csv"

COLUMNS = [
    "order", "url", "title", "address", "posted_date", "city", "district", "house_type",
    "area_m2", "width_m", "length_m", "road_width_m", "floors", "bedrooms",
    "has_dining_room", "has_kitchen", "has_terrace", "has_parking",
    "price_billion_vnd", "contact", "scraped_at",
]
BOOL_COLS = ["has_dining_room", "has_kitchen", "has_terrace", "has_parking"]

HN, HCM = "Hà Nội", "Hồ Chí Minh"
DISTRICTS = {
    HN: ["Ba Đình", "Hoàn Kiếm", "Hai Bà Trưng", "Đống Đa", "Tây Hồ", "Cầu Giấy", "Thanh Xuân",
         "Hoàng Mai", "Long Biên", "Bắc Từ Liêm", "Nam Từ Liêm", "Hà Đông", "Sơn Tây", "Thanh Trì",
         "Gia Lâm", "Đông Anh", "Sóc Sơn", "Hoài Đức", "Thanh Oai", "Chương Mỹ", "Mê Linh",
         "Đan Phượng", "Thường Tín", "Quốc Oai", "Thạch Thất", "Phúc Thọ", "Ba Vì", "Mỹ Đức",
         "Ứng Hòa", "Phú Xuyên"],
    HCM: [f"Quận {i}" for i in range(1, 13)] + [
        "Bình Thạnh", "Gò Vấp", "Phú Nhuận", "Tân Bình", "Tân Phú", "Bình Tân", "Thủ Đức",
        "Nhà Bè", "Hóc Môn", "Củ Chi", "Bình Chánh", "Cần Giờ"],
}
DISTRICT_ALIASES = {"hbt": "Hai Bà Trưng"}

MISSING_TOKENS = {"", "nan", "null", "none", "n/a", "na", "-", "--", "?", "không rõ", "khong ro"}


# %% Hàm chuẩn hóa chữ
# ---------------------------------------------------------------------------
# Hàm chuẩn hóa chữ
# ---------------------------------------------------------------------------
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]")


def fix_mojibake(s: str) -> str:
    """Sửa chữ UTF-8 bị đọc nhầm thành Latin-1, ví dụ 'Cáº§u Giáº¥y' -> 'Cầu Giấy'."""
    if re.search("[ÃÂÄÆáº»]", s):
        try:
            return s.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return s
    return s


def clean_text(x) -> str | float:
    """Chuẩn hóa một ô chữ: sửa font, NFC, bỏ HTML/emoji/khoảng trắng thừa."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = fix_mojibake(str(x))
    s = unicodedata.normalize("NFKC", s)  # NFC + đổi số full-width về số thường
    s = html.unescape(s).replace("\xa0", " ")
    s = re.sub(r"<[^>]+>", " ", s)
    s = EMOJI_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return np.nan if s.lower() in MISSING_TOKENS else s


def ascii_key(s: str) -> str:
    """Bỏ dấu, chữ thường, chỉ giữ chữ và số: 'Đống Đa' -> 'dongda'."""
    s = str(s).replace("Đ", "D").replace("đ", "d")
    s = unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", s)


# %% Thành phố, quận, loại nhà, có/không
# ---------------------------------------------------------------------------
# Thành phố, quận, loại nhà, có/không
# ---------------------------------------------------------------------------
CITY_KEYS = {
    "hanoi": HN, "hn": HN, "tphanoi": HN,
    "hochiminh": HCM, "hcm": HCM, "tphcm": HCM, "tphochiminh": HCM, "saigon": HCM,
    "thanhphohochiminh": HCM,
}


def parse_city(x) -> str | float:
    if pd.isna(x):
        return np.nan
    return CITY_KEYS.get(ascii_key(x), np.nan)


_DIST_INDEX = {}
for _city, _names in DISTRICTS.items():
    for _name in _names:
        _key = ascii_key(_name.replace("Quận ", ""))
        _DIST_INDEX[_key] = (_city, _name)


def parse_district(x, city=None) -> tuple[str, str] | None:
    """Trả về (thành phố, tên quận chuẩn) hoặc None nếu không nhận ra."""
    if pd.isna(x):
        return None
    s = re.sub(r"^\s*(quận|huyện|thị xã|thành phố|tp\.?|q\.?)\s*", "", str(x), flags=re.IGNORECASE)
    key = ascii_key(s)
    if not key:
        return None
    if key in DISTRICT_ALIASES:
        key = ascii_key(DISTRICT_ALIASES[key])
    if key in _DIST_INDEX:
        return _DIST_INDEX[key]
    if key.isdigit():
        return None
    pool = [k for k, (c, _) in _DIST_INDEX.items() if city is None or c == city]
    match = difflib.get_close_matches(key, pool, n=1, cutoff=0.8)
    return _DIST_INDEX[match[0]] if match else None


def parse_address(x) -> tuple[str | float, str | float]:
    """Tách (thành phố, quận) từ chuỗi địa chỉ."""
    city = district = np.nan
    if pd.isna(x):
        return city, district
    for part in re.split(r"[,\-]", str(x)):
        part = part.strip()
        if not part or part.startswith("Đường") or ascii_key(part) == "vietnam":
            continue
        c = parse_city(part)
        if not pd.isna(c):
            city = c
            continue
        d = parse_district(part)
        if d:
            district = d[1]
            if pd.isna(city):
                city = d[0]
    return city, district


def district_from_title(t, city) -> str | None:
    """Tìm tên quận trong tiêu đề, ví dụ 'Bán nhà Quận 10 37m2...' -> 'Quận 10'."""
    if pd.isna(t):
        return None
    m = re.search(r"quận (\d{1,2})\b", str(t), flags=re.IGNORECASE)
    if m and city in (HCM, None) and 1 <= int(m.group(1)) <= 12:
        return f"Quận {int(m.group(1))}"
    key = ascii_key(t)
    for c, names in DISTRICTS.items():
        if city is not None and not pd.isna(city) and c != city:
            continue
        for name in sorted(names, key=len, reverse=True):
            if not name.startswith("Quận ") and ascii_key(name) in key:
                return name
    return None


def parse_house_type(x) -> str | float:
    if pd.isna(x):
        return np.nan
    k = ascii_key(x)
    if "mat" in k:
        return "Nhà mặt tiền"
    if "hem" in k or "ngo" in k:
        return "Nhà trong hẻm"
    return np.nan


TRUE_TOKENS = {"true", "1", "y", "yes", "x", "co"}
FALSE_TOKENS = {"false", "0", "n", "no", "khong", "ko", "k"}
BOOL_PHRASES = {
    "has_kitchen": {"cobep": True, "beprieng": True},
    "has_terrace": {"cosanthuong": True, "sanphoi": True},
    "has_parking": {"otodocua": True, "xehoivaonha": True, "khongcochodau": False},
}


def parse_bool(x, col: str):
    if pd.isna(x):
        return np.nan
    k = ascii_key(x)
    if k in TRUE_TOKENS:
        return True
    if k in FALSE_TOKENS:
        return False
    return BOOL_PHRASES.get(col, {}).get(k, np.nan)


# %% Chuyển chữ thành số
# ---------------------------------------------------------------------------
# Chuyển chữ thành số
# ---------------------------------------------------------------------------
NUM = r"\d+(?:[.,]\d+)?"
VI_NUMBERS = {"mot": 1, "hai": 2, "ba": 3, "bon": 4, "tu": 4, "nam": 5, "sau": 6, "bay": 7,
              "tam": 8, "chin": 9, "muoi": 10}
EXCEL_DATE_RE = re.compile(r"^(\d{1,2})-thg(\d{1,2})$")  # '5-Thg4' là 4.5 bị Excel đổi


def to_float(s: str) -> float:
    """'4,5' -> 4.5 ; '4.5' -> 4.5"""
    return float(s.replace(",", "."))


def excel_date_to_number(s: str) -> float | None:
    m = EXCEL_DATE_RE.match(s.lower().replace(" ", ""))
    return float(f"{m.group(2)}.{m.group(1)}") if m else None


def parse_price(x) -> tuple[float, str]:
    """Trả về (giá, loại). Loại 'per_m2' nghĩa là giá theo triệu/m², cần nhân diện tích."""
    if pd.isna(x):
        return np.nan, ""
    s = str(x).lower()
    s = re.sub(r"giá\s*:|khoảng|vnđ|vnd|\(.*?\)", " ", s).strip()
    if any(w in s for w in ("thỏa thuận", "thoả thuận", "liên hệ")):
        return np.nan, ""
    ex = excel_date_to_number(s)
    if ex is not None:
        return ex, ""
    if s.startswith("$"):
        return float(re.sub(r"[^\d.]", "", s)) * 25_000 / 1e9, "usd"
    if "/m" in s:
        m = re.search(NUM, s)
        return (to_float(m.group()), "per_m2") if m else (np.nan, "")
    m = re.match(rf"^({NUM})\s*tỷ\s*(\d+)\s*triệu$", s)          # 3 tỷ 400 triệu
    if m:
        return to_float(m.group(1)) + int(m.group(2)) / 1000, ""
    m = re.match(r"^(\d+)\s*ty\s*(\d)$", s)                        # 5ty2
    if m:
        return int(m.group(1)) + int(m.group(2)) / 10, ""
    m = re.match(rf"^({NUM})\s*-\s*({NUM})\s*(tỷ|ty)?$", s)        # 4.5 - 5.0 tỷ
    if m:
        return (to_float(m.group(1)) + to_float(m.group(2))) / 2, ""
    m = re.match(rf"^({NUM})\s*triệu$", s)
    if m:
        return to_float(m.group(1)) / 1000, ""
    if re.fullmatch(r"\d{1,3}([.,]\d{3}){2,}\s*đ?", s):            # 7.500.000.000 / 3,900,000,000 đ
        return int(re.sub(r"\D", "", s)) / 1e9, ""
    m = re.match(r"^(\d+\.\d+)\.000$", s)                          # 4.35.000
    if m:
        return float(m.group(1)), ""
    m = re.match(rf"^({NUM}(?:e[+-]?\d+)?)\s*(tỷ|ty)?$", s)
    if m:
        v = float(m.group(1).replace(",", ".")) if "e" not in m.group(1) else float(m.group(1))
        return (v / 1e9 if v > 1e6 else v), ""
    return np.nan, ""


def parse_area(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).lower().replace("dt", "").replace("~", "").strip()
    s = re.sub(r"\(.*?\)", "", s).strip()                          # 60m2 (sổ 57m2)
    m = re.match(rf"^({NUM})\s*x\s*({NUM})$", s)                   # 4.0x12.0
    if m:
        return to_float(m.group(1)) * to_float(m.group(2))
    m = re.match(rf"^({NUM})\s*ha$", s)
    if m:
        return to_float(m.group(1)) * 10_000
    m = re.match(rf"^({NUM})\s*-\s*({NUM})", s)                    # 50-55 m2
    if m:
        return (to_float(m.group(1)) + to_float(m.group(2))) / 2
    m = re.match(rf"^(-?{NUM})", s)
    return abs(to_float(m.group(1))) if m else np.nan


def parse_length(x) -> float:
    """Chiều ngang/dài/độ rộng đường (m). Số >= 100 coi là cm."""
    if pd.isna(x):
        return np.nan
    s = str(x).lower()
    ex = excel_date_to_number(s)
    if ex is not None:
        return ex
    if "xe hơi" in s:          # hẻm xe hơi: thường rộng từ khoảng 4 m
        return 4.0
    if "ba gác" in s:          # hẻm ba gác: khoảng 2,5 m
        return 2.5
    k = ascii_key(s)
    if k in VI_NUMBERS:
        return float(VI_NUMBERS[k])
    m = re.match(rf"^(-?{NUM})", s)                                # 12.0/13.0 lấy số đầu
    if not m:
        return np.nan
    v = to_float(m.group(1))
    return v / 100 if v >= 100 else v


def parse_floors(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).lower()
    m = re.match(r"^1\s*trệt\s*(\d+)\s*lầu", s)                     # 1 trệt 3 lầu = 4 tầng
    if m:
        return 1 + int(m.group(1))
    first = re.split(r"[\s+]", s)[0]
    if ascii_key(first) in VI_NUMBERS:
        return float(VI_NUMBERS[ascii_key(first)])
    m = re.match(rf"^(-?{NUM})", s)                                # 4 tầng, 4 lầu, 3T, 5+lửng
    return to_float(m.group(1)) if m else np.nan


def parse_bedrooms(x) -> float:
    if pd.isna(x) or str(x).lower() in ("true", "false"):
        return np.nan
    m = re.match(r"^(-?\d+)", str(x))                              # 4PN, 4-5, 4+, 4.0.0
    return float(m.group(1)) if m else np.nan


# %% Ngày tháng
# ---------------------------------------------------------------------------
# Ngày tháng
# ---------------------------------------------------------------------------
RELATIVE_DAYS = {"hôm nay": 0, "vừa xong": 0, "hôm qua": 1}


def parse_datetime(x) -> pd.Timestamp:
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%d/%m/%Y %H:%M"):
        try:
            return pd.Timestamp(dt.datetime.strptime(s, fmt))
        except ValueError:
            pass
    return pd.NaT


def parse_posted_date(x, scraped: pd.Timestamp) -> pd.Timestamp:
    """Ngày đăng. Ngày kiểu dd/mm được ưu tiên; chỉ đọc mm/dd khi dd/mm không hợp lệ."""
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip().lower().replace("ngày đăng:", "").strip()
    base = scraped.normalize() if not pd.isna(scraped) else pd.NaT
    if s in RELATIVE_DAYS:
        return base - pd.Timedelta(days=RELATIVE_DAYS[s])
    m = re.match(r"^(\d+)\s*(ngày|tuần) trước$", s)
    if m:
        return base - pd.Timedelta(days=int(m.group(1)) * (7 if m.group(2) == "tuần" else 1))
    if re.fullmatch(r"\d{5}", s):                                  # số ngày kiểu Excel
        return pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(s))
    m = re.match(r"^thg (\d+) (\d+), (\d{4})$", s)
    if m:
        s = f"{m.group(2)}/{m.group(1)}/{m.group(3)}"
    s = s.split(" ")[0]
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%m/%d/%Y"):
        try:
            d = pd.Timestamp(dt.datetime.strptime(s, fmt))
        except ValueError:
            continue
        if d.year < 2015 or (not pd.isna(scraped) and d > scraped):
            return pd.NaT                                          # 01/01/1970 hoặc ngày tương lai
        return d
    return pd.NaT


# %% Đọc thông tin trong tiêu đề (dùng để đối chiếu)
# ---------------------------------------------------------------------------
# Đọc thông tin trong tiêu đề (dùng để đối chiếu)
# ---------------------------------------------------------------------------
def title_area(t) -> float:
    if pd.isna(t):
        return np.nan
    m = re.search(rf"({NUM})\s*m(2|²)?\b", str(t), flags=re.IGNORECASE)
    return to_float(m.group(1)) if m else np.nan


def title_price(t) -> float:
    if pd.isna(t):
        return np.nan
    m = re.search(rf"(?:giá|chỉ|-)\s*({NUM})\s*(tỷ|ty)", str(t), flags=re.IGNORECASE)
    return to_float(m.group(1)) if m else np.nan


# %% Các bước làm sạch
# ---------------------------------------------------------------------------
# Các bước làm sạch
# ---------------------------------------------------------------------------
def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def fix_file_level(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Bỏ tiêu đề lặp, dòng rỗng; sửa dòng bị lệch cột."""
    df = df[df["order"] != "order"]
    log.append(("Bỏ dòng tiêu đề lặp", len(df)))
    df = df[(df != "").any(axis=1)].copy()
    log.append(("Bỏ dòng rỗng", len(df)))

    # Dòng thiếu một ô ở cột title/address/posted_date: scraped_at bị trống và
    # contact chứa thời gian. Dời các cột từ city trở đi sang phải một ô.
    shifted = (df["scraped_at"] == "") & df["contact"].str.match(r"^\d{4}-\d{2}-\d{2}T|^\d{2}/\d{2}/\d{4} ")
    i_city = COLUMNS.index("city")
    for idx in df.index[shifted]:
        vals = df.loc[idx, COLUMNS].tolist()
        df.loc[idx, COLUMNS[i_city:]] = vals[i_city - 1:-1]
        df.loc[idx, ["title", "address", "posted_date"]] = ""      # không biết ô nào bị mất
    log.append((f"Sửa {int(shifted.sum())} dòng lệch cột", len(df)))
    return df


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = df[c].map(clean_text)
    return df


def fix_location(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa thành phố/quận, sửa đổi chỗ, đối chiếu với địa chỉ."""
    city = df["city"].map(parse_city)
    district_raw = df["district"].copy()

    # Thành phố và quận bị đổi chỗ cho nhau
    swap = city.isna() & df["district"].map(parse_city).notna()
    city[swap] = df.loc[swap, "district"].map(parse_city)
    district_raw[swap] = df.loc[swap, "city"]

    addr = df["address"].map(parse_address)
    addr_city = addr.map(lambda t: t[0])
    addr_dist = addr.map(lambda t: t[1])
    city = city.fillna(addr_city)

    districts, flags = [], []
    for raw, c, ad, title in zip(district_raw, city, addr_dist, df["title"]):
        d = parse_district(raw, c if not pd.isna(c) else None)
        # số trống ('1', '12'...) chỉ có ở TP.HCM
        if d is None and not pd.isna(raw) and re.sub(r"\D", "", str(raw)) and c == HCM:
            n = int(re.sub(r"\D", "", str(raw)))
            d = (HCM, f"Quận {n}") if 1 <= n <= 12 else None
        if d and (pd.isna(c) or d[0] == c):
            districts.append(d[1]); flags.append("")
        elif not pd.isna(ad):
            districts.append(ad); flags.append("lay_tu_dia_chi")
        elif (td := district_from_title(title, c)) is not None:
            districts.append(td); flags.append("lay_tu_tieu_de")
        else:
            districts.append(np.nan); flags.append("khong_xac_dinh" if d is None else "quan_sai_tp")
    df["city"] = city
    df["district"] = districts
    df["location_fix"] = flags
    return df


def parse_numbers(df: pd.DataFrame) -> pd.DataFrame:
    df["house_type"] = df["house_type"].map(parse_house_type)
    for c in BOOL_COLS:
        df[c] = df[c].map(lambda v, c=c: parse_bool(v, c))

    df["area_m2"] = df["area_m2"].map(parse_area)
    for c in ["width_m", "length_m", "road_width_m"]:
        df[c] = df[c].map(parse_length)
    df["floors"] = df["floors"].map(parse_floors)
    df["bedrooms"] = df["bedrooms"].map(parse_bedrooms)

    price = df["price_billion_vnd"].map(parse_price)
    df["price_billion_vnd"] = price.map(lambda t: t[0])
    per_m2 = price.map(lambda t: t[1] == "per_m2")
    df.loc[per_m2, "price_billion_vnd"] = df.loc[per_m2, "price_billion_vnd"] * df.loc[per_m2, "area_m2"] / 1000
    # Giá >= 100 tỷ gần như chắc chắn là ghi theo đơn vị triệu
    big = df["price_billion_vnd"] >= 100
    df.loc[big, "price_billion_vnd"] = df.loc[big, "price_billion_vnd"] / 1000

    df["scraped_at"] = df["scraped_at"].map(parse_datetime)
    df["posted_date"] = [parse_posted_date(p, s) for p, s in zip(df["posted_date"], df["scraped_at"])]
    return df


def fix_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """Giá trị vô lý thành NaN; đối chiếu diện tích/giá với tiêu đề; đảo ngang-dài."""
    valid = {
        "price_billion_vnd": (0.3, 200), "area_m2": (10, 1000), "width_m": (1, 50),
        "length_m": (1, 100), "road_width_m": (0.5, 60), "floors": (1, 15), "bedrooms": (1, 20),
    }
    for c, (lo, hi) in valid.items():
        df.loc[~df[c].between(lo, hi), c] = np.nan
    for c in ["floors", "bedrooms"]:
        df.loc[df[c] % 1 != 0, c] = np.nan

    # Diện tích gõ thừa số 0: so với tiêu đề hoặc ngang x dài
    t_area = df["title"].map(title_area)
    wl = df["width_m"] * df["length_m"]
    ref = t_area.fillna(wl)
    x10 = (df["area_m2"] / ref).between(8, 12)
    df.loc[x10, "area_m2"] = df.loc[x10, "area_m2"] / 10

    # Ngang lớn hơn dài thì đảo lại
    swap = df["width_m"] > df["length_m"]
    df.loc[swap, ["width_m", "length_m"]] = df.loc[swap, ["length_m", "width_m"]].values

    # Giá: đối chiếu với tiêu đề
    t_price = df["title"].map(title_price)
    df["price_from_title"] = df["price_billion_vnd"].isna() & t_price.notna()
    df.loc[df["price_from_title"], "price_billion_vnd"] = t_price[df["price_from_title"]]
    df["title_price_mismatch"] = (t_price.notna() & ~df["price_from_title"]
                                  & ((df["price_billion_vnd"] - t_price).abs() / t_price > 0.03))
    return df


def drop_duplicates(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Bỏ trùng y hệt, trùng url, và tin đăng lại (cùng đặc điểm + liên hệ, giá lệch <= 6%)."""
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "order"]).copy()
    log.append(("Bỏ dòng trùng y hệt", len(df)))

    df["url"] = df["url"].str.replace(r"[?#].*$", "", regex=True)
    df = df[~(df["url"].notna() & df.duplicated("url"))]
    log.append(("Bỏ tin trùng url", len(df)))

    # Tin đăng lại: cùng đặc điểm căn nhà, giá lệch <= 6%, và cùng số điện thoại
    # hoặc cùng tiêu đề (sau khi bỏ các tiền tố như "[ĐĂNG LẠI]").
    key = ["city", "district", "house_type", "area_m2", "width_m", "length_m", "road_width_m",
           "floors", "bedrooms"]
    df = df.sort_values("order", key=lambda s: pd.to_numeric(s, errors="coerce"))
    norm_title = (df["title"].fillna("").str.lower()
                  .str.replace(r"^(\[đăng lại\]|cần bán gấp:|hạ giá!)\s*", "", regex=True)
                  .map(ascii_key))
    phone = df["contact"].where(df["contact"].str.contains(r"\d{3}", na=False))
    keep = pd.Series(True, index=df.index)
    for _, g in df.groupby(key, dropna=False):
        if len(g) < 2:
            continue
        idx = list(g.index)
        for j in range(1, len(idx)):
            for i in range(j):
                a, b = idx[i], idx[j]
                if not keep[a]:
                    continue
                pa, pb = df.at[a, "price_billion_vnd"], df.at[b, "price_billion_vnd"]
                close = pd.isna(pa) or pd.isna(pb) or abs(pb - pa) / pa <= 0.06
                same_phone = not pd.isna(phone[a]) and phone[a] == phone[b]
                same_title = norm_title[a] != "" and norm_title[a] == norm_title[b]
                if close and (same_phone or same_title):
                    keep[b] = False
                    break
    df = df[keep]
    log.append(("Bỏ tin đăng lại", len(df)))
    return df


def finalize(df: pd.DataFrame, log: list) -> pd.DataFrame:
    df = df[df["price_billion_vnd"].notna()].copy()
    log.append(("Bỏ dòng không có giá", len(df)))
    df["order"] = pd.to_numeric(df["order"], errors="coerce").astype("Int64")
    df["price_billion_vnd"] = df["price_billion_vnd"].round(3)
    df["area_m2"] = df["area_m2"].round(1)
    for c in BOOL_COLS:
        df[c] = df[c].astype("boolean")
    for c in ["floors", "bedrooms"]:
        df[c] = df[c].astype("Int64")
    df["posted_date"] = pd.to_datetime(df["posted_date"]).dt.date
    cols = ["order", "url", "title", "posted_date", "city", "district", "house_type",
            "area_m2", "width_m", "length_m", "road_width_m", "floors", "bedrooms", *BOOL_COLS,
            "price_billion_vnd", "price_from_title", "title_price_mismatch", "location_fix"]
    return df[cols].reset_index(drop=True)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    log = [("Dữ liệu thô", len(raw))]
    df = fix_file_level(raw.copy(), log)
    df = normalize_text(df)
    df = fix_location(df)
    df = parse_numbers(df)
    df = fix_invalid_values(df)
    df = drop_duplicates(df, log)
    df = finalize(df, log)
    return df, pd.DataFrame(log, columns=["buoc", "so_dong"])


# %% Chạy cả file và lưu kết quả
def main():
    raw = load_raw()
    df, log = clean(raw)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False, encoding="utf-8")
    log.to_csv(LOG_PATH, index=False, encoding="utf-8")
    print(log.to_string(index=False))
    print(f"\nĐã lưu {len(df)} dòng vào {CLEAN_PATH.relative_to(ROOT)}")
    print("\nGiá trị thiếu còn lại:")
    print(df.isna().sum()[lambda s: s > 0].to_string())


if __name__ == "__main__":
    main()
