#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIIRS cloudiness + haze detector (float 0..1)
- Скачивает VNP02IMG (изображения) и VNP03IMG (гео) через earthaccess
- Автодетект групп/имён переменных в VNP02IMG
- День (I01,I02,I03,I05): мягкая cloudiness из NDSI / I2 / BT(I05)
- Ночь (I04,I05): мягкая cloudiness из BT(I04/I05)
- Haze = «сомнительные» пиксели: cloudiness в [HAZE_LOWER, BIN_THR)
- Сохраняет NetCDF: cloudiness, cloud_mask_bin, haze_mask, class_id, NDSI(день)
"""

import os
import re
import numpy as np
import xarray as xr
import earthaccess
from netCDF4 import Dataset

# ----------------- НАСТРОЙКИ -----------------
DATE_START = "2025-05-09"
DATE_END   = "2025-05-10"

# Хасанский район (lon_min, lat_min, lon_max, lat_max)
BBOX = (130.2, 42.2, 131.6, 43.2)

OUT_DIR = "./viirs_l1"   # одна папка для .nc результатов и скачанных файлов
SAVE_PNG = False         # PNG отключены

# Пороги дня (рефл./индексы/BT) и «мягкости» (делители для сигмоид)
NDSI_THR = 0.40
REFL_THR = 0.15
BT_THR   = 295.0
K_NDSI   = 0.10
K_REFL   = 0.05
K_BT     = 3.0

# Ночь (BT) и «мягкости»
THR_I4 = 285.0
THR_I5 = 275.0
K_BT4  = 3.0
K_BT5  = 3.0

# Бинарный порог облачности (для маски и класса)
BIN_THR    = 0.50
# Диапазон «дымки/сомнительных» пикселей
HAZE_LOWER = 0.35        # можно подбирать
# ---------------------------------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def login_earthdata():
    # Требуется ~/.netrc с учеткой Earthdata:
    # machine urs.earthdata.nasa.gov
    # login <user>
    # password <pass>
    earthaccess.login(strategy="netrc")

def download_one(short_name: str, temporal, bbox):
    print(f"\nПоиск {short_name} с {temporal[0]} по {temporal[1]} в BBOX {bbox} ...")
    matches = earthaccess.search_data(
        short_name=short_name,
        temporal=temporal,
        bounding_box=bbox,
        cloud_hosted=True,
    )
    if not matches:
        print(f"Ничего не найдено для {short_name}.")
        return []
    ensure_dir(OUT_DIR)
    print(f"Найдено: {len(matches)} файлов. Скачивание в {OUT_DIR} ...")
    return earthaccess.download(matches, OUT_DIR)

def pair_img_geo(img_files, geo_files):
    """Сопоставляем VNP02IMG и VNP03IMG по ключу AYYYYDDD.HHMM."""
    def key_from_name(p: str):
        m = re.search(r"A(\d{7})\.(\d{4})", os.path.basename(p))
        return f"{m.group(1)}.{m.group(2)}" if m else None

    gmap = {key_from_name(g): g for g in geo_files}
    pairs = []
    for f in img_files:
        k = key_from_name(f)
        if k and k in gmap:
            pairs.append((f, gmap[k]))
    return pairs

# ---------- Чтение VNP02IMG с автопоиском группы/имён ----------
def _find_var_regex(ds: xr.Dataset, patterns):
    for name in ds.data_vars:
        for pat in patterns:
            if re.fullmatch(pat, name):
                return ds[name]
    return None

def open_vnp02img_vars(path_vnp02img: str, debug=True):
    # Кандидаты групп
    candidate_groups = [None, "All_Data", "observation_data", "Data_Products", "scan_line_attributes"]
    with Dataset(path_vnp02img, "r") as nc:
        for g in nc.groups.keys():
            if g not in candidate_groups:
                candidate_groups.append(g)

    patt = {
        "I01": [r"I0?1(_Reflectance)?", r"I01_Reflectance", r"Radiance_I01", r"Reflectance_I01", r"I01"],
        "I02": [r"I0?2(_Reflectance)?", r"I02_Reflectance", r"Radiance_I02", r"Reflectance_I02", r"I02"],
        "I03": [r"I0?3(_Reflectance)?", r"I03_Reflectance", r"Radiance_I03", r"Reflectance_I03", r"I03"],
        "I04": [r"I0?4(_BrightnessTemperature)?", r"I04_BrightnessTemperature", r"BT_I04", r"BrightnessTemperature_I04", r"I04", r"Radiance_I04"],
        "I05": [r"I0?5(_BrightnessTemperature)?", r"I05_BrightnessTemperature", r"BT_I05", r"BrightnessTemperature_I05", r"I05", r"Radiance_I05"],
    }

    last_err = None
    for grp in candidate_groups:
        try:
            ds = xr.open_dataset(path_vnp02img, group=grp) if grp else xr.open_dataset(path_vnp02img)
            if debug:
                print(f"\nПробуем группу: {grp if grp else '<root>'}")
                print("Переменные (первые 20):", list(ds.data_vars)[:20])
            found = {}
            for k, pats in patt.items():
                v = _find_var_regex(ds, pats)
                if v is not None:
                    v.load()
                    found[k] = v.values
            if found:
                if debug:
                    print("Нашли каналы: ", sorted(found.keys()))
                return found
        except Exception as e:
            last_err = e
            if debug:
                print(f"  → не подошла группа {grp}: {e}")
    raise RuntimeError(f"Не удалось прочесть каналы из {path_vnp02img}. Последняя ошибка: {last_err}")

# ------------- Вспомогательные функции -------------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def radiance_to_bt_wien(L_wm2_sr_um: np.ndarray, lambda_um: float) -> np.ndarray:
    """
    Конверсия радианса (W m^-2 sr^-1 µm^-1) в яркостную температуру (K) по уравнению Планка (Вин).
    Защита от нулевых/отрицательных L.
    """
    L = np.asarray(L_wm2_sr_um, dtype=np.float64)
    L = np.maximum(L, 1e-12)  # чтоб не было деления на 0 / log(<=1)
    lam = float(lambda_um)

    # Константы Планка в «наборе» для единиц W m^-2 sr^-1 µm^-1
    c1 = 1.191042e8    # W µm^4 m^-2 sr^-1
    c2 = 1.4387752e4   # µm·K

    inside = 1.0 + (c1 / ((lam ** 5) * L))
    # защита от численной нестабильности
    inside = np.maximum(inside, 1.0 + 1e-12)
    T = c2 / (lam * np.log(inside))
    return T.astype(np.float32)

# ------------- Мягкая облачность + дымка -------------
def cloudiness_day(ri1, ri2, ri3, bi5,
                   ndsi_thr=NDSI_THR, refl_thr=REFL_THR, bt_thr=BT_THR,
                   k_ndsi=K_NDSI, k_refl=K_REFL, k_bt=K_BT):
    with np.errstate(divide='ignore', invalid='ignore'):
        ndsi = (ri1 - ri3) / (ri1 + ri3)

    c_ndsi = _sigmoid((ndsi - ndsi_thr) / k_ndsi)   # выше порога → облачнее
    c_refl = _sigmoid((ri2  - refl_thr) / k_refl)   # выше порога → облачнее
    c_bt   = _sigmoid((bt_thr - bi5)   / k_bt)      # ниже порога → облачнее

    cloudiness = 1.0 - (1.0 - c_ndsi) * (1.0 - c_refl) * (1.0 - c_bt)

    nan_mask = np.isnan(ri1) | np.isnan(ri2) | np.isnan(ri3) | np.isnan(bi5)
    cloudiness = np.where(nan_mask, 0.0, cloudiness).astype(np.float32)

    mask_bin = cloudiness >= BIN_THR
    cloudy   = int(np.count_nonzero(mask_bin))
    total    = int(mask_bin.size)
    pct      = (cloudy / total) * 100.0 if total > 0 else 0.0

    return cloudiness, mask_bin, pct, ndsi

def cloudiness_night(bi4_bt, bi5_bt,
                     thr4=THR_I4, thr5=THR_I5, k4=K_BT4, k5=K_BT5):
    c4 = _sigmoid((thr4 - bi4_bt) / k4)  # ниже thr → облачнее
    c5 = _sigmoid((thr5 - bi5_bt) / k5)

    cloudiness = 1.0 - (1.0 - c4) * (1.0 - c5)
    nan_mask = np.isnan(bi4_bt) | np.isnan(bi5_bt)
    cloudiness = np.where(nan_mask, 0.0, cloudiness).astype(np.float32)

    mask_bin = cloudiness >= BIN_THR
    cloudy   = int(np.count_nonzero(mask_bin))
    total    = int(mask_bin.size)
    pct      = (cloudy / total) * 100.0 if total > 0 else 0.0

    return cloudiness, mask_bin, pct

# ------------- Сохранение результата -------------
def save_float_mask_nc(path_nc, cloudiness, mask_bin=None, haze_mask=None, class_id=None, ndsi=None):
    """
    NetCDF:
      cloudiness (float32 0..1)
      cloud_mask_bin (uint8)
      haze_mask (uint8)
      class_id (uint8): 0=clear, 1=haze(prob), 2=cloud
      NDSI (float32) при дне
    """
    da = {"cloudiness": (("y","x"), cloudiness.astype("float32"))}
    if mask_bin is not None:
        da["cloud_mask_bin"] = (("y","x"), mask_bin.astype("uint8"))
    if haze_mask is not None:
        da["haze_mask"] = (("y","x"), haze_mask.astype("uint8"))
    if class_id is not None:
        da["class_id"] = (("y","x"), class_id.astype("uint8"))
    if ndsi is not None:
        da["NDSI"] = (("y","x"), ndsi.astype("float32"))
    xr.Dataset(da).to_netcdf(path_nc)
    print(f"Float-маска сохранена: {path_nc}")

# ------------- Основной процесс по паре -------------
def process_pair(img_path: str, geo_path: str):
    print(f"\nОбработка сцены:\n  IMG: {img_path}\n  GEO: {geo_path}")

    chans = open_vnp02img_vars(img_path, debug=True)
    have = set(chans.keys())

    ndsi = None
    # День
    if {"I01","I02","I03","I05"}.issubset(have):
        ri1, ri2, ri3 = chans["I01"], chans["I02"], chans["I03"]

        # I05 может быть BT или радиансом. Если максимум < 60 → точно не BT (радианс),
        # конвертируем: возьмем длину волны I05 ≈ 11.45 µm
        bi5 = chans["I05"].astype(np.float32)
        if np.nanmax(bi5) < 60.0:  # вероятно радиансы
            bi5 = radiance_to_bt_wien(bi5, 11.45)

        cloudiness, mask_bin, pct_cloud, ndsi = cloudiness_day(
            ri1, ri2, ri3, bi5,
            ndsi_thr=NDSI_THR, refl_thr=REFL_THR, bt_thr=BT_THR,
            k_ndsi=K_NDSI, k_refl=K_REFL, k_bt=K_BT
        )
        alg = "DAY (soft: NDSI/I2/BT)"
        bi4_bt = None

    # Ночь
    elif {"I04","I05"}.issubset(have):
        # VIIRS IMG I04≈3.74 µm, I05≈11.45 µm — часто как радиансы → конвертируем
        bi4 = chans["I04"].astype(np.float32)
        bi5 = chans["I05"].astype(np.float32)
        bi4_bt = bi4 if np.nanmax(bi4) > 60 else radiance_to_bt_wien(bi4, 3.74)
        bi5_bt = bi5 if np.nanmax(bi5) > 60 else radiance_to_bt_wien(bi5, 11.45)

        cloudiness, mask_bin, pct_cloud = cloudiness_night(
            bi4_bt, bi5_bt, thr4=THR_I4, thr5=THR_I5, k4=K_BT4, k5=K_BT5
        )
        alg = "NIGHT (soft: BT I4/I5)"
        ri1 = ri2 = ri3 = None

    else:
        print("Пропускаем: нет набора каналов ни для дня, ни для ночи.")
        return

    # --- Haze (сомнительные): [HAZE_LOWER, BIN_THR)
    haze_mask = (cloudiness >= HAZE_LOWER) & (cloudiness < BIN_THR)
    haze_pct  = (np.count_nonzero(haze_mask) / haze_mask.size * 100.0) if haze_mask.size else 0.0

    # Классы: 0=clear, 1=haze(prob), 2=cloud
    class_id = np.zeros_like(mask_bin, dtype=np.uint8)
    class_id[haze_mask] = 1
    class_id[mask_bin]  = 2

    print(f"{alg} → Доля облаков (thr={BIN_THR:.2f}): {pct_cloud:.2f}% | Haze: {haze_pct:.2f}%")

    # --- Сохранение NetCDF
    out_nc = os.path.join(OUT_DIR, os.path.basename(img_path).replace(".nc", "_cloudiness.nc"))
    save_float_mask_nc(out_nc, cloudiness, mask_bin, haze_mask, class_id, ndsi)

def main():
    login_earthdata()
    img_files = download_one("VNP02IMG", (DATE_START, DATE_END), BBOX)
    geo_files = download_one("VNP03IMG", (DATE_START, DATE_END), BBOX)
    if not img_files or not geo_files:
        print("Не хватает IMG или GEO файлов — проверь дату/регион/доступ.")
        return
    pairs = pair_img_geo(img_files, geo_files)
    if not pairs:
        print("Не удалось сопоставить пары VNP02IMG↔VNP03IMG по времени.")
        return
    for img_path, geo_path in pairs:
        process_pair(img_path, geo_path)

if __name__ == "__main__":
    main()
