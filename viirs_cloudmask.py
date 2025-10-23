#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viirs_cloudmask_demo.py — версия с float-плотностью облачности
- Скачивает VNP02IMG и VNP03IMG в одну папку OUT_DIR
- Находит каналы (автодетект групп/имён)
- Для дня (I01,I02,I03,I05) строит мягкую cloudiness 0..1 из NDSI/I2/BT
- Для ночи (I04,I05) строит мягкую cloudiness 0..1 из BT
- Сохраняет PNG (бинарная по порогу 0.5) и NetCDF с float-маской
"""

import os
import re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import earthaccess
from netCDF4 import Dataset

# ---------- НАСТРОЙКИ ----------
DATE_START = "2025-05-09"
DATE_END   = "2025-05-10"

# Хасанский район (lon_min, lat_min, lon_max, lat_max)
BBOX = (130.2, 42.2, 131.6, 43.2)

# Одна папка для всего (.nc и .png)
OUT_DIR = "./viirs_l1"

# Сохранять PNG?
SAVE_PNG = True

# Пороги (дневные)
NDSI_THR = 0.4
REFL_THR = 0.15
BT_THR   = 295.0

# "мягкость" порогов для сигмоид (чем больше k, тем плавнее)
K_NDSI = 0.10
K_REFL = 0.05
K_BT   = 3.0   # K, дневной BT в I5

# Ночь: пороги BT и мягкость
THR_I4 = 285.0
THR_I5 = 275.0
K_BT4  = 3.0
K_BT5  = 3.0

# Порог для бинарной визуализации из float cloudiness
BIN_THR = 0.5
# --------------------------------


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def login_earthdata():
    # Нужен .netrc с Earthdata (machine urs.earthdata.nasa.gov ...)
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


# ---- Чтение VNP02IMG с автопоиском группы/имён ----
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
        "I04": [r"I0?4(_BrightnessTemperature)?", r"I04_BrightnessTemperature", r"BT_I04", r"BrightnessTemperature_I04", r"I04"],
        "I05": [r"I0?5(_BrightnessTemperature)?", r"I05_BrightnessTemperature", r"BT_I05", r"BrightnessTemperature_I05", r"I05"],
    }

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
                    print("Нашли каналы:", sorted(found.keys()))
                return found
        except Exception as e:
            if debug:
                print(f"  → не подошла группа {grp}: {e}")
    raise RuntimeError(f"Не удалось прочесть каналы из {path_vnp02img}")


# ---------- Мягкие тесты / cloudiness ----------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def cloudiness_day(ri1, ri2, ri3, bi5,
                   ndsi_thr=NDSI_THR, refl_thr=REFL_THR, bt_thr=BT_THR,
                   k_ndsi=K_NDSI, k_refl=K_REFL, k_bt=K_BT):
    """Возвращает (cloudiness 0..1, mask_bin, pct, ndsi)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        ndsi = (ri1 - ri3) / (ri1 + ri3)

    # “мягкие” признаки (0..1)
    c_ndsi = _sigmoid((ndsi - ndsi_thr) / k_ndsi)      # выше порога → облачнее
    c_refl = _sigmoid((ri2  - refl_thr) / k_refl)      # выше порога → облачнее
    c_bt   = _sigmoid((bt_thr - bi5)   / k_bt)         # ниже порога → облачнее

    # Непрерывное "ИЛИ": 1 - ∏(1 - ci)
    cloudiness = 1.0 - (1.0 - c_ndsi) * (1.0 - c_refl) * (1.0 - c_bt)

    # NaN → 0
    nan_mask = np.isnan(ri1) | np.isnan(ri2) | np.isnan(ri3) | np.isnan(bi5)
    cloudiness = np.where(nan_mask, 0.0, cloudiness).astype(np.float32)

    # Бинарка для визуализации
    mask_bin = cloudiness > BIN_THR
    cloudy = int(np.count_nonzero(mask_bin))
    total  = int(mask_bin.size)
    pct    = (cloudy / total) * 100.0 if total > 0 else 0.0

    return cloudiness, mask_bin, pct, ndsi

def cloudiness_night(bi4, bi5,
                     thr4=THR_I4, thr5=THR_I5, k4=K_BT4, k5=K_BT5):
    """Возвращает (cloudiness 0..1, mask_bin, pct)."""
    c4 = _sigmoid((thr4 - bi4) / k4)   # ниже thr → облачнее
    c5 = _sigmoid((thr5 - bi5) / k5)

    cloudiness = 1.0 - (1.0 - c4) * (1.0 - c5)
    nan_mask = np.isnan(bi4) | np.isnan(bi5)
    cloudiness = np.where(nan_mask, 0.0, cloudiness).astype(np.float32)

    mask_bin = cloudiness > BIN_THR
    cloudy = int(np.count_nonzero(mask_bin))
    total  = int(mask_bin.size)
    pct    = (cloudy / total) * 100.0 if total > 0 else 0.0

    return cloudiness, mask_bin, pct


# ---------- Сохранение float-маски ----------
def save_float_mask_nc(path_nc, cloudiness, mask_bin=None, ndsi=None):
    """Сохраняет NetCDF с cloudiness (float 0..1) + опционально бинарку и NDSI."""
    import xarray as xr
    da = {"cloudiness": (("y","x"), cloudiness.astype("float32"))}
    if mask_bin is not None:
        da["cloud_mask_bin"] = (("y","x"), mask_bin.astype("uint8"))
    if ndsi is not None:
        da["NDSI"] = (("y","x"), ndsi.astype("float32"))
    xr.Dataset(da).to_netcdf(path_nc)
    print(f"Float-маска сохранена: {path_nc}")


# ---------- Основной процесс по паре ----------
def process_pair(img_path: str, geo_path: str):
    print(f"\nОбработка сцены:\n  IMG: {img_path}\n  GEO: {geo_path}")

    chans = open_vnp02img_vars(img_path, debug=True)
    have = set(chans.keys())

    # День
    if {"I01","I02","I03","I05"}.issubset(have):
        ri1, ri2, ri3, bi5 = chans["I01"], chans["I02"], chans["I03"], chans["I05"]
        cloudiness, mask_bin, pct, ndsi = cloudiness_day(
            ri1, ri2, ri3, bi5,
            ndsi_thr=NDSI_THR, refl_thr=REFL_THR, bt_thr=BT_THR,
            k_ndsi=K_NDSI, k_refl=K_REFL, k_bt=K_BT
        )
        alg = "DAY (soft: NDSI/I2/BT)"

    # Ночь
    elif {"I04","I05"}.issubset(have):
        bi4, bi5 = chans["I04"], chans["I05"]
        cloudiness, mask_bin, pct = cloudiness_night(
            bi4, bi5, thr4=THR_I4, thr5=THR_I5, k4=K_BT4, k5=K_BT5
        )
        alg = "NIGHT (soft: BT I4/I5)"
        ndsi = None
        ri1 = ri2 = ri3 = None

    else:
        print("Пропускаем: нет набора каналов ни для дня, ни для ночи.")
        return

    print(f"{alg} → Доля облаков (по порогу {BIN_THR:.2f}): {pct:.2f}%")

    # --- PNG (бинарка для наглядности)
    # if SAVE_PNG:
    #     ensure_dir(OUT_DIR)
    #     fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    #     ax = axes.ravel()

    #     if ri1 is not None:  # день
    #         ax[0].imshow(ri1); ax[0].set_title("I1")
    #         ax[1].imshow(ri2); ax[1].set_title("I2")
    #         ax[2].imshow(ri3); ax[2].set_title("I3")
    #         ax[3].imshow(bi5); ax[3].set_title("I5 (BT)")
    #         if ndsi is not None:
    #             im4 = ax[4].imshow(ndsi, vmin=-1, vmax=1); ax[4].set_title("NDSI")
    #         im5 = ax[5].imshow(mask_bin); ax[5].set_title("Cloud mask (bin)")
    #     else:                 # ночь
    #         ax[0].imshow(bi4); ax[0].set_title("I4 (BT)")
    #         ax[1].imshow(bi5); ax[1].set_title("I5 (BT)")
    #         ax[2].axis("off"); ax[3].axis("off"); ax[4].axis("off")
    #         ax[5].imshow(mask_bin); ax[5].set_title("Cloud mask (bin)")

    #     for a in ax: a.axis("off")
    #     fig.suptitle(f"{alg} — Cloud fraction (thr={BIN_THR:.2f}): {pct:.2f}%", y=0.98)
    #     plt.tight_layout()
    #     out_png = os.path.join(OUT_DIR, os.path.basename(img_path).replace(".nc", "_mask.png"))
    #     plt.savefig(out_png, dpi=150)
    #     plt.close(fig)
    #     print(f"PNG сохранён: {out_png}")

    # --- Float-маска → NetCDF
    out_nc = os.path.join(OUT_DIR, os.path.basename(img_path).replace(".nc", "_cloudiness.nc"))
    save_float_mask_nc(out_nc, cloudiness, mask_bin, ndsi)


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
