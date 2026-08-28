import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
"""
TC_AWS.py  (KST 버전 — TC.py 에서 파생)
=========================================
TC.py 는 SM2RAIN + ERA5 + GPM 삼중분석(TCA)이었다.
본 스크립트는 GPM 대신 지상관측 IDW_AWS 를 세 번째 자료로 넣어
  SM2RAIN + ERA5 + IDW_AWS 삼중분석(TCA) 을 같은 방식으로 수행한다.
  - 데이터 1 (active)  : SM2RAIN 강수량   ← SM2RAIN_KST.nc   (참조/재스케일 기준)
  - 데이터 2 (model)   : ERA5-Land 강수량 ← era5_land{yr}_KST.nc (70×70 → 격자 regrid)
  - 데이터 3 (passive) : IDW_AWS 강수량   ← da_IDWs.nc ('AWS (IDW)')
  - 검증 기준(독립)     : IDW_ASOS         ← da_IDWs.nc ('ASOS (IDW)')

* 평가규약: 보간=AWS, 평가=ASOS (AWS ≠ ASOS 이므로 AWS 를 TCA 입력으로 써도
  ASOS 는 여전히 준독립 검증 기준으로 유효). Thiessen 미사용.
* TCA 피팅(오차분산·가중치·ERA5/AWS 재스케일 계수)은 2021년 표본만으로 추정하고,
  병합(anomaly merge)·climatology 복원은 전 공통기간(2021–2025)에 적용한다.
  (SM2RAIN 캘리브 2021 과 동일한 "2021 학습 → 전 기간 적용" 방식)

실행: /Users/kim/miniconda3/bin/python3 code/use/TC/TC_AWS.py
"""

import os
import warnings
import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ============================================================
# 경로 설정  (KST)
# ============================================================
BASE_PATH    = '/Users/kim/cpuserver_data/personal_data/project_KIHS/result/ASCAT/precipitation'
KST_DIR      = '/Users/kim/cpuserver_data/personal_data/project_KIHS'
IDW_BOTH_PATH = "/Users/kim/DAS/projects/KIHS/IDW/both/da_IDWs.nc"   # 지상 IDW (AWS·ASOS 단일파일, KST)

SM2RAIN_PATH = os.path.join(KST_DIR, 'result/SM2RAIN/SM2RAIN_KST.nc')           # KST
ERA5_KST_DIR = os.path.join(KST_DIR, 'data/era5_KST')                           # era5_land{yr}_KST.nc

OUT_MERGE_PATH   = os.path.join(BASE_PATH, 'TCA/TC_AWS_KST_T.nc')
OUT_WEIGHTS_PATH = os.path.join(BASE_PATH, 'TCA/TC_AWS_weights_KST_T.nc')
FIG_DIR          = os.path.dirname(os.path.abspath(__file__))                   # use/TC (그림 저장)

YEARS       = [2021, 2022, 2023, 2024, 2025]
MIN_TRIPLET = 100
P_THRESH    = 0.05


# ============================================================
# ERA5 KST 전용 로더 + 격자 보간 (scipy 불필요 numpy bilinear)
# ============================================================
def _axis_matrix(src, tgt):
    """1축 선형보간 가중행렬 (len(tgt), len(src)). src 오름차순 가정."""
    src = np.asarray(src, dtype='float64'); tgt = np.asarray(tgt, dtype='float64')
    idx = np.clip(np.searchsorted(src, tgt) - 1, 0, len(src) - 2)
    x0, x1 = src[idx], src[idx + 1]
    w = np.clip((tgt - x0) / (x1 - x0), 0.0, 1.0)
    M = np.zeros((len(tgt), len(src)), dtype='float64')
    rows = np.arange(len(tgt)); M[rows, idx] = 1.0 - w; M[rows, idx + 1] = w
    return M


def _apply_bilinear(cube, Mlat, Mlon):
    s1 = np.tensordot(Mlat, cube, axes=([1], [0]))     # (TL, nlon, T)
    s2 = np.tensordot(Mlon, s1,   axes=([1], [1]))     # (TO, TL, T)
    return s2.transpose(1, 0, 2)


def _regrid_bilinear(cube, src_lat, src_lon, tgt_lat, tgt_lon, min_cov=0.5):
    """cube (nlat,nlon,T) → (TL,TO,T). NaN-aware 재정규화."""
    Mlat = _axis_matrix(src_lat, tgt_lat); Mlon = _axis_matrix(src_lon, tgt_lon)
    valid = np.isfinite(cube).astype('float64')
    cube0 = np.nan_to_num(cube, nan=0.0)
    num = _apply_bilinear(cube0, Mlat, Mlon)
    den = _apply_bilinear(valid, Mlat, Mlon)
    out = np.where(den > min_cov, num / np.where(den == 0, np.nan, den), np.nan)
    return out


def load_era5_kst_regrid(years, era5_dir, tgt_lat, tgt_lon):
    """연도별 era5_land{yr}_KST.nc(daily mm/day, 70×70) 로드·concat → SM2RAIN 격자로 보간."""
    arrs, times = [], []
    src_lat = src_lon = None
    for yr in years:
        p = os.path.join(era5_dir, f'era5_land{yr}_KST.nc')
        if not os.path.exists(p):
            continue
        ds = xr.open_dataset(p)
        da = ds['tp']
        lat = ds['lat'].values; lon = ds['lon'].values
        a = np.transpose(da.values.astype('float64'),
                         ([d.lower() for d in da.dims].index('lat'),
                          [d.lower() for d in da.dims].index('lon'),
                          [d.lower() for d in da.dims].index('time')))
        if lat[0] > lat[-1]:
            lat = lat[::-1]; a = a[::-1, :, :]
        if lon[0] > lon[-1]:
            lon = lon[::-1]; a = a[:, ::-1, :]
        arrs.append(a); times.append(pd.DatetimeIndex(ds['time'].values))
        src_lat, src_lon = lat, lon
        ds.close()
    if not arrs:
        raise FileNotFoundError(f'ERA5 KST 파일 없음: {era5_dir}')
    cube = np.concatenate(arrs, axis=2)
    t = times[0]
    for x in times[1:]:
        t = t.append(x)
    tl = np.sort(tgt_lat); to = np.sort(tgt_lon)
    rg = _regrid_bilinear(cube, src_lat, src_lon, tl, to)
    rg[rg < 0] = np.nan
    return rg, tl, to, t


def load_idw_both(varname_sub):
    """da_IDWs.nc 에서 varname_sub('aws'/'asos') 포함 변수를 (lat,lon,time) 오름차순으로 반환."""
    ds_i = xr.open_dataset(IDW_BOTH_PATH)
    var  = next(v for v in ds_i.data_vars if varname_sub in v.lower())
    da_i = ds_i[var]
    lat_c = ds_i['y'].values.squeeze(); lon_c = ds_i['x'].values.squeeze()
    if lat_c.ndim == 2:
        lat_c = lat_c[:, 0]; lon_c = lon_c[0, :]
    dl = [d.lower() for d in da_i.dims]
    arr_i = np.transpose(da_i.values.astype(np.float64),
                         (dl.index('y'), dl.index('x'), dl.index('time')))
    if lat_c[0] > lat_c[-1]:
        lat_c = lat_c[::-1]; arr_i = arr_i[::-1, :, :]
    if lon_c[0] > lon_c[-1]:
        lon_c = lon_c[::-1]; arr_i = arr_i[:, ::-1, :]
    arr_i[arr_i < 0] = np.nan
    t_i = pd.DatetimeIndex(ds_i['time'].values)
    ds_i.close()
    return arr_i, lat_c, lon_c, t_i


# ============================================================
# 1. 데이터 로딩
# ============================================================
print("=" * 60)
print("Step 1. 데이터 로딩  (TCA 삼중: SM2RAIN + ERA5 + IDW_AWS)")
print("=" * 60)

def load_precip(path, label=''):
    ds  = xr.open_dataset(path)
    var = next(v for v in ds.data_vars
               if v.lower() not in {'lat', 'lon', 'latitude', 'longitude', 'time'})
    da  = ds[var]

    def get_coord_1d(ds, keys, axis):
        for k in list(ds.coords) + list(ds.data_vars):
            if any(kw in k.lower() for kw in keys):
                v = ds[k].values.squeeze()
                if v.ndim == 2:
                    return v[:, 0] if axis == 0 else v[0, :]
                return v
        return None

    lat_1d  = get_coord_1d(ds, ['lat', 'latitude'],  axis=0)
    lon_1d  = get_coord_1d(ds, ['lon', 'longitude'], axis=1)
    time_1d = ds['time'].values

    dims    = [d.lower() for d in da.dims]
    lat_ax  = next((i for i, d in enumerate(dims) if 'lat' in d or d == 'y'), None)
    lon_ax  = next((i for i, d in enumerate(dims) if 'lon' in d or d == 'x'), None)
    time_ax = next((i for i, d in enumerate(dims) if 'time' in d), None)

    if None in (lat_ax, lon_ax, time_ax):
        sizes   = da.values.shape
        time_ax = int(np.argmax(sizes))
        rem     = [i for i in range(3) if i != time_ax]
        lat_ax, lon_ax = rem[0], rem[1]

    arr = np.transpose(da.values.astype(np.float64), (lat_ax, lon_ax, time_ax))

    if lat_1d is not None and lat_1d[0] > lat_1d[-1]:
        lat_1d = lat_1d[::-1];  arr = arr[::-1, :, :]
    if lon_1d is not None and lon_1d[0] > lon_1d[-1]:
        lon_1d = lon_1d[::-1];  arr = arr[:, ::-1, :]

    arr[arr < 0] = np.nan
    return arr, lat_1d, lon_1d, pd.DatetimeIndex(time_1d)


# ── SM2RAIN (단일 파일, KST) — 목표 격자 정의 ─────────────────
print("[ SM2RAIN (KST) ]")
arr_sm, lat_sm, lon_sm, time_sm = load_precip(SM2RAIN_PATH)
print(f"  SM2RAIN: {arr_sm.shape}  {time_sm[0].date()} ~ {time_sm[-1].date()}\n")

lat_1d = lat_sm
lon_1d = lon_sm

# ── IDW_AWS (단일 파일 da_IDWs.nc, KST) — GPM 대체 세 번째 자료 ──
print("[ IDW_AWS (KST, da_IDWs.nc) ]")
arr_aws, lat_aws, lon_aws, time_aws = load_idw_both('aws')
print(f"  IDW_AWS: {arr_aws.shape}  {time_aws[0].date()} ~ {time_aws[-1].date()}")

# AWS 격자가 SM2RAIN 격자와 다르면 in-memory 재격자 (보통 동일 58×46)
if arr_aws.shape[:2] != arr_sm.shape[:2] or not (
        np.allclose(lat_sm, lat_aws, atol=0.01) and np.allclose(lon_sm, lon_aws, atol=0.01)):
    print(f"  AWS 재격자: {arr_aws.shape[:2]} → {arr_sm.shape[:2]} (SM2RAIN 격자)")
    arr_aws = _regrid_bilinear(arr_aws, lat_aws, lon_aws, np.sort(lat_sm), np.sort(lon_sm))
    arr_aws[arr_aws < 0] = np.nan
    lat_aws, lon_aws = np.sort(lat_sm), np.sort(lon_sm)
print()

# ── ERA5 (연도별 KST + SM2RAIN 격자로 in-memory regrid) ───────
print("[ ERA5 (KST, 70×70 → SM2RAIN 격자 regrid) ]")
arr_era5, lat_era5, lon_era5, time_era5 = load_era5_kst_regrid(
    YEARS, ERA5_KST_DIR, lat_sm, lon_sm)
print(f"  ERA5: {arr_era5.shape}  {time_era5[0].date()} ~ {time_era5[-1].date()}\n")


# ============================================================
# 2. 공통 격자 확인
# ============================================================
print("Step 2. 공통 격자 확인")

assert arr_sm.shape[:2] == arr_aws.shape[:2] == arr_era5.shape[:2], \
    f"격자 크기 불일치: SM={arr_sm.shape[:2]}, AWS={arr_aws.shape[:2]}, ERA5={arr_era5.shape[:2]}"
assert np.allclose(lat_sm, lat_aws,  atol=0.01), "lat 불일치 (SM2RAIN vs AWS)"
assert np.allclose(lon_sm, lon_aws,  atol=0.01), "lon 불일치 (SM2RAIN vs AWS)"
assert np.allclose(lat_sm, lat_era5, atol=0.01), "lat 불일치 (SM2RAIN vs ERA5)"
assert np.allclose(lon_sm, lon_era5, atol=0.01), "lon 불일치 (SM2RAIN vs ERA5)"

n_lat, n_lon = lat_1d.shape[0], lon_1d.shape[0]
print(f"  격자 확인 완료: ({n_lat}, {n_lon})")


# ============================================================
# 3. 공통 기간 추출
# ============================================================
print("\nStep 3. 공통 기간 추출")

common_dates = time_sm.intersection(time_era5).intersection(time_aws)
print(f"  SM2RAIN : {time_sm[0].date()} ~ {time_sm[-1].date()}  ({len(time_sm)}일)")
print(f"  ERA5    : {time_era5[0].date()} ~ {time_era5[-1].date()}  ({len(time_era5)}일)")
print(f"  AWS     : {time_aws[0].date()} ~ {time_aws[-1].date()}  ({len(time_aws)}일)")
print(f"  공통    : {common_dates[0].date()} ~ {common_dates[-1].date()}  ({len(common_dates)}일)")

if len(common_dates) < MIN_TRIPLET:
    raise ValueError(f"공통 기간({len(common_dates)}일)이 MIN_TRIPLET({MIN_TRIPLET})보다 짧습니다.")

def select_dates(arr, time_idx, common):
    mask = time_idx.isin(common)
    return arr[:, :, mask]

X_sm   = select_dates(arr_sm,   time_sm,   common_dates)
X_era5 = select_dates(arr_era5, time_era5, common_dates)
X_aws  = select_dates(arr_aws,  time_aws,  common_dates)
T = len(common_dates)
print(f"  공통 기간 배열: SM={X_sm.shape}, ERA5={X_era5.shape}, AWS={X_aws.shape}")


# ============================================================
# 4. Anomaly 계산 + Climatology 분리
# ============================================================
print("\nStep 4. Climatology 분리 + Anomaly 계산")

def calc_climatology(arr_3d, dates):
    months = dates.month
    clim   = np.full_like(arr_3d, np.nan)
    for m in range(1, 13):
        idx = np.where(months == m)[0]
        if len(idx) == 0:
            continue
        mu = np.nanmean(arr_3d[:, :, idx], axis=2, keepdims=True)
        clim[:, :, idx] = mu
    anomaly = arr_3d - clim
    return anomaly, clim

A_sm,   clim_sm   = calc_climatology(X_sm,   common_dates)
A_era5, clim_era5 = calc_climatology(X_era5, common_dates)
A_aws,  clim_aws  = calc_climatology(X_aws,  common_dates)

print(f"  A_sm   range: {np.nanmin(A_sm):.3f} ~ {np.nanmax(A_sm):.3f}")
print(f"  A_era5 range: {np.nanmin(A_era5):.3f} ~ {np.nanmax(A_era5):.3f}")
print(f"  A_aws  range: {np.nanmin(A_aws):.3f} ~ {np.nanmax(A_aws):.3f}")

# ── TCA 피팅(오차분산·가중치·재스케일 계수)은 2021년 표본만 사용 ──
FIT_YEAR   = 2021
fit_mask   = (common_dates.year == FIT_YEAR)
A_sm_fit   = A_sm[:, :, fit_mask]
A_era5_fit = A_era5[:, :, fit_mask]
A_aws_fit  = A_aws[:, :, fit_mask]
print(f"  TCA 피팅 기간: {FIT_YEAR}  ({int(fit_mask.sum())}일)"
      f"  — 오차분산·가중치·재스케일 계수를 이 표본으로만 추정")


# ============================================================
# 5. TCA — error variance 추정 (ref=SM2RAIN)
# ============================================================
print("\nStep 5. TCA (error variance 추정)...")

def dong_tca_pixel(x, y, z, min_n):
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    n     = valid.sum()
    if n < min_n:
        return np.nan, np.nan, np.nan, n
    xv, yv, zv = x[valid], y[valid], z[valid]
    denom_y = float(yv @ zv)
    denom_z = float(zv @ yv)
    if abs(denom_y) < 1e-12 or abs(denom_z) < 1e-12:
        return np.nan, np.nan, np.nan, n
    ax_ay = float(xv @ zv) / denom_y
    ax_az = float(xv @ yv) / denom_z
    xs = xv
    ys = ax_ay * yv
    zs = ax_az * zv
    ex = float(np.mean((xs - ys) * (xs - zs)))
    ey = float(np.mean((ys - xs) * (ys - zs)))
    ez = float(np.mean((zs - xs) * (zs - ys)))
    return ex, ey, ez, n

shape = (n_lat, n_lon)
err2_sm   = np.full(shape, np.nan)
err2_era5 = np.full(shape, np.nan)
err2_aws  = np.full(shape, np.nan)
n_valid_map = np.zeros(shape, dtype=int)

for i in tqdm(range(n_lat), desc="  위도줄"):
    for j in range(n_lon):
        ex, ey, ez, nv    = dong_tca_pixel(A_sm_fit[i,j,:], A_era5_fit[i,j,:], A_aws_fit[i,j,:], MIN_TRIPLET)
        err2_sm[i,j]   = ex
        err2_era5[i,j] = ey
        err2_aws[i,j]  = ez
        n_valid_map[i,j] = nv

print(f"  TCA 완료: {int(np.sum(n_valid_map >= MIN_TRIPLET))} / {n_lat*n_lon} 픽셀")

# 음(–)/0 오차분산 대치(구멍 방지)
for _e in (err2_sm, err2_era5, err2_aws):
    _pos  = _e[np.isfinite(_e) & (_e > 0)]
    _fill = float(np.median(_pos)) if _pos.size else 1.0
    _bad  = np.isfinite(_e) & (_e <= 0)
    _e[_bad] = _fill
    print(f"    음/0 분산 대치: {int(_bad.sum())}픽셀 → {_fill:.4f}")

print(f"  σ²_SM2RAIN : mean={np.nanmean(err2_sm):.4f}  n_valid={np.sum(np.isfinite(err2_sm))}")
print(f"  σ²_ERA5    : mean={np.nanmean(err2_era5):.4f}  n_valid={np.sum(np.isfinite(err2_era5))}")
print(f"  σ²_AWS     : mean={np.nanmean(err2_aws):.4f}  n_valid={np.sum(np.isfinite(err2_aws))}")


# ============================================================
# 6. 가중치 계산
# ============================================================
print("\nStep 6. 가중치 계산")

denom = err2_sm*err2_era5 + err2_era5*err2_aws + err2_sm*err2_aws

w_sm   = err2_era5 * err2_aws  / denom
w_era5 = err2_sm   * err2_aws  / denom
w_aws  = err2_sm   * err2_era5 / denom

valid_w = np.isfinite(w_sm) & np.isfinite(w_era5) & np.isfinite(w_aws)

print(f"  가중치 유효 픽셀: {valid_w.sum()} / {n_lat*n_lon}")
print(f"  w_SM2RAIN mean={np.nanmean(w_sm):.3f}")
print(f"  w_ERA5    mean={np.nanmean(w_era5):.3f}")
print(f"  w_AWS     mean={np.nanmean(w_aws):.3f}")
print(f"  w_sum check: mean={np.nanmean(w_sm + w_era5 + w_aws):.6f} (≈1.0)")


# ============================================================
# 7. Anomaly 병합
# ============================================================
print("\nStep 7. Anomaly 병합...")

scale_era5 = np.full(shape, np.nan)
scale_aws  = np.full(shape, np.nan)

for i in range(n_lat):
    for j in range(n_lon):
        # 재스케일 계수도 2021 표본으로만 추정 (ref=SM2RAIN)
        s = A_sm_fit[i, j, :];  a = A_era5_fit[i, j, :];  g = A_aws_fit[i, j, :]
        valid = np.isfinite(s) & np.isfinite(a) & np.isfinite(g)
        if valid.sum() < MIN_TRIPLET:
            continue
        sv, av, gv = s[valid], a[valid], g[valid]
        d_y = float(av @ gv);  d_z = float(gv @ av)
        if abs(d_y) > 1e-12:
            scale_era5[i, j] = float(sv @ gv) / d_y
        if abs(d_z) > 1e-12:
            scale_aws[i, j]  = float(sv @ av) / d_z

M_anomaly = np.full((n_lat, n_lon, T), np.nan)

for i in tqdm(range(n_lat), desc="  위도줄"):
    for j in range(n_lon):
        if not valid_w[i, j]:
            continue
        if not (np.isfinite(scale_era5[i,j]) and np.isfinite(scale_aws[i,j])):
            continue
        xs = A_sm[i, j, :]
        ys = scale_era5[i, j] * A_era5[i, j, :]
        zs = scale_aws[i, j]  * A_aws[i, j, :]
        M_anomaly[i, j, :] = w_sm[i,j]*xs + w_era5[i,j]*ys + w_aws[i,j]*zs

print("  Anomaly 병합 완료")


# ============================================================
# 8. Climatology 복원
# ============================================================
print("\nStep 8. Climatology 복원...")

P_merged = M_anomaly + clim_sm
P_merged = np.maximum(P_merged, 0.0)

print(f"  P_merged range: {np.nanmin(P_merged):.3f} ~ {np.nanmax(P_merged):.3f} mm/day")
print(f"  유효 픽셀: {np.sum(np.any(np.isfinite(P_merged), axis=2))}")


# ============================================================
# 9. NetCDF 저장
# ============================================================
print(f"\nStep 9. NetCDF 저장")

ds_merge = xr.Dataset(
    data_vars={
        "precipitation": xr.DataArray(
            np.transpose(P_merged, (2, 0, 1)),
            dims=("time", "lat", "lon"),
            attrs={"long_name": "TCA-merged precipitation (SM2RAIN + ERA5 + IDW_AWS)",
                   "units": "mm/day",
                   "method": "Dong 2020 TCA",
                   "fit_period": "2021 only (weights & rescale estimated on 2021, applied 2021-2025)"}
        )
    },
    coords={
        "time": xr.DataArray(common_dates.values, dims="time"),
        "lat":  xr.DataArray(lat_1d, dims="lat",
                             attrs={"long_name": "latitude",  "units": "degrees_north"}),
        "lon":  xr.DataArray(lon_1d, dims="lon",
                             attrs={"long_name": "longitude", "units": "degrees_east"}),
    },
    attrs={"title":  "TCA Merged Precipitation with IDW_AWS (KST)",
           "inputs": "SM2RAIN_KST + ERA5_KST(regrid) + IDW_AWS",
           "eval":   "IDW_ASOS (independent)",
           "time_zone": "KST (UTC+9); daily boundary 00-24 KST",
           "period": f"{common_dates[0].date()} ~ {common_dates[-1].date()}"}
)
os.makedirs(os.path.dirname(OUT_MERGE_PATH), exist_ok=True)
ds_merge.to_netcdf(OUT_MERGE_PATH)
print(f"  병합 강수량 저장: {OUT_MERGE_PATH}")

ds_weights = xr.Dataset(
    data_vars={
        "err2_SM2RAIN": xr.DataArray(err2_sm,   dims=("lat","lon"),
            attrs={"long_name": "TCA error variance (SM2RAIN)", "units": "mm²/day²"}),
        "err2_ERA5":    xr.DataArray(err2_era5,  dims=("lat","lon"),
            attrs={"long_name": "TCA error variance (ERA5)",    "units": "mm²/day²"}),
        "err2_AWS":     xr.DataArray(err2_aws,   dims=("lat","lon"),
            attrs={"long_name": "TCA error variance (IDW_AWS)", "units": "mm²/day²"}),
        "w_SM2RAIN":    xr.DataArray(w_sm,        dims=("lat","lon"),
            attrs={"long_name": "TCA weight SM2RAIN"}),
        "w_ERA5":       xr.DataArray(w_era5,      dims=("lat","lon"),
            attrs={"long_name": "TCA weight ERA5"}),
        "w_AWS":        xr.DataArray(w_aws,       dims=("lat","lon"),
            attrs={"long_name": "TCA weight IDW_AWS"}),
        "scale_ERA5":   xr.DataArray(scale_era5,  dims=("lat","lon"),
            attrs={"long_name": "Scaling parameter ax/ay (ERA5→SM2RAIN ref)"}),
        "scale_AWS":    xr.DataArray(scale_aws,   dims=("lat","lon"),
            attrs={"long_name": "Scaling parameter ax/az (IDW_AWS→SM2RAIN ref)"}),
        "n_valid":      xr.DataArray(n_valid_map.astype(np.int16), dims=("lat","lon"),
            attrs={"long_name": "number of valid triplet observations"}),
    },
    coords={
        "lat": xr.DataArray(lat_1d, dims="lat"),
        "lon": xr.DataArray(lon_1d, dims="lon"),
    }
)
ds_weights.to_netcdf(OUT_WEIGHTS_PATH)
print(f"  TCA 진단 저장: {OUT_WEIGHTS_PATH}")
print("\n✅ 완료!")



#################fig#########################################################
# ============================================================
# V-0. 검증 기준 자료 준비 (IDW_ASOS — TCA 입력과 독립)
# ============================================================
print("\n[V-0] 검증 기준 자료 준비 (IDW_ASOS 로딩)...")

ASOS_val_full, lat_ref, lon_ref, time_asos_val = load_idw_both('asos')
print(f"  IDW_ASOS(da_IDWs): {ASOS_val_full.shape}  {time_asos_val[0].date()}~{time_asos_val[-1].date()}")
assert ASOS_val_full.shape[:2] == arr_sm.shape[:2], \
    f"검증자료 격자 불일치: ASOS={ASOS_val_full.shape[:2]} vs SM2RAIN={arr_sm.shape[:2]}"

asos_mask = time_asos_val.isin(common_dates)
aws_mask  = time_aws.isin(common_dates)
sm_mask   = time_sm.isin(common_dates)

REF_val  = ASOS_val_full[:, :, asos_mask]   # 검증 기준 (IDW_ASOS, 독립)
AWS_val  = arr_aws[:, :, aws_mask]          # TCA 입력 자료(AWS) — 참고 비교용
SM_val   = arr_sm[:, :, sm_mask]
ERA5_val = X_era5

# ── 남한(South Korea) 마스크: 검증 계산·그림은 남한 육지 픽셀만 ──
import shapefile as _shpf
from shapely.geometry import shape as _shape, Point as _Point
from shapely.ops import unary_union as _uu, transform as _tr
from shapely.prepared import prep as _prep
from pyproj import Transformer as _Tfm

KOREA_SHP = '/Users/kim/cpuserver_data/python_modules/kunhee/Data/SM2RAIN/Korea.shp'
_sk_poly = _uu([_shape(s.__geo_interface__) for s in _shpf.Reader(KOREA_SHP).shapes()])
_sk_poly = _tr(_Tfm.from_crs(5179, 4326, always_xy=True).transform, _sk_poly)
_sk_prep = _prep(_sk_poly.buffer(0.05))          # 해안 픽셀 포함 여유
SK_MASK = np.zeros((n_lat, n_lon), dtype=bool)
for _i in range(n_lat):
    for _j in range(n_lon):
        SK_MASK[_i, _j] = _sk_prep.contains(_Point(lon_1d[_j], lat_1d[_i]))
print(f"  남한 마스크: {int(SK_MASK.sum())} / {n_lat * n_lon} 픽셀 (검증·그림 남한 한정)")

for _a in (REF_val, AWS_val, SM_val, ERA5_val, P_merged):        # 3D (lat,lon,T)
    _a[~SK_MASK, :] = np.nan
for _a in (err2_sm, err2_era5, err2_aws, w_sm, w_era5, w_aws):    # 2D 진단(F-1·F-2)
    _a[~SK_MASK] = np.nan

print(f"  P_merged : {P_merged.shape}")
print(f"  REF_val  : {REF_val.shape}  ← IDW_ASOS (독립 검증 기준)")
print(f"  AWS_val  : {AWS_val.shape}  ← IDW_AWS (TCA 입력, 참고)")
print(f"  SM_val   : {SM_val.shape}")
print(f"  ERA5_val : {ERA5_val.shape}")


# ============================================================
# V-1. 픽셀별 Pearson R 지도 (4개 자료)
# ============================================================
from scipy.stats import pearsonr

def pixelwise_corr(pred, ref):
    nl, nx = pred.shape[:2]
    R = np.full((nl, nx), np.nan)
    t = min(pred.shape[2], ref.shape[2])
    p_ = pred[:, :, :t];  r_ = ref[:, :, :t]
    for i in range(nl):
        for j in range(nx):
            p = p_[i,j,:];  r = r_[i,j,:]
            ok = np.isfinite(p) & np.isfinite(r)
            if ok.sum() >= 10:
                R[i,j], _ = pearsonr(p[ok], r[ok])
    return R

R_merged  = pixelwise_corr(P_merged,  REF_val)
R_sm2rain = pixelwise_corr(SM_val,    REF_val)
R_aws     = pixelwise_corr(AWS_val,   REF_val)
R_era5    = pixelwise_corr(ERA5_val,  REF_val)

print(f"  R_merged  : mean={np.nanmean(R_merged):.3f}  median={np.nanmedian(R_merged):.3f}")
print(f"  R_SM2RAIN : mean={np.nanmean(R_sm2rain):.3f}  median={np.nanmedian(R_sm2rain):.3f}")
print(f"  R_AWS     : mean={np.nanmean(R_aws):.3f}  median={np.nanmedian(R_aws):.3f}")
print(f"  R_ERA5    : mean={np.nanmean(R_era5):.3f}  median={np.nanmedian(R_era5):.3f}")


# ============================================================
# 공통 유틸
# ============================================================
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.family']        = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def _edges(c):
    h = np.diff(c)/2.0;  e = np.empty(len(c)+1)
    e[1:-1] = c[:-1]+h;  e[0] = c[0]-h[0];  e[-1] = c[-1]+h[-1]
    return e

def _pcm(ax, lon_1d, lat_1d, data, **kw):
    lon2d, lat2d = np.meshgrid(_edges(lon_1d), _edges(lat_1d))
    return ax.pcolormesh(lon2d, lat2d, data, shading='flat',
                         transform=ccrs.PlateCarree(), zorder=2, **kw)

def _map_ax(ax, lon_1d, lat_1d, margin=(0.35, 0.22)):
    lom, lam = margin
    ax.set_extent([lon_1d.min()-lom, lon_1d.max()+lom,
                   lat_1d.min()-lam, lat_1d.max()+lam],
                  crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale('10m'),     facecolor='#ddeeff', zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('10m'),      facecolor='#f7f7f2', zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='#333', lw=0.7, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'),   edgecolor='#888',
                   lw=0.5, linestyle=':', zorder=3)
    gl = ax.gridlines(draw_labels=True, linestyle='--', lw=0.35, alpha=0.5,
                      x_inline=False, y_inline=False)
    gl.top_labels = False;  gl.right_labels = False
    gl.xlabel_style = {'size': 7};  gl.ylabel_style = {'size': 7}
    return ax

def calc_stats_maps(pred, ref):
    nl, nx = pred.shape[:2]
    R_map    = np.full((nl,nx), np.nan)
    RMSD_map = np.full((nl,nx), np.nan)
    Bias_map = np.full((nl,nx), np.nan)
    t = min(pred.shape[2], ref.shape[2])
    p_ = pred[:,:,:t];  r_ = ref[:,:,:t]
    for i in range(nl):
        for j in range(nx):
            p = p_[i,j,:];  r = r_[i,j,:]
            ok = np.isfinite(p) & np.isfinite(r)
            if ok.sum() < 10: continue
            R_map[i,j], _ = pearsonr(p[ok], r[ok])
            diff           = p[ok] - r[ok]
            Bias_map[i,j]  = np.mean(diff)
            RMSD_map[i,j]  = np.sqrt(np.mean((diff - diff.mean())**2))
    return R_map, RMSD_map, Bias_map

BASINS = {
    '한강 유역':   (36.8, 38.5, 126.0, 129.0),
    '낙동강 유역': (35.0, 37.0, 127.5, 129.5),
    '금강 유역':   (35.8, 37.2, 126.3, 128.0),
    '영산강 유역': (34.5, 35.8, 126.3, 127.5),
}

def _bmask(lat_1d, lon_1d, lat_min, lat_max, lon_min, lon_max):
    return ((lat_1d >= lat_min) & (lat_1d <= lat_max))[:,None] & \
           ((lon_1d >= lon_min) & (lon_1d <= lon_max))[None,:]

def _bmean(arr_3d, mask_2d):
    ii, jj = np.where(mask_2d)
    if len(ii) == 0: return np.full(arr_3d.shape[2], np.nan)
    return np.nanmean(arr_3d[ii,jj,:], axis=0)

FIG_PATH = os.path.join(BASE_PATH, 'TCA')
os.makedirs(FIG_PATH, exist_ok=True)

# 자료 공통 설정 (GPM → IDW_AWS 로 교체)
COL_DATA = [
    ('IDW_AWS',      AWS_val),
    ('SM2RAIN',      SM_val),
    ('ERA5',         ERA5_val),
    ('Merged (TCA)', P_merged),
]

STYLE = {
    'IDW_ASOS': dict(color='#757575', lw=1.4, ls='-',  marker=None, ms=0, label='IDW_ASOS (In situ)', alpha=0.95, zorder=5),
    'SM2RAIN':  dict(color='#2196F3', lw=0,   ls='',   marker='o',  ms=5, label='SM2RAIN',           alpha=0.65, zorder=2),
    'IDW_AWS':  dict(color='#8E24AA', lw=0,   ls='',   marker='^',  ms=5, label='IDW_AWS',           alpha=0.65, zorder=2),
    'ERA5':     dict(color='#27ae60', lw=0,   ls='',   marker='s',  ms=5, label='ERA5',              alpha=0.65, zorder=2),
    'Merged':   dict(color='#E53935', lw=0,   ls='',   marker='<',  ms=5, label='Merged (TCA)',      alpha=0.75, zorder=3),
}
TS_STYLE = {
    'IDW_ASOS': dict(color='#757575', lw=1.5, ls='-',  label='IDW_ASOS (In situ)', alpha=0.5,  zorder=5),
    'SM2RAIN':  dict(color='#2196F3', lw=0.9, ls='--', label='SM2RAIN',           alpha=0.85, zorder=2),
    'IDW_AWS':  dict(color='#8E24AA', lw=0.9, ls='-.', label='IDW_AWS',           alpha=0.85, zorder=2),
    'ERA5':     dict(color='#27ae60', lw=0.9, ls=':',  label='ERA5',              alpha=0.85, zorder=2),
    'Merged':   dict(color='#E53935', lw=1.1, ls='-',  label='Merged (TCA)',      alpha=0.95, zorder=3),
}


# ============================================================
# F-1. TCA Error Variance 공간분포 지도
# ============================================================
print("\n[F-1] TCA Error Variance 지도...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150,
    subplot_kw={'projection': ccrs.PlateCarree()})
plt.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.05, wspace=0.12)

ev_cfg = [
    (err2_sm,   'SM2RAIN', 0, 80),
    (err2_era5, 'ERA5',    0, 80),
    (err2_aws,  'IDW_AWS', 0, 80),
]

for ax, (data, label, vmin, vmax) in zip(axes, ev_cfg):
    _map_ax(ax, lon_1d, lat_1d)
    plot_data = np.where(data > 0, data, np.nan)
    im = _pcm(ax, lon_1d, lat_1d, plot_data, cmap='YlOrRd', vmin=vmin, vmax=vmax)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.06, shrink=0.85)
    cb.set_label('Error Variance (mm²/day²)', fontsize=9)
    ax.set_title(f'{label}\nError Variance', fontsize=11, fontweight='bold', pad=6)

plt.suptitle('F-1.  TCA Error Variance 공간분포  (SM2RAIN + ERA5 + IDW_AWS)',
             fontsize=13, fontweight='bold', y=1.02)
fig.savefig(os.path.join(FIG_DIR, 'F1_error_variance_AWS_KST.png'), dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# F-2. TCA 가중치 공간분포 지도
# ============================================================
print("\n[F-2] TCA 가중치 지도...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150,
    subplot_kw={'projection': ccrs.PlateCarree()})
plt.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.05, wspace=0.12)

wt_cfg = [
    (w_sm,   'SM2RAIN'),
    (w_era5, 'ERA5'),
    (w_aws,  'IDW_AWS'),
]

for ax, (data, label) in zip(axes, wt_cfg):
    _map_ax(ax, lon_1d, lat_1d)
    im = _pcm(ax, lon_1d, lat_1d, data, cmap='viridis_r', vmin=0, vmax=1)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.06, shrink=0.85)
    cb.set_label('Weight [-]', fontsize=9)
    ax.set_title(f'{label}\nWeight  (mean={np.nanmean(data):.3f})',
                 fontsize=11, fontweight='bold', pad=6)

w_sum = np.nansum(np.stack([w_sm, w_era5, w_aws], axis=0), axis=0)
plt.suptitle(f'F-2.  TCA Weights 공간분포  (w_sum mean={np.nanmean(w_sum[w_sum>0]):.3f})',
             fontsize=13, fontweight='bold', y=1.02)
fig.savefig(os.path.join(FIG_DIR, 'F2_weights_AWS_KST.png'), dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# F-3. 통계 지표 공간분포 지도 (R / ubRMSD / Bias) — 총 3장
# ============================================================
print("\n[F-3] 통계 지표 공간분포 지도...")

stat_results = {}
for ds_label, ds_arr in COL_DATA:
    R_m, RMSD_m, Bias_m = calc_stats_maps(ds_arr, REF_val)
    stat_results[ds_label] = {'R': R_m, 'ubRMSD': RMSD_m, 'Bias': Bias_m}
    print(f"  {ds_label:15s}: R={np.nanmean(R_m):.3f}  ubRMSD={np.nanmean(RMSD_m):.3f}  Bias={np.nanmean(Bias_m):+.3f}")

stat_cfgs = [
    ('R',      'Pearson R',  'RdYlGn', 'R [-]'),
    ('ubRMSD', 'ubRMSD',     'YlOrRd', 'ubRMSD (mm/day)'),
    ('Bias',   'Bias',       'RdBu_r', 'Bias (mm/day)'),
]

for stat_key, stat_name, cmap, cb_lbl in stat_cfgs:
    all_vals = np.concatenate([
        stat_results[lbl][stat_key][np.isfinite(stat_results[lbl][stat_key])].ravel()
        for lbl, _ in COL_DATA
    ])
    if stat_key == 'R':
        vmin, vmax = 0.0, 1.0
    elif stat_key == 'ubRMSD':
        vmin = 0.0;  vmax = float(np.nanpercentile(all_vals, 95))
    else:
        lim = float(np.nanpercentile(np.abs(all_vals), 95));  vmin, vmax = -lim, lim

    fig, axes = plt.subplots(1, 4, figsize=(22, 6), dpi=150,
                             subplot_kw={"projection": ccrs.PlateCarree()})
    plt.subplots_adjust(left=0.03, right=0.91, top=0.85, bottom=0.05, wspace=0.08)

    for col_idx, (ds_label, _) in enumerate(COL_DATA):
        ax   = axes[col_idx]
        data = stat_results[ds_label][stat_key]
        _map_ax(ax, lon_1d, lat_1d)
        im = _pcm(ax, lon_1d, lat_1d, data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f'{ds_label}\nmean={np.nanmean(data):.3f}   median={np.nanmedian(data):.3f}',
                     fontsize=10, fontweight='bold', pad=6)
        ax.text(0.02, 0.03, 'vs IDW_ASOS',
                transform=ax.transAxes, fontsize=7.5, va='bottom', fontstyle='italic',
                bbox=dict(fc='white', alpha=0.75, ec='#aaa', lw=0.4, boxstyle='round,pad=0.22'),
                zorder=5)

    cax = fig.add_axes([0.922, 0.12, 0.012, 0.72])
    cb  = fig.colorbar(im, cax=cax)
    cb.set_label(cb_lbl, fontsize=11)
    cb.ax.tick_params(labelsize=9)
    fig.suptitle(f'F-3.  {stat_name}  —  IDW_AWS / SM2RAIN / ERA5 / Merged (TCA)  vs  IDW_ASOS',
                 fontsize=13, fontweight='bold')
    fig.savefig(os.path.join(FIG_DIR, f'F3_{stat_key}_AWS_vs_IDW_ASOS_KST.png'),
                dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# F-4. 유역별 시계열 + 산점도 (유역별 1장씩)
# ============================================================
print("\n[F-4] 유역별 시계열 + 산점도...")

def scatter_stats(p, r):
    ok = np.isfinite(p) & np.isfinite(r)
    if ok.sum() < 10: return np.nan, np.nan, np.nan
    diff = p[ok] - r[ok]
    bias = float(np.mean(diff))
    ubrmsd = float(np.sqrt(np.mean((diff - bias)**2)))
    R, _ = pearsonr(p[ok], r[ok])
    return float(R), ubrmsd, bias

legend_ts = [
    mlines.Line2D([], [], color=TS_STYLE[k]['color'], lw=TS_STYLE[k]['lw'],
                  ls=TS_STYLE[k]['ls'], alpha=TS_STYLE[k].get('alpha',1),
                  label=TS_STYLE[k]['label'])
    for k in ['IDW_ASOS', 'SM2RAIN', 'IDW_AWS', 'ERA5', 'Merged']
]

date_arr = common_dates.to_pydatetime()

for basin_name, (lat_min, lat_max, lon_min, lon_max) in BASINS.items():
    bmask = _bmask(lat_1d, lon_1d, lat_min, lat_max, lon_min, lon_max)
    n_pix = int(bmask.sum())

    ts_ref  = _bmean(REF_val,  bmask)
    ts_sm   = _bmean(SM_val,   bmask)
    ts_aws  = _bmean(AWS_val,  bmask)
    ts_era5 = _bmean(ERA5_val, bmask)
    ts_mrg  = _bmean(P_merged, bmask)

    fig = plt.figure(figsize=(21, 5.4), dpi=150)
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                            left=0.06, right=0.97, top=0.76, bottom=0.18,
                            wspace=0.38, width_ratios=[3.2, 1.25, 1.25])

    # (a) 시계열
    ax_ts = fig.add_subplot(gs[0, 0])
    ax_ts.fill_between(date_arr, ts_ref, alpha=0.12, color='#1A237E')
    ax_ts.plot(date_arr, ts_ref,  **TS_STYLE['IDW_ASOS'])
    ax_ts.plot(date_arr, ts_sm,   **TS_STYLE['SM2RAIN'])
    ax_ts.plot(date_arr, ts_aws,  **TS_STYLE['IDW_AWS'])
    ax_ts.plot(date_arr, ts_era5, **TS_STYLE['ERA5'])
    ax_ts.plot(date_arr, ts_mrg,  **TS_STYLE['Merged'])
    ax_ts.set_title(f'(a) {basin_name}  —  시계열  (유역 내 {n_pix}픽셀 평균)',
                    fontsize=11, fontweight='bold', pad=6)
    ax_ts.set_ylabel('강수량 (mm/day)', fontsize=10)
    ax_ts.xaxis.set_major_locator(mticker.MaxNLocator(7))
    ax_ts.tick_params(axis='x', rotation=25, labelsize=9)
    ax_ts.tick_params(axis='y', labelsize=9)
    ax_ts.spines[['top','right']].set_visible(False)
    ax_ts.grid(axis='y', alpha=0.25, ls='--', lw=0.6)
    ax_ts.legend(handles=legend_ts, fontsize=8.5, ncol=5,
                 loc='upper center', bbox_to_anchor=(0.5, 1.22),
                 framealpha=0.95, edgecolor='#ccc', columnspacing=0.8)

    # (b) 전체 산점도 (4자료)
    ax_b = fig.add_subplot(gs[0, 1])
    lim_b = 0.0
    for key, ts_pred in [('SM2RAIN', ts_sm), ('IDW_AWS', ts_aws),
                          ('ERA5', ts_era5), ('Merged', ts_mrg)]:
        ok = np.isfinite(ts_pred) & np.isfinite(ts_ref)
        if ok.sum() < 5: continue
        ax_b.scatter(ts_ref[ok], ts_pred[ok], s=20,
                     alpha=STYLE[key].get('alpha', 0.6),
                     color=STYLE[key]['color'],
                     marker=STYLE[key]['marker'], zorder=3)
        lim_b = max(lim_b, float(np.nanmax(ts_ref[ok])), float(np.nanmax(ts_pred[ok])))

    lim_b *= 1.10
    ax_b.plot([0,lim_b],[0,lim_b], '--', color='gray', lw=1.0, alpha=0.7)
    ax_b.set_xlim(0,lim_b);  ax_b.set_ylim(0,lim_b)
    ax_b.set_xlabel('IDW_ASOS (mm/day)', fontsize=9)
    ax_b.set_ylabel('Estimated (mm/day)', fontsize=9)
    ax_b.set_title(f'(b) {basin_name}\nvs IDW_ASOS', fontsize=10, fontweight='bold')
    ax_b.tick_params(labelsize=9)
    ax_b.spines[['top','right']].set_visible(False)
    ax_b.grid(alpha=0.2, ls='--', lw=0.5)

    # 통계 텍스트박스
    y_txt = 0.97
    for key, ts_pred, fc in [
        ('SM2RAIN', ts_sm,   '#2196F3'),
        ('IDW_AWS', ts_aws,  '#8E24AA'),
        ('ERA5',    ts_era5, '#27ae60'),
        ('Merged',  ts_mrg,  '#E53935'),
    ]:
        R_, ubr_, bias_ = scatter_stats(ts_pred, ts_ref)
        if not np.isfinite(R_): continue
        txt = f"R={R_:.3f}\nubRMSD={ubr_:.3f}\nbias={bias_:+.3f}"
        ax_b.text(0.97, y_txt, txt, transform=ax_b.transAxes,
                  fontsize=7.5, va='top', ha='right', color=fc, fontweight='bold',
                  bbox=dict(fc='white', alpha=0.65, ec=fc, lw=0.8, boxstyle='round,pad=0.25'))
        y_txt -= 0.28

    # (c) Merged 단독 산점도
    ax_c = fig.add_subplot(gs[0, 2])
    ok_m = np.isfinite(ts_mrg) & np.isfinite(ts_ref)
    if ok_m.sum() >= 10:
        ax_c.scatter(ts_ref[ok_m], ts_mrg[ok_m], s=20,
                     color='#E53935', alpha=0.7, zorder=3)
        lim_c = max(float(np.nanmax(ts_ref[ok_m])), float(np.nanmax(ts_mrg[ok_m]))) * 1.10
        ax_c.plot([0,lim_c],[0,lim_c], '--', color='gray', lw=1.0, alpha=0.7)
        ax_c.set_xlim(0,lim_c);  ax_c.set_ylim(0,lim_c)
        R_m, ubr_m, bias_m = scatter_stats(ts_mrg, ts_ref)
        ax_c.text(0.05, 0.93,
                  f"R = {R_m:.3f}\nubRMSD = {ubr_m:.3f} mm/day\nbias = {bias_m:+.3f} mm/day",
                  transform=ax_c.transAxes, fontsize=9, va='top',
                  color='#E53935', fontweight='bold',
                  bbox=dict(fc='white', alpha=0.8, ec='#E53935', lw=0.8, boxstyle='round,pad=0.3'))

    ax_c.set_xlabel('IDW_ASOS (mm/day)', fontsize=9)
    ax_c.set_ylabel('Merged (mm/day)', fontsize=9)
    ax_c.set_title(f'(c) Merged (TCA)\nvs IDW_ASOS', fontsize=10, fontweight='bold')
    ax_c.tick_params(labelsize=9)
    ax_c.spines[['top','right']].set_visible(False)
    ax_c.grid(alpha=0.2, ls='--', lw=0.5)

    fig.suptitle(f'F-4.  {basin_name}  —  유역별 평균 시계열 & 산점도  (기준: IDW_ASOS)',
                 fontsize=13, fontweight='bold', y=1.01)

    safe_name = basin_name.replace(' ', '_')
    fig.savefig(os.path.join(FIG_DIR, f'F4_{safe_name}_AWS_KST.png'), dpi=150, bbox_inches='tight')
    plt.show()


print("\n✅ 검증 + 피규어 완료!")

#%%
