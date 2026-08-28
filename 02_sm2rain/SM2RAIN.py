"""
SM2RAIN.py  (기본 3-파라미터 SM2RAIN, KST)
==========================================
Brocca et al.(2014) 기본식으로 ASCAT 토양수분에서 강수를 역산한다. (GA 아님)

  물수지: Z·dθ/dt = P − a·θ^b        (유출 r, 증발산 e 는 강우 중 무시)
  강수 역산:  P(t) = Z·(dθ/dt) + a·θ(t)^b     ,  P<0 → 0
  파라미터 3개: a(배수계수), b(배수지수), Z(유효토심, mm)

입력 : ASCAT_daily_stack_KST.nc  (보간 안 한 원본 stack, KST 일경계)
       - sm_volumetric (y, x, time), 픽셀별 0~1 정규화하여 θ 로 사용
캘리브: 2021 IDW_AWS (KST) 기준 픽셀별 최적화 → 2022~2025 적용
출력 : SM2RAIN_KST.nc  (time, lat, lon / mm/day) + a,b,Z 파라미터 맵

* ASCAT(58×46) → AWS/SM2RAIN 격자(49×49) 는 numpy NaN-aware bilinear 로 보간
  (scipy 불필요). IF_KST 결측보간본이 아니라 원본 stack 을 그대로 사용.
"""

import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import minimize   # 최적화만 사용 (설치 필요)
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ============================================================
# 경로 및 설정
# ============================================================
DATA_PATH  = "/Users/kim/cpuserver_data/python_modules/kunhee/Results/SM2RAIN"       # IDW_AWS
ASCAT_PATH = "/Users/kim/cpuserver_data/personal_data/project_KIHS/data/layer/ASCAT_daily_stack_KST.nc"  # 보간 안 한 원본
OUTPUT_PATH = "/Users/kim/cpuserver_data/personal_data/project_KIHS/result/SM2RAIN/SM2RAIN_KST.nc"
PARAM_PATH = "/Users/kim/cpuserver_data/personal_data/project_KIHS/result/SM2RAIN/SM2RAIN_params_KST.nc"

YEARS = [2021, 2022, 2023, 2024, 2025]
CAL_YEAR = 2021
TEST_START_YEAR = 2022
MIN_VALID = 30
REUSE_SAVED_PARAMS = True   # True: 저장된 a,b,Z가 있으면 재최적화 없이 고정값 사용

# 파라미터 순서: a, b, Z  (기존 basic SM2RAIN_precipitation.nc 와 동일 경계)
#   a : 배수(drainage) 계수
#   b : 배수 비선형 지수
#   Z : 유효 토양층 깊이 [mm] (저류 변화항 Z·dθ/dt)
INIT_GUESS = [20.0, 10.0, 20.0]
BOUNDS = [
    (0.01, 500.0),   # a
    (0.01, 30.0),    # b
    (1.00, 150.0),    # Z
]

EPS = 1e-6


# ============================================================
# ASCAT → AWS 격자 numpy bilinear (NaN-aware, scipy 불필요)
# ============================================================
def _axis_matrix(src, tgt):
    src = np.asarray(src, dtype="float64"); tgt = np.asarray(tgt, dtype="float64")
    idx = np.clip(np.searchsorted(src, tgt) - 1, 0, len(src) - 2)
    x0, x1 = src[idx], src[idx + 1]
    w = np.clip((tgt - x0) / (x1 - x0), 0.0, 1.0)
    M = np.zeros((len(tgt), len(src)), dtype="float64")
    rows = np.arange(len(tgt)); M[rows, idx] = 1.0 - w; M[rows, idx + 1] = w
    return M


def _apply(cube, Mlat, Mlon):
    s1 = np.tensordot(Mlat, cube, axes=([1], [0]))
    s2 = np.tensordot(Mlon, s1, axes=([1], [1]))
    return s2.transpose(1, 0, 2)


def regrid_bilinear(cube, src_lat, src_lon, tgt_lat, tgt_lon, min_cov=0.5):
    """cube (nlat,nlon,T) → (TL,TO,T). 바다/결측 NaN 은 유효 가중 재정규화로 제외."""
    Mlat = _axis_matrix(src_lat, tgt_lat); Mlon = _axis_matrix(src_lon, tgt_lon)
    valid = np.isfinite(cube).astype("float64")
    cube0 = np.nan_to_num(cube, nan=0.0)
    num = _apply(cube0, Mlat, Mlon)
    den = _apply(valid, Mlat, Mlon)
    return np.where(den > min_cov, num / np.where(den == 0, np.nan, den), np.nan)


# ============================================================
# IDW_AWS 로딩
# ============================================================
# 단일 파일(da_IDWs.nc: 변수 'AWS (IDW)', 좌표 y/x 2D, dims time,y,x)을 1회만 로드해 캐시.
# 파일은 수정하지 않고, 배열/좌표만 정리해 (lat, lon, time) 로 연도별 슬라이싱해 반환.
_AWS_DS_CACHE = None

def load_aws_year(yr):
    global _AWS_DS_CACHE
    fp = "/Users/kim/DAS/projects/KIHS/IDW/both/da_IDWs.nc"
    if _AWS_DS_CACHE is None:
        _AWS_DS_CACHE = xr.open_dataset(fp)
    ds = _AWS_DS_CACHE

    # AWS 변수 명시 선택 ('aws' 포함, ASOS 제외)
    skip = {"lat", "lon", "latitude", "longitude", "time"}
    var = next((v for v in ds.data_vars if "aws" in v.lower()),
               next(v for v in ds.data_vars if v.lower() not in skip and "asos" not in v.lower()))
    da = ds[var]

    def find_coord(dataset, keys):
        for cname in list(dataset.coords) + list(dataset.data_vars):
            if any(k in cname.lower() for k in keys):
                return dataset[cname].values.squeeze()
        return None

    lat_c = find_coord(ds, ['y', "lat", "latitude"])
    lon_c = find_coord(ds, ['x', "lon", "longitude"])
    if lat_c.ndim == 2:
        lat_c = lat_c[:, 0]; lon_c = lon_c[0, :]
    if lat_c[0] > lat_c[-1]: lat_c = lat_c[::-1]
    if lon_c[0] > lon_c[-1]: lon_c = lon_c[::-1]

    dims = [d.lower() for d in da.dims]
    lat_ax = next((i for i, d in enumerate(dims) if "lat" in d or d == "y"), None)
    lon_ax = next((i for i, d in enumerate(dims) if "lon" in d or d == "x"), None)
    time_ax = next((i for i, d in enumerate(dims) if "time" in d), None)
    if None in (lat_ax, lon_ax, time_ax):
        sizes = da.values.shape
        time_ax = int(np.argmax(sizes))
        rem = [i for i in range(3) if i != time_ax]
        lat_ax, lon_ax = rem[0], rem[1]
    arr = np.transpose(da.values, (lat_ax, lon_ax, time_ax))   # (lat, lon, time)

    # 단일 파일에서 해당 연도만 슬라이싱 (연도별 배열 → 이후 연도순 concat 과 정합)
    tvals = pd.DatetimeIndex(ds["time"].values)
    ymask = (tvals.year == yr)
    arr = arr[:, :, ymask]
    print(f"  IDW_AWS_{yr}: {tuple(da.shape)} {da.dims} -> {arr.shape}  ({int(ymask.sum())}일)")
    return arr, lat_c, lon_c


def apply_flip(arr, key):
    return {"none": arr, "lat": arr[::-1, :, :], "lon": arr[:, ::-1, :],
            "both": arr[::-1, ::-1, :]}[key].copy()


# ============================================================
# 기본 SM2RAIN (3-파라미터)
# ============================================================
def sm2rain_basic(sm, a, b, Z):
    """P(t) = Z·(θ_t − θ_{t-1}) + a·θ_t^b ,  음수는 0 (역산의 자연 제약)."""
    sm = np.asarray(sm, dtype=float)
    out = np.full_like(sm, np.nan, dtype=float)
    finite = np.isfinite(sm)
    if finite.sum() < MIN_VALID:
        return out
    theta = np.clip(sm, EPS, 1.0 - EPS)
    dtheta = np.zeros_like(theta)
    dtheta[1:] = theta[1:] - theta[:-1]
    dtheta[~finite] = np.nan
    p_est = Z * dtheta + a * np.power(theta, b)
    p_est = np.where(np.isfinite(p_est), np.maximum(p_est, 0.0), np.nan)
    out[finite] = p_est[finite]
    return out


def objective(params, sm_1d, p_obs_1d):
    a, b, Z = params
    valid = np.isfinite(sm_1d) & np.isfinite(p_obs_1d)
    if valid.sum() < MIN_VALID:
        return 1e20
    p_est = sm2rain_basic(sm_1d[valid], a, b, Z)
    p_obs = p_obs_1d[valid]
    mse = np.nanmean((p_est - p_obs) ** 2)
    if not np.isfinite(mse):
        return 1e20
    return mse


def calibrate_pixel(sm_1d, p_obs_1d):
    starts = [INIT_GUESS, [5.0, 5.0, 10.0], [50.0, 15.0, 40.0], [1.0, 2.0, 5.0]]
    best, best_fun = None, np.inf
    for g in starts:
        try:
            res = minimize(objective, g, args=(sm_1d, p_obs_1d),
                           bounds=BOUNDS, method="L-BFGS-B",
                           options={"maxiter": 800, "ftol": 1e-8})
            if res.success and np.isfinite(res.fun) and res.fun < best_fun:
                best, best_fun = res, res.fun
        except Exception:
            continue
    if best is None or not np.all(np.isfinite(best.x)):
        return None
    return best.x


def _coords_match(ds, lat, lon):
    if "lat" in ds.coords and not np.allclose(ds["lat"].values, lat, equal_nan=True):
        return False
    if "lon" in ds.coords and not np.allclose(ds["lon"].values, lon, equal_nan=True):
        return False
    return True


def load_saved_params(lat, lon):
    """저장된 a,b,Z를 읽는다. 전용 파일이 없으면 기존 산출물에서 fallback."""
    for path in [PARAM_PATH, OUTPUT_PATH]:
        if not os.path.exists(path):
            continue
        try:
            with xr.open_dataset(path) as ds:
                if not all(v in ds for v in ("a", "b", "Z")):
                    print(f"  저장 파라미터 없음: {path}")
                    continue
                if ds["a"].shape != (len(lat), len(lon)) or not _coords_match(ds, lat, lon):
                    print(f"  저장 파라미터 격자 불일치: {path}")
                    continue
                a_saved = ds["a"].values.astype(float)
                b_saved = ds["b"].values.astype(float)
                Z_saved = ds["Z"].values.astype(float)
            print(f"  저장 파라미터 사용: {path}")
            return a_saved, b_saved, Z_saved
        except Exception as exc:
            print(f"  저장 파라미터 읽기 실패: {path} ({exc})")
    return None


def save_params(path, a_map, b_map, Z_map, lat, lon):
    ds_param = xr.Dataset(
        data_vars={
            "a": xr.DataArray(a_map, dims=("lat", "lon"),
                              attrs={"description": "drainage coefficient"}),
            "b": xr.DataArray(b_map, dims=("lat", "lon"),
                              attrs={"description": "drainage exponent"}),
            "Z": xr.DataArray(Z_map, dims=("lat", "lon"),
                              attrs={"units": "mm", "description": "effective soil depth"}),
        },
        coords={
            "lat": xr.DataArray(lat, dims="lat", attrs={"units": "degrees_north"}),
            "lon": xr.DataArray(lon, dims="lon", attrs={"units": "degrees_east"}),
        },
        attrs={
            "title": "SM2RAIN basic calibrated parameters",
            "model_equation": "P = Z*dtheta/dt + a*theta**b",
            "parameter_order": "a, b, Z",
            "calibration_period": str(CAL_YEAR),
            "source_SM": "ASCAT sm_volumetric normalized 0-1 per pixel (ASCAT_daily_stack_KST.nc, no interpolation)",
            "source_P_cal": "IDW_AWS",
            "bounds": str(BOUNDS),
        },
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    ds_param.to_netcdf(path)
    print(f"  파라미터 저장: {path}")


# ============================================================
# 메인
# ============================================================
print("=" * 60)
print("SM2RAIN (basic 3-param): 데이터 로딩")
print("=" * 60)

aws_data, aws_lat_from_file, aws_lon_from_file = {}, None, None
for yr in YEARS:
    arr, lat_c, lon_c = load_aws_year(yr)
    aws_data[yr] = arr
    if aws_lat_from_file is None:
        aws_lat_from_file, aws_lon_from_file = lat_c, lon_c

da_ascat = xr.open_dataset(ASCAT_PATH)
ascat_lat = da_ascat["lat"].values
ascat_lon = da_ascat["lon"].values
sm_da = da_ascat["sm_volumetric"].sel(time=slice(str(YEARS[0]), str(YEARS[-1])))
ascat_time = sm_da.time.values
SM_raw = sm_da.transpose("y", "x", "time").values
time_pd = pd.DatetimeIndex(ascat_time)
print(f"\n  ASCAT SM: {SM_raw.shape}  {time_pd[0].date()} ~ {time_pd[-1].date()}")

# 좌표 정렬 + 픽셀별 0~1 정규화
lat_1d = (ascat_lat[:, 0] if ascat_lat.ndim == 2 else ascat_lat).copy()
lon_1d = (ascat_lon[0, :] if ascat_lon.ndim == 2 else ascat_lon).copy()
SM = SM_raw.copy()
if lat_1d[0] > lat_1d[-1]:
    lat_1d = lat_1d[::-1]; SM = SM[::-1, :, :]
if lon_1d[0] > lon_1d[-1]:
    lon_1d = lon_1d[::-1]; SM = SM[:, ::-1, :]
SM_min = np.nanmin(SM, axis=2, keepdims=True)
SM_max = np.nanmax(SM, axis=2, keepdims=True)
SM_norm = (SM - SM_min) / (SM_max - SM_min + EPS)
print(f"  ASCAT lat {lat_1d[0]:.2f}~{lat_1d[-1]:.2f}  lon {lon_1d[0]:.2f}~{lon_1d[-1]:.2f}")

aws_lat_1d, aws_lon_1d = aws_lat_from_file, aws_lon_from_file
n_lat, n_lon = aws_data[CAL_YEAR].shape[:2]
print(f"  AWS  lat {aws_lat_1d[0]:.2f}~{aws_lat_1d[-1]:.2f} ({n_lat})  lon {aws_lon_1d[0]:.2f}~{aws_lon_1d[-1]:.2f} ({n_lon})")

cal_idx = np.where(time_pd.year == CAL_YEAR)[0]
test_idx = np.where(time_pd.year >= TEST_START_YEAR)[0]
out_idx = np.where(time_pd.year >= CAL_YEAR)[0]      # 저장 대상: 2021(학습)~2025 전체

# ASCAT SM → AWS 격자 (numpy bilinear). 오름차순 타깃으로 보간 후 배열 정렬 일치
tl = np.sort(aws_lat_1d); to = np.sort(aws_lon_1d)
def to_aws(t_idx):
    rg = regrid_bilinear(SM_norm[:, :, t_idx], lat_1d, lon_1d, tl, to)
    # aws 좌표가 내림차순이면 맞춰 뒤집기
    if aws_lat_1d[0] > aws_lat_1d[-1]: rg = rg[::-1, :, :]
    if aws_lon_1d[0] > aws_lon_1d[-1]: rg = rg[:, ::-1, :]
    return rg

_sm_cal = to_aws(cal_idx)
sm_vmask = np.sum(np.isfinite(_sm_cal), axis=2) >= MIN_VALID
sm_lat_c = float(np.sum(aws_lat_1d[:, None] * sm_vmask) / sm_vmask.sum())
sm_lon_c = float(np.sum(aws_lon_1d[None, :] * sm_vmask) / sm_vmask.sum())

# IDW_AWS flip 자동 감지 (유효영역 중심 최근접)
best_key, best_dist = "none", np.inf
for key in ["none", "lat", "lon", "both"]:
    cand = apply_flip(aws_data[CAL_YEAR], key)
    pv = np.sum(np.isfinite(cand), axis=2) >= MIN_VALID
    if pv.sum() == 0:
        continue
    p_lat_c = float(np.sum(aws_lat_1d[:, None] * pv) / pv.sum())
    p_lon_c = float(np.sum(aws_lon_1d[None, :] * pv) / pv.sum())
    dist = (p_lat_c - sm_lat_c) ** 2 + (p_lon_c - sm_lon_c) ** 2
    if dist < best_dist:
        best_dist, best_key = dist, key
print(f"  selected AWS flip: {best_key}")

P_all = np.concatenate([apply_flip(aws_data[yr], best_key) for yr in YEARS], axis=2)
P_trimmed = P_all[:, :, : len(ascat_time)]
P_cal = P_trimmed[:, :, cal_idx]

print("\n기간 분할")
print(f"  CAL  ({len(cal_idx)}일): {time_pd[cal_idx[0]].date()} ~ {time_pd[cal_idx[-1]].date()}")
print(f"  TEST ({len(test_idx)}일): {time_pd[test_idx[0]].date()} ~ {time_pd[test_idx[-1]].date()}")

SM_cal_aws = _sm_cal
SM_out_aws = to_aws(out_idx)          # 2021~2025 전체 (추정·저장용)
print(f"  SM_cal_aws {SM_cal_aws.shape}  SM_out_aws {SM_out_aws.shape}")
if SM_cal_aws.shape != P_cal.shape:
    raise ValueError(f"격자 불일치: SM_cal={SM_cal_aws.shape}, P_cal={P_cal.shape}")

# ============================================================
# 픽셀별 파라미터 준비: 저장값 재사용 또는 2021 캘리브
# ============================================================
print("\n픽셀별 SM2RAIN(basic) 파라미터 준비")
lat_len, lon_len = SM_cal_aws.shape[:2]
saved_params = load_saved_params(aws_lat_1d, aws_lon_1d) if REUSE_SAVED_PARAMS else None

if saved_params is not None:
    a_map, b_map, Z_map = saved_params
    print("  재최적화 건너뜀: 저장된 고정 a,b,Z 사용")
    if not os.path.exists(PARAM_PATH):
        save_params(PARAM_PATH, a_map, b_map, Z_map, aws_lat_1d, aws_lon_1d)
else:
    print("  저장 파라미터 없음 또는 재사용 비활성화 -> 2021 기준 새로 최적화")
    a_map = np.full((lat_len, lon_len), np.nan)
    b_map = np.full((lat_len, lon_len), np.nan)
    Z_map = np.full((lat_len, lon_len), np.nan)

    valid_mask = (
        (np.sum(np.isfinite(SM_cal_aws), axis=2) >= MIN_VALID)
        & (np.sum(np.isfinite(P_cal), axis=2) >= MIN_VALID)
    )
    print(f"  최적화 대상 픽셀: {valid_mask.sum()} / {lat_len * lon_len}")

    for i in tqdm(range(lat_len), desc="  위도줄"):
        for j in range(lon_len):
            if not valid_mask[i, j]:
                continue
            params = calibrate_pixel(SM_cal_aws[i, j, :], P_cal[i, j, :])
            if params is None:
                continue
            a_map[i, j], b_map[i, j], Z_map[i, j] = params
    print(f"  캘리브 성공: {np.isfinite(a_map).sum()} 픽셀")
    save_params(PARAM_PATH, a_map, b_map, Z_map, aws_lat_1d, aws_lon_1d)

for name, arr in [("a", a_map), ("b", b_map), ("Z", Z_map)]:
    print(f"  {name}: mean={np.nanmean(arr):.3f} min={np.nanmin(arr):.3f} max={np.nanmax(arr):.3f}")

# ============================================================
# 2021~2025 적용 (캘리브된 픽셀)  ※ 2021은 학습기간(in-sample fit)
# ============================================================
print("\nSM2RAIN(basic) 강수 추정 (2021~2025)")
n_out = SM_out_aws.shape[2]
P_sm2rain = np.full((lat_len, lon_len, n_out), np.nan)
calib_ok = np.isfinite(a_map)
for i in tqdm(range(lat_len), desc="  위도줄"):
    for j in range(lon_len):
        if not calib_ok[i, j]:
            continue
        P_sm2rain[i, j, :] = sm2rain_basic(SM_out_aws[i, j, :],
                                           a_map[i, j], b_map[i, j], Z_map[i, j])
print(f"  유효 추정값: {np.isfinite(P_sm2rain).sum()}")
print(f"  P range: {np.nanmin(P_sm2rain):.2f} ~ {np.nanmax(P_sm2rain):.2f} mm/day")

# ============================================================
# 저장
# ============================================================
print(f"\nNetCDF 저장 -> {OUTPUT_PATH}")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
test_dates = pd.DatetimeIndex(ascat_time[out_idx])   # 2021~2025 전체
out_lat = aws_lat_1d
out_lon = aws_lon_1d
ds_out = xr.Dataset(
    data_vars={
        "precipitation": xr.DataArray(
            np.transpose(P_sm2rain, (2, 0, 1)),
            dims=("time", "lat", "lon"),
            attrs={"long_name": "SM2RAIN (basic 3-param) estimated precipitation",
                   "units": "mm/day",
                   "description": ("Pixel-wise basic SM2RAIN P=Z*dtheta/dt + a*theta^b "
                                   "from ASCAT normalized SM (no gap-filling). "
                                   "Calibrated on 2021 IDW_AWS, applied 2021-2025 "
                                   "(2021 = in-sample/training year).")}),
        "a": xr.DataArray(a_map, dims=("lat", "lon"), attrs={"description": "drainage coefficient"}),
        "b": xr.DataArray(b_map, dims=("lat", "lon"), attrs={"description": "drainage exponent"}),
        "Z": xr.DataArray(Z_map, dims=("lat", "lon"), attrs={"units": "mm", "description": "effective soil depth"}),
    },
    coords={
        "time": xr.DataArray(test_dates.values, dims="time"),
        "lat": xr.DataArray(out_lat, dims="lat", attrs={"units": "degrees_north"}),
        "lon": xr.DataArray(out_lon, dims="lon", attrs={"units": "degrees_east"}),
    },
    attrs={
        "title": "SM2RAIN basic (3-param) Precipitation",
        "model_equation": "P = Z*dtheta/dt + a*theta**b",
        "parameter_order": "a, b, Z",
        "calibration_period": str(CAL_YEAR),
        "test_period": f"{test_dates[0].date()} ~ {test_dates[-1].date()}",
        "source_SM": "ASCAT sm_volumetric normalized 0-1 per pixel (ASCAT_daily_stack_KST.nc, no interpolation)",
        "source_P_cal": "IDW_AWS",
        "time_zone": "KST (UTC+9); daily boundary 00-24 KST",
        "bounds": str(BOUNDS),
    },
)
# NFS 상 기존/스테일(0바이트 손상) 파일을 덮어쓸 때 netCDF4 권한오류가 나므로
# 저장 직전 기존 파일을 제거하고 새로 생성한다.
if os.path.exists(OUTPUT_PATH):
    os.remove(OUTPUT_PATH)
ds_out.to_netcdf(OUTPUT_PATH)
print("  완료:", ds_out["precipitation"].shape, "(time, lat, lon)")
#%%
