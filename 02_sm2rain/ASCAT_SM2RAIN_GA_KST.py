"""
ASCAT_SM2RAIN_GA.py
===================
Loads the same ASCAT and IDW_AWS inputs as ASCAT_caliblartion.py, calibrates a
pixel-wise SM2RAIN-GreenAmpt model on 2021, applies it to 2022-2025, and writes
SM2RAIN_GA.nc. Plotting is intentionally omitted.
"""

import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import minimize
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ============================================================
# Paths and settings: same inputs as the current script
# ============================================================
DATA_PATH = "/Users/kim/cpuserver_data/python_modules/kunhee/Results/SM2RAIN"
# KST 버전: ASCAT(KST 일경계) ↔ IDW_AWS(KST 일경계) 로 시간대 일치.
ASCAT_PATH = "/Users/kim/cpuserver_data/personal_data/project_KIHS/data/layer/ASCAT_daily_stack_interp_KST.nc"
OUTPUT_PATH = "/Users/kim/cpuserver_data/personal_data/project_KIHS/result/SM2RAIN/SM2RAIN_GA_KST_T.nc"

YEARS = [2021, 2022, 2023, 2024, 2025]
CAL_YEAR = 2021
TEST_START_YEAR = 2022
MIN_VALID = 30

# Parameter order: A, C, Z, KS, PSI_F
# A     : multiplier for the Green-Ampt flux term
# C     : runoff/nonlinear saturation correction exponent in (1 - theta**C)
# Z     : effective soil layer depth for the storage tendency term [mm]
# KS    : effective saturated hydraulic conductivity at daily scale [mm/day]
# PSI_F : wetting-front suction head [mm]
INIT_GUESS = [0.05, 2.0, 10.0, 5.0, 100.0]
BOUNDS = [
    (0.0001, 10.0),   # A
    (0.10, 30.0),     # C
    (1.00, 200.0),    # Z
    (0.01, 200.0),    # KS
    (20.0, 1000.0),   # PSI_F
]

EPS = 1e-6
DENOM_FLOOR = 0.02
F_FLOOR = 0.1
P_MAX_REASONABLE = 500.0


# ============================================================
# Data loading helpers
# ============================================================
def load_aws_year(yr):
    """Return IDW_AWS precipitation as (lat, lon, time), plus 1D lat/lon."""
    fp = os.path.join(DATA_PATH, f"Precipitation_IDW_AWS_{yr}.nc")
    ds = xr.open_dataset(fp)

    skip = {"lat", "lon", "latitude", "longitude", "time"}
    var = next(v for v in ds.data_vars if v.lower() not in skip)
    da = ds[var]

    def find_coord(dataset, keywords):
        for cname in dataset.coords:
            if any(k in cname.lower() for k in keywords):
                return dataset[cname].values.squeeze()
        for vname in dataset.data_vars:
            if any(k in vname.lower() for k in keywords):
                return dataset[vname].values.squeeze()
        return None

    lat_coord = find_coord(ds, ["lat", "latitude"])
    lon_coord = find_coord(ds, ["lon", "longitude"])
    if lat_coord is None or lon_coord is None:
        raise ValueError(
            f"Could not find lat/lon in Precipitation_IDW_AWS_{yr}.nc\n"
            f"coords={list(ds.coords)} data_vars={list(ds.data_vars)}"
        )

    if lat_coord.ndim == 2:
        lat_coord = lat_coord[:, 0]
        lon_coord = lon_coord[0, :]

    if lat_coord[0] > lat_coord[-1]:
        lat_coord = lat_coord[::-1]
    if lon_coord[0] > lon_coord[-1]:
        lon_coord = lon_coord[::-1]

    dims = [d.lower() for d in da.dims]
    lat_ax = next((i for i, d in enumerate(dims) if "lat" in d), None)
    lon_ax = next((i for i, d in enumerate(dims) if "lon" in d), None)
    time_ax = next((i for i, d in enumerate(dims) if "time" in d), None)

    if None in (lat_ax, lon_ax, time_ax):
        sizes = da.values.shape
        time_ax = int(np.argmax(sizes))
        rem = [i for i in range(3) if i != time_ax]
        lat_ax, lon_ax = (rem[0], rem[1]) if sizes[rem[0]] >= sizes[rem[1]] else (rem[1], rem[0])

    arr = np.transpose(da.values, (lat_ax, lon_ax, time_ax))
    print(f"  IDW_AWS_{yr}: {da.values.shape} {da.dims} -> {arr.shape} (lat, lon, time)")
    return arr, lat_coord, lon_coord


def apply_flip(arr, key):
    if key == "none":
        return arr.copy()
    if key == "lat":
        return arr[::-1, :, :].copy()
    if key == "lon":
        return arr[:, ::-1, :].copy()
    if key == "both":
        return arr[::-1, ::-1, :].copy()
    raise ValueError(f"Unknown flip key: {key}")


def sm_to_aws(sm_norm, t_idx, lat_1d, lon_1d, aws_lat_1d, aws_lon_1d):
    da = xr.DataArray(
        sm_norm[:, :, t_idx],
        dims=("lat", "lon", "time"),
        coords={"lat": lat_1d, "lon": lon_1d, "time": np.arange(len(t_idx))},
    )
    return da.interp(lat=aws_lat_1d, lon=aws_lon_1d, method="linear").values


# ============================================================
# SM2RAIN-GreenAmpt model
# ============================================================
def sm2rain_ga(sm, A, C, Z, KS, PSI_F):
    """
    Daily SM2RAIN-GA estimate.

    P(t) = (Z*dtheta/dt + A*f_GA(t))/(1 - theta(t)**C)
    f_GA(t) = KS * (1 + PSI_F*delta_theta/F_eff(t))

    With daily ASCAT normalized SM, theta_s/theta_i and hydraulic parameters are
    treated as effective pixel-wise quantities. delta_theta is estimated from the
    local dynamic SM range, and F_eff is mapped from relative wetness to an
    equivalent water depth.
    """
    sm = np.asarray(sm, dtype=float)
    out = np.full_like(sm, np.nan, dtype=float)
    finite = np.isfinite(sm)
    if finite.sum() < MIN_VALID:
        return out

    theta = np.clip(sm, EPS, 1.0 - EPS)
    theta_i = np.nanpercentile(theta[finite], 5)
    theta_s = np.nanpercentile(theta[finite], 95)
    delta_theta = float(np.clip(theta_s - theta_i, 0.03, 0.60))

    dtheta = np.zeros_like(theta)
    dtheta[1:] = theta[1:] - theta[:-1]
    dtheta[~finite] = np.nan

    f_eff = Z * np.maximum(theta - theta_i, F_FLOOR / max(Z, EPS))
    f_ga = KS * (1.0 + (PSI_F * delta_theta) / np.maximum(f_eff, F_FLOOR))

    denom = 1.0 - np.power(theta, C)
    denom = np.maximum(denom, DENOM_FLOOR)

    p_est = (Z * dtheta + A * f_ga) / denom
    p_est = np.where(np.isfinite(p_est), np.maximum(p_est, 0.0), np.nan)
    p_est = np.where(p_est <= P_MAX_REASONABLE, p_est, P_MAX_REASONABLE)
    out[finite] = p_est[finite]
    return out


def objective(params, sm_1d, p_obs_1d):
    A, C, Z, KS, PSI_F = params
    valid = np.isfinite(sm_1d) & np.isfinite(p_obs_1d)
    if valid.sum() < MIN_VALID:
        return 1e20

    p_est = sm2rain_ga(sm_1d[valid], A, C, Z, KS, PSI_F)
    p_obs = p_obs_1d[valid]
    mse = np.nanmean((p_est - p_obs) ** 2)
    if not np.isfinite(mse):
        return 1e20

    # Small regularization keeps the effective GA parameters from drifting to
    # extreme but equivalent A*KS combinations.
    reg = 1e-6 * (A**2 + KS**2 + (PSI_F / 100.0) ** 2)
    return mse + reg


def calibrate_pixel(sm_1d, p_obs_1d):
    starts = [
        INIT_GUESS,
        [0.01, 1.0, 10.0, 2.0, 50.0],
        [0.10, 3.0, 30.0, 5.0, 150.0],
        [0.05, 5.0, 50.0, 10.0, 300.0],
    ]

    best_res = None
    best_fun = np.inf
    for guess in starts:
        try:
            res = minimize(
                objective,
                guess,
                args=(sm_1d, p_obs_1d),
                bounds=BOUNDS,
                method="L-BFGS-B",
                options={"maxiter": 800, "ftol": 1e-8},
            )
            if res.success and np.isfinite(res.fun) and res.fun < best_fun:
                best_res = res
                best_fun = res.fun
        except Exception:
            continue

    if best_res is None or not np.all(np.isfinite(best_res.x)):
        return None
    return best_res.x


# ============================================================
# Main workflow
# ============================================================
print("=" * 60)
print("SM2RAIN-GA: data loading")
print("=" * 60)

aws_data = {}
aws_lat_from_file = None
aws_lon_from_file = None
for yr in YEARS:
    arr, lat_c, lon_c = load_aws_year(yr)
    aws_data[yr] = arr
    if aws_lat_from_file is None:
        aws_lat_from_file = lat_c
        aws_lon_from_file = lon_c

da_ascat = xr.open_dataset(ASCAT_PATH)
ascat_lat = da_ascat["lat"].values
ascat_lon = da_ascat["lon"].values
sm_da = da_ascat["sm_volumetric"].sel(time=slice(str(YEARS[0]), str(YEARS[-1])))
ascat_time = sm_da.time.values
SM_raw = sm_da.transpose("y", "x", "time").values
time_pd = pd.DatetimeIndex(ascat_time)

print(f"\n  ASCAT SM: {SM_raw.shape}")
print(f"  ASCAT time: {time_pd[0].date()} ~ {time_pd[-1].date()}")

print("\nSM coordinate sorting and normalization")
lat_1d = (ascat_lat[:, 0] if ascat_lat.ndim == 2 else ascat_lat).copy()
lon_1d = (ascat_lon[0, :] if ascat_lon.ndim == 2 else ascat_lon).copy()
SM = SM_raw.copy()

if lat_1d[0] > lat_1d[-1]:
    lat_1d = lat_1d[::-1]
    SM = SM[::-1, :, :]
if lon_1d[0] > lon_1d[-1]:
    lon_1d = lon_1d[::-1]
    SM = SM[:, ::-1, :]

SM_min = np.nanmin(SM, axis=2, keepdims=True)
SM_max = np.nanmax(SM, axis=2, keepdims=True)
SM_norm = (SM - SM_min) / (SM_max - SM_min + EPS)

print(f"  ASCAT lat: {lat_1d[0]:.2f} ~ {lat_1d[-1]:.2f}")
print(f"  ASCAT lon: {lon_1d[0]:.2f} ~ {lon_1d[-1]:.2f}")

print("\nAWS grid and flip detection")
aws_lat_1d = aws_lat_from_file
aws_lon_1d = aws_lon_from_file
n_lat, n_lon = aws_data[CAL_YEAR].shape[:2]

print(f"  AWS lat: {aws_lat_1d[0]:.2f} ~ {aws_lat_1d[-1]:.2f} ({n_lat})")
print(f"  AWS lon: {aws_lon_1d[0]:.2f} ~ {aws_lon_1d[-1]:.2f} ({n_lon})")

cal_idx = np.where(time_pd.year == CAL_YEAR)[0]
test_idx = np.where(time_pd.year >= TEST_START_YEAR)[0]

_sm_aws_cal = sm_to_aws(SM_norm, cal_idx, lat_1d, lon_1d, aws_lat_1d, aws_lon_1d)
sm_vmask = np.sum(np.isfinite(_sm_aws_cal), axis=2) >= MIN_VALID
sm_lat_c = float(np.sum(aws_lat_1d[:, None] * sm_vmask) / sm_vmask.sum())
sm_lon_c = float(np.sum(aws_lon_1d[None, :] * sm_vmask) / sm_vmask.sum())

flip_candidates = {
    "none": aws_data[CAL_YEAR],
    "lat": aws_data[CAL_YEAR][::-1, :, :],
    "lon": aws_data[CAL_YEAR][:, ::-1, :],
    "both": aws_data[CAL_YEAR][::-1, ::-1, :],
}

best_key, best_dist = "none", np.inf
for key, cand in flip_candidates.items():
    pv = np.sum(np.isfinite(cand), axis=2) >= MIN_VALID
    if pv.sum() == 0:
        continue
    p_lat_c = float(np.sum(aws_lat_1d[:, None] * pv) / pv.sum())
    p_lon_c = float(np.sum(aws_lon_1d[None, :] * pv) / pv.sum())
    dist = (p_lat_c - sm_lat_c) ** 2 + (p_lon_c - sm_lon_c) ** 2
    print(f"  [{key:4s}] P center=({p_lat_c:.2f},{p_lon_c:.2f}) dist={dist:.4f}")
    if dist < best_dist:
        best_dist, best_key = dist, key

print(f"  selected flip: {best_key}")

P_all = np.concatenate([apply_flip(aws_data[yr], best_key) for yr in YEARS], axis=2)
P_trimmed = P_all[:, :, : len(ascat_time)]
P_cal = P_trimmed[:, :, cal_idx]
P_test = P_trimmed[:, :, test_idx]

print("\nPeriod split")
print(f"  CAL  ({len(cal_idx)} days): {time_pd[cal_idx[0]].date()} ~ {time_pd[cal_idx[-1]].date()}")
print(f"  TEST ({len(test_idx)} days): {time_pd[test_idx[0]].date()} ~ {time_pd[test_idx[-1]].date()}")

print("\nASCAT SM interpolation to AWS grid")
SM_cal_aws = sm_to_aws(SM_norm, cal_idx, lat_1d, lon_1d, aws_lat_1d, aws_lon_1d)
SM_test_aws = sm_to_aws(SM_norm, test_idx, lat_1d, lon_1d, aws_lat_1d, aws_lon_1d)

print(f"  SM_cal_aws : {SM_cal_aws.shape}")
print(f"  SM_test_aws: {SM_test_aws.shape}")
if SM_cal_aws.shape != P_cal.shape:
    raise ValueError(f"Shape mismatch: SM_cal={SM_cal_aws.shape}, P_cal={P_cal.shape}")

print("\nPixel-wise SM2RAIN-GA calibration")
lat_len, lon_len = SM_cal_aws.shape[:2]
A_map = np.full((lat_len, lon_len), np.nan)
C_map = np.full((lat_len, lon_len), np.nan)
Z_map = np.full((lat_len, lon_len), np.nan)
KS_map = np.full((lat_len, lon_len), np.nan)
PSI_map = np.full((lat_len, lon_len), np.nan)

valid_mask = (
    (np.sum(np.isfinite(SM_cal_aws), axis=2) >= MIN_VALID)
    & (np.sum(np.isfinite(P_cal), axis=2) >= MIN_VALID)
)
print(f"  optimization pixels: {valid_mask.sum()} / {lat_len * lon_len}")

for i in tqdm(range(lat_len), desc="  latitude rows"):
    for j in range(lon_len):
        if not valid_mask[i, j]:
            continue
        params = calibrate_pixel(SM_cal_aws[i, j, :], P_cal[i, j, :])
        if params is None:
            continue
        A_map[i, j], C_map[i, j], Z_map[i, j], KS_map[i, j], PSI_map[i, j] = params

print(f"  calibration success: {np.isfinite(A_map).sum()} pixels")
for name, arr in [
    ("A", A_map),
    ("C", C_map),
    ("Z", Z_map),
    ("KS", KS_map),
    ("PSI_F", PSI_map),
]:
    print(f"  {name}: mean={np.nanmean(arr):.3f} min={np.nanmin(arr):.3f} max={np.nanmax(arr):.3f}")

# ------------------------------------------------------------------
# Parameter transfer (육지 빈칸 채움)
#   캘리브: IDW_AWS 가 있는 픽셀(valid_mask, ~643) 에서 매개변수 학습
#   적용  : ASCAT 가 있는 모든 픽셀(apply_mask, ~육지 전체) 로 nearest-neighbor 전이 후 적용
#   계측 타깃(IDW_AWS)이 없는 육지 픽셀도 가장 가까운 캘리브 픽셀의 매개변수를 빌려 추정.
# ------------------------------------------------------------------
print("\nParameter transfer (nearest-neighbor)")
apply_mask = np.sum(np.isfinite(SM_test_aws), axis=2) >= MIN_VALID
calib_ok   = np.isfinite(A_map)
print(f"  calibrated pixels    : {int(calib_ok.sum())}  (IDW_AWS 있는 곳)")
print(f"  apply (ASCAT) pixels : {int(apply_mask.sum())}  (ASCAT 있는 육지)")

# 채워진 매개변수 맵 + 출처 플래그 (1=calibrated, 2=transferred, 0=none)
A_f, C_f, Z_f, KS_f, PSI_f = (m.copy() for m in (A_map, C_map, Z_map, KS_map, PSI_map))
calib_flag = calib_ok.astype(np.uint8)

cal_ij  = np.argwhere(calib_ok)
need_ij = np.argwhere(apply_mask & ~calib_ok)        # 적용대상인데 캘리브 안 된 픽셀
if len(cal_ij) and len(need_ij):
    # 격자 인덱스 공간에서 최근접 캘리브 픽셀 찾기 (scipy 불필요)
    d2 = ((need_ij[:, None, 0] - cal_ij[None, :, 0]) ** 2 +
          (need_ij[:, None, 1] - cal_ij[None, :, 1]) ** 2)
    src = cal_ij[d2.argmin(axis=1)]
    for fmap, cmap in [(A_f, A_map), (C_f, C_map), (Z_f, Z_map),
                       (KS_f, KS_map), (PSI_f, PSI_map)]:
        fmap[need_ij[:, 0], need_ij[:, 1]] = cmap[src[:, 0], src[:, 1]]
    calib_flag[need_ij[:, 0], need_ij[:, 1]] = 2
print(f"  transferred pixels   : {len(need_ij)}")

print("\nSM2RAIN-GA rainfall estimation (apply to all ASCAT land pixels)")
n_test = SM_test_aws.shape[2]
P_sm2rain_ga = np.full((lat_len, lon_len, n_test), np.nan)

for i in tqdm(range(lat_len), desc="  latitude rows"):
    for j in range(lon_len):
        if not apply_mask[i, j]:
            continue
        P_sm2rain_ga[i, j, :] = sm2rain_ga(
            SM_test_aws[i, j, :],
            A_f[i, j],
            C_f[i, j],
            Z_f[i, j],
            KS_f[i, j],
            PSI_f[i, j],
        )

n_valid_pix = int(np.sum(np.any(np.isfinite(P_sm2rain_ga), axis=2)))
print(f"  valid estimates: {np.isfinite(P_sm2rain_ga).sum()}  (유효 픽셀 {n_valid_pix})")
print(f"  P range: {np.nanmin(P_sm2rain_ga):.2f} ~ {np.nanmax(P_sm2rain_ga):.2f} mm/day")

print(f"\nWriting NetCDF -> {OUTPUT_PATH}")
test_dates = pd.DatetimeIndex(ascat_time[test_idx])
ds_out = xr.Dataset(
    data_vars={
        "precipitation": xr.DataArray(
            np.transpose(P_sm2rain_ga, (2, 0, 1)),
            dims=("time", "lat", "lon"),
            attrs={
                "long_name": "SM2RAIN-GA estimated precipitation",
                "units": "mm/day",
                "description": (
                    "Pixel-wise SM2RAIN-GreenAmpt from ASCAT normalized SM. "
                    "Calibrated on 2021 IDW_AWS (~643 px) and applied to all ASCAT "
                    "land pixels via nearest-neighbor parameter transfer (2022-2025)."
                ),
            },
        ),
        "A": xr.DataArray(A_f, dims=("lat", "lon"), attrs={"description": "GA flux multiplier (calibrated+transferred)"}),
        "C": xr.DataArray(C_f, dims=("lat", "lon"), attrs={"description": "runoff/saturation exponent (calibrated+transferred)"}),
        "Z": xr.DataArray(Z_f, dims=("lat", "lon"), attrs={"units": "mm", "description": "effective soil depth (calibrated+transferred)"}),
        "KS": xr.DataArray(KS_f, dims=("lat", "lon"), attrs={"units": "mm/day", "description": "effective saturated hydraulic conductivity (calibrated+transferred)"}),
        "PSI_F": xr.DataArray(PSI_f, dims=("lat", "lon"), attrs={"units": "mm", "description": "effective wetting-front suction head (calibrated+transferred)"}),
        "calib_flag": xr.DataArray(calib_flag, dims=("lat", "lon"),
            attrs={"description": "parameter source: 1=calibrated (IDW_AWS), 2=transferred (nearest-neighbor), 0=none",
                   "flag_values": "0,1,2"}),
    },
    coords={
        "time": xr.DataArray(test_dates.values, dims="time", attrs={"long_name": "time"}),
        "lat": xr.DataArray(aws_lat_1d, dims="lat", attrs={"long_name": "latitude", "units": "degrees_north"}),
        "lon": xr.DataArray(aws_lon_1d, dims="lon", attrs={"long_name": "longitude", "units": "degrees_east"}),
    },
    attrs={
        "title": "SM2RAIN-GreenAmpt Precipitation",
        "calibration_period": str(CAL_YEAR),
        "test_period": f"{test_dates[0].date()} ~ {test_dates[-1].date()}",
        "source_SM": "ASCAT sm_volumetric normalized to 0-1 per pixel",
        "source_P_cal": "IDW_AWS",
        "coverage_extension": "calibrated on IDW_AWS pixels; parameters nearest-neighbor transferred to all ASCAT land pixels for application",
        "time_zone": "KST (UTC+9); daily boundary 00-24 KST, aligned with IDW_AWS",
        "AWS_flip_applied": best_key,
        "model_equation": "P = (Z*dtheta/dt + A*f_GA)/(1 - theta**C)",
        "green_ampt_flux": "f_GA = KS*(1 + PSI_F*delta_theta/F_eff)",
        "parameter_order": "A, C, Z, KS, PSI_F",
        "bounds": str(BOUNDS),
    },
)

ds_out.to_netcdf(OUTPUT_PATH)

print("  done")
print(f"  precipitation shape: {ds_out['precipitation'].shape} (time, lat, lon)")
print(f"  time: {test_dates[0].date()} ~ {test_dates[-1].date()}")
print(f"  lat : {aws_lat_1d[0]:.2f} ~ {aws_lat_1d[-1]:.2f} ({len(aws_lat_1d)})")
print(f"  lon : {aws_lon_1d[0]:.2f} ~ {aws_lon_1d[-1]:.2f} ({len(aws_lon_1d)})")
#%%