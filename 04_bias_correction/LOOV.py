"""
LOOV.py  —  픽셀별 선형회귀 + 공간 Leave-One-Pixel-Out 교차검증 (KST, 2021–2025)
================================================================================
BC_LR.py 의 픽셀별 OLS(자기 픽셀 AWS로 적합)는 학습에 쓴 관측을 평가에도 써서
"자기 자신을 보고 맞춘" calibration 성격이 강하다(자기 픽셀 정보 누수).
LOOV 는 이를 공간 교차검증으로 바꾼다.

  1) BC_LR 와 동일하게 픽셀마다 독립 OLS 계수 β(i,j) = [a1..a4, b] 를 2021로 적합.
  2) 평가 대상 픽셀 (i,j) 의 계수는 그 픽셀 AWS 로 만든 β(i,j) 를 쓰지 않고,
     주변 4방향(상·하·좌·우) 유효 픽셀 계수의 평균 β_CV(i,j) 로 대체.
         β_CV_k(i,j) = (1/N) Σ_{(m,n)∈Ω(i,j)} β_k(m,n)
         Ω(i,j) = {(i-1,j),(i+1,j),(i,j-1),(i,j+1)} 중 계수가 유효한 픽셀, N=|Ω|
  3) 그 평균 계수를 target 픽셀의 SM2RAIN/ERA5/GPM/TCA 에 적용해 예측:
         P_CV(t,i,j) = β_CV_0 + β_CV_1·SM2RAIN + β_CV_2·GPM + β_CV_3·ERA5 + β_CV_4·TCA
  4) target 픽셀 AWS 는 β_CV 학습에 전혀 안 들어감 → 최종 평가에만 사용.
     "이 픽셀에 관측소가 없다고 가정하면 주변에서 학습한 보정관계가 얼마나 맞나?"
     ⇒ 일반적 픽셀별 회귀보다 공정한 spatial cross-validation.

성능표는 BC(자기계수, =BC_LR) 와 BC_CV(주변평균계수) 를 나란히 비교한다.

  예측변수  : SM2RAIN, GPM, ERA5, TCA   (모두 KST 일경계)
  학습 기준 : IDW_AWS
  독립 평가 : IDW_ASOS  (학습망 AWS ≠ 평가망 ASOS, Thiessen 미사용)
  분할      : cal 2021 / val 2022 / tst 2023–2025 (연도 홀드아웃)

입력 (전부 KST 일경계, 58×46):
  SM2RAIN  : result/SM2RAIN/SM2RAIN_KST.nc
  GPM      : data/gpm_KST/GPM_{yr}_KST.nc             (49×49 → 58×46 in-memory regrid)
  ERA5     : data/era5_KST/era5_land{yr}_KST.nc       (70×70 → 58×46 in-memory regrid)
  TCA      : result/ASCAT/precipitation/TCA/TC_KST_T.nc
  IDW_AWS/ASOS : /Users/kim/DAS/projects/KIHS/IDW/both/da_IDWs.nc ('AWS (IDW)'/'ASOS (IDW)')

* 남한 마스크(Korea.shp): 남한 밖 픽셀은 전부 NaN.

산출:
  result/ASCAT/precipitation/LOOV_KST.nc         (BC_LR + BC_CV + 입력 6변수)
  result/ASCAT/precipitation/LOOV_coef_KST.nc    (자기계수 + 주변평균계수 β_CV)

실행: /Users/kim/miniconda3/bin/python3 code/use/LOOV/LOOV.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

# ============================================================
# 경로 설정 (KST)
# ============================================================
KST_DIR   = '/Users/kim/cpuserver_data/personal_data/project_KIHS'
IDW_BOTH_PATH = '/Users/kim/DAS/projects/KIHS/IDW/both/da_IDWs.nc'   # 지상 IDW (AWS·ASOS 단일파일, KST)

SM2RAIN_PATH = os.path.join(KST_DIR, 'result/SM2RAIN/SM2RAIN_KST.nc')
GPM_KST_DIR  = os.path.join(KST_DIR, 'data/gpm_KST')                            # GPM_{yr}_KST.nc
ERA5_KST_DIR = os.path.join(KST_DIR, 'data/era5_KST')                           # era5_land{yr}_KST.nc
TCA_PATH     = os.path.join(KST_DIR, 'result/ASCAT/precipitation/TCA/TC_KST_T.nc')

OUT_DIR   = os.path.join(KST_DIR, 'result/ASCAT/precipitation')
OUT_BC_NC = os.path.join(OUT_DIR, 'LOOV_KST.nc')
OUT_COEF  = os.path.join(OUT_DIR, 'LOOV_coef_KST.nc')

YEARS    = [2021, 2022, 2023, 2024, 2025]
FEATURES = ['SM2RAIN', 'GPM', 'ERA5', 'TCA']
TARGET   = 'AWS'      # 학습 기준
EVAL     = 'ASOS'     # 독립 평가

# 학습은 2021만 (SM2RAIN·TCA 피팅과 통일: 2021 적합 → 2022–2025 적용)
CAL_SLICE = ('2021-01-01', '2021-12-31')   # 적합(fit)
VAL_SLICE = ('2022-01-01', '2022-12-31')   # 독립 검증(홀드아웃)
TST_SLICE = ('2023-01-01', '2025-12-31')   # 독립 시험(홀드아웃)

PREDICTORS = FEATURES     # 픽셀별 OLS 예측변수 (['TCA'] 로 두면 단일변수 회귀)
MIN_FIT    = 60           # 픽셀별 적합 최소 유효일수
MIN_NEIGH  = 1            # 공간 CV: 계수 평균에 필요한 최소 유효 이웃 픽셀 수 (상하좌우 중)


# ============================================================
# 로더 / 재격자 (BC_LR.py 와 동일 로직)
# ============================================================
def load_precip(path):
    """nc → arr(lat,lon,time), lat_1d, lon_1d, DatetimeIndex. 음수→NaN, 위/경도 오름차순."""
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
    ds.close()
    return arr, lat_1d, lon_1d, pd.DatetimeIndex(time_1d)


def _axis_matrix(src, tgt):
    src = np.asarray(src, dtype='float64'); tgt = np.asarray(tgt, dtype='float64')
    idx = np.clip(np.searchsorted(src, tgt) - 1, 0, len(src) - 2)
    x0, x1 = src[idx], src[idx + 1]
    w = np.clip((tgt - x0) / (x1 - x0), 0.0, 1.0)
    M = np.zeros((len(tgt), len(src)), dtype='float64')
    rows = np.arange(len(tgt)); M[rows, idx] = 1.0 - w; M[rows, idx + 1] = w
    return M


def _apply_bilinear(cube, Mlat, Mlon):
    s1 = np.tensordot(Mlat, cube, axes=([1], [0]))
    s2 = np.tensordot(Mlon, s1,   axes=([1], [1]))
    return s2.transpose(1, 0, 2)


def _regrid_bilinear(cube, src_lat, src_lon, tgt_lat, tgt_lon, min_cov=0.5):
    """NaN-aware bilinear regrid (nlat,nlon,T)→(TL,TO,T)."""
    Mlat = _axis_matrix(src_lat, tgt_lat); Mlon = _axis_matrix(src_lon, tgt_lon)
    valid = np.isfinite(cube).astype('float64')
    cube0 = np.nan_to_num(cube, nan=0.0)
    num = _apply_bilinear(cube0, Mlat, Mlon)
    den = _apply_bilinear(valid, Mlat, Mlon)
    return np.where(den > min_cov, num / np.where(den == 0, np.nan, den), np.nan)


def load_era5_kst_regrid(years, era5_dir, tgt_lat, tgt_lon):
    """era5_land{yr}_KST.nc(daily mm/day, 70×70) 로드·concat → SM2RAIN 격자 보간."""
    arrs, times = [], []
    src_lat = src_lon = None
    for yr in years:
        p = os.path.join(era5_dir, f'era5_land{yr}_KST.nc')
        if not os.path.exists(p):
            continue
        ds = xr.open_dataset(p)
        da = ds['tp']
        lat = ds['lat'].values.squeeze(); lon = ds['lon'].values.squeeze()
        dl = [d.lower() for d in da.dims]
        a = np.transpose(da.values.astype('float64'),
                         (dl.index('lat'), dl.index('lon'), dl.index('time')))
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
    """da_IDWs.nc 단일파일에서 varname_sub('aws'/'asos') 포함 변수를 (lat,lon,time) 오름차순으로 반환."""
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


def load_years(path_tpl, tag):
    """연도별 nc concat (동일 격자 가정: SM2RAIN 49×49)."""
    arrs, times = [], []
    lat = lon = None
    for yr in YEARS:
        p = path_tpl.format(yr=yr)
        if not os.path.exists(p):
            continue
        a, la, lo, t = load_precip(p)
        arrs.append(a); times.append(t); lat, lon = la, lo
    if not arrs:
        raise FileNotFoundError(f'{tag} 파일 없음: {path_tpl}')
    arr = np.concatenate(arrs, axis=2)
    t = times[0]
    for x in times[1:]:
        t = t.append(x)
    return arr, lat, lon, t


# ============================================================
# 평가지표 (all_process 스타일)
# ============================================================
def _fin(p, o):
    m = np.isfinite(p) & np.isfinite(o)
    return p[m], o[m]

def corr(p, o):
    p, o = _fin(p, o)
    return np.corrcoef(p, o)[0, 1] if p.size >= 2 else np.nan

def rmse(p, o):
    p, o = _fin(p, o)
    return np.sqrt(np.mean((p - o) ** 2)) if p.size else np.nan

def ubrmse(p, o):
    p, o = _fin(p, o)
    return np.sqrt(np.mean(((p - p.mean()) - (o - o.mean())) ** 2)) if p.size else np.nan

def bias(p, o):
    p, o = _fin(p, o)
    return np.mean(p - o) if p.size else np.nan


# ============================================================
# 1. 데이터 로딩 + 정합
# ============================================================
print("=" * 68)
print("LOOV.py  |  픽셀별 선형회귀 + 공간 Leave-One-Pixel-Out CV (KST, 2021–2025)")
print("=" * 68)

print("[ SM2RAIN (KST) — 목표격자 정의 ]")
arr_sm, lat, lon, t_sm = load_precip(SM2RAIN_PATH)
n_lat, n_lon = lat.shape[0], lon.shape[0]
print(f"  SM2RAIN : {arr_sm.shape}  {t_sm[0].date()} ~ {t_sm[-1].date()}")

print("[ GPM (KST) ]")
arr_gpm, la_g, lo_g, t_gpm = load_years(os.path.join(GPM_KST_DIR, 'GPM_{yr}_KST.nc'), 'GPM')
if arr_gpm.shape[:2] != arr_sm.shape[:2] or not (
        np.allclose(lat, la_g, atol=0.01) and np.allclose(lon, lo_g, atol=0.01)):
    print(f"  GPM 재격자: {arr_gpm.shape[:2]} → {arr_sm.shape[:2]} (SM2RAIN 격자)")
    arr_gpm = _regrid_bilinear(arr_gpm, la_g, lo_g, np.sort(lat), np.sort(lon))
    arr_gpm[arr_gpm < 0] = np.nan
    la_g, lo_g = np.sort(lat), np.sort(lon)
print(f"  GPM     : {arr_gpm.shape}  {t_gpm[0].date()} ~ {t_gpm[-1].date()}")

print("[ ERA5 (KST, 70×70 → SM2RAIN 격자 regrid) ]")
arr_era5, la_e, lo_e, t_era5 = load_era5_kst_regrid(YEARS, ERA5_KST_DIR, lat, lon)
print(f"  ERA5    : {arr_era5.shape}  {t_era5[0].date()} ~ {t_era5[-1].date()}")

print("[ TCA (KST) ]")
arr_tca, la_t, lo_t, t_tca = load_precip(TCA_PATH)
print(f"  TCA     : {arr_tca.shape}  {t_tca[0].date()} ~ {t_tca[-1].date()}")

print("[ IDW_AWS / IDW_ASOS (da_IDWs.nc 단일파일, KST) ]")
arr_aws,  la_a, lo_a, t_aws  = load_idw_both('aws')
arr_asos, la_s, lo_s, t_asos = load_idw_both('asos')
print(f"  AWS     : {arr_aws.shape}  {t_aws[0].date()} ~ {t_aws[-1].date()}")
print(f"  ASOS    : {arr_asos.shape}  {t_asos[0].date()} ~ {t_asos[-1].date()}")

for nm, (la, lo) in {'GPM': (la_g, lo_g), 'ERA5': (la_e, lo_e), 'TCA': (la_t, lo_t),
                     'AWS': (la_a, lo_a), 'ASOS': (la_s, lo_s)}.items():
    assert np.allclose(lat, la, atol=0.01) and np.allclose(lon, lo, atol=0.01), \
        f"격자 불일치: SM2RAIN vs {nm}"

# ── 남한(South Korea) 마스크 ──
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
        SK_MASK[_i, _j] = _sk_prep.contains(_Point(lon[_j], lat[_i]))
print(f"  남한 마스크: {int(SK_MASK.sum())} / {n_lat * n_lon} 픽셀 (남한 밖 NaN)")

for _a in (arr_sm, arr_gpm, arr_era5, arr_tca, arr_aws, arr_asos):
    _a[~SK_MASK, :] = np.nan

# ── 공통 기간 교집합 ──
common = (t_sm.intersection(t_gpm).intersection(t_era5)
             .intersection(t_tca).intersection(t_aws).intersection(t_asos)).sort_values()
print(f"\n  공통 기간 : {common[0].date()} ~ {common[-1].date()}  ({len(common)}일)")

sl = lambda arr, t: arr[:, :, t.isin(common)]
X = {'SM2RAIN': sl(arr_sm, t_sm), 'GPM': sl(arr_gpm, t_gpm), 'ERA5': sl(arr_era5, t_era5),
     'TCA': sl(arr_tca, t_tca), 'AWS': sl(arr_aws, t_aws), 'ASOS': sl(arr_asos, t_asos)}
T = len(common)

ds_all = xr.Dataset(
    {k: (('time', 'lat', 'lon'), np.transpose(v, (2, 0, 1))) for k, v in X.items()},
    coords={'time': common.values,
            'lat':  xr.DataArray(lat, dims='lat', attrs={'units': 'degrees_north'}),
            'lon':  xr.DataArray(lon, dims='lon', attrs={'units': 'degrees_east'})})


# ============================================================
# 2. 분할 정의 (연도 홀드아웃: 학습=2021 → 2022–2025 적용)
# ============================================================
cal_t = (common >= pd.Timestamp(CAL_SLICE[0])) & (common <= pd.Timestamp(CAL_SLICE[1]))
print("\n" + "=" * 68)
print(f"분할  cal {CAL_SLICE[0]}~{CAL_SLICE[1]} ({int(cal_t.sum())}일) → 2022–2025 적용")
print("=" * 68)


# ============================================================
# 3. 픽셀별 선형회귀 (2021 학습) — BC_LR 과 동일
#    각 픽셀마다 독립 OLS:  AWS ≈ a1·SM2RAIN + a2·GPM + a3·ERA5 + a4·TCA + b
# ============================================================
print("\n" + "=" * 68)
print(f"[1] 픽셀별 선형회귀 적합 (예측변수={PREDICTORS}, target={TARGET})")
print("=" * 68)

nP = len(PREDICTORS)
Xcube = np.stack([X[p] for p in PREDICTORS], axis=-1)   # (lat, lon, T, nP)
Yaws  = X[TARGET]                                        # (lat, lon, T)

coef = np.full((n_lat, n_lon, nP + 1), np.nan)          # [a1..a_nP, b(절편)]
nfit = 0
for i in range(n_lat):
    for j in range(n_lon):
        xt = Xcube[i, j, cal_t, :]                      # (ncal, nP)
        yt = Yaws[i, j, cal_t]                          # (ncal,)
        m = np.isfinite(yt) & np.isfinite(xt).all(axis=1)
        if m.sum() < MIN_FIT:
            continue
        A = np.column_stack([xt[m], np.ones(m.sum())])
        beta, *_ = np.linalg.lstsq(A, yt[m], rcond=None)
        coef[i, j] = beta
        nfit += 1
print(f"  OLS 적합 픽셀: {nfit} / {n_lat * n_lon}  (픽셀당 최소 {MIN_FIT}일)")


# ============================================================
# 4. 공간 Leave-One-Pixel-Out: 주변 4픽셀 계수 평균 β_CV
#    β_CV_k(i,j) = mean of β_k over 상·하·좌·우 유효 이웃  (자기 픽셀 계수 제외)
# ============================================================
print("\n" + "=" * 68)
print(f"[2] 공간 LOOV 계수 = 상·하·좌·우 이웃 계수 평균 (최소 이웃 {MIN_NEIGH}개)")
print("=" * 68)

NEIGH = [(-1, 0), (1, 0), (0, -1), (0, 1)]     # 상·하·좌·우
coef_cv = np.full_like(coef, np.nan)           # (n_lat, n_lon, nP+1)
n_nbr   = np.zeros((n_lat, n_lon), dtype=int)   # 픽셀별 사용된 유효 이웃 수
ncv = 0
for i in range(n_lat):
    for j in range(n_lon):
        stack = []
        for di, dj in NEIGH:
            ii, jj = i + di, j + dj
            if 0 <= ii < n_lat and 0 <= jj < n_lon and np.isfinite(coef[ii, jj]).all():
                stack.append(coef[ii, jj])
        n_nbr[i, j] = len(stack)
        if len(stack) >= MIN_NEIGH:
            coef_cv[i, j] = np.mean(np.stack(stack, axis=0), axis=0)
            ncv += 1
print(f"  LOOV 계수 산출 픽셀: {ncv} / {n_lat * n_lon}")
_vn = n_nbr[np.isfinite(coef_cv).all(axis=2)]
if _vn.size:
    print(f"  픽셀당 유효 이웃 수: 평균 {_vn.mean():.2f}, 최소 {_vn.min()}, 최대 {_vn.max()}")


# ============================================================
# 5. 전 공통기간 복원 (자기계수 BC_LR + 주변평균계수 BC_CV, 음수→0)
# ============================================================
print("\n" + "=" * 68)
print("[3] 전 공통기간 복원 (BC_LR: 자기계수 / BC_CV: 주변평균계수)")
print("=" * 68)

def apply_coef(coef_map):
    out = np.full((n_lat, n_lon, T), np.nan)
    for i in range(n_lat):
        for j in range(n_lon):
            b = coef_map[i, j]
            if not np.isfinite(b).all():
                continue
            xt = Xcube[i, j, :, :]                       # (T, nP)
            m = np.isfinite(xt).all(axis=1)
            if m.sum() == 0:
                continue
            out[i, j, m] = np.clip(xt[m] @ b[:nP] + b[nP], 0, None)
    return out

BC_map = apply_coef(coef)        # BC_LR (자기계수)
CV_map = apply_coef(coef_cv)     # BC_CV  (주변평균계수, spatial LOOV)
print(f"  BC_LR 범위: {np.nanmin(BC_map):.3f} ~ {np.nanmax(BC_map):.3f} mm/day")
print(f"  BC_CV 범위: {np.nanmin(CV_map):.3f} ~ {np.nanmax(CV_map):.3f} mm/day")

ds_all['BC_LR'] = (('time', 'lat', 'lon'), np.transpose(BC_map, (2, 0, 1)))
ds_all['BC_CV'] = (('time', 'lat', 'lon'), np.transpose(CV_map, (2, 0, 1)))


# ============================================================
# 6. 성능표 (cal/val/tst × {vs AWS, vs ASOS}) — BC_LR vs BC_CV 비교
# ============================================================
def flat(ds_sel):
    return pd.DataFrame({k: ds_sel[k].values.ravel()
                         for k in FEATURES + [TARGET, EVAL, 'BC_LR', 'BC_CV']})

def show(split, df, ref):
    tag = '종속(학습기준)' if ref == TARGET else '독립(평가망)'
    print(f"\n── {split.upper()}  vs {ref}  ({tag}) ──")
    print(f"  {'method':10s} {'R':>6} {'RMSE':>7} {'ubRMSE':>7} {'Bias':>7}")
    for mth in ['SM2RAIN', 'GPM', 'ERA5', 'TCA', 'BC_LR', 'BC_CV']:
        p, o = df[mth].values, df[ref].values
        print(f"  {mth:10s} {corr(p, o):>6.3f} {rmse(p, o):>7.2f} "
              f"{ubrmse(p, o):>7.2f} {bias(p, o):>7.2f}")

print("\n" + "=" * 68)
print("성능 요약  (BC_LR = 자기계수 calibration / BC_CV = 공간 LOOV)")
print("=" * 68)
for sname, sl_ in [('cal', CAL_SLICE), ('val', VAL_SLICE), ('tst', TST_SLICE)]:
    df = flat(ds_all.sel(time=slice(*sl_))).dropna(subset=FEATURES + [TARGET])
    print(f"\n[{sname}] {sl_[0]}~{sl_[1]}  유효표본 {len(df):,}")
    for ref in [TARGET, EVAL]:
        show(sname, df, ref)


# ============================================================
# 7. 저장 (BC nc + 자기계수 + 주변평균계수맵)
# ============================================================
ds_all['BC_LR'].attrs = {
    'long_name': 'Per-pixel linear-regression bias-corrected precipitation (self coefficients)',
    'units': 'mm/day', 'predictors': ', '.join(PREDICTORS), 'target': TARGET,
    'method': 'per-pixel OLS (AWS ~ predictors), fit 2021 → applied 2021-2025'}
ds_all['BC_CV'].attrs = {
    'long_name': 'Spatial leave-one-pixel-out CV bias-corrected precipitation (neighbor-averaged coefficients)',
    'units': 'mm/day', 'predictors': ', '.join(PREDICTORS), 'target': TARGET,
    'method': 'coefficients averaged over up/down/left/right neighbors (target pixel AWS excluded from fit)',
    'min_neighbors': MIN_NEIGH}
ds_all.attrs = {'title': 'Per-pixel LR + spatial LOOV Bias-Corrected Precipitation (KST)',
                'time_zone': 'KST (UTC+9)', 'predictors': ', '.join(PREDICTORS),
                'target': TARGET, 'eval': EVAL,
                'note': 'BC_LR=self-coef (calibration); BC_CV=neighbor-averaged coef (spatial cross-validation)'}

coef_ds = xr.Dataset(
    {f'coef_{p}': (('lat', 'lon'), coef[:, :, k]) for k, p in enumerate(PREDICTORS)},
    coords={'lat': lat, 'lon': lon})
coef_ds['intercept'] = (('lat', 'lon'), coef[:, :, nP])
for k, p in enumerate(PREDICTORS):
    coef_ds[f'coefCV_{p}'] = (('lat', 'lon'), coef_cv[:, :, k])
coef_ds['interceptCV'] = (('lat', 'lon'), coef_cv[:, :, nP])
coef_ds['n_neighbors'] = (('lat', 'lon'), n_nbr)

os.makedirs(OUT_DIR, exist_ok=True)
for _f in (OUT_BC_NC, OUT_COEF):
    if os.path.exists(_f):
        os.remove(_f)
ds_all.to_netcdf(OUT_BC_NC)
coef_ds.to_netcdf(OUT_COEF)
print(f"\n  저장: {OUT_BC_NC}")
print(f"  계수맵: {OUT_COEF}")
print("\n완료.")
#%%
