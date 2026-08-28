"""
ASCAT_interpolation.py
======================
ASCAT_daily_stack.nc 의 시간에 따라 생기는 NaN만 채우는 코드.

- 항상 NaN인 픽셀 (해양/경계 고정 마스크) → 건드리지 않음
- 날짜별로 생기는 NaN만 → 시간 보간 → 공간 보간 순서로 채움

출력: ASCAT_daily_stack_interp.nc
"""

import numpy as np
import netCDF4 as nc
from scipy.interpolate import griddata

# ============================================================
# 경로 설정
# ============================================================
# KST 버전: layer_KST.py 가 만든 KST 일경계 stack 을 보간한다.
INPUT_FILE  = '/Users/kim/cpuserver_data/personal_data/project_KIHS/data/layer/ASCAT_daily_stack_KST.nc'
OUTPUT_FILE = '/Users/kim/cpuserver_data/personal_data/project_KIHS/data/layer/ASCAT_daily_stack_interp_KST.nc'

DATA_VARS = ['sm', 'degree_of_saturation', 'sm_volumetric', 'optimum_water_content']

# ============================================================
# 데이터 로딩
# ============================================================
print("파일 읽는 중...")
ds_in = nc.Dataset(INPUT_FILE, 'r')

T = ds_in.dimensions['time'].size
Y = ds_in.dimensions['y'].size
X = ds_in.dimensions['x'].size

lat_2d = ds_in.variables['lat'][:]   # (58, 46)
lon_2d = ds_in.variables['lon'][:]

# 고정 마스크: 모든 변수에서 항상 NaN인 픽셀
print("고정 마스크 탐지 중...")
sm_raw = ds_in.variables['sm'][:]
if hasattr(sm_raw, 'data'):
    _tmp = sm_raw.data.copy()
    _tmp[sm_raw.mask] = np.nan
else:
    _tmp = np.array(sm_raw, dtype=float)

fixed_mask = np.all(np.isnan(_tmp), axis=0)   # (Y, X) - 항상 NaN인 픽셀
print(f"  고정 마스크 픽셀: {fixed_mask.sum()} / {Y*X}")
del _tmp

# ============================================================
# 보간 함수
# ============================================================
def time_interpolate(data_3d, fixed_mask):
    """
    시간 축 선형 보간.
    fixed_mask 픽셀은 건드리지 않음.
    """
    T, Y, X = data_3d.shape
    out = data_3d.copy()
    t_idx = np.arange(T, dtype=float)

    for y in range(Y):
        for x in range(X):
            if fixed_mask[y, x]:
                continue
            series = out[:, y, x]
            valid  = ~np.isnan(series)
            if valid.sum() < 2 or valid.all():
                continue
            out[:, y, x] = np.interp(t_idx, t_idx[valid], series[valid])

    return out


def spatial_interpolate(slice_2d, lat_2d, lon_2d, fixed_mask):
    """
    공간 보간 (시간 보간 후에도 남은 NaN 처리).
    fixed_mask 픽셀은 채우지 않음.
    """
    # 보간에 사용할 소스: fixed_mask도 아니고 NaN도 아닌 픽셀
    source = ~fixed_mask & ~np.isnan(slice_2d)
    if source.sum() < 4:
        return slice_2d

    points = np.column_stack([lat_2d[source], lon_2d[source]])
    values = slice_2d[source]

    # 채워야 할 픽셀: fixed_mask는 아닌데 NaN인 픽셀
    target = ~fixed_mask & np.isnan(slice_2d)
    if target.sum() == 0:
        return slice_2d

    xi = np.column_stack([lat_2d[target], lon_2d[target]])

    filled = griddata(points, values, xi, method='linear')

    # 볼록껍질 밖은 nearest로 마저 채움
    still_nan = np.isnan(filled)
    if still_nan.any():
        filled[still_nan] = griddata(points, values, xi[still_nan], method='nearest')

    out = slice_2d.copy()
    out[target] = filled
    return out


# ============================================================
# 출력 파일 생성
# ============================================================
ds_out = nc.Dataset(OUTPUT_FILE, 'w', format='NETCDF4')
for dname, dim in ds_in.dimensions.items():
    ds_out.createDimension(dname, len(dim))

for vname in ['lat', 'lon', 'time']:
    vin  = ds_in.variables[vname]
    vout = ds_out.createVariable(vname, vin.datatype, vin.dimensions)
    vout.setncatts({a: getattr(vin, a) for a in vin.ncattrs()})
    vout[:] = vin[:]

# ============================================================
# 변수별 보간
# ============================================================
for vname in DATA_VARS:
    print(f"\n[{vname}] 처리 중...")
    vin  = ds_in.variables[vname]

    # (time, y, x) 로드
    raw = vin[:]
    if hasattr(raw, 'data'):
        data = raw.data.copy()
        data[raw.mask] = np.nan
    else:
        data = np.array(raw, dtype=float)

    n_before = np.isnan(data).sum()
    n_temporal = np.isnan(data).sum() - np.sum(fixed_mask) * T  # 고정마스크 제외
    print(f"  결측 전: {n_before:,} px  (고정마스크 제외 시간성 결측: {(np.isnan(data) & ~fixed_mask[None,:,:]).sum():,})")

    # ① 시간 보간
    print("  [1/2] 시간 보간 중...")
    data = time_interpolate(data, fixed_mask)
    n_temporal_after = (np.isnan(data) & ~fixed_mask[None,:,:]).sum()
    print(f"  → 남은 시간성 결측: {n_temporal_after:,}")

    # ② 공간 보간 (시간 보간 후에도 남은 경우)
    if n_temporal_after > 0:
        print(f"  [2/2] 공간 보간 중...")
        for t in range(T):
            if t % 300 == 0:
                print(f"    step {t}/{T}...")
            if (~fixed_mask & np.isnan(data[t])).any():
                data[t] = spatial_interpolate(data[t], lat_2d, lon_2d, fixed_mask)
    else:
        print("  [2/2] 공간 보간 불필요")

    # 고정 마스크는 NaN 유지 확인
    n_fixed_remain = np.sum(np.isnan(data) & fixed_mask[None,:,:])
    n_temporal_final = (np.isnan(data) & ~fixed_mask[None,:,:]).sum()
    print(f"  최종 시간성 결측: {n_temporal_final} (목표: 0)")
    print(f"  고정 마스크 NaN 유지: {n_fixed_remain:,} (건드리지 않음 ✓)")

    # 저장
    vout = ds_out.createVariable(vname, 'f4', vin.dimensions,
                                  fill_value=np.float32(np.nan), zlib=True, complevel=4)
    vout.setncatts({a: getattr(vin, a) for a in vin.ncattrs() if a != '_FillValue'})
    vout[:] = data.astype(np.float32)

ds_in.close()
ds_out.close()
print(f"\n완료 → {OUTPUT_FILE}")
#%%