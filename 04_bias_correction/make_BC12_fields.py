"""
make_BC12_fields.py  —  BC_1 / BC_2 격자 예측 필드 생성 (non-LOPO, 빠른 적합)
=============================================================================
TCA_merging_clim.py 의 BC_1(Model1)·BC_2(Model2) 구성을 그대로 복제해
전역 LightGBM 을 2021 로 1회 적합 → 전기간(2021–2025) 격자 예측을 저장한다.
(LOPO 아님: 픽셀 제외 없이 전역 1회 적합이라 수 초~십수 초)

  features(BC_1, X1) : SM2RAIN, ERA5, GPM, TCA, lon, lat
  features(BC_2, X2) : X1 + AWS   (같은 날 AWS 입력 = gauge 융합)
  target             : AWS,  적합기간 = 2021

입력 : ds_merged_LR_TCA_SM2RAIN_ERA5_AWS_2021.nc  (또는 ds_merged_LR.nc)
산출 : BC12_fields_2021fit.nc   (BC_1, BC_2 : time,lat,lon)

실행: /Users/kim/miniconda3/bin/python3 code/use/TC/make_BC12_fields.py
"""
import os
import numpy as np
import pandas as pd
import xarray as xr

BASE   = '/Users/kim/cpuserver_data/personal_data/jaese/KIHS/output/260705'
IN_NC  = os.path.join(BASE, 'ds_merged_LR_TCA_SM2RAIN_ERA5_AWS_2021.nc')
# jaese 폴더는 읽기전용 → kim 소유 결과폴더에 저장
OUT_DIR = '/Users/kim/cpuserver_data/personal_data/project_KIHS/result/ASCAT/precipitation'
OUT_NC = os.path.join(OUT_DIR, 'BC12_fields_2021fit.nc')

BASE_FEATURES    = ['SM2RAIN', 'ERA5', 'GPM', 'TCA', 'lon', 'lat']
X1 = BASE_FEATURES
X2 = BASE_FEATURES + ['AWS']
POSITIVE_FEATURES = ['SM2RAIN', 'ERA5', 'GPM', 'TCA']
Y = 'AWS'
FIT_SEL = '2021'

from sklearn.ensemble import HistGradientBoostingRegressor
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

def make_bc_model(random_state, n_estimators=500):
    if LGBMRegressor is not None:
        return LGBMRegressor(
            n_estimators=n_estimators, learning_rate=0.03, num_leaves=31,
            min_child_samples=30, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, random_state=random_state, verbose=-1)
    return HistGradientBoostingRegressor(
        max_iter=n_estimators, learning_rate=0.03, max_leaf_nodes=31,
        l2_regularization=0.01, random_state=random_state)


print("로딩:", IN_NC)
ds = xr.open_dataset(IN_NC)
REFDIMS = ds['SM2RAIN'].dims                       # (time, lat, lon)
nt, nla, nlo = ds['SM2RAIN'].shape

def col(dss, name):
    """dss 의 name 변수를 SM2RAIN 격자로 broadcast → (time,lat,lon) flatten."""
    da = dss[name].broadcast_like(dss['SM2RAIN']).transpose(*REFDIMS)
    return da.data.astype('float64').flatten()

def build_df(dss):
    d = {c: col(dss, {'lon': 'LON', 'lat': 'LAT'}.get(c, c))
         for c in ['SM2RAIN', 'ERA5', 'GPM', 'TCA', 'AWS', 'lon', 'lat']}
    return pd.DataFrame(d)

def valid_ml_rows(df, features):
    cols = list(dict.fromkeys(features))
    valid = np.isfinite(df[cols]).all(axis=1)
    valid &= np.isfinite(df[POSITIVE_FEATURES]).all(axis=1)
    valid &= (df[POSITIVE_FEATURES] > 0).all(axis=1)
    return valid

def predict_grid(model, df, features, shape):
    pv = valid_ml_rows(df, features)
    pred = np.full(len(df), np.nan, dtype='float64')
    pred[pv.values] = np.maximum(model.predict(df.loc[pv, features]), 0.0)
    return pred.reshape(shape)

# ── 학습 (2021) ──
df_cal = build_df(ds.sel(time=FIT_SEL))
print(f"  적합기간 {FIT_SEL}: {len(df_cal):,} rows")
for name, X, rs in [('BC_1', X1, 42), ('BC_2', X2, 43)]:
    cal_valid = valid_ml_rows(df_cal, X) & np.isfinite(df_cal[Y]) & (df_cal[Y] > 0)
    globals()[f'm_{name}'] = make_bc_model(rs)
    globals()[f'm_{name}'].fit(df_cal.loc[cal_valid, X], df_cal.loc[cal_valid, Y])
    imp = dict(zip(X, np.round(globals()[f'm_{name}'].feature_importances_, 1)))
    print(f"  {name} 적합 (n={int(cal_valid.sum()):,})  변수중요도={imp}")

# ── 전기간 예측 ──
df_all = build_df(ds)
BC1 = predict_grid(m_BC_1, df_all, X1, (nt, nla, nlo))
BC2 = predict_grid(m_BC_2, df_all, X2, (nt, nla, nlo))
print(f"  BC_1 범위 {np.nanmin(BC1):.2f}~{np.nanmax(BC1):.2f} | BC_2 {np.nanmin(BC2):.2f}~{np.nanmax(BC2):.2f}")

out = xr.Dataset(
    {'BC_1': (REFDIMS, BC1), 'BC_2': (REFDIMS, BC2)},
    coords={'time': ds['time'].values, 'lat': ds['lat'].values, 'lon': ds['lon'].values},
    attrs={'title': 'Global LightGBM BC_1/BC_2 gridded fields (non-LOPO, fit 2021)',
           'BC_1_features': ', '.join(X1), 'BC_2_features': ', '.join(X2),
           'target': Y, 'note': 'BC_2 uses same-day AWS as input (fusion) — near-AWS by design'})
out['BC_1'].attrs = {'long_name': 'LightGBM BC without AWS input', 'units': 'mm/day'}
out['BC_2'].attrs = {'long_name': 'LightGBM BC with AWS input (fusion)', 'units': 'mm/day'}
ds.close()
if os.path.exists(OUT_NC):
    os.remove(OUT_NC)
out.to_netcdf(OUT_NC)
print("저장:", OUT_NC)
