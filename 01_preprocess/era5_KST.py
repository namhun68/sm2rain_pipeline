"""
era5_KST.py
===========
ERA5-Land 시간별 누계 강수(원본)를 누계 해제(de-accumulate) → KST(+9h) 변환
→ KST 하루(00~24시 KST) 경계로 재집계 → 연도별 일강수 nc 생성.

기존 era5_.py 가 tp[-1](23:00 UTC 누계)만 써서 'UTC 일경계'가 됐던 것과 달리,
이 코드는 시간별로 풀어 KST 일경계로 다시 합친다. (원본 파일은 건드리지 않음)

입력 : /Users/kim/data_2/ERA5_Land/Precipitation/{YYYY}/{YYYY.MM.DD}/ERA5_Land_P_{YYYYMMDD}.nc
        - valid_time: 24 (시간별, 00:00~23:00 UTC), tp: stepType=accum(누계), 단위 m
출력 : SAVE_DIR/era5_land{YYYY}_KST.nc   (lat, lon, time / 단위 mm/day, KST 일경계)
        SAVE_DIR/era5_land_P_KST.nc      (전체 병합)

핵심 절차
--------
1. 각 UTC 일 파일 → 누계 해제 → 시간별 강수 (ERA5-Land 00:00 함정 처리)
2. 각 시간값을 +9h 하여 KST 날짜에 배정 (KST 하루는 UTC 두 날에 걸침)
3. KST 날짜별로 합산 → KST 일강수
4. 연도별 저장
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

# ── 설정 ────────────────────────────────────────────────────────────────
ERA5_RAW_DIR = '/Users/kim/data_2/ERA5_Land/Precipitation'   # 시간별 누계 원본
SAVE_DIR     = '/Users/kim/cpuserver_data/personal_data/project_KIHS/data/era5_KST'
YEARS        = [2021, 2022, 2023, 2024, 2025]
KST_OFFSET_H = 9

# 한국 영역 subset (글로벌 1801x3600 → 한반도; 기존 일자료 격자에 맞춤)
LAT_MAX, LAT_MIN = 39.5, 32.5     # 글로벌 lat 은 내림차순(90→-90)
LON_MIN, LON_MAX = 123.5, 130.5   # 글로벌 lon 은 0~360

os.makedirs(SAVE_DIR, exist_ok=True)


# ── 한 UTC 일 파일 → 시간별 강수(mm) 배열 ─────────────────────────────────
def deaccumulate_day(nc_path):
    """
    반환: hourly(0~22시), lat, lon, ocean, cum23, art00
      hourly[H] = UTC 시각 (date + H시) 에 시작하는 1시간 강수 (H=0..22).
      ERA5-Land 00:00 스텝은 '전날 누계 꼬리'이므로 baseline 0 으로 처리.
      H=23(23~24시 UTC)은 '다음날 파일의 00:00 누계(=당일 총량) − cum23' 으로
      build 단계에서 정밀 계산한다.
      cum23 = 00:00→23:00 누계,  art00 = 이 날 00:00 값(=전날 총량).
    """
    with xr.open_dataset(nc_path) as ds:
        sub = ds.sel(latitude=slice(LAT_MAX, LAT_MIN),
                     longitude=slice(LON_MIN, LON_MAX))
        raw = sub['tp'].values.astype('float64') * 1000.0  # (24, ny, nx) m→mm, 누계
        lat = sub['latitude'].values
        lon = sub['longitude'].values

    if raw.shape[0] != 24:
        raise ValueError(f"시간축 길이가 24가 아님: {nc_path} ({raw.shape[0]})")

    # ERA5-Land 는 바다에서 전 시간 NaN → 육지/바다 마스크 (계산은 0으로 채워 진행)
    ocean = np.isnan(raw).all(axis=0)        # (ny, nx) True=바다(무효)
    raw = np.nan_to_num(raw, nan=0.0)

    art00 = raw[0].copy()              # 이 날 00:00 값 = 전날 총량
    tpc = raw.copy()
    tpc[0] = 0.0                       # 00:00 함정 → 당일 baseline 0

    hourly = np.zeros_like(tpc)        # (24, ny, nx)
    hourly[:23] = np.diff(tpc, axis=0) # H=0..22: tpc[H+1]-tpc[H]
    np.clip(hourly, 0.0, None, out=hourly)
    cum23 = tpc[23].copy()             # 00:00→23:00 누계
    return hourly, lat, lon, ocean, cum23, art00


# ── 일 파일 목록 (연속 KST 경계를 위해 직전 연말 하루 포함) ───────────────
def list_day_files(years, raw_dir):
    files = []
    need_years = sorted(set(years) | {min(years) - 1})  # 경계용 직전 해
    for yr in need_years:
        yr_dir = os.path.join(raw_dir, str(yr))
        if not os.path.isdir(yr_dir):
            continue
        for day_dir in sorted(os.listdir(yr_dir)):
            dpath = os.path.join(yr_dir, day_dir)
            if not os.path.isdir(dpath) or day_dir.startswith('.'):
                continue
            date_str = day_dir.replace('.', '')          # YYYYMMDD
            nc = os.path.join(dpath, f'ERA5_Land_P_{date_str}.nc')
            if os.path.exists(nc):
                try:
                    d = pd.Timestamp(date_str)
                except Exception:
                    continue
                # 직전 해는 12/31 하루만 (KST 1/1 경계 채우기용)
                if d.year == min(years) - 1 and not (d.month == 12 and d.day == 31):
                    continue
                files.append((d, nc))
    return sorted(files, key=lambda x: x[0])


# ── STEP 1. KST 일강수 생성 ──────────────────────────────────────────────
def build_kst_daily(years, raw_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    day_files = list_day_files(years, raw_dir)
    if not day_files:
        raise FileNotFoundError(f"입력 일 파일 없음: {raw_dir}")
    print(f"대상 일 파일: {len(day_files)}개 "
          f"({day_files[0][0].date()} ~ {day_files[-1][0].date()})")

    acc = {}              # kst_date(Timestamp) -> (ny,nx) 합산 강수
    hour_counts = {}      # kst_date(Timestamp) -> 합산된 1시간 값 개수
    lat_ref = lon_ref = None
    ocean_mask = None
    prev = None           # (직전 날짜, 직전 cum23) — 23~24시 정밀 계산용
    skipped_files = []     # 손상·읽기 오류 등으로 건너뛴 원본 파일

    def _add(kd, val):
        if kd not in acc:
            acc[kd] = np.zeros(val.shape, dtype='float64')
            hour_counts[kd] = 0
        acc[kd] += val
        hour_counts[kd] += 1

    for date, nc in tqdm(day_files, desc='de-accum + KST'):
        try:
            hourly, lat, lon, ocean, cum23, art00 = deaccumulate_day(nc)
        except Exception as e:
            tqdm.write(f"  [스킵] 원본 파일 읽기 실패: {nc} — {type(e).__name__}: {e}")
            skipped_files.append((date, nc, e))
            prev = None
            continue

        if lat_ref is None:
            lat_ref, lon_ref = lat, lon
            ocean_mask = ocean

        # ── 정밀화: 직전 날의 23~24시 UTC = (이 날 art00=전날총량) − (직전 cum23)
        if prev is not None:
            pdate, pcum23 = prev
            if (date - pdate) == pd.Timedelta(days=1):
                h23 = np.clip(art00 - pcum23, 0.0, None)
                kd = (pdate + pd.Timedelta(hours=23 + KST_OFFSET_H)).normalize()
                _add(kd, h23)

        # ── 이 날 0~22시 UTC
        for H in range(23):
            kd = (date + pd.Timedelta(hours=H + KST_OFFSET_H)).normalize()
            _add(kd, hourly[H])

        prev = (date, cum23)

    if skipped_files:
        print(f"  [경고] 읽기 실패 원본 파일 {len(skipped_files)}개를 건너뜀 "
              f"({skipped_files[0][0].date()} ~ {skipped_files[-1][0].date()})")

    # 요청 연도이면서 24시간이 모두 모인 날짜만 남긴다.
    # 입력 마지막 날 다음 KST 날짜나 결측 파일 주변의 partial 일자는 제외한다.
    candidates = sorted(d for d in acc if d.year in years)
    incomplete = [d for d in candidates if hour_counts[d] != 24]
    if incomplete:
        print(f"  [경고] 24시간 미만 partial KST 일자 {len(incomplete)}개 제외: "
              f"{incomplete[0].date()} ~ {incomplete[-1].date()}")
    dates = [d for d in candidates if hour_counts[d] == 24]
    if not dates:
        raise RuntimeError("24시간이 완전한 KST 일강수 날짜가 없습니다.")

    arr = np.stack([acc[d] for d in dates], axis=2)        # (ny, nx, time)
    time = pd.DatetimeIndex(dates)

    # 바다(무효) 픽셀은 NaN 으로 복원 (기존 era5 일자료와 동일하게)
    if ocean_mask is not None:
        arr[ocean_mask, :] = np.nan

    # lat 오름차순으로 통일 (원본은 내림차순)
    if lat_ref[0] > lat_ref[-1]:
        lat_ref = lat_ref[::-1]
        arr = arr[::-1, :, :]

    # 연도별 저장
    for yr in years:
        m = time.year == yr
        if not m.any():
            continue
        ds_out = xr.Dataset(
            {'tp': (['lat', 'lon', 'time'], arr[:, :, m])},
            coords={'lat': lat_ref, 'lon': lon_ref, 'time': time[m]},
        )
        ds_out['tp'].attrs['units'] = 'mm/day'
        ds_out['tp'].attrs['long_name'] = 'Total Precipitation (KST daily)'
        ds_out.attrs['time_zone'] = 'KST (UTC+9); daily boundary 00-24 KST'
        ds_out.attrs['method'] = 'hourly de-accumulation -> +9h -> KST daily sum'
        path = os.path.join(save_dir, f'era5_land{yr}_KST.nc')
        ds_out.to_netcdf(path)
        print(f"  저장: {path}  {arr[:, :, m].shape}  "
              f"({time[m][0].date()} ~ {time[m][-1].date()})")
    return arr, lat_ref, lon_ref, time


# ── STEP 2. 전체 병합 ────────────────────────────────────────────────────
def merge_yearly(years, save_dir):
    arrays, times = [], []
    lat_ref = lon_ref = None
    for yr in years:
        p = os.path.join(save_dir, f'era5_land{yr}_KST.nc')
        if not os.path.exists(p):
            continue
        with xr.open_dataset(p) as ds:
            lat = ds['lat'].values
            lon = ds['lon'].values
            if lat_ref is None:
                lat_ref, lon_ref = lat, lon
            elif not (np.array_equal(lat_ref, lat) and np.array_equal(lon_ref, lon)):
                raise ValueError(f"연도별 파일의 격자가 다름: {p}")
            arrays.append(ds['tp'].values)
            times.append(pd.DatetimeIndex(ds['time'].values))

    if not arrays:
        raise FileNotFoundError(f"병합할 연도별 KST 파일 없음: {save_dir}")

    full = np.concatenate(arrays, axis=2)
    t = times[0]
    for x in times[1:]:
        t = t.append(x)
    out = os.path.join(save_dir, 'era5_land_P_KST.nc')
    ds_out = xr.Dataset({'tp': (['lat', 'lon', 'time'], full)},
                        coords={'lat': lat_ref, 'lon': lon_ref, 'time': t})
    ds_out['tp'].attrs['units'] = 'mm/day'
    ds_out.attrs['time_zone'] = 'KST (UTC+9)'
    ds_out.to_netcdf(out)
    print(f"병합 저장: {out}  {full.shape}")
    return out


# ── 실행 ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("STEP 1. ERA5 누계해제 → KST 일강수")
    print("=" * 60)
    build_kst_daily(YEARS, ERA5_RAW_DIR, SAVE_DIR)

    print("=" * 60)
    print("STEP 2. 전체 병합")
    print("=" * 60)
    merge_yearly(YEARS, SAVE_DIR)
    print("\n완료. (다음: era5_.py 의 STEP 3 regrid 를 era5_land_P_KST.nc 로 돌리면 SM2RAIN 격자 정합)")
#%%
