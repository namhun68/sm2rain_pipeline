"""
export_BC_LR_csv.py — BC_LR(편향보정 강수)만 CSV 로 추출
================================================================================
입력  result/ASCAT/precipitation/BC_LR_AWS_KST.nc  의 BC_LR 변수
      (픽셀별 OLS: AWS ~ SM2RAIN+GPM+ERA5+TCA, 2021 적합 → 2021-2025 적용, KST)

출력  KIHS/DATA/
        BC_LR_KST_daily.csv     wide  행=날짜, 열=격자(lat_lon), 값=mm/day, 소수2자리
        BC_LR_grid_coords.csv   열이름 ↔ 위경도 대응표 (+ 유효일수)
        BC_LR_README.txt        산출 조건 요약

실행  python3 code/use/export_BC_LR_csv.py
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import xarray as xr

SRC = ("/Users/kim/cpuserver_data/personal_data/project_KIHS/result/ASCAT/"
       "precipitation/BC_LR_AWS_KST.nc")
OUT_DIR = "/Users/kim/Desktop/work/KIHS/DATA"
VAR = "BC_LR"
MIN_VALID_DAYS = 1          # 이보다 유효일이 적은 격자는 열에서 제외


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ds = xr.open_dataset(SRC)
    da = ds[VAR]
    T, NY, NX = da.shape
    lat = da.lat.values
    lon = da.lon.values
    times = pd.to_datetime(da.time.values)
    print(f"[입력] {os.path.basename(SRC)}  {VAR}  {T}일 × {NY}×{NX}")
    print(f"       {times[0]:%Y-%m-%d} ~ {times[-1]:%Y-%m-%d}")

    arr = da.values.reshape(T, NY * NX)
    valid_days = np.isfinite(arr).sum(axis=0)
    keep = valid_days >= MIN_VALID_DAYS
    print(f"       유효 격자 {keep.sum():,} / {NY*NX:,}")

    # 열 이름 = 위경도 (자기설명적). 좌표 대응표도 따로 저장한다.
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    lat_f, lon_f = LAT.ravel()[keep], LON.ravel()[keep]
    cols = [f"{a:.2f}_{o:.2f}" for a, o in zip(lat_f, lon_f)]

    df = pd.DataFrame(arr[:, keep], index=times.strftime("%Y-%m-%d"), columns=cols)
    df.index.name = "date"
    p_data = os.path.join(OUT_DIR, "BC_LR_KST_daily.csv")
    df.to_csv(p_data, float_format="%.2f")

    coords = pd.DataFrame({"column": cols, "lat": lat_f, "lon": lon_f,
                           "valid_days": valid_days[keep]})
    p_coord = os.path.join(OUT_DIR, "BC_LR_grid_coords.csv")
    coords.to_csv(p_coord, index=False, float_format="%.4f")

    finite = np.isfinite(arr[:, keep])
    readme = f"""BC_LR — 편향보정 일강수 (KIHS 다종자료 융합)
================================================================================
원본        {SRC}
변수        {VAR}  ({da.attrs.get('long_name', '')})
단위        mm/day
시간대      KST (UTC+9), 일 경계 00-24시
기간        {times[0]:%Y-%m-%d} ~ {times[-1]:%Y-%m-%d}  ({T}일)
격자        {NY} x {NX} @ 0.1도,  lat {lat.min():.1f}-{lat.max():.1f} / lon {lon.min():.1f}-{lon.max():.1f}
            이 중 값이 있는 {keep.sum()}개 격자만 열로 수록

산출 방법   픽셀별 최소자승 회귀 (per-pixel OLS)
              target     : AWS (IDW 보간)
              predictors : SM2RAIN, GPM, ERA5, TCA
              적합       : 2021년 / 적용 : 2021-2025
              평가       : ASOS (IDW 보간)

성능 (2022년 이후, 기준 ASOS)
              R 0.887 · ubRMSE 6.07 mm/day · bias +0.25 · KGE 0.767
              참고 - AWS 자체 R 0.919 (상한),  보정 전 TCA R 0.717

주의        여기 쓰인 TCA 는 AWS 를 삼중분석 멤버로 포함한다.
            학습 target 도 AWS 이므로 순환(leakage) 소지가 있고,
            R 0.887 은 그만큼 낙관적일 수 있다.
            AWS 를 멤버에서 뺀 판(BC_LR_KST.nc)의 성능은 R 0.844 이다.

파일        BC_LR_KST_daily.csv     행=날짜, 열=격자, 빈칸=결측
            BC_LR_grid_coords.csv   열이름 ↔ 위경도 대응 (+유효일수)
            열 이름 형식: "위도_경도"  예) 36.50_127.30

생성        {pd.Timestamp.now():%Y-%m-%d %H:%M}
"""
    p_readme = os.path.join(OUT_DIR, "BC_LR_README.txt")
    with open(p_readme, "w") as f:
        f.write(readme)

    print(f"\n[출력] {OUT_DIR}")
    for p in (p_data, p_coord, p_readme):
        print(f"  {os.path.basename(p):26s} {os.path.getsize(p)/1e6:7.2f} MB")
    print(f"\n  자료 {df.shape[0]:,}행 × {df.shape[1]:,}열")
    print(f"  유효값 {finite.sum():,} / {finite.size:,}  ({100*finite.mean():.1f}%)")
    print(f"  값 범위 {np.nanmin(arr):.2f} ~ {np.nanmax(arr):.2f} mm/day")


if __name__ == "__main__":
    main()
