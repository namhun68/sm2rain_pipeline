"""
export_bcg_grid.py — BC-G 격자 CSV (원자료 + 결측보간본) 생성
================================================================================
    python3 export_bcg_grid.py

출력  최종산출물/01_격자/
        BCG_grid_daily.csv        원자료 그대로 (결측 = 빈칸)
        BCG_grid_daily_filled.csv 결측을 보간해 채운 판
        BCG_grid_coords.csv       열이름 ↔ 위경도 대응 + 유효일수
        BCG_grid_fillflag.csv     칸별 출처 (0=원자료, 1=보간, 빈칸=끝내 결측)
        README.txt

보간 방법 — GPM 안내 비율 보간(ratio-guided spatial interpolation)
  BC-G 의 결측은 무작위가 아니다. 결측 격자는 유효 격자보다 계통적으로 건조하다
  (같은 날 ASOS 평균 2.48 vs 6.41 mm/day). 따라서 유효 격자 값을 그대로 공간
  보간해 끌어오면 건조한 칸을 젖은 값으로 채워 크게 과대추정된다.

  대신 결측 칸의 '비가 왔는지'는 결측이 없는 GPM 이 알고 있으므로, GPM 을
  공간 안내자로 삼아 비율만 보간한다.

      1) 같은 날 유효 칸에서 비율      r_k = BCG_k / GPM_k     (GPM_k > 0.1 인 칸)
      2) r 을 결측 칸으로 IDW 보간     r̂_ij  (거리 가중 p=2, 반경 150 km, 최소 3개)
      3) 채움                          BCG_ij = r̂_ij × GPM_ij
      4) GPM_ij ≤ 0.1 이면 0 으로 채운다 (위성이 무강수로 보는 칸)

  비율은 강수량 자체보다 공간적으로 완만해 보간이 안정적이고, 무엇보다
  결측 칸의 건·습 정보를 GPM 이 제공하므로 결측 편향이 상당 부분 상쇄된다.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if '__file__' in globals() else '/Users/kim/Desktop/work/code/use')
import basin_eval_core as B

OUT_DIR = '/Users/kim/Desktop/work/KIHS/DATA/최종산출물/01_격자'
RADIUS_KM = 150.0
POWER = 2.0
MIN_NB = 3
GPM_DRY = 0.1          # 이 이하면 위성이 무강수로 본다


def build_grid(lat, lon):
    LAT, LON = np.meshgrid(lat, lon, indexing='ij')
    return LAT.ravel(), LON.ravel()


def fill_day(bcg, gpm, X, Y, valid):
    """하루치 한 장을 채운다. bcg/gpm 은 1차원(격자) 배열."""
    out = bcg.copy()
    miss = ~np.isfinite(bcg)
    if not miss.any():
        return out, np.zeros(len(bcg), bool)

    filled = np.zeros(len(bcg), bool)
    # GPM 이 무강수로 보는 칸은 0 으로
    dry = miss & np.isfinite(gpm) & (gpm <= GPM_DRY)
    out[dry] = 0.0
    filled |= dry

    todo = miss & ~dry & np.isfinite(gpm)
    src = valid & np.isfinite(gpm) & (gpm > GPM_DRY)
    if not todo.any() or src.sum() < MIN_NB:
        return out, filled

    r = np.clip(bcg[src] / np.maximum(gpm[src], GPM_DRY), 0.0, 5.0)
    sx, sy = X[src], Y[src]
    for k in np.where(todo)[0]:
        d = np.hypot((X[k] - sx) * 88.0, (Y[k] - sy) * 111.0)   # 대략 km
        m = d <= RADIUS_KM
        if m.sum() < MIN_NB:
            idx = np.argsort(d)[:MIN_NB]
            m = np.zeros(len(d), bool)
            m[idx] = True
        w = 1.0 / np.maximum(d[m], 1.0) ** POWER
        out[k] = float((w * r[m]).sum() / w.sum()) * gpm[k]
        filled[k] = True
    return out, filled


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    ds = xr.open_dataset(B.NC)
    d2 = xr.open_dataset(B.NC_BC12)
    lat, lon = ds.lat.values, ds.lon.values
    times = pd.to_datetime(ds.time.values)
    G = d2['BC_2'].values.reshape(len(times), -1)
    P = ds['GPM'].values.reshape(len(times), -1)
    A = ds['ASOS'].values.reshape(len(times), -1)
    d2.close(); ds.close()

    X, Y = build_grid(lat, lon)
    keep = np.isfinite(G).sum(axis=0) > 0        # BC-G 가 한 번이라도 있는 칸
    print(f'격자 {len(lat)}×{len(lon)} = {G.shape[1]:,}칸 중 대상 {keep.sum():,}칸')
    print(f'기간 {times[0]:%Y-%m-%d} ~ {times[-1]:%Y-%m-%d} ({len(times):,}일)')

    Gk, Pk, Ak, Xk, Yk = G[:, keep], P[:, keep], A[:, keep], X[keep], Y[keep]
    F = np.full_like(Gk, np.nan)
    FL = np.zeros(Gk.shape, bool)
    for t in range(len(times)):
        v = np.isfinite(Gk[t])
        F[t], FL[t] = fill_day(Gk[t], Pk[t], Xk, Yk, v)
        if (t + 1) % 400 == 0:
            print(f'  {t+1}/{len(times)}일 ({time.time()-t0:.0f}초)')

    # ---------------- 검증: 유효 칸을 일부러 가리고 복원해 본다 ----------------
    rng = np.random.default_rng(0)
    ho_true, ho_fill = [], []
    for t in rng.choice(len(times), 200, replace=False):
        v = np.where(np.isfinite(Gk[t]))[0]
        if len(v) < 60:
            continue
        hide = rng.choice(v, size=max(1, len(v) // 5), replace=False)
        g = Gk[t].copy()
        g[hide] = np.nan
        f, _ = fill_day(g, Pk[t], Xk, Yk, np.isfinite(g))
        ok = np.isfinite(f[hide]) & np.isfinite(Gk[t][hide])
        ho_true.append(Gk[t][hide][ok])
        ho_fill.append(f[hide][ok])
    ho_true = np.concatenate(ho_true); ho_fill = np.concatenate(ho_fill)
    vr = float(np.corrcoef(ho_fill, ho_true)[0, 1])
    vb = float(ho_fill.mean() - ho_true.mean())
    vrm = float(np.sqrt(((ho_fill - ho_true) ** 2).mean()))
    print(f'\n[교차검증] 유효 칸 20%를 가리고 복원: n={len(ho_true):,} '
          f'R={vr:.3f} bias={vb:+.2f} RMSE={vrm:.2f} mm/day')

    # ---------------- ASOS 대비 ----------------
    def cmp(M, mask):
        m = mask & np.isfinite(M) & np.isfinite(Ak)
        s, o = M[m], Ak[m]
        return (float(np.corrcoef(s, o)[0, 1]), float(s.mean() - o.mean()),
                float(np.sqrt(((s - o) ** 2).mean())), int(m.sum()))
    orig_mask = np.isfinite(Gk)
    r0 = cmp(Gk, orig_mask)
    r1 = cmp(F, FL)                  # 보간으로 채운 칸만
    r2 = cmp(F, np.isfinite(F))      # 채운 판 전체
    print(f'[ASOS 대비] 원자료 칸      R={r0[0]:.3f} bias={r0[1]:+.2f} RMSE={r0[2]:.2f} (n={r0[3]:,})')
    print(f'            보간한 칸      R={r1[0]:.3f} bias={r1[1]:+.2f} RMSE={r1[2]:.2f} (n={r1[3]:,})')
    print(f'            채운 판 전체   R={r2[0]:.3f} bias={r2[1]:+.2f} RMSE={r2[2]:.2f} (n={r2[3]:,})')

    # ---------------- 저장 ----------------
    cols = [f'{a:.2f}_{o:.2f}' for a, o in zip(Yk, Xk)]
    idx = times.strftime('%Y-%m-%d')
    p1 = os.path.join(OUT_DIR, 'BCG_grid_daily.csv')
    pd.DataFrame(Gk, index=idx, columns=cols).rename_axis('date') \
        .to_csv(p1, float_format='%.2f')
    p2 = os.path.join(OUT_DIR, 'BCG_grid_daily_filled.csv')
    pd.DataFrame(F, index=idx, columns=cols).rename_axis('date') \
        .to_csv(p2, float_format='%.2f')
    p3 = os.path.join(OUT_DIR, 'BCG_grid_fillflag.csv')
    flag = np.where(np.isfinite(F), FL.astype(int), np.nan)
    pd.DataFrame(flag, index=idx, columns=cols).rename_axis('date') \
        .to_csv(p3, float_format='%.0f')
    p4 = os.path.join(OUT_DIR, 'BCG_grid_coords.csv')
    pd.DataFrame({'column': cols, 'lat': Yk, 'lon': Xk,
                  'valid_days_원자료': np.isfinite(Gk).sum(axis=0),
                  'valid_days_보간후': np.isfinite(F).sum(axis=0)}) \
        .to_csv(p4, index=False, float_format='%.4f')

    stats = dict(ncell=int(keep.sum()), T=len(times),
                 v0=float(np.isfinite(Gk).mean()), v1=float(np.isfinite(F).mean()),
                 nfill=int(FL.sum()), ndry=int((FL & (F == 0)).sum()),
                 vr=vr, vb=vb, vrm=vrm, r0=r0, r1=r1, r2=r2,
                 fully_missing=int((~np.isfinite(F)).all(axis=1).sum()))
    write_readme(times, stats)

    print(f'\n출력 {OUT_DIR}  ({time.time()-t0:.0f}초)')
    for f in sorted(os.listdir(OUT_DIR)):
        print(f'  {f:30s} {os.path.getsize(os.path.join(OUT_DIR,f))/1e6:7.2f} MB')
    return 0


def write_readme(times, s):
    txt = f"""BC-G 격자 단위 일강수 자료 (원자료 + 결측보간본)
================================================================================
다종자료 융합 강우장의 최종 산출물 BC-G 를 격자 단위로 정리한 자료.
유역 단위 자료(최종산출물/02_표준유역/)와 같은 원본에서 만들었으며,
이쪽은 공간 집계 없이 0.1° 격자 그대로다.

생성  {pd.Timestamp.now():%Y-%m-%d %H:%M}
기간  {times[0]:%Y-%m-%d} ~ {times[-1]:%Y-%m-%d}  ({s['T']:,}일)
격자  0.1° 정방, {s['ncell']:,}칸  (BC-G 가 한 번이라도 산출된 칸)
단위  mm/day,  시간대 KST(UTC+9)

--------------------------------------------------------------------------------
1. 파일 구성
--------------------------------------------------------------------------------
  BCG_grid_daily.csv          원자료. 결측은 빈칸        유효 {100*s['v0']:.1f}%
  BCG_grid_daily_filled.csv   결측을 보간해 채운 판      유효 {100*s['v1']:.1f}%
  BCG_grid_fillflag.csv       칸별 출처  0=원자료 / 1=보간 / 빈칸=끝내 결측
  BCG_grid_coords.csv         열이름 ↔ 위경도 + 칸별 유효일수

  형식  행 = 날짜(YYYY-MM-DD),  열 = "위도_경도"(예: 36.50_127.30),  값 = mm/day

  ★ 보간본을 쓸 때는 반드시 fillflag 를 함께 보기 바란다. 어느 칸이 관측 기반이고
    어느 칸이 채워 넣은 값인지 구분하지 않으면 정확도를 오해하게 된다.

--------------------------------------------------------------------------------
2. 결측의 성격  ★ 먼저 읽을 것
--------------------------------------------------------------------------------
  BC-G 의 결측은 ASCAT 위성의 관측 궤도(swath)에서 비롯되며, **무작위가 아니다.**

    · 하루 전체가 비는 날은 {s['T']-0:,}일 중 66일뿐이고, 나머지는 격자별로 흩어져 빈다.
      하루에 유효한 칸의 비율은 최대 66.6 % 로, 전국이 한 번에 채워지는 날은 없다.
    · 결측 칸은 유효 칸보다 계통적으로 건조하다.
      같은 날 ASOS 실측 평균이  유효 칸 6.41 mm/day  vs  결측 칸 2.48 mm/day.

  이 때문에 유효 칸 값을 그대로 공간 보간해 끌어오면 건조한 칸을 젖은 값으로
  채우게 되어 크게 과대추정된다. 아래 방법은 이 편향을 줄이기 위한 것이다.

--------------------------------------------------------------------------------
3. 보간 방법 — GPM 안내 비율 보간
--------------------------------------------------------------------------------
  결측 칸에 비가 왔는지는 결측이 없는 GPM IMERG 가 알고 있다. 그래서 강수량을
  직접 보간하지 않고, GPM 대비 비율만 보간한 뒤 GPM 을 곱해 되돌린다.

    1) 같은 날 유효 칸에서 비율 산정      r_k = BCG_k / GPM_k     (GPM_k > {GPM_DRY} 인 칸)
                                          r 은 0~5 로 제한
    2) r 을 결측 칸으로 공간 보간          역거리가중(IDW), 지수 p={POWER:.0f},
                                          반경 {RADIUS_KM:.0f} km, 최소 {MIN_NB}개 (부족하면 최근접 {MIN_NB}개)
    3) 되돌림                              BCG_ij = r̂_ij × GPM_ij
    4) GPM_ij ≤ {GPM_DRY} 인 칸             0 으로 채움 (위성이 무강수로 보는 칸)

  비율은 강수량 자체보다 공간적으로 완만해 보간이 안정적이며, 무엇보다 결측 칸의
  건·습 정보를 GPM 이 제공하므로 2절의 결측 편향이 상당 부분 상쇄된다.

  보간으로 채운 칸    {s['nfill']:,}개  (그중 GPM 무강수로 0 처리 {s['ndry']:,}개)
  끝내 못 채운 날     {s['fully_missing']}일  (GPM 까지 결측이라 안내자가 없는 날)

--------------------------------------------------------------------------------
4. 보간 정확도 (검증)
--------------------------------------------------------------------------------
  (a) 복원 검증 — 값이 있는 칸의 20 % 를 일부러 가리고 같은 방법으로 채운 뒤
      원래 값과 비교하였다. 무작위 200일 표본.

        표본 R = {s['vr']:.3f}   편의 = {s['vb']:+.2f} mm/day   RMSE = {s['vrm']:.2f} mm/day

  (b) ASOS 실측 대비 — 원자료 칸과 보간 칸을 나누어 비교하였다.

        구분              R        편의(mm/day)   RMSE(mm/day)   표본수
        원자료 칸      {s['r0'][0]:.3f}      {s['r0'][1]:+.2f}          {s['r0'][2]:.2f}       {s['r0'][3]:,}
        보간한 칸      {s['r1'][0]:.3f}      {s['r1'][1]:+.2f}          {s['r1'][2]:.2f}       {s['r1'][3]:,}
        채운 판 전체   {s['r2'][0]:.3f}      {s['r2'][1]:+.2f}          {s['r2'][2]:.2f}       {s['r2'][3]:,}

  보간 칸은 원자료 칸보다 정확도가 낮다. 이는 당연한 결과이며, 보간본은 시계열
  연속성이 필요한 용도(수문 모형 입력 등)에 쓰고, 정확도가 중요한 검증·통계
  분석에는 원자료(BCG_grid_daily.csv)를 쓰는 것을 권한다.

--------------------------------------------------------------------------------
5. 유역 단위 자료와의 관계
--------------------------------------------------------------------------------
  같은 원본에서 만든 유역 단위 자료가 최종산출물/02_표준유역/ 에 있다.
  그쪽은 국가 표준유역도(850개)로 잘라 유역 평균을 낸 것으로, 산정식은

      P_b(t) = Σ A_k · P_k(t) / Σ A_k        A_k = 격자 k 와 유역 b 의 교차면적

  이다. 0.1° 격자 한 칸(약 100 km²)이 표준유역 면적 중앙값(113 km²)과 같은 규모라
  유역 하나가 격자 여러 개에 걸치므로, 중심 격자값만 취하거나 산술평균하지 않고
  교차면적으로 가중하였다(면적은 위도 보정 적용). 유효 격자의 면적 합이 유역
  면적의 50 % 미만인 날은 결측 처리하였다.

  ※ 유역 단위 자료는 **보간 전 원자료**로 만들었다. 보간본 기준 유역 자료가
     필요하면 code/use/export_basin_csv.py 의 입력을 이 폴더의 filled 파일로
     바꾸어 다시 생성하면 된다.

--------------------------------------------------------------------------------
6. 원자료 및 생성 코드
--------------------------------------------------------------------------------
  BC-G     BC12_fields_2021fit.nc 의 BC_2
           ("LightGBM BC with AWS input (fusion)", 목표변수 IDW_AWS,
            학습 2021년 / 적용 2021~2025년)
  GPM      BC_LR_AWS_KST.nc 의 GPM (IMERG Final)
  ASOS     BC_LR_AWS_KST.nc 의 ASOS (IDW 보간)

  격자 CSV·보간   code/use/export_bcg_grid.py
  유역 CSV        code/use/export_basin_csv.py
"""
    with open(os.path.join(OUT_DIR, 'README.txt'), 'w', encoding='utf-8') as f:
        f.write(txt)


if __name__ == '__main__':
    sys.exit(main())
