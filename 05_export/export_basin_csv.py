"""
export_basin_csv.py — 표준유역 단위 일강수 CSV 추출
================================================================================
    python3 export_basin_csv.py

국가 표준유역도(850개)로 격자 강우장을 잘라 유역별 일강수 시계열을 만든다.
격자 단위 CSV(BC_LR_KST_daily.csv)와 달리, 수문 활용 단위인 유역 평균값이다.

출력  최종산출물/02_표준유역/
        BCG_basin_daily.csv     최종 산출물 BC-G      행=날짜, 열=표준유역코드
        BC_basin_daily.csv      BC (지상관측 미사용)
        ASOS_basin_daily.csv    IDW_ASOS (검증 기준)
        AWS_basin_daily.csv     IDW_AWS  (참고)
        basin_info.csv          유역 제원 + 격자 구성 + 산출물별 유효일수
        README.txt

산정 방법 — 격자·유역 교차면적 가중
    P_b(t) = Σ A_k·P_k(t) / Σ A_k      A_k = 격자 k 와 유역 b 의 교차면적
  0.1° 격자 한 칸이 약 100 km² 인데 표준유역 중앙값이 113 km² 라, 유역이 격자
  여러 개에 걸친다. 중심 격자값만 쓰거나 산술평균하면 유역 밖 강수가 섞인다.
  유효 격자 면적 합이 유역 면적의 50 % 미만인 날은 결측 처리한다.
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
import export_bcg_grid as GRID

OUT_DIR = '/Users/kim/Desktop/work/KIHS/DATA/최종산출물/02_표준유역'
MIN_AREA_FRAC = 0.5         # 유역 유효면적이 이보다 작은 날은 결측
VARS = {'BC_G': 'BCG', 'BC_G_FILLED': 'BCG_filled', 'BC': 'BC',
        'ASOS': 'ASOS', 'AWS': 'AWS'}


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    recs = B.read_dbf(B.SHP + '.dbf')
    offs = B.shp_offsets(B.SHP)

    ds = xr.open_dataset(B.NC)
    lat, lon = ds.lat.values, ds.lon.values
    times = pd.to_datetime(ds.time.values)
    ARR = {v: ds[v].values for v in ds.data_vars}
    d2 = xr.open_dataset(B.NC_BC12)
    for name, src in B.BC12_MAP.items():
        ARR[name] = d2[src].values
    d2.close()
    ds.close()

    # ---- BC-G 결측보간 격자를 만든다 (grid_BCG 와 동일한 함수·설정) ----
    print('BC-G 결측 보간 중...', end=' ', flush=True)
    G = ARR['BC_G'].reshape(len(times), -1)
    P = ARR['GPM'].reshape(len(times), -1)
    LATf, LONf = GRID.build_grid(lat, lon)
    Ffull = np.full_like(G, np.nan)
    for t in range(len(times)):
        Ffull[t], _ = GRID.fill_day(G[t], P[t], LONf, LATf,
                                    np.isfinite(G[t]))
    ARR['BC_G_FILLED'] = Ffull.reshape(len(times), len(lat), len(lon))
    print(f'유효 {100*np.isfinite(Ffull).mean():.1f}%')

    print(f'격자 {len(lat)}×{len(lon)} · {len(times)}일 '
          f'({times[0]:%Y-%m-%d}~{times[-1]:%Y-%m-%d})')

    series = {v: {} for v in VARS}
    rows = []
    skipped = []
    for k, r in enumerate(recs):
        code = r['SBSN_CD'].strip()
        name = r['SBSN_NM'].strip()
        try:
            geom = B.read_polygon(B.SHP, *offs[k])
        except Exception as e:
            skipped.append((code, name, f'도형 오류: {e}'))
            continue
        w = B.cell_weights(geom, lat, lon)
        if not w:
            skipped.append((code, name, '격자와 겹치지 않음'))
            continue
        tot = sum(a for _, _, a in w)
        area = float(geom.area * (111.32 ** 2)
                     * np.cos(np.deg2rad(geom.centroid.y)))

        rec = {'표준유역코드': code, '표준유역명': name,
               '대권역코드': r['BBSN_CD'].strip(),
               '중권역코드': r['MBSN_CD'].strip(),
               '면적_km2': round(area, 1),
               '중심위도': round(float(geom.centroid.y), 4),
               '중심경도': round(float(geom.centroid.x), 4),
               '중첩격자수': len(w),
               '최대격자기여율_pct': round(100 * max(a for _, _, a in w) / tot, 1)}

        for v in VARS:
            arr = ARR[v]
            num = np.zeros(len(times))
            den = np.zeros(len(times))
            for (i, j, a) in w:
                x = arr[:, i, j]
                ok = np.isfinite(x)
                num[ok] += a * x[ok]
                den[ok] += a
            out = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
            out[den < MIN_AREA_FRAC * tot] = np.nan
            series[v][code] = out
            rec[f'유효일_{VARS[v]}'] = int(np.isfinite(out).sum())
        rows.append(rec)
        if (k + 1) % 200 == 0:
            print(f'  {k+1}/{len(recs)} 처리 ({time.time()-t0:.0f}초)')

    info = pd.DataFrame(rows)
    idx = times.strftime('%Y-%m-%d')

    print()
    paths = []
    for v, tag in VARS.items():
        df = pd.DataFrame(series[v], index=idx)
        df.index.name = 'date'
        df = df[info['표준유역코드'].tolist()]      # 열 순서를 유역코드 순으로
        p = os.path.join(OUT_DIR, f'{tag}_basin_daily.csv')
        df.to_csv(p, float_format='%.2f')
        paths.append(p)
        fin = np.isfinite(df.values)
        print(f'  {tag:5s} {df.shape[0]:,}행 × {df.shape[1]:,}열 · '
              f'유효 {100*fin.mean():4.1f}% · {os.path.getsize(p)/1e6:5.2f} MB')

    p_info = os.path.join(OUT_DIR, 'basin_info.csv')
    info.to_csv(p_info, index=False, encoding='utf-8-sig')
    paths.append(p_info)
    if skipped:
        p_skip = os.path.join(OUT_DIR, '_제외유역.csv')
        pd.DataFrame(skipped, columns=['표준유역코드', '표준유역명', '사유']) \
          .to_csv(p_skip, index=False, encoding='utf-8-sig')
        paths.append(p_skip)

    write_readme(info, times, skipped, series)
    print(f'\n출력 {OUT_DIR}  ({time.time()-t0:.0f}초)')
    for p in sorted(os.listdir(OUT_DIR)):
        print(f'  {p:26s} {os.path.getsize(os.path.join(OUT_DIR,p))/1e6:7.2f} MB')
    return 0


def write_readme(info, times, skipped, series):
    # 결측 편향 진단 — README 에 넣을 수치
    M = np.column_stack([series['BC_G'][c] for c in info['표준유역코드']])
    O = np.column_stack([series['ASOS'][c] for c in info['표준유역코드']])
    fin = np.isfinite(M)
    ofin = np.isfinite(O)
    rain = ofin & (O >= 0.1)
    dry = ofin & (O < 0.1)
    obs_on = float(np.nanmean(np.where(fin & ofin, O, np.nan)))
    obs_off = float(np.nanmean(np.where(~fin & ofin, O, np.nan)))
    both = fin & ofin
    bad_annual = float(np.nanmean(M[both])) * 365.25
    good_annual = float(np.nanmean(O[ofin])) * 365.25
    asos_valid = float(ofin.mean())
    Fm = np.column_stack([series['BC_G_FILLED'][c] for c in info['표준유역코드']])
    bcg_v = float(fin.mean()); bcg_m = float(np.nanmean(M))
    fil_v = float(np.isfinite(Fm).mean()); fil_m = float(np.nanmean(Fm))
    aso_m = float(np.nanmean(O))
    cf = np.isfinite(Fm) & ofin
    fil_r = float(np.corrcoef(Fm[cf], O[cf])[0, 1])
    fil_b = float((Fm[cf] - O[cf]).mean())
    yrs = pd.DatetimeIndex(times).year
    _al = []
    for y in (2022, 2023, 2024):
        k = yrs == y
        fs = np.nansum(Fm[k]) / Fm.shape[1]
        os_ = np.nansum(O[k]) / O.shape[1]
        _al.append(f'      {y}    BC-G 보간본 {fs:6,.0f} mm    '
                   f'IDW_ASOS {os_:6,.0f} mm    차이 {100*(fs/os_-1):+5.1f}%')
    annual_lines = '\n'.join(_al)

    n = len(info)
    a = info['면적_km2']
    c = info['중첩격자수']
    ev = {t: info[f'유효일_{t}'] for t in VARS.values()}
    T = len(times)
    txt = f"""표준유역 단위 일강수 자료
================================================================================
국가 표준유역도(850개)로 다종자료 융합 강우장을 잘라 산출한 유역 평균 일강수.
수문 모형·유역 단위 분석에 바로 쓸 수 있도록 격자가 아닌 유역 단위로 정리하였다.

생성  {pd.Timestamp.now():%Y-%m-%d %H:%M}
기간  {times[0]:%Y-%m-%d} ~ {times[-1]:%Y-%m-%d}  ({T:,}일)
단위  mm/day,  시간대 KST(UTC+9), 일 경계 00~24시
유역  {n:,}개  (전체 850개 중 격자와 겹치는 유역)

--------------------------------------------------------------------------------
1. 파일 구성
--------------------------------------------------------------------------------
  BCG_basin_daily.csv         BC-G 원자료   최종 산출물, 결측은 빈칸
  BCG_filled_basin_daily.csv  BC-G 보간본   결측을 채운 판  ★ 총량 계산은 이쪽
  BC_basin_daily.csv          BC            지상관측 미사용 편향보정
  ASOS_basin_daily.csv   IDW_ASOS   종관기상관측 보간 — 검증 기준
  AWS_basin_daily.csv    IDW_AWS    방재기상관측 보간 — 참고
  basin_info.csv         유역 제원 · 격자 구성 · 산출물별 유효일수
  _제외유역.csv           격자와 겹치지 않아 제외된 유역 ({len(skipped)}개)

  자료 CSV 형식   행 = 날짜(YYYY-MM-DD),  열 = 표준유역코드,  값 = mm/day
                  빈칸 = 결측.  열 이름과 유역명 대응은 basin_info.csv 참조

--------------------------------------------------------------------------------
2. 산출물 설명
--------------------------------------------------------------------------------
  BC-G   위성 토양수분 역산(SM2RAIN) · 위성강수(GPM IMERG) · 재분석(ERA5-Land)을
         삼중병치분석(TCA)으로 융합한 뒤, 같은 날 지상관측(IDW_AWS)을 입력특징에
         더해 LightGBM 으로 편향보정한 산출물. 관측이 존재하는 조건에서 최고
         정확도를 목표로 하는 관측 보강형이다.
  BC     동일한 체계에서 지상관측을 입력특징으로 쓰지 않은 산출물.
         관측이 없는 지역으로의 일반화에 유리하다.

  BC-G 보간본  BC-G 격자의 결측을 채운 뒤 같은 방식으로 유역 평균을 낸 자료.
         보간 방법은 GPM 안내 비율 보간이며 상세 내용은
         최종산출물/01_격자/README.txt 3절 참조. 요약하면,
           1) 같은 날 값이 있는 칸에서 비율 r = BC-G / GPM 산정
           2) r 을 결측 칸으로 역거리가중 보간 (지수 2, 반경 150 km, 최소 3개)
           3) BC-G = r̂ × GPM 으로 되돌림
           4) GPM ≤ 0.1 mm 인 칸은 0 으로 채움
         강수량을 직접 보간하지 않고 비율만 보간하는 이유는 5절에 있다.

  두 산출물 모두 2021년으로 학습하고 2021~2025년에 적용하였으므로,
  독립적인 성능 평가는 2022년 이후 구간에서 수행하여야 한다.

--------------------------------------------------------------------------------
3. 유역 평균 산정 방법
--------------------------------------------------------------------------------
  격자·유역 교차면적 가중평균을 사용하였다.

      P_b(t) = Σ A_k · P_k(t) / Σ A_k        A_k = 격자 k 와 유역 b 의 교차면적

  원자료 격자는 0.1°(위도 36° 부근 약 100 km²)이고 표준유역 면적 중앙값은
  {a.median():.0f} km²(최소 {a.min():.0f}, 최대 {a.max():.0f})이므로, 유역 하나가 격자 여러 개에 걸친다.
  중심 격자값만 취하거나 중첩 격자를 산술평균하면 유역 밖 강수가 섞이므로
  교차면적으로 가중하였다. 면적은 위도 보정을 적용해 산정하였다.

  결측 격자는 분자·분모에서 함께 제외하되, 유효 격자의 면적 합이 유역 면적의
  {MIN_AREA_FRAC*100:.0f}% 미만인 날은 부분 관측에 따른 왜곡을 피하기 위해 결측 처리하였다.

  유역별 중첩 격자수   중앙 {c.median():.0f}개 (최소 {c.min()}, 최대 {c.max()})
  최대 격자 기여율     중앙 {info['최대격자기여율_pct'].median():.0f}%

--------------------------------------------------------------------------------
4. 해상도에 관한 주의  ★
--------------------------------------------------------------------------------
  원자료 격자(약 100 km²)가 표준유역 면적(중앙 {a.median():.0f} km²)과 같은 규모다.
  따라서 '유역 평균'이라 하더라도 실질적으로는 격자 1~2개의 정보로 유역 하나를
  대표하는 것이며, 유역 내부의 공간 변동은 표현되지 않는다.
  면적이 작은 유역일수록 이 한계가 크므로, basin_info.csv 의 '중첩격자수'와
  '최대격자기여율_pct'를 함께 확인하고 사용하기 바란다.

  면적 50 km² 미만 유역   {int((a < 50).sum())}개
  최대 격자 기여율 70% 이상 유역   {int((info['최대격자기여율_pct'] >= 70).sum())}개

--------------------------------------------------------------------------------
5. 결측 현황
--------------------------------------------------------------------------------
  산출물별 유역당 유효일수 (전체 {T:,}일 기준)

      자료      중앙값      최소     최대     비율
      BC-G    {ev['BCG'].median():>7,.0f}  {ev['BCG'].min():>7,}  {ev['BCG'].max():>7,}   {100*ev['BCG'].median()/T:>4.0f}%
      BC      {ev['BC'].median():>7,.0f}  {ev['BC'].min():>7,}  {ev['BC'].max():>7,}   {100*ev['BC'].median()/T:>4.0f}%
      ASOS    {ev['ASOS'].median():>7,.0f}  {ev['ASOS'].min():>7,}  {ev['ASOS'].max():>7,}   {100*ev['ASOS'].median()/T:>4.0f}%
      AWS     {ev['AWS'].median():>7,.0f}  {ev['AWS'].min():>7,}  {ev['AWS'].max():>7,}   {100*ev['AWS'].median()/T:>4.0f}%

  BC-G·BC 의 결측은 ASCAT 위성의 관측 주기에서 비롯된다. 극궤도 위성이라 매일
  같은 지점을 관측하지 못하며, 토양수분이 없는 날은 SM2RAIN 이 계산되지 않아
  후속 단계도 함께 빈다.

  ★★ 결측이 무작위가 아니다 — 총량 계산 금지 ★★

  BC-G·BC 의 결측일은 무강수일에 치우쳐 있다. 즉 값이 있는 날은 비가 온 날일
  확률이 높다.

      ASOS 기준 강수일  중 BC-G 값이 있는 비율   {100*(fin&rain).sum()/max(rain.sum(),1):.1f}%
      ASOS 기준 무강수일 중 BC-G 값이 있는 비율   {100*(fin&dry).sum()/max(dry.sum(),1):.1f}%

      같은 날 ASOS 평균강수   BC-G 값이 있는 날 {obs_on:.2f} mm/day
                              BC-G 가 빈 날     {obs_off:.2f} mm/day

  따라서 BC-G·BC 열을 그대로 더해 연·월 총량을 구하면 크게 과대추정된다.
  실제로 값이 있는 날만 평균해 365 를 곱하면 연 {bad_annual:,.0f} mm 가 나오는데,
  같은 기간 ASOS 실제 연강수는 약 {good_annual:,.0f} mm 이다.

  ★ 이 문제는 BCG_filled_basin_daily.csv 로 해결된다

  보간본은 결측 칸을 GPM 안내 비율 보간으로 채운 뒤 유역 평균을 낸 것이라,
  건조한 결측일도 제 값(대개 0)으로 들어간다. 그래서 총량을 그대로 계산할 수 있다.

      자료            유효율    유효일 평균     단순 연환산
      BC-G 원자료     {100*bcg_v:4.1f}%    {bcg_m:5.2f} mm/day   {bcg_m*365.25:6,.0f} mm   ← 과대 (쓰지 말 것)
      BC-G 보간본     {100*fil_v:4.1f}%    {fil_m:5.2f} mm/day   {fil_m*365.25:6,.0f} mm
      IDW_ASOS        {100*asos_valid:4.1f}%    {aso_m:5.2f} mm/day   {aso_m*365.25:6,.0f} mm   ← 실측 기준

  완전 연도 기준 유역평균 연강수 (보간본 vs ASOS)
{annual_lines}
  ASOS 대비 보간본 성능   R = {fil_r:.3f},  편의 = {fil_b:+.3f} mm/day

  올바른 사용법
    · 총량·월합·연강수가 필요하면  BCG_filled_basin_daily.csv 를 쓴다.
    · 정확도가 중요한 검증·통계에는 BCG_basin_daily.csv(원자료)를 쓴다.
      보간으로 채운 값은 관측 기반이 아니므로 성능을 낙관적으로 만들 수 있다.
    · 두 산출물을 비교할 때는 반드시 공통으로 값이 있는 날만 골라 비교한다.
    · 유역별 유효일수는 basin_info.csv 의 유효일_BCG / 유효일_BCG_filled 열 참조.

--------------------------------------------------------------------------------
6. 검증 결과 요약 (참고)
--------------------------------------------------------------------------------
  미계측 표준유역 3곳(논산천상류·조종천상류·유등천상류)에서 IDW_ASOS 를 기준으로
  평가한 결과는 다음과 같다 (2022~2025, BC-G 유효일 기준).

      유역          BC-G    BC     GPM    ERA5
      논산천상류    0.941  0.807  0.767  0.724     ← KGE
      조종천상류    0.881  0.804  0.847  0.768
      유등천상류    0.871  0.682  0.731  0.689

  BC-G 는 세 유역 모두에서 위성 강수(GPM)·재분석 강수(ERA5)를 상회하였다.
  다만 강우 강도가 클수록 과소추정하는 경향이 공통적으로 확인되므로,
  홍수 관련 활용 시에는 이 특성을 고려하여야 한다.
  상세 내용은 최종보고서 3.4.2 절 참조.

--------------------------------------------------------------------------------
7. 원자료 및 재현
--------------------------------------------------------------------------------
  유역 경계   국가 표준유역도 std_basin_850 (WGS84)
  격자 강우장 BC12_fields_2021fit.nc (BC_1=BC, BC_2=BC-G)
              BC_LR_AWS_KST.nc (ASOS, AWS)
  생성 코드   code/use/export_basin_csv.py
"""
    p = os.path.join(OUT_DIR, 'README.txt')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(txt)


if __name__ == '__main__':
    sys.exit(main())
