#!/usr/bin/env python3
"""티센 다각형 기반 표준유역 일강수 — 기존 방식 비교군 산출.

    python3 export_thiessen_basin.py

과업 산출물(BC-G)은 격자–유역 교차면적 가중으로 유역 평균을 낸다.  이 스크립트는
국내 실무에서 널리 쓰이는 **티센 다각형** 방식으로 같은 유역·같은 기간의 값을
따로 만들어, 두 방식을 나란히 견줄 수 있게 한다.

산정 방법
    지점 가중치   유역을 250 m 격자로 잘게 나눠 각 칸을 최근접 관측지점에
                  배정하고, 그 면적 비율을 지점 가중치로 삼는다 (EPSG:5179).
                  폴리곤 교차를 직접 푸는 것과 결과가 같고 훨씬 간단하다.
    유역 일강수   P_b(t) = Σ w_i·P_i(t) / Σ w_i
                  그날 값이 있는 지점만 더하고 가중치를 다시 정규화한다.
                  값이 있는 지점이 하나도 없으면 그날은 결측이다.

자료
    관측    기상청 시자료 (종관 ASOS + 방재 AWS 752지점)
            하루는 KST 01시 ~ 익일 00시.  연 파일 경계에서 하루가 둘로 나뉘므로
            같은 날짜끼리 합친다.
    유역    국가 표준유역도 850.  산출 대상은 격자 산출물과 같은 848개.

출력  cpuserver .../project_KIHS/data/thiessen/
        THIESSEN_basin_daily.csv     행=날짜, 열=표준유역코드 (mm/day)
        THIESSEN_basin_daily.nc      같은 값의 NetCDF (time × basin)
        THIESSEN_basin_weights.csv   유역코드 · 지점번호 · 가중치 · 관측망
        THIESSEN_README.txt
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'analysis'))
import basin_eval_core as B          # noqa: E402
from thiessen_basin import to5179    # noqa: E402

WORK = Path(os.environ.get('KIHS_WORK', '/Users/kim/Desktop/work'))
DATA = WORK / 'KIHS/DATA'
CPU = Path(os.environ.get('KIHS_CPUSERVER', '/Users/kim/cpuserver_data'))
OUT = CPU / 'personal_data/project_KIHS/data/thiessen'
AWSC = str(DATA / 'AWS/Data_AWS_hourly_%d.csv')
REF = DATA / '최종산출물/02_표준유역/BCG_basin_daily.csv'   # 유역 목록·기간 기준
CACHE = DATA / 'stn_daily_2021_2025.pkl'

YEARS = range(2021, 2026)
START, END = '2021-01-01', '2025-05-01'
CELL = 250.0        # 가중치 산정용 세부 격자 (m)
NEAR_N = 80         # 유역 중심에서 가까운 지점만 후보로 둔다 (속도)


# ────────────────────────────────────────────────────────────── 관측
def station_daily() -> tuple[pd.DataFrame, pd.DataFrame]:
    """지점별 일강수와 좌표.  한 번 만들어 두고 캐시에서 읽는다."""
    if CACHE.exists():
        import pickle
        return pickle.load(open(CACHE, 'rb'))
    import pickle
    frames, meta = [], []
    for y in YEARS:
        parts = [ch for ch in pd.read_csv(
            AWSC % y, usecols=['일시', '지점', '위도', '경도', '강수량'],
            chunksize=3_000_000)]
        d = pd.concat(parts, ignore_index=True)
        d['날짜'] = (pd.to_datetime(d['일시'])
                   - pd.Timedelta(hours=1)).dt.normalize()
        frames.append(d.groupby(['날짜', '지점'])['강수량']
                      .sum(min_count=1).unstack())
        meta.append(d.drop_duplicates('지점').set_index('지점')[['위도', '경도']])
        print(f'  {y} 읽음', flush=True)
    # 연 파일 경계일(12-31)은 두 파일에 나뉘어 있다 → 같은 날짜끼리 더한다
    S = (pd.concat(frames).groupby(level=0).sum(min_count=1)
         .sort_index().loc[START:END])
    C = pd.concat(meta).groupby(level=0).first()
    pickle.dump((S, C), open(CACHE, 'wb'))
    return S, C


# ────────────────────────────────────────────────────────────── 가중치
def thiessen_weights(geom, sx, sy, codes):
    """유역 하나의 티센 가중치 {지점번호: 비율} 와 유효면적(km²)."""
    x0, y0, x1, y1 = geom.bounds
    cx, cy = to5179(np.array([(x0 + x1) / 2]), np.array([(y0 + y1) / 2]))
    cand = np.argsort((sx - cx[0]) ** 2 + (sy - cy[0]) ** 2)[:NEAR_N]

    # 섬을 낀 유역은 MultiPolygon 이므로 조각을 모두 훑는다
    parts = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
    shells = []
    for g in parts:
        r = np.asarray(g.exterior.coords)
        rx, ry = to5179(r[:, 0], r[:, 1])
        holes = []
        for h in g.interiors:
            hh = np.asarray(h.coords)
            hx, hy = to5179(hh[:, 0], hh[:, 1])
            holes.append(np.column_stack([hx, hy]))
        shells.append((np.column_stack([rx, ry]), holes))

    allxy = np.vstack([s for s, _ in shells])
    gx = np.arange(allxy[:, 0].min(), allxy[:, 0].max() + CELL, CELL)
    gy = np.arange(allxy[:, 1].min(), allxy[:, 1].max() + CELL, CELL)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([GX.ravel(), GY.ravel()])

    ins = np.zeros(len(pts), bool)
    for shell, holes in shells:
        m = MplPath(shell).contains_points(pts)
        for h in holes:
            m &= ~MplPath(h).contains_points(pts)
        ins |= m
    if not ins.any():
        return {}, 0.0

    px, py = pts[ins, 0], pts[ins, 1]
    d2 = ((px[:, None] - sx[cand][None, :]) ** 2
          + (py[:, None] - sy[cand][None, :]) ** 2)
    cnt = np.bincount(np.argmin(d2, axis=1), minlength=len(cand))
    tot = cnt.sum()
    w = {int(codes[cand[i]]): c / tot for i, c in enumerate(cnt) if c}
    return w, tot * CELL ** 2 / 1e6


# ────────────────────────────────────────────────────────────── 본체
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print('1. 지점 일강수')
    S, C = station_daily()
    codes = C.index.to_numpy()
    sx, sy = to5179(C['경도'].to_numpy(float), C['위도'].to_numpy(float))
    print(f'   {len(S)}일 × {S.shape[1]}지점   '
          f'(ASOS {(codes < 300).sum()} · AWS {(codes >= 300).sum()})')

    print('2. 유역 목록')
    basins = pd.read_csv(REF, nrows=0).columns[1:].tolist()   # 격자 산출물과 동일
    recs = B.read_dbf(B.SHP + '.dbf')
    offs = B.shp_offsets(B.SHP)
    key = next(k for k in recs[0] if 'BAS' in k.upper() or 'CD' in k.upper())
    idx = {r[key].strip(): i for i, r in enumerate(recs)}
    print(f'   대상 {len(basins)}개 (표준유역도 {len(recs)}개 중)')

    print('3. 티센 가중치')
    t0 = time.time()
    W, area, rows = {}, {}, []
    for n, code in enumerate(basins, 1):
        if code not in idx:
            W[code] = {}
            continue
        try:
            g = B.read_polygon(B.SHP, *offs[idx[code]])
        except Exception:
            W[code] = {}
            continue
        w, a = thiessen_weights(g, sx, sy, codes)
        W[code], area[code] = w, a
        for st, v in sorted(w.items(), key=lambda x: -x[1]):
            rows.append({'표준유역코드': code, '지점번호': st,
                         '가중치': round(v, 6),
                         '관측망': 'ASOS' if st < 300 else 'AWS'})
        if n % 200 == 0:
            print(f'   {n}/{len(basins)}   {time.time() - t0:.0f}s', flush=True)
    nst = pd.Series({k: len(v) for k, v in W.items()})
    print(f'   완료 {time.time() - t0:.0f}s   '
          f'기여지점 중앙값 {nst.median():.0f}개 (최소 {nst.min()} 최대 {nst.max()})')

    print('4. 유역 일강수')
    V = S.to_numpy(float)
    have = np.isfinite(V)
    col = {c: i for i, c in enumerate(S.columns)}
    out = np.full((len(S), len(basins)), np.nan)
    for j, code in enumerate(basins):
        w = W.get(code) or {}
        ii = [col[k] for k in w if k in col]
        if not ii:
            continue
        ww = np.array([w[k] for k in w if k in col])
        num = np.nansum(np.where(have[:, ii], V[:, ii], 0.0) * ww, axis=1)
        den = (have[:, ii] * ww).sum(axis=1)
        out[:, j] = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    T = pd.DataFrame(out, index=S.index, columns=basins)
    T.index.name = 'date'

    print('5. 저장')
    p = OUT / 'THIESSEN_basin_daily.csv'
    T.to_csv(p, float_format='%.2f')
    print('  ', p)

    pw = OUT / 'THIESSEN_basin_weights.csv'
    pd.DataFrame(rows).to_csv(pw, index=False, encoding='utf-8-sig')
    print('  ', pw)

    try:
        import xarray as xr
        ds = xr.Dataset(
            {'precipitation': (('time', 'basin'), T.to_numpy(np.float32))},
            coords={'time': T.index.values,
                    'basin': np.array(basins, dtype='U8')},
            attrs={'title': '티센 다각형 기반 표준유역 일강수',
                   'method': '250 m 최근접지점 배정 면적비 가중, 결측지점 재정규화',
                   'stations': f'ASOS+AWS {S.shape[1]}지점 (기상청 시자료)',
                   'day_boundary': 'KST 01시~익일 00시',
                   'units': 'mm/day',
                   'note': '과업 산출물이 아니라 기존 방식 비교군'})
        ds['precipitation'].attrs['units'] = 'mm/day'
        pn = OUT / 'THIESSEN_basin_daily.nc'
        ds.to_netcdf(pn)
        print('  ', pn)
    except Exception as e:
        print('   NetCDF 저장 실패 —', e)

    ok = T.notna().sum()
    with open(OUT / 'THIESSEN_README.txt', 'w', encoding='utf-8') as f:
        f.write(f"""티센 다각형 기반 표준유역 일강수 — 기존 방식 비교군
{'=' * 78}
과업명   다종자료 융합 강우 공간분포 산정 기술 개발
발주     한국수자원조사기술원      수행  광주과학기술원(GIST)

이 자료는 과업 산출물이 아니다.  과업 산출물(BC-G)은 격자-유역 교차면적 가중으로
유역 평균을 낸다.  국내 실무에서 널리 쓰이는 티센 다각형 방식과 견주려고 같은
유역·같은 기간에 대해 따로 만든 비교군이다.

기간     {T.index.min():%Y-%m-%d} ~ {T.index.max():%Y-%m-%d}  ({len(T):,}일)
유역     {len(basins)}개 (격자 산출물과 동일)
관측     기상청 시자료 {S.shape[1]}지점 — 종관 ASOS {(codes < 300).sum()}, 방재 AWS {(codes >= 300).sum()}
단위     mm/day,  시간대 KST,  하루 = 01시 ~ 익일 00시

산정 방법
  1) 유역을 250 m 격자로 잘게 나눠 각 칸을 최근접 관측지점에 배정한다.
     투영은 EPSG:5179 (한국 중부원점 TM).  그 면적 비율이 지점 가중치다.
  2) P_b(t) = sum(w_i * P_i(t)) / sum(w_i)
     그날 값이 있는 지점만 더하고 가중치를 다시 정규화한다.
     값이 있는 지점이 하나도 없으면 그날은 결측(빈칸)이다.

파일
  THIESSEN_basin_daily.csv     행=날짜, 열=표준유역코드
  THIESSEN_basin_daily.nc      같은 값 (time x basin)
  THIESSEN_basin_weights.csv   유역코드, 지점번호, 가중치, 관측망

유효일수
  중앙값 {ok.median():.0f}일 / {len(T)}일    최소 {ok.min()}일    최대 {ok.max()}일
  기여지점 수 중앙값 {nst.median():.0f}개 (최소 {nst.min()}, 최대 {nst.max()})

읽을 때 주의
  · 티센은 지점이 대표하는 면적을 기하학적으로만 정한다.  강수의 실제 공간 구조를
    반영하지 못하고, 유역 안팎에 지점이 없으면 먼 지점 하나가 유역 전체를
    대표하게 된다.  기여지점이 1~2개인 유역의 값은 그렇게 읽어야 한다.
  · 종관관측(ASOS)이 섞여 있으므로, ASOS 를 기준으로 이 자료를 평가하면
    일부 유역에서 순환이 생긴다.  독립 검증에는 쓸 수 없다.
""")
    print('  ', OUT / 'THIESSEN_README.txt')

    print(f'\n유효일수 중앙값 {ok.median():.0f}일 / {len(T)}일')
    print(T.iloc[:, :6].tail(3).to_string())
    return T


if __name__ == '__main__':
    main()
