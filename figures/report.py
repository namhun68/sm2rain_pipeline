#!/usr/bin/env python3
"""KIHS 최종보고서 그림 모음 — 한 파일로 모은 판.

흩어져 있던 event_maps.py · event_grid_figs.py · fig21_kge_map.py 를 합쳤다.
자료 읽기와 지도 그리기 도구를 공유하므로 그림마다 같은 규약이 적용된다.

────────────────────────────────────────────────────────────────────────────
만드는 그림
────────────────────────────────────────────────────────────────────────────
  이름     파일                                          내용
  map      fig_event/M1~M3_<유역>_배치도.png             티센 분할 · 0.1° 격자 · 관측소
  series   fig_event/T1_사상전후_시계열.png               사상 앞뒤 일강수 (7종)
  sums     fig_event/S1_사상누적.png                     사상 누적 (7종)
  grid     fig_event/G1~G3_<유역>_격자비교<TAG>.png       티센·IDW_AWS·BC-G·BC·GPM 격자 비교
  graph    fig_event/C1_티센_BCG_비교그래프<TAG>.png      일별 + 구간 누적 막대
  kge      fig_basin/N1_전국유역_KGE지도_경계선.png       전국 표준유역 KGE (남한 경계선)

  grid 는 수치표 fig_event/event_thiessen_vs_bcg<TAG>.csv 도 같이 쓴다.

────────────────────────────────────────────────────────────────────────────
실행
────────────────────────────────────────────────────────────────────────────
    python3 report.py                 # 전부
    python3 report.py grid graph      # 고른 것만
    python3 report.py kge

  grid · graph 는 격자 NetCDF 가 필요해 cpuserver 마운트가 있어야 한다.
  나머지(map · series · sums · kge)는 캐시 pkl 만으로 돌아가므로 랩탑에서도 된다.

────────────────────────────────────────────────────────────────────────────
지켜야 할 규약
────────────────────────────────────────────────────────────────────────────
  · 하루는 KST 01시 ~ 익일 00시.  AWS 시자료는 (시각 − 1h) 로 묶는다.
  · 사상 구간은 기준자료(IDW_ASOS) 일강수가 1 mm 이상으로 이어지는 구간.
  · BC 계열은 ASCAT 관측폭 때문에 결측일이 있다.  공정하게 견주려고
    유역 내 모든 격자가 유효한 날만 다섯 자료 모두에서 합산하고, 빠진 날짜는
    그림 안에 적는다.
  · 모든 값은 **구간 누적**이다.  날짜별로 먼저 더한 뒤 면적으로 가중한다.
        티센   Σ wi·(Σt Pi,t)          wi = 지점 티센 면적비
        격자   Σ aj·(Σt Pj,t) / Σ aj    aj = 격자–유역 교차면적
  · 배치는 비율이 아니라 **자 단위(inch)** 로 잡는다.  겹칠 여지를 없앤다.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.collections import PatchCollection
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Polygon as MplPoly
from matplotlib.patches import Rectangle
from matplotlib.path import Path as MplPath
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'analysis'))
import basin_eval_core as B          # noqa: E402
import national_eval as N            # noqa: E402
from thiessen_basin import to5179    # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
# § 0  설정
# ══════════════════════════════════════════════════════════════════════════
ROOT = Path('/Users/kim/Desktop/work/KIHS')
OUT_E = ROOT / 'fig_event'                      # 호우사상 검토용
OUT_B = ROOT / 'fig_basin'                      # 전국·유역 평가용
AWSC = str(ROOT / 'DATA/AWS/Data_AWS_hourly_%d.csv')
THI = str(ROOT / 'DATA/thiessen_aws.pkl')
NCBC = ('/Users/kim/cpuserver_data/personal_data/project_KIHS/result/'
        'ASCAT/precipitation/BC12_fields_2021fit.nc')
STN_CACHE = OUT_E / '_stn_daily.pkl'

TAG = '_v3'          # grid·graph 세트 이름.  바꾸면 이전 세트를 덮지 않는다.

BASINS = ['논산천상류', '조종천상류', '유등천상류']
PEAK = {'논산천상류': '2023-07-14',              # 기준자료 최대 일강수일
        '조종천상류': '2022-06-30',
        '유등천상류': '2023-07-14'}
WET = 1.0            # 사상 구간을 잇는 최소 일강수 (mm)
PAD_G = 4            # 비교 그래프에서 사상 앞뒤로 더 보여줄 날 수
PAD_S = 6            # 사상 전후 시계열에서 더 보여줄 날 수

# ── 격자 산출물.  BC 계열이 왜 과소추정되는지 보이려면 학습 목표인
#    IDW_AWS 를 바로 위에 놓고 견주어야 한다.
PROD = ['AWS', 'BC_G', 'BC', 'GPM']
PLAB = {'AWS': 'IDW_AWS (BC 학습 목표)', 'BC_G': 'BC-G (AWS 입력 포함)',
        'BC': 'BC (AWS 입력 없음)', 'GPM': 'GPM (입력 위성강수)'}
PCOL = {'AWS': '#00798C', 'BC_G': '#8E3B46', 'BC': '#D1495B', 'GPM': '#66A182'}
TARGET = 'AWS'               # BC 계열이 재현하도록 학습된 자료
FOLLOW = ('BC_G', 'BC')      # 학습 목표 대비 얼마나 못 따라갔는지 함께 적는다

C_THI = '#9A6FB0'
LAB_T = '티센 (기존 방식)'
SER = ['THI'] + PROD
SLAB = {'THI': LAB_T, **PLAB}
SCOL = {'THI': C_THI, **PCOL}

# ── M·T·S 는 보고서 검토용이라 일곱 자료를 모두 싣는다
SHOW7 = ['ASOS', 'THIESSEN', 'AWS', 'BC_G', 'BC', 'GPM', 'ERA5']
LAB7 = {'ASOS': 'IDW_ASOS (기준)', 'THIESSEN': '티센', 'AWS': 'IDW_AWS',
        'BC_G': 'BC-G', 'BC': 'BC', 'GPM': 'GPM', 'ERA5': 'ERA5'}
COL7 = {'ASOS': '#111111', 'THIESSEN': '#D1495B', 'AWS': '#2E86AB',
        'BC_G': '#F18F01', 'BC': '#8D6A9F', 'GPM': '#4C956C', 'ERA5': '#8C8C8C'}

FSM, FSG = 25, 21    # 지도용 / 그래프용 기본 글자 크기
FADE = .30           # 유역 밖을 옅게 까는 투명도
TXT_FC, TXT_EC = 'white', '#4A5560'   # 지도 위 글자 = 흰색 + 은색 테두리
CMAP = plt.get_cmap('YlGnBu')

# 구간마다 자릿수가 달라(수 mm ~ 수백 mm) 등간격 눈금으로는 한 장에 못 담는다.
# 강수도에서 흔히 쓰는 비등간격 계급을 써서 그림 하나가 색눈금 하나를 같이 쓴다.
BASE_LV = [0, 1, 5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500,
           600, 800]


# ══════════════════════════════════════════════════════════════════════════
# § 1  공통 도구
# ══════════════════════════════════════════════════════════════════════════
def set_font() -> None:
    for c in ('AppleGothic', 'NanumGothic', 'Malgun Gothic'):
        if any(c in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams['font.family'] = c
            break
    plt.rcParams['axes.unicode_minus'] = False


def stroke(w: float = 3.2, fg: str = 'white'):
    """글자 테두리.  흰 글씨를 밝은 바탕 위에서도 읽히게 한다."""
    return [pe.withStroke(linewidth=w, foreground=fg)]


def rain_levels(vmax: float) -> np.ndarray:
    """자료 최대값을 덮는 데까지만 강수 계급을 쓴다."""
    for k, v in enumerate(BASE_LV):
        if v >= vmax:
            return np.array(BASE_LV[:k + 1], float)
    return np.array(BASE_LV, float)


def pct(new: float, base: float) -> str:
    """기준이 너무 작으면 백분율이 의미를 잃으므로 적지 않는다."""
    return f'{100 * (new / base - 1):+.0f} %' if base >= 5.0 else '—'


def boxer(fig, W: float, H: float):
    """자 단위(inch)로 축을 놓는 도우미.  (왼쪽, 위, 폭, 높이) 를 받는다."""
    def bx(xi, yi, w, h):
        return fig.add_axes([xi / W, 1 - (yi + h) / H, w / W, h / H])
    return bx


# ══════════════════════════════════════════════════════════════════════════
# § 2  자료
# ══════════════════════════════════════════════════════════════════════════
def stn_daily(year: int, stns) -> tuple[pd.DataFrame, pd.DataFrame]:
    """AWS 시자료 → 지점별 일강수(mm).  하루는 01시 ~ 익일 00시 (KST)."""
    fr = []
    for ch in pd.read_csv(AWSC % year,
                          usecols=['일시', '지점', '위도', '경도', '강수량'],
                          chunksize=2_000_000):
        fr.append(ch[ch['지점'].isin(stns)])
    d = pd.concat(fr, ignore_index=True)
    t = pd.to_datetime(d['일시'])
    d['날짜'] = (t - pd.Timedelta(hours=1)).dt.normalize()
    g = d.groupby(['날짜', '지점'])['강수량'].sum(min_count=1).unstack()
    c = d.drop_duplicates('지점').set_index('지점')[['위도', '경도']]
    return g, c


def load_basins():
    """유역 도형·면적가중 시계열 · 티센 가중치 · 지점 일강수.

    전부 캐시(pkl)에서 읽으므로 cpuserver 마운트가 없어도 된다.
    """
    D = B.load()
    B.add_thiessen(D)
    W = pickle.load(open(THI, 'rb'))['weights']

    years = sorted({pd.Timestamp(v).year for v in PEAK.values()})
    stns = sorted({s for n in BASINS for s in W[n][0]})
    if STN_CACHE.exists():
        S, C = pickle.load(open(STN_CACHE, 'rb'))
    else:
        S, cs = {}, []
        for y in years:
            g, c = stn_daily(y, stns)
            S[y] = g
            cs.append(c)
        C = pd.concat(cs).groupby(level=0).first()
        OUT_E.mkdir(parents=True, exist_ok=True)
        pickle.dump((S, C), open(STN_CACHE, 'wb'))
    return D, W, S, C


def load_grids():
    """격자 산출물(BC·BC-G·IDW_AWS·GPM).  cpuserver 마운트가 필요하다."""
    import xarray as xr
    if not os.path.exists(NCBC):
        raise SystemExit(
            f'격자 자료를 찾지 못했습니다:\n  {NCBC}\n'
            'cpuserver 를 마운트한 뒤 다시 실행하거나, grid·graph 를 빼고\n'
            '  python3 report.py map series sums kge\n'
            '로 실행하세요.')
    ds = xr.open_dataset(NCBC)
    G = {'arr': {'BC_G': ds['BC_2'].values, 'BC': ds['BC_1'].values},
         'time': pd.to_datetime(ds.time.values),
         'lat': ds.lat.values.astype(float),
         'lon': ds.lon.values.astype(float)}
    ds.close()
    d2 = xr.open_dataset(B.NC)                   # IDW_AWS · GPM 은 본 파일에
    assert len(pd.to_datetime(d2.time.values)) == len(G['time']), '기간이 다릅니다'
    for v in ('AWS', 'GPM'):
        G['arr'][v] = d2[v].values
    d2.close()
    assert set(PROD) <= set(G['arr']), '읽지 못한 산출물이 있습니다'
    return G


def event_span(s: pd.Series, peak: str):
    """peak 를 품은 연속 강우 구간."""
    i = s.index.get_loc(pd.Timestamp(peak))
    a = b = i
    while a - 1 >= 0 and s.iloc[a - 1] >= WET:
        a -= 1
    while b + 1 < len(s) and s.iloc[b + 1] >= WET:
        b += 1
    return s.index[a], s.index[b]


def periods(a, b, peak):
    """(이름, 시작, 끝) 네 구간.  마지막은 사상 전체."""
    p = pd.Timestamp(peak)
    d = pd.Timedelta(days=1)
    return [('피크 전', a, p - d), ('피크 중', p, p),
            ('피크 후', p + d, b), ('사상 전체', a, b)]


def prep(name, D, W, S, C, G):
    """유역 하나에 필요한 값을 한 번에 모은다."""
    ww, _ = W[name]
    stn = sorted(ww)
    a, b = event_span(D['series'][name]['ASOS'], PEAK[name])
    days = pd.date_range(a, b, freq='D')
    cells = D['info'][name]['cells']

    k = pd.Index(G['time']).get_indexer(days)
    cv = {v: np.array([[G['arr'][v][t, i, j] for (i, j, _) in cells] for t in k])
          for v in PROD}
    ok = np.isfinite(cv['BC_G']).all(axis=1)     # 공통 유효일 (BC 계열만 결측)

    return dict(name=name, geom=D['geoms'][name], info=D['info'][name],
                cells=cells, stn=stn,
                w=np.array([ww[s] for s in stn]),
                area=np.array([c[2] for c in cells]),
                lon=C.loc[stn, '경도'].to_numpy(float),
                lat=C.loc[stn, '위도'].to_numpy(float),
                a=a, b=b, days=days, cell_v=cv,
                stn_v=S[a.year].reindex(days)[stn].to_numpy(float),
                ok=ok, per=periods(a, b, PEAK[name]))


def sums(P, lo, hi, only_ok: bool = True):
    """구간 누적 — 지점별 · 격자별 누적과, 그것을 면적가중한 유역 누적."""
    m = (P['days'] >= lo) & (P['days'] <= hi)
    n_all = int(m.sum())
    if only_ok:
        m = m & P['ok']
    v = P['stn_v'][m]
    if not np.isfinite(v).all():
        print(f'  [주의] {P["name"]} {lo:%m-%d}~{hi:%m-%d} 지점 결측 '
              f'{int((~np.isfinite(v)).sum())}건 — 그 지점은 그날을 빼고 더한다')
    sv = np.nansum(v, axis=0)
    cell = {q: P['cell_v'][q][m].sum(axis=0) for q in PROD}
    mean = {q: float(cell[q] @ P['area'] / P['area'].sum()) for q in PROD}
    return dict(days=P['days'][m], n_all=n_all, stn=sv, cell=cell, mean=mean,
                thi=float(sv @ P['w']))


def thi_daily(sv, w):
    """일별 티센 유역평균.  결측 지점이 있으면 남은 가중치로 나눠 준다."""
    ok = np.isfinite(sv)
    den = (ok * w).sum(axis=1)
    num = np.nansum(np.where(ok, sv, 0.0) * w, axis=1)
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


# ══════════════════════════════════════════════════════════════════════════
# § 3  지도 도구
# ══════════════════════════════════════════════════════════════════════════
def inside_mask(geom, GX, GY):
    """유역 안쪽 판정 (구멍 제외)."""
    pts = np.column_stack([GX.ravel(), GY.ravel()])
    ins = MplPath(np.asarray(geom.exterior.coords)).contains_points(pts)
    for h in geom.interiors:
        ins &= ~MplPath(np.asarray(h.coords)).contains_points(pts)
    return ins.reshape(GX.shape)


def nearest_stn(GX, GY, lon, lat):
    """화면 격자마다 최근접 지점 번호 (티센 분할)."""
    px, py = to5179(GX.ravel(), GY.ravel())
    sx, sy = to5179(np.asarray(lon), np.asarray(lat))
    d2 = (px[:, None] - sx[None, :]) ** 2 + (py[:, None] - sy[None, :]) ** 2
    return np.argmin(d2, axis=1).reshape(GX.shape)


def view_extent(P, G):
    """유역 · 중첩 격자 · 기여 지점을 모두 담는 화면 범위."""
    x0, y0, x1, y1 = P['geom'].bounds
    cx = [G['lon'][j] for _, j, _ in P['cells']]
    cy = [G['lat'][i] for i, _, _ in P['cells']]
    x0 = min(x0, min(cx) - .05, P['lon'].min())
    x1 = max(x1, max(cx) + .05, P['lon'].max())
    y0 = min(y0, min(cy) - .05, P['lat'].min())
    y1 = max(y1, max(cy) + .05, P['lat'].max())
    mx, my = (x1 - x0) * .04, (y1 - y0) * .04
    return x0 - mx, x1 + mx, y0 - my, y1 + my


def raster(P, G, ext, n: int = 520):
    """화면을 잘게 나눠 유역 안팎 · 최근접지점 · 소속격자를 배정한다."""
    x0, x1, y0, y1 = ext
    gx = np.linspace(x0, x1, n)
    gy = np.linspace(y0, y1, max(int(n * (y1 - y0) / (x1 - x0)), 8))
    GX, GY = np.meshgrid(gx, gy)

    ins = inside_mask(P['geom'], GX, GY)
    near = nearest_stn(GX, GY, P['lon'], P['lat'])

    kk = np.full(GX.shape, -1)
    ii = np.rint((gy[:, None] - G['lat'][0]) / .1).astype(int)
    jj = np.rint((gx[None, :] - G['lon'][0]) / .1).astype(int)
    for k, (i, j, _) in enumerate(P['cells']):
        kk[(ii == i) & (jj == j)] = k
    return gx, gy, ins, near, kk


def ctx_cells(P, G, ext, days, var):
    """화면 안 0.1° 격자의 구간 누적 — 유역 밖까지 옅게 깔아 줄 배경."""
    x0, x1, y0, y1 = ext
    js = np.where((G['lon'] > x0 - .05) & (G['lon'] < x1 + .05))[0]
    iss = np.where((G['lat'] > y0 - .05) & (G['lat'] < y1 + .05))[0]
    k = pd.Index(G['time']).get_indexer(pd.DatetimeIndex(days))
    a = G['arr'][var][np.ix_(k, iss, js)]
    v = np.where(np.isfinite(a).all(axis=0), a.sum(axis=0), np.nan)
    ex = np.r_[G['lon'][js] - .05, G['lon'][js][-1] + .05]
    ey = np.r_[G['lat'][iss] - .05, G['lat'][iss][-1] + .05]
    return ex, ey, v


def label_spots(P, ext, pw, ph, fs):
    """지점 이름표 자리(축 비율 좌표).

    지점마다 여덟 방향 · 두 거리의 후보를 놓고 ① 다른 이름표와 겹침
    ② 지점 표시를 덮음 ③ 패널 밖으로 나감 ④ 제 지점에서 멀어짐 순으로
    벌점을 매겨 가장 싼 자리를 고른다.  가중치가 큰 지점부터 자리를 잡는다.
    """
    x0, x1, y0, y1 = ext
    hw = (2.0 * fs + 6 * fs * .56 + 8) / 72 / 2 / pw    # 이름표 반너비(비율)
    hh = (3 * fs * 1.25 + 8) / 72 / 2 / ph              # 이름표 반높이(비율)
    fx = (P['lon'] - x0) / (x1 - x0)
    fy = (P['lat'] - y0) / (y1 - y0)
    n, gap = len(fx), .018

    cand = []
    for k in (1.0, 1.75):
        cand += [(0, k * (hh + gap)), (0, -k * (hh + gap)),
                 (k * (hw + gap), 0), (-k * (hw + gap), 0),
                 (k * hw * .95, k * hh * 1.15), (-k * hw * .95, k * hh * 1.15),
                 (k * hw * .95, -k * hh * 1.15), (-k * hw * .95, -k * hh * 1.15)]

    def ov(ax_, ay_, bx_, by_, w, h):
        return (max(0., w - abs(ax_ - bx_)) * max(0., h - abs(ay_ - by_))
                / (w * h))

    out, done = [None] * n, []
    for i in np.argsort(-np.asarray(P['w'])):           # 큰 지점부터 좋은 자리
        best, cost = None, 1e18
        for dx, dy in cand:
            lx, ly = fx[i] + dx, fy[i] + dy
            cx = float(np.clip(lx, hw + .006, 1 - hw - .006))
            cy = float(np.clip(ly, hh + .006, 1 - hh - .006))
            c = 40 * (abs(cx - lx) / hw + abs(cy - ly) / hh)   # 밖으로 밀린 만큼
            for bx_, by_ in done:                             # 이름표끼리
                c += 120 * ov(cx, cy, bx_, by_, 2 * hw, 2 * hh)
            for j in range(n):                                # 지점 표시 위
                c += 45 * ov(cx, cy, fx[j], fy[j], hw + .012, hh + .014)
            c += 3 * np.hypot(dx / hw, dy / hh)               # 제 지점과의 거리
            if c < cost:
                cost, best = c, (cx, cy)
        out[i] = best
        done.append(best)
    return out


# ══════════════════════════════════════════════════════════════════════════
# § 4-1  G — 격자 비교도  (티센 · IDW_AWS · BC-G · BC · GPM)
# ══════════════════════════════════════════════════════════════════════════
def fig_grid(P, G, path):
    per = P['per']
    S = [sums(P, lo, hi) for _, lo, hi in per]
    nc, nr = len(per), 1 + len(PROD)

    ext = view_extent(P, G)
    x0, x1, y0, y1 = ext
    asp = 1 / np.cos(np.deg2rad((y0 + y1) / 2))
    ratio = (y1 - y0) * asp / (x1 - x0)          # 패널 높이 / 너비

    PW = min(6.8, 6.4 / ratio)
    PH = PW * ratio
    L, R, GXP = 1.85, .5, .40
    TOP, TIT, GYP = 2.15, 1.60, 1.02
    CG, CH, CL, LEG, BOT = 1.05, .38, .78, .95, .35
    W = L + nc * PW + (nc - 1) * GXP + R
    H = TOP + TIT + nr * PH + (nr - 1) * GYP + CG + CH + CL + LEG + BOT

    fig = plt.figure(figsize=(W, H))
    bx = boxer(fig, W, H)

    gx, gy, ins, near, kk = raster(P, G, ext)
    spot = label_spots(P, ext, PW, PH, FSM - 8)
    ring = np.asarray(P['geom'].exterior.coords)
    CTX = {(c, v): ctx_cells(P, G, ext, S[c]['days'], v)
           for c in range(nc) for v in PROD}

    vmax = max(max(s['stn'].max(), max(s['cell'][v].max() for v in PROD),
                   max(np.nanmax(CTX[(c, v)][2]) for v in PROD))
               for c, s in enumerate(S))
    lv = rain_levels(vmax)
    nrm = BoundaryNorm(lv, CMAP.N)

    for c, ((nm, lo, hi), s) in enumerate(zip(per, S)):
        xi = L + c * (PW + GXP)
        for r in range(nr):
            ax = bx(xi, TOP + TIT + r * (PH + GYP), PW, PH)
            col = C_THI if r == 0 else PCOL[PROD[r - 1]]

            if r == 0:                                   # 티센 — 지점 기반
                ax.pcolormesh(gx, gy, s['stn'][near], cmap=CMAP, norm=nrm,
                              shading='auto', alpha=FADE, zorder=1)
                ax.pcolormesh(gx, gy, np.where(ins, s['stn'][near], np.nan),
                              cmap=CMAP, norm=nrm, shading='auto', zorder=2)
                ax.contour(gx, gy, near.astype(float),
                           levels=np.arange(len(P['stn'])) - .5,
                           colors='white', linewidths=2.2, zorder=3)
            else:                                        # 0.1° 격자 산출물
                v = PROD[r - 1]
                ex, ey, ctx = CTX[(c, v)]
                ax.pcolormesh(ex, ey, ctx, cmap=CMAP, norm=nrm,
                              shading='flat', alpha=FADE, zorder=1)
                ax.pcolormesh(gx, gy,
                              np.where(ins & (kk >= 0),
                                       s['cell'][v][np.maximum(kk, 0)], np.nan),
                              cmap=CMAP, norm=nrm, shading='auto', zorder=2)
                for xe in ex:
                    ax.axvline(xe, color='white', lw=2.0, zorder=3)
                for ye in ey:
                    ax.axhline(ye, color='white', lw=2.0, zorder=3)

            ax.plot(ring[:, 0], ring[:, 1], color='#111111', lw=3.4, zorder=5)
            for h in P['geom'].interiors:
                hh = np.asarray(h.coords)
                ax.plot(hh[:, 0], hh[:, 1], color='#111111', lw=2.2, zorder=5)

            if r == 0:                                   # 지점 표시 + 값
                for k, st in enumerate(P['stn']):
                    ax.plot(P['lon'][k], P['lat'][k],
                            marker='*' if st < 300 else 'o',
                            ms=34 if st < 300 else 19, mfc='#D1495B',
                            mec='#111111', mew=2.2, ls='none', zorder=6)
                    ax.annotate(f'{st}\n{s["stn"][k]:.0f} mm\n'
                                f'(가중 {P["w"][k] * 100:.0f}%)',
                                xy=(P['lon'][k], P['lat'][k]),
                                xytext=spot[k], textcoords='axes fraction',
                                ha='center', va='center', fontsize=FSM - 8,
                                fontweight='bold', zorder=7, linespacing=1.18,
                                color=TXT_FC, path_effects=stroke(4.6, TXT_EC),
                                arrowprops=dict(arrowstyle='-', color=TXT_FC,
                                                lw=2.0, ls=':', shrinkA=2,
                                                shrinkB=9,
                                                path_effects=stroke(4.0, TXT_EC)))
            else:                                        # 격자 값
                tot = P['area'].sum()
                for k, (i, j, a) in enumerate(P['cells']):
                    cval = s['cell'][PROD[r - 1]][k]
                    # 면적 기여율은 행마다 같으므로 첫 격자 행에만 적는다
                    lab = (f'{cval:.0f} mm\n(면적 {a / tot * 100:.0f}%)'
                           if r == 1 else f'{cval:.0f} mm')
                    ax.text(G['lon'][j], G['lat'][i], lab, ha='center',
                            va='center', fontsize=FSM - 8, fontweight='bold',
                            zorder=7, linespacing=1.18, color=TXT_FC,
                            path_effects=stroke(4.2, TXT_EC))

            val = s['thi'] if r == 0 else s['mean'][PROD[r - 1]]
            txt = f'유역 누적 {val:.0f} mm'
            if r:
                txt += f'\n티센 {pct(val, s["thi"])}'
                if PROD[r - 1] in FOLLOW:
                    txt += f'   ·   IDW_AWS {pct(val, s["mean"][TARGET])}'
            ax.text(.5, -.028, txt, transform=ax.transAxes, ha='center',
                    va='top', fontsize=FSM - 4, fontweight='bold', color=col,
                    linespacing=1.35)

            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_aspect(asp)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(1.6)
            if c == 0:
                ax.set_ylabel(LAB_T if r == 0 else PLAB[PROD[r - 1]],
                              fontsize=FSM, fontweight='bold', color=col,
                              labelpad=16)
            if r == 0:
                nd = len(s['days'])
                sub = f'{lo:%m-%d}' if lo == hi else f'{lo:%m-%d} ~ {hi:%m-%d}'
                cnt = f'{nd}일' if nd == s['n_all'] else f"{s['n_all']}일 중 {nd}일"
                ax.set_title(f'{nm}\n{sub}  ({cnt})', fontsize=FSM + 3,
                             fontweight='bold', pad=16)

    # ── 그림 하나에 색눈금 하나
    span = nc * PW + (nc - 1) * GXP
    cw = .58 * span
    cax = bx(L + (span - cw) / 2, TOP + TIT + nr * PH + (nr - 1) * GYP + CG,
             cw, CH)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap=CMAP), cax=cax,
                      orientation='horizontal', ticks=lv, spacing='uniform')
    cb.ax.set_xticklabels([f'{v:.0f}' for v in lv])
    cax.set_xlabel('구간 누적 강수량 (mm)  —  다섯 자료·네 구간이 같은 색눈금을 쓴다',
                   fontsize=FSM - 4, labelpad=8)
    cax.tick_params(labelsize=FSM - 8)

    drop = [d for d in P['days'] if d not in set(S[-1]['days'])]
    note = (f'BC 계열 결측일 제외 → {", ".join(f"{d:%m-%d}" for d in drop)}'
            if drop else 'BC 계열 결측 없음')
    inf = P['info']
    fig.text(.5, 1 - .42 / H,
             f'{P["name"]}   {P["a"]:%Y-%m-%d} ~ {P["b"]:%Y-%m-%d} 호우사상'
             '  —  티센 · IDW_AWS · BC-G · BC · GPM',
             ha='center', va='top', fontsize=FSM + 8, fontweight='bold')
    fig.text(.5, 1 - 1.12 / H,
             f'유역면적 {inf["area"]:.0f} km²   ·   티센 기여지점 {len(P["stn"])}개소'
             f'   ·   0.1° 중첩격자 {inf["ncell"]}칸   ·   {note}',
             ha='center', va='top', fontsize=FSM - 1, color='#333333')

    fig.legend(handles=[
        Line2D([], [], color='#111111', lw=3.4, label='표준유역 경계'),
        Line2D([], [], color='#BBBBBB', lw=2.6,
               label='흰 선 = 티센 분할 · 0.1° 격자 경계'),
        Line2D([], [], marker='*', ms=22, ls='none', mfc='#D1495B',
               mec='#111111', label='종관관측 ASOS'),
        Line2D([], [], marker='o', ms=14, ls='none', mfc='#D1495B',
               mec='#111111', label='방재관측 AWS'),
        Patch(fc='#BBD6E8', ec='#888888',
              label='진한 색 = 유역 안 (유역 누적에 쓰인 값)'),
        Patch(fc='#EAF2F8', ec='#CCCCCC', label='옅은 색 = 유역 밖 (참고)'),
        Patch(fc='none', ec='none', label='(면적 %) = 그 칸의 유역 면적 기여율'),
    ], loc='lower center', bbox_to_anchor=(.5, (BOT * .25) / H), ncol=4,
        fontsize=FSM - 6, framealpha=.95)

    fig.savefig(path, dpi=150, facecolor='white')
    plt.close(fig)
    return S


# ══════════════════════════════════════════════════════════════════════════
# § 4-2  C — 비교 그래프  (좌 일별 · 우 구간 누적)
# ══════════════════════════════════════════════════════════════════════════
def fig_graph(PS, G, path):
    n = len(PS)
    PW1, PW2 = 16.0, 11.0
    L, R, CGP = 1.45, .5, 1.6
    TOP, RGP, BOT = 3.25, 2.35, 1.6
    PH = 6.6
    W = L + PW1 + CGP + PW2 + R
    H = TOP + n * PH + (n - 1) * RGP + BOT

    fig = plt.figure(figsize=(W, H))
    bx = boxer(fig, W, H)
    shade = {'피크 전': '#E4EEF8', '피크 중': '#FBE0C2', '피크 후': '#E4F0E0'}
    nb = len(SER)
    off = (np.arange(nb) - (nb - 1) / 2) * (.84 / nb)

    for r, P in enumerate(PS):
        yi = TOP + r * (PH + RGP)
        win = pd.date_range(P['days'][0] - pd.Timedelta(days=PAD_G),
                            P['days'][-1] + pd.Timedelta(days=PAD_G), freq='D')
        k = pd.Index(G['time']).get_indexer(win)
        day = {}
        for v in PROD:
            cv = np.array([[G['arr'][v][t, i, j] for (i, j, _) in P['cells']]
                           for t in k])
            day[v] = np.where(np.isfinite(cv).all(axis=1),
                              cv @ P['area'] / P['area'].sum(), np.nan)
        sv = P['S'][P['a'].year].reindex(win)[P['stn']].to_numpy(float)
        day['THI'] = thi_daily(sv, P['w'])

        # ── 왼쪽: 일별
        ax = bx(L, yi, PW1, PH)
        x = np.arange(len(win))
        wl = list(win)
        for q, (nm, lo, hi) in enumerate(P['per'][:3]):
            if hi < lo:
                continue
            i0, i1 = wl.index(lo), wl.index(hi)
            ax.axvspan(i0 - .5, i1 + .5, color=shade[nm], zorder=0)
            ax.text((i0 + i1) / 2, .975 if q % 2 == 0 else .875, nm,
                    transform=ax.get_xaxis_transform(), ha='center', va='top',
                    fontsize=FSG - 3, fontweight='bold', color='#444444',
                    path_effects=stroke(3.4, 'white'))
        for q, v in enumerate(SER):
            ax.bar(x + off[q], np.nan_to_num(day[v]), width=.84 / nb,
                   color=SCOL[v], ec='#333333', lw=.8, zorder=2)
        ms = x[~np.isfinite(day['BC_G'])]
        if len(ms):
            ax.plot(ms + off[1], np.zeros(len(ms)), marker='x', ms=15, mew=3.4,
                    ls='none', color=SCOL['BC_G'], zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{d:%m-%d}' for d in win], rotation=55,
                           ha='right', fontsize=FSG - 5)
        ax.set_ylabel('일강수량 (mm)', fontsize=FSG)
        ax.tick_params(axis='y', labelsize=FSG - 4)
        ax.set_title(f'{P["name"]}  —  일별   '
                     f'(사상 {P["a"]:%Y-%m-%d} ~ {P["b"]:%Y-%m-%d})',
                     fontsize=FSG + 4, fontweight='bold', pad=14)
        ax.grid(axis='y', alpha=.3, ls=':')
        ax.set_xlim(-.7, len(win) - .3)
        ax.set_ylim(0, np.nanmax([np.nanmax(day[v]) for v in SER]) * 1.22)

        # ── 오른쪽: 구간 누적
        ax = bx(L + PW1 + CGP, yi, PW2, PH)
        nmz = [p[0] for p in P['per']]
        val = {v: [s['thi'] if v == 'THI' else s['mean'][v] for s in P['S4']]
               for v in SER}
        xb = np.arange(len(nmz))
        top = max(max(val[v]) for v in SER)
        for q, v in enumerate(SER):
            ax.bar(xb + off[q], val[v], width=.84 / nb, color=SCOL[v],
                   ec='#333333', lw=1.0)
            for i in range(len(nmz)):
                ax.text(xb[i] + off[q], val[v][i] + top * .022,
                        f'{val[v][i]:.0f}', ha='center', rotation=90,
                        va='bottom', fontsize=FSG - 7, fontweight='bold',
                        color=SCOL[v])
        ax.set_xticks(xb)
        ax.set_xticklabels(
            [f'{nm}\n(' + (f'{len(s["days"])}일' if len(s['days']) == s['n_all']
                           else f'{s["n_all"]}일 중 {len(s["days"])}일') + ')'
             for nm, s in zip(nmz, P['S4'])], fontsize=FSG - 4)
        ax.set_ylabel('구간 누적 강수량 (mm)', fontsize=FSG)
        ax.tick_params(axis='y', labelsize=FSG - 4)
        ax.set_ylim(0, top * 1.30)
        ax.set_title(f'{P["name"]}  —  구간 누적', fontsize=FSG + 4,
                     fontweight='bold', pad=14)
        ax.grid(axis='y', alpha=.3, ls=':')

    fig.text(.5, 1 - .40 / H,
             '세 유역 호우사상  —  티센 · IDW_AWS · BC-G · BC · GPM',
             ha='center', va='top', fontsize=FSG + 9, fontweight='bold')
    fig.text(.5, 1 - 1.05 / H,
             '구간 누적은 BC 계열이 유효한 날만 다섯 자료 모두에서 더한 값이다',
             ha='center', va='top', fontsize=FSG - 1, color='#333333')
    fig.legend(handles=[Patch(fc=SCOL[v], ec='#333333', label=SLAB[v])
                        for v in SER]
               + [Line2D([], [], marker='x', ms=15, mew=3.4, ls='none',
                         color=SCOL['BC_G'], label='BC 계열 결측일')],
               loc='upper center', bbox_to_anchor=(.5, 1 - 1.55 / H), ncol=5,
               fontsize=FSG - 1, framealpha=.95)

    fig.savefig(path, dpi=150, facecolor='white')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# § 4-3  M — 유역 배치도  (티센 분할 · 0.1° 격자 · 관측소)
# ══════════════════════════════════════════════════════════════════════════
def fig_layout(name, D, W, C, path):
    geom, info = D['geoms'][name], D['info'][name]
    ww, _ = W[name]
    stn = sorted(ww)
    lon = C.loc[stn, '경도'].to_numpy()
    lat = C.loc[stn, '위도'].to_numpy()

    fig, ax = plt.subplots(figsize=(13, 11.5))
    x0, y0, x1, y1 = geom.bounds
    gx = np.linspace(x0, x1, 420)
    gy = np.linspace(y0, y1, int(420 * (y1 - y0) / (x1 - x0)) + 1)
    GX, GY = np.meshgrid(gx, gy)
    near = nearest_stn(GX, GY, lon, lat).astype(float)
    near[~inside_mask(geom, GX, GY)] = np.nan
    ax.pcolormesh(gx, gy, near, cmap=plt.get_cmap('Pastel1', max(len(stn), 3)),
                  vmin=-.5, vmax=len(stn) - .5, shading='auto', zorder=1)

    for ilat, ilon, a in info['cells']:                  # 0.1° 산출 격자
        cx, cy = D['lon'][ilon], D['lat'][ilat]
        ax.add_patch(Rectangle((cx - .05, cy - .05), .1, .1, fill=False,
                               ec='#2E86AB', lw=2.4, ls='--', zorder=3))
        ax.text(cx, cy + .043, f'{a / info["area"] * 100:.0f}%', ha='center',
                va='top', fontsize=FSM - 5, color='#2E86AB',
                fontweight='bold', zorder=4)

    r = np.asarray(geom.exterior.coords)
    ax.plot(r[:, 0], r[:, 1], color='#111111', lw=3.4, zorder=5)
    for h in geom.interiors:
        hh = np.asarray(h.coords)
        ax.plot(hh[:, 0], hh[:, 1], color='#111111', lw=2.0, zorder=5)

    for k, s in enumerate(stn):
        asos = s < 300
        ax.plot(lon[k], lat[k], marker='*' if asos else 'o',
                ms=32 if asos else 20, mfc='#D1495B' if asos else '#FFFFFF',
                mec='#111111', mew=2.2, ls='none', zorder=6)
        ax.annotate(f'{s}\n{ww[s] * 100:.0f}%', (lon[k], lat[k]),
                    textcoords='offset points', xytext=(0, 26), ha='center',
                    fontsize=FSM - 4, fontweight='bold', zorder=7,
                    bbox=dict(fc='white', ec='#999999', alpha=.85, pad=1.6))

    x0 = min(x0, lon.min()) - .03
    x1 = max(x1, lon.max()) + .03
    y0 = min(y0, lat.min()) - .03
    y1 = max(y1, lat.max()) + .03
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect(1 / np.cos(np.deg2rad((y0 + y1) / 2)))
    ax.set_xlabel('경도 (°E)')
    ax.set_ylabel('위도 (°N)')
    ax.set_title(f'{name}  —  티센 분할 · 0.1° 산출격자 · 강우관측소\n'
                 f'유역면적 {info["area"]:.0f} km²  ·  기여 지점 {len(stn)}개소'
                 f'  ·  중첩 격자 {info["ncell"]}칸', pad=16)
    ax.grid(alpha=.25, ls=':')
    ax.legend(handles=[
        Line2D([], [], color='#111111', lw=3.4, label='표준유역 경계'),
        Line2D([], [], color='#2E86AB', lw=2.4, ls='--',
               label='0.1° 산출격자 (% = 면적 기여율)'),
        Line2D([], [], marker='*', ms=22, ls='none', mfc='#D1495B',
               mec='#111111', label='종관관측 ASOS'),
        Line2D([], [], marker='o', ms=14, ls='none', mfc='#FFFFFF',
               mec='#111111', label='방재관측 AWS'),
    ], loc='upper center', bbox_to_anchor=(.5, -.10), ncol=2, framealpha=.92,
        fontsize=FSM - 4)
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor='white')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# § 4-4  T · S — 사상 전후 시계열과 사상 누적  (7종 비교)
# ══════════════════════════════════════════════════════════════════════════
def window7(s: pd.Series, peak: str):
    """사상 구간 (a, b) 과 앞뒤를 덧붙인 표시 구간 (lo, hi)."""
    a, b = event_span(s, peak)
    i0, i1 = s.index.get_loc(a), s.index.get_loc(b)
    lo = s.index[max(0, i0 - PAD_S)]
    hi = s.index[min(len(s) - 1, i1 + PAD_S)]
    return a, b, lo, hi


def fig_series(D, spans, path):
    fig, axes = plt.subplots(3, 1, figsize=(15, 20))
    for ax, name in zip(axes, BASINS):
        s = D['series'][name]
        a, b, lo, hi = spans[name]
        w = s.loc[lo:hi]
        x = np.arange(len(w))
        ax.axvspan(list(w.index).index(a) - .5, list(w.index).index(b) + .5,
                   color='#FDE2C0', alpha=.75, zorder=0, label='사상 구간')
        ax.bar(x, w['ASOS'], color='#CCCCCC', width=.78, zorder=1,
               label=LAB7['ASOS'])
        for v in SHOW7[1:]:
            ax.plot(x, w[v], marker='o', ms=6, lw=2.4, color=COL7[v],
                    label=LAB7[v], zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([d.strftime('%m-%d') for d in w.index],
                           rotation=45, ha='right')
        ax.set_ylabel('일강수량 (mm)')
        miss = x[w['BC_G'].isna().to_numpy()]
        if len(miss):
            ax.plot(miss, np.zeros(len(miss)), marker='x', ms=13, mew=3.2,
                    ls='none', color=COL7['BC_G'], zorder=3,
                    label='BC-G 결측' if ax is axes[0] else None)
        ax.set_title(f'{name}  —  사상 구간 {a.date()} ~ {b.date()}', pad=10)
        ax.grid(axis='y', alpha=.3, ls=':')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, fontsize=FSM - 4, framealpha=.95,
               loc='upper center', bbox_to_anchor=(.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, .945))
    fig.savefig(path, dpi=170, facecolor='white')
    plt.close(fig)


def event_sums7(D, spans) -> pd.DataFrame:
    """사상 구간 합계.  '_n' 은 유효일수, '_c' 는 공통 유효일 합이다."""
    rows = []
    for name in BASINS:
        s = D['series'][name]
        a, b, _, _ = spans[name]
        w = s.loc[a:b]
        ok = w[SHOW7].notna().all(axis=1)
        r = {'유역': name, '기간': f'{a.date()} ~ {b.date()}',
             '일수': len(w), '공통일수': int(ok.sum())}
        for v in SHOW7:
            r[v] = w[v].sum(min_count=1)
            r[v + '_n'] = int(w[v].notna().sum())
            r[v + '_c'] = w.loc[ok, v].sum(min_count=1)
        rows.append(r)
    return pd.DataFrame(rows)


def fig_sums(T, path):
    fig, axes = plt.subplots(1, 3, figsize=(20.5, 8.5))
    for ax, (_, r) in zip(axes, T.iterrows()):
        v = [r[k] for k in SHOW7]
        ax.bar(range(len(SHOW7)), v, color=[COL7[k] for k in SHOW7],
               ec='#333333', lw=1.2)
        ax.axhline(r['ASOS'], color='#111111', lw=2.0, ls='--')
        for k, y in enumerate(v):
            miss = r[SHOW7[k] + '_n'] < r['일수']
            ax.text(k, y + max(v) * .025, f'{y:.0f}' + ('*' if miss else ''),
                    ha='center', fontsize=FSM - 5, fontweight='bold')
        ax.set_xticks(range(len(SHOW7)))
        ax.set_xticklabels([LAB7[k].replace(' (기준)', '') for k in SHOW7],
                           rotation=40, ha='right')
        ax.set_ylabel('사상 누적 강수량 (mm)')
        ax.set_title(f'{r["유역"]}\n{r["기간"]}  ({r["일수"]}일)', pad=10,
                     fontsize=FSM - 2)
        ax.grid(axis='y', alpha=.3, ls=':')
        ax.set_ylim(0, max(v) * 1.16)
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor='white')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# § 4-5  N1 — 전국 표준유역 KGE 지도  (보고서 그림 21 교체본)
# ══════════════════════════════════════════════════════════════════════════
KGE_VMIN, KGE_VMAX = 0.4, 1.0
ISLAND = 3e-3        # 경계선에 남길 최소 면적 (deg², 약 30 km²)
C_NA = '#E6E8EB'     # 평가 대상이 아닌 유역


def basin_shapes():
    """표준유역 도형과 코드.  세 패널이 같이 쓰도록 한 번만 읽는다."""
    recs = B.read_dbf(N.SHP + '.dbf')
    offs = B.shp_offsets(N.SHP)
    key = next(k for k in recs[0] if 'BAS' in k.upper() or 'CD' in k.upper())
    out = []
    for rec, (o, l) in zip(recs, offs):
        try:
            out.append((rec[key], B.read_polygon(N.SHP, o, l)))
        except Exception:
            continue
    return out


def national_border(shapes):
    """유역 850개를 합쳐 만든 남한 경계선 (외곽 + 섬)."""
    u = unary_union([g for _, g in shapes])
    gs = [u] if u.geom_type == 'Polygon' else list(u.geoms)
    rings = []
    for g in gs:
        if g.area < ISLAND:        # 점처럼 찍히는 작은 섬은 선이 뭉쳐 뺀다
            continue
        rings.append(np.asarray(g.exterior.coords))
        rings += [np.asarray(h.coords) for h in g.interiors]
    return rings, u.bounds


def fig_kge_map(path):
    S = N.stats()
    per = S['per_basin'].set_index('code')
    shapes = basin_shapes()
    rings, (x0, y0, x1, y1) = national_border(shapes)
    mx, my = (x1 - x0) * .035, (y1 - y0) * .030

    plt.rcParams.update({'font.size': FSM, 'axes.titlesize': FSM + 3,
                         'axes.titleweight': 'bold',
                         'xtick.labelsize': FSM - 6, 'ytick.labelsize': FSM - 6})
    fig, axs = plt.subplots(1, 3, figsize=(21, 10.2))
    for k, (ax, p) in enumerate(zip(axs, N.PRODS)):
        pats, vals, na = [], [], []
        for c, g in shapes:
            ok = c in per.index and np.isfinite(per.loc[c, p])
            for gg in ([g] if g.geom_type == 'Polygon' else list(g.geoms)):
                poly = MplPoly(np.asarray(gg.exterior.coords), closed=True)
                (pats if ok else na).append(poly)
                if ok:
                    vals.append(per.loc[c, p])
        ax.add_collection(PatchCollection(na, facecolor=C_NA,
                                          edgecolor='#8A97A5', linewidth=.35,
                                          zorder=2))
        pc = PatchCollection(pats, cmap='RdYlBu', edgecolor='#8A97A5',
                             linewidth=.35, zorder=3)
        pc.set_array(np.asarray(vals))
        pc.set_clim(KGE_VMIN, KGE_VMAX)
        ax.add_collection(pc)

        for r in rings:                       # 남한 경계선
            ax.plot(r[:, 0], r[:, 1], color='#111111', lw=1.5, zorder=5,
                    solid_joinstyle='round')

        ax.set_xlim(x0 - mx, x1 + mx)
        ax.set_ylim(y0 - my, y1 + my)
        ax.set_aspect(1 / np.cos(np.deg2rad((y0 + y1) / 2)))
        ax.set_xticks(np.arange(126, 131))
        ax.set_yticks(np.arange(33, 39))
        ax.set_xticklabels([f'{v:.0f}°' for v in np.arange(126, 131)])
        ax.set_yticklabels([f'{v:.0f}°' for v in np.arange(33, 39)])
        ax.set_xlabel('경도 (°E)', fontsize=FSM - 4, labelpad=8)
        if k == 0:
            ax.set_ylabel('위도 (°N)', fontsize=FSM - 4, labelpad=8)
        else:
            ax.tick_params(labelleft=False)
        ax.grid(True, ls=':', lw=1.1, color='#8A97A5', alpha=.55, zorder=1)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(1.5)
            s.set_color('#333333')
        ax.set_title(f'{N.LABEL[p]}\n중앙값 KGE {per[p].median():.3f}', pad=14)

    fig.legend(handles=[
        Patch(fc=C_NA, ec='#8A97A5', label='평가 제외 (기준자료 부족)'),
        Line2D([], [], color='#111111', lw=1.8, label='남한 경계')],
        loc='lower center', bbox_to_anchor=(.45, .012), ncol=2,
        fontsize=FSM - 6, framealpha=.95)
    cb = fig.colorbar(pc, ax=axs, fraction=.021, pad=.022,
                      ticks=np.arange(.4, 1.01, .1))
    cb.set_label('KGE  (IDW_ASOS 기준, 2022~2025)', fontsize=FSM - 1,
                 labelpad=14)
    cb.ax.tick_params(labelsize=FSM - 6)
    cb.outline.set_linewidth(1.3)

    fig.savefig(path, dpi=190, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    return {N.LABEL[p]: float(per[p].median()) for p in N.PRODS}


# ══════════════════════════════════════════════════════════════════════════
# § 5  실행
# ══════════════════════════════════════════════════════════════════════════
NEEDS_GRID = {'grid', 'graph'}          # 격자 NetCDF 가 필요한 그림
ALL = ['map', 'series', 'sums', 'grid', 'graph', 'kge']


def run(which) -> None:
    set_font()
    OUT_E.mkdir(parents=True, exist_ok=True)
    OUT_B.mkdir(parents=True, exist_ok=True)
    want = set(which)

    if want & {'map', 'series', 'sums', 'grid', 'graph'}:
        D, W, S, C = load_basins()

    # ── M  유역 배치도
    if 'map' in want:
        for k, name in enumerate(BASINS, 1):
            p = OUT_E / f'M{k}_{name}_배치도.png'
            fig_layout(name, D, W, C, p)
            print('저장', p)

    # ── T · S  사상 전후 시계열 / 사상 누적
    if want & {'series', 'sums'}:
        spans = {n: window7(D['series'][n]['ASOS'], PEAK[n]) for n in BASINS}
        if 'series' in want:
            p = OUT_E / 'T1_사상전후_시계열.png'
            fig_series(D, spans, p)
            print('저장', p)
        if 'sums' in want:
            T = event_sums7(D, spans)
            p = OUT_E / 'S1_사상누적.png'
            fig_sums(T, p)
            print('저장', p)
            print(T[['유역', '기간', '일수'] + SHOW7].to_string(
                index=False, float_format=lambda v: f'{v:.1f}'))

    # ── G · C  격자 비교도 / 비교 그래프
    if want & NEEDS_GRID:
        G = load_grids()
        PS, rows = [], []
        for k, name in enumerate(BASINS, 1):
            P = prep(name, D, W, S, C, G)
            if 'grid' in want:
                p = OUT_E / f'G{k}_{name}_격자비교{TAG}.png'
                P['S4'] = fig_grid(P, G, p)
                print('저장', p)
            else:
                P['S4'] = [sums(P, lo, hi) for _, lo, hi in P['per']]
            P['S'] = S
            PS.append(P)
            for (nm, lo, hi), s in zip(P['per'], P['S4']):
                r = {'유역': name, '구간': nm, '기간': f'{lo:%m-%d}~{hi:%m-%d}',
                     '전체일수': s['n_all'], '사용일수': len(s['days']),
                     '티센': s['thi']}
                for v in PROD:
                    r[v] = s['mean'][v]
                    r[v + '_티센대비%'] = (100 * (s['mean'][v] / s['thi'] - 1)
                                       if s['thi'] >= 5 else np.nan)
                rows.append(r)

        if 'graph' in want:
            p = OUT_E / f'C1_티센_BCG_비교그래프{TAG}.png'
            fig_graph(PS, G, p)
            print('저장', p)

        T = pd.DataFrame(rows)
        T.to_csv(OUT_E / f'event_thiessen_vs_bcg{TAG}.csv', index=False,
                 encoding='utf-8-sig')
        print()
        print(T.to_string(index=False, float_format=lambda v: f'{v:.1f}'))

    # ── N1  전국 KGE 지도
    if 'kge' in want:
        p = OUT_B / 'N1_전국유역_KGE지도_경계선.png'
        med = fig_kge_map(p)
        print('저장', p)
        print('  중앙값 KGE  ' + '  '.join(f'{k} {v:.3f}' for k, v in med.items()))


def main() -> None:
    ap = argparse.ArgumentParser(
        description='KIHS 최종보고서 그림 생성 (인자 없으면 전부)')
    ap.add_argument('which', nargs='*', choices=ALL, help=' '.join(ALL))
    run(ap.parse_args().which or ALL)      # 인자가 없으면 전부


if __name__ == '__main__':
    main()
