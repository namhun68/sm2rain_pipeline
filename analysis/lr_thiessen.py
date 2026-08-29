#!/usr/bin/env python3
"""티센을 목표로 한 격자별 선형회귀와 전국 평가.

편향보정의 목표자료를 IDW_AWS 대신 티센 유역평균으로 바꾸면 어떻게 되는지
본다.  격자 한 칸마다 선형회귀를 따로 적합하고, 목표는 그 격자가 속한
표준유역의 티센 일강수다.  만들어진 격자장을 848개 표준유역으로 면적가중
집계해 티센 기준으로 BC-G · BC · BC-LR · TCA 와 나란히 평가한다.

    설명변수  SM2RAIN, ERA5, GPM, TCA      (BC 와 같은 입력, 지상관측 제외)
    목표      격자가 속한 표준유역의 티센 일강수
    적합      2021년
    평가      2022-01-01 ~ 2025-05-01

격자의 소속 유역은 겹치는 면적이 가장 큰 유역으로 정한다.  0.1° 한 칸이
약 100 km2 이고 표준유역 중앙값이 113 km2 라 한 칸이 여러 유역에 걸치지만,
목표를 하나로 정해야 회귀가 성립한다.

목표와 평가기준이 같은 자료이므로 티센 점수가 좋게 나오기 쉬운 판이다.
"더 낫다"가 아니라 "제 목표를 얼마나 따라가는가"로 읽고, 전국 확장에서
실제로 볼 것은 유역별 편차와 관측밀도 의존성이다.

산출
    KIHS/DATA/LR_THI_grid.nc          격자장 · 격자별 계수 · 적합일수
    KIHS/DATA/LR_THI_basin_eval.csv   유역별 지표

실행
    python3 lr_thiessen.py            격자장을 만들고 평가·그림까지
    python3 lr_thiessen.py --no-fit   이미 만든 격자장을 읽어 평가만
"""
from __future__ import annotations

import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import font_manager
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPoly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import basin_eval_core as B

for _c in ('AppleGothic', 'NanumGothic', 'Malgun Gothic'):
    if any(_c in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = _c
        break
plt.rcParams['axes.unicode_minus'] = False

# ────────────────────────────────────────────────────────────── 설정
ROOT = next((r for r in ('/home/cpuserver_data', '/Users/kim/cpuserver_data')
             if os.path.isdir(r)), '/Users/kim/cpuserver_data')
PJ = f'{ROOT}/personal_data/project_KIHS'
F_THI = f'{PJ}/data/thiessen/THIESSEN_basin_daily.nc'
WORK = '/Users/kim/Desktop/work/KIHS/DATA'
F_W = f'{WORK}/basin_cell_weights.pkl'      # 유역×격자 교차면적 (없으면 만든다)
F_OUT = f'{WORK}/LR_THI_grid.nc'            # 이 스크립트가 만드는 격자장

SAVE = None              # 그림 저장 폴더. None 이면 화면에만 띄운다
FIT_YEAR = '2021'
EVAL0, EVAL1 = '2022-01-01', '2025-05-01'
MIN_FIT = 60             # 적합에 쓸 수 있는 날이 이보다 적으면 그 격자는 비운다
MIN_AREA_FRAC = 0.5      # 유역 유효면적이 이보다 작은 날은 결측 (납품 CSV 와 같음)

X_BASE = ['SM2RAIN', 'ERA5', 'GPM', 'TCA']
PRODS = ['LR_THI', 'BC_G', 'BC', 'BC_LR', 'TCA']
LAB = {'LR_THI': 'LR (격자별, 목표 티센)', 'BC_G': 'BC-G', 'BC': 'BC',
       'BC_LR': 'BC_LR (격자별, 목표 AWS)', 'TCA': 'TCA (보정 전)',
       'THI': '티센'}
COL = {'LR_THI': '#0F7B8A', 'BC_G': '#8E3B46', 'BC': '#D1495B',
       'BC_LR': '#6C8EBF', 'TCA': '#EDAE49'}


# ────────────────────────────────────────────────────────────── 자료
def cell_weights():
    """유역×격자 교차면적.  한 번 만들어 캐시에 둔다."""
    if os.path.exists(F_W):
        return pickle.load(open(F_W, 'rb'))
    print('유역×격자 교차면적 계산 중...', end=' ', flush=True)
    recs = B.read_dbf(B.SHP + '.dbf')
    offs = B.shp_offsets(B.SHP)
    ds = xr.open_dataset(B.NC)
    lat, lon = ds.lat.values, ds.lon.values
    ds.close()
    W = {}
    for k, r in enumerate(recs):
        try:
            geom = B.read_polygon(B.SHP, *offs[k])
        except Exception:
            continue
        w = B.cell_weights(geom, lat, lon)
        if not w:
            continue
        W[r['SBSN_CD'].strip()] = {
            'w': w, 'name': r['SBSN_NM'].strip(), 'BBSN': r['BBSN_CD'].strip(),
            'cx': float(geom.centroid.x), 'cy': float(geom.centroid.y),
            'area': float(geom.area * 111.32 ** 2
                          * np.cos(np.deg2rad(geom.centroid.y)))}
    out = {'W': W, 'lat': lat, 'lon': lon}
    pickle.dump(out, open(F_W, 'wb'))
    print(f'유역 {len(W)}개')
    return out


def load_all():
    Wc = cell_weights()
    W, lat, lon = Wc['W'], Wc['lat'], Wc['lon']

    ds = xr.open_dataset(B.NC)
    t = pd.to_datetime(ds.time.values)
    A = {v: ds[v].values for v in X_BASE + ['BC_LR']}
    ds.close()
    d2 = xr.open_dataset(B.NC_BC12)
    A['BC'], A['BC_G'] = d2['BC_1'].values, d2['BC_2'].values
    d2.close()

    th = xr.open_dataset(F_THI)
    THI = pd.DataFrame(th['precipitation'].values,
                       index=pd.to_datetime(th.time.values),
                       columns=[str(c) for c in th.basin.values]).reindex(t)
    th.close()
    return W, lat, lon, t, A, THI


def dominant_basin(W, lat, lon):
    """격자마다 겹치는 면적이 가장 큰 유역.  (owner[i,j], 격자 소속 목록)"""
    best = np.zeros((len(lat), len(lon)))
    owner = np.full((len(lat), len(lon)), '', dtype=object)
    for code, d in W.items():
        for i, j, a in d['w']:
            if a > best[i, j]:
                best[i, j], owner[i, j] = a, code
    return owner


# ────────────────────────────────────────────────────────────── 격자 적합
def fit_grid(W, lat, lon, t, A, THI):
    owner = dominant_basin(W, lat, lon)
    itr = t.year == int(FIT_YEAR)
    P = np.full((len(t), len(lat), len(lon)), np.nan, np.float32)
    C = np.full((len(X_BASE) + 1, len(lat), len(lon)), np.nan, np.float32)
    nfit = np.zeros((len(lat), len(lon)), int)

    t0, done = time.time(), 0
    for i in range(len(lat)):
        for j in range(len(lon)):
            code = owner[i, j]
            if not code or code not in THI.columns:
                continue
            X = np.column_stack([A[v][:, i, j] for v in X_BASE])
            y = THI[code].to_numpy()
            ok = np.isfinite(X).all(1)
            m = ok & np.isfinite(y) & itr
            if m.sum() < MIN_FIT:
                continue
            Xm = np.column_stack([X[m], np.ones(m.sum())])
            c, *_ = np.linalg.lstsq(Xm, y[m], rcond=None)
            P[ok, i, j] = np.maximum(
                np.column_stack([X[ok], np.ones(ok.sum())]) @ c, 0.0)
            C[:, i, j] = c
            nfit[i, j] = int(m.sum())
            done += 1
    print(f'  적합한 격자 {done}개 / {int((owner != "").sum())}개 '
          f'({time.time() - t0:.0f}s)')
    return P, C, nfit, owner


def save_grid(P, C, nfit, lat, lon, t):
    ds = xr.Dataset(
        {'LR_THI': (('time', 'lat', 'lon'), P),
         'coef': (('term', 'lat', 'lon'), C),
         'n_fit': (('lat', 'lon'), nfit)},
        coords={'time': t, 'lat': lat, 'lon': lon,
                'term': X_BASE + ['intercept']},
        attrs={'title': 'Per-pixel linear regression, target = basin Thiessen',
               'predictors': ', '.join(X_BASE), 'target': 'Thiessen (basin)',
               'method': f'per-pixel OLS, fit {FIT_YEAR} -> applied all',
               'time_zone': 'KST (UTC+9)', 'units': 'mm/day'})
    ds.to_netcdf(F_OUT)
    print(f'  저장 {F_OUT}')


# ────────────────────────────────────────────────────────────── 유역 집계
def to_basin(W, arr, t) -> pd.DataFrame:
    """격자장 → 유역 면적가중 일강수.  유효면적이 절반 미만인 날은 결측."""
    out = {}
    for code, d in W.items():
        tot = sum(a for _, _, a in d['w'])
        num = np.zeros(len(t))
        den = np.zeros(len(t))
        for i, j, a in d['w']:
            x = arr[:, i, j]
            ok = np.isfinite(x)
            num[ok] += a * x[ok]
            den[ok] += a
        v = np.where(den >= MIN_AREA_FRAC * tot, num / np.where(den > 0, den, 1),
                     np.nan)
        out[code] = v
    return pd.DataFrame(out, index=t)


def kge(sim, obs):
    m = np.isfinite(sim) & np.isfinite(obs)
    if m.sum() < 60:
        return np.nan, np.nan, np.nan, np.nan
    s, o = sim[m], obs[m]
    if s.std() == 0 or o.std() == 0 or o.mean() == 0:
        return np.nan, np.nan, np.nan, np.nan
    r = float(np.corrcoef(s, o)[0, 1])
    a, b = float(s.std() / o.std()), float(s.mean() / o.mean())
    return (1 - float(np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2)),
            r, b, float(np.sqrt(((s - o) ** 2).mean())))


def evaluate(BAS: dict[str, pd.DataFrame], THI: pd.DataFrame, common=True):
    """유역마다 산출물별 지표·절대량."""
    codes = [c for c in THI.columns if c in BAS['LR_THI'].columns]
    rows = []
    for c in codes:
        o = THI[c].loc[EVAL0:EVAL1]
        cols = {k: BAS[k][c].loc[EVAL0:EVAL1] for k in PRODS}
        if common:
            m = o.notna()
            for v in cols.values():
                m &= v.notna()
            o = o[m]
            cols = {k: v[m] for k, v in cols.items()}
        if o.notna().sum() < 60:
            continue
        peak = o.idxmax() if o.notna().any() else None
        yp = {}
        for yr, gg in o.groupby(o.index.year):
            if gg.notna().sum() >= 30 and gg.max() > 0:
                yp[yr] = gg.idxmax()
        for k, v in cols.items():
            g, r, b, rm = kge(v.to_numpy(), o.to_numpy())
            rows.append({'code': c, '산출': k, 'n': int(o.notna().sum()),
                         'KGE': g, 'R': r, 'RMSE': rm,
                         '누적': float(v.sum()), '기준누적': float(o.sum()),
                         '누적비': float(v.sum() / o.sum()) if o.sum() else np.nan,
                         '최대일': float(v.max()), '기준최대일': float(o.max()),
                         '피크비': float(v.loc[peak] / o.loc[peak])
                         if peak is not None and o.loc[peak] > 0 else np.nan,
                         '연피크비': float(np.nanmedian(
                             [v.loc[d] / o.loc[d] for d in yp.values()]))
                         if yp else np.nan})
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────── 그림
def _poly(ax, W, val: dict, vmin, vmax, cmap, title, unit):
    recs = B.read_dbf(B.SHP + '.dbf')
    offs = B.shp_offsets(B.SHP)
    pats, cols = [], []
    for k, r in enumerate(recs):
        code = r['SBSN_CD'].strip()
        if code not in W:
            continue
        try:
            g = B.read_polygon(B.SHP, *offs[k])
        except Exception:
            continue
        for gg in (g.geoms if g.geom_type == 'MultiPolygon' else [g]):
            pats.append(MplPoly(np.asarray(gg.exterior.coords), closed=True))
            cols.append(val.get(code, np.nan))
    pc = PatchCollection(pats, cmap=cmap, edgecolor='#FFFFFF', linewidths=.15)
    pc.set_array(np.array(cols, float))
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    ax.set_xlim(125.5, 129.7)
    ax.set_ylim(33.0, 38.7)
    ax.set_aspect(1 / np.cos(np.deg2rad(36)))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=11)
    cb = plt.colorbar(pc, ax=ax, fraction=.045, pad=.02)
    cb.set_label(unit, fontsize=9)


def figures(tab: pd.DataFrame, W) -> None:
    def out(fig, stem):
        if SAVE:
            os.makedirs(SAVE, exist_ok=True)
            f = os.path.join(SAVE, stem + '.png')
            fig.savefig(f, bbox_inches='tight', facecolor='white')
            print('  저장', f)
        else:
            plt.show()

    piv = {m: tab.pivot(index='code', columns='산출', values=m)
           for m in ('KGE', '누적비', '피크비', 'RMSE')}

    # G1 ─ 유역 지도
    for m, (vmin, vmax, cmap, unit) in {
            'KGE': (0, 1, 'viridis', 'KGE'),
            '누적비': (.5, 1.5, 'RdBu_r', '산출/티센'),
            '피크비': (0, 2, 'RdBu_r', '연 최대일 산출/티센')}.items():
        fig, axs = plt.subplots(1, len(PRODS), figsize=(3.4 * len(PRODS), 6.4),
                                dpi=130)
        for ax, k in zip(axs, PRODS):
            _poly(ax, W, piv[m][k].to_dict(), vmin, vmax, cmap,
                  f'{LAB[k]}\n중앙값 {piv[m][k].median():.2f}', unit)
        fig.suptitle(f'전국 표준유역 {m}   티센 기준  {EVAL0} ~ {EVAL1}',
                     fontweight='bold')
        fig.tight_layout(rect=(0, 0, 1, .95))
        out(fig, f'G_지도_{m}')

    # G2 ─ 분포
    fig, axs = plt.subplots(1, 4, figsize=(20, 5.2), dpi=130)
    for ax, (m, lo, hi) in zip(axs, [('KGE', -.5, 1), ('R', 0, 1),
                                     ('누적비', .4, 1.8), ('피크비', 0, 2.2)]):
        p = tab.pivot(index='code', columns='산출', values=m)
        ax.boxplot([p[k].dropna() for k in PRODS], tick_labels=PRODS,
                   showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor='#EEF1F4'), medianprops=dict(color='k'))
        for i, k in enumerate(PRODS, 1):
            ax.scatter(np.random.normal(i, .06, len(p[k].dropna())),
                       p[k].dropna(), s=3, alpha=.15, color=COL[k])
        if m in ('누적비', '피크비'):
            ax.axhline(1, color='#C0392B', lw=1, ls='--')
        ax.set_ylim(lo, hi)
        ax.set_title(m)
        ax.grid(axis='y', alpha=.3)
        ax.tick_params(axis='x', labelrotation=30, labelsize=9)
    fig.suptitle(f'전국 {tab["code"].nunique()}개 표준유역 분포  티센 기준',
                 fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .94))
    out(fig, 'G_분포')


# ────────────────────────────────────────────────────────────── 본체
def main() -> None:
    W, lat, lon, t, A, THI = load_all()
    print(f'격자 {len(lat)}×{len(lon)} · {len(t)}일 · 유역 {len(W)}개')

    if '--no-fit' in sys.argv and os.path.exists(F_OUT):
        P = xr.open_dataset(F_OUT)['LR_THI'].values
        print(f'  읽음 {F_OUT}')
    else:
        P, C, nfit, owner = fit_grid(W, lat, lon, t, A, THI)
        save_grid(P, C, nfit, lat, lon, t)

    print('유역 집계 중...', flush=True)
    BAS = {'LR_THI': to_basin(W, P, t)}
    for k in ('BC_G', 'BC', 'BC_LR', 'TCA'):
        BAS[k] = to_basin(W, A[k], t)

    tab = evaluate(BAS, THI)
    print(f'\n■ 티센 기준 전국 평가  {EVAL0} ~ {EVAL1}   '
          f'유역 {tab["code"].nunique()}개  (다섯 산출물이 다 있는 날만)')
    g = tab.groupby('산출')[['n', 'KGE', 'R', 'RMSE', '누적비', '피크비',
                            '연피크비']]
    print('  ─ 중앙값 ─')
    print(g.median().reindex(PRODS).round(3).to_string())
    print('  ─ 평균 ─')
    print(g.mean().reindex(PRODS).round(3).to_string())
    print('  ─ 절대 강우량 합계 [mm] (전 유역 평균) ─')
    a = tab.groupby('산출')[['누적', '기준누적', '최대일', '기준최대일']].mean()
    print(a.reindex(PRODS).round(1).to_string())

    tab.to_csv(f'{WORK}/LR_THI_basin_eval.csv', index=False)
    print(f'\n  저장 {WORK}/LR_THI_basin_eval.csv')
    figures(tab, W)


if __name__ == '__main__':
    main()
