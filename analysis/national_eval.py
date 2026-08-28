"""
national_eval.py — 전국 표준유역(848개) 단위 BC-G 확대 검증
================================================================================
    import national_eval as N
    S = N.stats()          # 모든 표(dict) — 캐시됨
    N.figures()            # 그림 4장 저장 (fig_basin/)

3.4.2 는 미계측 유역 3곳을 깊게 본 것이고, 이 모듈은 같은 산출물을
전국 표준유역으로 넓혀 통계적으로 본다.

자료   최종산출물/02_표준유역/BCG_basin_daily.csv        (원자료, 보간 전)
       최종산출물/03_참고자료/{ASOS,AWS,BC}_basin_daily.csv
기준   IDW_ASOS.  BC·BC-G 는 IDW_AWS 로 학습하므로 ASOS 가 독립 기준이다.
기간   2022-01-01 ~ 2025-05-01.  2021년은 편향보정 학습구간이라 제외한다.
비교   모든 산출물을 BC-G 유효일에 한정해 같은 날로 비교한다.
       BC-G 결측이 무강수일에 치우쳐 있으므로 이 조건을 걸지 않으면
       산출물마다 다른 날을 채점하게 된다.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

ROOT = '/Users/kim/Desktop/work/KIHS/DATA/최종산출물'
OUT_DIR = '/Users/kim/Desktop/work/KIHS/fig_basin'
CACHE = '/Users/kim/Desktop/work/KIHS/DATA/national_eval_cache.pkl'
SHP = '/Users/kim/Desktop/work/KIHS/DATA/std_basin_850/std_basin_850'

START = '2022-01-01'
# 대권역코드는 한강권 10~13, 낙동강권 20~25 처럼 십의 자리가 권역을 뜻한다.
DAEGWON = {1: '한강', 2: '낙동강', 3: '금강', 4: '섬진강',
           5: '영산강', 6: '제주도'}
PRODS = ['BC_G', 'BC', 'AWS']
LABEL = {'BC_G': 'BC-G', 'BC': 'BC', 'AWS': 'IDW_AWS'}
COLOR = {'BC_G': '#C0392B', 'BC': '#E08A2E', 'AWS': '#2E86C1'}

BINS = [(0, 1, '무강우\n(<1)'), (1, 5, '약한 비\n(1~5)'), (5, 10, '보통\n(5~10)'),
        (10, 20, '다소 강함\n(10~20)'), (20, 50, '강함\n(20~50)'),
        (50, 80, '호우\n(50~80)'), (80, 1e9, '집중호우\n(≥80)')]
SEASON = {'봄(3~5월)': [3, 4, 5], '여름(6~8월)': [6, 7, 8],
          '가을(9~11월)': [9, 10, 11], '겨울(12~2월)': [12, 1, 2]}


# ------------------------------------------------------------------ 자료
def _read(p):
    return pd.read_csv(os.path.join(ROOT, p), index_col=0, parse_dates=True)


def load():
    bcg = _read('02_표준유역/BCG_basin_daily.csv')
    ref = _read('03_참고자료/ASOS_basin_daily.csv')
    cols = [c for c in bcg.columns if c in ref.columns]
    D = {'BC_G': bcg[cols], 'ASOS': ref[cols],
         'BC': _read('03_참고자료/BC_basin_daily.csv')[cols],
         'AWS': _read('03_참고자료/AWS_basin_daily.csv')[cols]}
    D = {k: v.loc[START:] for k, v in D.items()}
    info = pd.read_csv(os.path.join(ROOT, '02_표준유역/basin_info.csv'),
                       dtype={'표준유역코드': str})
    return D, info, cols


# ------------------------------------------------------------------ 지표
def kge(sim, obs):
    """(KGE, R, alpha, beta).  결측은 호출 전에 걸러 넣는다."""
    r = float(np.corrcoef(sim, obs)[0, 1])
    a = float(sim.std() / obs.std())
    b = float(sim.mean() / obs.mean())
    return 1 - float(np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2)), r, a, b


def pooled(sim, obs):
    k = ~(np.isnan(sim) | np.isnan(obs))
    s, o = sim[k], obs[k]
    g, r, a, b = kge(s, o)
    return dict(n=int(k.sum()), KGE=g, R=r, alpha=a, beta=b,
                RMSE=float(np.sqrt(((s - o) ** 2).mean())),
                bias=float((s - o).mean()),
                rbias=float((s.mean() - o.mean()) / o.mean() * 100))


def contingency(sim, obs, th):
    H = int(((sim >= th) & (obs >= th)).sum())
    M = int(((sim < th) & (obs >= th)).sum())
    F = int(((sim >= th) & (obs < th)).sum())
    return dict(H=H, M=M, F=F, POD=H / (H + M) if H + M else np.nan,
                FAR=F / (H + F) if H + F else np.nan,
                CSI=H / (H + M + F) if H + M + F else np.nan)


# ------------------------------------------------------------------ 집계
def stats(rebuild: bool = False) -> dict:
    if not rebuild and os.path.exists(CACHE):
        with open(CACHE, 'rb') as f:
            return pickle.load(f)

    D, info, cols = load()
    ok = D['BC_G'].notna() & D['ASOS'].notna()
    O = D['ASOS'].values[ok.values]

    S = {'period': (str(D['ASOS'].index.min().date()),
                    str(D['ASOS'].index.max().date())),
         'n_basin': len(cols), 'n_sample': int(ok.values.sum())}

    # (1) 전체 pooled
    S['pooled'] = {p: pooled(D[p].values[ok.values], O) for p in PRODS}

    # (2) 유역별
    rows = []
    for c in cols:
        o = D['ASOS'][c].where(ok[c])
        m = o.notna()
        if m.sum() < 100:
            continue
        row = {'code': c, 'n': int(m.sum())}
        for p in PRODS:
            s = D[p][c].where(ok[c])
            k = m & s.notna()
            if k.sum() < 100:
                row[p] = np.nan
                continue
            g, r, _, _ = kge(s[k].values, o[k].values)
            row[p] = g
            row[p + '_R'] = r
            row[p + '_RMSE'] = float(np.sqrt(((s[k] - o[k]) ** 2).mean()))
        rows.append(row)
    per = pd.DataFrame(rows).merge(
        info.rename(columns={'표준유역코드': 'code'}), on='code')
    per['권역'] = (per['대권역코드'] // 10).map(DAEGWON)
    S['per_basin'] = per

    # (3) 권역별 · 면적별 · 격자수별 중앙값
    def med(key, order=None):
        t = per.groupby(key, observed=True)[PRODS].median()
        t.insert(0, '유역수', per.groupby(key, observed=True).size())
        return t.reindex(order) if order else t

    S['by_region'] = med('권역', ['한강', '낙동강', '금강', '섬진강',
                                  '영산강', '제주도'])
    per['면적구간'] = pd.cut(per['면적_km2'], [0, 50, 100, 150, 200, 1e4],
                             labels=['50 미만', '50~100', '100~150',
                                     '150~200', '200 이상'])
    S['by_area'] = med('면적구간')
    per['격자군'] = pd.cut(per['중첩격자수'], [0, 2, 4, 6, 100],
                           labels=['1~2', '3~4', '5~6', '7 이상'])
    S['by_ncell'] = med('격자군')

    # (4) 계절 · 연도
    idx = D['ASOS'].index
    S['by_season'] = {}
    for nm, mo in SEASON.items():
        sel = ok.values & np.asarray(idx.month.isin(mo))[:, None]
        S['by_season'][nm] = {p: pooled(D[p].values[sel],
                                        D['ASOS'].values[sel]) for p in PRODS}
    S['by_year'] = {}
    for y in sorted(set(idx.year)):
        sel = ok.values & np.asarray(idx.year == y)[:, None]
        S['by_year'][y] = {p: pooled(D[p].values[sel],
                                     D['ASOS'].values[sel]) for p in PRODS}

    # (5) 강우강도 구간
    V = {p: D[p].values[ok.values] for p in PRODS}
    rows = []
    for lo, hi, nm in BINS:
        k = (O >= lo) & (O < hi)
        if k.sum() < 5:
            continue
        o = float(O[k].mean())
        r = {'구간': nm.replace('\n', ' '), '일수': int(k.sum()), '기준': o}
        for p in PRODS:
            r[p] = float(V[p][k].mean())
            r[p + '_rb'] = (r[p] - o) / o * 100 if o > 0 else np.nan
        rows.append(r)
    S['by_intensity'] = pd.DataFrame(rows)

    # (6) 호우 탐지
    S['detect'] = {th: {p: contingency(V[p], O, th) for p in PRODS}
                   for th in (30, 50, 80)}

    with open(CACHE, 'wb') as f:
        pickle.dump(S, f)
    return S


# ------------------------------------------------------------------ 그림
def _style():
    import matplotlib as mpl
    for f in ('Apple SD Gothic Neo', 'AppleGothic', 'NanumGothic'):
        if f in {x.name for x in mpl.font_manager.fontManager.ttflist}:
            mpl.rcParams['font.family'] = f
            break
    mpl.rcParams.update({
        'axes.unicode_minus': False, 'figure.dpi': 110, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'savefig.facecolor': 'white',
        'font.size': 18, 'axes.titlesize': 21, 'axes.titleweight': 'bold',
        'axes.labelsize': 19, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
        'legend.fontsize': 16, 'axes.linewidth': 1.4,
        'xtick.major.width': 1.6, 'ytick.major.width': 1.6,
        'axes.grid': True, 'grid.alpha': 0.25, 'lines.linewidth': 2.6,
        'legend.frameon': False,
    })


def figures(only=None):
    import matplotlib.pyplot as plt
    _style()
    S = stats()
    os.makedirs(OUT_DIR, exist_ok=True)
    made = []
    for n, fn in ((1, _fig_map), (2, _fig_cdf), (3, _fig_season_intensity),
                  (4, _fig_detect), (5, _fig_monthly),
                  (6, _fig_series), (7, _fig_region),
                  (8, _fig_scatter)):
        if only and n not in only:
            continue
        p = fn(S, plt)
        made.append(p)
        print('  저장', os.path.basename(p))
    return made


def _fig_map(S, plt):
    """전국 표준유역 KGE 공간분포 (3패널)."""
    import basin_eval_core as B
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon as MplPoly

    recs = B.read_dbf(SHP + '.dbf')
    offs = B.shp_offsets(SHP)
    key = next(k for k in recs[0] if 'BAS' in k.upper() or 'CD' in k.upper())
    per = S['per_basin'].set_index('code')

    fig, axs = plt.subplots(1, 3, figsize=(17, 10.5))
    for ax, p in zip(axs, PRODS):
        pats, vals = [], []
        for rec, (o, l) in zip(recs, offs):
            c = rec[key]
            if c not in per.index or not np.isfinite(per.loc[c, p]):
                continue
            try:
                g = B.read_polygon(SHP, o, l)
            except Exception:
                continue
            gs = [g] if g.geom_type == 'Polygon' else list(g.geoms)
            for gg in gs:
                pats.append(MplPoly(np.asarray(gg.exterior.coords), closed=True))
                vals.append(per.loc[c, p])
        pc = PatchCollection(pats, cmap='RdYlBu', edgecolor='#FFFFFF',
                             linewidth=0.25)
        pc.set_array(np.asarray(vals))
        pc.set_clim(0.4, 1.0)
        ax.add_collection(pc)
        ax.autoscale_view()
        ax.set_aspect(1 / np.cos(np.deg2rad(36)))
        ax.set_title(f'{LABEL[p]}  (중앙값 {per[p].median():.3f})')
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(pc, ax=axs, fraction=0.022, pad=0.02)
    cb.set_label('KGE', fontsize=19)
    p = os.path.join(OUT_DIR, 'N1_전국유역_KGE지도.png')
    fig.savefig(p); plt.close(fig)
    return p


def _fig_cdf(S, plt):
    """유역별 KGE 누적분포 + 권역별 상자그림."""
    per = S['per_basin']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 7.6))
    for p in PRODS:
        v = np.sort(per[p].dropna().values)
        a1.plot(v, np.arange(1, len(v) + 1) / len(v) * 100,
                color=COLOR[p], label=f'{LABEL[p]}  (중앙값 {np.median(v):.3f})')
    a1.axvline(0.7, ls=':', color='#555', lw=2)
    a1.text(0.7, 4, ' KGE 0.7', color='#555', fontsize=15)
    a1.set_xlim(0.2, 1.0); a1.set_ylim(0, 100)
    a1.set_xlabel('KGE'); a1.set_ylabel('누적 유역 비율 (%)')
    a1.set_title('(a) 유역별 KGE 누적분포'); a1.legend(loc='upper left')

    order = ['한강', '낙동강', '금강', '섬진강', '영산강', '제주도']
    w, off = 0.26, {-1: -0.27, 0: 0.0, 1: 0.27}
    for i, p in enumerate(PRODS):
        data = [per.loc[per['권역'] == r, p].dropna().values for r in order]
        bp = a2.boxplot(data, positions=np.arange(len(order)) + off[i - 1],
                        widths=w, patch_artist=True, showfliers=False,
                        medianprops=dict(color='#222', lw=2.2))
        for b in bp['boxes']:
            b.set_facecolor(COLOR[p]); b.set_alpha(0.75); b.set_edgecolor('#333')
        a2.plot([], [], color=COLOR[p], lw=8, label=LABEL[p])
    a2.set_xticks(range(len(order))); a2.set_xticklabels(order)
    a2.set_ylabel('KGE'); a2.set_title('(b) 대권역별 KGE 분포')
    a2.legend(loc='lower left', ncol=3)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, 'N2_전국유역_KGE분포.png')
    fig.savefig(p); plt.close(fig)
    return p


def _fig_season_intensity(S, plt):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16.5, 7.6))
    names = list(SEASON)
    x = np.arange(len(names))
    for i, p in enumerate(PRODS):
        v = [S['by_season'][n][p]['KGE'] for n in names]
        b = a1.bar(x + (i - 1) * 0.27, v, 0.26, color=COLOR[p], label=LABEL[p])
        a1.bar_label(b, fmt='%.3f', fontsize=12, padding=2)
    a1.set_xticks(x); a1.set_xticklabels(names)
    a1.set_ylim(0.55, 1.0); a1.set_ylabel('KGE')
    a1.set_title('(a) 계절별 KGE'); a1.legend(ncol=3, loc='lower left')

    T = S['by_intensity']
    x = np.arange(len(T))
    for i, p in enumerate(PRODS):
        a2.bar(x + (i - 1) * 0.27, T[p + '_rb'], 0.26, color=COLOR[p],
               label=LABEL[p])
    a2.axhline(0, color='#333', lw=2)
    a2.set_xticks(x)
    a2.set_xticklabels([b[2] for b in BINS][:len(T)], fontsize=13)
    a2.set_ylabel('상대편의 (%)')
    a2.set_title('(b) 강우강도 구간별 상대편의')
    a2.legend(ncol=3)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, 'N3_전국_계절_강도별.png')
    fig.savefig(p); plt.close(fig)
    return p


def _fig_detect(S, plt):
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 6.8))
    ths = [30, 50, 80]
    for ax, m in zip(axs, ('POD', 'FAR', 'CSI')):
        x = np.arange(len(ths))
        for i, p in enumerate(PRODS):
            v = [S['detect'][t][p][m] for t in ths]
            b = ax.bar(x + (i - 1) * 0.27, v, 0.26, color=COLOR[p],
                       label=LABEL[p])
            ax.bar_label(b, fmt='%.2f', fontsize=12, padding=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{t} mm/일' for t in ths])
        ax.set_ylim(0, 1.0)
        ax.set_title(f'({"abc"[ths.index(30) + list("abc").index("a")]}) {m}'
                     if False else
                     f'({chr(97 + ("POD", "FAR", "CSI").index(m))}) {m}')
    axs[0].set_ylabel('값'); axs[0].legend(ncol=1, loc='lower left')
    fig.tight_layout()
    p = os.path.join(OUT_DIR, 'N4_전국_호우탐지.png')
    fig.savefig(p); plt.close(fig)
    return p


def _fig_monthly(S, plt):
    """월별 자료 가용성과 재현 성능."""
    D, _, _ = load()
    ok = D['BC_G'].notna() & D['ASOS'].notna()
    mon, avail, kg, obs = [], [], [], []
    for m in range(1, 13):
        sel = D['ASOS'].index.month == m
        tot = int(D['ASOS'][sel].notna().values.sum())
        val = int(ok[sel].values.sum())
        s_ = D['BC_G'].values[sel][ok[sel].values]
        o_ = D['ASOS'].values[sel][ok[sel].values]
        g, _, _, _ = kge(s_, o_)
        mon.append(m); avail.append(val / tot * 100); kg.append(g)
        obs.append(float(o_.mean()))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6.8))
    b = a1.bar(mon, avail, 0.62, color='#5B8FB9')
    a1.bar_label(b, fmt='%.0f', fontsize=12, padding=2)
    ax = a1.twinx(); ax.grid(False)
    ax.plot(mon, obs, 'o-', color='#C0392B', ms=9)
    ax.set_ylabel('IDW_ASOS 평균 일강수 (mm/일)', color='#C0392B')
    ax.tick_params(axis='y', colors='#C0392B')
    a1.set_xticks(mon); a1.set_ylim(0, 62)
    a1.set_xlabel('월'); a1.set_ylabel('BC-G 유효율 (%)')
    a1.set_title('(a) 월별 자료 가용성과 강수량')

    b = a2.bar(mon, kg, 0.62, color='#C0392B')
    a2.bar_label(b, fmt='%.3f', fontsize=11, padding=2)
    a2.set_xticks(mon); a2.set_ylim(0.6, 1.02)
    a2.set_xlabel('월'); a2.set_ylabel('KGE')
    a2.set_title('(b) 월별 재현 성능 (BC-G)')
    fig.tight_layout()
    p = os.path.join(OUT_DIR, 'N5_전국_월별가용성.png')
    fig.savefig(p); plt.close(fig)
    return p


def _fig_series(S, plt):
    """전국 유역평균 일강수 시계열과 연강수 총량."""
    D, _, _ = load()
    ok = (D['BC_G'].notna() & D['ASOS'].notna())
    o = D['ASOS'].where(ok).mean(1)
    g = D['BC_G'].where(ok).mean(1)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(16, 10),
                                 gridspec_kw=dict(height_ratios=[2, 1]))
    a1.fill_between(o.index, 0, o.values, color='#B9C6D2', label='IDW_ASOS')
    a1.plot(g.index, g.values, color='#C0392B', lw=1.8, label='BC-G')
    a1.set_ylabel('전국 유역평균 일강수 (mm/일)')
    a1.set_title('(a) 전국 표준유역 평균 일강수 시계열 (BC-G 유효일)')
    a1.legend(ncol=2, loc='upper left')

    f = pd.read_csv(os.path.join(ROOT, '02_표준유역/BCG_filled_basin_daily.csv'),
                    index_col=0, parse_dates=True)
    a = pd.read_csv(os.path.join(ROOT, '03_참고자료/ASOS_basin_daily.csv'),
                    index_col=0, parse_dates=True)
    yrs = [2022, 2023, 2024]
    x = np.arange(len(yrs))
    fv = [f[f.index.year == y].sum().mean() for y in yrs]
    av = [a[a.index.year == y].sum().mean() for y in yrs]
    for i, (v, c, lb) in enumerate(((av, '#2E86C1', 'IDW_ASOS'),
                                    (fv, '#C0392B', 'BC-G (보간본)'))):
        bb = a2.bar(x + (i - 0.5) * 0.34, v, 0.32, color=c, label=lb)
        a2.bar_label(bb, fmt='%.0f', fontsize=13, padding=2)
    a2.set_xticks(x); a2.set_xticklabels([f'{y}년' for y in yrs])
    a2.set_ylabel('연강수량 (mm)'); a2.set_ylim(0, 1900)
    a2.set_title('(b) 전국 표준유역 평균 연강수 총량')
    a2.legend(ncol=2)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, 'N6_전국_시계열_연총량.png')
    fig.savefig(p); plt.close(fig)
    return p


def _fig_region(S, plt):
    """대권역별 종합 성능 요약."""
    D, _, _ = load()
    per = S['per_basin']
    ok = D['BC_G'].notna() & D['ASOS'].notna()
    order = ['한강', '낙동강', '금강', '섬진강', '영산강', '제주도']
    kg, rb, csi = [], [], []
    for r in order:
        cs = list(per.loc[per['권역'] == r, 'code'])
        m = ok[cs].values
        o = D['ASOS'][cs].values[m]
        g = D['BC_G'][cs].values[m]
        k, _, _, _ = kge(g, o)
        kg.append(k)
        rb.append((g.mean() - o.mean()) / o.mean() * 100)
        c = contingency(g, o, 50)
        csi.append(c['CSI'])
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 7.0))
    for ax, v, ttl, fmt in ((axs[0], kg, '(a) KGE', '%.3f'),
                            (axs[1], rb, '(b) 상대편의 (%)', '%.1f'),
                            (axs[2], csi, '(c) 호우 탐지 CSI (50 mm/일)', '%.3f')):
        b = ax.bar(order, v, 0.6,
                   color=['#C0392B' if x == '제주도' else '#2E86C1'
                          for x in order])
        ax.bar_label(b, fmt=fmt, fontsize=16, padding=3)
        ax.set_title(ttl)
        ax.tick_params(axis='x', labelrotation=20)
    axs[1].axhline(0, color='#333', lw=2)
    axs[0].set_ylim(0, 1.05); axs[2].set_ylim(0, 0.85)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, 'N7_전국_권역별요약.png')
    fig.savefig(p); plt.close(fig)
    return p


def _fig_scatter(S, plt):
    """전국 유역-일 표본의 기준 대비 산점도와 잔차 분포."""
    D, _, _ = load()
    ok = (D['BC_G'].notna() & D['ASOS'].notna()).values
    o = D['ASOS'].values[ok]
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 6.4))
    for ax, p_ in zip(axs, PRODS):
        s = D[p_].values[ok]
        ax.hexbin(o, s, gridsize=70, bins='log', cmap='viridis',
                  extent=(0, 200, 0, 200), mincnt=1)
        ax.plot([0, 200], [0, 200], '--', color='#C0392B', lw=2.4)
        st = S['pooled'][p_]
        ax.set_xlim(0, 200); ax.set_ylim(0, 200)
        ax.set_xlabel('IDW_ASOS (mm/일)')
        ax.set_title(f"({chr(97 + PRODS.index(p_))}) {LABEL[p_]}\n"
                     f"R={st['R']:.3f}  RMSE={st['RMSE']:.2f}  "
                     f"편의={st['bias']:+.2f} mm/일", fontsize=17)
        ax.grid(alpha=0.2)
    axs[0].set_ylabel('산출물 (mm/일)')
    fig.tight_layout()
    p = os.path.join(OUT_DIR, 'N8_전국_산점도.png')
    fig.savefig(p); plt.close(fig)
    return p


if __name__ == '__main__':
    S = stats(rebuild=True)
    print('표본', S['n_sample'], '유역', S['n_basin'])
    for p in PRODS:
        print(f"  {LABEL[p]:8s}", {k: round(v, 3) if isinstance(v, float) else v
                                   for k, v in S['pooled'][p].items()})
    figures()
