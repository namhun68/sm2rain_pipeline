#!/usr/bin/env python3
"""보고서 그림 — 티센 목표 재적합 검토.

report.py 가 본문 그림을 만든다면, 이 파일은 편향보정의 목표자료를 티센으로
바꾼 판(analysis/lr_thiessen.py)을 전국 규모로 견주는 그림을 만든다.
기준은 티센 유역 일강수이고 평가기간은 2022-01-01 ~ 2025-05-01 이다.

    R1   전국유역_KGE지도            산출물 5종 KGE
    R2   전국유역_누적비지도          산출/티센 누적비
    R3   전국유역_연최대일재현비지도    연 최대일 재현비
    R4   유역분포                   KGE·R·누적비·연최대일 상자그림
    R5   대상유역_피크전후            3개 유역 피크 ±5일
    R6   전국_월평균시계열            월평균 일강수
    R7   전국_일강수시계열            일강수
    R8   전국_누적강수               누적강수
    R9   전국_산점도                 일별 산점도
    R10  전국_월별평균               월별 평균 일강수
    R11  전국_강도별재현비            강우강도 구간별 산출/티센
    R12  전국_연누적                 연 누적강수

    수치표 report_lr_지표.csv · report_lr_피크.csv · report_lr_전국통계.csv
           report_lr_강도별재현비.csv

셀(#%%) 단위로 위에서부터 실행한다.  SAVE 에 폴더를 넣으면 저장까지 하고
글씨·해상도가 보고서 규격으로 올라간다.  None 이면 화면에만 띄운다.

앞서 analysis/lr_thiessen.py 를 돌려 격자장을 만들어 두어야 한다.
    KIHS/DATA/LR_THI_grid.nc

────────────────────────────────────────────────────────────────────────────
지켜야 할 규약
────────────────────────────────────────────────────────────────────────────
  · 기준은 티센 유역 일강수 (ASOS+AWS 752지점, KST 01시~익일 00시).
  · LR 은 목표와 기준이 같은 자료다.  유리한 판인데도 못 이긴다는 것이
    요지이므로, 그림에서 "더 낫다"로 읽히지 않게 캡션을 단다.
  · 다섯 산출물이 같은 날짜에서 비교되도록 표본을 맞춘다.
  · 유역 단위 평가와 전국 평균은 다른 것을 잰다.  유역마다 다른 날 오는
    국지 사상이 전국 평균에서는 희석되므로 섞어 쓰지 않는다.
"""
#%% 설정 ─────────────────────────────────────────────────────────────────
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import font_manager

_F = globals().get('__file__')          # 셀 실행이면 __file__ 이 없다
_HERE = os.path.dirname(os.path.abspath(_F)) if _F else os.getcwd()
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, os.pardir, 'analysis')))
import lr_thiessen as G      # 자료 읽기·집계·지도 도구를 공유한다
import basin_eval_core as B

TAB = '/Users/kim/Desktop/work/KIHS/DATA'

SAVE = None              # 저장할 폴더.  None 이면 띄우기만 한다
#   예: SAVE = '/Users/kim/Desktop/work/KIHS/fig_basin'

PRODS = ['BC_G', 'LR_THI', 'BC_LR', 'BC', 'TCA']
LAB = {'BC_G': 'BC-G (최종산출물)', 'LR_THI': 'LR (목표 티센)',
       'BC_LR': 'LR (목표 IDW_AWS)', 'BC': 'BC', 'TCA': 'TCA (보정 전)',
       'THI': '티센'}
SHORT = {'BC_G': 'BC-G', 'LR_THI': 'LR\n(목표 티센)', 'BC_LR': 'LR\n(목표 AWS)',
         'BC': 'BC', 'TCA': 'TCA'}
COL = {'BC_G': '#8E3B46', 'LR_THI': '#0F7B8A', 'BC_LR': '#6C8EBF',
       'BC': '#D1495B', 'TCA': '#EDAE49'}
BASINS = {'논산천상류': '301301', '조종천상류': '101503', '유등천상류': '300903'}
SPAN = ('2022-01-01', '2025-05-01')   # R7 전국 시계열 구간 (= 평가기간)

#  저장할 때는 보고서 규격으로, 띄울 때는 화면에 들어가게
FS = 20 if SAVE else 11
DPI = 200 if SAVE else 110
WMAP = 5.2 if SAVE else 2.9      # 지도 한 칸 폭
HMAP = 9.6 if SAVE else 5.4
WBOX, HBOX = (26, 8.4) if SAVE else (15, 4.8)     # R4
WPAN, HPAN = (27, 8.0) if SAVE else (16, 4.6)     # R5 · R6
WTS, HTS = (26, 5.2) if SAVE else (16, 3.6)       # R7  전체 폭, 유역 한 줄 높이

for _c in ('AppleGothic', 'NanumGothic', 'Malgun Gothic'):
    if any(_c in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = _c
        break
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': FS,
                     'axes.titlesize': FS + 2, 'axes.titleweight': 'bold',
                     'axes.labelsize': FS, 'xtick.labelsize': FS - 3,
                     'ytick.labelsize': FS - 3, 'legend.fontsize': FS - 4,
                     'figure.titlesize': FS + 5, 'figure.titleweight': 'bold'})


def keep(fig, stem: str) -> None:
    """SAVE 를 줬으면 저장까지 한다."""
    if SAVE:
        os.makedirs(SAVE, exist_ok=True)
        f = f'{SAVE}/{stem}.png'
        fig.savefig(f, bbox_inches='tight', facecolor='white')
        print('  저장', f)


#%% 자료 읽기 ─────────────────────────────────────────────────────────────
W, lat, lon, t, A, THI = G.load_all()
A['LR_THI'] = xr.open_dataset(G.F_OUT)['LR_THI'].values
AWS = A['AWS'] if 'AWS' in A else xr.open_dataset(B.NC)['AWS'].values
BAS = {k: G.to_basin(W, A[k], t) for k in PRODS}
print('유역', len(W), ' 격자', A['LR_THI'].shape, ' 기간',
      t.min().date(), '~', t.max().date())


#%% 유역별 지표 ──────────────────────────────────────────────────────────
rows = []
for c in THI.columns:
    if c not in BAS['BC_G'].columns:
        continue
    o = THI[c].loc[G.EVAL0:G.EVAL1]
    cols = {k: BAS[k][c].loc[G.EVAL0:G.EVAL1] for k in PRODS}
    m = o.notna()
    for v in cols.values():
        m &= v.notna()
    if m.sum() < 60:
        continue
    o, cols = o[m], {k: v[m] for k, v in cols.items()}
    yp = [g.idxmax() for _, g in o.groupby(o.index.year)
          if g.notna().sum() >= 30 and g.max() > 0]
    for k, v in cols.items():
        kg, r, _, rm = G.kge(v.to_numpy(), o.to_numpy())
        rows.append({'code': c, '산출': k, 'n': int(m.sum()), 'KGE': kg,
                     'R': r, 'RMSE': rm, '누적': float(v.sum()),
                     '기준누적': float(o.sum()),
                     '누적비': float(v.sum() / o.sum()),
                     '연최대일비': float(np.median(
                         [v.loc[d] / o.loc[d] for d in yp])) if yp else np.nan})
tab = pd.DataFrame(rows)

print(f'\n■ 전국 {tab["code"].nunique()}개 표준유역   티센 기준   '
      f'{G.EVAL0} ~ {G.EVAL1}\n  ─ 중앙값 ─')
print(tab.groupby('산출')[['n', 'KGE', 'R', 'RMSE', '누적비', '연최대일비']]
      .median().reindex(PRODS).round(3).to_string())
if SAVE:
    tab.to_csv(f'{TAB}/report_lr_지표.csv', index=False)
    print(f'  표 {TAB}/report_lr_지표.csv')


#%% 지도 도구 ────────────────────────────────────────────────────────────
def maps(metric, vmin, vmax, cmap, unit, title, stem):
    piv = tab.pivot(index='code', columns='산출', values=metric)
    fig, axs = plt.subplots(1, len(PRODS), figsize=(WMAP * len(PRODS), HMAP),
                            dpi=DPI)
    for ax, k in zip(axs, PRODS):
        G._poly(ax, W, piv[k].to_dict(), vmin, vmax, cmap,
                f'{LAB[k]}\n중앙값 {piv[k].median():.2f}', unit)
        ax.title.set_fontsize(FS)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, .95))
    keep(fig, stem)
    plt.show()


#%% R1  전국유역 KGE 지도 ────────────────────────────────────────────────
maps('KGE', 0, 1, 'viridis', 'KGE',
     f'전국 표준유역 KGE   티센 기준   {G.EVAL0} ~ {G.EVAL1}',
     'R1_전국유역_KGE지도')


#%% R2  전국유역 누적비 지도 ─────────────────────────────────────────────
maps('누적비', .6, 1.4, 'RdBu_r', '산출 / 티센',
     f'전국 표준유역 누적강수비   티센 기준   {G.EVAL0} ~ {G.EVAL1}',
     'R2_전국유역_누적비지도')


#%% R3  전국유역 연 최대일 재현비 지도 ───────────────────────────────────
maps('연최대일비', 0, 1.6, 'RdBu_r', '산출 / 티센',
     f'전국 표준유역 연 최대일 재현비   티센 기준   {G.EVAL0} ~ {G.EVAL1}',
     'R3_전국유역_연최대일재현비지도')


#%% R4  유역 분포 ────────────────────────────────────────────────────────
items = [('KGE', -.2, 1, None), ('R', .6, 1, None),
         ('누적비', .6, 1.6, 1.0), ('연최대일비', 0, 1.6, 1.0)]
fig, axs = plt.subplots(1, 4, figsize=(WBOX, HBOX), dpi=DPI)
for ax, (m, lo, hi, ref) in zip(axs, items):
    p = tab.pivot(index='code', columns='산출', values=m)
    ax.boxplot([p[k].dropna() for k in PRODS], showfliers=False, widths=.6,
               patch_artist=True, boxprops=dict(facecolor='#EEF1F4', lw=1.4),
               medianprops=dict(color='k', lw=2), whiskerprops=dict(lw=1.4),
               capprops=dict(lw=1.4))
    for i, k in enumerate(PRODS, 1):
        v = p[k].dropna()
        ax.scatter(np.random.normal(i, .07, len(v)), v, s=6, alpha=.18,
                   color=COL[k], zorder=0)
    if ref is not None:
        ax.axhline(ref, color='#C0392B', lw=1.6, ls='--')
    ax.set_xticks(range(1, len(PRODS) + 1))
    ax.set_xticklabels([SHORT[k] for k in PRODS], fontsize=FS - 5)
    ax.set_ylim(lo, hi)
    ax.set_title(m)
    ax.grid(axis='y', alpha=.3)
fig.suptitle(f'전국 {tab["code"].nunique()}개 표준유역 분포   티센 기준   '
             f'{G.EVAL0} ~ {G.EVAL1}')
fig.tight_layout(rect=(0, 0, 1, .93))
keep(fig, 'R4_유역분포')
plt.show()


#%% R5  대상 유역 피크 전후 ──────────────────────────────────────────────
WIN = 5
fig, axs = plt.subplots(1, 3, figsize=(WPAN, HPAN + .6), dpi=DPI)
recs = []
for ax, (name, code) in zip(axs, BASINS.items()):
    o = THI[code].loc[G.EVAL0:G.EVAL1]
    m = o.notna()
    for k in PRODS:
        m &= BAS[k][code].loc[G.EVAL0:G.EVAL1].notna()
    d0 = o[m].idxmax()
    sl = slice(d0 - pd.Timedelta(days=WIN), d0 + pd.Timedelta(days=WIN))
    w = pd.DataFrame({'THI': THI[code].loc[sl],
                      **{k: BAS[k][code].loc[sl] for k in PRODS}})
    x = np.arange(len(w))
    bw = .8 / (len(PRODS) + 1)
    ax.bar(x - len(PRODS) / 2 * bw, w['THI'], bw * .92, color='#444444',
           label=LAB['THI'])
    for i, k in enumerate(PRODS):
        ax.bar(x + (i - len(PRODS) / 2 + 1) * bw, w[k].fillna(0), bw * .92,
               color=COL[k], label=LAB[k])
    ip = list(w.index).index(d0)
    ax.axvspan(ip - .5, ip + .5, color='#C8A200', alpha=.13, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{d:%m-%d}' for d in w.index], rotation=60,
                       ha='right', fontsize=FS - 5)
    ax.set_title(f'{name}   {d0:%Y-%m-%d}')
    ax.set_ylabel('일강수 [mm]')
    ax.grid(axis='y', alpha=.3)
    if ax is axs[0]:
        ax.legend(fontsize=FS - 6)
    recs.append({'유역': name, '피크일': f'{d0:%Y-%m-%d}',
                 '티센_피크일': round(float(w.loc[d0, 'THI']), 1),
                 **{f'{k}_피크일': round(float(w.loc[d0, k]), 1) for k in PRODS},
                 '티센_사상': round(float(w['THI'].sum()), 1),
                 **{f'{k}_사상': round(float(w[k].sum()), 1) for k in PRODS}})
fig.suptitle(f'대상 유역 피크일 전후 ±{WIN}일')
fig.tight_layout(rect=(0, 0, 1, .93))
keep(fig, 'R5_대상유역_피크전후')
plt.show()

peak = pd.DataFrame(recs)
print(peak.to_string(index=False))
if SAVE:
    peak.to_csv(f'{TAB}/report_lr_피크.csv', index=False)
    print(f'  표 {TAB}/report_lr_피크.csv')


#%% 전국 면적가중 평균 ───────────────────────────────────────────────────
#   날마다 다섯 산출물과 티센이 모두 있는 유역만 쓰고 그 유역들의 면적으로
#   가중치를 다시 정규화한다.
import matplotlib.dates as mdates

codes = [c for c in THI.columns if c in BAS['BC_G'].columns]
area = np.array([W[c]['area'] for c in codes], float)
OK = np.isfinite(THI[codes].loc[SPAN[0]:SPAN[1]].to_numpy())
for k in PRODS:
    OK &= np.isfinite(BAS[k][codes].loc[SPAN[0]:SPAN[1]].to_numpy())
idx = THI[codes].loc[SPAN[0]:SPAN[1]].index
den = (OK * area).sum(1)
good = den > 0


def national(df):
    v = np.where(OK, np.nan_to_num(df[codes].loc[SPAN[0]:SPAN[1]].to_numpy()), 0)
    return pd.Series(np.where(good, (v * area).sum(1) / np.where(good, den, 1),
                              np.nan), index=idx)


NAT = {'THI': national(THI), **{k: national(BAS[k]) for k in PRODS}}
o = NAT['THI'].dropna()
print(f'전국 평균에 쓴 유역 수  중앙값 {pd.Series(OK.sum(1)).median():.0f} / '
      f'{len(codes)}   유효일 {len(o)} / {len(idx)}일')

rows = []
for k in PRODS:
    v = NAT[k].reindex(o.index)
    m = v.notna()
    kg, r, _, rm = G.kge(v[m].to_numpy(), o[m].to_numpy())
    rows.append({'산출': k, 'n': int(m.sum()), 'KGE': kg, 'R': r, 'RMSE': rm,
                 '편의': float(v[m].mean() - o[m].mean()),
                 '누적[mm]': float(v[m].sum()),
                 '누적비': float(v[m].sum() / o[m].sum()),
                 '최대일[mm]': float(v[m].max()),
                 '최대일비': float(v[m].max() / o[m].max())})
nstat = pd.DataFrame(rows).set_index('산출')
print(f'\n■ 전국 면적가중 일강수 통계   티센 기준   {SPAN[0]} ~ {SPAN[1]}   '
      f'{len(o)}일')
print(nstat.round(3).to_string())
if SAVE:
    nstat.to_csv(f'{TAB}/report_lr_전국통계.csv')
    print(f'  표 {TAB}/report_lr_전국통계.csv')


def xdate(ax, minticks=8, maxticks=16):
    loc = mdates.AutoDateLocator(minticks=minticks, maxticks=maxticks)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.tick_params(axis='x', labelrotation=30)
    for lb in ax.get_xticklabels():
        lb.set_horizontalalignment('right')


WSER, HSER = (18, 4.6) if SAVE else (14, 3.6)     # 시계열 한 장
WSQ = 7.6 if SAVE else 5.6                        # 정사각형 한 장
WBAR, HBAR = (13, 6.4) if SAVE else (9, 4.8)      # 막대 한 장


#%% R6  전국 월평균 일강수 시계열 ────────────────────────────────────────
mo = {k: v.resample('MS').mean() for k, v in NAT.items()}
fig, ax = plt.subplots(figsize=(WSER, HSER), dpi=DPI)
ax.fill_between(mo['THI'].index, 0, mo['THI'].values, color='#D7DCE1',
                zorder=0)
ax.plot(mo['THI'].index, mo['THI'].values, color='k', lw=2.2, label=LAB['THI'])
for k in PRODS:
    ax.plot(mo[k].index, mo[k].values, color=COL[k], lw=1.5, label=LAB[k])
ax.set_ylabel('월평균 일강수 [mm/일]')
ax.set_ylim(bottom=0)
ax.margins(x=.01)
ax.grid(alpha=.25)
ax.legend(ncols=3, fontsize=FS - 5)
ax.set_title(f'전국 면적가중 월평균 일강수   {SPAN[0]} ~ {SPAN[1]}')
xdate(ax)
fig.tight_layout()
keep(fig, 'R6_전국_월평균시계열')
plt.show()


#%% R7  전국 일강수 시계열 ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(WSER, HSER), dpi=DPI)
ax.fill_between(NAT['THI'].index, 0, NAT['THI'].values, color='#D7DCE1',
                zorder=0)
ax.plot(NAT['THI'].index, NAT['THI'].values, color='k', lw=1.4,
        label=LAB['THI'])
for k in PRODS:
    ax.plot(NAT[k].index, NAT[k].values, color=COL[k], lw=.9, alpha=.9,
            label=LAB[k])
ax.set_ylabel('일강수 [mm/일]')
ax.set_ylim(bottom=0)
ax.margins(x=.01)
ax.grid(alpha=.25)
ax.legend(ncols=3, fontsize=FS - 5)
ax.set_title(f'전국 면적가중 일강수   {SPAN[0]} ~ {SPAN[1]}')
xdate(ax)
fig.tight_layout()
keep(fig, 'R7_전국_일강수시계열')
plt.show()


#%% R8  전국 누적강수 ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(WSER, HSER), dpi=DPI)
ax.fill_between(o.index, 0, o.cumsum().values, color='#D7DCE1', zorder=0)
ax.plot(o.index, o.cumsum().values, color='k', lw=2.4, label=LAB['THI'])
for k in PRODS:
    v = NAT[k].reindex(o.index).cumsum()
    ax.plot(v.index, v.values, color=COL[k], lw=1.8, label=LAB[k])
    ax.annotate(f'{nstat.loc[k, "누적비"]:.2f}', (v.index[-1], v.iloc[-1]),
                xytext=(6, 0), textcoords='offset points', va='center',
                color=COL[k], fontsize=FS - 5, fontweight='bold')
ax.set_ylabel('누적강수 [mm]')
ax.margins(x=.02)
ax.grid(alpha=.25)
ax.legend(loc='upper left', fontsize=FS - 5)
ax.set_title(f'전국 면적가중 누적강수   {SPAN[0]} ~ {SPAN[1]}')
xdate(ax)
fig.tight_layout()
keep(fig, 'R8_전국_누적강수')
plt.show()


#%% R9  전국 일강수 산점도 ─────────────────────────────────────────────
hi = float(max(o.max(), max(NAT[k].max() for k in PRODS))) * 1.06
fig, ax = plt.subplots(figsize=(WSQ, WSQ), dpi=DPI)
for k in PRODS:
    ax.scatter(o, NAT[k].reindex(o.index), s=16, alpha=.45, color=COL[k],
               edgecolors='none', label=LAB[k])
ax.plot([0, hi], [0, hi], 'k--', lw=1.4)
ax.set_xlim(0, hi)
ax.set_ylim(0, hi)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('티센 [mm/일]')
ax.set_ylabel('산출 [mm/일]')
ax.grid(alpha=.25)
ax.legend(fontsize=FS - 5)
ax.set_title(f'전국 면적가중 일강수 산점도   {SPAN[0][:4]}–{SPAN[1][:4]}')
fig.tight_layout()
keep(fig, 'R9_전국_산점도')
plt.show()


#%% R10  전국 월별 평균 ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(WBAR, HBAR), dpi=DPI)
x = np.arange(1, 13)
ax.fill_between(x, 0, o.groupby(o.index.month).mean().values, color='#D7DCE1',
                zorder=0)
ax.plot(x, o.groupby(o.index.month).mean(), 'k-o', lw=2.4, ms=6,
        label=LAB['THI'])
for k in PRODS:
    v = NAT[k].reindex(o.index)
    ax.plot(x, v.groupby(v.index.month).mean(), '-o', lw=1.8, ms=5,
            color=COL[k], label=LAB[k])
ax.set_xticks(x)
ax.set_xlabel('월')
ax.set_ylabel('평균 일강수 [mm/일]')
ax.set_ylim(bottom=0)
ax.grid(alpha=.25)
ax.legend(fontsize=FS - 5)
ax.set_title(f'전국 월별 평균 일강수   {SPAN[0][:4]}–{SPAN[1][:4]}')
fig.tight_layout()
keep(fig, 'R10_전국_월별평균')
plt.show()


#%% R11  강우강도 구간별 재현비 ─────────────────────────────────────────
edges = [-.01, .5, 2, 5, 10, 20, 1000]
names = ['~0.5', '0.5~2', '2~5', '5~10', '10~20', '20~']
band = pd.cut(o, edges, labels=names)
HI = 2.4                 # 약비 구간은 값이 커서 잘라 그리고 숫자를 적는다
fig, ax = plt.subplots(figsize=(WBAR, HBAR), dpi=DPI)
xx = np.arange(len(names))
w = .8 / len(PRODS)
band_tab = {}
for i, k in enumerate(PRODS):
    v = NAT[k].reindex(o.index)
    ratio = np.array([v[band == b].sum() / o[band == b].sum() for b in names])
    band_tab[k] = ratio
    px = xx + (i - (len(PRODS) - 1) / 2) * w
    ax.bar(px, np.minimum(ratio, HI), w * .9, color=COL[k], label=LAB[k])
    for xi, rr in zip(px, ratio):
        if rr > HI:
            ax.annotate(f'{rr:.0f}', (xi, HI), xytext=(0, 3),
                        textcoords='offset points', ha='center',
                        fontsize=FS - 6, color=COL[k], fontweight='bold')
ax.axhline(1, color='#C0392B', lw=1.6, ls='--')
ax.set_ylim(0, HI * 1.12)
ax.set_xticks(xx)
ax.set_xticklabels([f'{b}\n{int((band == b).sum())}일' for b in names],
                   fontsize=FS - 5)
ax.set_xlabel('티센 일강수 [mm/일]')
ax.set_ylabel('산출 / 티센')
ax.grid(axis='y', alpha=.25)
ax.legend(fontsize=FS - 5)
ax.set_title(f'강우강도 구간별 재현비   {SPAN[0][:4]}–{SPAN[1][:4]}')
fig.tight_layout()
keep(fig, 'R11_전국_강도별재현비')
plt.show()

bt = pd.DataFrame(band_tab, index=names).T
bt.columns = [f'{c} ({int((band == c).sum())}일)' for c in names]
print('\n■ 강우강도 구간별 재현비 (산출/티센)')
print(bt.round(2).to_string())
if SAVE:
    bt.to_csv(f'{TAB}/report_lr_강도별재현비.csv')
    print(f'  표 {TAB}/report_lr_강도별재현비.csv')


#%% R12  연도별 누적 ────────────────────────────────────────────────────
yrs = sorted(o.index.year.unique())
fig, ax = plt.subplots(figsize=(WBAR, HBAR), dpi=DPI)
xx = np.arange(len(yrs))
w = .8 / (len(PRODS) + 1)
b0 = ax.bar(xx - len(PRODS) / 2 * w, o.groupby(o.index.year).sum(), w * .9,
            color='#444444', label=LAB['THI'])
ax.bar_label(b0, fmt='%.0f', fontsize=FS - 7, padding=2)
for i, k in enumerate(PRODS):
    v = NAT[k].reindex(o.index)
    ax.bar(xx + (i - len(PRODS) / 2 + 1) * w, v.groupby(v.index.year).sum(),
           w * .9, color=COL[k], label=LAB[k])
ax.set_xticks(xx)
ax.set_xticklabels([f'{y}년' for y in yrs])
ax.set_ylabel('연 누적강수 [mm]')
ax.grid(axis='y', alpha=.25)
ax.legend(fontsize=FS - 5)
ax.set_title(f'전국 연 누적강수   (2025년은 5월 1일까지)')
fig.tight_layout()
keep(fig, 'R12_전국_연누적')
plt.show()
