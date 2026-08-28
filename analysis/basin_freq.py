"""
basin_freq.py — 미계측 유역 확률강수량(빈도별) 비교
================================================================================
    python3 basin_freq.py

논산천상류 · 조종천상류 · 유등천상류에서 2·5·10·20·50·100년 빈도
확률강수량을 ASOS 기준과 각 산출물로 각각 산정하고 그 비를 본다.

★ 자료기간이 4년이다. 이 사실이 해석 전체를 지배한다.
   100년 빈도는 기록의 25배를 외삽하는 것이라 절대값은 신뢰할 수 없다.
   그래서 이 분석의 결론은 "확률강수량이 얼마인가"가 아니라
   **"같은 방법·같은 기간에 자료원만 바꾸면 설계강우가 몇 % 달라지는가"** 이다.
   양쪽이 같은 기간·같은 방법을 쓰므로 기간 부족의 영향이 상당 부분 상쇄된다.

방법
  · POT(Peaks-Over-Threshold) + GPD.  연최대치(AMS)는 표본이 4개뿐이라 쓸 수 없다.
  · 독립사상 분리: 3일 이상 간격 (declustering).
  · 임계값은 자료원마다 **같은 초과율**이 되도록 잡는다.
    같은 값으로 자르면 편의가 큰 자료는 표본수부터 달라져 비교가 흐려진다.
  · 결측 보정: BC_LR 은 결측일이 3분의 1이라, 유효기간을 실제 유효일수로 센다.
    ASOS 와 공통으로 유효한 날만 쓴다.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if '__file__' in globals() else '/Users/kim/Desktop/work/code/use')
import basin_eval_core as B

RETURN_PERIODS = [2, 5, 10, 20, 50, 100]
FREQ_START, FREQ_END = '2021-01-01', '2024-12-31'   # 완전한 4개 연도만
PEAKS_PER_YEAR = 3.0        # 목표 초과율 λ [건/년]
MIN_SEP_DAYS = 3            # 독립사상 분리 간격


def decluster(s: pd.Series, min_sep: int = MIN_SEP_DAYS) -> pd.Series:
    """큰 값부터 골라가며 min_sep 일 안에 이미 뽑힌 사상이 있으면 버린다."""
    picked = []
    for t, v in s.sort_values(ascending=False).items():
        if all(abs((t - p).days) >= min_sep for p in picked):
            picked.append(t)
    return s.loc[sorted(picked)]


def pot_return_levels(x: pd.Series, n_years: float,
                      periods=RETURN_PERIODS, lam=PEAKS_PER_YEAR):
    """POT+GPD 확률강수량.  x = 일강수 시계열(결측 제거된 것)."""
    k = max(8, int(round(lam * n_years)))
    peaks_all = decluster(x[x > 0])
    if len(peaks_all) < k:
        k = len(peaks_all)
    if k < 8:
        return None
    peaks = peaks_all.sort_values(ascending=False).iloc[:k]
    u = float(peaks.min())
    exc = peaks.values - u
    exc = exc[exc > 0]
    if len(exc) < 6:
        return None
    xi, loc, sigma = stats.genpareto.fit(exc, floc=0.0)
    rate = len(peaks) / n_years            # 연 초과 건수 λ
    out = {}
    for T in periods:
        m = rate * T
        if abs(xi) < 1e-6:
            q = u + sigma * np.log(m)
        else:
            q = u + (sigma / xi) * (m ** xi - 1.0)
        out[T] = float(q)
    return {'levels': out, 'u': u, 'xi': float(xi), 'sigma': float(sigma),
            'n_peak': int(len(peaks)), 'rate': float(rate),
            'n_years': float(n_years), 'max': float(x.max())}


def analyse(D, products=None):
    if products is None:
        products = B.PRODUCTS
    res = {}
    for name, df in D['series'].items():
        d = df.loc[FREQ_START:FREQ_END]
        ref_ok = d[B.REF].notna()
        out = {}
        for p in (B.REF,) + tuple(products):
            both = ref_ok & d[p].notna()          # ASOS 와 공통 유효일
            x = d.loc[both, p].dropna()
            ny = len(x) / 365.25
            r = pot_return_levels(x, ny)
            if r:
                r['n_days'] = int(len(x))
                out[p] = r
        res[name] = out
    return res


def ams_table(D):
    """연최대 일강수 (참고용). 표본이 4개라 빈도해석에는 못 쓴다."""
    rows = []
    for name, df in D['series'].items():
        d = df.loc[FREQ_START:FREQ_END]
        both = d[B.REF].notna() & d[B.MAIN].notna()
        for yr in sorted(set(d.index.year)):
            s = d[(d.index.year == yr) & both]
            if not len(s):
                continue
            io = s[B.REF].idxmax()
            ib = s[B.MAIN].idxmax()
            rows.append({
                '유역': name, '연도': yr, '공통유효일': len(s),
                'ASOS 연최대': round(float(s.loc[io, B.REF]), 1),
                '발생일': f'{io:%m-%d}',
                'BC_LR 같은날': round(float(s.loc[io, B.MAIN]), 1),
                'BC_LR 연최대': round(float(s.loc[ib, B.MAIN]), 1),
                '재현율(%)': round(100 * float(s.loc[io, B.MAIN])
                                 / float(s.loc[io, B.REF]), 1),
            })
    return pd.DataFrame(rows)


def main(ref='ASOS'):
    B.set_reference(ref, drop=('AWS',) if ref == 'ASOS' else ())
    D = B.load()
    pd.set_option('display.width', 220)
    circ = B.CIRCULAR.get(ref, ())
    print(f'평가 기준 = {ref}' + (f'   (순환 주의: {", ".join(circ)})' if circ else ''))

    print('=' * 100)
    print('0. 유역 내 관측소 — 미계측 여부')
    print('=' * 100)
    print('  논산천상류  ASOS 0개 · AWS 0개 (최근접 12.8 km)   → 완전 미계측')
    print('  조종천상류  ASOS 0개 · AWS 1개 (유역 내)          → ASOS 기준 미계측')
    print('  유등천상류  ASOS 0개 · AWS 0개 (최근접 12.3 km)   → 완전 미계측')

    print()
    print('=' * 100)
    print(f'1. 연최대 일강수 (참고) — {FREQ_START}~{FREQ_END}, ASOS·BC_LR 공통 유효일')
    print('=' * 100)
    print(ams_table(D).to_string(index=False))

    res = analyse(D)

    print()
    print('=' * 100)
    print('2. POT+GPD 적합 진단')
    print('=' * 100)
    rows = []
    for name, o in res.items():
        for p, r in o.items():
            rows.append({'유역': name, '자료': p, '유효일': r['n_days'],
                         '유효기간(년)': round(r['n_years'], 2),
                         '사상수': r['n_peak'], 'λ(건/년)': round(r['rate'], 2),
                         '임계 u(mm)': round(r['u'], 1),
                         '형상 ξ': round(r['xi'], 3),
                         '규모 σ': round(r['sigma'], 2),
                         '관측최대(mm)': round(r['max'], 1)})
    print(pd.DataFrame(rows).to_string(index=False))

    print()
    print('=' * 100)
    print('3. 확률강수량 (mm/일)')
    print('=' * 100)
    for name, o in res.items():
        print(f'\n── {name} ──')
        rows = []
        for p, r in o.items():
            rows.append({'자료': p,
                         **{f'{T}년': round(r['levels'][T], 1)
                            for T in RETURN_PERIODS}})
        print(pd.DataFrame(rows).to_string(index=False))

    print()
    print('=' * 100)
    print(f'4. ★ {B.REF} 대비 재현율 (%)  — 100%면 설계강우가 같다는 뜻')
    print('=' * 100)
    for name, o in res.items():
        if B.REF not in o:
            continue
        ref = o[B.REF]['levels']
        print(f'\n── {name} ──')
        rows = []
        for p, r in o.items():
            if p == B.REF:
                continue
            rows.append({'자료': p,
                         **{f'{T}년': round(100 * r['levels'][T] / ref[T], 1)
                            for T in RETURN_PERIODS}})
        print(pd.DataFrame(rows).to_string(index=False))

    print()
    print('=' * 100)
    print(f'5. {B.REF} 기준 T년 임계 초과일을 BC_LR 이 잡아내는가')
    print('=' * 100)
    rows = []
    for name, o in res.items():
        if B.REF not in o or B.MAIN not in o:
            continue
        d = D['series'][name].loc[FREQ_START:FREQ_END]
        d = d[d[B.REF].notna() & d[B.MAIN].notna()]
        for T in RETURN_PERIODS:
            th = o[B.REF]['levels'][T]
            hit_obs = d[d[B.REF] >= th]
            n_obs = len(hit_obs)
            n_cap = int((hit_obs[B.MAIN] >= th).sum()) if n_obs else 0
            rows.append({'유역': name, '빈도': f'{T}년',
                         f'{B.REF} 임계(mm)': round(th, 1),
                         f'{B.REF} 초과일수': n_obs,
                         'BC_LR 도 초과': n_cap,
                         '탐지율(%)': round(100 * n_cap / n_obs, 0)
                         if n_obs else None})
    print(pd.DataFrame(rows).to_string(index=False))

    print()
    print('=' * 100)
    print('주의')
    print('=' * 100)
    print('  · 자료기간 4년. 100년 빈도는 기록의 25배 외삽이라 절대값은 신뢰할 수 없다.')
    print('  · 의미가 있는 것은 "자료원만 바꿨을 때의 비(4번 표)"다.')
    print('  · BC_LR 은 결측일이 많아 유효기간이 ASOS 보다 짧다 (2번 표).')
    print('  · 정식 확률강수량은 30년 이상 기록으로 산정해야 한다.')
    return res


if __name__ == '__main__':
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument('--ref', default='ASOS', choices=['ASOS', 'AWS'])
    _a = _ap.parse_args([] if (hasattr(sys, 'ps1') or 'ipykernel' in sys.modules)
                        else None)
    main(_a.ref)
