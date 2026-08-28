#!/usr/bin/env python3
"""단계 실행기 — 산출 순서를 그대로 돌린다.

    python3 run.py --list                 어떤 단계가 있는지
    python3 run.py --check                지금 이 컴퓨터에서 무엇이 되는지
    python3 run.py export analysis        고른 단계만
    python3 run.py figures                그림만
    python3 run.py --from 04              그 단계부터 끝까지

각 단계에 필요한 입력이 없으면 실행하지 않고 무엇이 없는지 알려 준다.
scipy·h5py 가 필요한 단계가 있어 파이썬을 고를 수 있게 두었다.

    python3 run.py 02 --python /Users/kim/miniconda3/bin/python3
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import paths

HERE = Path(__file__).resolve().parent

# 단계 이름 → (설명, 실행할 스크립트들)
#   순서가 곧 산출 순서다.  같은 단계 안의 스크립트는 위에서 아래로 돈다.
STAGES: dict[str, tuple[str, list[str]]] = {
    '01_preprocess': (
        '입력 자료를 KST 일경계로 맞춘다',
        ['01_preprocess/era5_KST.py',
         '01_preprocess/gpm_KST.py',
         '01_preprocess/IF_KST.py']),
    '02_sm2rain': (
        '토양수분에서 강수를 역산한다 (basic 3-parameter 채택)',
        ['02_sm2rain/SM2RAIN.py']),
    '03_tca': (
        '삼중병치분석으로 SM2RAIN·GPM·ERA5 를 오차분산 가중 병합한다',
        ['03_tca/TC.py']),
    '04_bias_correction': (
        '지상관측을 목표로 편향보정한다 (BC · BC-G 격자장)',
        ['04_bias_correction/make_BC12_fields.py']),
    '05_export': (
        '납품용 격자 · 표준유역 CSV 를 만든다',
        ['05_export/export_bcg_grid.py',
         '05_export/export_basin_csv.py',
         '05_export/export_thiessen_basin.py']),
    'analysis': (
        '전국 표준유역 단위로 평가한다 (표 · 지표)',
        ['analysis/national_eval.py']),
    'figures': (
        '보고서 그림을 만든다',
        ['figures/report.py']),
}

ORDER = list(STAGES)


def resolve(names) -> list[str]:
    """'04' 같은 짧은 이름도 받아 준다."""
    out = []
    for n in names:
        hit = [k for k in ORDER if k == n or k.startswith(n) or n in k]
        if not hit:
            raise SystemExit(f'그런 단계가 없습니다: {n}\n  {ORDER}')
        out.append(hit[0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stages', nargs='*', help='실행할 단계 (없으면 전부)')
    ap.add_argument('--list', action='store_true', help='단계 목록만 보기')
    ap.add_argument('--check', action='store_true', help='입력 존재 여부만 보기')
    ap.add_argument('--from', dest='start', help='이 단계부터 끝까지')
    ap.add_argument('--python', default=sys.executable, help='쓸 파이썬')
    ap.add_argument('--dry', action='store_true', help='실행하지 않고 명령만 보기')
    ap.add_argument('--force', action='store_true', help='입력이 없어도 밀어붙이기')
    a = ap.parse_args()

    if a.list:
        for k in ORDER:
            what, scripts = STAGES[k]
            print(f'{k:18s} {what}')
            for s in scripts:
                print(f'{"":20s}{s}')
        return

    want = ORDER[ORDER.index(resolve([a.start])[0]):] if a.start else \
        (resolve(a.stages) if a.stages else ORDER)

    if a.check:
        paths.check(want)
        return

    for k in want:
        what, scripts = STAGES[k]
        print(f'\n{"=" * 74}\n■ {k} — {what}\n{"=" * 74}')
        if not a.force and k in paths.NEEDS and not paths.check([k]):
            print(f'\n[건너뜀] {k} 의 필수 입력이 없습니다. '
                  '마운트를 확인하거나 --force 로 밀어붙이세요.')
            continue
        for s in scripts:
            p = HERE / s
            if not p.exists():
                print(f'[없음] {s}')
                continue
            cmd = [a.python, str(p)]
            print('$', ' '.join(cmd))
            if a.dry:
                continue
            r = subprocess.run(cmd, cwd=p.parent)
            if r.returncode:
                raise SystemExit(f'\n[중단] {s} 가 코드 {r.returncode} 로 끝났습니다.')


if __name__ == '__main__':
    main()
