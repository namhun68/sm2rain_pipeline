#!/usr/bin/env python3
"""자료 경로를 한곳에 모은 것.

각 단계 스크립트는 아직 제 안에 절대경로를 그대로 들고 있다.  이 파일은
그 경로들을 한 장에 모아 두고 **지금 이 컴퓨터에서 무엇이 있고 무엇이
없는지** 바로 확인하게 해 준다.  경로를 옮겼다면 여기와 해당 스크립트를
같이 고치면 된다.

    python3 paths.py            # 전부 확인
    python3 paths.py 04 05      # 그 단계에 필요한 것만

환경변수로 뿌리 경로를 덮어쓸 수 있다.
    KIHS_WORK      기본 /Users/kim/Desktop/work
    KIHS_CPUSERVER 기본 /Users/kim/cpuserver_data
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

WORK = Path(os.environ.get('KIHS_WORK', '/Users/kim/Desktop/work'))
CPU = Path(os.environ.get('KIHS_CPUSERVER', '/Users/kim/cpuserver_data'))

KIHS = WORK / 'KIHS'
DATA = KIHS / 'DATA'                      # 로컬 자료 (유역도 · AWS 시자료 · 캐시)
FINAL = DATA / '최종산출물'                # 납품 CSV
FIG_E = KIHS / 'fig_event'                # 호우사상 그림
FIG_B = KIHS / 'fig_basin'                # 전국·유역 그림

PJ = CPU / 'personal_data/project_KIHS'   # 파이프라인 중간·최종 격자
PRECIP = PJ / 'result/ASCAT/precipitation'
SM2R = PJ / 'result/SM2RAIN'
THIES = PJ / 'data/thiessen'              # 티센 산출물 (기존 방식 비교군)
SHAPE = PJ / 'data/shape'                 # 국가 표준유역도

# ──────────────────────────────────────────────────────────────────────────
# 단계별로 필요한 것 —  (경로, 설명, 필수 여부)
# ──────────────────────────────────────────────────────────────────────────
NEEDS: dict[str, list[tuple[Path, str, bool]]] = {
    '01_preprocess': [
        (Path('/Users/kim/data_1/GPM'), 'GPM IMERG 30분 원본 HDF5', True),
        (Path('/Users/kim/data_2/ERA5_Land/Precipitation'),
         'ERA5-Land 시간별 누계', True),
        (PJ / 'data/layer/ASCAT_daily_stack_KST.nc',
         'ASCAT 일 stack (보간 전)', True),
        (PJ / 'data/gpm_KST', 'GPM KST 일자료 (출력)', False),
        (PJ / 'data/era5_KST', 'ERA5 KST 일자료 (출력)', False),
        (PJ / 'data/layer/ASCAT_daily_stack_interp_KST.nc',
         'ASCAT 결측보간본 (IF_KST 출력)', False),
    ],
    '02_sm2rain': [
        (PJ / 'data/layer/ASCAT_daily_stack_KST.nc', 'ASCAT 일 stack', True),
        (Path('/Users/kim/DAS/projects/KIHS/IDW/both/da_IDWs.nc'),
         'IDW_ASOS · IDW_AWS 격자장 (캘리브 타깃)', True),
        (SM2R / 'SM2RAIN_KST.nc', 'SM2RAIN basic 산출 (출력, 채택본)', False),
        (SM2R / 'SM2RAIN_GA_KST_T.nc', 'SM2RAIN-GA 산출 (참고, 미채택)', False),
    ],
    '03_tca': [
        (SM2R / 'SM2RAIN_KST.nc', 'SM2RAIN 강수', True),
        (PJ / 'data/gpm_KST', 'GPM KST', True),
        (PJ / 'data/era5_KST', 'ERA5 KST', True),
        (CPU / 'python_modules/kunhee/Data/SM2RAIN/Korea.shp', '남한 마스크', True),
        (PRECIP / 'TCA', 'TCA 병합 결과 (출력)', False),
    ],
    '04_bias_correction': [
        (PRECIP / 'TCA', 'TCA 병합 결과', True),
        (Path('/Users/kim/DAS/projects/KIHS/IDW/both/da_IDWs.nc'),
         'IDW_ASOS · IDW_AWS (학습 목표 · 검증)', True),
        (CPU / 'personal_data/jaese/KIHS/output/260705',
         'BC12 입력 병합자료 (ds_merged_*)', True),
        (PRECIP / 'BC_LR_AWS_KST.nc',
         '픽셀별 선형회귀 결과 + SM2RAIN·GPM·ERA5·TCA·AWS·ASOS 격자', False),
        (PRECIP / 'BC12_fields_2021fit.nc',
         'LightGBM BC_1(BC) · BC_2(BC-G) 격자장', False),
    ],
    '05_export': [
        (PRECIP / 'BC12_fields_2021fit.nc', 'BC · BC-G 격자장', True),
        (PRECIP / 'BC_LR_AWS_KST.nc', '참고 산출물 격자', True),
        (SHAPE / 'std_basin_850.shp', '국가 표준유역도 850', True),
        (DATA / 'AWS', '기상청 시자료 (티센 산정용)', True),
        (FINAL, '납품 CSV (출력)', False),
        (THIES / 'THIESSEN_basin_daily.nc', '티센 유역 일강수 (출력)', False),
    ],
    'analysis': [
        (FINAL / '02_표준유역/BCG_basin_daily.csv', '유역 일강수 (BC-G)', True),
        (FINAL / '03_참고자료/ASOS_basin_daily.csv', '유역 일강수 (기준 ASOS)', True),
        (SHAPE / 'std_basin_850.shp', '국가 표준유역도 850', True),
        (DATA / 'basin_eval_cache.pkl', '유역 추출 캐시', False),
        (DATA / 'national_eval_cache.pkl', '전국 평가 캐시', False),
        (DATA / 'thiessen_aws.pkl', '티센 가중치·시계열 (3개 유역)', False),
        (THIES / 'THIESSEN_basin_daily.nc', '티센 유역 일강수 (전국 848)', False),
        (PRECIP / 'BC12_fields_2021fit.nc', 'BC · BC-G 격자장', True),
        (PRECIP / 'BC_LR_AWS_KST.nc', 'SM2RAIN·GPM·ERA5·TCA 격자', True),
        (DATA / 'basin_cell_weights.pkl', '유역×격자 교차면적 (없으면 만든다)', False),
        (DATA / 'LRG_THI_grid.nc', '최종 산출물 LR-G 격자장 (출력)', False),
    ],
    'figures': [
        (DATA / 'basin_eval_cache.pkl', '유역 캐시 (map·series·sums)', True),
        (DATA / 'thiessen_aws.pkl', '티센 가중치', True),
        (DATA / 'national_eval_cache.pkl', '전국 평가 캐시 (kge)', True),
        (PRECIP / 'BC12_fields_2021fit.nc', 'BC · BC-G 격자 (grid·graph)', False),
        (PJ / 'result/ASCAT/precipitation/BC_LR_AWS_KST.nc',
         'IDW_AWS · GPM 격자 (grid·graph)', False),
        (DATA / 'LRG_THI_grid.nc', '최종 산출물 LR-G 격자장 (report_lr)', False),
    ],
}

STAGES = list(NEEDS)


def check(stages=None) -> bool:
    """단계별 입력 존재 여부를 찍고, 필수가 다 있으면 True."""
    ok_all = True
    for st in (stages or STAGES):
        key = next((k for k in STAGES if k.startswith(st) or st in k), None)
        if key is None:
            print(f'[{st}] 그런 단계가 없습니다.  {STAGES}')
            ok_all = False
            continue
        print(f'\n■ {key}')
        for p, what, must in NEEDS[key]:
            ok = p.exists()
            mark = 'O' if ok else ('X' if must else '·')
            tag = '' if must else '  (출력/선택)'
            print(f'  {mark}  {what}{tag}\n       {p}')
            if must and not ok:
                ok_all = False
    return ok_all


if __name__ == '__main__':
    good = check(sys.argv[1:] or None)
    print('\n' + ('필수 입력이 모두 있습니다.' if good else
                  '필수 입력 가운데 없는 것이 있습니다 (위의 X).  '
                  '마운트를 확인하거나 그 단계는 건너뛰세요.'))
