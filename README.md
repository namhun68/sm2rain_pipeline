# 다종자료 융합 강우 공간분포 산정

위성 토양수분에서 역산한 강수(SM2RAIN), 위성 강수(GPM IMERG), 재분석(ERA5-Land)을
삼중병치분석으로 병합하고 지상관측을 목표로 편향보정하여, 남한 전역의 격자 일강수
자료를 산출한다. 결과는 격자와 국가 표준유역 두 단위로 제공한다.

발주 한국수자원조사기술원(KIHS) · 수행 광주과학기술원(GIST)

| | |
|---|---|
| 최종 산출물 | LR-G — 지상관측 융합 · 티센 목표 격자별 선형회귀 강우장 |
| 공간 | 0.1° 정방 격자 991칸 · 국가 표준유역 848개 |
| 기간 | 2021-01-01 ~ 2025-05-01 (1,575일) |
| 단위 | mm/day · KST · 일 경계 00–24시 |

---

## 방법

다섯 단계를 거친다. 각 단계는 앞 단계의 출력을 입력으로 받는다.

```
   ASCAT 토양수분 ┐
   GPM IMERG      ├─→ 01_preprocess ─→ 02_sm2rain ─→ 03_tca ─→ 04_bias_correction
   ERA5-Land      │                                                    │
   ASOS · AWS     ┘                                                    ↓
                                                                  05_export
                                                                       ↓
                                                          analysis ─→ figures
```

| 단계 | 내용 |
|---|---|
| **01 전처리** | 위성·재분석은 UTC, 지상관측은 KST 일경계다. 시간별·30분 원자료에서 다시 집계해 전부 KST 00–24시로 맞춘다. ASCAT 결측은 지수필터로 보간한다. |
| **02 SM2RAIN** | 토양수분의 시간 변화에서 강수를 역산한다. `P = Z·Δθ + a·θ^b` 3-파라미터 형태를 쓰고 2021년 지상관측으로 픽셀별 보정한다. |
| **03 삼중병치분석** | SM2RAIN·GPM·ERA5는 관측 원리가 서로 달라 오차가 독립이다. 참값 없이 각 자료의 오차분산을 추정해 그 역수를 가중치로 병합한다. |
| **04 편향보정** | 병합장에 남은 계통 편의를 LightGBM으로 보정한다. 지상관측 입력을 쓰지 않는 BC와, 같은 날 관측을 융합하는 BC-G 두 가지를 산출한다. |
| **05 산출** | 격자 자료와 표준유역 평균을 CSV·NetCDF로 낸다. 유역 평균은 격자–유역 교차면적 가중이다. |

### 산출물

| 이름 | 설명 |
|---|---|
| **LR-G** | 지상관측을 함께 넣고 티센을 목표로 한 격자별 선형회귀 강우장. 최종 산출물 |
| **BC-G** | 같은 날 지상관측을 입력으로 융합한 LightGBM 편향보정. LR-G 의 직전 단계 |
| **BC** | 지상관측을 입력으로 쓰지 않는 편향보정 강우장. 미계측 지역 확장용 |
| TCA | 삼중병치 병합장. 편향보정 전 |
| SM2RAIN | 토양수분 역산 강수 |
| 티센 | 지점 면적비 가중 유역평균. LR-G 의 학습 목표이자 평가 기준 |
| BC-LR | 픽셀별 선형회귀 편향보정. 기계학습 이전 방식과의 비교군 |

---

## 저장소 구성

```
01_preprocess/       era5_KST · gpm_KST · IF_KST · ASOS_AWS_preprocessing
02_sm2rain/          SM2RAIN · ASCAT_SM2RAIN_GA_KST
03_tca/              TC · TC_AWS
04_bias_correction/  make_BC12_fields · BC_LR_AWS · LOOV
05_export/           export_bcg_grid · export_basin_csv · export_thiessen_basin
                     export_BC_LR_csv
analysis/            basin_eval_core · national_eval · thiessen_basin
                     lr_thiessen · basin_freq · gauge_coverage
figures/             report · report_lr

paths.py             자료 경로 정의와 존재 확인
run.py               단계 실행기
```

| 스크립트 | 역할 |
|---|---|
| `01_preprocess/era5_KST.py` | ERA5-Land 시간별 누계 해제 후 KST 일합산 |
| `01_preprocess/gpm_KST.py` | GPM 30분 자료 KST 일합산 및 재격자화 |
| `01_preprocess/IF_KST.py` | ASCAT 일 stack |
| `01_preprocess/ASOS_AWS_preprocessing.py` | 지상관측 정리 |
| `02_sm2rain/SM2RAIN.py` | 토양수분 역산 강수 (채택) |
| `02_sm2rain/ASCAT_SM2RAIN_GA_KST.py` | 5-파라미터 변형 (검토 결과 미채택) |
| `03_tca/TC.py` | 삼중병치분석 병합 |
| `04_bias_correction/make_BC12_fields.py` | BC·BC-G 격자장 생성 |
| `04_bias_correction/BC_LR_AWS.py` | 픽셀별 선형회귀 편향보정 |
| `04_bias_correction/LOOV.py` | 지점 제외 교차검증 |
| `05_export/export_bcg_grid.py` | 격자 CSV (원자료·보간본·플래그) |
| `05_export/export_basin_csv.py` | 표준유역 CSV 및 유역 제원 |
| `05_export/export_thiessen_basin.py` | 티센 유역 일강수 |
| `analysis/basin_eval_core.py` | 유역 추출·면적가중 평균·평가지표 |
| `analysis/national_eval.py` | 전국 표준유역 성능 평가 |
| `analysis/thiessen_basin.py` | 티센 가중치 산정 |
| `analysis/lr_thiessen.py` | 최종 산출물 LR-G — 티센 목표 격자별 선형회귀와 전국 평가 |
| `figures/report.py` | 결과 그림 |
| `figures/report_lr.py` | 최종 산출물 LR-G 전국 검증 그림 |

---

## 실행

```bash
python3 run.py --list      # 단계 목록
python3 run.py --check     # 필요한 입력이 갖춰졌는지 확인
python3 run.py             # 전체 실행
python3 run.py --from 05   # 05단계부터 끝까지
python3 run.py figures     # 그림만
```

단계마다 필요한 입력을 먼저 확인하고, 갖춰지지 않았으면 건너뛰면서 무엇이 없는지
알린다. 경로는 [`paths.py`](paths.py)에 모여 있고 환경변수 `KIHS_WORK`,
`KIHS_CPUSERVER`로 바꿀 수 있다.

```bash
python3 paths.py 04 05     # 특정 단계의 입력만 확인
```

전처리와 편향보정 단계는 원자료와 서버 마운트를 필요로 한다. 산출 이후 단계는
캐시 파일만으로 실행된다.

### 그림

```bash
python3 figures/report.py                 # 전체
python3 figures/report.py grid graph      # 선택 실행
```

| 이름 | 내용 |
|---|---|
| `map` | 유역별 티센 분할, 산출 격자, 관측소 배치 |
| `series` | 호우사상 전후 일강수 시계열 |
| `sums` | 호우사상 누적 강수량 |
| `grid` | 자료별 격자 강수 비교 (피크 전·중·후·전체) |
| `graph` | 자료별 일별 및 구간 누적 비교 |
| `kge` | 전국 표준유역 KGE 공간분포 |

---

## 입력 자료

| 자료 | 종류 | 해상도 | 용도 |
|---|---|---|---|
| ASCAT SOMO12 | 위성 토양수분 | 12.5 km · swath | SM2RAIN 입력 |
| GPM IMERG | 위성 강수 | 0.1° · 30분 | 병합 입력 |
| ERA5-Land | 재분석 강수 | 0.1° · 시간별 | 병합 입력 |
| ASOS | 종관기상관측 117지점 | 지점 · 시간별 | 독립 검증 |
| AWS | 방재기상관측 635지점 | 지점 · 시간별 | 보정 학습 목표 |
| 수자원단위지도 | 국가 표준유역 850 | 폴리곤 | 유역 집계 |

---

## 설계 규약

- **일 경계는 KST 00–24시로 통일한다.** 위성·재분석은 UTC 기준이라 지상관측과
  하루가 어긋난다. 이미 일합산된 자료를 옮기면 라벨만 바뀌므로 시간별 원자료에서
  다시 집계한다.

- **학습 목표와 검증 기준을 분리한다.** 편향보정은 방재관측(AWS)을 목표로 학습하고
  성능 평가는 종관관측(ASOS)으로 한다. 관측망이 서로 달라 준독립이다. 보정
  적합연도 2021년을 제외한 2022년 이후가 독립검증 구간이다.

- **유역 평균은 격자–유역 교차면적 가중으로 구한다.** 0.1° 격자 한 칸이 약 100 km²,
  표준유역 면적 중앙값이 113 km²라 유역 하나가 격자 4–7개에 걸친다. 중심 격자값이나
  단순 평균을 쓰면 유역 밖 강수가 섞인다.

- **티센 가중치는 250 m 격자 최근접지점 배정으로 구한다.** 폴리곤 교차를 직접 푸는
  것과 결과가 같고 구현이 단순하다. 관측이 결측인 날은 남은 지점의 가중치를 다시
  정규화한다.

- **중간 해상도와 최종 해상도를 구분한다.** ASCAT 재격자화는 0.125°에서 이뤄지고
  최종 산출물은 0.1°다.

- 지점번호 300 미만은 종관관측(ASOS), 300 이상은 방재관측(AWS)이다.

---

## 요구 환경

Python 3.10 이상, `numpy` `pandas` `xarray` `scipy` `shapely` `matplotlib`
`lightgbm` `h5py`. 자료 파일(`*.nc` `*.csv` `*.pkl`)과 그림은 저장소에 포함하지
않는다.
