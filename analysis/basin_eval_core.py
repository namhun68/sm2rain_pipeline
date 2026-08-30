"""
basin_eval_core.py — 표준유역 단위로 최종산출물(BC_LR)을 ASOS 기준 평가
================================================================================
논산천상류 · 조종천상류 · 유등천상류 3개 표준유역에서
BC_LR(최종 통합본)을 포함한 각 산출물을 ASOS(IDW 보간) 기준으로 평가한다.

    import basin_eval_core as B
    D = B.load()                      # 유역추출 + 면적가중 시계열 (캐시됨)
    D['series']['논산천상류']          # DataFrame: 열=산출물, 행=날짜
    B.metrics(sim, obs)               # 지표 한 벌

핵심 설계
  · 유역 평균은 **격자-유역 교차면적 가중**이다.
    유역이 140~160 km2 인데 0.1도 격자 한 칸이 약 100 km2 라 유역이 격자
    4~7개에 걸친다. 단순 포함격자 평균을 쓰면 유역 밖 강수가 섞인다.
  · geopandas 없이 돈다. shapefile 을 직접 읽고 shapely 로만 처리한다.
  · 유역 유효면적이 절반 미만인 날은 결측 처리한다 (부분 관측 왜곡 방지).

BC_LR 은 2021년으로 적합했으므로 2022년부터가 독립검증 구간이다.
"""
from __future__ import annotations

import os
import pickle
import struct

import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

# ------------------------------------------------------------------ 설정
#  표준유역도.  서버 자료폴더에 두고, 없으면 로컬 사본을 쓴다
SHP = next((p for p in (
    '/Users/kim/cpuserver_data/personal_data/project_KIHS/data/shape/std_basin_850',
    '/Users/kim/Desktop/work/KIHS/DATA/std_basin_850/std_basin_850')
    if os.path.exists(p + '.shp')),
    '/Users/kim/cpuserver_data/personal_data/project_KIHS/data/shape/std_basin_850')
_P = '/Users/kim/cpuserver_data/personal_data/project_KIHS/result/ASCAT/precipitation/'
NC = _P + 'BC_LR_AWS_KST.nc'          # SM2RAIN/GPM/ERA5/TCA/AWS/ASOS/BC_LR

# 보고서의 BC · BC-G 는 별도 파일에 BC_1 / BC_2 로 들어 있다.
#   BC_1  "LightGBM BC without AWS input"      → 보고서 BC   (미계측 지역 확장용)
#   BC_2  "LightGBM BC with AWS input(fusion)" → 보고서 BC-G (관측 보강용)
#   파일 주석: "BC_2 uses same-day AWS as input — near-AWS by design"
NC_BC12 = _P + 'BC12_fields_2021fit.nc'
BC12_MAP = {'BC': 'BC_1', 'BC_G': 'BC_2'}

CACHE = '/Users/kim/Desktop/work/KIHS/DATA/basin_eval_cache.pkl'

TARGETS = {'301301': '논산천상류', '101503': '조종천상류', '300903': '유등천상류'}

# 시계열은 항상 이만큼 만들어 두고, 비교표에서만 고른다.
ALL_VARS = ['BC', 'BC_G', 'BC_LR', 'TCA', 'AWS', 'GPM', 'ERA5', 'SM2RAIN',
            'ASOS', 'THIESSEN']

# 티센(AWS) 은 격자가 아니라 지점 가중 평균이라 캐시와 별도로 만들어 둔다.
THIESSEN_PKL = '/Users/kim/Desktop/work/KIHS/DATA/thiessen_aws.pkl'
PRODUCTS_ALL = [v for v in ALL_VARS if v != 'ASOS']

# 기본 비교군.  대상이 '미계측 유역'이므로 주력은 BC 다.
#   보고서 2.2.4 [267]: "기본 편향보정(BC)은 … 관측이 없는 지역으로의 일반화에
#   유리하다. 반면 BC-G 는 … 관측 보강형 산출물로 해석된다."
# BC-G 는 같은 날 AWS 를 입력으로 먹으므로 미계측 유역에는 쓸 수 없다.
# 표에는 '관측이 있을 때의 상한'으로만 남긴다.
# BC_LR(픽셀별 선형회귀)은 보고서가 '기존 방식'으로 규정하므로 기본 비교군에서 뺀다.
DEFAULT_PRODUCTS = ['BC_G', 'BC', 'TCA', 'GPM', 'ERA5', 'SM2RAIN']

# 상세 분석(산점도·강도별·계절별·빈도)의 주 대상 산출물.
#   BC-G 가 보고서의 최종 산출물이다. 유역 내 관측소가 없어도 IDW_AWS 격자장은
#   존재하므로 미계측 유역에도 적용된다. 기준 ASOS 는 AWS 와 다른 관측망이므로
#   BC-G 를 ASOS 로 평가하는 것은 순환이 아니다.
MAIN = 'BC_G'

# 기준자료. set_reference() 로 바꾼다.
#   ASOS  96지점.  BC_LR·TCA 와 독립이라 순환이 없다. 대신 대상 유역에서
#         최근접 지점이 11~26 km 라 기준 자체의 공간대표성이 약하다.
#   AWS   556지점. 훨씬 조밀해 공간대표성은 낫다. 그러나 BC_LR 의 학습 target
#         이고 TCA 의 삼중분석 멤버라, 이 둘의 점수는 순환으로 부풀려진다.
#         GPM·ERA5·SM2RAIN 은 AWS 와 독립이므로 영향을 받지 않는다.
REF = 'ASOS'
PRODUCTS = list(DEFAULT_PRODUCTS)


def add_thiessen(D) -> bool:
    """유역 시계열에 티센(AWS) 열을 붙인다. 파일이 없으면 조용히 지나간다."""
    import os
    import pickle
    if not os.path.exists(THIESSEN_PKL):
        return False
    t = pickle.load(open(THIESSEN_PKL, 'rb'))['series']
    n = 0
    for name, df in D['series'].items():
        if name in t:
            df['THIESSEN'] = t[name].reindex(df.index)
            n += 1
    return n > 0


def set_reference(ref: str, products=None, drop: tuple[str, ...] = ()) -> None:
    """평가 기준자료를 바꾼다. 기준으로 쓴 변수는 비교군에서 자동으로 빠진다."""
    global REF, PRODUCTS
    if ref not in ALL_VARS:
        raise ValueError(f'{ref} 는 없는 변수입니다. {ALL_VARS}')
    REF = ref
    base = list(products) if products else list(DEFAULT_PRODUCTS)
    if ref == 'AWS' and 'ASOS' not in base:
        base.append('ASOS')          # AWS 기준일 때 독립 관측망을 참고로 남긴다
    PRODUCTS = [v for v in base if v != ref and v not in drop]


# 기준에 대해 순환(circularity)이 있는 산출물 — 표에 ※ 로 표시한다
#   BC   : 목표변수가 IDW_AWS  → AWS 기준이면 순환
#   BC_G : 같은 날 AWS 를 입력으로 융합 → AWS 기준이면 강한 순환
#   TCA  : AWS 를 삼중병치 멤버로 포함 → AWS 기준이면 순환
#   ASOS 기준에서는 넷 다 독립이므로 순환이 없다.
CIRCULAR = {'AWS': ('BC', 'BC_G', 'BC_LR', 'TCA'), 'ASOS': ()}

# 지상관측을 입력으로 쓰는 산출물 (표에 † 로 표시)
GAUGE_FUSED = ('BC_G',)
FIT_YEAR = 2021                 # BC_LR 적합연도 → 독립검증은 2022부터
INDEP_START = '2022-01-01'

# 산출물 색 (그림 전체에서 공유)
COLORS = {
    'BC':      '#D1495B',       # 주력 산출물(미계측용) — 강조색
    'BC_G':    '#8E3B46',       # 관측 보강형 — 같은 계열의 짙은 색
    'BC_LR':   '#E08A96',       # 기존 픽셀별 선형회귀 (참고)
    'TCA':     '#EDAE49',
    'AWS':     '#00798C',
    'GPM':     '#66A182',
    'ERA5':    '#6C91BF',
    'SM2RAIN': '#B08EA2',
    'ASOS':    '#2E2E2E',       # 기준
    'THIESSEN': '#9A6FB0',      # 기존 방식 (티센 다각형)
}
BASIN_COLORS = {'논산천상류': '#2E5EAA', '조종천상류': '#D1495B',
                '유등천상류': '#3B8C6E'}

LABEL = {'BC': 'BC', 'BC_G': 'BC-G', 'BC_LR': 'BC-LR (기존)',
         'TCA': 'TCA (보정전)', 'AWS': 'IDW_AWS', 'GPM': 'GPM', 'ERA5': 'ERA5',
         'SM2RAIN': 'SM2RAIN', 'ASOS': 'IDW_ASOS',
         'THIESSEN': '티센(AWS)'}


# ------------------------------------------------------------------ shapefile
def read_dbf(path: str) -> list[dict]:
    """dbf 속성표. std_basin_850 은 cpg 가 UTF-8 이다."""
    with open(path, 'rb') as f:
        nrec, hlen, rlen = struct.unpack('<IHH', f.read(32)[4:12])
        flds = []
        for _ in range((hlen - 33) // 32):
            d = f.read(32)
            flds.append((d[:11].split(b'\x00')[0].decode('latin1'), d[16]))
        f.seek(hlen)
        out = []
        for _ in range(nrec):
            rec = f.read(rlen)
            off, row = 1, {}
            for nm, ln in flds:
                row[nm] = rec[off:off + ln].decode('utf-8', 'replace').strip()
                off += ln
            out.append(row)
    return out


def shp_offsets(path: str) -> list[tuple[int, int]]:
    """shx 로 레코드별 (바이트오프셋, 길이)."""
    with open(path + '.shx', 'rb') as f:
        f.seek(24)
        flen = struct.unpack('>I', f.read(4))[0] * 2
        f.seek(100)
        out = []
        for _ in range((flen - 100) // 8):
            o, l = struct.unpack('>II', f.read(8))
            out.append((o * 2, l * 2))
    return out


def read_polygon(path: str, offset: int, length: int):
    """폴리곤 레코드 → shapely 도형.

    shapefile 규약상 껍질은 시계방향(부호면적 음수), 구멍은 반시계방향이다.
    이 부호로 둘을 가른 뒤 구멍을 제 껍질에 붙인다.
    """
    with open(path + '.shp', 'rb') as f:
        f.seek(offset + 8)
        buf = f.read(length)
    stype = struct.unpack('<i', buf[:4])[0]
    if stype != 5:
        raise ValueError(f'폴리곤이 아닙니다 (shape type {stype})')
    nparts, npts = struct.unpack('<ii', buf[36:44])
    parts = struct.unpack(f'<{nparts}i', buf[44:44 + 4 * nparts])
    p0 = 44 + 4 * nparts
    xy = np.frombuffer(buf[p0:p0 + 16 * npts], dtype='<f8').reshape(npts, 2)
    idx = list(parts) + [npts]

    shells, holes = [], []
    for i in range(nparts):
        ring = xy[idx[i]:idx[i + 1]]
        if len(ring) < 4:
            continue
        a = 0.5 * np.sum(ring[:-1, 0] * ring[1:, 1] - ring[1:, 0] * ring[:-1, 1])
        (holes if a > 0 else shells).append(ring)

    polys = []
    for s in shells:
        sp = Polygon(s)
        if not sp.is_valid:
            sp = sp.buffer(0)
        mine = [h for h in holes
                if sp.contains(Polygon(h).representative_point())]
        p = Polygon(s, mine) if mine else sp
        polys.append(p if p.is_valid else p.buffer(0))
    return unary_union(polys) if len(polys) > 1 else polys[0]


# ------------------------------------------------------------------ 가중치
def cell_weights(geom, lat, lon):
    """격자셀 × 유역 교차면적 → [(ilat, ilon, km2), ...]"""
    dlat = float(abs(lat[1] - lat[0]))
    dlon = float(abs(lon[1] - lon[0]))
    minx, miny, maxx, maxy = geom.bounds
    out = []
    for i, la in enumerate(lat):
        if la + dlat / 2 < miny or la - dlat / 2 > maxy:
            continue
        for j, lo in enumerate(lon):
            if lo + dlon / 2 < minx or lo - dlon / 2 > maxx:
                continue
            inter = geom.intersection(
                box(lo - dlon / 2, la - dlat / 2, lo + dlon / 2, la + dlat / 2))
            if inter.is_empty:
                continue
            km2 = inter.area * (111.32 ** 2) * np.cos(np.deg2rad(la))
            if km2 > 1e-6:
                out.append((i, j, float(km2)))
    return out


# ------------------------------------------------------------------ 지표
def metrics(sim, obs, rain_th: float = 1.0) -> dict:
    """R, KGE, NSE, bias, RMSE, ubRMSE, α, β + 강우탐지 POD/FAR/CSI."""
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    m = np.isfinite(sim) & np.isfinite(obs)
    s, o = sim[m], obs[m]
    n = int(len(s))
    if n < 30:
        return {'n': n}
    r = float(np.corrcoef(s, o)[0, 1])
    bias = float(s.mean() - o.mean())
    alpha = float(s.std() / o.std()) if o.std() > 0 else np.nan
    beta = float(s.mean() / o.mean()) if o.mean() > 0 else np.nan
    hit = int(((s >= rain_th) & (o >= rain_th)).sum())
    fa = int(((s >= rain_th) & (o < rain_th)).sum())
    ms = int(((s < rain_th) & (o >= rain_th)).sum())
    return {
        'n': n, 'R': r, 'bias': bias,
        'rbias': float(100 * bias / o.mean()) if o.mean() > 0 else np.nan,
        'RMSE': float(np.sqrt(((s - o) ** 2).mean())),
        'ubRMSE': float(np.sqrt((((s - s.mean()) - (o - o.mean())) ** 2).mean())),
        'alpha': alpha, 'beta': beta,
        'KGE': float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)),
        'NSE': float(1 - ((s - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()),
        'POD': hit / (hit + ms) if hit + ms else np.nan,
        'FAR': fa / (hit + fa) if hit + fa else np.nan,
        'CSI': hit / (hit + fa + ms) if hit + fa + ms else np.nan,
        'sim_mean': float(s.mean()), 'obs_mean': float(o.mean()),
        'sim_sum': float(s.sum()), 'obs_sum': float(o.sum()),
    }


# ------------------------------------------------------------------ 본체
def build() -> dict:
    """유역 추출 → 면적가중 유역평균 시계열."""
    recs = read_dbf(SHP + '.dbf')
    offs = shp_offsets(SHP)
    ds = xr.open_dataset(NC)
    lat, lon = ds.lat.values, ds.lon.values
    times = pd.to_datetime(ds.time.values)

    # BC / BC-G 를 별도 파일에서 붙인다. 격자·기간이 동일한지 확인한다.
    d2 = xr.open_dataset(NC_BC12)
    assert d2.sizes['lat'] == ds.sizes['lat'] and d2.sizes['lon'] == ds.sizes['lon'], \
        'BC12 파일의 격자가 다릅니다'
    assert len(pd.to_datetime(d2.time.values)) == len(times), \
        'BC12 파일의 기간이 다릅니다'
    ARR = {v: ds[v].values for v in ds.data_vars}
    for name, src in BC12_MAP.items():
        ARR[name] = d2[src].values
    d2.close()

    info, series, geoms = {}, {}, {}
    for code, name in TARGETS.items():
        k = next(i for i, r in enumerate(recs) if r['SBSN_CD'].strip() == code)
        geom = read_polygon(SHP, *offs[k])
        w = cell_weights(geom, lat, lon)
        tot = sum(a for _, _, a in w)
        geoms[name] = geom
        info[name] = {
            'code': code, 'ncell': len(w), 'cells': w,
            'area': float(geom.area * (111.32 ** 2)
                          * np.cos(np.deg2rad(geom.centroid.y))),
            'cx': float(geom.centroid.x), 'cy': float(geom.centroid.y),
            'top_frac': 100 * max(a for _, _, a in w) / tot,
        }
        d = {}
        # 시계열은 있는 변수를 전부 만들어 둔다. 비교표에서만 고른다.
        for var in ALL_VARS:
            if var not in ARR:
                continue
            arr = ARR[var]
            num = np.zeros(len(times))
            den = np.zeros(len(times))
            for (i, j, a) in w:
                v = arr[:, i, j]
                ok = np.isfinite(v)
                num[ok] += a * v[ok]
                den[ok] += a
            out = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
            out[den < 0.5 * tot] = np.nan      # 유역 절반도 못 덮은 날은 버린다
            d[var] = out
        series[name] = pd.DataFrame(d, index=times)

    ds.close()
    return {'info': info, 'series': series, 'geoms': geoms,
            'lat': lat, 'lon': lon}


def load(rebuild: bool = False) -> dict:
    """캐시가 있으면 읽고, 없으면 만들어 저장한다 (유역추출이 수십 초 걸린다)."""
    if not rebuild and os.path.exists(CACHE):
        with open(CACHE, 'rb') as f:
            return pickle.load(f)
    D = build()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, 'wb') as f:
        pickle.dump(D, f)
    return D


def summary_table(D: dict, start: str | None = INDEP_START) -> pd.DataFrame:
    """유역 × 산출물 지표표."""
    rows = []
    for name, df in D['series'].items():
        d = df.loc[start:]
        for var in PRODUCTS:
            m = metrics(d[var].values, d[REF].values)
            if 'R' not in m:
                continue
            rows.append({'유역': name, '자료': var, **{
                k: round(m[k], 3) for k in
                ('n', 'R', 'KGE', 'NSE', 'bias', 'rbias', 'RMSE', 'ubRMSE',
                 'alpha', 'beta', 'POD', 'FAR', 'CSI')}})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    D = load(rebuild=True)
    print('=' * 84)
    for name, v in D['info'].items():
        print(f"{name}({v['code']})  {v['area']:6.1f} km²  "
              f"격자 {v['ncell']}개 (최대 {v['top_frac']:.1f}%)  "
              f"중심 {v['cy']:.3f}N {v['cx']:.3f}E")
    print('=' * 84)
    print(summary_table(D).to_string(index=False))
