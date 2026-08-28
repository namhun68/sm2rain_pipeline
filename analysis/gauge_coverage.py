"""
gauge_coverage.py — 표준유역별 지상관측 지점 배치 (미계측 여부 판정용)
================================================================================
    import gauge_coverage as G
    T = G.table()      # 유역코드 → asos_in / aws_in / d_asos / d_aws  (캐시)

지점 좌표는 기상청 지상관측 시간자료(DATA/AWS/Data_AWS_hourly_2022.csv)의
지점·위도·경도에서 얻는다. 기상청 지점번호 규약상 300 미만이 종관기상관측
(ASOS), 300 이상이 방재기상관측(AWS)이다.

거리는 유역 중심에서 지점까지의 직선거리다. 이 정의로 계산하면 보고서
2.2.7.1 의 값(논산천상류 15.1 km · 조종천상류 25.7 km · 유등천상류 11.5 km)이
그대로 재현된다.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from shapely.geometry import Point

import basin_eval_core as B

SRC = '/Users/kim/Desktop/work/KIHS/DATA/AWS/Data_AWS_hourly_2022.csv'
CACHE = '/Users/kim/Desktop/work/KIHS/DATA/gauge_coverage.pkl'


def stations():
    it = pd.read_csv(SRC, encoding='utf-8-sig',
                     usecols=['지점', '위도', '경도'], chunksize=2_000_000)
    g = pd.concat([c.groupby('지점')[['위도', '경도']].first() for c in it])
    g = g[~g.index.duplicated()].dropna()
    return g[g.index < 300], g[g.index >= 300]


def table(rebuild: bool = False) -> pd.DataFrame:
    if not rebuild and os.path.exists(CACHE):
        return pd.read_pickle(CACHE)
    asos, aws = stations()
    recs = B.read_dbf(B.SHP + '.dbf')
    offs = B.shp_offsets(B.SHP)
    key = next(k for k in recs[0] if 'BAS' in k.upper() or 'CD' in k.upper())
    rows = []
    for rec, (o, l) in zip(recs, offs):
        try:
            geom = B.read_polygon(B.SHP, o, l)
        except Exception:
            continue
        cx, cy = geom.centroid.x, geom.centroid.y
        k = 111.32 * np.cos(np.deg2rad(cy))
        da = np.hypot((asos['경도'] - cx) * k, (asos['위도'] - cy) * 110.57)
        dw = np.hypot((aws['경도'] - cx) * k, (aws['위도'] - cy) * 110.57)
        rows.append(dict(
            code=rec[key],
            asos_in=sum(1 for _, r in asos.iterrows()
                        if geom.contains(Point(r['경도'], r['위도']))),
            aws_in=sum(1 for _, r in aws.iterrows()
                       if geom.contains(Point(r['경도'], r['위도']))),
            d_asos=float(da.min()), d_aws=float(dw.min())))
    T = pd.DataFrame(rows).set_index('code')
    T.to_pickle(CACHE)
    return T


# 미계측 판정 : 유역 안에 ASOS·AWS 지점이 모두 없고, 유역 중심에서
#              최근접 ASOS 지점까지 D_MIN km 이상 떨어진 유역
D_MIN = 15.0


def ungauged(T=None):
    T = table() if T is None else T
    return T.index[(T.asos_in == 0) & (T.aws_in == 0) & (T.d_asos >= D_MIN)]


if __name__ == '__main__':
    T = table(rebuild=True)
    print('유역', len(T))
    print('ASOS 내부 없음      ', int((T.asos_in == 0).sum()))
    print('ASOS·AWS 모두 없음  ', int(((T.asos_in == 0) & (T.aws_in == 0)).sum()))
    print(f'그중 d_asos>={D_MIN}km  ', len(ungauged(T)))
    print(T.loc[['301301', '101503', '300903']].round(1))
