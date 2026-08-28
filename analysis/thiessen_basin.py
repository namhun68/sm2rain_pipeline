#!/usr/bin/env python3
"""표준유역의 티센 가중치를 구하고, AWS 지점 강수로 유역평균을 산정한다.

과업에서는 티센을 쓰지 않았다(면적가중 격자평균을 썼다). 이 스크립트는
"기존 방식(티센)과 비교하면 어떤가" 를 확인하기 위한 별도 계산이다.

가중치는 유역을 250 m 격자로 잘게 나눈 뒤 각 칸을 최근접 지점에 배정하여
그 면적 비율로 구한다. 폴리곤 교차를 직접 푸는 것과 결과가 같고 훨씬 간단하다.
"""
import numpy as np
import basin_eval_core as B

CELL = 250.0          # 가중치 산정용 세부 격자 (m)


def to5179(lon, lat):
    """WGS84 → EPSG:5179 (한국 중부원점 TM). pyproj 없이 직접 계산."""
    a, f = 6378137.0, 1 / 298.257222101
    e2 = f * (2 - f)
    k0, lat0, lon0 = 0.9996, np.deg2rad(38.0), np.deg2rad(127.5)
    FE, FN = 1000000.0, 2000000.0
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    e_2 = e2 / (1 - e2)
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    T = np.tan(lat) ** 2
    C = e_2 * np.cos(lat) ** 2
    A = (lon - lon0) * np.cos(lat)

    def M(p):
        return a * ((1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * p
                    - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024)
                    * np.sin(2 * p)
                    + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * np.sin(4 * p)
                    - (35 * e2**3 / 3072) * np.sin(6 * p))

    x = FE + k0 * N * (A + (1 - T + C) * A**3 / 6
                       + (5 - 18 * T + T**2 + 72 * C - 58 * e_2) * A**5 / 120)
    y = FN + k0 * (M(lat) - M(lat0)
                   + N * np.tan(lat) * (A**2 / 2
                                        + (5 - T + 9 * C + 4 * C**2) * A**4 / 24
                                        + (61 - 58 * T + T**2 + 600 * C
                                           - 330 * e_2) * A**6 / 720))
    return x, y


def in_poly(px, py, rings):
    """셸에 들고 홀에 안 든 점만 True."""
    inside = np.zeros(px.shape, bool)
    for ring, is_hole in rings:
        rx, ry = ring[:, 0], ring[:, 1]
        c = np.zeros(px.shape, bool)
        n = len(rx)
        j = n - 1
        for i in range(n):
            cond = ((ry[i] > py) != (ry[j] > py))
            with np.errstate(divide='ignore', invalid='ignore'):
                xint = (rx[j] - rx[i]) * (py - ry[i]) / (ry[j] - ry[i]) + rx[i]
            c ^= cond & (px < xint)
            j = i
        inside = (inside & ~c) if is_hole else (inside | c)
    return inside


def weights(rings_ll, stn_lon, stn_lat, cell=CELL):
    """유역(경위도 링) 과 지점 목록에서 티센 가중치를 구한다."""
    rings = []
    for ring, is_hole in rings_ll:
        x, y = to5179(ring[:, 0], ring[:, 1])
        rings.append((np.column_stack([x, y]), is_hole))
    allx = np.concatenate([r[:, 0] for r, _ in rings])
    ally = np.concatenate([r[:, 1] for r, _ in rings])
    gx = np.arange(allx.min(), allx.max() + cell, cell)
    gy = np.arange(ally.min(), ally.max() + cell, cell)
    GX, GY = np.meshgrid(gx, gy)
    m = in_poly(GX, GY, rings)
    if not m.any():
        return {}, 0.0
    px, py = GX[m], GY[m]
    sx, sy = to5179(np.asarray(stn_lon), np.asarray(stn_lat))
    d2 = (px[:, None] - sx[None, :]) ** 2 + (py[:, None] - sy[None, :]) ** 2
    near = np.argmin(d2, axis=1)
    cnt = np.bincount(near, minlength=len(sx))
    tot = cnt.sum()
    w = {i: c / tot for i, c in enumerate(cnt) if c}
    return w, tot * cell ** 2 / 1e6      # 가중치, 면적 km²
