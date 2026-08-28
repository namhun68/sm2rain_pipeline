import os
import sys
import platform
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
warnings.filterwarnings("ignore")

if platform.system() == 'Darwin': # Mac Mini
    base_FP = '/Users/geobambus'
    cpuserver_data_FP = os.path.join(base_FP, 'cpuserver_data')
    george_FP = os.path.join(cpuserver_data_FP, 'python_modules', 'kunhee')
elif platform.system() == 'Linux': # Workstation
    base_FP = '/home/kunhee'
    if os.path.exists(base_FP):
        cpuserver_data_FP = os.path.join(base_FP, 'cpuserver_data')
        nas_FP = os.path.join(base_FP, 'NAS')
        das_FP = os.path.join(base_FP, 'DAS')
        george_FP = os.path.join(cpuserver_data_FP, 'python_modules', 'kunhee')
    else: # CPU Server
        base_FP = '/data'
        cpuserver_data_FP = '/data'
        nas_FP = '/data'
        das_FP = '/data'
        george_FP = os.path.join('cpuserver_data', 'python_modules', 'kunhee')

sys.path.append(os.path.join(george_FP, 'george_FP'))
import HydroAI.Grid as hGrid

grid_res = 0.25 
eqd_lon, eqd_lat = hGrid.generate_lon_lat_eqdgrid(grid_res)

def process_and_cut_stations(data_fp, meta_fp, network_type, out_dir, valid_dates):
    print(f"Processing {network_type} data...")
    
    df = pd.read_csv(data_fp)

    if '경도관서' not in df.columns or '위도관서' not in df.columns:
        try:
            meta_df = pd.read_csv(meta_fp, encoding='cp949')
        except UnicodeDecodeError:
            meta_df = pd.read_csv(meta_fp, encoding='utf-8')
            
        meta_df.columns = meta_df.columns.str.strip()
        df = df.merge(meta_df[['지점', '경도관서', '위도관서']], on='지점', how='left')

    df['일시'] = pd.to_datetime(df['일시'], errors='coerce')
    df['일시'] = df['일시'].dt.normalize()
    df = df[df['일시'].isin(valid_dates)]
    
    daily_df = df.groupby(['지점', '일시']).agg({
        '강수량': 'sum',
        '경도관서': 'first',
        '위도관서': 'first'
    }).reset_index()

    os.makedirs(out_dir, exist_ok=True)
    stn_groups = daily_df.groupby('지점')
    
    for stn_id, stn_data in tqdm(stn_groups, desc=f"Saving {network_type} stations"):
        stn_lon = stn_data['경도관서'].iloc[0]
        stn_lat = stn_data['위도관서'].iloc[0]

        if pd.isna(stn_lon) or pd.isna(stn_lat):
            continue

        dist_sq = (eqd_lon - stn_lon)**2 + (eqd_lat - stn_lat)**2
        row_idx, col_idx = np.unravel_index(dist_sq.argmin(), dist_sq.shape)
        filename = f"{network_type}_{stn_lon:.4f}_{stn_lat:.4f}_{col_idx}_{row_idx}.csv"
        save_path = os.path.join(out_dir, filename)

        stn_data.rename(columns={'일시': 'Date', '강수량': 'Precipitation'}, inplace=True)
        stn_data[['Date', 'Precipitation']].to_csv(save_path, index=False)

KIHS_FP = os.path.join(das_FP, 'projects', 'KIHS')
asos_out_dir = os.path.join(KIHS_FP, 'Precipitation_raw', 'ASOS')
aws_out_dir = os.path.join(KIHS_FP, 'Precipitation_raw', 'AWS')

time_FP = os.path.join(KIHS_FP, 'KIHS_data_all.nc')
with xr.open_dataset(time_FP) as ds:
    valid_dates = pd.to_datetime(ds['time'].values).normalize()

asos_data_path = os.path.join(KIHS_FP, 'Precipitation', 'ASOS', 'Data_ASOS.csv')
asos_meta_path = os.path.join(KIHS_FP, 'Precipitation', 'ASOS', 'Station_ASOS.csv')
process_and_cut_stations(asos_data_path, asos_meta_path, 'ASOS', asos_out_dir, valid_dates)

aws_data_path = os.path.join(KIHS_FP, 'Precipitation', 'AWS', 'Data_AWS.csv')
aws_meta_path = os.path.join(KIHS_FP, 'Precipitation', 'AWS', 'Station_AWS.csv')
process_and_cut_stations(aws_data_path, aws_meta_path, 'AWS', aws_out_dir, valid_dates)