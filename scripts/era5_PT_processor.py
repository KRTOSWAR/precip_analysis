import os
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.io.shapereader as shpreader
from shapely.vectorized import contains

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_NC  = '../data/raw/Era5/ERA5_TP.nc'
OUTPUT_NC = '../data/processed/era5/era5_tp_mozambique.nc'


os.makedirs(os.path.dirname(OUTPUT_NC), exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1 — Inspect
# ---------------------------------------------------------------------------
print("── Inspecting ERA5 Total Precipitation file ─────────")
ds_raw = xr.open_dataset(INPUT_NC)
print(ds_raw)
print("\n  Variables   :", list(ds_raw.data_vars))
print("  Dimensions  :", dict(ds_raw.dims))

# ---------------------------------------------------------------------------
# Step 2 — Normalise coordinates
# ---------------------------------------------------------------------------
rename_map = {}
if 'latitude'  in ds_raw.dims: rename_map['latitude']  = 'lat'
if 'longitude' in ds_raw.dims: rename_map['longitude'] = 'lon'
if rename_map:
    ds_raw = ds_raw.rename(rename_map)
    print(f"\n  Renamed: {rename_map}")

if ds_raw['lat'].values[0] > ds_raw['lat'].values[-1]:
    ds_raw = ds_raw.isel(lat=slice(None, None, -1))
    print("  Flipped lat to ascending order")

# ---------------------------------------------------------------------------
# Step 3 — Unit conversion: ERA5 TP is in metres/month → mm/month
# ---------------------------------------------------------------------------
TP_VAR = [v for v in ds_raw.data_vars if 'tp' in v.lower() 
          or 'precip' in v.lower() 
          or 'precipitation' in v.lower()][0]

print(f"\n  Precipitation variable found : '{TP_VAR}'")
print(f"  Current units : {ds_raw[TP_VAR].attrs.get('units', 'not set')}")

# ERA5 stores TP in metres — convert to mm
ds_raw[TP_VAR] = ds_raw[TP_VAR] * 1000
ds_raw[TP_VAR].attrs['units']     = 'mm/month'
ds_raw[TP_VAR].attrs['long_name'] = 'Total precipitation'
print("  Converted: m/month → mm/month  (×1000)")

# ---------------------------------------------------------------------------
# Step 4 — Bounding box clip
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = -27.0, -10.0
LON_MIN, LON_MAX =  30.0,  41.0

ds_bbox = ds_raw.sel(
    lat=slice(LAT_MIN, LAT_MAX),
    lon=slice(LON_MIN, LON_MAX)
)

print(f"\n  After bbox clip → shape: {dict(ds_bbox.dims)}")

# ---------------------------------------------------------------------------
# Step 5 — Mozambique polygon mask
# ---------------------------------------------------------------------------
shpfile  = shpreader.natural_earth(resolution='10m',
                                   category='cultural',
                                   name='admin_0_countries')
reader   = shpreader.Reader(shpfile)
moz_geom = next(
    r.geometry for r in reader.records()
    if r.attributes['ADMIN'] == 'Mozambique'
)

lons_2d, lats_2d = np.meshgrid(ds_bbox.lon.values, ds_bbox.lat.values)
inside = contains(moz_geom, lons_2d.ravel(), lats_2d.ravel()).reshape(lons_2d.shape)
mask   = xr.DataArray(inside, dims=['lat', 'lon'],
                      coords={'lat': ds_bbox.lat, 'lon': ds_bbox.lon})

ds_moz = ds_bbox.where(mask)

print(f"  Valid pixel fraction : {float(mask.mean().values):.1%}")

# ---------------------------------------------------------------------------
# Step 6 — Quick sanity check on rainfall values
# ---------------------------------------------------------------------------
vals = ds_moz[TP_VAR].values
print(f"\n── Precipitation sanity check ───────────────────────")
print(f"  Min : {np.nanmin(vals):.2f} mm")
print(f"  Max : {np.nanmax(vals):.2f} mm")
print(f"  Mean: {np.nanmean(vals):.2f} mm/month")

if np.nanmax(vals) > 2000:
    print("  ⚠ Very high values — check if conversion already applied")
elif np.nanmax(vals) < 1:
    print("  ⚠ Very low values — may still be in metres, check units")
else:
    print("  ✓ Values look reasonable for monthly precipitation")

# ---------------------------------------------------------------------------
# Step 7 — Metadata and save
# ---------------------------------------------------------------------------
ds_moz.attrs.update({
    'title'      : 'ERA5 Total Precipitation Monthly — Mozambique',
    'source'     : 'ECMWF ERA5 reanalysis',
    'variable'   : 'Total precipitation',
    'region'     : 'Mozambique',
    'history'    : f'Clipped and converted m→mm on {pd.Timestamp.today().date()}',
    'Conventions': 'CF-1.8',
})

ds_moz['lat'].attrs = {'long_name': 'latitude',  'units': 'degrees_north', 'axis': 'Y'}
ds_moz['lon'].attrs = {'long_name': 'longitude', 'units': 'degrees_east',  'axis': 'X'}

encoding = {
    TP_VAR: {
        'dtype'     : 'float32',
        'zlib'      : True,
        'complevel' : 4,
        '_FillValue': np.float32(np.nan),
    }
}

if os.path.exists(OUTPUT_NC):
    os.remove(OUTPUT_NC)

print(f"\nWriting → {OUTPUT_NC} ...")
ds_moz.to_netcdf(OUTPUT_NC, encoding=encoding, mode='w')

with xr.open_dataset(OUTPUT_NC) as ds_check:
    print("\n── Verification ─────────────────────────────────────")
    print(ds_check)
    print(f"  File size : {os.path.getsize(OUTPUT_NC)/1e6:.1f} MB")
    if 'time' in ds_check.dims:
        print(f"  Time range: {str(ds_check.time.values[0])[:10]} → "
              f"{str(ds_check.time.values[-1])[:10]}")

print("\n✓ ERA5 Total Precipitation — Mozambique clip complete.")