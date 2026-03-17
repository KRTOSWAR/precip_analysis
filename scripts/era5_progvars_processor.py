import os
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.io.shapereader as shpreader
from shapely.vectorized import contains

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_NC  = '../data/raw/Era5/era5_pressure_850_monthly_africa.nc'
OUTPUT_NC = '../data/processed/era5/era5_progvars_mozambique.nc'

os.makedirs(os.path.dirname(OUTPUT_NC), exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1 — Inspect
# ---------------------------------------------------------------------------
print("── Inspecting ERA5 pressure-level file ──────────────")
ds_raw = xr.open_dataset(INPUT_NC)
print(ds_raw)
print("\n  Variables      :", list(ds_raw.data_vars))
print("  Dimensions     :", dict(ds_raw.dims))
print("  Coordinates    :", list(ds_raw.coords))
if 'pressure_level' in ds_raw.dims:
    print("  Pressure levels:", ds_raw['pressure_level'].values, "hPa")

# ---------------------------------------------------------------------------
# Step 2 — Normalise coordinate names
#   ERA5 uses 'valid_time' instead of 'time', and 'latitude'/'longitude'
#   We standardise everything for downstream consistency
# ---------------------------------------------------------------------------
rename_map = {}
if 'latitude'   in ds_raw.dims: rename_map['latitude']   = 'lat'
if 'longitude'  in ds_raw.dims: rename_map['longitude']  = 'lon'
if 'valid_time' in ds_raw.dims: rename_map['valid_time'] = 'time'

if rename_map:
    ds_raw = ds_raw.rename(rename_map)
    print(f"\n  Renamed coordinates: {rename_map}")

# Flip latitude to ascending order if needed
if ds_raw['lat'].values[0] > ds_raw['lat'].values[-1]:
    ds_raw = ds_raw.isel(lat=slice(None, None, -1))
    print("  Flipped lat to ascending order")

print(f"\n  Time range     : {str(ds_raw.time.values[0])[:10]} → "
      f"{str(ds_raw.time.values[-1])[:10]}")
print(f"  Time steps     : {ds_raw.dims['time']}")

# ---------------------------------------------------------------------------
# Step 3 — Bounding box clip (fast coarse crop before masking)
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = -27.0, -10.0
LON_MIN, LON_MAX =  30.0,  41.0

ds_bbox = ds_raw.sel(
    lat=slice(LAT_MIN, LAT_MAX),
    lon=slice(LON_MIN, LON_MAX)
)

print(f"\n── After bounding box clip ───────────────────────────")
print(f"  lat : {ds_bbox.lat.values[0]:.2f}° → {ds_bbox.lat.values[-1]:.2f}°  "
      f"({ds_bbox.dims['lat']} pixels)")
print(f"  lon : {ds_bbox.lon.values[0]:.2f}° → {ds_bbox.lon.values[-1]:.2f}°  "
      f"({ds_bbox.dims['lon']} pixels)")
print(f"  Full shape: {dict(ds_bbox.dims)}")

# ---------------------------------------------------------------------------
# Step 4 — Mozambique polygon mask
#   The mask is 2-D (lat, lon) — xarray broadcasts it automatically
#   across the time and pressure_level dimensions via .where()
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
inside = contains(
    moz_geom,
    lons_2d.ravel(),
    lats_2d.ravel()
).reshape(lons_2d.shape)

# 2-D mask — broadcasts over (time, pressure_level) automatically
mask = xr.DataArray(
    inside,
    dims=['lat', 'lon'],
    coords={'lat': ds_bbox.lat, 'lon': ds_bbox.lon}
)

ds_moz = ds_bbox.where(mask)

print(f"\n  Valid pixel fraction : {float(mask.mean().values):.1%} of bounding box")

# ---------------------------------------------------------------------------
# Step 5 — Per-variable sanity check
# ---------------------------------------------------------------------------
VAR_META = {
    'q': {'long_name': 'Specific humidity',         'units': 'kg kg-1'},
    't': {'long_name': 'Temperature',               'units': 'K'},
    'u': {'long_name': 'U-component of wind',       'units': 'm s-1'},
    'v': {'long_name': 'V-component of wind',       'units': 'm s-1'},
    'w': {'long_name': 'Vertical velocity',         'units': 'Pa s-1'},
    'z': {'long_name': 'Geopotential',              'units': 'm2 s-2'},
}

print("\n── Per-variable statistics (area mean over Mozambique) ──")
print(f"  {'Var':<5} {'Long name':<30} {'Units':<12} {'Min':>10} {'Max':>10} {'Mean':>10}")
print(f"  {'─'*5} {'─'*30} {'─'*12} {'─'*10} {'─'*10} {'─'*10}")

for var in ds_moz.data_vars:
    vals = ds_moz[var].values
    vmin  = np.nanmin(vals)
    vmax  = np.nanmax(vals)
    vmean = np.nanmean(vals)
    meta  = VAR_META.get(var, {})
    print(f"  {var:<5} {meta.get('long_name',''):<30} "
          f"{meta.get('units',''):<12} "
          f"{vmin:>10.3f} {vmax:>10.3f} {vmean:>10.3f}")

    # Attach clean metadata to each variable
    ds_moz[var].attrs['long_name'] = meta.get('long_name', ds_moz[var].attrs.get('long_name', ''))
    ds_moz[var].attrs['units']     = meta.get('units',     ds_moz[var].attrs.get('units', ''))

# ---------------------------------------------------------------------------
# Step 6 — Coordinate metadata
# ---------------------------------------------------------------------------
ds_moz['lat'].attrs  = {'long_name': 'latitude',       'units': 'degrees_north', 'axis': 'Y'}
ds_moz['lon'].attrs  = {'long_name': 'longitude',      'units': 'degrees_east',  'axis': 'X'}
ds_moz['time'].attrs = {'long_name': 'time',           'axis': 'T'}

if 'pressure_level' in ds_moz.dims:
    ds_moz['pressure_level'].attrs = {
        'long_name'     : 'pressure level',
        'units'         : 'hPa',
        'axis'          : 'Z',
        'positive'      : 'down',
    }

# ---------------------------------------------------------------------------
# Step 7 — Global metadata
# ---------------------------------------------------------------------------
ds_moz.attrs.update({
    'title'          : 'ERA5 Pressure-Level Monthly — Mozambique',
    'source'         : 'ECMWF ERA5 reanalysis',
    'variables'      : 'q, t, u, v, w, z',
    'pressure_levels': str(ds_moz['pressure_level'].values.tolist())
                       if 'pressure_level' in ds_moz.dims else 'N/A',
    'region'         : 'Mozambique',
    'history'        : f'Clipped on {pd.Timestamp.today().date()}',
    'Conventions'    : 'CF-1.8',
})

# ---------------------------------------------------------------------------
# Step 8 — Save with per-variable compression
# ---------------------------------------------------------------------------
encoding = {
    var: {
        'dtype'     : 'float32',
        'zlib'      : True,
        'complevel' : 4,
        '_FillValue': np.float32(np.nan),
    }
    for var in ds_moz.data_vars
}

if os.path.exists(OUTPUT_NC):
    os.remove(OUTPUT_NC)

print(f"\nWriting → {OUTPUT_NC} ...")
ds_moz.to_netcdf(OUTPUT_NC, encoding=encoding, mode='w')

# ---------------------------------------------------------------------------
# Step 9 — Verify
# ---------------------------------------------------------------------------
with xr.open_dataset(OUTPUT_NC) as ds_check:
    print("\n── Verification ─────────────────────────────────────")
    print(ds_check)
    print(f"\n  File size  : {os.path.getsize(OUTPUT_NC)/1e6:.1f} MB")
    print(f"  Variables  : {list(ds_check.data_vars)}")
    print(f"  Time range : {str(ds_check.time.values[0])[:10]} → "
          f"{str(ds_check.time.values[-1])[:10]}")
    if 'pressure_level' in ds_check.dims:
        print(f"  Levels     : {ds_check.pressure_level.values} hPa")
    print(f"  Shape      : (time={ds_check.dims['time']}, "
          f"pressure_level={ds_check.dims.get('pressure_level','N/A')}, "
          f"lat={ds_check.dims['lat']}, lon={ds_check.dims['lon']})")

print("\n✓ ERA5 pressure-level data — Mozambique clip complete.")