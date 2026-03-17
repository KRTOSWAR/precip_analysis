import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

import rioxarray
import cartopy.io.shapereader as shpreader
from shapely.geometry import mapping

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHIRPS_DIR    = '../data/raw/chirps'
OUTPUT_DIR    = '../data/processed/chirps'
CHIRPS_NODATA = -9999

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1 — Get Mozambique boundary geometry via Cartopy
# ---------------------------------------------------------------------------
print("Loading Mozambique boundary from Cartopy Natural Earth ...")

# Download / cache the 1:10m admin-0 shapefile (done once, then cached locally)
shpfilename = shpreader.natural_earth(
    resolution="10m",       # highest res available: 10m | 50m | 110m
    category="cultural",
    name="admin_0_countries"
)

reader    = shpreader.Reader(shpfilename)
countries = list(reader.records())

# Filter to Mozambique — field name is 'NAME_LONG' or 'ADMIN'
mozambique = next(
    rec for rec in countries
    if rec.attributes["ADMIN"] == "Mozambique"
)

moz_geom      = mozambique.geometry      # Shapely (Multi)Polygon
moz_geom_list = [mapping(moz_geom)]      # rioxarray expects a list of GeoJSON-like dicts

print(f"  Mozambique geometry type : {moz_geom.geom_type}")

# ---------------------------------------------------------------------------
# Step 2 — Load, clip, save, and stack CHIRPS files
# ---------------------------------------------------------------------------
print("\nLoading and clipping CHIRPS files to Mozambique ...")

chirps_files = sorted(glob.glob(os.path.join(CHIRPS_DIR, "chirps-v3.0.*.*.tif")))
if not chirps_files:
    raise FileNotFoundError(
        f"No CHIRPS .tif files found in {CHIRPS_DIR}. "
        "Check that the download completed successfully."
    )

print(f"  Found {len(chirps_files)} CHIRPS files\n")

chirps_list  = []
chirps_times = []

for fpath in chirps_files:
    # --- Parse date from filename  e.g. chirps-v3.0.1981.01.tif ---
    stem  = Path(fpath).stem        # chirps-v3.0.1981.01
    parts = stem.split(".")         # ['chirps-v3', '0', '1981', '01']
    year  = int(parts[-2])
    month = int(parts[-1])
    date  = pd.Timestamp(year=year, month=month, day=1)

    # --- Open raster ---
    da = rioxarray.open_rasterio(fpath, masked=True).squeeze("band", drop=True)
    da = da.rename({"y": "lat", "x": "lon"})

    # --- Ensure CRS is set (CHIRPS is always WGS-84) ---
    if da.rio.crs is None:
        da = da.rio.write_crs("EPSG:4326")

    # --- Explicitly tell rioxarray which dims are spatial ---
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

    # --- Clip to Mozambique boundary ---
    # drop=True     → crops bounding box to Mozambique extent (saves memory)
    # all_touched=True → include pixels whose edges touch the boundary
    da_moz = da.rio.clip(
        moz_geom_list,
        crs="EPSG:4326",
        drop=True,
        invert=False,
        all_touched=True
    )

    # --- Save clipped GeoTIFF to OUTPUT_DIR ---
    out_name = Path(fpath).name.replace(".tif", "_mozambique.tif")
    out_path = os.path.join(OUTPUT_DIR, out_name)
    da_moz.rio.to_raster(out_path)

    # --- Collect array for downstream stacking ---
    arr = da_moz.values.astype(np.float32)
    arr[arr <= CHIRPS_NODATA] = np.nan      # replace nodata with NaN

    chirps_list.append(arr)
    chirps_times.append(date)

    print(f"  ✓ {Path(fpath).name}  →  shape {arr.shape}  |  saved: {out_name}")

# ---------------------------------------------------------------------------
# Step 3 — Build stacked array  (T, H, W)
# ---------------------------------------------------------------------------
chirps_times = pd.DatetimeIndex(chirps_times)
chirps_data  = np.stack(chirps_list, axis=0)    # (T, H_moz, W_moz)

print(f"\n{'='*60}")
print(f"  Files saved to    : {OUTPUT_DIR}")
print(f"  Stacked shape     : {chirps_data.shape}  (T, H, W)")
print(f"  Time range        : {chirps_times[0].date()}  →  {chirps_times[-1].date()}")
print(f"  NaN fraction      : {np.isnan(chirps_data).mean():.1%}")
print(f"  Min rainfall      : {np.nanmin(chirps_data):.2f} mm")
print(f"  Max rainfall      : {np.nanmax(chirps_data):.2f} mm")
print(f"  Mean rainfall     : {np.nanmean(chirps_data):.2f} mm")
print(f"{'='*60}")