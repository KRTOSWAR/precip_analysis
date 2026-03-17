import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROCESSED_DIR = '../data/processed/chirps'
OUTPUT_NC     = '../data/processed/chirps/chirps_mozambique.nc'
CHIRPS_NODATA = -9999

# ---------------------------------------------------------------------------
# Load all clipped GeoTIFFs and build a labelled xarray DataArray
# ---------------------------------------------------------------------------
print("Loading clipped CHIRPS files ...")

chirps_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*_mozambique.tif")))
if not chirps_files:
    raise FileNotFoundError(f"No clipped files found in {PROCESSED_DIR}")

da_list      = []
chirps_times = []

for fpath in chirps_files:
    # Parse date from filename
    stem  = Path(fpath).stem                        # chirps-v3.0.1981.01_mozambique
    parts = stem.replace("_mozambique", "").split(".")
    year  = int(parts[-2])
    month = int(parts[-1])
    date  = pd.Timestamp(year=year, month=month, day=1)

    da = rioxarray.open_rasterio(fpath, masked=True).squeeze("band", drop=True)
    arr = da.values.astype(np.float32)
    arr[arr <= CHIRPS_NODATA] = np.nan

    # Wrap back into a DataArray with named spatial coords
    da_clean = xr.DataArray(
        arr,
        dims=["lat", "lon"],
        coords={
            "lat": da.y.values,
            "lon": da.x.values,
        }
    )

    da_list.append(da_clean)
    chirps_times.append(date)

print(f"  Loaded {len(da_list)} files")

# ---------------------------------------------------------------------------
# Stack along time → single DataArray (time, lat, lon)
# ---------------------------------------------------------------------------
chirps_times = pd.DatetimeIndex(chirps_times)

da_stacked = xr.concat(da_list, dim="time")
da_stacked = da_stacked.assign_coords(time=chirps_times)
da_stacked.name = "precip"

# ---------------------------------------------------------------------------
# Add metadata — makes the .nc self-describing and package-friendly
# ---------------------------------------------------------------------------
da_stacked.attrs = {
    "long_name"    : "Monthly precipitation",
    "standard_name": "precipitation_amount",
    "units"        : "mm/month",
    "source"       : "CHIRPS v3.0",
    "region"       : "Mozambique",
    "clipped_by"   : "Natural Earth admin_0_countries boundary",
}

da_stacked["lat"].attrs = {
    "long_name"     : "latitude",
    "standard_name" : "latitude",
    "units"         : "degrees_north",
    "axis"          : "Y",
}

da_stacked["lon"].attrs = {
    "long_name"     : "longitude",
    "standard_name" : "longitude",
    "units"         : "degrees_east",
    "axis"          : "X",
}

da_stacked["time"].attrs = {
    "long_name" : "time",
    "axis"      : "T",
}

# Wrap in a Dataset (standard NetCDF structure)
ds = da_stacked.to_dataset(name="precip")

ds.attrs = {
    "title"       : "CHIRPS v3.0 Monthly Precipitation — Mozambique",
    "institution" : "Climate Hazards Center, UC Santa Barbara",
    "history"     : f"Clipped and converted on {pd.Timestamp.today().date()}",
    "Conventions" : "CF-1.8",
}

# ---------------------------------------------------------------------------
# Save to NetCDF with compression
# ---------------------------------------------------------------------------
encoding = {
    "precip": {
        "dtype"      : "float32",
        "zlib"       : True,        # enable compression
        "complevel"  : 4,           # 1 (fast) → 9 (small); 4 is a good balance
        "chunksizes" : (12, da_stacked.sizes["lat"], da_stacked.sizes["lon"]),
        "_FillValue" : np.float32(np.nan),
    }
}

print(f"\nWriting NetCDF → {OUTPUT_NC} ...")
ds.to_netcdf(OUTPUT_NC, encoding=encoding, mode="w")

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
ds_check = xr.open_dataset(OUTPUT_NC)
print("\n── Verification ─────────────────────────────────────")
print(ds_check)
print(f"\n  File size : {os.path.getsize(OUTPUT_NC) / 1e6:.1f} MB")
print(f"  Time range: {str(ds_check.time.values[0])[:10]} → "
      f"{str(ds_check.time.values[-1])[:10]}")
print(f"  Shape     : {ds_check['precip'].shape}  (time, lat, lon)")
print("\n✓ NetCDF written successfully. Ready for precipitation indices.")
ds_check.close()