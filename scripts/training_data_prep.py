import os
import numpy as np
import xarray as xr
import pandas as pd

# ---------------------------------------------------------------------------
# Paths — point to your processed Mozambique files
# ---------------------------------------------------------------------------
ERA5_850_FILE  = '../data/processed/era5/era5_progvars_mozambique.nc'
ERA5_TP_FILE   = '../data/processed/era5/era5_tp_mozambique.nc'
CHIRPS_NC_FILE = '../data/processed/chirps/chirps_mozambique.nc'
OUTPUT_DIR     = '../data/processed/training_data'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
#SCALE_FACTOR = 5          # ERA5 0.25 deg --> CHIRPS 0.05 deg

ERA5_850_VARS = ["q", "t", "u", "v", "w", "z"]   # 850 hPa channels

# Temporal split boundaries
TRAIN_END = "2012-12"
VAL_END   = "2017-12"

# ERA5 0.50 deg → CHIRPS 0.05 deg
SCALE_FACTOR   = 10
ERA5_TARGET_RES = 0.50    # deg — target coarsened resolution

# ---------------------------------------------------------------------------
# Step 1 — Load processed ERA5 850 hPa
# ---------------------------------------------------------------------------
print("── Loading ERA5 850 hPa ─────────────────────────────")
ds_850 = xr.open_dataset(ERA5_850_FILE)

# Normalise coord names just in case
if "latitude"  in ds_850.dims: ds_850 = ds_850.rename({"latitude": "lat"})
if "longitude" in ds_850.dims: ds_850 = ds_850.rename({"longitude": "lon"})
if "valid_time" in ds_850.dims: ds_850 = ds_850.rename({"valid_time": "time"})

# Select only 850 hPa level 
if "level" in ds_850.dims:
    ds_850 = ds_850.sel(level=850, drop=True)
if "pressure_level" in ds_850.dims:
    ds_850 = ds_850.sel(pressure_level=850, drop=True)

# Ensure ascending lat order
if ds_850["lat"].values[0] > ds_850["lat"].values[-1]:
    ds_850 = ds_850.isel(lat=slice(None, None, -1))

ds_850["time"] = pd.to_datetime(ds_850["time"].values)

# Collect 850 hPa channels — try short name then any partial match
era5_850_arrays = []
found_vars      = []
for var in ERA5_850_VARS:
    if var in ds_850.data_vars:
        era5_850_arrays.append(ds_850[var].values.astype(np.float32))
        found_vars.append(var)
    else:
        # Try case-insensitive partial match
        match = next((v for v in ds_850.data_vars if var.lower() in v.lower()), None)
        if match:
            era5_850_arrays.append(ds_850[match].values.astype(np.float32))
            found_vars.append(match)
            print(f"  '{var}' matched to '{match}'")
        else:
            print(f" Variable '{var}' not found — skipping")

era5_times = pd.DatetimeIndex(ds_850["time"].values)
era5_lats  = ds_850["lat"].values
era5_lons  = ds_850["lon"].values

print(f"  Variables found : {found_vars}")
print(f"  Shape per var   : {era5_850_arrays[0].shape}  (T, lat, lon)")
print(f"  Time range      : {era5_times[0].date()} → {era5_times[-1].date()}")
print(f"  Lat range       : {era5_lats.min():.2f} → {era5_lats.max():.2f}")
print(f"  Lon range       : {era5_lons.min():.2f} → {era5_lons.max():.2f}")


# ---------------------------------------------------------------------------
# Step 2 — Load ERA5 Total Precipitation and regrid to 850 hPa grid
# ---------------------------------------------------------------------------
print("\n── Loading ERA5 Total Precipitation ─────────────────")
ds_tp = xr.open_dataset(ERA5_TP_FILE)

# Normalise coords
if "latitude"  in ds_tp.dims:  ds_tp = ds_tp.rename({"latitude": "lat"})
if "longitude" in ds_tp.dims:  ds_tp = ds_tp.rename({"longitude": "lon"})
if "valid_time" in ds_tp.dims: ds_tp = ds_tp.rename({"valid_time": "time"})

if ds_tp["lat"].values[0] > ds_tp["lat"].values[-1]:
    ds_tp = ds_tp.isel(lat=slice(None, None, -1))

ds_tp["time"] = pd.to_datetime(ds_tp["time"].values)

# Auto-detect the precipitation variable name
tp_var = next(
    (v for v in ds_tp.data_vars
     if any(k in v.lower() for k in ["tp", "precip", "precipitation"])),
    list(ds_tp.data_vars)[0]   # fallback: first variable
)
print(f"  TP variable     : '{tp_var}'")
print(f"  TP units        : {ds_tp[tp_var].attrs.get('units', 'not set')}")
print(f"  TP time range   : {pd.to_datetime(ds_tp['time'].values[0]).date()} → "
      f"{pd.to_datetime(ds_tp['time'].values[-1]).date()}")

# Regrid TP to the 850 hPa lat/lon grid (bilinear) so both inputs
# share exactly the same spatial coordinates before stacking
print("  Regridding TP → 850 hPa grid (bilinear) ...")
ds_tp_regrid = ds_tp[tp_var].interp(
    lat=era5_lats,
    lon=era5_lons,
    method="linear"
)

# Align TP time axis to ERA5 850 hPa time axis (inner join)
common_tp_times = pd.DatetimeIndex(ds_tp["time"].values).intersection(era5_times)
ds_tp_aligned   = ds_tp_regrid.sel(time=common_tp_times)
ds_850_aligned  = ds_850.sel(time=common_tp_times)

print(f"  Common timesteps after alignment : {len(common_tp_times)}")

# Re-extract 850 hPa channels on the aligned time axis
era5_850_arrays = []
for var in found_vars:
    era5_850_arrays.append(ds_850_aligned[var].values.astype(np.float32))

era5_times = common_tp_times

# Stack all ERA5 channels: 850 hPa variables + TP → (T, lat, lon, C)
tp_array    = ds_tp_aligned.values.astype(np.float32)   # (T, lat, lon)
all_channels = era5_850_arrays + [tp_array]
channel_names = found_vars + [tp_var]

era5_data = np.stack(all_channels, axis=-1)             # (T, H_lr, W_lr, C)

print(f"\n  ERA5 final shape  : {era5_data.shape}  (T, H_lr, W_lr, C={len(channel_names)})")
print(f"  Channels          : {channel_names}")


# ---------------------------------------------------------------------------
# Step 2b — Coarsen ERA5 from 0.25 deg → 0.50 deg
# ---------------------------------------------------------------------------
print("\n── Coarsening ERA5 0.25° → 0.50° ───────────────────")

# Build a coarsened lat/lon grid at the target resolution
# Snap to the target resolution so coordinates are clean multiples
coarse_lats = np.arange(
    np.round(era5_lats.min() / ERA5_TARGET_RES) * ERA5_TARGET_RES,
    np.round(era5_lats.max() / ERA5_TARGET_RES) * ERA5_TARGET_RES + ERA5_TARGET_RES / 2,
    ERA5_TARGET_RES
)
coarse_lons = np.arange(
    np.round(era5_lons.min() / ERA5_TARGET_RES) * ERA5_TARGET_RES,
    np.round(era5_lons.max() / ERA5_TARGET_RES) * ERA5_TARGET_RES + ERA5_TARGET_RES / 2,
    ERA5_TARGET_RES
)

print(f"  ERA5 before  : {era5_data.shape}  @ ~0.25°")

# Coarsen each channel using xarray (area-average — more physically correct
# than subsampling for atmospheric fields)
era5_coarse_list = []
for c in range(era5_data.shape[-1]):
    da_fine = xr.DataArray(
        era5_data[..., c],                          # (T, H, W)
        dims=["time", "lat", "lon"],
        coords={
            "time": era5_times,
            "lat" : era5_lats,
            "lon" : era5_lons,
        }
    )
    # Bilinear interpolation onto the coarser grid
    # (use method="linear" for smooth fields like T, q, z, u, v, w
    #  and method="linear" for TP — never nearest for coarsening)
    da_coarse = da_fine.interp(
        lat=coarse_lats,
        lon=coarse_lons,
        method="linear"
    )
    era5_coarse_list.append(da_coarse.values.astype(np.float32))

era5_data = np.stack(era5_coarse_list, axis=-1)   # (T, H_lr, W_lr, C)
era5_lats = coarse_lats
era5_lons = coarse_lons

print(f"  ERA5 after   : {era5_data.shape}  @ 0.50°")
print(f"  New scale factor vs CHIRPS : {SCALE_FACTOR}×  "
      f"({ERA5_TARGET_RES}° → {abs(float(chirps_lat_target[1]-chirps_lat_target[0])):.2f}°)")

# Sanity check: coarsened values should be in the same physical range
for c, name in enumerate(channel_names):
    fine_mean   = np.nanmean(era5_coarse_list[c])
    print(f"  [{c}] {name:30s}  mean={fine_mean:.4e}")


# ---------------------------------------------------------------------------
# Step 3 — Load CHIRPS from .nc (already clipped and stacked)
# ---------------------------------------------------------------------------
print("\n── Loading CHIRPS (processed .nc) ───────────────────")
ds_chirps = xr.open_dataset(CHIRPS_NC_FILE)

# Ensure ascending lat
if ds_chirps["lat"].values[0] > ds_chirps["lat"].values[-1]:
    ds_chirps = ds_chirps.isel(lat=slice(None, None, -1))

chirps_times = pd.DatetimeIndex(ds_chirps["time"].values)
chirps_lats  = ds_chirps["lat"].values
chirps_lons  = ds_chirps["lon"].values
chirps_data  = ds_chirps["precip"].values.astype(np.float32)  # (T, H_hr, W_hr)

print(f"  CHIRPS shape    : {chirps_data.shape}")
print(f"  CHIRPS time     : {chirps_times[0].date()} → {chirps_times[-1].date()}")
print(f"  CHIRPS lat      : {chirps_lats.min():.2f} → {chirps_lats.max():.2f}")
print(f"  CHIRPS lon      : {chirps_lons.min():.2f} → {chirps_lons.max():.2f}")
print(f"  Resolution      : ~{abs(chirps_lats[1]-chirps_lats[0]):.4f}°")

# ---------------------------------------------------------------------------
# Step 4 — Align spatial grids
#   ERA5 @ 0.25 deg  →  CHIRPS @ 0.05 deg  (SCALE_FACTOR = 5)
#   Snap a common bounding box to ERA5 grid lines, then verify ratio holds
# ---------------------------------------------------------------------------
print("\n── Aligning spatial grids ───────────────────────────")

res_era5   = 0.25
res_chirps = abs(float(chirps_lats[1] - chirps_lats[0]))   # ~0.05

# Overlapping bounding box, snapped to ERA5 grid
lat_min = np.round(max(era5_lats.min(), chirps_lats.min()) / res_era5) * res_era5
lat_max = np.round(min(era5_lats.max(), chirps_lats.max()) / res_era5) * res_era5
lon_min = np.round(max(era5_lons.min(), chirps_lons.min()) / res_era5) * res_era5
lon_max = np.round(min(era5_lons.max(), chirps_lons.max()) / res_era5) * res_era5

print(f"  Snapped bbox  : lat [{lat_min}, {lat_max}]  lon [{lon_min}, {lon_max}]")

# ERA5 indices within snapped box
era5_lat_idx = np.where((era5_lats >= lat_min - 1e-6) & (era5_lats <= lat_max + 1e-6))[0]
era5_lon_idx = np.where((era5_lons >= lon_min - 1e-6) & (era5_lons <= lon_max + 1e-6))[0]

era5_lat_crop = era5_lats[era5_lat_idx]
era5_lon_crop = era5_lons[era5_lon_idx]
era5_data     = era5_data[:, era5_lat_idx, :, :][:, :, era5_lon_idx, :]

# CHIRPS target coords (5× finer, built from ERA5 snapped grid)
chirps_lat_target = np.arange(lat_max, lat_min - res_chirps / 2, -res_chirps)
chirps_lon_target = np.arange(lon_min, lon_max + res_chirps / 2,  res_chirps)

# Verify scale factor holds exactly
assert len(chirps_lat_target) == len(era5_lat_crop) * SCALE_FACTOR, (
    f"Lat mismatch: {len(chirps_lat_target)} ≠ {len(era5_lat_crop)} × {SCALE_FACTOR}")
assert len(chirps_lon_target) == len(era5_lon_crop) * SCALE_FACTOR, (
    f"Lon mismatch: {len(chirps_lon_target)} ≠ {len(era5_lon_crop)} × {SCALE_FACTOR}")

print(f"  ERA5  grid    : {len(era5_lat_crop)} × {len(era5_lon_crop)}")
print(f"  CHIRPS grid   : {len(chirps_lat_target)} × {len(chirps_lon_target)}")
print(f"  Scale factor  : {SCALE_FACTOR}×  ✓")

# Regrid CHIRPS to snapped target grid (nearest neighbour preserves rainfall values)
print("  Resampling CHIRPS to snapped grid ...")
chirps_da = xr.DataArray(
    chirps_data,
    dims=["time", "lat", "lon"],
    coords={"time": chirps_times, "lat": chirps_lats, "lon": chirps_lons}
)
chirps_da = chirps_da.interp(
    lat=chirps_lat_target,
    lon=chirps_lon_target,
    method="nearest"
)
chirps_data = chirps_da.values   # (T, H_hr, W_hr)

print(f"  ERA5  final   : {era5_data.shape}")
print(f"  CHIRPS final  : {chirps_data.shape}")

# ---------------------------------------------------------------------------
# Step 5 — Align time axes
# ---------------------------------------------------------------------------
print("\n── Aligning time axes ───────────────────────────────")

common_times = chirps_times.intersection(era5_times)
print(f"  Common months : {len(common_times)}  "
      f"({common_times[0].date()} → {common_times[-1].date()})")

era5_idx   = np.array([np.where(era5_times   == t)[0][0] for t in common_times])
chirps_idx = np.array([np.where(chirps_times == t)[0][0] for t in common_times])

era5_data   = era5_data[era5_idx]     # (N, H_lr, W_lr, C)
chirps_data = chirps_data[chirps_idx] # (N, H_hr, W_hr)
times       = common_times

# Add channel dim to CHIRPS target: (N, H_hr, W_hr, 1)
chirps_data = chirps_data[..., np.newaxis]

# ---------------------------------------------------------------------------
# Step 6 — Land mask (from CHIRPS — valid in > 80% of months)
# ---------------------------------------------------------------------------
land_mask_hr = (np.sum(~np.isnan(chirps_data[..., 0]), axis=0) / len(times)) > 0.8
land_mask_lr = (np.sum(~np.isnan(era5_data[..., 0]),   axis=0) / len(times)) > 0.8

print(f"\n  Land pixels (CHIRPS) : {land_mask_hr.sum():,} / {land_mask_hr.size:,}  "
      f"({100*land_mask_hr.mean():.1f}%)")
print(f"  Land pixels (ERA5)   : {land_mask_lr.sum():,} / {land_mask_lr.size:,}  "
      f"({100*land_mask_lr.mean():.1f}%)")

# ---------------------------------------------------------------------------
# Step 7 — Temporal split
# ---------------------------------------------------------------------------
print("\n── Splitting train / val / test ─────────────────────")

train_mask = times <= TRAIN_END
val_mask   = (times > TRAIN_END) & (times <= VAL_END)
test_mask  = times > VAL_END

X_train, y_train = era5_data[train_mask], chirps_data[train_mask]
X_val,   y_val   = era5_data[val_mask],   chirps_data[val_mask]
X_test,  y_test  = era5_data[test_mask],  chirps_data[test_mask]

print(f"  Train : {X_train.shape[0]} months  ({times[train_mask][0].date()} → {times[train_mask][-1].date()})")
print(f"  Val   : {X_val.shape[0]}   months  ({times[val_mask][0].date()}   → {times[val_mask][-1].date()})")
print(f"  Test  : {X_test.shape[0]}  months  ({times[test_mask][0].date()}  → {times[test_mask][-1].date()})")

# ---------------------------------------------------------------------------
# Step 8 — Normalise inputs (train stats only)
#   ERA5 channels     → z-score  (unbounded variables)
#   CHIRPS target     → log1p    (heavy-tailed rainfall distribution)
# ---------------------------------------------------------------------------
print("\n── Normalising ──────────────────────────────────────")
n_channels = X_train.shape[-1]

norm_mean = np.zeros(n_channels, dtype=np.float32)
norm_std  = np.zeros(n_channels, dtype=np.float32)

for c in range(n_channels):
    vals = X_train[..., c].ravel()
    vals = vals[np.isfinite(vals)]
    norm_mean[c] = vals.mean()
    norm_std[c]  = vals.std()
    print(f"  [{c}] {channel_names[c]:30s}  mean={norm_mean[c]:.4e}  std={norm_std[c]:.4e}")

# Replace NaN with channel mean before normalising (ocean pixels)
for c in range(n_channels):
    for arr in [X_train, X_val, X_test]:
        nan_mask = np.isnan(arr[..., c])
        arr[..., c][nan_mask] = norm_mean[c]

X_train = (X_train - norm_mean) / (norm_std + 1e-8)
X_val   = (X_val   - norm_mean) / (norm_std + 1e-8)
X_test  = (X_test  - norm_mean) / (norm_std + 1e-8)

# Log1p transform on CHIRPS target (then normalise)
print("\n  CHIRPS target → log1p transform")
y_train_log = np.log1p(np.nan_to_num(y_train, nan=0.0))
y_val_log   = np.log1p(np.nan_to_num(y_val,   nan=0.0))
y_test_log  = np.log1p(np.nan_to_num(y_test,  nan=0.0))

chirps_log_mean = y_train_log[land_mask_hr[np.newaxis,...,np.newaxis].repeat(
    y_train_log.shape[0], axis=0)].mean()
chirps_log_std  = y_train_log[land_mask_hr[np.newaxis,...,np.newaxis].repeat(
    y_train_log.shape[0], axis=0)].std()

y_train_norm = (y_train_log - chirps_log_mean) / (chirps_log_std + 1e-8)
y_val_norm   = (y_val_log   - chirps_log_mean) / (chirps_log_std + 1e-8)
y_test_norm  = (y_test_log  - chirps_log_mean) / (chirps_log_std + 1e-8)

print(f"  CHIRPS log-mean : {chirps_log_mean:.4f}")
print(f"  CHIRPS log-std  : {chirps_log_std:.4f}")

# ---------------------------------------------------------------------------
# Step 9 — Save
# ---------------------------------------------------------------------------
print(f"\n── Saving to {OUTPUT_DIR} ───────────────────────────")

# Remove old files to avoid Windows lock issues
for fname in ["train.npz", "val.npz", "test.npz", "metadata.npz"]:
    fpath = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(fpath):
        os.remove(fpath)

np.savez_compressed(
    os.path.join(OUTPUT_DIR, "train.npz"),
    X=X_train, y=y_train_norm,
    y_raw=y_train,                          # keep raw values for evaluation
    times=times[train_mask].astype(str)
)
np.savez_compressed(
    os.path.join(OUTPUT_DIR, "val.npz"),
    X=X_val, y=y_val_norm,
    y_raw=y_val,
    times=times[val_mask].astype(str)
)
np.savez_compressed(
    os.path.join(OUTPUT_DIR, "test.npz"),
    X=X_test, y=y_test_norm,
    y_raw=y_test,
    times=times[test_mask].astype(str)
)
np.savez_compressed(
    os.path.join(OUTPUT_DIR, "metadata.npz"),
    # Grids
    era5_lats=era5_lat_crop,
    era5_lons=era5_lon_crop,
    chirps_lats=chirps_lat_target,
    chirps_lons=chirps_lon_target,
    # Masks
    land_mask_hr=land_mask_hr,
    land_mask_lr=land_mask_lr,
    # ERA5 normalisation (z-score)
    norm_mean=norm_mean,
    norm_std=norm_std,
    channel_names=np.array(channel_names),
    # CHIRPS normalisation (log1p + z-score)
    chirps_log_mean=np.float32(chirps_log_mean),
    chirps_log_std=np.float32(chirps_log_std),
    # Config
    scale_factor=np.int32(SCALE_FACTOR),
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n── Summary ──────────────────────────────────────────")
print(f"  Input channels    : {len(channel_names)}  →  {channel_names}")
print(f"  X shape (train)   : {X_train.shape}  (N, H_lr, W_lr, C)")
print(f"  y shape (train)   : {y_train_norm.shape}  (N, H_hr, W_hr, 1)")
print(f"  Scale factor      : {SCALE_FACTOR}×  ({res_era5}° → {res_chirps}°)")
print()
for fname in ["train.npz","val.npz","test.npz","metadata.npz"]:
    p = os.path.join(OUTPUT_DIR, fname)
    print(f"  {fname:20s}  {os.path.getsize(p)/1e6:.1f} MB")
print("\n✓ Training data ready for downscaling model.")