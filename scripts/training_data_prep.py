import os
import numpy as np
import xarray as xr
import pandas as pd

# ---------------------------------------------------------------------------
# Paths — point to your processed Mozambique files
# ---------------------------------------------------------------------------
ERA5_850_FILE  = '../data/processed/era5/era5_progvars_mozambique.nc'
ERA5_TP_FILE   = '../data/processed/era5_tp_mozambique.nc'
CHIRPS_NC_FILE = '../data/processed/chirps/chirps_mozambique.nc'
OUTPUT_DIR     = '../data/processed/training_data'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ERA5_850_VARS  = ["q", "t", "u", "v", "w", "z"]   # 850 hPa channels
TRAIN_END      = "2012-12"
VAL_END        = "2017-12"
SCALE_FACTOR   = 10        # ERA5 0.50 deg → CHIRPS 0.05 deg
ERA5_TARGET_RES = 0.50     # deg — target coarsened resolution

# ---------------------------------------------------------------------------
# Step 1 — Load processed ERA5 850 hPa
# ---------------------------------------------------------------------------
print("── Loading ERA5 850 hPa ─────────────────────────────")
ds_850 = xr.open_dataset(ERA5_850_FILE)

if "latitude"   in ds_850.dims: ds_850 = ds_850.rename({"latitude":  "lat"})
if "longitude"  in ds_850.dims: ds_850 = ds_850.rename({"longitude": "lon"})
if "valid_time" in ds_850.dims: ds_850 = ds_850.rename({"valid_time": "time"})

if "level"          in ds_850.dims: ds_850 = ds_850.sel(level=850,          drop=True)
if "pressure_level" in ds_850.dims: ds_850 = ds_850.sel(pressure_level=850, drop=True)

if ds_850["lat"].values[0] > ds_850["lat"].values[-1]:
    ds_850 = ds_850.isel(lat=slice(None, None, -1))

# Normalise timestamps to month-start (day=1, no sub-day component)
era5_times = (pd.DatetimeIndex(ds_850["time"].values)
              .normalize()
              .map(lambda t: t.replace(day=1)))
ds_850["time"] = era5_times

era5_lats = ds_850["lat"].values
era5_lons = ds_850["lon"].values

# Collect 850 hPa channels — try short name then case-insensitive partial match
era5_850_arrays = []
found_vars      = []
for var in ERA5_850_VARS:
    if var in ds_850.data_vars:
        era5_850_arrays.append(ds_850[var].values.astype(np.float32))
        found_vars.append(var)
    else:
        match = next((v for v in ds_850.data_vars if var.lower() in v.lower()), None)
        if match:
            era5_850_arrays.append(ds_850[match].values.astype(np.float32))
            found_vars.append(match)
            print(f"  '{var}' matched to '{match}'")
        else:
            print(f"  ⚠  Variable '{var}' not found — skipping")

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

if "latitude"   in ds_tp.dims: ds_tp = ds_tp.rename({"latitude":  "lat"})
if "longitude"  in ds_tp.dims: ds_tp = ds_tp.rename({"longitude": "lon"})
if "valid_time" in ds_tp.dims: ds_tp = ds_tp.rename({"valid_time": "time"})

if ds_tp["lat"].values[0] > ds_tp["lat"].values[-1]:
    ds_tp = ds_tp.isel(lat=slice(None, None, -1))

# Normalise TP timestamps to month-start — same logic as 850 hPa
tp_times = (pd.DatetimeIndex(ds_tp["time"].values)
            .normalize()
            .map(lambda t: t.replace(day=1)))
ds_tp["time"] = tp_times

# Auto-detect precipitation variable name
tp_var = next(
    (v for v in ds_tp.data_vars
     if any(k in v.lower() for k in ["tp", "precip", "precipitation"])),
    list(ds_tp.data_vars)[0]
)
print(f"  TP variable     : '{tp_var}'")
print(f"  TP units        : {ds_tp[tp_var].attrs.get('units', 'not set')}")
print(f"  TP time range   : {tp_times[0].date()} → {tp_times[-1].date()}")

# Regrid TP onto the 850 hPa spatial grid (bilinear)
print("  Regridding TP → 850 hPa grid (bilinear) ...")
ds_tp_regrid = ds_tp[tp_var].interp(
    lat=era5_lats,
    lon=era5_lons,
    method="linear"
)

# Inner-join on time — now guaranteed to work because both are month-start
common_tp_times = tp_times.intersection(era5_times)
print(f"  Common timesteps after alignment : {len(common_tp_times)}")

if len(common_tp_times) == 0:
    raise ValueError(
        "No common timesteps found between ERA5 850 hPa and ERA5 TP. "
        "Check that both files cover overlapping date ranges."
    )

ds_tp_aligned  = ds_tp_regrid.sel(time=common_tp_times)
ds_850_aligned = ds_850.sel(time=common_tp_times)

# Re-extract 850 hPa arrays on the aligned time axis
era5_850_arrays = [ds_850_aligned[var].values.astype(np.float32)
                   for var in found_vars]
era5_times = common_tp_times

# Stack all channels: 850 hPa vars + TP → (T, lat, lon, C)
tp_array      = ds_tp_aligned.values.astype(np.float32)
all_channels  = era5_850_arrays + [tp_array]
channel_names = found_vars + [tp_var]

era5_data = np.stack(all_channels, axis=-1)   # (T, H_lr, W_lr, C)

print(f"\n  ERA5 stacked shape : {era5_data.shape}  (T, H_lr, W_lr, C={len(channel_names)})")
print(f"  Channels           : {channel_names}")

# ---------------------------------------------------------------------------
# Step 2b — Coarsen ERA5 from native 0.25° → 0.50°
# ---------------------------------------------------------------------------
print("\n── Coarsening ERA5 0.25° → 0.50° ───────────────────")
print(f"  ERA5 before  : {era5_data.shape}  @ ~0.25°")

# Build clean coarsened grid snapped to multiples of ERA5_TARGET_RES
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

era5_coarse_list = []
for c in range(era5_data.shape[-1]):
    da_fine = xr.DataArray(
        era5_data[..., c],
        dims=["time", "lat", "lon"],
        coords={"time": era5_times, "lat": era5_lats, "lon": era5_lons}
    )
    da_coarse = da_fine.interp(lat=coarse_lats, lon=coarse_lons, method="linear")
    era5_coarse_list.append(da_coarse.values.astype(np.float32))

era5_data = np.stack(era5_coarse_list, axis=-1)   # (T, H_lr, W_lr, C)
era5_lats = coarse_lats
era5_lons = coarse_lons

print(f"  ERA5 after   : {era5_data.shape}  @ {ERA5_TARGET_RES}°")
print(f"  Scale factor : {SCALE_FACTOR}×  "
      f"({ERA5_TARGET_RES}° → {round(ERA5_TARGET_RES / SCALE_FACTOR, 4)}°)")

# Sanity check — means should be physically plausible
for c, name in enumerate(channel_names):
    print(f"  [{c}] {name:30s}  mean={np.nanmean(era5_coarse_list[c]):.4e}")

# ---------------------------------------------------------------------------
# Step 3 — Load CHIRPS from processed .nc
# ---------------------------------------------------------------------------
print("\n── Loading CHIRPS (processed .nc) ───────────────────")
ds_chirps = xr.open_dataset(CHIRPS_NC_FILE)

if ds_chirps["lat"].values[0] > ds_chirps["lat"].values[-1]:
    ds_chirps = ds_chirps.isel(lat=slice(None, None, -1))

chirps_times = pd.DatetimeIndex(ds_chirps["time"].values)
chirps_lats  = ds_chirps["lat"].values
chirps_lons  = ds_chirps["lon"].values
chirps_data  = ds_chirps["precip"].values.astype(np.float32)   # (T, H_hr, W_hr)

print(f"  CHIRPS shape : {chirps_data.shape}")
print(f"  Time range   : {chirps_times[0].date()} → {chirps_times[-1].date()}")
print(f"  Lat range    : {chirps_lats.min():.2f} → {chirps_lats.max():.2f}")
print(f"  Lon range    : {chirps_lons.min():.2f} → {chirps_lons.max():.2f}")
print(f"  Resolution   : ~{abs(chirps_lats[1]-chirps_lats[0]):.4f}°")

# ---------------------------------------------------------------------------
# Step 4 — Align spatial grids
#   Snap a common bounding box to the coarsened ERA5 grid (ERA5_TARGET_RES),
#   then verify the SCALE_FACTOR holds exactly.
# ---------------------------------------------------------------------------
print("\n── Aligning spatial grids ───────────────────────────")

res_era5   = ERA5_TARGET_RES                                   # ← uses 0.50, not hardcoded 0.25
res_chirps = abs(float(chirps_lats[1] - chirps_lats[0]))       # ~0.05

# Overlapping bounding box snapped to ERA5 grid lines
lat_min = np.round(max(era5_lats.min(), chirps_lats.min()) / res_era5) * res_era5
lat_max = np.round(min(era5_lats.max(), chirps_lats.max()) / res_era5) * res_era5
lon_min = np.round(max(era5_lons.min(), chirps_lons.min()) / res_era5) * res_era5
lon_max = np.round(min(era5_lons.max(), chirps_lons.max()) / res_era5) * res_era5

print(f"  Snapped bbox : lat [{lat_min}, {lat_max}]  lon [{lon_min}, {lon_max}]")

# Crop ERA5 to snapped box
era5_lat_idx = np.where((era5_lats >= lat_min - 1e-6) & (era5_lats <= lat_max + 1e-6))[0]
era5_lon_idx = np.where((era5_lons >= lon_min - 1e-6) & (era5_lons <= lon_max + 1e-6))[0]

era5_lat_crop = era5_lats[era5_lat_idx]
era5_lon_crop = era5_lons[era5_lon_idx]
era5_data     = era5_data[:, era5_lat_idx, :, :][:, :, era5_lon_idx, :]

# Build CHIRPS target coords at exactly SCALE_FACTOR × ERA5 resolution
n_lat = len(era5_lat_crop) * SCALE_FACTOR
n_lon = len(era5_lon_crop) * SCALE_FACTOR

chirps_lat_target = np.linspace(lat_max, lat_min, n_lat)   # descending (N→S)
chirps_lon_target = np.linspace(lon_min, lon_max, n_lon)   # ascending  (W→E)

# Verify the ratio holds exactly before proceeding
assert len(chirps_lat_target) == len(era5_lat_crop) * SCALE_FACTOR, (
    f"Lat mismatch: {len(chirps_lat_target)} ≠ {len(era5_lat_crop)} × {SCALE_FACTOR}")
assert len(chirps_lon_target) == len(era5_lon_crop) * SCALE_FACTOR, (
    f"Lon mismatch: {len(chirps_lon_target)} ≠ {len(era5_lon_crop)} × {SCALE_FACTOR}")

print(f"  ERA5  grid   : {len(era5_lat_crop)} × {len(era5_lon_crop)}  @ {res_era5}°")
print(f"  CHIRPS grid  : {len(chirps_lat_target)} × {len(chirps_lon_target)}  @ {res_chirps}°")
print(f"  Scale factor : {SCALE_FACTOR}×  ✓")

# Resample CHIRPS to snapped target grid (nearest preserves rainfall values)
print("  Resampling CHIRPS to snapped grid (nearest neighbour) ...")
chirps_da = xr.DataArray(
    chirps_data,
    dims=["time", "lat", "lon"],
    coords={"time": chirps_times, "lat": chirps_lats, "lon": chirps_lons}
)
chirps_da   = chirps_da.interp(lat=chirps_lat_target, lon=chirps_lon_target, method="nearest")
chirps_data = chirps_da.values   # (T, H_hr, W_hr)

print(f"  ERA5  final  : {era5_data.shape}")
print(f"  CHIRPS final : {chirps_data.shape}")

# ---------------------------------------------------------------------------
# Step 5 — Align time axes
# ---------------------------------------------------------------------------
print("\n── Aligning time axes ───────────────────────────────")

common_times = chirps_times.intersection(era5_times)
print(f"  Common months : {len(common_times)}  "
      f"({common_times[0].date()} → {common_times[-1].date()})")

era5_idx   = np.array([np.where(era5_times   == t)[0][0] for t in common_times])
chirps_idx = np.array([np.where(chirps_times == t)[0][0] for t in common_times])

era5_data   = era5_data[era5_idx]      # (N, H_lr, W_lr, C)
chirps_data = chirps_data[chirps_idx]  # (N, H_hr, W_hr)
times       = common_times

# Add channel dim to CHIRPS target: (N, H_hr, W_hr, 1)
chirps_data = chirps_data[..., np.newaxis]

# ---------------------------------------------------------------------------
# Step 6 — Land masks
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

print(f"  Train : {X_train.shape[0]} months  "
      f"({times[train_mask][0].date()} → {times[train_mask][-1].date()})")
print(f"  Val   : {X_val.shape[0]}   months  "
      f"({times[val_mask][0].date()}   → {times[val_mask][-1].date()})")
print(f"  Test  : {X_test.shape[0]}  months  "
      f"({times[test_mask][0].date()}  → {times[test_mask][-1].date()})")

# ---------------------------------------------------------------------------
# Step 8 — Normalise
#   ERA5 inputs  → z-score per channel (train stats only)
#   CHIRPS target → log1p then z-score (handles heavy-tailed distribution)
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

# Fill ocean/boundary NaNs with channel mean before normalising
for c in range(n_channels):
    for arr in [X_train, X_val, X_test]:
        arr[..., c][np.isnan(arr[..., c])] = norm_mean[c]

X_train = (X_train - norm_mean) / (norm_std + 1e-8)
X_val   = (X_val   - norm_mean) / (norm_std + 1e-8)
X_test  = (X_test  - norm_mean) / (norm_std + 1e-8)

# CHIRPS: log1p transform then z-score (stats from land pixels only)
print("\n  CHIRPS target → log1p + z-score")
y_train_log = np.log1p(np.nan_to_num(y_train, nan=0.0))
y_val_log   = np.log1p(np.nan_to_num(y_val,   nan=0.0))
y_test_log  = np.log1p(np.nan_to_num(y_test,  nan=0.0))

# Compute stats over land pixels only (expand mask to match array shape)
land_4d = np.broadcast_to(
    land_mask_hr[np.newaxis, ..., np.newaxis],
    y_train_log.shape
)
chirps_log_mean = float(y_train_log[land_4d].mean())
chirps_log_std  = float(y_train_log[land_4d].std())

y_train_norm = (y_train_log - chirps_log_mean) / (chirps_log_std + 1e-8)
y_val_norm   = (y_val_log   - chirps_log_mean) / (chirps_log_std + 1e-8)
y_test_norm  = (y_test_log  - chirps_log_mean) / (chirps_log_std + 1e-8)

print(f"  CHIRPS log-mean : {chirps_log_mean:.4f}")
print(f"  CHIRPS log-std  : {chirps_log_std:.4f}")

# ---------------------------------------------------------------------------
# Step 9 — Save
# ---------------------------------------------------------------------------
print(f"\n── Saving to {OUTPUT_DIR} ───────────────────────────")

for fname in ["train.npz", "val.npz", "test.npz", "metadata.npz"]:
    fpath = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(fpath):
        os.remove(fpath)

np.savez_compressed(
    os.path.join(OUTPUT_DIR, "train.npz"),
    X=X_train, y=y_train_norm,
    y_raw=y_train,
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
    era5_lats=era5_lat_crop,
    era5_lons=era5_lon_crop,
    chirps_lats=chirps_lat_target,
    chirps_lons=chirps_lon_target,
    land_mask_hr=land_mask_hr,
    land_mask_lr=land_mask_lr,
    norm_mean=norm_mean,
    norm_std=norm_std,
    channel_names=np.array(channel_names),
    chirps_log_mean=np.float32(chirps_log_mean),
    chirps_log_std=np.float32(chirps_log_std),
    scale_factor=np.int32(SCALE_FACTOR),
    era5_target_res=np.float32(ERA5_TARGET_RES),
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n── Summary ──────────────────────────────────────────")
print(f"  Input channels    : {len(channel_names)}  →  {channel_names}")
print(f"  X shape (train)   : {X_train.shape}  (N, H_lr, W_lr, C)")
print(f"  y shape (train)   : {y_train_norm.shape}  (N, H_hr, W_hr, 1)")
print(f"  Scale factor      : {SCALE_FACTOR}×  ({ERA5_TARGET_RES}° → {res_chirps}°)")
print()
for fname in ["train.npz", "val.npz", "test.npz", "metadata.npz"]:
    p = os.path.join(OUTPUT_DIR, fname)
    print(f"  {fname:20s}  {os.path.getsize(p)/1e6:.1f} MB")
print("\n✓ Training data ready for downscaling model.")