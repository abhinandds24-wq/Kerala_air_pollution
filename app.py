# app.py - Basic kriging-enabled Kerala Pollution Dashboard (Option A)
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from shapely.geometry import shape, Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from pykrige.ok import OrdinaryKriging
import plotly.express as px
import gdown

st.set_page_config(page_title="Kerala Pollution Dashboard (Kriging)", layout="wide")

# -------------------------
# CONFIG
# -------------------------
DATA_URL = "https://drive.google.com/uc?id=1M6I2ku_aWGkWz0GypktKXeRJPjNhlsM2"
LOCAL_FILE = "kerala_pollution.csv"            # file downloaded by gdown
BOUNDARY_PATH = "/mnt/data/state (1).geojson"  # local geojson you uploaded

# -------------------------
# LOAD DATA (gdown if needed)
# -------------------------
@st.cache_data
def load_data():
    # download only once
    if not os.path.exists(LOCAL_FILE):
        with st.spinner("Downloading large dataset from Google Drive..."):
            gdown.download(DATA_URL, LOCAL_FILE, quiet=False)

    df = pd.read_csv(LOCAL_FILE)

    # normalize column names
    df.columns = [c.strip() for c in df.columns]

    # parse date and ensure numeric lat/lon
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    df = df.dropna(subset=["date", "lat", "lon"]).reset_index(drop=True)
    return df

# -------------------------
# LOAD KERALA GEOJSON (local uploaded file)
# -------------------------
@st.cache_data
def load_kerala_polygon():
    if not os.path.exists(BOUNDARY_PATH):
        st.error(f"Kerala boundary file not found at: {BOUNDARY_PATH}")
        st.stop()

    with open(BOUNDARY_PATH, "r", encoding="utf-8") as f:
        gj = json.load(f)

    features = gj["features"] if "features" in gj else [gj]
    polys = []
    for feat in features:
        geom = feat.get("geometry") if isinstance(feat, dict) else feat
        shp = shape(geom)
        if isinstance(shp, (Polygon, MultiPolygon)):
            polys.append(shp)

    return unary_union(polys)

# -------------------------
# Clip points to polygon
# -------------------------
def clip_points_to_polygon(df, polygon):
    pts = [Point(xy) for xy in zip(df["lon"].values, df["lat"].values)]
    mask = np.array([polygon.contains(p) for p in pts])
    return df.loc[mask].reset_index(drop=True)

# -------------------------
# Detrend helper (linear on lon, lat)
# -------------------------
def detrend_linear(df, value_col):
    # design matrix: [1, lon, lat]
    X = np.vstack([np.ones(len(df)), df["lon"].values, df["lat"].values]).T
    y = df[value_col].values
    # solve least squares
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    trend = X.dot(coef)
    resid = y - trend
    return resid, coef  # resid array, coef (3,)

def predict_trend_grid(gx, gy, coef):
    # gx: grid lon array, gy: grid lat array -> produce trend grid same shape as z
    GX, GY = np.meshgrid(gx, gy)
    XX = np.vstack([np.ones(GX.size), GX.ravel(), GY.ravel()]).T
    trend_flat = XX.dot(coef)
    return trend_flat.reshape(GX.shape)

# -------------------------
# Kriging (ordinary) on residuals
# -------------------------
def do_ordinary_kriging_on_residuals(df_points, value_col, grid_res=150, variogram_model="spherical"):
    # prepare arrays
    lons = df_points["lon"].values.astype(float)
    lats = df_points["lat"].values.astype(float)
    vals = df_points[value_col].values.astype(float)

    # build grid with small padding
    pad_x = (lons.max() - lons.min()) * 0.02 if (lons.max() > lons.min()) else 0.01
    pad_y = (lats.max() - lats.min()) * 0.02 if (lats.max() > lats.min()) else 0.01
    gx = np.linspace(lons.min() - pad_x, lons.max() + pad_x, grid_res)
    gy = np.linspace(lats.min() - pad_y, lats.max() + pad_y, grid_res)

    OK = OrdinaryKriging(lons, lats, vals, variogram_model=variogram_model, verbose=False, enable_plotting=False)
    z, ss = OK.execute("grid", gx, gy)  # z shape: (ny, nx)
    return gx, gy, z, ss

# -------------------------
# Mask grid by polygon
# -------------------------
def mask_grid_to_polygon(gx, gy, z, polygon):
    xx, yy = np.meshgrid(gx, gy)
    flat_lon = xx.ravel()
    flat_lat = yy.ravel()
    flat_z = z.ravel()
    pts = [Point(xy) for xy in zip(flat_lon, flat_lat)]
    mask = np.array([polygon.contains(p) for p in pts])
    if mask.sum() == 0:
        return pd.DataFrame(columns=["lon","lat","value"])
    return pd.DataFrame({
        "lon": flat_lon[mask],
        "lat": flat_lat[mask],
        "value": flat_z[mask]
    })

# -------------------------
# MAIN
# -------------------------
df_all = load_data()
kerala_poly = load_kerala_polygon()

st.sidebar.header("Controls")

# auto detect pollutant columns
candidate_pollutants = [c for c in df_all.columns if c.upper() in ["AOD","NO2","SO2","CO","O3"]]
if not candidate_pollutants:
    # fallback numeric columns (excluding lat/lon)
    numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
    candidate_pollutants = [c for c in numeric_cols if c not in ["lat","lon"]]
pollutant = st.sidebar.selectbox("Select pollutant", candidate_pollutants)

mode = st.sidebar.radio("View Mode", ["Interactive Map","Daily Slice","Monthly Slice","Heatmap","Kriging Smooth Map"])

# sample size for display / kriging
sample_size = st.sidebar.slider("Sample Size (for plotting / kriging)", 500, 5000, 2000, step=100)
grid_res = st.sidebar.slider("Kriging grid resolution", 80, 200, 140, step=10)
variogram_model = st.sidebar.selectbox("Variogram model", ["spherical","exponential","gaussian"], index=0)

# log transform option
use_log = st.sidebar.checkbox("Log-transform pollutant (if > 0)", value=False)

# date selection depending on mode
if mode == "Daily Slice":
    date_min = df_all["date"].min().date()
    date_max = df_all["date"].max().date()
    sel_date = st.sidebar.date_input("Select date (daily slice)", value=date_min, min_value=date_min, max_value=date_max)
    # filter for that date
    df_slice = df_all[df_all["date"].dt.date == pd.to_datetime(sel_date).date()].copy()
elif mode == "Monthly Slice":
    # compute unique year-months
    df_all["year_month"] = df_all["date"].dt.to_period("M").astype(str)
    months = sorted(df_all["year_month"].unique())
    sel_month = st.sidebar.selectbox("Select year-month", months, index=len(months)-1)
    df_slice = df_all[df_all["year_month"] == sel_month].copy()
else:
    # not a time-sliced mode
    df_slice = df_all.copy()

# basic filtering
df_slice = df_slice.dropna(subset=["lat","lon",pollutant]).reset_index(drop=True)
if df_slice.empty:
    st.error("No data for selected slice / date range.")
    st.stop()

# clip to Kerala
df_slice = clip_points_to_polygon(df_slice, kerala_poly)
if df_slice.empty:
    st.error("No points fall inside Kerala for selection.")
    st.stop()

# sampling for display/kriging
df_display = df_slice.sample(min(sample_size, len(df_slice)), random_state=42).reset_index(drop=True)

st.title("🌏 Kerala Air Pollution Dashboard (Basic Kriging)")
st.write(f"Mode: **{mode}**, Pollutant: **{pollutant}** — Points used: {len(df_display):,}")

# ---------- Interactive Map ----------
if mode == "Interactive Map":
    fig = px.scatter_mapbox(df_display, lat="lat", lon="lon", color=pollutant, size=pollutant,
                            hover_data=["date"], zoom=7, height=750, color_continuous_scale="Turbo")
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

# ---------- Daily Slice & Monthly Slice & Heatmap (simple displays) ----------
elif mode in ["Daily Slice","Monthly Slice"]:
    fig = px.scatter_mapbox(df_display, lat="lat", lon="lon", color=pollutant, size=pollutant,
                            hover_data=["date"], zoom=7, height=750, color_continuous_scale="Turbo")
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

elif mode == "Heatmap":
    fig = px.density_mapbox(df_display, lat="lat", lon="lon", z=pollutant,
                            radius=25, center=dict(lat=df_slice["lat"].mean(), lon=df_slice["lon"].mean()),
                            zoom=7, height=750, color_continuous_scale="Turbo")
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

# ---------- Kriging ----------
elif mode == "Kriging Smooth Map":
    st.subheader("Ordinary Kriging (detrend residual kriging)")

    # ensure enough points
    if len(df_display) < 10:
        st.error("Not enough points for stable kriging (need more points). Try increasing sample size or choose monthly aggregation.")
        st.stop()

    # prepare values and optional log transform
    vals_col = pollutant
    df_pts = df_display.copy()
    if use_log:
        # make sure values positive
        df_pts = df_pts[df_pts[pollutant] > 0].copy()
        if df_pts.empty:
            st.error("No positive values to log-transform.")
            st.stop()
        df_pts["val_trans"] = np.log(df_pts[pollutant].values)
        back_transform = np.exp
    else:
        df_pts["val_trans"] = df_pts[pollutant].values
        back_transform = lambda x: x

    # detrend linear on lon,lat
    resid, coef = detrend_linear(df_pts, "val_trans")
    df_pts["resid"] = resid

    # perform kriging on residuals
    with st.spinner("Performing kriging on residuals..."):
        gx, gy, z_resid, ss = do_ordinary_kriging_on_residuals(df_pts, "resid", grid_res=grid_res, variogram_model=variogram_model)

    # predict trend on grid and add back
    trend_grid = predict_trend_grid(gx, gy, coef)
    # z_resid is shape (ny, nx) as returned by pykrige (ny=len(gy), nx=len(gx))
    z_total = z_resid + trend_grid

    # back transform
    try:
        z_final = back_transform(z_total)
    except Exception:
        # if back transform fails for negative values, clip then transform
        z_final = back_transform(np.clip(z_total, a_min=1e-6, a_max=None))

    # mask to Kerala polygon and convert to dataframe for plotting
    grid_df = mask_grid_to_polygon(gx, gy, z_final, kerala_poly)
    if grid_df.empty:
        st.error("Kriged grid masked to Kerala is empty. Try increasing grid resolution or sample size.")
        st.stop()

    # optional kriging variance plot (std)
    show_variance = st.sidebar.checkbox("Show kriging standard deviation map", value=False)
    if show_variance:
        std_grid = np.sqrt(ss)  # standard deviation on residuals grid
        # add trend back to variance is not straightforward — we show residual std
        var_df = mask_grid_to_polygon(gx, gy, std_grid, kerala_poly)
        fig_var = px.density_mapbox(var_df, lat="lat", lon="lon", z="value", radius=10,
                                   zoom=7, height=600, color_continuous_scale="Viridis")
        fig_var.update_layout(mapbox_style="open-street-map", title="Kriging Std (residual)")
        st.plotly_chart(fig_var, use_container_width=True)

    # main kriged map
    fig = px.density_mapbox(grid_df, lat="lat", lon="lon", z="value", radius=10,
                            center=dict(lat=df_slice["lat"].mean(), lon=df_slice["lon"].mean()),
                            zoom=7, height=700, color_continuous_scale="Turbo")
    fig.update_layout(mapbox_style="open-street-map", title=f"Kriged {pollutant}")

    # overlay sample points
    if st.sidebar.checkbox("Overlay sample points", value=True):
        fig.add_scattermapbox(lat=df_pts["lat"], lon=df_pts["lon"], mode="markers",
                              marker=dict(size=6, color="black"), name="samples")

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.write("Notes: This is the basic kriging pipeline (detrend + ordinary kriging on residuals). For research-grade results add variogram fitting, anisotropy, external drift and spatial cross-validation.")
