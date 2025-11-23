# app.py — Kerala Pollution Dashboard (Monthly & Seasonal Kriging + Yearly Animation)
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

st.set_page_config(page_title="Kerala Pollution Dashboard — Kriging & Animation", layout="wide")

# -------------------------
# CONFIG
# -------------------------
# Large dataset: try local first, otherwise download from Drive
LOCAL_DATA_PATHS = [
    "/mnt/data/df_final.csv",      # environment upload path (if present)
    "/mnt/data/Ernakulam_Daily_AQI_2018_2024_with_LatLon.csv",
    "/mnt/data/Kerala_S5P_Cleaned_2018_2025.csv"
]
DATA_URL = "https://drive.google.com/uc?id=1M6I2ku_aWGkWz0GypktKXeRJPjNhlsM2"  # fallback gdown link
LOCAL_FILE = "kerala_pollution.csv"  # file name to save downloaded CSV

# Boundary: repo file (user selected B). Also try raw GitHub fallback
BOUNDARY_PATH = "kerala_boundary.geojson"
GITHUB_RAW_BOUNDARY = "https://raw.githubusercontent.com/Abhinand-1/air_pollution/main/kerala_boundary.geojson"

# Fast-mode defaults (Option A)
DEFAULT_SAMPLE = 1000
DEFAULT_GRID = 60

# -------------------------
# DATA LOADING / HELPERS
# -------------------------
@st.cache_data(show_spinner=False)
def find_local_csv():
    """Return the first existing CSV path from LOCAL_DATA_PATHS or None."""
    for p in LOCAL_DATA_PATHS:
        if os.path.exists(p):
            return p
    return None

@st.cache_data(show_spinner=False)
def load_data():
    # 1) try known local upload paths
    local = find_local_csv()
    if local is not None:
        csv_path = local
    else:
        # 2) try previously downloaded LOCAL_FILE
        if os.path.exists(LOCAL_FILE):
            csv_path = LOCAL_FILE
        else:
            # 3) download via gdown
            with st.spinner("Downloading large dataset from Google Drive (first run)..."):
                try:
                    gdown.download(DATA_URL, LOCAL_FILE, quiet=False)
                    csv_path = LOCAL_FILE
                except Exception as e:
                    st.error(f"Failed to download dataset from Google Drive: {e}")
                    st.stop()

    # read CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Failed to read CSV at {csv_path}: {e}")
        st.stop()

    # normalize columns and parse
    df.columns = [c.strip() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
    df["lon"] = pd.to_numeric(df.get("lon"), errors="coerce")
    df = df.dropna(subset=["date","lat","lon"]).reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def load_kerala_polygon():
    # Try repo file first
    if os.path.exists(BOUNDARY_PATH):
        path = BOUNDARY_PATH
        with open(path, "r", encoding="utf-8") as f:
            gj = json.load(f)
    else:
        # try raw GitHub fallback
        try:
            import urllib.request
            with urllib.request.urlopen(GITHUB_RAW_BOUNDARY) as resp:
                gj = json.load(resp)
        except Exception as e:
            st.error(f"Kerala boundary geojson not found locally ({BOUNDARY_PATH}) and could not fetch from GitHub: {e}")
            st.stop()

    features = gj["features"] if "features" in gj else [gj]
    polys = []
    for feat in features:
        geom = feat.get("geometry") if isinstance(feat, dict) else feat
        shp = shape(geom)
        if isinstance(shp, (Polygon, MultiPolygon)):
            polys.append(shp)
    if not polys:
        st.error("No polygon geometry found in boundary geojson.")
        st.stop()
    return unary_union(polys)

# clipping helper
def clip_points_to_polygon(df, polygon):
    pts = [Point(xy) for xy in zip(df["lon"].values, df["lat"].values)]
    mask = np.array([polygon.contains(p) for p in pts])
    return df.loc[mask].reset_index(drop=True)

# detrend helper
def detrend_linear(df, value_col):
    X = np.vstack([np.ones(len(df)), df["lon"].values, df["lat"].values]).T
    y = df[value_col].values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    trend = X.dot(coef)
    resid = y - trend
    return resid, coef

def predict_trend_grid(gx, gy, coef):
    GX, GY = np.meshgrid(gx, gy)
    XX = np.vstack([np.ones(GX.size), GX.ravel(), GY.ravel()]).T
    trend_flat = XX.dot(coef)
    return trend_flat.reshape(GX.shape)

# kriging helpers
def do_ordinary_kriging_on_residuals(df_points, value_col, grid_res=DEFAULT_GRID, variogram_model="spherical"):
    lons = df_points["lon"].values.astype(float)
    lats = df_points["lat"].values.astype(float)
    vals = df_points[value_col].values.astype(float)
    pad_x = (lons.max() - lons.min()) * 0.02 if (lons.max() > lons.min()) else 0.01
    pad_y = (lats.max() - lats.min()) * 0.02 if (lats.max() > lats.min()) else 0.01
    gx = np.linspace(lons.min() - pad_x, lons.max() + pad_x, grid_res)
    gy = np.linspace(lats.min() - pad_y, lats.max() + pad_y, grid_res)
    OK = OrdinaryKriging(lons, lats, vals, variogram_model=variogram_model, verbose=False, enable_plotting=False)
    z, ss = OK.execute("grid", gx, gy)
    return gx, gy, z, ss

def mask_grid_to_polygon(gx, gy, z, polygon):
    xx, yy = np.meshgrid(gx, gy)
    flat_lon = xx.ravel()
    flat_lat = yy.ravel()
    flat_z = z.ravel()
    pts = [Point(xy) for xy in zip(flat_lon, flat_lat)]
    mask = np.array([polygon.contains(p) for p in pts])
    if mask.sum() == 0:
        return pd.DataFrame(columns=["lon","lat","value"])
    return pd.DataFrame({"lon": flat_lon[mask], "lat": flat_lat[mask], "value": flat_z[mask]})

# seasons helper
def add_season_col(df):
    def season_from_dt(d):
        m = d.month
        if m in [12,1,2]:
            return "Winter"
        if m in [3,4,5]:
            return "Summer"
        if m in [6,7,8,9]:
            return "Monsoon"
        return "Post-monsoon"
    df["season"] = df["date"].apply(season_from_dt)
    return df

# -------------------------
# LOAD everything
# -------------------------
df_all = load_data()
kerala_poly = load_kerala_polygon()

# -------------------------
# UI: sidebar controls
# -------------------------
st.sidebar.header("Controls")

# detect pollutant columns
candidate = [c for c in df_all.columns if c.upper() in ["AOD","NO2","SO2","CO","O3"]]
if not candidate:
    numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
    candidate = [c for c in numeric_cols if c not in ["lat","lon"]]
pollutant = st.sidebar.selectbox("Pollutant", candidate)

view_mode = st.sidebar.radio("View", [
    "Interactive Map",
    "Monthly Mean Kriging",
    "Seasonal Kriging",
    "Heatmap",
    "Yearly Heatmap Animation (2018–2025)",
    "Daily Slice (points only)"
])

# fast-mode defaults (Option A)
sample_size = st.sidebar.slider("Sample size (for kriging/display)", 200, 2000, DEFAULT_SAMPLE, step=100)
grid_res = st.sidebar.slider("Grid resolution (fast mode)", 40, 120, DEFAULT_GRID, step=10)
variogram_model = st.sidebar.selectbox("Variogram model", ["spherical","exponential","gaussian"], index=0)
use_log = st.sidebar.checkbox("Log-transform pollutant (if >0)", value=False)

# selectors for monthly/seasonal/daily
sel_month, sel_season = None, None
if view_mode == "Monthly Mean Kriging":
    df_all["year_month"] = df_all["date"].dt.to_period("M").astype(str)
    months = sorted(df_all["year_month"].unique())
    sel_month = st.sidebar.selectbox("Select month (YYYY-MM)", months, index=max(0, len(months)-1))
    df_slice = df_all[df_all["year_month"] == sel_month].copy()
elif view_mode == "Seasonal Kriging":
    df_all = add_season_col(df_all)
    seasons = ["Winter","Summer","Monsoon","Post-monsoon"]
    sel_season = st.sidebar.selectbox("Select season", seasons, index=2)
    df_slice = df_all[df_all["season"] == sel_season].copy()
elif view_mode == "Daily Slice (points only)":
    date_min = df_all["date"].min().date()
    date_max = df_all["date"].max().date()
    sel_date = st.sidebar.date_input("Select date", value=date_min, min_value=date_min, max_value=date_max)
    df_slice = df_all[df_all["date"].dt.date == pd.to_datetime(sel_date).date()].copy()
else:
    df_slice = df_all.copy()

# common filtering
df_slice = df_slice.dropna(subset=["lat","lon",pollutant]).reset_index(drop=True)
if df_slice.empty:
    st.error("No data for selected period/selection.")
    st.stop()

# clip to Kerala and sample
df_slice = clip_points_to_polygon(df_slice, kerala_poly)
if df_slice.empty:
    st.error("No points fall inside Kerala for selection.")
    st.stop()

df_sample = df_slice.sample(min(sample_size, len(df_slice)), random_state=42).reset_index(drop=True)

# Title
st.title("Kerala Pollution Dashboard — Kriging & Yearly Animation (Fast)")
st.write(f"Mode: **{view_mode}** — Pollutant: **{pollutant}** — Points used: {len(df_sample):,}")

# -------------------------
# VISUAL MODES
# -------------------------

# Interactive
if view_mode == "Interactive Map":
    fig = px.scatter_mapbox(df_sample, lat="lat", lon="lon", color=pollutant, size=pollutant,
                            hover_data=["date"], zoom=7, height=700, color_continuous_scale="Turbo")
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

# Daily slice (points only)
elif view_mode == "Daily Slice (points only)":
    fig = px.scatter_mapbox(df_sample, lat="lat", lon="lon", color=pollutant, size=pollutant,
                            hover_data=["date"], zoom=7, height=700, color_continuous_scale="Turbo")
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

# Heatmap (single frame)
elif view_mode == "Heatmap":
    fig = px.density_mapbox(df_sample, lat="lat", lon="lon", z=pollutant,
                            radius=20, center=dict(lat=df_slice["lat"].mean(), lon=df_slice["lon"].mean()),
                            zoom=7, height=700, color_continuous_scale="Turbo")
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

# Monthly mean kriging and Seasonal kriging (same pipeline)
elif view_mode in ["Monthly Mean Kriging", "Seasonal Kriging"]:
    st.subheader(f"Kriging — {view_mode}")
    if len(df_sample) < 8:
        st.error("Too few points for kriging; increase sample size or choose a broader period.")
        st.stop()

    # prepare values (log optional)
    df_pts = df_sample.copy()
    if use_log:
        df_pts = df_pts[df_pts[pollutant] > 0].copy()
        if df_pts.empty:
            st.error("No positive values available for log transform.")
            st.stop()
        df_pts["val_trans"] = np.log(df_pts[pollutant].values)
        back_transform = np.exp
    else:
        df_pts["val_trans"] = df_pts[pollutant].values
        back_transform = lambda x: x

    # detrend and krige residuals
    resid, coef = detrend_linear(df_pts, "val_trans")
    df_pts["resid"] = resid

    with st.spinner("Running Ordinary Kriging on residuals..."):
        gx, gy, z_resid, ss = do_ordinary_kriging_on_residuals(df_pts, "resid", grid_res=grid_res, variogram_model=variogram_model)

    # add trend back and back-transform
    trend_grid = predict_trend_grid(gx, gy, coef)
    z_total = z_resid + trend_grid
    try:
        z_final = back_transform(z_total)
    except Exception:
        z_final = back_transform(np.clip(z_total, a_min=1e-6, a_max=None))

    grid_df = mask_grid_to_polygon(gx, gy, z_final, kerala_poly)
    if grid_df.empty:
        st.error("Kriged grid masked to Kerala is empty. Try adjusting grid resolution or sample size.")
        st.stop()

    # optional std map
    if st.sidebar.checkbox("Show kriging residual std (uncertainty)", value=False):
        std_grid = np.sqrt(ss)
        var_df = mask_grid_to_polygon(gx, gy, std_grid, kerala_poly)
        fig_var = px.density_mapbox(var_df, lat="lat", lon="lon", z="value", radius=10,
                                   zoom=7, height=600, color_continuous_scale="Viridis", labels={"value":"std"})
        fig_var.update_layout(mapbox_style="open-street-map", title="Kriging residual standard deviation")
        st.plotly_chart(fig_var, use_container_width=True)

    # final kriged map
    title_label = sel_month if view_mode == "Monthly Mean Kriging" else sel_season
    fig = px.density_mapbox(grid_df, lat="lat", lon="lon", z="value", radius=8,
                            center=dict(lat=df_slice["lat"].mean(), lon=df_slice["lon"].mean()),
                            zoom=7, height=700, color_continuous_scale="Turbo")
    fig.update_layout(mapbox_style="open-street-map", title=f"Kriged {pollutant} — {title_label}")
    if st.sidebar.checkbox("Overlay sample points", value=True):
        fig.add_scattermapbox(lat=df_pts["lat"], lon=df_pts["lon"], mode="markers",
                              marker=dict(size=5, color="black"), name="samples")
    st.plotly_chart(fig, use_container_width=True)

# Yearly heatmap animation
elif view_mode == "Yearly Heatmap Animation (2018–2025)":
    st.subheader("🎞 Yearly Heatmap Animation (2018–2025)")

    df_year = df_all.copy()
    df_year["year"] = df_year["date"].dt.year
    df_year = df_year[(df_year["year"] >= 2018) & (df_year["year"] <= 2025)]
    df_year = df_year.dropna(subset=["lat","lon",pollutant])

    if df_year.empty:
        st.error("No data available for 2018–2025.")
        st.stop()

    # optionally sample per year to keep responsiveness
    max_per_year = st.sidebar.slider("Max points per year (animation)", 2000, 10000, 4000, step=500)
    df_anim = df_year.groupby("year").apply(lambda g: g.sample(min(len(g), max_per_year), random_state=1)).reset_index(drop=True)

    fig = px.density_mapbox(
        df_anim,
        lat="lat",
        lon="lon",
        z=pollutant,
        radius=18,
        animation_frame="year",
        center=dict(lat=df_anim["lat"].mean(), lon=df_anim["lon"].mean()),
        zoom=7,
        height=750,
        color_continuous_scale="Turbo",
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.write("Notes: Monthly/Seasonal kriging runs on single time-slices (correct approach). This is fast-mode (Option A). For research-grade variogram fitting, anisotropy, and full CV add the extended pipeline.")
