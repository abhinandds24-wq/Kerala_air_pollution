
import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv('df_final.csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce')

pollutant = st.sidebar.selectbox('Select Pollutant', ['AOD','NO2','SO2','CO','O3'])

df_sample = df.sample(5000, random_state=42)

st.title('Kerala Air Pollution Dashboard')
st.write(f'Displaying: {pollutant}')

fig = px.scatter_mapbox(
    df_sample,
    lat='lat', lon='lon',
    color=pollutant,
    size=pollutant,
    zoom=7,
    color_continuous_scale='Turbo',
    hover_data=['date','AOD','NO2','SO2','CO','O3']
)

fig.update_layout(mapbox_style='open-street-map')
st.plotly_chart(fig)
