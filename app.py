# app.py
# --- Importación de librerías ---
import streamlit as st
import pandas as pd
import numpy as np
from influxdb_client import InfluxDBClient
from sklearn.linear_model import LinearRegression
import altair as alt
from datetime import timedelta

# --- Configuración de página ---
st.set_page_config(page_title="Tablero IoT - Extreme Manufacturing", layout="wide")
st.title("🏭 Tablero de Monitoreo IoT")
st.markdown("Visualización de datos de **temperatura, humedad y vibración** desde InfluxDB")

# --- Parámetros de conexión (usa tus credenciales) ---
INFLUXDB_URL = "https://us-east-1-1.aws.cloud2.influxdata.com"
INFLUXDB_TOKEN = "JcKXoXE30JQvV9Ggb4-zv6sQc0Zh6B6Haz5eMRW0FrJEduG2KcFJN9-7RoYvVORcFgtrHR-Q_ly-52pD7IC6JQ=="
INFLUXDB_ORG = "0925ccf91ab36478"
INFLUXDB_BUCKET = "EXTREME_MANUFACTURING"

# --- Sidebar (Controles) ---
st.sidebar.header("⚙️ Controles")

# Slider para elegir cantidad de días de antigüedad
dias = st.sidebar.slider("Seleccionar número de días hacia atrás", min_value=1, max_value=30, value=7)
start = f"-{dias}d"
stop = "now()"

# Selección de fuente de datos
fuente = st.sidebar.radio("Fuente de datos", ["DHT22 (Temperatura/Humedad)", "MPU6050 (Vibración)"])

# Selección de variable para predicción
pred_var = st.sidebar.selectbox("Variable para predecir:", ["temperatura", "humedad", "accel_x", "accel_y", "accel_z"])

# --- Funciones de consulta ---
@st.cache_data(ttl=300)
def get_data_dht22(start="-7d", stop="now()"):
    query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: {start}, stop: {stop})
      |> filter(fn: (r) => r._measurement == "studio-dht22")
      |> filter(fn: (r) => r._field == "humedad" or r._field == "temperatura" or r._field == "sensacion_termica")
      |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
      |> sort(columns: ["_time"])
    '''
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    df = client.query_api().query_data_frame(org=INFLUXDB_ORG, query=query)
    client.close()
    if "_time" in df.columns:
        df = df.set_index("_time")
        df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(ttl=300)
def get_data_mpu6050(start="-7d", stop="now()"):
    query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: {start}, stop: {stop})
      |> filter(fn: (r) => r._measurement == "mpu6050")
      |> filter(fn: (r) =>
          r._field == "accel_x" or r._field == "accel_y" or r._field == "accel_z" or
          r._field == "gyro_x" or r._field == "gyro_y" or r._field == "gyro_z" or
          r._field == "temperature")
      |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
      |> sort(columns: ["_time"])
    '''
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    df = client.query_api().query_data_frame(org=INFLUXDB_ORG, query=query)
    client.close()
    if "_time" in df.columns:
        df = df.set_index("_time")
        df.index = pd.to_datetime(df.index)
    return df

# --- Funciones de análisis ---
def kpis(df, var):
    if df.empty or var not in df.columns:
        return None
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"{var} actual", f"{df[var].iloc[-1]:.2f}")
    col2.metric("Promedio", f"{df[var].mean():.2f}")
    col3.metric("Máximo", f"{df[var].max():.2f}")
    col4.metric("Mínimo", f"{df[var].min():.2f}")


def plot_timeseries(df, variables):
    if df.empty:
        st.warning("No hay datos disponibles.")
        return
    df_plot = df[variables].reset_index().melt(id_vars="_time", var_name="variable", value_name="valor")
    chart = (
        alt.Chart(df_plot)
        .mark_line()
        .encode(x="_time:T", y="valor:Q", color="variable:N", tooltip=["_time:T", "variable:N", "valor:Q"])
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def predict_linear(series: pd.Series, horizon=20):
    s = series.dropna()
    if s.empty:
        return pd.Series([])
    X = np.arange(len(s)).reshape(-1, 1)
    y = s.values
    model = LinearRegression().fit(X, y)
    future_X = np.arange(len(s), len(s) + horizon).reshape(-1, 1)
    preds = model.predict(future_X)
    freq = s.index.inferred_freq or "1h"
    future_index = pd.date_range(start=s.index[-1], periods=horizon + 1, freq=freq)[1:]
    return pd.Series(preds, index=future_index)

# --- Carga de datos según fuente ---
if fuente.startswith("DHT22"):
    df = get_data_dht22(start, stop)
    st.subheader(f"🌡️ Datos del sensor DHT22 (últimos {dias} días)")
    variables = [v for v in ["temperatura", "humedad", "sensacion_termica"] if v in df.columns]
else:
    df = get_data_mpu6050(start, stop)
    st.subheader(f"📈 Datos del sensor MPU6050 (últimos {dias} días)")
    variables = [v for v in ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "temperature"] if v in df.columns]

# --- Visualización ---
if not df.empty:
    for var in variables:
        st.markdown(f"### {var}")
        kpis(df, var)
        plot_timeseries(df, [var])
else:
    st.warning("No se encontraron datos para el rango consultado.")

# --- Predicción ---
st.subheader("🔮 Predicción (modelo lineal simple)")
if pred_var in df.columns and st.button("Generar predicción"):
    pred = predict_linear(df[pred_var], horizon=30)
    if not pred.empty:
        pred_df = pd.DataFrame({"Tiempo": pred.index, "Predicción": pred.values})
        st.line_chart(pred_df.set_index("Tiempo"))
        st.success(f"Predicción generada correctamente para '{pred_var}'.")
    else:
        st.warning("No hay suficientes datos para predecir.")
