# app.py
import streamlit as st
import pandas as pd
import numpy as np
from influxdb_client import InfluxDBClient
from sklearn.linear_model import LinearRegression
import altair as alt

# ---------------- CONFIGURACIÓN ----------------
st.set_page_config(page_title="Tablero IoT - Sensores", layout="wide")

st.title("🌡️ Tablero de Monitoreo IoT")
st.markdown("Visualización y análisis de **temperatura**, **humedad** y **vibración** desde InfluxDB")

# ---------- CREDENCIALES INFLUXDB ----------
# ⚠️ Reemplaza estos valores por los tuyos
INFLUX_URL = "https://us-east-1-1.aws.cloud2.influxdata.com"
INFLUX_TOKEN = "JcKXoXE30JQvV9Ggb4-zv6sQc0Zh6B6Haz5eMRW0FrJEduG2KcFJN9-7RoYvVORcFgtrHR-Q_ly-52pD7IC6JQ=="
INFLUX_ORG = "0925ccf91ab36478"
INFLUX_BUCKET = "EXTREME_MANUFACTURING"

# ---------- FUNCIONES ----------
@st.cache_data(ttl=300)
def get_data(measurement, start="-24h"):
    """Consulta datos desde InfluxDB."""
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    query_api = client.query_api()

    flux_query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {start})
      |> filter(fn: (r) => r._measurement == "{measurement}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> sort(columns: ["_time"])
    '''
    df = query_api.query_data_frame(flux_query)
    if "_time" in df.columns:
        df = df.set_index("_time")
        df.index = pd.to_datetime(df.index)
    client.close()
    return df


def kpis(df, var):
    """Muestra indicadores resumen."""
    if df.empty or var not in df.columns:
        return None
    col1, col2, col3, col4 = st.columns(4)
    actual = df[var].iloc[-1]
    promedio = df[var].mean()
    maximo = df[var].max()
    minimo = df[var].min()
    col1.metric(f"{var} actual", f"{actual:.2f}")
    col2.metric("Promedio", f"{promedio:.2f}")
    col3.metric("Máximo", f"{maximo:.2f}")
    col4.metric("Mínimo", f"{minimo:.2f}")


def plot_timeseries(df, variables):
    """Grafica series temporales."""
    if df.empty:
        st.warning("No hay datos para mostrar.")
        return
    df_plot = df[variables].reset_index().melt(id_vars="_time", var_name="variable", value_name="valor")
    chart = (
        alt.Chart(df_plot)
        .mark_line()
        .encode(
            x="_time:T",
            y="valor:Q",
            color="variable:N",
            tooltip=["_time:T", "variable:N", "valor:Q"]
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def predict_linear(series: pd.Series, horizon=20):
    """Predice tendencia lineal simple."""
    s = series.dropna()
    if s.empty:
        return pd.Series([])
    X = np.arange(len(s)).reshape(-1, 1)
    y = s.values
    model = LinearRegression().fit(X, y)
    future_X = np.arange(len(s), len(s) + horizon).reshape(-1, 1)
    preds = model.predict(future_X)
    freq = s.index.inferred_freq or "1min"
    future_index = pd.date_range(start=s.index[-1], periods=horizon + 1, freq=freq)[1:]
    return pd.Series(preds, index=future_index)


def detect_anomalies(series: pd.Series, window=10, z_thresh=3.0):
    """Detecta anomalías usando Z-score."""
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    z = (series - mean) / std
    return z.abs() > z_thresh


# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Controles")

rango = st.sidebar.selectbox("Rango de tiempo", ["Última hora", "Últimas 24h", "Últimos 7 días"])
if rango == "Última hora":
    start = "-1h"
elif rango == "Últimas 24h":
    start = "-24h"
else:
    start = "-7d"

variables = st.sidebar.multiselect(
    "Variables a mostrar",
    ["temperatura", "humedad", "vibracion"],
    default=["temperatura", "humedad", "vibracion"]
)

# ---------- CONSULTA DE DATOS ----------
st.subheader("📡 Datos desde InfluxDB")

data_frames = []

if "temperatura" in variables or "humedad" in variables:
    df_dht = get_data("dht22", start)
    data_frames.append(df_dht)
if "vibracion" in variables:
    df_mpu = get_data("mpu6050", start)
    data_frames.append(df_mpu)

if data_frames:
    df_all = pd.concat(data_frames, axis=1).sort_index()
else:
    st.warning("Selecciona al menos una variable.")
    st.stop()

# ---------- VISUALIZACIÓN ----------
st.subheader("📈 Indicadores y gráficos")
for var in variables:
    if var in df_all.columns:
        st.markdown(f"### {var.capitalize()}")
        kpis(df_all, var)
        plot_timeseries(df_all, [var])

# ---------- PREDICCIÓN ----------
st.subheader("🔮 Predicción de tendencias (modelo lineal simple)")

var_pred = st.selectbox("Seleccionar variable para predecir:", variables)

if st.button("Generar predicción"):
    if var_pred not in df_all.columns or df_all[var_pred].dropna().empty:
        st.warning("No hay datos para predecir.")
    else:
        pred = predict_linear(df_all[var_pred], horizon=30)
        pred_df = pd.DataFrame({
            "Tiempo": pred.index,
            "Predicción": pred.values
        })
        st.line_chart(pred_df.set_index("Tiempo"))
        st.success("Predicción generada correctamente ✅")

# ---------- DETECCIÓN DE ANOMALÍAS ----------
st.subheader("🚨 Detección de anomalías")

var_anom = st.selectbox("Seleccionar variable para detectar anomalías:", variables)

if st.button("Detectar anomalías"):
    if var_anom not in df_all.columns or df_all[var_anom].dropna().empty:
        st.warning("No hay datos para analizar.")
    else:
        anoms = detect_anomalies(df_all[var_anom])
        df_show = pd.DataFrame({
            "Tiempo": df_all.index,
            "Valor": df_all[var_anom].values,
            "Anomalía": anoms.values
        })
        puntos = df_show[df_show["Anomalía"]]
        st.write(f"Se detectaron {len(puntos)} posibles anomalías.")
        chart = (
            alt.Chart(df_show)
            .mark_line(color="steelblue")
            .encode(x="Tiempo:T", y="Valor:Q")
            + alt.Chart(puntos)
            .mark_point(color="red", size=80)
            .encode(x="Tiempo:T", y="Valor:Q")
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
