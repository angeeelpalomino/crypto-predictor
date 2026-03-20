"""
╔══════════════════════════════════════════════════════════════╗
║         ANALIZADOR MULTI-CRIPTO — 8 MONEDAS                 ║
║  Corre solo cada día con GitHub Actions                      ║
║  Genera reporte HTML con señales de inversión                ║
╚══════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

import os, json
from datetime import datetime, date

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # sin pantalla (necesario en GitHub Actions)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error,
                             mean_absolute_percentage_error)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU, LSTM, Input
from tensorflow.keras.optimizers import Adam

# ─────────────────────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────
MONEDAS = {
    "BTC-USD"  : "Bitcoin",
    "ETH-USD"  : "Ethereum",
    "SOL-USD"  : "Solana",
    "XRP-USD"  : "Ripple",
    "ADA-USD"  : "Cardano",
    "DOGE-USD" : "Dogecoin",
    "TRX-USD"  : "TRON",
    "MATIC-USD": "Polygon",
}

TRAIN_START  = "2017-01-01"
TRAIN_END    = "2023-12-31"
TEST_START   = "2024-01-01"
TEST_END     = date.today().strftime("%Y-%m-%d")   # hasta hoy
EPOCHS       = 20
VENTANA      = 30
TIPO_CAMBIO  = 17.2          # MXN por USD (ajusta si necesitas)
OUTPUT_DIR   = "reporte"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
#  FUNCIONES BASE
# ─────────────────────────────────────────────────────────────
def descargar(ticker):
    hist = yf.Ticker(ticker).history(start=TRAIN_START, end=TEST_END)
    if hist.empty:
        raise ValueError(f"Sin datos para {ticker}")
    hist.index = pd.to_datetime(hist.index, utc=True).tz_localize(None).normalize()
    return hist["Close"].dropna()


def preparar_df(y):
    fecha_ref      = pd.to_datetime("2017-01-01")
    dias           = (y.index - fecha_ref).days
    dia_semana     = y.index.weekday
    return pd.DataFrame({
        "y"             : y.values,
        "dias_2017"     : dias,
        "dia_sin"       : np.sin(2 * np.pi * dia_semana / 7),
        "dia_cos"       : np.cos(2 * np.pi * dia_semana / 7),
        "retorno_diario": y.pct_change().fillna(0).values,
        "fecha_str"     : y.index.strftime("%Y-%m-%d").to_numpy(),
    }, index=y.index)


def escalar(df_train, df_test):
    cols    = ["y","dias_2017","dia_sin","dia_cos","retorno_diario"]
    scaler  = StandardScaler()
    tr_sc   = df_train.copy()
    te_sc   = df_test.copy()
    tr_sc[cols] = scaler.fit_transform(df_train[cols])
    te_sc[cols] = scaler.transform(df_test[cols])
    return tr_sc, te_sc, scaler


def crear_secuencias(df_sc, ventana):
    feats  = df_sc.drop(columns=["fecha_str"]).values
    fechas = df_sc["fecha_str"].values
    X, nd, ow, tw, fnd, fow, ftw = [], [], [], [], [], [], []
    for i in range(len(feats) - ventana - 10):
        X.append(feats[i:i+ventana])
        nd.append(feats[i+ventana][0])
        ow.append(feats[i+ventana+4][0])
        tw.append(feats[i+ventana+9][0])
        fnd.append(fechas[i+ventana])
        fow.append(fechas[i+ventana+4])
        ftw.append(fechas[i+ventana+9])
    return (np.array(X), np.array(nd), np.array(ow), np.array(tw),
            np.array(fnd), np.array(fow), np.array(ftw))


def build_gru(v, nf, lr=0.001):
    m = Sequential([
        Input(shape=(v, nf)),
        GRU(50, return_sequences=True, activation="relu"),
        Dropout(0.2),
        GRU(50, return_sequences=True, activation="relu"),
        Dropout(0.2),
        GRU(50, activation="relu"),
        Dropout(0.2),
        Dense(1),
    ])
    m.compile(optimizer=Adam(lr), loss="mean_squared_error")
    return m


def build_lstm(v, nf, lr=0.001):
    m = Sequential([
        Input(shape=(v, nf)),
        LSTM(100, activation="relu", return_sequences=True),
        Dropout(0.2),
        LSTM(100, activation="relu"),
        Dropout(0.2),
        Dense(50, activation="relu"),
        Dropout(0.2),
        Dense(1),
    ])
    m.compile(optimizer=Adam(lr), loss="mean_squared_error")
    return m


def entrenar(builder, Xtr, ytr, Xte, yte, nf):
    m    = builder(VENTANA, nf)
    m.fit(Xtr, ytr, epochs=EPOCHS, batch_size=32,
          validation_split=0.2, verbose=0)
    pred = m.predict(Xte, verbose=0).flatten()
    return (pred,
            mean_absolute_percentage_error(yte, pred),
            mean_absolute_error(yte, pred))


def señal(pred, actual, umbral=0.02):
    cambio = (pred - actual) / (abs(actual) + 1e-9)
    if   cambio >  umbral: return "COMPRA",  cambio * 100, "#22c55e"
    elif cambio < -umbral: return "VENTA",   cambio * 100, "#ef4444"
    else:                  return "ESPERA",  cambio * 100, "#f59e0b"


# ─────────────────────────────────────────────────────────────
#  LOOP PRINCIPAL — analizar cada moneda
# ─────────────────────────────────────────────────────────────
resumen = []

for ticker, nombre in MONEDAS.items():
    print(f"\n{'='*55}")
    print(f"  Procesando {nombre} ({ticker})")
    print(f"{'='*55}")

    try:
        # 1. Datos
        y         = descargar(ticker)
        df        = preparar_df(y)
        df_train  = df.loc[TRAIN_START:TRAIN_END]
        df_test   = df.loc[TEST_START:]
        tr_sc, te_sc, _ = escalar(df_train, df_test)

        if len(df_test) < VENTANA + 11:
            print(f"  [!] Datos de test insuficientes, saltando...")
            continue

        # 2. Secuencias
        Xtr, ytr_nd, ytr_ow, ytr_tw, _, _, _ = crear_secuencias(tr_sc, VENTANA)
        Xte, yte_nd, yte_ow, yte_tw, d_nd, d_ow, d_tw = crear_secuencias(te_sc, VENTANA)
        nf = Xtr.shape[2]

        # 3. Entrenar mejor modelo (GRU lr=0.001 como base)
        print(f"  Entrenando GRU +1d  ...", end=" ")
        p_nd, mape_nd, mae_nd = entrenar(build_gru, Xtr, ytr_nd, Xte, yte_nd, nf)
        print(f"MAPE={mape_nd*100:.1f}%")

        print(f"  Entrenando GRU +5d  ...", end=" ")
        p_ow, mape_ow, _      = entrenar(build_gru, Xtr, ytr_ow, Xte, yte_ow, nf)
        print(f"MAPE={mape_ow*100:.1f}%")

        print(f"  Entrenando GRU +10d ...", end=" ")
        p_tw, mape_tw, _      = entrenar(build_gru, Xtr, ytr_tw, Xte, yte_tw, nf)
        print(f"MAPE={mape_tw*100:.1f}%")

        # 4. Señales
        ultimo_real = yte_nd[-1]
        a1,  c1,  color1  = señal(p_nd[-1], ultimo_real)
        a5,  c5,  color5  = señal(p_ow[-1], ultimo_real)
        a10, c10, color10 = señal(p_tw[-1], ultimo_real)

        precio_usd = float(y.iloc[-1])
        precio_mxn = precio_usd * TIPO_CAMBIO

        # 5. Score compuesto (menor MAPE + señal positiva = mejor)
        # Penaliza si las 3 señales apuntan en distinto sentido
        señales_vals = [c1, c5, c10]
        consenso = sum(1 for s in señales_vals if s > 0) - sum(1 for s in señales_vals if s < 0)
        score = (1 / (mape_nd + 0.01)) * (1 + consenso * 0.1)

        # 6. Gráfica de predicción
        fig, axs = plt.subplots(3, 1, figsize=(12, 9))
        fig.suptitle(f"{nombre} ({ticker})  —  {date.today()}", fontsize=13)
        for ax, (titulo, yte, pred, fechas) in zip(axs, [
            ("+1 día",   yte_nd, p_nd, pd.to_datetime(d_nd)),
            ("+5 días",  yte_ow, p_ow, pd.to_datetime(d_ow)),
            ("+10 días", yte_tw, p_tw, pd.to_datetime(d_tw)),
        ]):
            ax.plot(fechas, yte,  label="Real",        color="black",     lw=1.5)
            ax.plot(fechas, pred, label="Predicción",  color="steelblue", lw=1, ls="--")
            ax.set_title(titulo, fontsize=10)
            ax.set_ylabel("Precio escalado")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.tight_layout()
        img_path = os.path.join(OUTPUT_DIR, f"{ticker.replace('-','_')}.png")
        plt.savefig(img_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  Gráfica guardada: {img_path}")

        resumen.append({
            "ticker"    : ticker,
            "nombre"    : nombre,
            "precio_usd": round(precio_usd, 6),
            "precio_mxn": round(precio_mxn, 4),
            "mape_nd"   : round(mape_nd * 100, 2),
            "mape_ow"   : round(mape_ow * 100, 2),
            "mape_tw"   : round(mape_tw * 100, 2),
            "señal_1d"  : a1,  "cambio_1d"  : round(c1,  2), "color_1d"  : color1,
            "señal_5d"  : a5,  "cambio_5d"  : round(c5,  2), "color_5d"  : color5,
            "señal_10d" : a10, "cambio_10d" : round(c10, 2), "color_10d" : color10,
            "score"     : round(score, 4),
            "img"       : f"{ticker.replace('-','_')}.png",
        })

    except Exception as e:
        print(f"  ERROR con {ticker}: {e}")
        continue

# ─────────────────────────────────────────────────────────────
#  ORDENAR POR SCORE (mejor primero)
# ─────────────────────────────────────────────────────────────
resumen.sort(key=lambda x: x["score"], reverse=True)

# Guardar JSON para uso externo
with open(os.path.join(OUTPUT_DIR, "señales.json"), "w", encoding="utf-8") as f:
    json.dump(resumen, f, ensure_ascii=False, indent=2)

# ─────────────────────────────────────────────────────────────
#  GENERAR REPORTE HTML
# ─────────────────────────────────────────────────────────────
def badge(accion, cambio, color):
    arrow = "▲" if cambio > 0 else ("▼" if cambio < 0 else "●")
    return (f'<span style="background:{color};color:#fff;padding:3px 10px;'
            f'border-radius:6px;font-weight:600;font-size:13px">'
            f'{arrow} {accion} ({cambio:+.1f}%)</span>')


filas_tabla = ""
for i, m in enumerate(resumen):
    medal = ["🥇","🥈","🥉","4.","5.","6.","7.","8."][i] if i < 8 else f"{i+1}."
    filas_tabla += f"""
    <tr>
      <td style="font-size:20px;text-align:center">{medal}</td>
      <td><strong>{m['nombre']}</strong><br>
          <span style="color:#888;font-size:12px">{m['ticker']}</span></td>
      <td style="text-align:right"><strong>${m['precio_usd']:.6f}</strong><br>
          <span style="color:#888;font-size:12px">${m['precio_mxn']:.4f} MXN</span></td>
      <td style="text-align:center">{badge(m['señal_1d'],  m['cambio_1d'],  m['color_1d'])}</td>
      <td style="text-align:center">{badge(m['señal_5d'],  m['cambio_5d'],  m['color_5d'])}</td>
      <td style="text-align:center">{badge(m['señal_10d'], m['cambio_10d'], m['color_10d'])}</td>
      <td style="text-align:center">
        <span style="font-size:12px;color:#666">Error pred:</span><br>
        <strong>{m['mape_nd']}%</strong>
      </td>
    </tr>
    <tr>
      <td colspan="7" style="padding:0 0 24px 0;background:#f9f9f9">
        <img src="{m['img']}" style="width:100%;max-width:900px;display:block;margin:8px auto;border-radius:8px">
      </td>
    </tr>"""


html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reporte Cripto — {date.today()}</title>
<style>
  body       {{ font-family: system-ui, sans-serif; max-width: 1000px;
                margin: 0 auto; padding: 20px; background: #f4f4f4; color: #222; }}
  h1         {{ font-size: 22px; margin-bottom: 4px; }}
  .sub       {{ color: #666; font-size: 14px; margin-bottom: 24px; }}
  table      {{ width: 100%; border-collapse: collapse; background: #fff;
                border-radius: 12px; overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
  th         {{ background: #1e293b; color: #fff; padding: 12px 10px;
                font-size: 13px; text-align: center; }}
  td         {{ padding: 12px 10px; border-bottom: 1px solid #eee;
                vertical-align: middle; }}
  tr:last-child td {{ border-bottom: none; }}
  .aviso     {{ background: #fef9c3; border-left: 4px solid #ca8a04;
                padding: 12px 16px; border-radius: 6px; font-size: 13px;
                margin-top: 28px; }}
  .ganador   {{ background: #f0fdf4; }}
  .footer    {{ text-align:center; color:#aaa; font-size:12px; margin-top:32px; }}
</style>
</head>
<body>

<h1>📊 Reporte de Criptomonedas</h1>
<div class="sub">Actualizado: {datetime.now().strftime('%d/%m/%Y %H:%M')} UTC &nbsp;|&nbsp;
Modelos: GRU (VENTANA={VENTANA} días, EPOCHS={EPOCHS}) &nbsp;|&nbsp;
Tipo de cambio: ${TIPO_CAMBIO} MXN/USD</div>

<table>
  <thead>
    <tr>
      <th>#</th>
      <th>Moneda</th>
      <th>Precio</th>
      <th>Señal +1 día</th>
      <th>Señal +5 días</th>
      <th>Señal +10 días</th>
      <th>Confianza</th>
    </tr>
  </thead>
  <tbody>
    {filas_tabla}
  </tbody>
</table>

<div class="aviso">
  ⚠️ <strong>Advertencia:</strong> Este reporte es generado automáticamente por modelos de
  aprendizaje profundo entrenados con datos históricos. <strong>No es asesoría financiera.</strong>
  Los mercados de criptomonedas son altamente volátiles. Nunca inviertas más de lo que
  puedes perder. Consulta múltiples fuentes antes de tomar decisiones.
</div>

<div class="footer">Generado automáticamente con GitHub Actions · GRU/LSTM · yfinance</div>
</body>
</html>"""

html_path = os.path.join(OUTPUT_DIR, "index.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n{'='*55}")
print(f"  REPORTE GENERADO: {html_path}")
print(f"{'='*55}")

# Imprimir resumen en consola
print(f"\n{'RANK':<5} {'MONEDA':<12} {'PRECIO USD':<14} {'SEÑAL +1d':<10} {'MAPE':>6}")
print("-" * 55)
for i, m in enumerate(resumen, 1):
    print(f"  {i:<4} {m['nombre']:<12} ${m['precio_usd']:<13.6f} {m['señal_1d']:<10} {m['mape_nd']:>5.1f}%")
