"""
ANALIZADOR — CRIPTOMONEDAS + ACCIONES
Genera reporte HTML con dos secciones
"""

import warnings
warnings.filterwarnings("ignore")

import os, json
from datetime import datetime, date

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU, Input
from tensorflow.keras.optimizers import Adam

# ── CONFIGURACIÓN ──────────────────────────────────────────
CRIPTOS = {
    "BTC-USD" : "Bitcoin",
    "ETH-USD" : "Ethereum",
    "BNB-USD" : "Binance Coin",
    "SOL-USD" : "Solana",
    "ADA-USD" : "Cardano",
}

ACCIONES = {
    "AAPL" : "Apple",
    "GOOGL": "Alphabet (Google)",
    "MSFT" : "Microsoft",
    "AMZN" : "Amazon",
    "META" : "Meta Platforms",
}

TRAIN_START = "2017-01-01"
TRAIN_END   = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = date.today().strftime("%Y-%m-%d")
EPOCHS      = 20
VENTANA     = 30
TIPO_CAMBIO = 17.2
OUTPUT_DIR  = "reporte"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── FUNCIONES ──────────────────────────────────────────────
def descargar(ticker):
    hist = yf.Ticker(ticker).history(start=TRAIN_START, end=TEST_END)
    if hist.empty:
        raise ValueError(f"Sin datos para {ticker}")
    hist.index = pd.to_datetime(hist.index, utc=True).tz_localize(None).normalize()
    return hist["Close"].dropna()

def preparar_df(y):
    fecha_ref  = pd.to_datetime("2017-01-01")
    dias       = (y.index - fecha_ref).days
    dia_semana = y.index.weekday
    return pd.DataFrame({
        "y"             : y.values,
        "dias_2017"     : dias,
        "dia_sin"       : np.sin(2 * np.pi * dia_semana / 7),
        "dia_cos"       : np.cos(2 * np.pi * dia_semana / 7),
        "retorno_diario": y.pct_change().fillna(0).values,
        "fecha_str"     : y.index.strftime("%Y-%m-%d").to_numpy(),
    }, index=y.index)

def escalar(df_train, df_test):
    cols   = ["y","dias_2017","dia_sin","dia_cos","retorno_diario"]
    scaler = StandardScaler()
    tr_sc  = df_train.copy()
    te_sc  = df_test.copy()
    tr_sc[cols] = scaler.fit_transform(df_train[cols])
    te_sc[cols] = scaler.transform(df_test[cols])
    return tr_sc, te_sc

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

def build_gru(v, nf):
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
    m.compile(optimizer=Adam(0.001), loss="mean_squared_error")
    return m

def entrenar(Xtr, ytr, Xte, yte, nf):
    m = build_gru(VENTANA, nf)
    m.fit(Xtr, ytr, epochs=EPOCHS, batch_size=32,
          validation_split=0.2, verbose=0)
    pred = m.predict(Xte, verbose=0).flatten()
    return pred, mean_absolute_percentage_error(yte, pred)

def señal(pred, actual, umbral=0.02):
    cambio = (pred - actual) / (abs(actual) + 1e-9)
    if   cambio >  umbral: return "COMPRA",  cambio * 100, "#22c55e"
    elif cambio < -umbral: return "VENTA",   cambio * 100, "#ef4444"
    else:                  return "ESPERA",  cambio * 100, "#f59e0b"

def analizar(ticker, nombre, es_cripto=True):
    tipo = "CRIPTO" if es_cripto else "ACCION"
    print(f"\n  [{tipo}] {nombre} ({ticker})")
    try:
        y        = descargar(ticker)
        df       = preparar_df(y)
        df_train = df.loc[TRAIN_START:TRAIN_END]
        df_test  = df.loc[TEST_START:]
        tr_sc, te_sc = escalar(df_train, df_test)

        if len(df_test) < VENTANA + 11:
            print(f"    Datos insuficientes, saltando...")
            return None

        Xtr, ytr_nd, ytr_ow, ytr_tw, _, _, _ = crear_secuencias(tr_sc, VENTANA)
        Xte, yte_nd, yte_ow, yte_tw, d_nd, d_ow, d_tw = crear_secuencias(te_sc, VENTANA)
        nf = Xtr.shape[2]

        print(f"    +1d...", end=" ", flush=True)
        p_nd, mape_nd = entrenar(Xtr, ytr_nd, Xte, yte_nd, nf)
        print(f"{mape_nd*100:.1f}%  +5d...", end=" ", flush=True)
        p_ow, mape_ow = entrenar(Xtr, ytr_ow, Xte, yte_ow, nf)
        print(f"{mape_ow*100:.1f}%  +10d...", end=" ", flush=True)
        p_tw, mape_tw = entrenar(Xtr, ytr_tw, Xte, yte_tw, nf)
        print(f"{mape_tw*100:.1f}%")

        ultimo = yte_nd[-1]
        a1,  c1,  col1  = señal(p_nd[-1], ultimo)
        a5,  c5,  col5  = señal(p_ow[-1], ultimo)
        a10, c10, col10 = señal(p_tw[-1], ultimo)

        precio_usd = float(y.iloc[-1])
        precio_mxn = precio_usd * TIPO_CAMBIO
        consenso   = sum(1 for c in [c1,c5,c10] if c > 0) - sum(1 for c in [c1,c5,c10] if c < 0)
        score      = (1 / (mape_nd + 0.01)) * (1 + consenso * 0.1)

        # Gráfica
        fig, axs = plt.subplots(3, 1, figsize=(12, 9))
        fig.suptitle(f"{nombre} ({ticker})  —  {date.today()}", fontsize=13)
        for ax, (titulo, yte, pred, fechas) in zip(axs, [
            ("+1 día",   yte_nd, p_nd, pd.to_datetime(d_nd)),
            ("+5 días",  yte_ow, p_ow, pd.to_datetime(d_ow)),
            ("+10 días", yte_tw, p_tw, pd.to_datetime(d_tw)),
        ]):
            ax.plot(fechas, yte,  label="Real",       color="black",     lw=1.5)
            ax.plot(fechas, pred, label="Predicción", color="steelblue", lw=1, ls="--")
            ax.set_title(titulo, fontsize=10)
            ax.set_ylabel("Precio escalado")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.tight_layout()
        img_path = os.path.join(OUTPUT_DIR, f"{ticker.replace('-','_')}.png")
        plt.savefig(img_path, dpi=100, bbox_inches="tight")
        plt.close()

        return {
            "ticker"    : ticker,
            "nombre"    : nombre,
            "precio_usd": round(precio_usd, 4),
            "precio_mxn": round(precio_mxn, 2),
            "mape_nd"   : round(mape_nd * 100, 2),
            "señal_1d"  : a1,  "cambio_1d"  : round(c1,  2), "color_1d"  : col1,
            "señal_5d"  : a5,  "cambio_5d"  : round(c5,  2), "color_5d"  : col5,
            "señal_10d" : a10, "cambio_10d" : round(c10, 2), "color_10d" : col10,
            "score"     : round(score, 4),
            "img"       : f"{ticker.replace('-','_')}.png",
            "es_cripto" : es_cripto,
        }
    except Exception as e:
        print(f"    ERROR: {e}")
        return None

# ── LOOP PRINCIPAL ─────────────────────────────────────────
print("\n" + "="*55)
print("  ANALIZANDO CRIPTOMONEDAS")
print("="*55)
resultados_cripto = [r for ticker, nombre in CRIPTOS.items()
                     if (r := analizar(ticker, nombre, True)) is not None]

print("\n" + "="*55)
print("  ANALIZANDO ACCIONES")
print("="*55)
resultados_acciones = [r for ticker, nombre in ACCIONES.items()
                       if (r := analizar(ticker, nombre, False)) is not None]

resultados_cripto.sort(key=lambda x: x["score"], reverse=True)
resultados_acciones.sort(key=lambda x: x["score"], reverse=True)

todos = resultados_cripto + resultados_acciones
with open(os.path.join(OUTPUT_DIR, "señales.json"), "w", encoding="utf-8") as f:
    json.dump(todos, f, ensure_ascii=False, indent=2)

# ── GENERAR HTML ───────────────────────────────────────────
def badge(accion, cambio, color):
    arrow = "▲" if cambio > 0 else ("▼" if cambio < 0 else "●")
    return (f'<span style="background:{color};color:#fff;padding:3px 10px;'
            f'border-radius:6px;font-weight:600;font-size:13px">'
            f'{arrow} {accion} ({cambio:+.1f}%)</span>')

def tabla_html(resultados, titulo, emoji):
    medals = ["🥇","🥈","🥉","4.","5.","6.","7.","8.","9.","10."]
    filas  = ""
    for i, m in enumerate(resultados):
        medal = medals[i] if i < len(medals) else f"{i+1}."
        filas += f"""
        <tr>
          <td style="font-size:20px;text-align:center">{medal}</td>
          <td><strong>{m['nombre']}</strong><br>
              <span style="color:#888;font-size:12px">{m['ticker']}</span></td>
          <td style="text-align:right">
              <strong>${m['precio_usd']:,.4f}</strong><br>
              <span style="color:#888;font-size:12px">${m['precio_mxn']:,.2f} MXN</span>
          </td>
          <td style="text-align:center">{badge(m['señal_1d'],  m['cambio_1d'],  m['color_1d'])}</td>
          <td style="text-align:center">{badge(m['señal_5d'],  m['cambio_5d'],  m['color_5d'])}</td>
          <td style="text-align:center">{badge(m['señal_10d'], m['cambio_10d'], m['color_10d'])}</td>
          <td style="text-align:center">
              <span style="font-size:11px;color:#666">Error pred:</span><br>
              <strong>{m['mape_nd']}%</strong>
          </td>
        </tr>
        <tr>
          <td colspan="7" style="padding:0 0 20px 0;background:#f9f9f9">
            <img src="{m['img']}" style="width:100%;max-width:900px;
                 display:block;margin:8px auto;border-radius:8px">
          </td>
        </tr>"""
    return f"""
    <h2 style="margin-top:36px;font-size:20px;border-left:4px solid #334155;
               padding-left:12px">{emoji} {titulo}</h2>
    <table>
      <thead>
        <tr>
          <th>#</th><th>Nombre</th><th>Precio</th>
          <th>Señal +1 día</th><th>Señal +5 días</th><th>Señal +10 días</th>
          <th>Confianza</th>
        </tr>
      </thead>
      <tbody>{filas}</tbody>
    </table>"""

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reporte Inversiones — {date.today()}</title>
<style>
  body  {{ font-family:system-ui,sans-serif; max-width:1000px;
           margin:0 auto; padding:20px; background:#f4f4f4; color:#222; }}
  h1    {{ font-size:24px; margin-bottom:4px; }}
  .sub  {{ color:#666; font-size:13px; margin-bottom:8px; }}
  table {{ width:100%; border-collapse:collapse; background:#fff;
           border-radius:12px; overflow:hidden;
           box-shadow:0 2px 8px rgba(0,0,0,.08); margin-bottom:32px; }}
  th    {{ background:#1e293b; color:#fff; padding:12px 10px;
           font-size:13px; text-align:center; }}
  td    {{ padding:12px 10px; border-bottom:1px solid #eee;
           vertical-align:middle; }}
  .aviso  {{ background:#fef9c3; border-left:4px solid #ca8a04;
             padding:12px 16px; border-radius:6px; font-size:13px;
             margin-top:28px; }}
  .footer {{ text-align:center; color:#aaa; font-size:12px; margin-top:32px; }}
</style>
</head>
<body>
<h1>📊 Reporte Diario de Inversiones</h1>
<div class="sub">
  Actualizado: {datetime.now().strftime('%d/%m/%Y %H:%M')} UTC &nbsp;|&nbsp;
  Modelo: GRU (ventana={VENTANA} días, epochs={EPOCHS}) &nbsp;|&nbsp;
  Tipo de cambio: ${TIPO_CAMBIO} MXN/USD
</div>
{tabla_html(resultados_cripto,   "Criptomonedas", "🪙")}
{tabla_html(resultados_acciones, "Acciones",      "📈")}
<div class="aviso">
  ⚠️ <strong>Advertencia:</strong> Reporte generado automáticamente con modelos de
  aprendizaje profundo. <strong>No es asesoría financiera.</strong>
  Los mercados son volátiles. Nunca inviertas más de lo que puedes perder.
</div>
<div class="footer">Generado automáticamente con GitHub Actions · GRU · yfinance</div>
</body>
</html>"""

with open(os.path.join(OUTPUT_DIR, "index.html"), "w", encoding="utf-8") as f:
    f.write(html)

print(f"\nREPORTE GENERADO: {OUTPUT_DIR}/index.html")
print(f"\n{'TIPO':<8} {'RANK':<4} {'NOMBRE':<22} {'SEÑAL':>8} {'ERROR':>6}")
print("-"*55)
for i, m in enumerate(resultados_cripto, 1):
    print(f"{'CRIPTO':<8} {i:<4} {m['nombre']:<22} {m['señal_1d']:>8} {m['mape_nd']:>5.1f}%")
for i, m in enumerate(resultados_acciones, 1):
    print(f"{'ACCION':<8} {i:<4} {m['nombre']:<22} {m['señal_1d']:>8} {m['mape_nd']:>5.1f}%")
