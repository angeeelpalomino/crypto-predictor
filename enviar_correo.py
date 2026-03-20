"""
Envía el reporte HTML por correo automáticamente.
Se ejecuta después de analizar_cryptos.py
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import date

# ─────────────────────────────────────────────────────────────
#  CONFIGURACIÓN — viene de los secrets de GitHub
# ─────────────────────────────────────────────────────────────
GMAIL_USER    = os.environ["GMAIL_USER"]
GMAIL_PASS    = os.environ["GMAIL_PASSWORD"]
DESTINATARIO  = os.environ.get("DESTINATARIO", GMAIL_USER)
REPORTE_PATH  = "reporte/index.html"

# ─────────────────────────────────────────────────────────────
#  LEER EL REPORTE GENERADO
# ─────────────────────────────────────────────────────────────
if not os.path.exists(REPORTE_PATH):
    raise FileNotFoundError(f"No se encontró el reporte en {REPORTE_PATH}")

with open(REPORTE_PATH, "r", encoding="utf-8") as f:
    html_content = f.read()

# ─────────────────────────────────────────────────────────────
#  CONSTRUIR EL CORREO
# ─────────────────────────────────────────────────────────────
msg = MIMEMultipart("alternative")
msg["Subject"] = f"📊 Señales Cripto — {date.today().strftime('%d/%m/%Y')}"
msg["From"]    = GMAIL_USER
msg["To"]      = DESTINATARIO

# Texto plano por si el HTML no carga
texto_plano = f"""
Reporte de Criptomonedas — {date.today().strftime('%d/%m/%Y')}

El reporte HTML está adjunto. Ábrelo en tu navegador para ver
las señales de inversión del día con gráficas completas.

Generado automáticamente con GitHub Actions.
"""

# Adjuntar versión HTML (se ve directo en Gmail)
msg.attach(MIMEText(texto_plano, "plain"))
msg.attach(MIMEText(html_content, "html"))

# También adjuntar el HTML como archivo descargable
with open(REPORTE_PATH, "rb") as f:
    adjunto = MIMEBase("application", "octet-stream")
    adjunto.set_payload(f.read())
    encoders.encode_base64(adjunto)
    adjunto.add_header(
        "Content-Disposition",
        f"attachment; filename=reporte_cripto_{date.today()}.html"
    )
    msg.attach(adjunto)

# ─────────────────────────────────────────────────────────────
#  ENVIAR
# ─────────────────────────────────────────────────────────────
print(f"Enviando reporte a {DESTINATARIO}...")

try:
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_USER, GMAIL_PASS)
        server.sendmail(GMAIL_USER, DESTINATARIO, msg.as_string())
    print(f"Correo enviado exitosamente a {DESTINATARIO}")
except Exception as e:
    print(f"Error al enviar correo: {e}")
    raise
