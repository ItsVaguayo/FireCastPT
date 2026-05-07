# =============================================================================
# FireCast PT — Generador de negativos sintéticos (FWI encadenado)
# =============================================================================
# 
# Instalación de dependencias:
#   pip install pandas openpyxl requests tqdm
# =============================================================================

import math
import random
import time
from datetime import date, timedelta

import pandas as pd
import requests
from tqdm import tqdm

random.seed(42)

# =============================================================================
# PARÁMETROS
# =============================================================================

ARCHIVO_EXCEL  = "Registos_Incendios_SGIF_2021_2025.xlsx"
ARCHIVO_SALIDA = "firecast_pt_dataset.csv"
MAX_FILAS      = 500    # 0 = todo el dataset
PAUSA_API      = 0.1    # segundos entre llamadas a Open-Meteo
DIAS_PREVIOS   = 30     # días de histórico para encadenar el FWI

# =============================================================================
# TABLAS FWI
# =============================================================================

LE = {1:6.5, 2:7.5, 3:9.0, 4:12.8, 5:13.9, 6:13.9,
      7:12.4, 8:10.9, 9:9.4, 10:8.0, 11:7.0, 12:6.0}

LF = {1:-1.6, 2:-1.6, 3:-1.6, 4:0.9, 5:3.8, 6:5.8,
      7:6.4,  8:5.0,  9:2.4, 10:0.4, 11:-1.6, 12:-1.6}

# =============================================================================
# FÓRMULAS FWI (sin cambios respecto a la versión anterior)
# =============================================================================

def calcular_fwi(temp, rh, ws, pr, mes, ffmc0=85.0, dmc0=6.0, dc0=15.0):
    """
    Calcula todos los índices FWI para UN día.
    Los parámetros ffmc0, dmc0, dc0 son los valores del día anterior.
    """
    rh   = max(1.0, min(rh, 99.9))
    temp = max(-40.0, temp)
    ws   = max(0.0, ws)
    pr   = max(0.0, pr)

    # --- FFMC ---
    mo = 147.2 * (101 - ffmc0) / (59.5 + ffmc0)
    if pr > 0.5:
        rf = pr - 0.5
        if mo > 150:
            mo += 42.91 * math.exp(-0.034 * rf) - 4.0 * rf + 0.18 * (21.1 - temp) * (1 - math.exp(-0.115 * rh))
        else:
            mo += 42.91 * math.exp(-0.034 * rf) - 4.0 * rf
        mo = min(mo, 250.0)
    ed = 0.942 * rh**0.679 + 11 * math.exp((rh-100)/10) + 0.18 * (21.1-temp) * (1-math.exp(-0.115*rh))
    ew = 0.618 * rh**0.753 + 10 * math.exp((rh-100)/10) + 0.18 * (21.1-temp) * (1-math.exp(-0.115*rh))
    if mo > ed:
        kd = (0.424*(1-(rh/100)**1.7) + 0.0694*ws**0.5*(1-(rh/100)**8)) * 0.581 * math.exp(0.0365*temp)
        m  = ed + (mo - ed) * 10**(-kd)
    elif mo < ew:
        kw = (0.424*(1-(100-rh)/100)**1.7 + 0.0694*ws**0.5*(1-((100-rh)/100)**8)) * 0.581 * math.exp(0.0365*temp)
        m  = ew - (ew - mo) * 10**(-kw)
    else:
        m = mo
    ffmc = max(0.0, min(59.5 * (250-m) / (147.2+m), 101.0))

    # --- ISI ---
    mo2 = 147.2 * (101-ffmc) / (59.5+ffmc)
    ff  = 91.9 * math.exp(-0.1386*mo2) * (1 + mo2**5.31 / 4.93e7)
    isi = 0.208 * math.exp(0.05039*ws) * ff

    # --- DMC ---
    if pr > 1.5:
        re   = 0.92 * pr - 1.27
        mo_d = 20 + math.exp(5.6348 - dmc0/43.43)
        b    = 100/(0.5+0.3*dmc0) if dmc0 <= 33 else (14-1.3*math.log(dmc0) if dmc0 <= 65 else 6.2*math.log(dmc0)-17.2)
        mr   = mo_d + 1000*re/(48.77+b*re)
        dmc0 = max(0.0, 244.72 - 43.43*math.log(mr-20))
    dmc = dmc0 + 100 * 1.894*(temp+1.1)*(100-rh)*LE.get(mes,9.0)*1e-6 if temp > -1.1 else dmc0

    # --- DC ---
    if pr > 2.8:
        rd   = 0.83 * pr - 1.27
        qr   = 800 * math.exp(-dc0/400) + 3.937*rd
        dc0  = max(0.0, 400*math.log(800/qr)) if qr > 0 else dc0
    v  = 0.36*(temp+2.8) + LF.get(mes, 3.8)
    dc = dc0 + 0.5*v if v > 0 else dc0

    # --- BUI ---
    if dmc == 0:
        bui = 0.0
    elif dmc <= 0.4*dc:
        bui = 0.8*dmc*dc / (dmc+0.4*dc)
    else:
        bui = dmc - (1 - 0.8*dc/(dmc+0.4*dc)) * (0.92+(0.0114*dmc)**1.7)
    bui = max(0.0, bui)

    # --- FWI ---
    fd  = 0.626*bui**0.809 + 2.0 if bui <= 80 else 1000/(25+108.64*math.exp(-0.023*bui))
    b   = 0.1*isi*fd
    fwi = math.exp(2.72*(0.434*math.log(b))**0.647) if b > 1 else b

    # --- DSR ---
    dsr = 0.0272 * fwi**1.77

    return {
        "FFMC": round(ffmc, 4),
        "ISI":  round(isi,  4),
        "DMC":  round(dmc,  4),
        "DC":   round(dc,   4),
        "BUI":  round(bui,  4),
        "FWI":  round(fwi,  4),
        "DSR":  round(dsr,  6),
    }

# =============================================================================
# OPEN-METEO — ahora acepta un rango de fechas
# =============================================================================

def obtener_meteo_rango(lat, lon, fecha_inicio, fecha_fin):
    """
    Descarga datos horarios de Open-Meteo para un rango de fechas.
    Devuelve una lista de dicts, uno por día, con {temp, rh, ws, pr}.
    Si hay error devuelve None.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "start_date":      fecha_inicio.isoformat(),
        "end_date":        fecha_fin.isoformat(),
        "hourly":          "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "timezone":        "Europe/Lisbon",
        "wind_speed_unit": "kmh",
    }
    try:
        respuesta = requests.get(url, params=params, timeout=20)
        respuesta.raise_for_status()
        h = respuesta.json()["hourly"]

        # Open-Meteo devuelve listas de 24 valores por día
        # Agrupamos en bloques de 24 horas
        n_dias = (fecha_fin - fecha_inicio).days + 1
        dias   = []

        for d in range(n_dias):
            inicio = d * 24
            fin    = inicio + 24

            temp_dia = h["temperature_2m"][inicio:fin]
            rh_dia   = h["relative_humidity_2m"][inicio:fin]
            ws_dia   = h["wind_speed_10m"][inicio:fin]
            pr_dia   = h["precipitation"][inicio:fin]

            dias.append({
                "temp": temp_dia[12] if temp_dia[12] is not None else 15.0,  # hora 12h
                "rh":   rh_dia[12]   if rh_dia[12]   is not None else 50.0,
                "ws":   ws_dia[12]   if ws_dia[12]    is not None else 5.0,
                "pr":   sum(v for v in pr_dia if v is not None),  # total del día
                "mes":  (fecha_inicio + timedelta(days=d)).month,
            })

        return dias

    except Exception as e:
        print(f"  ⚠ Error Open-Meteo ({lat:.3f}, {lon:.3f}, {fecha_inicio}→{fecha_fin}): {e}")
        return None


def calcular_fwi_encadenado(lat, lon, fecha_objetivo):
    """
    Descarga los DIAS_PREVIOS días anteriores a fecha_objetivo más el propio día,
    y calcula el FWI en cadena pasando el resultado de cada día al siguiente.

    Devuelve los índices FWI del día objetivo y los datos meteorológicos,
    o None si hay error en la API.
    """
    fecha_inicio = fecha_objetivo - timedelta(days=DIAS_PREVIOS)
    fecha_fin    = fecha_objetivo

    dias = obtener_meteo_rango(lat, lon, fecha_inicio, fecha_fin)
    if dias is None:
        return None, None

    # Valores iniciales estándar — solo se usan el primer día del histórico
    ffmc = 85.0
    dmc  = 6.0
    dc   = 15.0

    resultado = None

    # Recorremos todos los días en orden, encadenando el FWI
    for dia in dias:
        resultado = calcular_fwi(
            temp  = dia["temp"],
            rh    = dia["rh"],
            ws    = dia["ws"],
            pr    = dia["pr"],
            mes   = dia["mes"],
            ffmc0 = ffmc,   # ← resultado del día anterior
            dmc0  = dmc,
            dc0   = dc,
        )
        # El resultado de hoy es el punto de partida de mañana
        ffmc = resultado["FFMC"]
        dmc  = resultado["DMC"]
        dc   = resultado["DC"]

    # El último resultado es el del día objetivo
    meteo_objetivo = dias[-1]
    return resultado, meteo_objetivo

# =============================================================================
# SELECCIÓN DE FECHA NEGATIVA (sin cambios)
# =============================================================================

def buscar_fecha_negativa(fecha_incendio, lat, lon, fire_set, indice):
    """
    Busca una fecha sin incendio para la misma ubicación.
    Mismo mes, año anterior o posterior, día ±7-21 días.
    """
    año_negativo = fecha_incendio.year - 1 if indice % 2 == 0 else fecha_incendio.year + 1

    for _ in range(15):
        desplazamiento = random.randint(7, 21) * random.choice([-1, 1])
        try:
            nueva_fecha = date(año_negativo, fecha_incendio.month, fecha_incendio.day) + timedelta(days=desplazamiento)
        except ValueError:
            nueva_fecha = date(año_negativo, fecha_incendio.month, 15) + timedelta(days=desplazamiento)

        if nueva_fecha.month != fecha_incendio.month:
            continue

        clave = (round(lat, 3), round(lon, 3), nueva_fecha.year, nueva_fecha.month, nueva_fecha.day)
        if clave not in fire_set and 2018 <= nueva_fecha.year <= 2025:
            return nueva_fecha

    return None

# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

print("Cargando Excel...")
df = pd.read_excel(ARCHIVO_EXCEL, sheet_name="SGIF_2021_2025")
df = df.dropna(subset=["Latitude", "Longitude", "FWI"])
df["fecha"] = pd.to_datetime(df["DataHoraAlerta"]).dt.date

if MAX_FILAS > 0:
    df = df.head(MAX_FILAS)
    print(f"Modo test: procesando solo {MAX_FILAS} filas")

print(f"Incendios a procesar: {len(df)}")

fire_set = set(
    zip(df["Latitude"].round(3), df["Longitude"].round(3),
        pd.to_datetime(df["DataHoraAlerta"]).dt.year,
        pd.to_datetime(df["DataHoraAlerta"]).dt.month,
        pd.to_datetime(df["DataHoraAlerta"]).dt.day)
)

filas = []

for i, (_, fila) in enumerate(tqdm(df.iterrows(), total=len(df))):

    lat       = fila["Latitude"]
    lon       = fila["Longitude"]
    fecha_pos = fila["fecha"]
    mes       = int(fila["Mes"])

    # ------------------------------------------------------------------
    # POSITIVO: FWI encadenado del día del incendio
    # ------------------------------------------------------------------
    indices_pos, meteo_pos = calcular_fwi_encadenado(lat, lon, fecha_pos)

    if indices_pos is None:
        continue

    filas.append({
        "Latitude":  lat,
        "Longitude": lon,
        "Distrito":  fila["Distrito"],
        "Concelho":  fila["Concelho"],
        "Mes":       mes,
        "Dia":       int(fila["Dia"]),
        "temp_12h":  round(meteo_pos["temp"], 2),
        "rh_12h":    round(meteo_pos["rh"],   2),
        "ws_12h":    round(meteo_pos["ws"],   2),
        "pr_dia":    round(meteo_pos["pr"],   2),
        **indices_pos,
        "incendio":  1,
    })

    time.sleep(PAUSA_API)

    # ------------------------------------------------------------------
    # NEGATIVO: FWI encadenado de la fecha sin incendio
    # ------------------------------------------------------------------
    fecha_neg = buscar_fecha_negativa(fecha_pos, lat, lon, fire_set, i)
    if fecha_neg is None:
        continue

    indices_neg, meteo_neg = calcular_fwi_encadenado(lat, lon, fecha_neg)
    if indices_neg is None:
        continue

    filas.append({
        "Latitude":  lat,
        "Longitude": lon,
        "Distrito":  fila["Distrito"],
        "Concelho":  fila["Concelho"],
        "Mes":       fecha_neg.month,
        "Dia":       fecha_neg.day,
        "temp_12h":  round(meteo_neg["temp"], 2),
        "rh_12h":    round(meteo_neg["rh"],   2),
        "ws_12h":    round(meteo_neg["ws"],   2),
        "pr_dia":    round(meteo_neg["pr"],   2),
        **indices_neg,
        "incendio":  0,
    })

    time.sleep(PAUSA_API)

# Mezclar y guardar
resultado = pd.DataFrame(filas).sample(frac=1, random_state=42).reset_index(drop=True)
resultado.to_csv(ARCHIVO_SALIDA, index=False, encoding="utf-8-sig")

print(f"\nDataset guardado en: {ARCHIVO_SALIDA}")
print(f"  Total filas  : {len(resultado)}")
print(f"  Incendios    : {resultado['incendio'].sum()}")
print(f"  No incendios : {(resultado['incendio'] == 0).sum()}")
print(f"  FWI medio positivos : {resultado[resultado.incendio==1]['FWI'].mean():.2f}")
print(f"  FWI medio negativos : {resultado[resultado.incendio==0]['FWI'].mean():.2f}")