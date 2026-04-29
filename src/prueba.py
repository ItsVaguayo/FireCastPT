import requests

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 38.72,
    "longitude": -9.14,
    "start_date": "2023-01-01",
    "end_date": "2023-01-02",
    "daily": ",".join([
        "temperature_2m_max",
        "temperature_2m_min",
        "relative_humidity_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
        "shortwave_radiation_sum",
        "et0_fao_evapotranspiration"
    ]),
    "timezone": "Europe/Lisbon"
}

respuesta = requests.get(url, params=params)
datos = respuesta.json()
print(type(datos))
print(datos["daily"].keys())
print(datos["daily"])
print("Número de días:", len(datos["daily"]["time"]))