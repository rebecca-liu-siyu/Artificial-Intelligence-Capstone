import json
import csv

with open("weather_ori.json", "r", encoding="utf-8") as f:
    data = json.load(f)

locations = data["records"]["location"]

csv_columns = [
    "StationID", "StationName", "StationNameEN", "StationAttribute",
    "DateTime", "AirPressure", "AirTemperature", "RelativeHumidity",
    "WindSpeed", "WindDirection", "Precipitation", "SunshineDuration"
]

csv_data = []

for location in locations:
    station_info = location["station"]
    station_id = station_info["StationID"]
    station_name = station_info["StationName"]
    station_name_en = station_info["StationNameEN"]
    station_attr = station_info["StationAttribute"]

    for obs in location["stationObsTimes"]["stationObsTime"]:
        weather = obs["weatherElements"]
        row = {
            "StationID": station_id,
            "StationName": station_name,
            "StationNameEN": station_name_en,
            "StationAttribute": station_attr,
            "DateTime": obs["DateTime"],
            "AirPressure": weather["AirPressure"],
            "AirTemperature": weather["AirTemperature"],
            "RelativeHumidity": weather["RelativeHumidity"],
            "WindSpeed": weather["WindSpeed"],
            "WindDirection": weather["WindDirection"],
            "Precipitation": weather["Precipitation"],
            "SunshineDuration": weather["SunshineDuration"]
        }
        csv_data.append(row)

csv_filename = "weather.csv"
with open(csv_filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_columns)
    writer.writeheader()
    writer.writerows(csv_data)

print(f"sucessful")
