from geopy.geocoders import Nominatim
import requests
geolocator = Nominatim(user_agent="weather-app") 
def get_weather_forecast(location: str, date: str):
    """Retrieves the weather using Open-Meteo API for a given location (city) and a date (yyyy-mm-dd). 
    Returns a dictionary with time, temperature, humidity, precipitation, and windspeed for each hour."""
    location = geolocator.geocode(location)
    if location:
        try:
            response = requests.get(
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={location.latitude}&longitude={location.longitude}"
                f"&hourly=temperature_2m,relativehumidity_2m,precipitation,windspeed_10m,weathercode"
                f"&start_date={date}&end_date={date}"
            )
            data = response.json()
            hourly_data = data["hourly"]
            return {
                "time": hourly_data["time"],
                "temperature": hourly_data["temperature_2m"],
                "humidity": hourly_data["relativehumidity_2m"],
                "precipitation": hourly_data["precipitation"],
                "windspeed": hourly_data["windspeed_10m"],
                "weathercode": hourly_data["weathercode"]
            }
        except Exception as e:
            return {"error": str(e)}
    else:
        return {"error": "Location not found"}
if __name__ == "__main__":
    # Example usage
    location = "ho chi minh"
    date = "2025-04-14"
    forecast = get_weather_forecast(location, date)
    print(f"Weather forecast for {location} on {date}: {forecast}")