from .base import create_function_tool
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Get current weather for a location"""
    return {
        "temperature": 25,
        "weather_description": "Sunny",
        "humidity": 60,
        "wind_speed": 10
    }
get_weather_tool = create_function_tool(
            get_weather,
            name="get_weather",
            description="Get current weather information for a location"
        )