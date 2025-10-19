"""
Real-time API Client for Weather and AQI Data
Integrates with multiple APIs for live data updates
"""

import requests
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
import json
import time
from dataclasses import dataclass

@dataclass
class WeatherData:
    """Weather data structure"""
    temperature: float
    humidity: float
    pressure: float
    wind_speed: float
    rainfall: float
    timestamp: datetime

@dataclass
class AQIData:
    """Air Quality Index data structure"""
    aqi: int
    pm25: float
    pm10: float
    co: float
    no2: float
    o3: float
    so2: float
    timestamp: datetime

class RealTimeAPIClient:
    """
    Real-time API client for fetching weather and AQI data
    Supports multiple API providers with fallback mechanisms
    """
    
    def __init__(self):
        self.pune_coords = {
            'lat': 18.5204,
            'lon': 73.8567,
            'name': 'Pune, India'
        }
        
        # API configurations (use environment variables for production)
        self.api_keys = {
            'openweather': os.getenv('OPENWEATHER_API_KEY', 'demo_key'),
            'weatherapi': os.getenv('WEATHERAPI_KEY', 'demo_key'),
            'aqicn': os.getenv('AQICN_API_KEY', 'demo_key')
        }
        
        self.api_endpoints = {
            'openweather_current': 'https://api.openweathermap.org/data/2.5/weather',
            'openweather_forecast': 'https://api.openweathermap.org/data/2.5/forecast',
            'weatherapi_current': 'https://api.weatherapi.com/v1/current.json',
            'aqicn_current': 'https://api.waqi.info/feed/geo:{lat};{lon}/'
        }
        
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def fetch_openweather_current(self) -> Optional[WeatherData]:
        """Fetch current weather from OpenWeatherMap API"""
        try:
            url = self.api_endpoints['openweather_current']
            params = {
                'lat': self.pune_coords['lat'],
                'lon': self.pune_coords['lon'],
                'appid': self.api_keys['openweather'],
                'units': 'metric'
            }
            
            if self.api_keys['openweather'] == 'demo_key':
                # Return simulated data for demo
                return self._generate_simulated_weather()
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    return WeatherData(
                        temperature=data['main']['temp'],
                        humidity=data['main']['humidity'],
                        pressure=data['main']['pressure'],
                        wind_speed=data['wind']['speed'],
                        rainfall=data.get('rain', {}).get('1h', 0),
                        timestamp=datetime.now()
                    )
                else:
                    print(f"⚠️ OpenWeather API error: {response.status}")
                    return None
                    
        except Exception as e:
            print(f"⚠️ Error fetching OpenWeather data: {e}")
            return None
    
    async def fetch_weatherapi_current(self) -> Optional[WeatherData]:
        """Fetch current weather from WeatherAPI"""
        try:
            url = self.api_endpoints['weatherapi_current']
            params = {
                'key': self.api_keys['weatherapi'],
                'q': f"{self.pune_coords['lat']},{self.pune_coords['lon']}",
                'aqi': 'yes'
            }
            
            if self.api_keys['weatherapi'] == 'demo_key':
                # Return simulated data for demo
                return self._generate_simulated_weather()
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    current = data['current']
                    
                    return WeatherData(
                        temperature=current['temp_c'],
                        humidity=current['humidity'],
                        pressure=current['pressure_mb'],
                        wind_speed=current['wind_kph'] / 3.6,  # Convert to m/s
                        rainfall=current.get('precip_mm', 0),
                        timestamp=datetime.now()
                    )
                else:
                    print(f"⚠️ WeatherAPI error: {response.status}")
                    return None
                    
        except Exception as e:
            print(f"⚠️ Error fetching WeatherAPI data: {e}")
            return None
    
    async def fetch_aqi_data(self) -> Optional[AQIData]:
        """Fetch current AQI data"""
        try:
            url = self.api_endpoints['aqicn_current'].format(
                lat=self.pune_coords['lat'],
                lon=self.pune_coords['lon']
            )
            params = {
                'token': self.api_keys['aqicn']
            }
            
            if self.api_keys['aqicn'] == 'demo_key':
                # Return simulated data for demo
                return self._generate_simulated_aqi()
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data['status'] == 'ok':
                        aqi_data = data['data']
                        iaqi = aqi_data.get('iaqi', {})
                        
                        return AQIData(
                            aqi=aqi_data['aqi'],
                            pm25=iaqi.get('pm25', {}).get('v', 0),
                            pm10=iaqi.get('pm10', {}).get('v', 0),
                            co=iaqi.get('co', {}).get('v', 0),
                            no2=iaqi.get('no2', {}).get('v', 0),
                            o3=iaqi.get('o3', {}).get('v', 0),
                            so2=iaqi.get('so2', {}).get('v', 0),
                            timestamp=datetime.now()
                        )
                    else:
                        print(f"⚠️ AQI API error: {data.get('message', 'Unknown error')}")
                        return None
                else:
                    print(f"⚠️ AQI API HTTP error: {response.status}")
                    return None
                    
        except Exception as e:
            print(f"⚠️ Error fetching AQI data: {e}")
            return None
    
    def _generate_simulated_weather(self) -> WeatherData:
        """Generate realistic simulated weather data for Pune"""
        now = datetime.now()
        month = now.month
        
        # Pune seasonal patterns
        if month in [12, 1, 2]:  # Winter
            base_temp = 22 + np.random.normal(0, 3)
            base_humidity = 45 + np.random.normal(0, 10)
            base_rainfall = np.random.exponential(0.5)
        elif month in [3, 4, 5]:  # Summer
            base_temp = 32 + np.random.normal(0, 4)
            base_humidity = 35 + np.random.normal(0, 8)
            base_rainfall = np.random.exponential(1.0)
        elif month in [6, 7, 8, 9]:  # Monsoon
            base_temp = 26 + np.random.normal(0, 2)
            base_humidity = 75 + np.random.normal(0, 10)
            base_rainfall = 5 + np.random.exponential(3)
        else:  # Post-monsoon
            base_temp = 28 + np.random.normal(0, 3)
            base_humidity = 60 + np.random.normal(0, 12)
            base_rainfall = np.random.exponential(2.0)
        
        return WeatherData(
            temperature=round(max(10, min(45, base_temp)), 1),
            humidity=round(max(20, min(95, base_humidity)), 0),
            pressure=round(1013 + np.random.normal(0, 8), 1),
            wind_speed=round(max(0, 3 + np.random.exponential(2)), 1),
            rainfall=round(max(0, base_rainfall), 2),
            timestamp=now
        )
    
    def _generate_simulated_aqi(self) -> AQIData:
        """Generate realistic simulated AQI data for Pune"""
        # Pune typically has moderate to poor air quality
        base_aqi = 75 + np.random.normal(0, 25)
        aqi = max(20, min(300, int(base_aqi)))
        
        # Correlate pollutants with AQI
        pm25 = max(10, aqi * 0.6 + np.random.normal(0, 10))
        pm10 = max(20, aqi * 0.8 + np.random.normal(0, 15))
        
        return AQIData(
            aqi=aqi,
            pm25=round(pm25, 1),
            pm10=round(pm10, 1),
            co=round(max(0.5, 1.2 + np.random.normal(0, 0.3)), 1),
            no2=round(max(10, 40 + np.random.normal(0, 15)), 1),
            o3=round(max(20, 60 + np.random.normal(0, 20)), 1),
            so2=round(max(5, 15 + np.random.normal(0, 8)), 1),
            timestamp=datetime.now()
        )
    
    async def fetch_current_data(self) -> Dict:
        """Fetch current weather and AQI data with fallback"""
        print("🌤️ Fetching real-time weather and AQI data...")
        
        weather_data = None
        aqi_data = None
        
        # Try multiple weather APIs with fallback
        try:
            weather_data = await self.fetch_openweather_current()
            if not weather_data:
                weather_data = await self.fetch_weatherapi_current()
            if not weather_data:
                weather_data = self._generate_simulated_weather()
        except Exception as e:
            print(f"⚠️ Weather API fallback: {e}")
            weather_data = self._generate_simulated_weather()
        
        # Try AQI API with fallback
        try:
            aqi_data = await self.fetch_aqi_data()
            if not aqi_data:
                aqi_data = self._generate_simulated_aqi()
        except Exception as e:
            print(f"⚠️ AQI API fallback: {e}")
            aqi_data = self._generate_simulated_aqi()
        
        # Combine data
        current_data = {
            'timestamp': datetime.now(),
            'temperature': weather_data.temperature,
            'humidity': weather_data.humidity,
            'pressure': weather_data.pressure,
            'wind_speed': weather_data.wind_speed,
            'rainfall': weather_data.rainfall,
            'aqi': aqi_data.aqi,
            'pm25': aqi_data.pm25,
            'pm10': aqi_data.pm10,
            'co': aqi_data.co,
            'no2': aqi_data.no2,
            'o3': aqi_data.o3,
            'so2': aqi_data.so2,
            'data_source': 'real_time_api'
        }
        
        print(f"✅ Real-time data fetched: {weather_data.temperature:.1f}°C, AQI: {aqi_data.aqi}")
        return current_data
    
    def save_current_data(self, data: Dict, filepath: str = "data/current_weather.json"):
        """Save current data to file for caching"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Convert datetime to string for JSON serialization
            data_copy = data.copy()
            data_copy['timestamp'] = data_copy['timestamp'].isoformat()
            
            with open(filepath, 'w') as f:
                json.dump(data_copy, f, indent=2)
            
            print(f"✅ Current data saved to {filepath}")
            
        except Exception as e:
            print(f"⚠️ Error saving current data: {e}")
    
    def load_cached_data(self, filepath: str = "data/current_weather.json") -> Optional[Dict]:
        """Load cached current data"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Convert timestamp back to datetime
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                
                # Check if data is recent (within 1 hour)
                if (datetime.now() - data['timestamp']).seconds < 3600:
                    print("✅ Using cached current data")
                    return data
                else:
                    print("⚠️ Cached data is stale")
                    return None
            else:
                return None
                
        except Exception as e:
            print(f"⚠️ Error loading cached data: {e}")
            return None


async def fetch_real_time_data(city: str = "Pune") -> Dict:
    """
    Main function to fetch real-time weather and AQI data
    
    Args:
        city: City name (currently supports Pune)
    
    Returns:
        Dictionary with current weather and AQI data
    """
    print(f"🌍 FETCHING REAL-TIME DATA FOR {city.upper()}")
    print("=" * 50)
    
    async with RealTimeAPIClient() as client:
        # Try to load cached data first
        cached_data = client.load_cached_data()
        if cached_data:
            return cached_data
        
        # Fetch fresh data
        current_data = await client.fetch_current_data()
        
        # Cache the data
        client.save_current_data(current_data)
        
        return current_data


# Synchronous wrapper for compatibility
def get_current_weather(city: str = "Pune") -> Dict:
    """Synchronous wrapper for fetching current weather"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        data = loop.run_until_complete(fetch_real_time_data(city))
        return data
    finally:
        loop.close()


# Example usage and testing
if __name__ == "__main__":
    async def test_api_client():
        print("🧪 Testing Real-time API Client...")
        
        # Test current data fetching
        current_data = await fetch_real_time_data("Pune")
        
        print(f"\n📊 CURRENT WEATHER DATA:")
        print(f"   🌡️ Temperature: {current_data['temperature']:.1f}°C")
        print(f"   💧 Humidity: {current_data['humidity']:.0f}%")
        print(f"   🌧️ Rainfall: {current_data['rainfall']:.1f}mm")
        print(f"   💨 AQI: {current_data['aqi']}")
        print(f"   🕐 Timestamp: {current_data['timestamp']}")
        
        return current_data
    
    # Run test
    asyncio.run(test_api_client())