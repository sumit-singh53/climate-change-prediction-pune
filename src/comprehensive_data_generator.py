#!/usr/bin/env python3
"""
Comprehensive Climate Data Generator
Generates realistic historical climate data for Pune from 2000 to current date
Based on actual climate patterns and trends for Maharashtra region
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple
import math

from config import DATABASE_CONFIG, PUNE_LOCATIONS

class PuneClimateDataGenerator:
    """Generate realistic climate data based on Pune's actual climate patterns"""
    
    def __init__(self):
        self.db_path = DATABASE_CONFIG["sqlite_path"]
        
        # Pune climate characteristics (based on historical data)
        self.climate_params = {
            'temperature': {
                'winter': {'mean': 22, 'std': 4, 'min': 10, 'max': 32},
                'summer': {'mean': 32, 'std': 5, 'min': 20, 'max': 42},
                'monsoon': {'mean': 26, 'std': 3, 'min': 18, 'max': 35},
                'post_monsoon': {'mean': 25, 'std': 4, 'min': 15, 'max': 35}
            },
            'humidity': {
                'winter': {'mean': 55, 'std': 15, 'min': 25, 'max': 85},
                'summer': {'mean': 45, 'std': 12, 'min': 20, 'max': 75},
                'monsoon': {'mean': 85, 'std': 10, 'min': 65, 'max': 95},
                'post_monsoon': {'mean': 70, 'std': 12, 'min': 45, 'max': 90}
            },
            'rainfall': {
                'winter': {'mean': 2, 'std': 5, 'max_daily': 25},
                'summer': {'mean': 5, 'std': 10, 'max_daily': 50},
                'monsoon': {'mean': 8, 'std': 15, 'max_daily': 150},
                'post_monsoon': {'mean': 3, 'std': 8, 'max_daily': 40}
            },
            'pressure': {
                'mean': 1013, 'std': 8, 'min': 995, 'max': 1030
            },
            'wind_speed': {
                'winter': {'mean': 8, 'std': 4, 'min': 2, 'max': 25},
                'summer': {'mean': 12, 'std': 6, 'min': 3, 'max': 35},
                'monsoon': {'mean': 15, 'std': 8, 'min': 5, 'max': 45},
                'post_monsoon': {'mean': 10, 'std': 5, 'min': 3, 'max': 30}
            },
            'aqi': {
                'winter': {'mean': 120, 'std': 40, 'min': 50, 'max': 300},
                'summer': {'mean': 150, 'std': 50, 'min': 60, 'max': 400},
                'monsoon': {'mean': 80, 'std': 30, 'min': 30, 'max': 200},
                'post_monsoon': {'mean': 110, 'std': 35, 'min': 40, 'max': 250}
            }
        }
        
        # Climate change trends (per decade)
        self.climate_trends = {
            'temperature': 0.5,  # +0.5°C per decade
            'humidity': -2,      # -2% per decade (due to urbanization)
            'rainfall': -5,      # -5% per decade (changing patterns)
            'aqi': 15,          # +15 AQI points per decade (urbanization)
            'wind_speed': -0.5   # -0.5 km/h per decade
        }
    
    def get_season(self, month: int) -> str:
        """Get season based on month for Pune climate"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'summer'
        elif month in [6, 7, 8, 9]:
            return 'monsoon'
        else:  # 10, 11
            return 'post_monsoon'
    
    def apply_climate_trend(self, base_value: float, year: int, variable: str) -> float:
        """Apply climate change trend to base value"""
        years_since_2000 = year - 2000
        decades = years_since_2000 / 10.0
        
        if variable in self.climate_trends:
            trend = self.climate_trends[variable] * decades
            if variable == 'rainfall':
                # Rainfall trend is percentage-based
                return base_value * (1 + trend / 100)
            else:
                return base_value + trend
        return base_value
    
    def generate_daily_weather(self, date: datetime, location_id: str) -> Dict:
        """Generate realistic daily weather data"""
        year = date.year
        month = date.month
        day = date.day
        season = self.get_season(month)
        
        # Base temperature with seasonal variation
        temp_params = self.climate_params['temperature'][season]
        base_temp = np.random.normal(temp_params['mean'], temp_params['std'])
        
        # Add daily variation (sinusoidal pattern)
        day_of_year = date.timetuple().tm_yday
        seasonal_variation = 3 * math.sin(2 * math.pi * day_of_year / 365.25)
        base_temp += seasonal_variation
        
        # Apply climate change trend
        temperature = self.apply_climate_trend(base_temp, year, 'temperature')
        temperature = max(temp_params['min'], min(temp_params['max'], temperature))
        
        # Humidity (inversely correlated with temperature)
        humid_params = self.climate_params['humidity'][season]
        base_humidity = np.random.normal(humid_params['mean'], humid_params['std'])
        # Temperature-humidity correlation
        temp_effect = (temperature - temp_params['mean']) * -1.5
        humidity = base_humidity + temp_effect
        humidity = self.apply_climate_trend(humidity, year, 'humidity')
        humidity = max(humid_params['min'], min(humid_params['max'], humidity))
        
        # Rainfall (seasonal patterns)
        rain_params = self.climate_params['rainfall'][season]
        if season == 'monsoon':
            # Higher probability of rain in monsoon
            rain_prob = 0.7
        elif season == 'winter':
            rain_prob = 0.1
        else:
            rain_prob = 0.3
        
        if np.random.random() < rain_prob:
            rainfall = max(0, np.random.exponential(rain_params['mean']))
            rainfall = min(rainfall, rain_params['max_daily'])
        else:
            rainfall = 0
        
        rainfall = self.apply_climate_trend(rainfall, year, 'rainfall')
        rainfall = max(0, rainfall)
        
        # Pressure (relatively stable with small variations)
        pressure_params = self.climate_params['pressure']
        pressure = np.random.normal(pressure_params['mean'], pressure_params['std'])
        pressure = max(pressure_params['min'], min(pressure_params['max'], pressure))
        
        # Wind speed (seasonal variation)
        wind_params = self.climate_params['wind_speed'][season]
        wind_speed = np.random.normal(wind_params['mean'], wind_params['std'])
        wind_speed = self.apply_climate_trend(wind_speed, year, 'wind_speed')
        wind_speed = max(wind_params['min'], min(wind_params['max'], wind_speed))
        
        # Wind direction (random but with seasonal preferences)
        if season == 'monsoon':
            # Westerly winds during monsoon
            wind_direction = np.random.normal(270, 45) % 360
        else:
            wind_direction = np.random.uniform(0, 360)
        
        return {
            'timestamp': date.isoformat(),
            'location_id': location_id,
            'temperature': round(temperature, 1),
            'humidity': round(humidity, 1),
            'pressure': round(pressure, 1),
            'wind_speed': round(wind_speed, 1),
            'wind_direction': round(wind_direction, 1),
            'precipitation': round(rainfall, 1),
            'solar_radiation': round(max(0, 20 - humidity/5 + temperature/2), 1),
            'uv_index': round(max(0, min(11, (temperature - 15) / 3)), 1),
            'visibility': round(max(1, 15 - humidity/10), 1),
            'cloud_cover': round(min(100, humidity + rainfall * 5), 1),
            'data_source': 'generated_historical',
            'quality_score': 0.95
        }
    
    def generate_daily_aqi(self, date: datetime, location_id: str, weather_data: Dict) -> Dict:
        """Generate realistic daily AQI data correlated with weather"""
        year = date.year
        season = self.get_season(date.month)
        
        # Base AQI with seasonal variation
        aqi_params = self.climate_params['aqi'][season]
        base_aqi = np.random.normal(aqi_params['mean'], aqi_params['std'])
        
        # Weather correlations
        temp_effect = (weather_data['temperature'] - 25) * 2  # Higher temp = higher AQI
        humidity_effect = (weather_data['humidity'] - 60) * -0.5  # Higher humidity = lower AQI
        wind_effect = (weather_data['wind_speed'] - 10) * -1.5  # Higher wind = lower AQI
        rain_effect = weather_data['precipitation'] * -2  # Rain clears air
        
        aqi = base_aqi + temp_effect + humidity_effect + wind_effect + rain_effect
        aqi = self.apply_climate_trend(aqi, year, 'aqi')
        aqi = max(aqi_params['min'], min(aqi_params['max'], aqi))
        
        # Generate component pollutants based on AQI
        pm25 = max(0, aqi * 0.4 + np.random.normal(0, 10))
        pm10 = max(pm25, pm25 * 1.5 + np.random.normal(0, 15))
        no2 = max(0, aqi * 0.3 + np.random.normal(0, 8))
        so2 = max(0, aqi * 0.2 + np.random.normal(0, 5))
        co = max(0, aqi * 0.01 + np.random.normal(0, 0.5))
        o3 = max(0, aqi * 0.25 + np.random.normal(0, 12))
        
        # Determine dominant pollutant
        pollutants = {'PM2.5': pm25, 'PM10': pm10, 'NO2': no2, 'SO2': so2, 'CO': co, 'O3': o3}
        dominant_pollutant = max(pollutants, key=pollutants.get)
        
        return {
            'timestamp': date.isoformat(),
            'location_id': location_id,
            'pm25': round(pm25, 1),
            'pm10': round(pm10, 1),
            'no2': round(no2, 1),
            'so2': round(so2, 1),
            'co': round(co, 2),
            'o3': round(o3, 1),
            'aqi': round(aqi, 0),
            'dominant_pollutant': dominant_pollutant,
            'data_source': 'generated_historical',
            'quality_score': 0.95
        }
    
    def generate_historical_data(self, start_year: int = 2000, end_year: int = None) -> Tuple[List[Dict], List[Dict]]:
        """Generate comprehensive historical data from start_year to end_year"""
        if end_year is None:
            end_year = datetime.now().year
        
        print(f"🌍 Generating climate data for Pune ({start_year}-{end_year})")
        
        weather_records = []
        aqi_records = []
        
        # Generate data for each location
        for location_id, location_config in PUNE_LOCATIONS.items():
            print(f"📍 Processing location: {location_config.name}")
            
            current_date = datetime(start_year, 1, 1)
            end_date = datetime(end_year + 1, 1, 1)
            
            while current_date < end_date:
                # Generate weather data
                weather_data = self.generate_daily_weather(current_date, location_id)
                weather_records.append(weather_data)
                
                # Generate AQI data (correlated with weather)
                aqi_data = self.generate_daily_aqi(current_date, location_id, weather_data)
                aqi_records.append(aqi_data)
                
                current_date += timedelta(days=1)
        
        print(f"✅ Generated {len(weather_records):,} weather records")
        print(f"✅ Generated {len(aqi_records):,} AQI records")
        
        return weather_records, aqi_records
    
    def save_to_database(self, weather_records: List[Dict], aqi_records: List[Dict]):
        """Save generated data to database"""
        conn = sqlite3.connect(self.db_path)
        
        # Clear existing data
        conn.execute("DELETE FROM weather_historical")
        conn.execute("DELETE FROM air_quality_historical")
        
        # Insert weather data
        weather_df = pd.DataFrame(weather_records)
        weather_df.to_sql('weather_historical', conn, if_exists='append', index=False)
        
        # Insert AQI data
        aqi_df = pd.DataFrame(aqi_records)
        aqi_df.to_sql('air_quality_historical', conn, if_exists='append', index=False)
        
        conn.commit()
        conn.close()
        
        print(f"💾 Saved all data to database: {self.db_path}")

def generate_comprehensive_climate_data():
    """Main function to generate comprehensive climate data"""
    generator = PuneClimateDataGenerator()
    
    # Generate data from 2000 to current year
    weather_records, aqi_records = generator.generate_historical_data(2000, datetime.now().year)
    
    # Save to database
    generator.save_to_database(weather_records, aqi_records)
    
    return len(weather_records), len(aqi_records)

if __name__ == "__main__":
    weather_count, aqi_count = generate_comprehensive_climate_data()
    print(f"\n🎉 Climate data generation completed!")
    print(f"📊 Total records: {weather_count:,} weather + {aqi_count:,} AQI")