"""
Enhanced Data Collection Module
Collects weather and air quality data from multiple sources with location-wise coverage
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import json
import logging
from typing import Dict, List, Optional, Tuple
from config import PUNE_LOCATIONS, API_CONFIG, DATABASE_CONFIG
import time

class EnhancedDataCollector:
    """Enhanced data collector with multi-location support and advanced features"""
    
    def __init__(self):
        self.db_path = DATABASE_CONFIG['sqlite_path']
        self.setup_database()
        self.session = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def setup_database(self):
        """Initialize database with enhanced schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Weather data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weather_historical (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                location_id TEXT,
                temperature REAL,
                humidity REAL,
                pressure REAL,
                wind_speed REAL,
                wind_direction REAL,
                precipitation REAL,
                solar_radiation REAL,
                uv_index REAL,
                visibility REAL,
                cloud_cover REAL,
                data_source TEXT,
                quality_score REAL
            )
        ''')
        
        # Air quality data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS air_quality_historical (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                location_id TEXT,
                pm25 REAL,
                pm10 REAL,
                no2 REAL,
                so2 REAL,
                co REAL,
                o3 REAL,
                aqi REAL,
                dominant_pollutant TEXT,
                data_source TEXT,
                quality_score REAL
            )
        ''')
        
        # Location metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS location_metadata (
                location_id TEXT PRIMARY KEY,
                name TEXT,
                latitude REAL,
                longitude REAL,
                district TEXT,
                zone TEXT,
                elevation REAL,
                land_use TEXT,
                population_density REAL,
                traffic_density TEXT,
                industrial_activity TEXT
            )
        ''')
        
        # Data quality metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                location_id TEXT,
                data_type TEXT,
                completeness_score REAL,
                accuracy_score REAL,
                consistency_score REAL,
                timeliness_score REAL,
                overall_quality REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        
        # Initialize location metadata
        self.initialize_location_metadata()
    
    def initialize_location_metadata(self):
        """Initialize location metadata in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for loc_id, loc_config in PUNE_LOCATIONS.items():
            cursor.execute('''
                INSERT OR REPLACE INTO location_metadata
                (location_id, name, latitude, longitude, district, zone)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (loc_id, loc_config.name, loc_config.lat, loc_config.lon,
                  loc_config.district, loc_config.zone))
        
        conn.commit()
        conn.close()
    
    async def fetch_weather_data(self, location_id: str, days_back: int = 7) -> pd.DataFrame:
        """Fetch comprehensive weather data for a location"""
        location = PUNE_LOCATIONS[location_id]
        
        # Current weather
        current_params = {
            'latitude': location.lat,
            'longitude': location.lon,
            'current': [
                'temperature_2m', 'relative_humidity_2m', 'surface_pressure',
                'wind_speed_10m', 'wind_direction_10m', 'precipitation',
                'shortwave_radiation', 'uv_index', 'visibility', 'cloud_cover'
            ],
            'timezone': 'Asia/Kolkata'
        }
        
        # Historical weather
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        historical_params = {
            'latitude': location.lat,
            'longitude': location.lon,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'hourly': [
                'temperature_2m', 'relative_humidity_2m', 'surface_pressure',
                'wind_speed_10m', 'wind_direction_10m', 'precipitation',
                'shortwave_radiation', 'uv_index', 'visibility', 'cloud_cover'
            ],
            'timezone': 'Asia/Kolkata'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                # Fetch current data
                async with session.get(API_CONFIG['open_meteo']['weather_url'], 
                                     params=current_params) as response:
                    current_data = await response.json()
                
                # Fetch historical data
                async with session.get(API_CONFIG['open_meteo']['historical_url'], 
                                     params=historical_params) as response:
                    historical_data = await response.json()
            
            # Process and combine data
            weather_df = self.process_weather_data(current_data, historical_data, location_id)
            return weather_df
            
        except Exception as e:
            self.logger.error(f"Error fetching weather data for {location_id}: {e}")
            return pd.DataFrame()
    
    async def fetch_air_quality_data(self, location_id: str, days_back: int = 7) -> pd.DataFrame:
        """Fetch comprehensive air quality data for a location"""
        location = PUNE_LOCATIONS[location_id]
        
        # Current air quality
        current_params = {
            'latitude': location.lat,
            'longitude': location.lon,
            'current': ['pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide', 
                       'sulphur_dioxide', 'ozone', 'european_aqi'],
            'timezone': 'Asia/Kolkata'
        }
        
        # Historical air quality
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        historical_params = {
            'latitude': location.lat,
            'longitude': location.lon,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'hourly': ['pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide', 
                      'sulphur_dioxide', 'ozone', 'european_aqi'],
            'timezone': 'Asia/Kolkata'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                # Fetch current data
                async with session.get(API_CONFIG['open_meteo']['air_quality_url'], 
                                     params=current_params) as response:
                    current_data = await response.json()
                
                # Fetch historical data
                async with session.get(API_CONFIG['open_meteo']['air_quality_url'], 
                                     params=historical_params) as response:
                    historical_data = await response.json()
            
            # Process and combine data
            aqi_df = self.process_air_quality_data(current_data, historical_data, location_id)
            return aqi_df
            
        except Exception as e:
            self.logger.error(f"Error fetching air quality data for {location_id}: {e}")
            return pd.DataFrame()
    
    def process_weather_data(self, current_data: Dict, historical_data: Dict, location_id: str) -> pd.DataFrame:
        """Process weather data from API responses"""
        weather_records = []
        
        # Process historical data
        if 'hourly' in historical_data:
            hourly = historical_data['hourly']
            times = hourly.get('time', [])
            
            for i, timestamp in enumerate(times):
                record = {
                    'timestamp': timestamp,
                    'location_id': location_id,
                    'temperature': hourly.get('temperature_2m', [None] * len(times))[i],
                    'humidity': hourly.get('relative_humidity_2m', [None] * len(times))[i],
                    'pressure': hourly.get('surface_pressure', [None] * len(times))[i],
                    'wind_speed': hourly.get('wind_speed_10m', [None] * len(times))[i],
                    'wind_direction': hourly.get('wind_direction_10m', [None] * len(times))[i],
                    'precipitation': hourly.get('precipitation', [None] * len(times))[i],
                    'solar_radiation': hourly.get('shortwave_radiation', [None] * len(times))[i],
                    'uv_index': hourly.get('uv_index', [None] * len(times))[i],
                    'visibility': hourly.get('visibility', [None] * len(times))[i],
                    'cloud_cover': hourly.get('cloud_cover', [None] * len(times))[i],
                    'data_source': 'open_meteo_historical',
                    'quality_score': self.calculate_data_quality_score(record)
                }
                weather_records.append(record)
        
        # Process current data
        if 'current' in current_data:
            current = current_data['current']
            record = {
                'timestamp': current.get('time'),
                'location_id': location_id,
                'temperature': current.get('temperature_2m'),
                'humidity': current.get('relative_humidity_2m'),
                'pressure': current.get('surface_pressure'),
                'wind_speed': current.get('wind_speed_10m'),
                'wind_direction': current.get('wind_direction_10m'),
                'precipitation': current.get('precipitation'),
                'solar_radiation': current.get('shortwave_radiation'),
                'uv_index': current.get('uv_index'),
                'visibility': current.get('visibility'),
                'cloud_cover': current.get('cloud_cover'),
                'data_source': 'open_meteo_current',
                'quality_score': self.calculate_data_quality_score(record)
            }
            weather_records.append(record)
        
        return pd.DataFrame(weather_records)
    
    def process_air_quality_data(self, current_data: Dict, historical_data: Dict, location_id: str) -> pd.DataFrame:
        """Process air quality data from API responses"""
        aqi_records = []
        
        # Process historical data
        if 'hourly' in historical_data:
            hourly = historical_data['hourly']
            times = hourly.get('time', [])
            
            for i, timestamp in enumerate(times):
                pm25 = hourly.get('pm2_5', [None] * len(times))[i]
                pm10 = hourly.get('pm10', [None] * len(times))[i]
                
                record = {
                    'timestamp': timestamp,
                    'location_id': location_id,
                    'pm25': pm25,
                    'pm10': pm10,
                    'no2': hourly.get('nitrogen_dioxide', [None] * len(times))[i],
                    'so2': hourly.get('sulphur_dioxide', [None] * len(times))[i],
                    'co': hourly.get('carbon_monoxide', [None] * len(times))[i],
                    'o3': hourly.get('ozone', [None] * len(times))[i],
                    'aqi': hourly.get('european_aqi', [None] * len(times))[i],
                    'dominant_pollutant': self.calculate_dominant_pollutant(record),
                    'data_source': 'open_meteo_historical',
                    'quality_score': self.calculate_data_quality_score(record)
                }
                aqi_records.append(record)
        
        # Process current data
        if 'current' in current_data:
            current = current_data['current']
            record = {
                'timestamp': current.get('time'),
                'location_id': location_id,
                'pm25': current.get('pm2_5'),
                'pm10': current.get('pm10'),
                'no2': current.get('nitrogen_dioxide'),
                'so2': current.get('sulphur_dioxide'),
                'co': current.get('carbon_monoxide'),
                'o3': current.get('ozone'),
                'aqi': current.get('european_aqi'),
                'dominant_pollutant': self.calculate_dominant_pollutant(record),
                'data_source': 'open_meteo_current',
                'quality_score': self.calculate_data_quality_score(record)
            }
            aqi_records.append(record)
        
        return pd.DataFrame(aqi_records)
    
    def calculate_dominant_pollutant(self, record: Dict) -> str:
        """Calculate the dominant pollutant based on AQI contribution"""
        pollutants = {
            'PM2.5': record.get('pm25', 0),
            'PM10': record.get('pm10', 0),
            'NO2': record.get('no2', 0),
            'SO2': record.get('so2', 0),
            'CO': record.get('co', 0),
            'O3': record.get('o3', 0)
        }
        
        # Simple heuristic - highest concentration relative to standards
        standards = {
            'PM2.5': 25,   # WHO guideline
            'PM10': 50,    # WHO guideline
            'NO2': 40,     # WHO guideline
            'SO2': 20,     # WHO guideline
            'CO': 10000,   # WHO guideline
            'O3': 100      # WHO guideline
        }
        
        ratios = {}
        for pollutant, value in pollutants.items():
            if value and pollutant in standards:
                ratios[pollutant] = value / standards[pollutant]
        
        if ratios:
            return max(ratios, key=ratios.get)
        return 'Unknown'
    
    def calculate_data_quality_score(self, record: Dict) -> float:
        """Calculate data quality score based on completeness and validity"""
        total_fields = len(record)
        non_null_fields = sum(1 for v in record.values() if v is not None)
        
        completeness = non_null_fields / total_fields
        
        # Additional quality checks can be added here
        # (e.g., range validation, consistency checks)
        
        return completeness
    
    async def collect_all_locations_data(self, days_back: int = 7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Collect data for all Pune locations"""
        weather_tasks = []
        aqi_tasks = []
        
        for location_id in PUNE_LOCATIONS.keys():
            weather_tasks.append(self.fetch_weather_data(location_id, days_back))
            aqi_tasks.append(self.fetch_air_quality_data(location_id, days_back))
        
        # Execute all tasks concurrently
        weather_results = await asyncio.gather(*weather_tasks, return_exceptions=True)
        aqi_results = await asyncio.gather(*aqi_tasks, return_exceptions=True)
        
        # Combine results
        all_weather_data = []
        all_aqi_data = []
        
        for result in weather_results:
            if isinstance(result, pd.DataFrame) and not result.empty:
                all_weather_data.append(result)
        
        for result in aqi_results:
            if isinstance(result, pd.DataFrame) and not result.empty:
                all_aqi_data.append(result)
        
        weather_df = pd.concat(all_weather_data, ignore_index=True) if all_weather_data else pd.DataFrame()
        aqi_df = pd.concat(all_aqi_data, ignore_index=True) if all_aqi_data else pd.DataFrame()
        
        return weather_df, aqi_df
    
    def save_to_database(self, weather_df: pd.DataFrame, aqi_df: pd.DataFrame):
        """Save collected data to database"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Save weather data
            if not weather_df.empty:
                weather_df.to_sql('weather_historical', conn, if_exists='append', index=False)
                self.logger.info(f"Saved {len(weather_df)} weather records to database")
            
            # Save air quality data
            if not aqi_df.empty:
                aqi_df.to_sql('air_quality_historical', conn, if_exists='append', index=False)
                self.logger.info(f"Saved {len(aqi_df)} air quality records to database")
            
            conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error saving to database: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    async def run_data_collection(self, days_back: int = 7):
        """Run complete data collection process"""
        self.logger.info("Starting enhanced data collection for all Pune locations...")
        
        start_time = time.time()
        weather_df, aqi_df = await self.collect_all_locations_data(days_back)
        
        if not weather_df.empty or not aqi_df.empty:
            self.save_to_database(weather_df, aqi_df)
            
            collection_time = time.time() - start_time
            self.logger.info(f"Data collection completed in {collection_time:.2f} seconds")
            self.logger.info(f"Weather records: {len(weather_df)}, AQI records: {len(aqi_df)}")
        else:
            self.logger.warning("No data collected")


async def main():
    """Main function for testing data collection"""
    collector = EnhancedDataCollector()
    await collector.run_data_collection(days_back=3)

if __name__ == "__main__":
    asyncio.run(main())