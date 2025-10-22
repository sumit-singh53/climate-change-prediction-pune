"""
Real-time Data Collection Module
Fetches live weather and air quality data from APIs for all Pune locations
"""

import asyncio
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import aiohttp
import pandas as pd

from config import API_CONFIG, DATABASE_CONFIG, PUNE_LOCATIONS, REALTIME_CONFIG


class RealtimeDataCollector:
    """Real-time data collector focused on API data sources"""

    def __init__(self):
        self.db_path = DATABASE_CONFIG["sqlite_path"]
        self.setup_database()
        self.last_update = {}
        self.cache = {}

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def setup_database(self):
        """Initialize database with enhanced schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Real-time weather data table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS realtime_weather (
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
                feels_like REAL,
                dew_point REAL,
                data_source TEXT,
                quality_score REAL
            )
        """
        )

        # Real-time air quality data table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS realtime_air_quality (
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
                aqi_category TEXT,
                dominant_pollutant TEXT,
                health_recommendation TEXT,
                data_source TEXT,
                quality_score REAL
            )
        """
        )

        # Data collection log table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS collection_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                location_id TEXT,
                data_type TEXT,
                status TEXT,
                records_collected INTEGER,
                error_message TEXT
            )
        """
        )

        conn.commit()
        conn.close()

        # Initialize location metadata
        self.initialize_location_metadata()

    def initialize_location_metadata(self):
        """Initialize location metadata in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS location_metadata (
                location_id TEXT PRIMARY KEY,
                name TEXT,
                latitude REAL,
                longitude REAL,
                district TEXT,
                zone TEXT,
                elevation REAL,
                population_density REAL,
                last_updated DATETIME
            )
        """
        )

        for loc_id, loc_config in PUNE_LOCATIONS.items():
            cursor.execute(
                """
                INSERT OR REPLACE INTO location_metadata
                (location_id, name, latitude, longitude, district, zone, elevation, population_density, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    loc_id,
                    loc_config.name,
                    loc_config.lat,
                    loc_config.lon,
                    loc_config.district,
                    loc_config.zone,
                    getattr(loc_config, 'elevation', 0.0),
                    getattr(loc_config, 'population_density', 0.0),
                    datetime.now().isoformat(),
                ),
            )

        conn.commit()
        conn.close()

    def should_update_data(self, location_id: str, data_type: str) -> bool:
        """Check if data should be updated based on cache and timing"""
        cache_key = f"{location_id}_{data_type}"

        if not REALTIME_CONFIG["enable_caching"]:
            return True

        if cache_key not in self.last_update:
            return True

        time_since_update = datetime.now() - self.last_update[cache_key]
        cache_duration = timedelta(minutes=REALTIME_CONFIG["cache_duration_minutes"])

        return time_since_update > cache_duration

    async def fetch_weather_data(self, location_id: str) -> Optional[Dict]:
        """Fetch comprehensive weather data for a location"""
        if not self.should_update_data(location_id, "weather"):
            return self.cache.get(f"{location_id}_weather")

        location = PUNE_LOCATIONS[location_id]

        # Current weather with extended parameters
        params = {
            "latitude": location.lat,
            "longitude": location.lon,
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "surface_pressure",
                "wind_speed_10m",
                "wind_direction_10m",
                "precipitation",
                "shortwave_radiation",
                "uv_index",
                "visibility",
                "cloud_cover",
                "apparent_temperature",
                "dew_point_2m",
            ],
            "timezone": "Asia/Kolkata",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    API_CONFIG["open_meteo"]["weather_url"], params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        processed_data = self.process_weather_data(data, location_id)

                        # Cache the data
                        self.cache[f"{location_id}_weather"] = processed_data
                        self.last_update[f"{location_id}_weather"] = datetime.now()

                        return processed_data
                    else:
                        self.logger.error(
                            f"Weather API error for {location_id}: {response.status}"
                        )
                        return None

        except Exception as e:
            self.logger.error(f"Error fetching weather data for {location_id}: {e}")
            return None

    async def fetch_air_quality_data(self, location_id: str) -> Optional[Dict]:
        """Fetch comprehensive air quality data for a location"""
        if not self.should_update_data(location_id, "air_quality"):
            return self.cache.get(f"{location_id}_air_quality")

        location = PUNE_LOCATIONS[location_id]

        params = {
            "latitude": location.lat,
            "longitude": location.lon,
            "current": [
                "pm2_5",
                "pm10",
                "carbon_monoxide",
                "nitrogen_dioxide",
                "sulphur_dioxide",
                "ozone",
                "european_aqi",
                "us_aqi",
            ],
            "timezone": "Asia/Kolkata",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    API_CONFIG["open_meteo"]["air_quality_url"], params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        processed_data = self.process_air_quality_data(
                            data, location_id
                        )

                        # Cache the data
                        self.cache[f"{location_id}_air_quality"] = processed_data
                        self.last_update[f"{location_id}_air_quality"] = datetime.now()

                        return processed_data
                    else:
                        self.logger.error(
                            f"Air quality API error for {location_id}: {response.status}"
                        )
                        return None

        except Exception as e:
            self.logger.error(f"Error fetching air quality data for {location_id}: {e}")
            return None

    def process_weather_data(self, data: Dict, location_id: str) -> Dict:
        """Process weather data from API response"""
        if "current" not in data:
            return {}

        current = data["current"]

        return {
            "timestamp": current.get("time", datetime.now().isoformat()),
            "location_id": location_id,
            "temperature": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "pressure": current.get("surface_pressure"),
            "wind_speed": current.get("wind_speed_10m"),
            "wind_direction": current.get("wind_direction_10m"),
            "precipitation": current.get("precipitation"),
            "solar_radiation": current.get("shortwave_radiation"),
            "uv_index": current.get("uv_index"),
            "visibility": current.get("visibility"),
            "cloud_cover": current.get("cloud_cover"),
            "feels_like": current.get("apparent_temperature"),
            "dew_point": current.get("dew_point_2m"),
            "data_source": "open_meteo_realtime",
            "quality_score": self.calculate_data_quality_score(current),
        }

    def process_air_quality_data(self, data: Dict, location_id: str) -> Dict:
        """Process air quality data from API response"""
        if "current" not in data:
            return {}

        current = data["current"]

        # Calculate AQI and determine category
        pm25 = current.get("pm2_5", 0)
        aqi = current.get("european_aqi", current.get("us_aqi", 0))

        return {
            "timestamp": current.get("time", datetime.now().isoformat()),
            "location_id": location_id,
            "pm25": pm25,
            "pm10": current.get("pm10"),
            "no2": current.get("nitrogen_dioxide"),
            "so2": current.get("sulphur_dioxide"),
            "co": current.get("carbon_monoxide"),
            "o3": current.get("ozone"),
            "aqi": aqi,
            "aqi_category": self.get_aqi_category(aqi),
            "dominant_pollutant": self.calculate_dominant_pollutant(current),
            "health_recommendation": self.get_health_recommendation(aqi),
            "data_source": "open_meteo_realtime",
            "quality_score": self.calculate_data_quality_score(current),
        }

    def get_aqi_category(self, aqi: float) -> str:
        """Get AQI category based on value"""
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"

    def get_health_recommendation(self, aqi: float) -> str:
        """Get health recommendation based on AQI"""
        if aqi <= 50:
            return "Air quality is good. Ideal for outdoor activities."
        elif aqi <= 100:
            return "Air quality is acceptable. Sensitive individuals should consider limiting outdoor activities."
        elif aqi <= 150:
            return "Sensitive groups should reduce outdoor activities."
        elif aqi <= 200:
            return "Everyone should limit outdoor activities."
        elif aqi <= 300:
            return "Avoid outdoor activities. Keep windows closed."
        else:
            return "Health alert! Avoid all outdoor activities."

    def calculate_dominant_pollutant(self, data: Dict) -> str:
        """Calculate the dominant pollutant"""
        pollutants = {
            "PM2.5": data.get("pm2_5", 0),
            "PM10": data.get("pm10", 0),
            "NO2": data.get("nitrogen_dioxide", 0),
            "SO2": data.get("sulphur_dioxide", 0),
            "CO": data.get("carbon_monoxide", 0),
            "O3": data.get("ozone", 0),
        }

        # WHO guidelines for comparison
        standards = {
            "PM2.5": 15,  # WHO guideline
            "PM10": 45,  # WHO guideline
            "NO2": 25,  # WHO guideline
            "SO2": 40,  # WHO guideline
            "CO": 4000,  # WHO guideline (mg/m³)
            "O3": 100,  # WHO guideline
        }

        ratios = {}
        for pollutant, value in pollutants.items():
            if value and pollutant in standards and standards[pollutant] > 0:
                ratios[pollutant] = value / standards[pollutant]

        if ratios:
            return max(ratios, key=ratios.get)
        return "PM2.5"  # Default

    def calculate_data_quality_score(self, data: Dict) -> float:
        """Calculate data quality score based on completeness"""
        total_fields = len(data)
        non_null_fields = sum(1 for v in data.values() if v is not None)

        if total_fields == 0:
            return 0.0

        return non_null_fields / total_fields

    async def collect_all_locations_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Collect data for all Pune locations"""
        weather_tasks = []
        air_quality_tasks = []

        for location_id in PUNE_LOCATIONS.keys():
            weather_tasks.append(self.fetch_weather_data(location_id))
            air_quality_tasks.append(self.fetch_air_quality_data(location_id))

        # Execute all tasks concurrently
        weather_results = await asyncio.gather(*weather_tasks, return_exceptions=True)
        air_quality_results = await asyncio.gather(
            *air_quality_tasks, return_exceptions=True
        )

        # Filter successful results
        weather_data = [r for r in weather_results if isinstance(r, dict) and r]
        air_quality_data = [r for r in air_quality_results if isinstance(r, dict) and r]

        return weather_data, air_quality_data

    def save_to_database(self, weather_data: List[Dict], air_quality_data: List[Dict]):
        """Save collected data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Save weather data
            for data in weather_data:
                cursor.execute(
                    """
                    INSERT INTO realtime_weather 
                    (timestamp, location_id, temperature, humidity, pressure, wind_speed, 
                     wind_direction, precipitation, solar_radiation, uv_index, visibility, 
                     cloud_cover, feels_like, dew_point, data_source, quality_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        data["timestamp"],
                        data["location_id"],
                        data["temperature"],
                        data["humidity"],
                        data["pressure"],
                        data["wind_speed"],
                        data["wind_direction"],
                        data["precipitation"],
                        data["solar_radiation"],
                        data["uv_index"],
                        data["visibility"],
                        data["cloud_cover"],
                        data["feels_like"],
                        data["dew_point"],
                        data["data_source"],
                        data["quality_score"],
                    ),
                )

            # Save air quality data
            for data in air_quality_data:
                cursor.execute(
                    """
                    INSERT INTO realtime_air_quality 
                    (timestamp, location_id, pm25, pm10, no2, so2, co, o3, aqi, 
                     aqi_category, dominant_pollutant, health_recommendation, 
                     data_source, quality_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        data["timestamp"],
                        data["location_id"],
                        data["pm25"],
                        data["pm10"],
                        data["no2"],
                        data["so2"],
                        data["co"],
                        data["o3"],
                        data["aqi"],
                        data["aqi_category"],
                        data["dominant_pollutant"],
                        data["health_recommendation"],
                        data["data_source"],
                        data["quality_score"],
                    ),
                )

            # Log collection status
            cursor.execute(
                """
                INSERT INTO collection_log 
                (timestamp, location_id, data_type, status, records_collected)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    datetime.now().isoformat(),
                    "all_locations",
                    "weather_air_quality",
                    "success",
                    len(weather_data) + len(air_quality_data),
                ),
            )

            conn.commit()
            self.logger.info(
                f"Saved {len(weather_data)} weather and {len(air_quality_data)} air quality records"
            )

        except Exception as e:
            self.logger.error(f"Error saving to database: {e}")
            conn.rollback()
        finally:
            conn.close()

    def get_latest_data(self, location_id: str = None) -> Dict:
        """Get latest data for dashboard display"""
        conn = sqlite3.connect(self.db_path)

        try:
            # Get latest weather data
            weather_query = """
                SELECT * FROM realtime_weather 
                WHERE timestamp = (SELECT MAX(timestamp) FROM realtime_weather)
            """
            if location_id:
                weather_query += f" AND location_id = '{location_id}'"

            weather_df = pd.read_sql_query(weather_query, conn)

            # Get latest air quality data
            air_quality_query = """
                SELECT * FROM realtime_air_quality 
                WHERE timestamp = (SELECT MAX(timestamp) FROM realtime_air_quality)
            """
            if location_id:
                air_quality_query += f" AND location_id = '{location_id}'"

            air_quality_df = pd.read_sql_query(air_quality_query, conn)

            return {
                "weather": weather_df.to_dict("records"),
                "air_quality": air_quality_df.to_dict("records"),
                "last_updated": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error retrieving latest data: {e}")
            return {"weather": [], "air_quality": [], "last_updated": None}
        finally:
            conn.close()

    async def run_continuous_collection(self):
        """Run continuous data collection"""
        self.logger.info("Starting continuous real-time data collection...")

        while True:
            try:
                start_time = time.time()

                # Collect data for all locations
                weather_data, air_quality_data = await self.collect_all_locations_data()

                # Save to database
                if weather_data or air_quality_data:
                    self.save_to_database(weather_data, air_quality_data)

                collection_time = time.time() - start_time
                self.logger.info(
                    f"Collection cycle completed in {collection_time:.2f} seconds"
                )

                # Wait for next collection cycle
                await asyncio.sleep(REALTIME_CONFIG["update_interval_minutes"] * 60)

            except Exception as e:
                self.logger.error(f"Error in continuous collection: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying


async def main():
    """Main function for testing"""
    collector = RealtimeDataCollector()

    # Run one collection cycle
    weather_data, air_quality_data = await collector.collect_all_locations_data()

    if weather_data or air_quality_data:
        collector.save_to_database(weather_data, air_quality_data)
        print(
            f"✅ Collected {len(weather_data)} weather and {len(air_quality_data)} air quality records"
        )
    else:
        print("⚠️ No data collected")


if __name__ == "__main__":
    asyncio.run(main())
