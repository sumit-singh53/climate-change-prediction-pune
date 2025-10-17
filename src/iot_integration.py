"""
IoT Integration Module for Real-time Environmental Data Collection
Supports MQTT, HTTP endpoints, and various sensor protocols
"""

import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import pandas as pd
import paho.mqtt.client as mqtt
from flask import Flask, request, jsonify
import logging
from config import IOT_CONFIG, DATABASE_CONFIG, PUNE_LOCATIONS

class IoTDataCollector:
    """Collects and processes real-time IoT sensor data"""
    
    def __init__(self):
        self.mqtt_client = None
        self.flask_app = Flask(__name__)
        self.db_path = DATABASE_CONFIG['sqlite_path']
        self.setup_database()
        self.setup_mqtt()
        self.setup_http_endpoints()
        self.data_buffer = []
        self.buffer_lock = threading.Lock()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def setup_database(self):
        """Initialize SQLite database for IoT data storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create IoT sensor data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS iot_sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                location_id TEXT,
                sensor_type TEXT,
                sensor_id TEXT,
                value REAL,
                unit TEXT,
                quality_score REAL,
                metadata TEXT
            )
        ''')
        
        # Create sensor registry table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_registry (
                sensor_id TEXT PRIMARY KEY,
                location_id TEXT,
                sensor_type TEXT,
                installation_date DATETIME,
                calibration_date DATETIME,
                status TEXT,
                metadata TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def setup_mqtt(self):
        """Setup MQTT client for sensor data collection"""
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                self.logger.info("Connected to MQTT broker")
                # Subscribe to all sensor topics
                for topic in IOT_CONFIG['sensor_topics'].values():
                    client.subscribe(f"{topic}/+/+")  # topic/location/sensor_id
            else:
                self.logger.error(f"Failed to connect to MQTT broker: {rc}")
        
        def on_message(client, userdata, msg):
            try:
                topic_parts = msg.topic.split('/')
                sensor_type = topic_parts[1]
                location_id = topic_parts[2]
                sensor_id = topic_parts[3]
                
                data = json.loads(msg.payload.decode())
                self.process_sensor_data(sensor_type, location_id, sensor_id, data)
                
            except Exception as e:
                self.logger.error(f"Error processing MQTT message: {e}")
        
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = on_connect
        self.mqtt_client.on_message = on_message
    
    def setup_http_endpoints(self):
        """Setup HTTP endpoints for sensor data submission"""
        
        @self.flask_app.route('/api/sensor-data', methods=['POST'])
        def receive_sensor_data():
            try:
                data = request.json
                required_fields = ['sensor_type', 'location_id', 'sensor_id', 'value', 'timestamp']
                
                if not all(field in data for field in required_fields):
                    return jsonify({'error': 'Missing required fields'}), 400
                
                self.process_sensor_data(
                    data['sensor_type'],
                    data['location_id'],
                    data['sensor_id'],
                    data
                )
                
                return jsonify({'status': 'success'}), 200
                
            except Exception as e:
                self.logger.error(f"Error in HTTP endpoint: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.flask_app.route('/api/sensor-status/<sensor_id>', methods=['GET'])
        def get_sensor_status(sensor_id):
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM sensor_registry WHERE sensor_id = ?
                ''', (sensor_id,))
                
                result = cursor.fetchone()
                conn.close()
                
                if result:
                    return jsonify({
                        'sensor_id': result[0],
                        'location_id': result[1],
                        'sensor_type': result[2],
                        'status': result[5]
                    })
                else:
                    return jsonify({'error': 'Sensor not found'}), 404
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def process_sensor_data(self, sensor_type: str, location_id: str, sensor_id: str, data: Dict):
        """Process incoming sensor data and store in database"""
        try:
            timestamp = data.get('timestamp', datetime.now().isoformat())
            value = data.get('value')
            unit = data.get('unit', '')
            quality_score = data.get('quality_score', 1.0)
            metadata = json.dumps(data.get('metadata', {}))
            
            # Validate data quality
            if not self.validate_sensor_data(sensor_type, value):
                self.logger.warning(f"Invalid data from sensor {sensor_id}: {value}")
                quality_score = 0.0
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO iot_sensor_data 
                (timestamp, location_id, sensor_type, sensor_id, value, unit, quality_score, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, location_id, sensor_type, sensor_id, value, unit, quality_score, metadata))
            
            conn.commit()
            conn.close()
            
            # Add to buffer for real-time processing
            with self.buffer_lock:
                self.data_buffer.append({
                    'timestamp': timestamp,
                    'location_id': location_id,
                    'sensor_type': sensor_type,
                    'sensor_id': sensor_id,
                    'value': value,
                    'quality_score': quality_score
                })
            
            self.logger.info(f"Processed data from {sensor_id}: {sensor_type}={value}")
            
        except Exception as e:
            self.logger.error(f"Error processing sensor data: {e}")
    
    def validate_sensor_data(self, sensor_type: str, value: float) -> bool:
        """Validate sensor data based on expected ranges"""
        validation_ranges = {
            'temperature': (-50, 60),  # Celsius
            'humidity': (0, 100),      # Percentage
            'pm25': (0, 1000),         # µg/m³
            'pm10': (0, 2000),         # µg/m³
            'co2': (300, 5000),        # ppm
            'noise': (20, 120),        # dB
            'pressure': (800, 1200),   # hPa
            'wind_speed': (0, 50)      # m/s
        }
        
        if sensor_type in validation_ranges:
            min_val, max_val = validation_ranges[sensor_type]
            return min_val <= value <= max_val
        
        return True  # Unknown sensor types pass validation
    
    def register_sensor(self, sensor_id: str, location_id: str, sensor_type: str, metadata: Dict = None):
        """Register a new sensor in the system"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO sensor_registry
                (sensor_id, location_id, sensor_type, installation_date, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (sensor_id, location_id, sensor_type, datetime.now().isoformat(), 
                  'active', json.dumps(metadata or {})))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Registered sensor {sensor_id} at {location_id}")
            
        except Exception as e:
            self.logger.error(f"Error registering sensor: {e}")
    
    def get_recent_data(self, location_id: str = None, hours: int = 24) -> pd.DataFrame:
        """Get recent sensor data for analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT timestamp, location_id, sensor_type, sensor_id, value, quality_score
                FROM iot_sensor_data
                WHERE timestamp > datetime('now', '-{} hours')
            '''.format(hours)
            
            if location_id:
                query += f" AND location_id = '{location_id}'"
            
            query += " ORDER BY timestamp DESC"
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error retrieving recent data: {e}")
            return pd.DataFrame()
    
    def start_mqtt_listener(self):
        """Start MQTT client in background thread"""
        def mqtt_loop():
            try:
                self.mqtt_client.connect(IOT_CONFIG['mqtt_broker'], IOT_CONFIG['mqtt_port'], 60)
                self.mqtt_client.loop_forever()
            except Exception as e:
                self.logger.error(f"MQTT connection error: {e}")
        
        mqtt_thread = threading.Thread(target=mqtt_loop, daemon=True)
        mqtt_thread.start()
        self.logger.info("MQTT listener started")
    
    def start_http_server(self, host='0.0.0.0', port=5000):
        """Start HTTP server for sensor data collection"""
        self.flask_app.run(host=host, port=port, debug=False)
    
    def cleanup_old_data(self):
        """Clean up old sensor data based on retention policy"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=IOT_CONFIG['data_retention_days'])
            
            cursor.execute('''
                DELETE FROM iot_sensor_data 
                WHERE timestamp < ?
            ''', (cutoff_date.isoformat(),))
            
            deleted_rows = cursor.rowcount
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cleaned up {deleted_rows} old sensor records")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")


class SensorSimulator:
    """Simulates IoT sensors for testing purposes"""
    
    def __init__(self, collector: IoTDataCollector):
        self.collector = collector
        self.running = False
    
    def simulate_sensors(self, duration_minutes: int = 60):
        """Simulate sensor data for testing"""
        import random
        import time
        
        self.running = True
        end_time = time.time() + (duration_minutes * 60)
        
        # Register simulated sensors
        for location_id in PUNE_LOCATIONS.keys():
            for sensor_type in ['temperature', 'humidity', 'pm25', 'pm10']:
                sensor_id = f"sim_{location_id}_{sensor_type}_001"
                self.collector.register_sensor(sensor_id, location_id, sensor_type)
        
        while self.running and time.time() < end_time:
            for location_id in PUNE_LOCATIONS.keys():
                # Simulate temperature sensor
                temp_value = random.uniform(15, 45)  # Pune temperature range
                self.collector.process_sensor_data(
                    'temperature', location_id, f"sim_{location_id}_temperature_001",
                    {'value': temp_value, 'unit': 'C', 'quality_score': random.uniform(0.8, 1.0)}
                )
                
                # Simulate humidity sensor
                humidity_value = random.uniform(30, 90)
                self.collector.process_sensor_data(
                    'humidity', location_id, f"sim_{location_id}_humidity_001",
                    {'value': humidity_value, 'unit': '%', 'quality_score': random.uniform(0.8, 1.0)}
                )
                
                # Simulate PM2.5 sensor
                pm25_value = random.uniform(10, 150)  # Pune AQI range
                self.collector.process_sensor_data(
                    'pm25', location_id, f"sim_{location_id}_pm25_001",
                    {'value': pm25_value, 'unit': 'µg/m³', 'quality_score': random.uniform(0.7, 1.0)}
                )
            
            time.sleep(30)  # Send data every 30 seconds
    
    def stop(self):
        """Stop sensor simulation"""
        self.running = False


if __name__ == "__main__":
    # Example usage
    collector = IoTDataCollector()
    
    # Start MQTT listener
    collector.start_mqtt_listener()
    
    # Start sensor simulation for testing
    simulator = SensorSimulator(collector)
    
    # Run simulation in background
    import threading
    sim_thread = threading.Thread(target=simulator.simulate_sensors, args=(10,), daemon=True)
    sim_thread.start()
    
    # Start HTTP server
    print("Starting IoT data collection server...")
    print("MQTT topics: sensors/{type}/{location}/{sensor_id}")
    print("HTTP endpoint: POST /api/sensor-data")
    collector.start_http_server()