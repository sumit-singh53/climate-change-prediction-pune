# API Documentation

This document describes the APIs available in the Enhanced Climate & AQI Prediction System.

## 🔌 IoT Data Submission API

### Base URL
```
http://localhost:5000
```

### Authentication
Currently no authentication required. Future versions will include API key authentication.

---

## Endpoints

### 1. Submit Sensor Data

Submit real-time sensor data to the system.

**Endpoint:** `POST /api/sensor-data`

**Content-Type:** `application/json`

**Request Body:**
```json
{
  "sensor_type": "temperature",
  "location_id": "pune_central",
  "sensor_id": "temp_001",
  "value": 28.5,
  "unit": "C",
  "timestamp": "2024-01-15T10:30:00Z",
  "quality_score": 0.95,
  "metadata": {
    "battery_level": 85,
    "signal_strength": -45
  }
}
```

**Parameters:**
- `sensor_type` (string, required): Type of sensor
  - Supported: `temperature`, `humidity`, `pm25`, `pm10`, `co2`, `noise`, `pressure`, `wind_speed`
- `location_id` (string, required): Location identifier
  - Supported: `pune_central`, `pimpri_chinchwad`, `hadapsar`, `kothrud`, `wakad`, `baner`, `katraj`, `wagholi`
- `sensor_id` (string, required): Unique sensor identifier
- `value` (number, required): Sensor reading value
- `unit` (string, optional): Unit of measurement
- `timestamp` (string, optional): ISO 8601 timestamp (defaults to current time)
- `quality_score` (number, optional): Data quality score 0.0-1.0 (defaults to 1.0)
- `metadata` (object, optional): Additional sensor metadata

**Response:**
```json
{
  "status": "success",
  "message": "Data received successfully",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Status Codes:**
- `200 OK`: Data received successfully
- `400 Bad Request`: Invalid request format or missing required fields
- `422 Unprocessable Entity`: Data validation failed
- `500 Internal Server Error`: Server error

**Example cURL:**
```bash
curl -X POST http://localhost:5000/api/sensor-data \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_type": "temperature",
    "location_id": "pune_central",
    "sensor_id": "temp_001",
    "value": 28.5,
    "unit": "C",
    "quality_score": 0.95
  }'
```

---

### 2. Get Sensor Status

Retrieve status information for a specific sensor.

**Endpoint:** `GET /api/sensor-status/{sensor_id}`

**Parameters:**
- `sensor_id` (string, required): Unique sensor identifier

**Response:**
```json
{
  "sensor_id": "temp_001",
  "location_id": "pune_central",
  "sensor_type": "temperature",
  "status": "active",
  "last_seen": "2024-01-15T10:30:00Z",
  "data_points_24h": 2880
}
```

**Status Codes:**
- `200 OK`: Sensor found
- `404 Not Found`: Sensor not found
- `500 Internal Server Error`: Server error

---

## 📡 MQTT Protocol

### Connection Details
- **Broker:** `localhost`
- **Port:** `1883`
- **Protocol:** MQTT v3.1.1

### Topic Structure
```
sensors/{sensor_type}/{location_id}/{sensor_id}
```

**Examples:**
- `sensors/temperature/pune_central/temp_001`
- `sensors/pm25/hadapsar/air_quality_001`
- `sensors/humidity/kothrud/humid_001`

### Message Format
```json
{
  "value": 28.5,
  "unit": "C",
  "timestamp": "2024-01-15T10:30:00Z",
  "quality_score": 0.95,
  "metadata": {
    "battery_level": 85
  }
}
```

### Python Example
```python
import paho.mqtt.client as mqtt
import json
from datetime import datetime

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")

client = mqtt.Client()
client.on_connect = on_connect
client.connect("localhost", 1883, 60)

# Publish sensor data
topic = "sensors/temperature/pune_central/temp_001"
data = {
    "value": 28.5,
    "unit": "C",
    "timestamp": datetime.now().isoformat(),
    "quality_score": 0.95
}

client.publish(topic, json.dumps(data))
client.loop_forever()
```

---

## 📊 Data Validation

### Sensor Value Ranges
The system validates incoming sensor data against expected ranges:

| Sensor Type | Min Value | Max Value | Unit |
|-------------|-----------|-----------|------|
| temperature | -50 | 60 | °C |
| humidity | 0 | 100 | % |
| pm25 | 0 | 1000 | µg/m³ |
| pm10 | 0 | 2000 | µg/m³ |
| co2 | 300 | 5000 | ppm |
| noise | 20 | 120 | dB |
| pressure | 800 | 1200 | hPa |
| wind_speed | 0 | 50 | m/s |

### Quality Scoring
Data quality is automatically calculated based on:
- **Completeness**: Presence of required fields
- **Validity**: Values within expected ranges
- **Consistency**: Comparison with nearby sensors
- **Timeliness**: Data freshness

---

## 🔧 Error Handling

### Error Response Format
```json
{
  "error": "Invalid sensor type",
  "code": "INVALID_SENSOR_TYPE",
  "details": {
    "received": "invalid_type",
    "supported": ["temperature", "humidity", "pm25", "pm10"]
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Common Error Codes
- `MISSING_REQUIRED_FIELD`: Required field not provided
- `INVALID_SENSOR_TYPE`: Unsupported sensor type
- `INVALID_LOCATION`: Unknown location ID
- `VALUE_OUT_OF_RANGE`: Sensor value outside valid range
- `INVALID_TIMESTAMP`: Malformed timestamp
- `DATABASE_ERROR`: Internal database error

---

## 📈 Rate Limits

### Current Limits
- **HTTP API**: 1000 requests per hour per IP
- **MQTT**: 10,000 messages per hour per client

### Rate Limit Headers
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642248000
```

---

## 🔮 Future API Features

### Planned Endpoints
- `GET /api/predictions/{location_id}` - Get predictions for location
- `GET /api/data/{location_id}` - Get historical data
- `POST /api/alerts` - Configure alerts
- `GET /api/health` - System health check

### Authentication (Coming Soon)
```bash
curl -X POST http://localhost:5000/api/sensor-data \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"sensor_type": "temperature", ...}'
```

---

## 📞 Support

For API support:
- Create an issue on GitHub
- Check the troubleshooting section in README.md
- Review system logs for error details