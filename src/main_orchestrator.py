"""
Main Orchestrator for Enhanced Climate and AQI Prediction System
Coordinates data collection, model training, IoT integration, and dashboard
"""

import asyncio
import threading
import schedule
import time
import logging
from datetime import datetime
from typing import Dict, Any

from config import MODEL_CONFIG, IOT_CONFIG
from enhanced_data_collector import EnhancedDataCollector
from iot_integration import IoTDataCollector, SensorSimulator
from advanced_ml_models import AdvancedMLModels
from realtime_dashboard import RealtimeDashboard

class SystemOrchestrator:
    """Main orchestrator for the entire system"""
    
    def __init__(self):
        # Initialize components
        self.data_collector = EnhancedDataCollector()
        self.iot_collector = IoTDataCollector()
        self.ml_models = AdvancedMLModels()
        self.dashboard = RealtimeDashboard()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('outputs/logs/system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # System status
        self.system_status = {
            'data_collection': 'stopped',
            'iot_collection': 'stopped',
            'model_training': 'stopped',
            'dashboard': 'stopped'
        }
    
    async def run_data_collection(self):
        """Run periodic data collection"""
        self.logger.info("Starting data collection service...")
        self.system_status['data_collection'] = 'running'
        
        try:
            while True:
                self.logger.info("Running scheduled data collection...")
                await self.data_collector.run_data_collection(days_back=1)
                
                # Wait for next collection cycle (every hour)
                await asyncio.sleep(3600)
                
        except Exception as e:
            self.logger.error(f"Data collection error: {e}")
            self.system_status['data_collection'] = 'error'
    
    def run_iot_collection(self):
        """Run IoT data collection service"""
        self.logger.info("Starting IoT collection service...")
        self.system_status['iot_collection'] = 'running'
        
        try:
            # Start MQTT listener
            self.iot_collector.start_mqtt_listener()
            
            # Start HTTP server in a separate thread
            def start_http_server():
                self.iot_collector.start_http_server(host='0.0.0.0', port=5000)
            
            http_thread = threading.Thread(target=start_http_server, daemon=True)
            http_thread.start()
            
            self.logger.info("IoT collection services started")
            
        except Exception as e:
            self.logger.error(f"IoT collection error: {e}")
            self.system_status['iot_collection'] = 'error'
    
    def run_model_training(self):
        """Run periodic model training"""
        self.logger.info("Starting model training service...")
        self.system_status['model_training'] = 'running'
        
        def train_models():
            try:
                self.logger.info("Running scheduled model training...")
                self.ml_models.train_all_models(days_back=180)
                self.logger.info("Model training completed successfully")
            except Exception as e:
                self.logger.error(f"Model training error: {e}")
                self.system_status['model_training'] = 'error'
        
        # Schedule model training
        schedule.every().day.at("02:00").do(train_models)  # Train at 2 AM daily
        schedule.every().week.do(train_models)  # Also train weekly
        
        # Run initial training
        train_models()
        
        # Keep scheduler running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def run_dashboard(self):
        """Run the dashboard service"""
        self.logger.info("Starting dashboard service...")
        self.system_status['dashboard'] = 'running'
        
        try:
            # Run Streamlit dashboard
            import subprocess
            import os
            
            dashboard_path = os.path.join(os.path.dirname(__file__), 'realtime_dashboard.py')
            subprocess.run(['streamlit', 'run', dashboard_path, '--server.port=8501'])
            
        except Exception as e:
            self.logger.error(f"Dashboard error: {e}")
            self.system_status['dashboard'] = 'error'
    
    def start_sensor_simulation(self, duration_minutes: int = 1440):  # 24 hours default
        """Start sensor simulation for testing"""
        self.logger.info("Starting sensor simulation...")
        
        simulator = SensorSimulator(self.iot_collector)
        
        def run_simulation():
            simulator.simulate_sensors(duration_minutes)
        
        sim_thread = threading.Thread(target=run_simulation, daemon=True)
        sim_thread.start()
        
        return simulator
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'timestamp': datetime.now().isoformat(),
            'services': self.system_status,
            'uptime': time.time() - self.start_time if hasattr(self, 'start_time') else 0
        }
    
    def start_all_services(self, simulate_sensors: bool = True):
        """Start all system services"""
        self.start_time = time.time()
        self.logger.info("Starting Enhanced Climate & AQI Prediction System...")
        
        # Start IoT collection service
        iot_thread = threading.Thread(target=self.run_iot_collection, daemon=True)
        iot_thread.start()
        
        # Start sensor simulation if requested
        if simulate_sensors:
            self.start_sensor_simulation()
        
        # Start data collection service
        data_collection_thread = threading.Thread(
            target=lambda: asyncio.run(self.run_data_collection()), 
            daemon=True
        )
        data_collection_thread.start()
        
        # Start model training service
        training_thread = threading.Thread(target=self.run_model_training, daemon=True)
        training_thread.start()
        
        # Give services time to start
        time.sleep(5)
        
        # Start dashboard (this will block)
        self.run_dashboard()
    
    def stop_all_services(self):
        """Stop all system services"""
        self.logger.info("Stopping all services...")
        
        for service in self.system_status:
            self.system_status[service] = 'stopped'
        
        self.logger.info("All services stopped")


def main():
    """Main function to start the system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Climate & AQI Prediction System')
    parser.add_argument('--mode', choices=['full', 'dashboard', 'data-collection', 'training'], 
                       default='full', help='System mode to run')
    parser.add_argument('--simulate-sensors', action='store_true', 
                       help='Start sensor simulation for testing')
    parser.add_argument('--no-dashboard', action='store_true', 
                       help='Skip dashboard startup')
    
    args = parser.parse_args()
    
    orchestrator = SystemOrchestrator()
    
    try:
        if args.mode == 'full':
            orchestrator.start_all_services(simulate_sensors=args.simulate_sensors)
        
        elif args.mode == 'dashboard':
            orchestrator.run_dashboard()
        
        elif args.mode == 'data-collection':
            asyncio.run(orchestrator.run_data_collection())
        
        elif args.mode == 'training':
            orchestrator.run_model_training()
    
    except KeyboardInterrupt:
        orchestrator.logger.info("Received shutdown signal")
        orchestrator.stop_all_services()
    
    except Exception as e:
        orchestrator.logger.error(f"System error: {e}")
        orchestrator.stop_all_services()


if __name__ == "__main__":
    main()