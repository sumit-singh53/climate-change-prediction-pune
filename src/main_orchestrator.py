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

from config import MODEL_CONFIG, REALTIME_CONFIG
from enhanced_data_collector import EnhancedDataCollector
from realtime_data_collector import RealtimeDataCollector
from advanced_ml_models import AdvancedMLModels
from realtime_dashboard import RealtimeDashboard

class SystemOrchestrator:
    """Main orchestrator for the entire system"""
    
    def __init__(self):
        # Initialize components
        self.data_collector = EnhancedDataCollector()
        self.realtime_collector = RealtimeDataCollector()
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
            'realtime_collection': 'stopped',
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
    
    async def run_realtime_collection(self):
        """Run real-time data collection service"""
        self.logger.info("Starting real-time data collection service...")
        self.system_status['realtime_collection'] = 'running'
        
        try:
            # Start continuous real-time data collection
            await self.realtime_collector.run_continuous_collection()
            
        except Exception as e:
            self.logger.error(f"Real-time collection error: {e}")
            self.system_status['realtime_collection'] = 'error'
    
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
    
    async def start_initial_data_collection(self):
        """Start initial data collection to populate database"""
        self.logger.info("Starting initial data collection...")
        
        try:
            # Collect initial data for all locations
            weather_data, air_quality_data = await self.realtime_collector.collect_all_locations_data()
            
            if weather_data or air_quality_data:
                self.realtime_collector.save_to_database(weather_data, air_quality_data)
                self.logger.info(f"Initial data collection completed: {len(weather_data)} weather, {len(air_quality_data)} air quality records")
            else:
                self.logger.warning("No initial data collected")
                
        except Exception as e:
            self.logger.error(f"Initial data collection error: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'timestamp': datetime.now().isoformat(),
            'services': self.system_status,
            'uptime': time.time() - self.start_time if hasattr(self, 'start_time') else 0
        }
    
    def start_all_services(self, collect_initial_data: bool = True):
        """Start all system services"""
        self.start_time = time.time()
        self.logger.info("Starting Enhanced Climate & AQI Prediction System...")
        
        # Start initial data collection if requested
        if collect_initial_data:
            asyncio.run(self.start_initial_data_collection())
        
        # Start real-time collection service
        realtime_thread = threading.Thread(
            target=lambda: asyncio.run(self.run_realtime_collection()), 
            daemon=True
        )
        realtime_thread.start()
        
        # Start historical data collection service
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
    parser.add_argument('--collect-initial-data', action='store_true', 
                       help='Collect initial data on startup')
    parser.add_argument('--no-dashboard', action='store_true', 
                       help='Skip dashboard startup')
    
    args = parser.parse_args()
    
    orchestrator = SystemOrchestrator()
    
    try:
        if args.mode == 'full':
            orchestrator.start_all_services(collect_initial_data=args.collect_initial_data)
        
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