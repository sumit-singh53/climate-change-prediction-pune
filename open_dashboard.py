#!/usr/bin/env python3
"""
Quick script to open the climate prediction dashboard
"""

import webbrowser
import time
import requests

def check_dashboard_status(url):
    """Check if dashboard is running"""
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    print("🌍 Climate Prediction Dashboard Launcher")
    print("=" * 50)
    
    # Dashboard URLs
    main_dashboard = "http://localhost:8501"
    enhanced_dashboard = "http://localhost:8502"
    
    print("🔍 Checking dashboard status...")
    
    # Check main dashboard
    if check_dashboard_status(main_dashboard):
        print(f"✅ Main Dashboard: {main_dashboard}")
        main_running = True
    else:
        print(f"❌ Main Dashboard: {main_dashboard} (not running)")
        main_running = False
    
    # Check enhanced dashboard
    if check_dashboard_status(enhanced_dashboard):
        print(f"✅ Enhanced Dashboard: {enhanced_dashboard}")
        enhanced_running = True
    else:
        print(f"❌ Enhanced Dashboard: {enhanced_dashboard} (not running)")
        enhanced_running = False
    
    print("\n" + "=" * 50)
    
    if main_running or enhanced_running:
        print("🚀 Opening dashboard(s) in your browser...")
        
        if main_running:
            print(f"📊 Opening Main Dashboard: {main_dashboard}")
            webbrowser.open(main_dashboard)
            time.sleep(2)
        
        if enhanced_running:
            print(f"🌟 Opening Enhanced Dashboard: {enhanced_dashboard}")
            webbrowser.open(enhanced_dashboard)
        
        print("\n✅ Dashboard(s) opened successfully!")
        print("\n💡 Features you can explore:")
        print("   📈 Real-time environmental data")
        print("   🗺️ Interactive location maps")
        print("   🔮 ML predictions and forecasts")
        print("   📊 Data quality metrics")
        print("   ⚙️ Dashboard controls and filters")
        
    else:
        print("❌ No dashboards are currently running.")
        print("\n💡 To start the dashboards, run:")
        print("   python run_system.py --mode dashboard")
        print("   or")
        print("   streamlit run src/realtime_dashboard.py")

if __name__ == "__main__":
    main()