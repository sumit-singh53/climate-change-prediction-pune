#!/usr/bin/env python3
"""
Launcher script for the Pune Climate Dashboard
Run this script to start the Streamlit dashboard
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit dashboard"""
    
    print("🌡️ PUNE CLIMATE DASHBOARD LAUNCHER")
    print("=" * 50)
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    dashboard_path = script_dir / "streamlit_dashboard.py"
    
    # Check if dashboard file exists
    if not dashboard_path.exists():
        print(f"❌ Dashboard file not found: {dashboard_path}")
        sys.exit(1)
    
    print(f"📁 Dashboard location: {dashboard_path}")
    print("🚀 Starting Streamlit dashboard...")
    print("🌐 Dashboard will open in your browser at: http://localhost:8501")
    print("\n💡 Tips:")
    print("   • Use Ctrl+C to stop the dashboard")
    print("   • Refresh browser if dashboard doesn't load immediately")
    print("   • Check terminal for any error messages")
    print("\n" + "=" * 50)
    
    try:
        # Launch Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ]
        
        subprocess.run(cmd, cwd=script_dir)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Dashboard stopped by user")
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install it with: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()