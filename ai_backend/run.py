import uvicorn
import os
import sys
import psutil
import signal
from app.main import app

def kill_process_on_port(port):
    for proc in psutil.process_iter(['pid', 'name', 'connections']):
        try:
            for conn in proc.info['connections']:
                if conn.laddr.port == port:
                    os.kill(proc.info['pid'], signal.SIGTERM)
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

if __name__ == "__main__":
    # Add project root to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)

    PORT = 8000
    kill_process_on_port(PORT)
    
    # Start server with proper module path
    uvicorn.run(app, 
                host="0.0.0.0", 
                port=PORT, 
                reload=True,
                reload_dirs=[project_root])