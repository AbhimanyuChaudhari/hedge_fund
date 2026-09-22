import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import signal
from data.ingest.tick_collector import run_collector

def shutdown(sig, frame):
    print("\nShutting down...")
    sys.exit(0)

signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)

if __name__ == '__main__':
    print('=== Hedge Fund Live Pipeline ===')
    print('Architecture: WebSocket → memory → S3 (no Redis)')
    run_collector()