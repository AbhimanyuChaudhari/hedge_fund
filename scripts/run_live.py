import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import signal
from data.ingest.tick_collector import TickCollector
from data.ingest.candle_builder import CandleBuilder
from data.ingest.instrument_manager import InstrumentManager

def main():
    manager           = InstrumentManager()
    manager.get_all_instruments()
    tokens, token_map = manager.get_tokens_and_symbols()

    print(f"Trading {len(tokens)} instruments:")
    for t, s in token_map.items():
        print(f"  {s} ({t})")

    collector = TickCollector(instrument_tokens=tokens)
    builder   = CandleBuilder(token_map=token_map, flush_interval=60)

    def shutdown(sig, frame):
        print("\nShutting down...")
        collector.stop()
        builder.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    collector.start()
    builder.start()

    print("Live pipeline running. Press Ctrl+C to stop.")
    while True:
        time.sleep(1)

if __name__ == '__main__':
    main()