import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import redis
import json
from config.settings import settings

class RedisClient:
    def __init__(self):
        self.client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True
        )
        self.client.ping()

    def store_tick(self, symbol: str, tick: dict):
        key = f"tick:{symbol}"
        self.client.setex(key, 3600, json.dumps(tick))

    def get_tick(self, symbol: str) -> dict:
        key = f"tick:{symbol}"
        data = self.client.get(key)
        return json.loads(data) if data else None

    def push_to_stream(self, symbol: str, tick: dict):
        key = f"stream:{symbol}"
        self.client.xadd(key, tick, maxlen=1000)

    def get_stream(self, symbol: str, count: int = 100) -> list:
        key = f"stream:{symbol}"
        return self.client.xrevrange(key, count=count)

    def store_orderbook(self, symbol: str, depth: dict):
        key = f"orderbook:{symbol}"
        self.client.setex(key, 3600, json.dumps(depth))

    def get_orderbook(self, symbol: str) -> dict:
        key = f"orderbook:{symbol}"
        data = self.client.get(key)
        return json.loads(data) if data else None