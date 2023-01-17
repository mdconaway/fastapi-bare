import aioredis
from typing import Union
from aioredis import Redis
from app.config.adapters import adapters


class RedisClient:
    client: Union[Redis, None] = None

    async def getClient(self):
        if not self.client:
            self.client = await aioredis.from_url(
                f"redis://{adapters.REDIS_HOST}:{adapters.REDIS_PORT}",
                max_connections=10,
                encoding="utf8",
                decode_responses=True,
            )
        return self.client


redis = RedisClient()
