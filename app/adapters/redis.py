import redis as redis_
from typing import Union
from redis import Redis
from app.config import adapters


class RedisClient:
    client: Union[Redis, None] = None
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_MAX_CONNECTIONS: int

    def __init__(
        self, redis_host="localhost", redis_port=6379, redis_max_connections=10
    ):
        self.REDIS_HOST = redis_host
        self.REDIS_PORT = redis_port
        self.REDIS_MAX_CONNECTIONS = redis_max_connections

    def getClient(self):
        if not self.client:
            self.client = redis_.from_url(
                f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}",
                max_connections=self.REDIS_MAX_CONNECTIONS,
                encoding="utf8",
                decode_responses=True,
            )
        return self.client


redis = RedisClient(
    redis_host=adapters.REDIS_HOST,
    redis_port=adapters.REDIS_PORT,
    redis_max_connections=adapters.REDIS_MAX_CONNECTIONS,
)
