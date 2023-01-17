from app.config._base import Base
from pydantic import PostgresDsn, validator, EmailStr
from typing import Optional, Dict, Any, Union


class Adapters(Base):
    DATABASE_USER: str
    DATABASE_PASSWORD: str
    DATABASE_HOST: str
    DATABASE_PORT: Union[int, str]
    DATABASE_NAME: str
    REDIS_HOST: str
    REDIS_PORT: str
    DB_POOL_SIZE = 83
    WEB_CONCURRENCY = 9
    POOL_SIZE = max(DB_POOL_SIZE // WEB_CONCURRENCY, 5)
    MAX_OVERFLOW = 64
    ASYNC_DATABASE_URI: Optional[str]

    @validator("ASYNC_DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        return PostgresDsn.build(
            scheme="postgresql+asyncpg",
            user=values.get("DATABASE_USER"),
            password=values.get("DATABASE_PASSWORD"),
            host=values.get("DATABASE_HOST"),
            port=str(values.get("DATABASE_PORT")),
            path=f"/{values.get('DATABASE_NAME') or ''}",
        )

    FIRST_SUPERUSER_EMAIL: EmailStr
    FIRST_SUPERUSER_PASSWORD: str


adapters = Adapters()
