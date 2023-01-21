from app.config._base import Base
from pydantic import PostgresDsn, validator, EmailStr
from typing import Optional, Dict, Any, Union


class Adapters(Base):
    # Postgresql config
    DATABASE_USER: str
    DATABASE_PASSWORD: str
    DATABASE_HOST: str
    DATABASE_PORT: Union[int, str]
    DATABASE_NAME: str
    DATABASE_POOL_SIZE: int
    DATABASE_MAX_OVERFLOW: int
    DATABASE_URI: Optional[str]
    DATABASE_SUPERUSER_EMAIL: EmailStr
    DATABASE_SUPERUSER_PASSWORD: str

    @validator("DATABASE_URI", pre=True)
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

    # Redis config
    REDIS_HOST: str
    REDIS_PORT: str
    REDIS_MAX_CONNECTIONS: Union[int, str]


adapters = Adapters()
