from app.config._base import Base
from pydantic import validator, AnyHttpUrl
from typing import Optional
import secrets


class Sessions(Base):
    SESSION_COOKIE_NAME: str
    SESSION_SECRET_KEY: str
    SESSION_MAX_AGE: int

    @validator("SESSION_SECRET_KEY", pre=True)
    def validate_session_secret(cls, v: Optional[str]) -> str:
        if isinstance(v, str):
            return v
        return secrets.token_urlsafe(32)


sessions = Sessions()
