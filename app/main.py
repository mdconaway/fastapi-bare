import logging
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from fastapi_async_sqlalchemy import SQLAlchemyMiddleware
from app.config import general, adapters, http
from app.adapters import redis, postgresql
from app.router import application as ApplicationRouter
from app.middleware import RequestLogger
from os import path


logging.config.fileConfig(
    path.join(path.dirname(path.abspath(__file__)), "logging.conf"),
    disable_existing_loggers=False,
)
logger = logging.getLogger(__name__)

app = FastAPI(title=general.PROJECT_NAME, version=general.API_VERSION)

app.add_middleware(
    SQLAlchemyMiddleware,
    db_url=adapters.ASYNC_DATABASE_URI,
    engine_args={
        "echo": False,
        "pool_pre_ping": True,
        "pool_size": adapters.POOL_SIZE,
        "max_overflow": 64,
    },
)

# Set all CORS origins enabled
if http.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in http.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.add_middleware(RequestLogger)

app.include_router(ApplicationRouter)

# You can do any init hooks below
@app.on_event("startup")
async def on_startup():
    await postgresql.addPostgresqlExtension()
    redis_client = await redis.getClient()
    # FastAPICache.init(RedisBackend(redis_client), prefix="fastapi-cache")
    print("startup fastapi")
