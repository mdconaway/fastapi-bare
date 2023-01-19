from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from fastapi_async_sqlalchemy import db, SQLAlchemyMiddleware
from sqlmodel import text
from app.config import general, adapters, http
from app.adapters import redis
from app.router import application as ApplicationRouter

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


async def add_postgresql_extension() -> None:
    async with db():
        query = text("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        return await db.session.execute(query)


app.include_router(ApplicationRouter)

"""
You can do any init hooks below
"""


@app.on_event("startup")
async def on_startup():
    await add_postgresql_extension()
    redis_client = await redis.getClient()
    """FastAPICache.init(RedisBackend(redis_client), prefix="fastapi-cache")"""
    print("startup fastapi")
