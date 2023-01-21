import logging
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from fastapi_async_sqlalchemy import SQLAlchemyMiddleware
from app.config import general, adapters, http, sessions
from app.adapters import redis, postgresql
from app.router import application as ApplicationRouter
from app.middleware import RequestLogger
from starlette_session import SessionMiddleware
from starlette_session.backends import BackendType
from datetime import timedelta

app = FastAPI(title=general.PROJECT_NAME, version=general.API_VERSION)


@app.on_event("startup")
async def bootstrap():
    logger = logging.getLogger(__name__)
    redis_client = await redis.getClient()
    await postgresql.addPostgresqlExtension()

    app.add_middleware(RequestLogger)

    # Set all CORS origins enabled
    if http.HTTP_CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in http.HTTP_CORS_ORIGINS],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Add session storage/retrieval to incoming requests
    app.add_middleware(
        SessionMiddleware,
        secret_key=sessions.SESSION_SECRET_KEY,
        cookie_name=sessions.SESSION_COOKIE_NAME,
        backend_type=BackendType.redis,
        backend_client=redis_client,
        https_only=False,
        same_site="lax",  # lax or strict
        max_age=timedelta(days=int(sessions.SESSION_MAX_AGE)),  # in seconds
    )

    # This enabled Depends to access the DB from controllers
    app.add_middleware(
        SQLAlchemyMiddleware,
        db_url=adapters.DATABASE_URI,
        engine_args={
            "echo": False,
            "pool_pre_ping": True,
            "pool_size": adapters.DATABASE_POOL_SIZE,
            "max_overflow": adapters.DATABASE_MAX_OVERFLOW,
        },
    )

    app.include_router(ApplicationRouter)
    # FastAPICache.init(RedisBackend(redis_client), prefix="fastapi-cache")
    logger.info(
        "{0}, {1}: Bootstrap complete".format(general.PROJECT_NAME, general.API_VERSION)
    )
    # You can do any init hooks below
