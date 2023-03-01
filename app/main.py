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

"""
from app.resources.user import resource as UserResource
from app.resources.post import resource as PostResource
from app.resources.group import resource as GroupResource
"""

logger = logging.getLogger(__name__)
redis_client = redis.getClient()

app = FastAPI(title=general.PROJECT_NAME, version=general.API_VERSION)

# As of fastapi 0.91.0+, all middlewares have to be defined immediately
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

# This enables "Depends" to access the DB from controllers
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


@app.on_event("startup")
async def bootstrap():
    await postgresql.addPostgresqlExtension()
    # FastAPICache.init(RedisBackend(redis_client), prefix="fastapi-cache")

    # Because of how fastapi and sqlalchemy populate the relationship mappers, the CRUD router
    # can't be fully loaded until after the fastapi server starts. Make sure you only mount
    # the ApplicationRouter in the bootstrapper. Fortunately, routers can be added lazily, which
    # forces fastapi to re-index the routes and update the openapi.json.
    app.include_router(ApplicationRouter)

    logger.info(f"{general.PROJECT_NAME}, {general.API_VERSION}: Bootstrap complete")
    # You can do any init hooks below
    """
    new_user = await UserResource.repository.create(data=UserResource.repository.model(
       first_name="bilbo",
       last_name="baggins",
       email="bilbo@baggins.net",
       hashed_password="83hpio;a8dsnv900q9j309nq3ap97hq9hf;84pt9q84fo;a8ehf"
    ))

    new_post = await PostResource.repository.create(data=PostResource.repository.model(
      user_id=new_user.id,
      content="Bilbo was here"
    ))

    new_group1 = await GroupResource.repository.create(data=GroupResource.repository.model(
      name="Bilbo's Team"
    ))

    new_group2 = await GroupResource.repository.create(data=GroupResource.repository.model(
      name="Mordor's Team"
    ))

    total_relations = await UserResource.repository.set_many_many_relations(
        id=new_user.id,
        relation="groups",
        relations=[new_group1.id,new_group2.id],
    )
    print(total_relations)
    total_altered = await UserResource.repository.set_one_many_relations(
        id=new_user.id,
        relation=UserResource._relations["posts"],
        relations=[new_post.id],
    )
    print(total_altered)
    """
