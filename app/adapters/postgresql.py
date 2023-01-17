from typing import AsyncGenerator, Union
from sqlalchemy.orm import sessionmaker
from app.config import adapters
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlmodel.ext.asyncio.session import AsyncSession

"""
Something about this logic... seems like the session might die
"""


class AsyncDatabaseSession:
    def __init__(self):
        self.engine = create_async_engine(
            adapters.ASYNC_DATABASE_URI,
            echo=True,
            future=True,
            pool_size=adapters.POOL_SIZE,
            max_overflow=adapters.MAX_OVERFLOW,
        )
        self.session = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )()

    def __getattr__(self, name):
        return getattr(self.session, name)

    async def commitOrRollback(self):
        try:
            await self.commit()
        except Exception:
            await self.rollback()
            raise


postgresql = AsyncDatabaseSession()

"""
class PostgresqlAdapter:
    connect_args: dict[str, bool] = {"check_same_thread": False}
    engine: Union[AsyncEngine, None] = None
    sessionLocal = None

    def __init__(self):
        self.engine = create_async_engine(
            adapters.ASYNC_DATABASE_URI,
            echo=True,
            future=True,
            pool_size=adapters.POOL_SIZE,
            max_overflow=adapters.MAX_OVERFLOW,
        )
        self.sessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )()

    async def getSession(self) -> AsyncGenerator[AsyncSession, None]:
        async with self.sessionLocal() as session:
            try:
                yield session
                try:
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise
            finally:
                await session.close()


postgresql = PostgresqlAdapter()
"""
