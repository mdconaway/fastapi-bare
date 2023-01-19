from typing import AsyncGenerator, Union
from sqlalchemy.orm import sessionmaker
from app.config import adapters
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import text


class PostgresqlAdapter:
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
            future=True,
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    # Since this returns an async generator, to use it elsewhere, it
    # should be invoked using the following syntax.
    #
    # async for session in postgresql.getSession(): session
    #
    # which will iterate through the generator context and yield the
    # product into a local variable named session.
    # Coding this method in this way also means classes interacting
    # with the adapter dont have to handle commiting thier
    # transactions, or rolling them back. It will happen here after
    # the yielded context cedes control of the event loop back to
    # the adapter. If the database explodes, the rollback happens.
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

    async def addPostgresqlExtension(self) -> None:
        async for session in self.getSession():
            session
        query = text("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        return await session.execute(query)


postgresql = PostgresqlAdapter()
