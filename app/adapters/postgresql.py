from app.config import adapters
from app.utils.cruddy import PostgresqlAdapter

postgresql = PostgresqlAdapter(
    connection_uri=adapters.ASYNC_DATABASE_URI,
    pool_size=adapters.POOL_SIZE,
    max_overflow=adapters.MAX_OVERFLOW,
)
