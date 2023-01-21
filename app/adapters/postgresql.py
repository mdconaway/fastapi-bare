from app.config import adapters
from app.utils.cruddy import PostgresqlAdapter

postgresql = PostgresqlAdapter(
    connection_uri=adapters.DATABASE_URI,
    pool_size=adapters.DATABASE_POOL_SIZE,
    max_overflow=adapters.DATABASE_MAX_OVERFLOW,
)
