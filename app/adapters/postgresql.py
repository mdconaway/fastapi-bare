from app.config import adapters
from fastapi_cruddy_framework import PostgresqlAdapter

postgresql = PostgresqlAdapter(
    connection_uri=adapters.DATABASE_URI,
    pool_size=adapters.DATABASE_POOL_SIZE,
    max_overflow=adapters.DATABASE_MAX_OVERFLOW,
)
