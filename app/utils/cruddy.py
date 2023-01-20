# This is a candidate to become a library. You're welcome Python community.
# Love,
# A Sails / Ember lover.

import math
from fastapi import APIRouter, Path, Query
from sqlalchemy import update as _update, delete as _delete, or_, text, func, column
from sqlalchemy.sql import select
from sqlalchemy.orm import sessionmaker, declared_attr
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlmodel import text
from sqlmodel.ext.asyncio.session import AsyncSession
from typing import AsyncGenerator, Union, TypeVar, Optional, Generic, List  # TypedDict
from pydantic.generics import GenericModel
from sqlmodel import Field, SQLModel as _SQLModel
from datetime import datetime

# Look into this for making generic factories??
# https://shanenullain.medium.com/abstract-factory-in-python-with-generic-typing-b9ceca2bf89e

T = TypeVar("T")


class BulkDTO(GenericModel, Generic[T]):
    total_pages: int
    total_records: int
    data: List


class MetaObject(GenericModel, Generic[T]):
    page_number: int
    page_size: int
    total_pages: int
    total_records: int


class PageResponse(GenericModel, Generic[T]):
    # The response for a pagination query.
    message: str
    meta: MetaObject
    data: List[T]


class ResponseSchema(_SQLModel):
    # The response for a single object return
    message: str
    data: Optional[T] = None


class SQLModel(_SQLModel):
    @declared_attr  # type: ignore
    def __tablename__(cls) -> str:
        return cls.__name__


class BaseIdModel(SQLModel):
    id: Optional[int] = Field(
        default=None,
        primary_key=True,
        index=True,
        nullable=False,
    )
    updated_at: Optional[datetime] = Field(
        default_factory=datetime.utcnow, sa_column_kwargs={"onupdate": datetime.utcnow}
    )
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)


class ExampleUpdate(SQLModel):
    updateable_field: str


class ExampleCreate(ExampleUpdate):
    create_only_field: str


class Example(BaseIdModel, ExampleCreate, table=True):
    db_only_field: str


def Controller(
    prefix="/example",
    tags=["example"],
    single_schema=ResponseSchema,
    many_schema=PageResponse,
    meta_schema=MetaObject,
    update_model=ExampleUpdate,
    create_model=ExampleCreate,
    id_type=int,
    repository=...,
) -> APIRouter:

    controller = APIRouter(prefix=prefix, tags=tags)

    @controller.post("", response_model=single_schema, response_model_exclude_none=True)
    async def create(data: create_model):
        await repository.create(data=data)
        return single_schema(message="Successfully created data !")

    @controller.patch(
        "/{id}", response_model=single_schema, response_model_exclude_none=True
    )
    async def update(id: id_type = Path(..., alias="id"), *, data: update_model):
        await repository.update(id=id, data=data)
        return single_schema(message="Successfully updated data !")

    @controller.delete(
        "/{id}", response_model=single_schema, response_model_exclude_none=True
    )
    async def delete(
        id: id_type = Path(..., alias="id"),
    ):
        await repository.delete(id=id)
        return single_schema(message="Successfully deleted data !")

    @controller.get(
        "/{id}", response_model=single_schema, response_model_exclude_none=True
    )
    async def get_by_id(id: id_type = Path(..., alias="id")):
        data = await repository.get_by_id(id=id)
        return single_schema(message="Successfully fetch data by id !", data=data)

    @controller.get("", response_model=many_schema, response_model_exclude_none=True)
    async def get_all(
        page: int = 1,
        limit: int = 10,
        columns: str = Query(None, alias="columns"),
        sort: str = Query(None, alias="sort"),
        filter: str = Query(None, alias="filter"),
    ):
        result: BulkDTO = await repository.get_all(
            page=page, limit=limit, columns=columns, sort=sort, filter=filter
        )
        meta = {
            "page_number": page,
            "page_size": limit,
            "total_pages": result.total_pages,
            "total_records": result.total_records,
        }
        return many_schema(
            message="Successfully fetch data list !",
            meta=meta_schema(**meta),
            data=result.data,
        )

    return controller


class AbstractRepository:
    def __init__(
        self, adapter=..., update_model=..., create_model=..., model=..., id_type=int
    ):
        self.adapter = adapter
        self.update_model = update_model
        self.create_model = create_model
        self.model = model
        self.id_type = id_type

        async def create(data: create_model):
            # create user data
            session = await adapter.getSession()
            await session.add(model(**data.dict()))
            # return a value?

        self.create = create

        async def get_by_id(id: id_type):
            # retrieve user data by id
            session = await adapter.getSession()
            query = select(model).where(model.id == id)
            result = (await session.execute(query)).scalar_one_or_none()
            return result

        self.get_by_id = get_by_id

        async def update(id: id_type, data: update_model):
            session = await adapter.getSession()
            # update user data
            query = (
                _update(model)
                .where(model.id == id)
                .values(**data.dict())
                .execution_options(synchronize_session="fetch")
            )
            await session.execute(query)
            # return a value?

        self.update = update

        async def delete(id: id_type):
            # delete user data by id
            session = await adapter.getSession()
            query = _delete(model).where(model.id == id)
            await session.execute(query)
            # return a value?

        self.delete = delete

        async def get_all(
            page: int = 1,
            limit: int = 10,
            columns: str = None,
            sort: str = None,
            filter: str = None,
        ):
            session = await adapter.getSession()
            query = select(from_obj=model, columns="*")

            # select columns dynamically
            if columns is not None and columns != "all":
                # we need column format data like this --> [column(id),column(name),column(sex)...]
                query = select(
                    from_obj=model,
                    columns=list(map(lambda x: column(x), columns.split("-"))),
                )

            # select filter dynamically
            if filter is not None and filter != "null":
                # we need filter format data like this  --> {'name': 'an','country':'an'}
                # convert string to dict format
                criteria = dict(x.split("*") for x in filter.split("-"))
                criteria_list = []
                # check every key in dict. are there any table attributes that are the same as the dict key ?
                for attr, value in criteria.items():
                    _attr = getattr(model, attr)
                    # filter format
                    search = "%{}%".format(value)
                    # criteria list
                    criteria_list.append(_attr.like(search))
                query = query.filter(or_(*criteria_list))

            # select sort dynamically
            if sort is not None and sort != "null":
                # we need sort format data like this --> ['id','name']
                query = query.order_by(text(",".join(sort.split("-"))))

            # count query
            count_query = select(func.count(1)).select_from(query)
            offset_page = page - 1
            # pagination
            query = query.offset(offset_page * limit).limit(limit)
            # total record
            total_record = (await session.execute(count_query)).scalar() or 0
            # total page
            total_page = math.ceil(total_record / limit)

            # result
            result = (await session.execute(query)).fetchall()

            # possible pass in outside functions to map/alter data?
            return BulkDTO(
                total_pages=total_page, total_records=total_record, data=result
            )

        self.get_all = get_all


# The default adapter for CruddyResource
# Currently has a bug. yield isn't halting properly
# Queries are being sent AFTER the session has been told to close
# which is creating non-fatal error messages over time.
class PostgresqlAdapter:
    engine: Union[AsyncEngine, None] = None
    sessionLocal = None

    def __init__(self, connection_uri="", pool_size=4, max_overflow=64):
        self.engine = create_async_engine(
            connection_uri,
            echo=True,
            future=True,
            pool_size=pool_size,
            max_overflow=max_overflow,
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
    async def _getSession(self) -> AsyncGenerator[AsyncSession, None]:
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

    # streamlines external calls to getSession to allow simple await
    async def getSession(self) -> Union[AsyncSession, None]:
        async for session in self._getSession():
            session
        return session

    async def addPostgresqlExtension(self) -> None:
        session = await self.getSession()
        query = text("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        await session.execute(query)


class Resource:
    adapter: PostgresqlAdapter = None
    repository: AbstractRepository = None
    controller: APIRouter = None

    def __init__(
        self,
        prefix="/example",
        tags=["example"],
        response_single_schema=ResponseSchema,
        response_many_schema=PageResponse,
        response_meta_schema=MetaObject,
        resource_update_model=ExampleUpdate,
        resource_create_model=ExampleCreate,
        resource_model=Example,
        id_type=int,
        adapter=None,
        connection_uri="",
        pool_size=4,
        max_overflow=64,
    ):
        if None == adapter:
            self.adapter = PostgresqlAdapter(connection_uri, pool_size, max_overflow)
        else:
            self.adapter = adapter

        self.repository = AbstractRepository(
            adapter=self.adapter,
            update_model=resource_update_model,
            create_model=resource_create_model,
            model=resource_model,
            id_type=id_type,
        )

        self.controller = Controller(
            prefix=prefix,
            tags=tags,
            single_schema=response_single_schema,
            many_schema=response_many_schema,
            meta_schema=response_meta_schema,
            update_model=resource_update_model,
            create_model=resource_create_model,
            id_type=id_type,
            repository=self.repository,
        )
