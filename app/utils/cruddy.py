# This is a candidate to become a library. You're welcome Python community.
# Love,
# A Sails / Ember lover.

import math
import os
import sys
import glob
import importlib.util
from os import path
from fastapi import APIRouter, Path, Query, Depends
from sqlalchemy import (
    update as _update,
    delete as _delete,
    or_,
    and_,
    not_,
    text,
    func,
    column,
)
from sqlalchemy.sql import select
from sqlalchemy.orm import sessionmaker, declared_attr
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from contextlib import asynccontextmanager
from sqlmodel import text
from sqlmodel.ext.asyncio.session import AsyncSession
from types import ModuleType
from typing import Union, TypeVar, Optional, Generic, List, Dict
from pydantic.generics import GenericModel
from pydantic.types import Json
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
    page: int
    limit: int
    pages: int
    records: int


class PageResponse(GenericModel, Generic[T]):
    # The response for a pagination query.
    meta: MetaObject
    data: List[T]


class ResponseSchema(_SQLModel):
    # The response for a single object return
    data: Optional[T] = None


class SQLModel(_SQLModel):
    @declared_attr  # type: ignore
    def __tablename__(cls) -> str:
        return cls.__name__


class BaseIntIdModel(SQLModel):
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


class Example(BaseIntIdModel, ExampleCreate, table=True):
    db_only_field: str


def asemblePolicies(*args: (List)):
    merged = []
    for policy_set in args:
        for individual_policy in policy_set:
            merged.append(Depends(individual_policy))
    return merged


def Controller(
    repository=...,
    prefix="/example",
    tags=["example"],
    id_type=int,
    single_schema=ResponseSchema,
    many_schema=PageResponse,
    meta_schema=MetaObject,
    update_model=ExampleUpdate,
    create_model=ExampleCreate,
    policies_universal=[],
    policies_create=[],
    policies_update=[],
    policies_delete=[],
    policies_get_one=[],
    policies_get_many=[],
) -> APIRouter:

    controller = APIRouter(prefix=prefix, tags=tags)

    @controller.post(
        "",
        response_model=single_schema,
        response_model_exclude_none=True,
        dependencies=asemblePolicies(policies_universal, policies_create),
    )
    async def create(data: create_model):
        data = await repository.create(data=data)
        # Add error logic?
        return single_schema(data=data)

    @controller.patch(
        "/{id}",
        response_model=single_schema,
        response_model_exclude_none=True,
        dependencies=asemblePolicies(policies_universal, policies_update),
    )
    async def update(id: id_type = Path(..., alias="id"), *, data: update_model):
        data = await repository.update(id=id, data=data)
        # Add error logic?
        return single_schema(data=data)

    @controller.delete(
        "/{id}",
        response_model=single_schema,
        response_model_exclude_none=True,
        dependencies=asemblePolicies(policies_universal, policies_delete),
    )
    async def delete(
        id: id_type = Path(..., alias="id"),
    ):
        data = await repository.delete(id=id)
        # Add error logic?
        return single_schema(data=data)

    @controller.get(
        "/{id}",
        response_model=single_schema,
        response_model_exclude_none=True,
        dependencies=asemblePolicies(policies_universal, policies_get_one),
    )
    async def get_by_id(id: id_type = Path(..., alias="id")):
        data = await repository.get_by_id(id=id)
        return single_schema(data=data)

    @controller.get(
        "",
        response_model=many_schema,
        response_model_exclude_none=True,
        dependencies=asemblePolicies(policies_universal, policies_get_many),
    )
    async def get_all(
        page: int = 1,
        limit: int = 10,
        columns: List[str] = Query(None, alias="columns"),
        sort: List[str] = Query(None, alias="sort"),
        where: Json = Query(None, alias="where"),
    ):
        result: BulkDTO = await repository.get_all(
            page=page, limit=limit, columns=columns, sort=sort, where=where
        )
        meta = {
            "page": page,
            "limit": limit,
            "pages": result.total_pages,
            "records": result.total_records,
        }
        return many_schema(
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
        self.op_map = {
            "*and": and_,
            "*or": or_,
            "*not": not_,
        }

        async def create(data: create_model):
            # create user data
            async with adapter.getSession() as session:
                record = model(**data.dict())
                await session.add(record)
            return record
            # return a value?

        self.create = create

        async def get_by_id(id: id_type):
            # retrieve user data by id
            query = select(model).where(model.id == id)
            async with adapter.getSession() as session:
                result = (await session.execute(query)).scalar_one_or_none()
            return result

        self.get_by_id = get_by_id

        async def update(id: id_type, data: update_model):
            # update user data
            query = (
                _update(model)
                .where(model.id == id)
                .values(**data.dict())
                .execution_options(synchronize_session="fetch")
            )
            async with adapter.getSession() as session:
                result = await session.execute(query)

            if result.rowcount == 1:
                return await self.get_by_id(id=id)

            return None
            # return a value?

        self.update = update

        async def delete(id: id_type):
            # delete user data by id
            record = await self.get_by_id(id=id)
            query = (
                _delete(model)
                .where(model.id == id)
                .execution_options(synchronize_session="fetch")
            )
            async with adapter.getSession() as session:
                result = await session.execute(query)

            if result.rowcount == 1:
                return record

            return None
            # return a value?

        self.delete = delete

        async def get_all(
            page: int = 1,
            limit: int = 10,
            columns: List[str] = None,
            sort: List[str] = None,
            where: Json = None,
        ):
            select_columns = (
                list(map(lambda x: column(x), columns))
                if columns is not None and columns != []
                else "*"
            )
            query = select(from_obj=model, columns=select_columns)

            # build an arbitrarily deep query with a JSON dictionary
            # a query object is a JSON object that generally looks like
            # all boolean operators, or field level operators, begin with a
            # * character. This will nearly always translate down to the sqlalchemy
            # level, where it is up to the model class to determine what operations
            # are possible on each model attribute.
            # The top level query object is an implicit AND.
            # To do an OR, the base key of the search must be *or, as below examples:
            # {"*or":{"first_name":"bilbo","last_name":"baggins"}}
            # {"*or":{"first_name":{"*contains":"bilbo"},"last_name":"baggins"}}
            # {"*or":{"first_name":{"*endswith":"bilbo"},"last_name":"baggins","*and":{"email":{"*contains":"@"},"first_name":{"*contains":"helga"}}}}
            # {"*or":{"first_name":{"*endswith":"bilbo"},"last_name":"baggins","*and":[{"email":{"*contains":"@"}},{"email":{"*contains":"helga"}}]}}
            # The following query would be an implicit *and:
            # [{"first_name":{"*endswith":"bilbo"}},{"last_name":"baggins"}]
            # As would the following query:
            # {"first_name":{"*endswith":"bilbo"},"last_name":"baggins"}
            if isinstance(where, dict) or isinstance(where, list):
                query = query.filter(and_(*self.query_forge(model=model, where=where)))

            # select sort dynamically
            if sort is not None and sort != []:
                # we need sort format data like this --> ['id asc','name desc', 'email']
                def splitter(sort_string: str):
                    parts = sort_string.split(" ")
                    getter = "asc"
                    if len(parts) == 2:
                        getter = parts[1]
                    return getattr(getattr(self.model, parts[0]), getter)

                sorts = list(map(splitter, sort))
                for field in sorts:
                    query = query.order_by(field())

            # count query
            count_query = select(func.count(1)).select_from(query)
            offset_page = page - 1
            # pagination
            query = query.offset(offset_page * limit).limit(limit)
            # total record

            async with adapter.getSession() as session:
                total_record = (await session.execute(count_query)).scalar() or 0
                # result
                result = (await session.execute(query)).fetchall()

            # possible pass in outside functions to map/alter data?
            # total page
            total_page = math.ceil(total_record / limit)
            return BulkDTO(
                total_pages=total_page, total_records=total_record, data=result
            )

        self.get_all = get_all

    # Initial, simple, query forge. Invalid attrs or ops are just dropped.
    # Improvements to make:
    # 1. Table joins for relationships.
    # 2. Make relationships searchable too!
    # 3. Maybe throw an error if a bad search field is sent? (Will help UI devs)
    def query_forge(self, model, where: Union[Dict, List[Dict]]):
        level_criteria = []
        if not (isinstance(where, list) or isinstance(where, dict)):
            return []
        if isinstance(where, list):
            list_of_lists = list(
                map(lambda x: self.query_forge(model=model, where=x), where)
            )
            for l in list_of_lists:
                level_criteria += l
            return level_criteria
        for k, v in where.items():
            isOp = False
            if k in self.op_map:
                isOp = self.op_map[k]
            if isinstance(v, dict) and isOp != False:
                level_criteria.append(isOp(*self.query_forge(model=model, where=v)))
            elif isinstance(v, list) and isOp != False:
                level_criteria.append(isOp(*self.query_forge(model=model, where=v)))
            elif not isinstance(v, dict) and not isOp and hasattr(model, k):
                level_criteria.append(getattr(model, k).like(v))
            elif (
                isinstance(v, dict)
                and not isOp
                and hasattr(model, k)
                and len(v.items()) == 1
            ):
                k2 = list(v.keys())[0]
                v2 = v[k2]
                mattr = getattr(model, k)
                if (
                    isinstance(k2, str)
                    and not isinstance(v2, dict)
                    and k2[0] == "*"
                    and hasattr(mattr, k2.replace("*", ""))
                ):
                    level_criteria.append(getattr(mattr, k2.replace("*", ""))(v2))
        return level_criteria


# The default adapter for CruddyResource
class PostgresqlAdapter:
    engine: Union[AsyncEngine, None] = None

    def __init__(self, connection_uri="", pool_size=4, max_overflow=64):
        self.engine = create_async_engine(
            connection_uri,
            echo=True,
            future=True,
            pool_size=pool_size,
            max_overflow=max_overflow,
        )

    # Since this returns an async generator, to use it elsewhere, it
    # should be invoked using the following syntax.
    #
    # async with postgresql.getSession() as session:
    #
    # which will iterate through the generator context and yield the
    # product into a local variable named session.
    # Coding this method in this way also means classes interacting
    # with the adapter dont have to handle commiting thier
    # transactions, or rolling them back. It will happen here after
    # the yielded context cedes control of the event loop back to
    # the adapter. If the database explodes, the rollback happens.

    def asyncSessionGenerator(self):
        return sessionmaker(
            autocommit=False,
            autoflush=False,
            future=True,
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    @asynccontextmanager
    async def getSession(self):
        try:
            asyncSession = self.asyncSessionGenerator()
            async with asyncSession() as session:
                yield session
                await session.commit()
        except:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def addPostgresqlExtension(self) -> None:
        query = text("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        async with self.getSession() as session:
            await session.execute(query)


# Next step: add input parameter for adding policies to each action
class Resource:
    adapter: PostgresqlAdapter = None
    repository: AbstractRepository = None
    controller: APIRouter = None

    def __init__(
        self,
        adapter=None,
        connection_uri="",
        pool_size=4,
        max_overflow=64,
        prefix="/example",
        tags=["example"],
        response_single_schema=ResponseSchema,
        response_many_schema=PageResponse,
        response_meta_schema=MetaObject,
        resource_update_model=ExampleUpdate,
        resource_create_model=ExampleCreate,
        resource_model=Example,
        id_type=int,
        policies_universal=[],
        policies_create=[],
        policies_update=[],
        policies_delete=[],
        policies_get_one=[],
        policies_get_many=[],
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
            repository=self.repository,
            prefix=prefix,
            tags=tags,
            id_type=id_type,
            single_schema=response_single_schema,
            many_schema=response_many_schema,
            meta_schema=response_meta_schema,
            update_model=resource_update_model,
            create_model=resource_create_model,
            policies_universal=policies_universal,
            policies_create=policies_create,
            policies_update=policies_update,
            policies_delete=policies_delete,
            policies_get_one=policies_get_one,
            policies_get_many=policies_get_many,
        )


def getModuleDir(application_module) -> str:
    return path.dirname(os.path.abspath(application_module.__file__))


def getDirectoryModules(
    application_module: ModuleType = ..., sub_module_path="resources"
):
    app_root = getModuleDir(application_module)
    app_root_name = path.split(app_root)[1]
    normalized_sub_path = os.path.normpath(sub_module_path)
    submodule_tokens = normalized_sub_path.split(os.sep)
    modules = glob.glob(path.join(app_root, sub_module_path, "*.py"))
    full_module_base = [app_root_name] + submodule_tokens
    loaded_modules = []
    for m in modules:
        file_name = path.basename(m)
        module_name = os.path.splitext(file_name)[0]
        if "__init__" != module_name:
            m_module_tokens = full_module_base + [module_name]
            full_module_name = ".".join(m_module_tokens)
            spec = importlib.util.spec_from_file_location(full_module_name, m)
            abstract_module = importlib.util.module_from_spec(spec)
            loaded_modules.append((module_name, abstract_module))
            sys.modules[full_module_name] = abstract_module
            spec.loader.exec_module(abstract_module)
    return loaded_modules


def CreateRouterFromResources(
    application_module: ModuleType = ...,
    resource_path: str = "resources",
    common_resource_name: str = "resource",
) -> APIRouter:
    modules = getDirectoryModules(
        application_module=application_module, sub_module_path=resource_path
    )
    router = APIRouter()
    for m in modules:
        module = m[1]
        router.include_router(
            getattr(getattr(module, common_resource_name), "controller")
        )
    return router
