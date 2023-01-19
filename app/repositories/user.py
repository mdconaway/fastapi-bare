import math
from sqlalchemy import update, delete, or_, text, func, column
from sqlalchemy.sql import select
from app.utils.uuid import UUID
from app.adapters import postgresql
from app.models.user import User, UserCreate, UserUpdate
from app.schemas.response import PageResponse, MetaObject


class UserRepository:
    @staticmethod
    async def create(data: UserCreate):
        # create user data
        async for session in postgresql.getSession():
            session
        await session.add(User(**data.dict()))
        # return a value?

    @staticmethod
    async def get_by_id(id: UUID):
        # retrieve user data by id
        async for session in postgresql.getSession():
            session
        query = select(User).where(User.id == id)
        result = (await session.execute(query)).scalar_one_or_none()
        return result

    @staticmethod
    async def update(id: UUID, data: UserUpdate):
        async for session in postgresql.getSession():
            session
        # update user data
        query = (
            update(User)
            .where(User.id == id)
            .values(**data.dict())
            .execution_options(synchronize_session="fetch")
        )
        await session.execute(query)
        # return a value?

    @staticmethod
    async def delete(id: UUID):
        # delete user data by id
        async for session in postgresql.getSession():
            session
        query = delete(User).where(User.id == id)
        await session.execute(query)
        # return a value?

    @staticmethod
    async def get_all(
        page: int = 1,
        limit: int = 10,
        columns: str = None,
        sort: str = None,
        filter: str = None,
    ):
        async for session in postgresql.getSession():
            session

        query = select(from_obj=User, columns="*")

        # select columns dynamically
        if columns is not None and columns != "all":
            # we need column format data like this --> [column(id),column(name),column(sex)...]
            query = select(
                from_obj=User,
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
                _attr = getattr(User, attr)
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
        # session = await postgresql.getSession()
        total_record = (await session.execute(count_query)).scalar() or 0
        # total page
        total_page = math.ceil(total_record / limit)

        # result
        # session = await postgresql.getSession()
        result = (await session.execute(query)).fetchall()

        return PageResponse(
            meta=MetaObject(
                page_number=page,
                page_size=limit,
                total_pages=total_page,
                total_record=total_record,
            ),
            data=result,
            message="Successfully fetch data list !",
        )
