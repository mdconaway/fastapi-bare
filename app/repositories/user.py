import math

from sqlalchemy import update, delete, or_, text, func, column
from sqlalchemy.sql import select
from app.utils.uuid import UUID
from app.adapters import postgresql
from app.models.user import User, UserCreate, UserUpdate, PageResponse


class UserRepository:
    @staticmethod
    async def create(create_form: UserCreate):
        """create person data"""
        postgresql.add(
            User(
                name=create_form.name,
                sex=create_form.sex,
                birth_date=create_form.birth_date,
                birth_place=create_form.birth_place,
                country=create_form.country,
            )
        )
        await postgresql.commitOrRollback()
        # return a value?

    @staticmethod
    async def get_by_id(id: UUID):
        """retrieve person data by id"""
        query = select(User).where(User.id == id)
        result = (await postgresql.execute(query)).scalar_one_or_none()
        await postgresql.commitOrRollback()
        return result

    @staticmethod
    async def update(id: UUID, update_form: UserUpdate):
        """update person data"""

        query = (
            update(User)
            .where(User.id == id)
            .values(**update_form.dict())
            .execution_options(synchronize_session="fetch")
        )
        await postgresql.execute(query)
        await postgresql.commitOrRollback()
        # return a value?

    @staticmethod
    async def delete(id: UUID):
        """delete person data by id"""

        query = delete(User).where(User.id == id)
        await postgresql.execute(query)
        await postgresql.commitOrRollback()
        # return a value?

    @staticmethod
    async def get_all(
        page: int = 1,
        limit: int = 10,
        columns: str = None,
        sort: str = None,
        filter: str = None,
    ):
        query = select(from_obj=User, columns="*")

        # select columns dynamically
        if columns is not None and columns != "all":
            # we need column format data like this --> [column(id),column(name),column(sex)...]

            query = select(from_obj=User, columns=convert_columns(columns))

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
            query = query.order_by(text(convert_sort(sort)))

        # count query
        count_query = select(func.count(1)).select_from(query)

        offset_page = page - 1
        # pagination
        query = query.offset(offset_page * limit).limit(limit)

        # total record
        total_record = (await postgresql.execute(count_query)).scalar() or 0

        # total page
        total_page = math.ceil(total_record / limit)

        # result
        result = (await postgresql.execute(query)).fetchall()
        await postgresql.commitOrRollback()
        return PageResponse(
            page_number=page,
            page_size=limit,
            total_pages=total_page,
            total_record=total_record,
            content=result,
        )


def convert_sort(sort):
    """
    # separate string using split('-')
    split_sort = sort.split('-')
    # join to list with ','
    new_sort = ','.join(split_sort)
    """
    return ",".join(sort.split("-"))


def convert_columns(columns):
    """
    # seperate string using split ('-')
    new_columns = columns.split('-')

    # add to list with column format
    column_list = []
    for data in new_columns:
        column_list.append(data)

    # we use lambda function to make code simple

    """
    return list(map(lambda x: column(x), columns.split("-")))
