# from fastapi_crudrouter import SQLAlchemyCRUDRouter
# from app.utils.asynccrudrouter import AsyncSQLAlchemyCRUDRouter
# from app.models.user import User, UserCreate, UserUpdate
# from app.adapters import postgresql
# """
# It calls it a router... but its more like a controller
# """
# user = AsyncSQLAlchemyCRUDRouter(
#    schema=User,
#    create_schema=UserCreate,
#    update_schema=UserUpdate,
#    db_model=User,
#    db=postgresql.getSession,
# )

from fastapi import APIRouter, Path, Query
from app.utils.uuid import UUID
from app.repositories.user import UserRepository
from app.models.user import UserCreate, UserUpdate, ResponseSchema

controller = APIRouter(prefix="/user", tags=["user"])


@controller.post("", response_model=ResponseSchema, response_model_exclude_none=True)
async def create(create_form: UserCreate):
    await UserRepository.create(create_form)
    return ResponseSchema(detail="Successfully created data !")


@controller.patch(
    "/{id}", response_model=ResponseSchema, response_model_exclude_none=True
)
async def update(id: UUID = Path(..., alias="id"), *, update_form: UserUpdate):
    await UserRepository.update(id, update_form)
    return ResponseSchema(detail="Successfully updated data !")


@controller.delete(
    "/{id}", response_model=ResponseSchema, response_model_exclude_none=True
)
async def delete(
    id: UUID = Path(..., alias="id"),
):
    await UserRepository.delete(id)
    return ResponseSchema(detail="Successfully deleted data !")


@controller.get(
    "/{id}", response_model=ResponseSchema, response_model_exclude_none=True
)
async def get_by_id(id: UUID = Path(..., alias="id")):
    result = await UserRepository.get_by_id(id)
    return ResponseSchema(
        detail="Successfully fetch person data by id !", result=result
    )


@controller.get("", response_model=ResponseSchema, response_model_exclude_none=True)
async def get_all(
    page: int = 1,
    limit: int = 10,
    columns: str = Query(None, alias="columns"),
    sort: str = Query(None, alias="sort"),
    filter: str = Query(None, alias="filter"),
):
    result = await UserRepository.get_all(page, limit, columns, sort, filter)
    return ResponseSchema(
        detail="Successfully fetch person data by id !", result=result
    )
