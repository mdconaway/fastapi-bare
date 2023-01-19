from fastapi import APIRouter, Path, Query
from app.utils.uuid import UUID
from app.repositories.user import UserRepository
from app.models.user import UserCreate, UserUpdate
from app.schemas.response import ResponseSchema, PageResponse

controller = APIRouter(prefix="/user", tags=["user"])


@controller.post("", response_model=ResponseSchema, response_model_exclude_none=True)
async def create(data: UserCreate):
    await UserRepository.create(data=data)
    return ResponseSchema(message="Successfully created data !")


@controller.patch(
    "/{id}", response_model=ResponseSchema, response_model_exclude_none=True
)
async def update(id: UUID = Path(..., alias="id"), *, data: UserUpdate):
    await UserRepository.update(id=id, data=data)
    return ResponseSchema(message="Successfully updated data !")


@controller.delete(
    "/{id}", response_model=ResponseSchema, response_model_exclude_none=True
)
async def delete(
    id: UUID = Path(..., alias="id"),
):
    await UserRepository.delete(id=id)
    return ResponseSchema(message="Successfully deleted data !")


@controller.get(
    "/{id}", response_model=ResponseSchema, response_model_exclude_none=True
)
async def get_by_id(id: UUID = Path(..., alias="id")):
    data = await UserRepository.get_by_id(id=id)
    return ResponseSchema(message="Successfully fetch data by id !", data=data)


@controller.get("", response_model=PageResponse, response_model_exclude_none=True)
async def get_all(
    page: int = 1,
    limit: int = 10,
    columns: str = Query(None, alias="columns"),
    sort: str = Query(None, alias="sort"),
    filter: str = Query(None, alias="filter"),
):
    return await UserRepository.get_all(
        page=page, limit=limit, columns=columns, sort=sort, filter=filter
    )
