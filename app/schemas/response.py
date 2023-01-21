from pydantic.generics import GenericModel
from typing import TypeVar, Optional, Generic, List
from sqlmodel import SQLModel


T = TypeVar("T")


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


class ResponseSchema(SQLModel):
    # The response for a single object return
    message: str
    data: Optional[T] = None
