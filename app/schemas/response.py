from pydantic.generics import GenericModel
from typing import TypeVar, Optional, Generic, List
from sqlmodel import SQLModel

# Look into this for making generic factories??
# https://shanenullain.medium.com/abstract-factory-in-python-with-generic-typing-b9ceca2bf89e

T = TypeVar("T")


class MetaObject(GenericModel, Generic[T]):
    page_number: int
    page_size: int
    total_pages: int
    total_record: int


class PageResponse(GenericModel, Generic[T]):
    # The response for a pagination query.
    message: str
    meta: MetaObject
    data: List[T]


class ResponseSchema(SQLModel):
    # The response for a single object return
    message: str
    data: Optional[T] = None
