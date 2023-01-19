from typing import Optional
from datetime import datetime
from sqlmodel import Field, SQLModel, Column, DateTime
from pydantic import EmailStr
from app.models._base import BaseUUIDModel


# The way the CRUD Router works, it needs an update, create, and base model.
# If you always structure model files in this order, you can extend from the
# minimal number of attrs that can be updated, all the way up to the maximal
# attrs in the base model.
class UserUpdate(SQLModel):
    first_name: str
    last_name: str
    email: EmailStr = Field(
        nullable=True, index=True, sa_column_kwargs={"unique": True}
    )
    is_active: bool = Field(default=True)
    is_superuser: bool = Field(default=False)
    birthdate: Optional[datetime] = Field(
        sa_column=Column(DateTime(timezone=True), nullable=True)
    )  # birthday with timezone
    phone: Optional[str]
    state: Optional[str]
    country: Optional[str]
    address: Optional[str]


class UserCreate(UserUpdate):
    pass


class User(BaseUUIDModel, UserUpdate, table=True):
    hashed_password: Optional[str] = Field(nullable=False, index=True)
