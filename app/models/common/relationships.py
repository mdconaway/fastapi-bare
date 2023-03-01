from typing import Optional
from sqlmodel import Field
from app.utils.cruddy import UUID, CruddyModel


# Many to many relationships require a manually defined "link" model.
# This model will house the table that stores many<->many relation rows.
class GroupUserLink(CruddyModel, table=True):
    user_id: Optional[UUID] = Field(
        default=None, foreign_key="User.id", primary_key=True
    )
    group_id: Optional[UUID] = Field(
        default=None, foreign_key="Group.id", primary_key=True
    )