from app.utils.cruddy import Resource, UUID
from app.adapters import postgresql
from app.models.user import (
    User,
    UserCreate,
    UserUpdate,
    UserSingleResponse,
    UserPageResponse,
)
from app.schemas.response import MetaObject
from app.policies.verify_session import verify_session


resource = Resource(
    adapter=postgresql,
    prefix="/user",
    tags=["user"],
    response_single_schema=UserSingleResponse,
    response_many_schema=UserPageResponse,
    response_meta_schema=MetaObject,
    resource_update_model=UserUpdate,
    resource_create_model=UserCreate,
    resource_model=User,
    id_type=UUID,
    policies_universal=[verify_session],
)
