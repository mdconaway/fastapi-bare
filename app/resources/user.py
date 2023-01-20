from app.utils.cruddy import Resource
from app.adapters import postgresql
from app.utils.uuid import UUID
from app.models.user import User, UserCreate, UserUpdate
from app.schemas.response import MetaObject, PageResponse, ResponseSchema

resource = Resource(
    adapter=postgresql,
    prefix="/user",
    tags=["user"],
    response_single_schema=ResponseSchema,
    response_many_schema=PageResponse,
    response_meta_schema=MetaObject,
    resource_update_model=UserUpdate,
    resource_create_model=UserCreate,
    resource_model=User,
    id_type=UUID,
)

controller = resource.controller
repository = resource.repository
adapter = resource.adapter
