from app.utils.cruddy import Resource, UUID
from app.adapters import postgresql
from app.models.group import (
    Group,
    GroupCreate,
    GroupUpdate,
    GroupView,
)
from app.schemas.response import MetaObject
from app.policies.verify_session import verify_session


resource = Resource(
    adapter=postgresql,
    response_schema=GroupView,
    response_meta_schema=MetaObject,
    resource_update_model=GroupUpdate,
    resource_create_model=GroupCreate,
    resource_model=Group,
    id_type=UUID,
    policies_universal=[verify_session],
)
