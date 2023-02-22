from app.utils.cruddy import Resource, UUID
from app.adapters import postgresql
from app.models.post import (
    Post,
    PostCreate,
    PostUpdate,
    PostSingleResponse,
    PostPageResponse,
)
from app.schemas.response import MetaObject
from app.policies.verify_session import verify_session


resource = Resource(
    adapter=postgresql,
    prefix="/post",
    tags=["post"],
    response_single_schema=PostSingleResponse,
    response_many_schema=PostPageResponse,
    response_meta_schema=MetaObject,
    resource_update_model=PostUpdate,
    resource_create_model=PostCreate,
    resource_model=Post,
    id_type=UUID,
    policies_universal=[verify_session],
)
