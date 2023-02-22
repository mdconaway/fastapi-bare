from typing import Optional, TypeVar, List, TYPE_CHECKING
from sqlmodel import Field, Relationship
from app.utils.cruddy import UUID, CruddyGenericModel, CruddyModel, CruddyUUIDModel
from app.schemas.response import MetaObject

if TYPE_CHECKING:
    from app.models.user import User

T = TypeVar("T")

# The way the CRUD Router works, it needs an update, create, and base model.
# If you always structure model files in this order, you can extend from the
# minimal number of attrs that can be updated, all the way up to the maximal
# attrs in the base model. CRUD JSON serialization schemas are also exposed
# for modification, and it makes sense to keep your response schemas defined
# in the same location as the view model used to represent records to the
# client.

# The "Update" model variant describes all fields that can be affected by a
# client's PATCH action. Generally, the update model should have the fewest
# number of available fields for a client to manipulate.
class PostUpdate(CruddyModel):
    content: str


# The "Create" model variant expands on the update model, above, and adds
# any new fields that may be writeable only the first time a record is
# generated. This allows the POST action to accept update-able fields, as
# well as one-time writeable fields.
class PostCreate(PostUpdate):
    user_id: UUID = Field(foreign_key="User.id")
    user: "User" = Relationship(back_populates="posts")


# The "View" model describes all fields that should typcially be present
# in any JSON responses to the client. This should, at a minimum, include
# the identity field for the model, as well as any server-side fields that
# are important but tamper resistant, such as created_at or updated_at
# fields. This should be used when defining single responses and paged
# responses, as in the schemas below.
class PostView(CruddyUUIDModel, PostCreate):
    pass


# The "Base" model describes the actual table as it should be reflected in
# Postgresql, etc. It is generally unsafe to use this model in actions, or
# in JSON representations, as it may contain hidden fields like passwords
# or other server-internal state or tracking information. Keep your "Base"
# models separated from all other interactive derivations.
class Post(PostView, table=True):
    pass


# The "Single Response" model is a generic model used to define how a
# singleton should be represented in JSON format when communicating between
# the client and the server. This is essentially just a communications
# schema, but it should be maintained in your model files as it necessarily
# leverages the "View" model as an embedded component of data
# serialization.
class PostSingleResponse(CruddyGenericModel):
    post: Optional[PostView] = None

    def __init__(self, data):
        super().__init__(post=PostView(data))


# The "Page Response" model is a generic model used to define how a
# paged resource should be represented in JSON format when
# communicating between the client and the server. This is important for
# defining how query results are transported, and alters how the "get many"
# action is represented back to the client. Again, this is essentially
# just a communications schema, but it should be maintained in your model
# files as it necessarily leverages the "View" model as an embedded
# component of data serialization. The "View" model in this case
# representing each singleton record within the "get many" list.
class PostPageResponse(CruddyGenericModel):
    posts: List[PostView]
    meta: MetaObject

    def __init__(self, data: List[T] = [], meta: MetaObject = MetaObject()):
        super().__init__(posts=map(lambda x: PostView(x), data), meta=meta)
