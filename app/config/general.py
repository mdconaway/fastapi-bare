from app.config._base import Base


class General(Base):
    PROJECT_NAME: str
    API_VERSION: str = "v1"


general = General()
