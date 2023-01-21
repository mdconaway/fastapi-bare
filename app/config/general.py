from app.config._base import Base


class General(Base):
    PROJECT_NAME: str
    API_VERSION: str


general = General()
