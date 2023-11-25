import enum
from sqlalchemy import Column, Enum, Integer, String, Text
from . import Base


class Status(str, enum.Enum):
    Active = 1
    Inactive = 0


class APIKey(Base):
    __tablename__ = "api_key"

    id = Column(Integer, primary_key=True)
    key = Column(String(100), nullable=False)
    note = Column(Text, nullable=False)
    status = Column(Enum(Status))
