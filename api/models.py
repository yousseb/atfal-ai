from pydantic import BaseModel


class Box(BaseModel):
    x1: int
    x2: int
    y1: int
    y2: int
