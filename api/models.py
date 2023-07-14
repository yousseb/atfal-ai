from typing import Annotated
from pydantic import BaseModel


class Box(BaseModel):
    x1: Annotated[int, "X1"]
    x2: Annotated[int, "X2"]
    y1: Annotated[int, "Y1"]
    y2: Annotated[int, "Y2"]
