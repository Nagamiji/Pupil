from pydantic import BaseModel
from typing import List, Optional

class PathPoint(BaseModel):
    type: str
    coords: List[float]

class CanvasObject(BaseModel):
    type: str
    path: Optional[List[PathPoint]] = None

class CanvasData(BaseModel):
    objects: List[CanvasObject]

class KhmerDigitResponse(BaseModel):
    predicted_digit: str
    writing_quality: str

class KhmerCharacterResponse(BaseModel):
    predicted_character: str
    is_correct: bool