from pydantic import BaseModel
class TryOnRequest(BaseModel):
    person_image_url: str
    article_url: str = None
    type: str = None