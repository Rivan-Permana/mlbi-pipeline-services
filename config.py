from typing import List
from pydantic import Field, field_validator

CORS_ORIGINS: List[str] = Field(default_factory=lambda: ["*"], validation_alias="CORS_ORIGINS")
