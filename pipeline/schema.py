from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime

class SecurityEvent(BaseModel):
    event_id: str = Field(..., description="Unique event identifier")
    camera_id: str = Field(..., description="Camera identifier")
    timestamp: str = Field(..., description="Timestamp in ISO or formatted string")
    category: str = Field(..., description="Classified incident label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    summary: str = Field(..., description="Concise scene summary from VLM")
    keyframe_path: str = Field(..., description="Path to saved snapshot")
