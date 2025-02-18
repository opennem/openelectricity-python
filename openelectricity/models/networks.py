from pydantic import BaseModel, Field


class Network(BaseModel):
    """A network in the OpenElectricity system."""

    code: str = Field(..., description="Network code")
    country: str = Field(..., description="Country code")
    label: str = Field(..., description="Network label/name")
    timezone_offset: str = Field(..., description="Timezone offset")
