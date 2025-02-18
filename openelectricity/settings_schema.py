from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    env: str = Field(default="development", validation_alias=AliasChoices("ENV"))
    api_key: str = Field(..., validation_alias=AliasChoices("OPENELECTRICITY_API_KEY"))
    base_url: str = Field(
        default="https://api.openelectricity.org.au/v4/",
        validation_alias=AliasChoices("OPENELECTRICITY_API_URL"),
    )

    @property
    def is_development(self) -> bool:
        return self.env in ["development", "dev"]

    @property
    def is_production(self) -> bool:
        return self.env in ["production", "prod"]

    class Config:
        env_file = ".env"


settings = Settings()
