from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_topic_prefix: str = "mes"
    collector_url: str = "http://localhost:8001"
    mes_url: str = "http://localhost:8000"
    mlflow_tracking_uri: str = "http://localhost:5000"
    model_name: str = "vibration-anomaly"
    model_dir: str = "data"
    anomaly_streak: int = 3
    cors_origins: list[str] = ["http://localhost:3001"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
