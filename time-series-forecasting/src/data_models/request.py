import json
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
import requests
from pydantic import BaseModel, Field


class EnergyDataRequest(BaseModel):
    """Configuration for an Energi Data Service API request."""

    base_url: str = "https://api.energidataservice.dk/dataset"
    dataset: str = "DayAheadPrices"
    start: date
    end: date | Literal["now"]
    price_areas: list[Literal["DK1", "DK2"]] = ["DK1"]
    sort: str = "TimeDK"
    limit: int = Field(default=0, ge=0)
    save_path: Path = Path("data/raw")

    def to_api_params(self) -> dict:

        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat() if isinstance(self.end, date) else self.end,
            "filter": json.dumps({"PriceArea": self.price_areas}),
            "sort": self.sort,
            "limit": self.limit,
        }
        
    def download_data(self) -> pd.DataFrame:
        """Download energy data."""

        url = f"{self.base_url}/{self.dataset}"
        self.save_path.mkdir(parents=True, exist_ok=True)

        response = requests.get(
            url,
            params=self.to_api_params(),
            timeout=60,
        )
        response.raise_for_status()
        
        df = pd.DataFrame(response.json()["records"])
        
        file_path = self.save_path / f"{self.dataset}.parquet"
        df.to_parquet(file_path, index=False)

        print(f"Data downloaded successfully. Saved at {self.save_path}")