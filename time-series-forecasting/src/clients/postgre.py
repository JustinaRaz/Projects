import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    MetaData,
    String,
    Table,
    create_engine,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import URL


class PostgresClient:
    """Client for connecting to and inserting data into PostgreSQL."""

    def __init__(self):
        self.processing_size = 1000
        load_dotenv()

        connection_url = URL.create(
            drivername="postgresql+psycopg",
            username=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"),
            host=os.getenv("PG_HOST", "localhost"),
            port=int(os.getenv("PG_PORT", "5432")),
            database=os.getenv("PG_DATABASE"),
        )

        self.engine = create_engine(connection_url) # Establishes a connection

        metadata = MetaData()

        self.day_ahead_prices = Table( # Defines the data the way it is in SQL
            "day_ahead_prices",
            metadata,
            Column("time_utc", DateTime(timezone=True), primary_key=True),
            Column("time_dk", DateTime(timezone=False), nullable=False),
            Column("price_area", String(3), primary_key=True),
            Column("price_eur_mwh", Float),
            Column("price_dkk_mwh", Float),
        )

    def insert_day_ahead_prices(self, df: pd.DataFrame) -> int:
        """Insert or update day-ahead prices from a DataFrame.
        The output is the number of processed records."""

        column_mapping = {
            "TimeUTC": "time_utc",
            "TimeDK": "time_dk",
            "PriceArea": "price_area",
            "DayAheadPriceEUR": "price_eur_mwh",
            "DayAheadPriceDKK": "price_dkk_mwh",
        }

        df = df.rename(columns=column_mapping).copy() # Creates a separate one rather than modifying the original.

        # Convert timestamps
        df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
        df["time_dk"] = pd.to_datetime(df["time_dk"])

        # Remove duplicate keys within the incoming batch
        df = df.drop_duplicates(
            subset=["time_utc", "price_area"],
            keep="last",
        )

        # Convert missing values to None
        df = df.astype(object).where(pd.notna(df), None)

        records = df.to_dict(orient="records")

        # Inserting process:

        with self.engine.begin() as connection:
            for start in range(0, len(records), self.processing_size):
                batch = records[start:start + self.processing_size]

                statement = insert(self.day_ahead_prices).values(batch)

                statement = statement.on_conflict_do_update(
                    index_elements=["time_utc", "price_area"],
                    set_={
                        "time_dk": statement.excluded.time_dk,
                        "price_eur_mwh": statement.excluded.price_eur_mwh,
                        "price_dkk_mwh": statement.excluded.price_dkk_mwh,
                    },
                )

                connection.execute(statement)

        return len(records)

    def close(self) -> None:
        """Close the database connection pool."""

        self.engine.dispose()