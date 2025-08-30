from sqlmodel import SQLModel, create_engine
from dotenv import load_dotenv, find_dotenv
from models import Export  # noqa: F401
import os

load_dotenv(find_dotenv())


DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, echo=True)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
