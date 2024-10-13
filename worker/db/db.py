import os
from contextlib import contextmanager

from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase


class Base(DeclarativeBase):
    __abstract__ = True
    # metadata = MetaData()


engine = create_engine(
    #os.getenv('DB_URL'),
    "mysql+pymysql://root:Uet123@127.0.0.1/iotdata",
    pool_size=30,
    max_overflow=10,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=engine, expire_on_commit=False
)


@contextmanager
def get_session() -> Session:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()