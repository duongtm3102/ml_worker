import os
from contextlib import contextmanager

from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase


class Base(DeclarativeBase):
    __abstract__ = True
    # metadata = MetaData()

db_url = os.getenv('DB_URL')
if not db_url:
    db_url = "mysql+pymysql://root:Uet123@mysql/iotdata"

engine = create_engine(
    db_url,
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
        session.commit() 
    except Exception:
        session.rollback()
        raise
    finally:
        session.close() 
