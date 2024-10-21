from .db import Base, engine
import datetime
from typing import Optional
from sqlalchemy import Integer, DateTime, Float, Text, String, BLOB, LargeBinary
from sqlalchemy.orm import Mapped, mapped_column


class HouseData(Base):
    __tablename__ = 'house_data'

    house_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    reg_date: Mapped[datetime.datetime] = mapped_column(DateTime, primary_key=True)
    slice_gap: Mapped[int] = mapped_column(Integer, primary_key=True)
    slice_index: Mapped[int] = mapped_column(Integer)
    year: Mapped[int] = mapped_column(Integer)
    month: Mapped[int] = mapped_column(Integer)
    day: Mapped[int] = mapped_column(Integer)
    avg: Mapped[Optional[float]] = mapped_column(Float)


class HousePredArima(Base):
    __tablename__ = 'house_data_forecast_arima'

    house_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    reg_date: Mapped[datetime.datetime] = mapped_column(DateTime)
    slice_gap: Mapped[int] = mapped_column(Integer, primary_key=True)
    slice_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    year: Mapped[int] = mapped_column(Integer, primary_key=True)
    month: Mapped[int] = mapped_column(Integer, primary_key=True)
    day: Mapped[int] = mapped_column(Integer, primary_key=True)
    avg: Mapped[Optional[float]] = mapped_column(Float)


class HousePredNeuralProphet(Base):
    __tablename__ = 'house_data_forecast_neuralprophet'

    house_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    reg_date: Mapped[datetime.datetime] = mapped_column(DateTime, primary_key=True)
    slice_gap: Mapped[int] = mapped_column(Integer, primary_key=True)
    slice_index: Mapped[int] = mapped_column(Integer)
    year: Mapped[int] = mapped_column(Integer)
    month: Mapped[int] = mapped_column(Integer)
    day: Mapped[int] = mapped_column(Integer)
    avg: Mapped[Optional[float]] = mapped_column(Float)


class HousePredXGBoost(Base):
    __tablename__ = 'house_data_forecast_xgboost'

    house_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    reg_date: Mapped[datetime.datetime] = mapped_column(DateTime)
    slice_gap: Mapped[int] = mapped_column(Integer, primary_key=True)
    slice_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    year: Mapped[int] = mapped_column(Integer, primary_key=True)
    month: Mapped[int] = mapped_column(Integer, primary_key=True)
    day: Mapped[int] = mapped_column(Integer, primary_key=True)
    avg: Mapped[Optional[float]] = mapped_column(Float)


class PredictModel(Base):
    __tablename__ = 'predict_model'
    # id: Mapped[int] = mapped_column(Integer, primary_key=True)
    house_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slice_gap: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(255), primary_key=True)
    model: Mapped[bytes] = mapped_column(LargeBinary(length=(2**32)-1))
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime)
    

Base.metadata.create_all(bind=engine)