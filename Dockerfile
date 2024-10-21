FROM python:3.10

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . /app
WORKDIR /app

ENV PYTHONPATH=/app
CMD ["python", "worker/arima_model.py"]
