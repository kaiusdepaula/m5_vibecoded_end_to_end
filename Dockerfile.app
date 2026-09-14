FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml ./
RUN uv pip install --system --no-cache -r pyproject.toml

COPY config ./config
COPY src ./src

ENV PYTHONPATH=/app/src

EXPOSE 8000
CMD ["uvicorn", "m5_forecast.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
