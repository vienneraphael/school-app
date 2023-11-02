FROM python:3.10-slim-buster

# System deps:
RUN pip install "poetry==1.6.1"

COPY ./app /app
COPY poetry.lock pyproject.toml .
WORKDIR /app


# Project initialization:
RUN poetry install --no-dev --no-interaction

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
