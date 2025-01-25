FROM python:3.10

ARG PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=on \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=500

RUN apt-get update && apt-get install -y gcc
RUN python -m pip install --upgrade pip
RUN python -m pip install poetry

RUN poetry config virtualenvs.in-project true

ENV APP_ROOT=/app
WORKDIR $APP_ROOT/src
COPY . ./

RUN poetry install --no-root

CMD ["/app/src/.venv/bin/uvicorn", "model.app:app", "--host", "0.0.0.0", "--port", "8080"]
