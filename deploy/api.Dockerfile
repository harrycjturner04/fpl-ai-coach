FROM python:3.14-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml ./
COPY ingestion ingestion
COPY prediction prediction
COPY optimisation optimisation
COPY features features
COPY evaluation evaluation
COPY reasoning reasoning
COPY orchestration orchestration
COPY presentation presentation
RUN pip install --no-deps . \
 && pip install fastapi uvicorn pandas pyarrow requests "psycopg[binary]" "pyjwt[crypto]" google-cloud-run "pulp>=4.2,<5" "highspy>=1.7"
RUN useradd --create-home app
USER app
CMD ["sh", "-c", "exec uvicorn presentation.api.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
