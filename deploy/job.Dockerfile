FROM python:3.14-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 KERAS_BACKEND=torch
WORKDIR /app
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml ./
COPY ingestion ingestion
COPY prediction prediction
COPY optimisation optimisation
COPY features features
COPY evaluation evaluation
COPY reasoning reasoning
COPY orchestration orchestration
COPY presentation presentation
RUN pip install ".[cloud]"
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home app
USER app
ENTRYPOINT ["python", "-m", "orchestration.job"]
