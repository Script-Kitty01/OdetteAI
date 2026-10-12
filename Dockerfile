FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir .

# Mount trained run directories at /models/runs, e.g.:
#   docker build -t fluxguard .
#   docker run -p 8000:8000 -v ${PWD}/models/runs:/models/runs -e FLUXGUARD_MODELS_ROOT=/models/runs fluxguard
ENV FLUXGUARD_MODELS_ROOT=/models/runs
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "fluxguard.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
