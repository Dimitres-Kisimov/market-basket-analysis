FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The image is self-verifying: lint + tests run at build time.
RUN pip install --no-cache-dir ruff pytest \
    && python -m ruff check . \
    && python -m pytest -q

CMD ["python", "-m", "basket", "--deliverables"]
