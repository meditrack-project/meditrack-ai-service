FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim AS production
LABEL maintainer="meditrack" version="1.0.0"
WORKDIR /app
RUN addgroup --system meditrack \
 && adduser --system meditrack --ingroup meditrack
COPY --from=builder /usr/local/lib/python3.12/site-packages \
                    /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY ./app ./app
USER meditrack
EXPOSE 4004
HEALTHCHECK --interval=30s --timeout=10s \
            --start-period=40s --retries=3 \
  CMD python -c \
  "import urllib.request; urllib.request.urlopen('http://localhost:4004/health')" \
  || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "4004"]
