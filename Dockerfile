FROM python:3.12-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PORT=8080
WORKDIR /app
RUN --mount=type=secret,id=cloud_ca,required=false,mode=0444 \
    sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && if [ -f /run/secrets/cloud_ca ]; then hydra_build_ca=/run/secrets/cloud_ca; else hydra_build_ca=/etc/ssl/certs/ca-certificates.crt; fi \
    && apt-get -o Acquire::https::CaInfo="$hydra_build_ca" update \
    && apt-get -o Acquire::https::CaInfo="$hydra_build_ca" install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt constraints.txt ./
RUN --mount=type=secret,id=cloud_ca,required=false,mode=0444 \
    if [ -f /run/secrets/cloud_ca ]; then export PIP_CERT=/run/secrets/cloud_ca; fi \
    && python -m pip install -r requirements.txt && python -m pip check
RUN useradd --create-home --uid 10001 hydra
COPY --chown=hydra:hydra . .
RUN mkdir -p /app/data /app/logs && chown hydra:hydra /app/data /app/logs
USER hydra
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8080')+'/api/health',timeout=3)" || exit 1
CMD ["python", "main.py"]
