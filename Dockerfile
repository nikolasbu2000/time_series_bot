FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base r-base-dev \
    build-essential gfortran \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    curl unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN Rscript -e "install.packages(c('jsonlite','readr'), repos='https://cloud.r-project.org')"

# apt (no GitHub API)
RUN set -eux; \
    apt_url_main="https://github.com/HugoGruson/apt/archive/refs/heads/main.zip"; \
    apt_url_master="https://github.com/HugoGruson/apt/archive/refs/heads/master.zip"; \
    (curl -fSL --retry 5 --retry-delay 2 -o /tmp/apt.zip "$apt_url_main" \
      || curl -fSL --retry 5 --retry-delay 2 -o /tmp/apt.zip "$apt_url_master"); \
    unzip -q /tmp/apt.zip -d /tmp; \
    R CMD INSTALL /tmp/apt-*; \
    rm -rf /tmp/apt.zip /tmp/apt-*

RUN Rscript -e "install.packages(c('rugarch','rmgarch'), repos='https://cloud.r-project.org')"

COPY . /app

EXPOSE 8501
CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true --browser.gatherUsageStats=false"]
