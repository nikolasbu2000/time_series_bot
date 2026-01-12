FROM python:3.11-slim

# System deps: R + build toolchain + curl/unzip (für GitHub ZIP Download)
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base r-base-dev \
    build-essential \
    gfortran \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    curl unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# R base deps
RUN Rscript -e "install.packages(c('jsonlite','readr'), repos='https://cloud.r-project.org')"

# --- apt: install from GitHub ZIP via codeload (no GitHub API) ---
RUN curl -L -o /tmp/apt.zip https://codeload.github.com/HugoGruson/apt/zip/refs/heads/master \
    && unzip -q /tmp/apt.zip -d /tmp \
    && R CMD INSTALL /tmp/apt-* \
    && rm -rf /tmp/apt.zip /tmp/apt-*

# GARCH deps
RUN Rscript -e "install.packages(c('rugarch','rmgarch'), repos='https://cloud.r-project.org')"

# App code
COPY . /app

EXPOSE 8501

# Render setzt PORT; wir nutzen Shell-Expansion mit $PORT
CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true --browser.gatherUsageStats=false"]
