# Dockerfile
FROM python:3.11-slim

# --- System deps: R + build tools + libs for common R packages ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base r-base-dev \
    git \
    build-essential \
    gfortran \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libpng-dev \
    libjpeg-dev \
    libtiff5-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python deps ---
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# --- R deps ---
# jsonlite/readr/remotes first, then install apt from GitHub (CRAN sometimes missing/outdated)
RUN Rscript -e "install.packages(c('jsonlite','readr','remotes'), repos='https://cloud.r-project.org')" \
 && Rscript -e "remotes::install_github('HugoGruson/apt')" \
 && Rscript -e "install.packages(c('rugarch','rmgarch'), repos='https://cloud.r-project.org')"

# --- App code ---
COPY . /app

# Streamlit on Render: bind to 0.0.0.0 and use $PORT provided by Render
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

CMD sh -c "streamlit run app.py --server.port ${PORT:-8501}"
