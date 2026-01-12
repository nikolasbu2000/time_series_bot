FROM python:3.11-slim

# --- Linux deps für R + gängige R Pakete (Compilation/SSL/Curl/XML) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base r-base-dev \
    build-essential \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# R deps (CRAN)
RUN Rscript -e "install.packages(c('jsonlite','readr'), repos='https://cloud.r-project.org')" \
 && Rscript -e "install.packages(c('apt','rugarch','rmgarch'), repos='https://cloud.r-project.org')"

# App code
COPY . /app

# Streamlit läuft auf PORT (Render/Fly geben PORT als ENV)
ENV PORT=8501
EXPOSE 8501

CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT} --server.headless=true --browser.gatherUsageStats=false"]
