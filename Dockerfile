FROM python:3.11-slim

# --- System deps: Python build + R + Compiler/BLAS/LAPACK + SSL/CURL/XML ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base r-base-dev \
    build-essential gfortran \
    libopenblas-dev liblapack-dev \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps zuerst (besseres caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# R deps von CRAN (WICHTIG: apt von CRAN, nicht GitHub)
RUN Rscript -e "install.packages(c('jsonlite','readr','apt','rugarch','rmgarch'), repos='https://cloud.r-project.org')"

# App code
COPY . /app

ENV PORT=8501
EXPOSE 8501

# WICHTIG: Shell-Form, damit $PORT expandiert (sonst kommt '${PORT}' als String an)
CMD sh -c "streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true --browser.gatherUsageStats=false"
