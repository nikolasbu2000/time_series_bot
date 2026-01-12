FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base r-base-dev \
    build-essential \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN Rscript -e "install.packages(c('jsonlite','readr'), repos='https://cloud.r-project.org')" \
 && Rscript -e "install.packages(c('apt','rugarch','rmgarch'), repos='https://cloud.r-project.org')"

COPY . /app

EXPOSE 8501

# IMPORTANT: shell expansion for $PORT
CMD sh -c "python -m streamlit run app.py --server.address=0.0.0.0 --server.port ${PORT:-8501} --server.headless true --browser.gatherUsageStats false"
