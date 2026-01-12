FROM python:3.11-slim

# ===== System + R Dependencies =====
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base \
    r-base-dev \
    build-essential \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# ===== Arbeitsverzeichnis =====
WORKDIR /app

# ===== Python Dependencies =====
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ===== R Packages (CRAN) =====
RUN Rscript -e "install.packages(c('jsonlite','readr'), repos='https://cloud.r-project.org')" \
 && Rscript -e "install.packages(c('apt','rugarch','rmgarch'), repos='https://cloud.r-project.org')"

# ===== App Code =====
COPY . .

# ===== Render / Streamlit =====
ENV PORT=8501
EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=${PORT}", "--server.headless=true", "--browser.gatherUsageStats=false"]
