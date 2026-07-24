# App Runner requires linux/amd64. Pin the platform in FROM so builds on
# ARM developer machines (Mac M-series, ARM Linux) still produce an amd64
# image locally. The deploy script also passes --platform linux/amd64 to
# docker buildx as belt-and-suspenders.
FROM --platform=linux/amd64 python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p data

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Streamlit + reverse proxy (App Runner, Nginx, ALB) notes:
# * `--server.enableXsrfProtection=false` — required behind proxies that
#   rewrite the Origin header (App Runner's ALB does). With XSRF ON,
#   Streamlit forces CORS ON, which then rejects the WebSocket upgrade
#   from the proxied Origin and the app hangs on the "loading" skeleton.
# * `--server.enableCORS=false` — same reason.
# Together these disable the two same-origin protections that assume
# direct-to-container access. Safe for internal/demo deployments where
# App Runner's own auth / IP allowlist is the outer boundary.
CMD ["streamlit", "run", "src/dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]
