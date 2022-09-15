FROM python:3.8-slim-buster

# Install curl for health check
RUN apt update && \
    apt install -y --no-install-recommends \
        curl libyaml-cpp-dev libyaml-dev && \
    apt autoremove -y && \
    apt clean all && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

# Install python packages
RUN pip install mlflow boto3 pymysql pyyaml

# We run on port 5000
EXPOSE 5000

HEALTHCHECK CMD curl -f http://localhost:5000/health || exit 1
