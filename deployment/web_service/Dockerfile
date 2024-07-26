# Stage 1: Install dependencies
FROM python:3.10-slim AS image

# Install pipenv and dependencies
RUN pip install --upgrade pip \
    && pip install pipenv

# Set the working directory
WORKDIR /app

# Copy Pipfile and Pipfile.lock
COPY Pipfile Pipfile.lock ./

# Install Python dependencies
RUN pipenv lock --requirements > requirements.txt \
    && pip install --no-cache-dir -r requirements.txt

# Stage 2: Build the final image
FROM python:3.10-slim

# Install Google Cloud SDK
RUN apt-get update && apt-get install -y curl \
    && curl -sSL https://sdk.cloud.google.com | bash \
    && exec -l $SHELL \
    && gcloud components install alpha beta

# Set the working directory
WORKDIR /app

# Copy dependencies from the builder stage
COPY --from=image /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=image /usr/local/bin /usr/local/bin

# Copy the application files and model binary
COPY [predict.py" "best_log_reg.bin" "./"]

# Copy GCS credentials if needed
# COPY /home/t.sukanmi/ProjectCLS/mlops-429011-853523c01da8.json /app/credentials.json

# Set environment variable for GCS credentials
# ENV GOOGLE_APPLICATION_CREDENTIALS="/home/t.sukanmi/ProjectCLS/mlops-429011-853523c01da8.json"

# Expose port 9696
EXPOSE 9696

# Use gunicorn to serve the application
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
