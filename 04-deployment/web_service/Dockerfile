FROM python:3.10-bullseye

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

# Copy Pipfile and Pipfile.lock
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install dependencies
RUN pipenv install --system --deploy

# Copy the application files
COPY ["predict.py", "./"]

# Install Google Cloud SDK
RUN curl -sSL https://sdk.cloud.google.com | bash -s -- --install-dir=/usr/local/gcloud --disable-prompts \
    && /usr/local/gcloud/google-cloud-sdk/bin/gcloud components install alpha beta

# Set environment variable for Google Cloud SDK
ENV PATH=$PATH:/usr/local/gcloud/google-cloud-sdk/bin

# Copy the local binary model
COPY ["best_log_reg.bin", "./"]

# Expose port 9696
EXPOSE 9696

# Use gunicorn to serve the application
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
