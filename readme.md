scrape_real_imp.py is run for create storage for scrape and will scrape picture as TARGET_TOTAL on topic and specific site in queries(can also just add topic no need to always add specific site).(still have like 0.66% picture that not real life  picture ex. infographic or something like that)
and the make_7_day_pic.py is random pic DAYS * IMAGES_PER_DAY to store in 1 folder and DAYS subfolders it also contain log for picture that already use so if you run later it will not get the picture that already use.
https://pypi.org/project/duckduckgo-search/#duckduckgo-search-operators
library for scrape that I use for clrarification on some detail.

# How to run Web Application
## Requirements
```
1.Node.js
2.Python
```
## Backend
```
1. cd backend
2. pip install -r requirements.txt
3. python api.py
```

## Frontend
```
1. npm install
2. update api endpoint on page.jsx (your backend api endpoint, will do .env later)
3. npm run build
4. npm start
```

# Project Setup Instructions
This project leverages Docker to simplify the setup and deployment of Apache Airflow and MLflow. Follow the steps below to get started.
## Prerequisites
```
Docker installed and running
Docker Compose installed
Setting Up Airflow and MLflow
```
```
Navigate to the Docker setup directory:
cd airflow/airflow-docker/
Start the services using Docker Compose:
docker-compose up
This command will spin up all necessary containers for both Airflow and MLflow.
Wait a few moments for all containers to fully initialize, then access the UIs:
Airflow Web UI: http://localhost:8080/
MLflow Tracking UI: http://localhost:5500/
Running the Project
```
```
Run the Image Scrape DAG first.
This DAG collects additional images necessary for training.
The original training images are from Kaggle and can be accessed here:
Shutterstock Dataset for AI vs Human Gen Image
Model Training and MLflow Logging
Once the training DAG runs, it will automatically log metrics, parameters, and models to MLflow.
Configuration Notes
```
```
Environment variables and dependency configurations can be adjusted in the .env file located inside airflow-docker/.
This is also where you can specify additional Python dependencies needed by your DAGs.
To stop the services, press Ctrl+C in the terminal, or run:
docker-compose down
```
