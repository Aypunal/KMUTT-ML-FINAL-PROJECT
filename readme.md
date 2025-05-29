# 🖼️ AI-Generated Image Detection Web Application

This project provides a full-stack web application to detect whether an image is real or AI-generated. It leverages learning models, a data scraping pipeline, MLflow for experiment tracking, and Apache Airflow for automation.

---

## 📂 Project Structure

```
backend/               ← Flask API for inference
frontend/              ← Next.js frontend for user interaction
airflow/               ← Dockerized setup for Airflow & MLflow
train/                 ← Training shits
```

---

## 🧠 Key Features

- Real vs. AI-generated image classification
- Learning models: Traditional CNN, MobileNetV2, EfficientNetB0
- MLflow for experiment tracking
- Apache Airflow for CI/CD automation
- Docker for reproducible deployment

---

## 🕹️ Web Application Usage

### ✅ Requirements

- **Node.js**
- **Python 3.x**

---

### 🔧 Backend Setup

```bash
cd backend
pip install -r requirements.txt
python api.py
```

---

### 🎨 Frontend Setup

```bash
npm install
# update api endpoint on page.jsx (your backend api endpoint, will do .env later)
npm run build
npm start
```

---

## 📦 Docker-Based Setup (Airflow + MLflow)

We use Docker Compose to simplify usage of Apache Airflow and MLflow.

### ⚙️ Prerequisites

- Docker
- Docker Compose

---

### 🚀 Launch Services

```bash
cd airflow/airflow-docker/
docker-compose up
```

This command initializes the containers. Once ready, access:

- **Airflow UI**: http://localhost:8080/
- **MLflow UI**: http://localhost:5500/

To stop services:

```bash
docker-compose down
```

---

## 📈 MLflow Integration

MLflow is integrated to track experiments, log metrics, store trained models, and visualize performance comparisons.

### 🧪 How to Use

1. Trigger the **Model Training DAG** from Airflow.
2. MLflow will automatically:
   - Log metrics and parameters
   - Save plots and model artifacts
   - Assign a unique run ID for traceability
3. Open **http://localhost:5500/** to compare and analyze experiments.


---

## 🖼️ Image Scraping & Dataset Generation

### `scrape_real_imp.py`

- Scrapes real images from **topics** and **sites**.
- Sets a `TARGET_TOTAL` number of images to collect.
- ~0.6% of the results may include non-photo visuals like infographics.
- Scraping powered by [duckduckgo-search](https://pypi.org/project/duckduckgo-search/).

### `make_7_day_pic.py`

- Randomly selects `DAYS * IMAGES_PER_DAY` images.
- Organizes them into folders by day.

---

## 📁 Dataset Sources

- **Real images**: Scraped from Pexels, Unsplash
- **AI-generated images**: Generated using Stable Diffusion, or collected from Midjourney/DALL·E communities
- **Kaggle dataset**: [Shutterstock Dataset – AI vs Human Images](https://www.kaggle.com/datasets/shreyasraghav/shutterstock-dataset-for-ai-vs-human-gen-image)
