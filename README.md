# Vehicle Insurance Cross-Sell Prediction — MLOps Pipeline

An end-to-end **MLOps pipeline** that predicts whether a health insurance customer is also interested in vehicle insurance (cross-sell prediction). Built with a modular, production-grade architecture featuring automated CI/CD, cloud model storage, and a live inference API.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Tech Stack](#2-tech-stack)
3. [Project Architecture](#3-project-architecture)
4. [Project Structure](#4-project-structure)
5. [ML Pipeline Components](#5-ml-pipeline-components)
6. [Dataset & Features](#6-dataset--features)
7. [Setup & Installation](#7-setup--installation)
8. [MongoDB Atlas Setup](#8-mongodb-atlas-setup)
9. [AWS Setup](#9-aws-setup)
10. [Environment Variables](#10-environment-variables)
11. [Running the Application](#11-running-the-application)
12. [API Endpoints](#12-api-endpoints)
13. [CI/CD Pipeline (GitHub Actions + AWS)](#13-cicd-pipeline-github-actions--aws)
14. [Docker](#14-docker)
15. [Project Flow Summary](#15-project-flow-summary)

---

## 1. Problem Statement

Insurance companies often want to cross-sell vehicle insurance to existing health insurance customers. Manually identifying interested customers is expensive and inefficient. This project builds a **binary classification model** that predicts customer response (`1` = Interested, `0` = Not Interested) based on demographic and vehicle-related features.

---

## 2. Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.10 |
| **ML Framework** | Scikit-learn (RandomForestClassifier) |
| **Data Storage** | MongoDB Atlas |
| **Cloud Storage** | AWS S3 |
| **Web Framework** | FastAPI + Uvicorn |
| **Templating** | Jinja2 |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |
| **Deployment** | AWS EC2 + AWS ECR |
| **Environment** | Conda |

---

## 3. Project Architecture

```
User Browser
    │
    ▼
FastAPI App (app.py)  ──── /train ──▶  TrainPipeline
    │                                      │
    │ POST /                                ├── DataIngestion   (MongoDB Atlas)
    │                                      ├── DataValidation  (schema check)
    ▼                                      ├── DataTransformation (preprocessing)
PredictionPipeline                         ├── ModelTrainer    (RandomForest)
    │                                      ├── ModelEvaluation (vs S3 model)
    ▼                                      └── ModelPusher     (→ AWS S3)
AWS S3 Bucket
(model-registry)
```

### CI/CD Flow

```
Git Push → GitHub Actions
              │
              ├── CI: Build Docker Image → Push to AWS ECR
              └── CD: Pull Image from ECR → Run on EC2 (self-hosted runner)
```

---

## 4. Project Structure

```
vehicle-insurance-mlops-pipeline/
│
├── .github/
│   └── workflows/
│       └── aws.yaml                  # GitHub Actions CI/CD pipeline
│
├── config/
│   ├── schema.yaml                   # Dataset schema for validation
│   └── model.yaml                    # Model configuration
│
├── notebook/
│   ├── data.csv                      # Raw dataset
│   ├── mongoDB_demo.ipynb            # Notebook to push data to MongoDB
│   └── exp-notebook.ipynb            # EDA & Feature Engineering notebook
│
├── src/
│   ├── cloud_storage/
│   │   └── aws_storage.py            # S3 read/write operations
│   │
│   ├── components/
│   │   ├── data_ingestion.py         # Fetch from MongoDB, train/test split
│   │   ├── data_validation.py        # Schema & data integrity checks
│   │   ├── data_transformation.py    # Preprocessing, encoding, scaling
│   │   ├── model_trainer.py          # RandomForest training & evaluation
│   │   ├── model_evaluation.py       # Compare new vs production model
│   │   └── model_pusher.py           # Push accepted model to AWS S3
│   │
│   ├── configuration/
│   │   ├── mongo_db_connection.py    # MongoDB client setup
│   │   └── aws_connection.py         # AWS S3 session setup
│   │
│   ├── constants/
│   │   └── __init__.py               # All project-wide constants
│   │
│   ├── data_access/
│   │   └── proj1_data.py             # MongoDB → DataFrame utility
│   │
│   ├── entity/
│   │   ├── config_entity.py          # Dataclasses for component configs
│   │   ├── artifact_entity.py        # Dataclasses for component outputs
│   │   ├── estimator.py              # MyModel wrapper (preprocessor + model)
│   │   └── s3_estimator.py           # S3-backed model loader/predictor
│   │
│   ├── exception/
│   │   └── __init__.py               # Custom exception handler
│   │
│   ├── logger/
│   │   └── __init__.py               # Centralized logger setup
│   │
│   ├── pipline/
│   │   ├── training_pipeline.py      # Orchestrates all training stages
│   │   └── prediction_pipeline.py    # Handles live inference requests
│   │
│   └── utils/
│       └── main_utils.py             # File I/O, YAML, numpy utilities
│
├── static/                           # CSS and static assets
├── templates/
│   └── vehicledata.html              # Prediction form UI
│
├── app.py                            # FastAPI entry point
├── demo.py                           # Quick pipeline test script
├── template.py                       # Project scaffolding script
├── setup.py                          # Local package installation
├── pyproject.toml                    # Build system config
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Container definition
└── .github/workflows/aws.yaml        # CI/CD workflow
```

---

## 5. ML Pipeline Components

The training pipeline runs **6 sequential stages**, each producing an **artifact** that is consumed by the next stage.

### Stage 1 — Data Ingestion
- Connects to **MongoDB Atlas** and fetches the vehicle insurance collection.
- Saves the raw data as a CSV in the local feature store.
- Splits data into **train (75%) / test (25%)** sets.
- **Output:** `DataIngestionArtifact` (paths to train/test CSV files)

### Stage 2 — Data Validation
- Validates the dataset against `config/schema.yaml`.
- Checks for required columns, correct data types, and data integrity.
- Generates a validation **report** (YAML format).
- **Output:** `DataValidationArtifact`

### Stage 3 — Data Transformation
- Encodes categorical columns (`Gender`, `Vehicle_Age`, `Vehicle_Damage`).
- Applies `StandardScaler` to numerical columns (`Age`, `Vintage`).
- Applies `MinMaxScaler` to skewed columns (`Annual_Premium`).
- Handles **class imbalance** using SMOTE (via `imblearn`).
- Saves the fitted preprocessing pipeline as `preprocessing.pkl`.
- **Output:** `DataTransformationArtifact` (`.npy` arrays + preprocessing object)

### Stage 4 — Model Trainer
- Trains a **RandomForestClassifier** with tuned hyperparameters:
  - `n_estimators = 200`
  - `max_depth = 10`
  - `criterion = entropy`
  - `min_samples_split = 7`, `min_samples_leaf = 6`
  - `random_state = 101`
- Evaluates accuracy, F1-score, precision, and recall.
- Bundles the preprocessing object + model into a single `MyModel` object and saves it as `model.pkl`.
- **Output:** `ModelTrainerArtifact`

### Stage 5 — Model Evaluation
- Loads the **production model from AWS S3** (if one exists).
- Compares the new model's performance against the production model.
- Accepts the new model only if improvement exceeds the threshold: `0.02` (2%).
- **Output:** `ModelEvaluationArtifact`

### Stage 6 — Model Pusher
- If the new model is accepted, pushes `model.pkl` to the **AWS S3 bucket** (`um-mlopsproj` / `model-registry`).
- **Output:** `ModelPusherArtifact`

---

## 6. Dataset & Features

**Dataset:** Vehicle Insurance Cross-Sell dataset (~380,000 records)

| Feature | Type | Description |
|---|---|---|
| `Gender` | Categorical | Male / Female |
| `Age` | Numerical | Customer age |
| `Driving_License` | Binary | 0 = No, 1 = Yes |
| `Region_Code` | Numerical | Region identifier |
| `Previously_Insured` | Binary | 0 = No prior vehicle insurance, 1 = Yes |
| `Vehicle_Age` | Categorical | `< 1 Year`, `1-2 Year`, `> 2 Years` |
| `Vehicle_Damage` | Categorical | `Yes` / `No` |
| `Annual_Premium` | Numerical | Annual health insurance premium amount |
| `Policy_Sales_Channel` | Numerical | Sales channel code |
| `Vintage` | Numerical | Days customer has been associated with company |
| **`Response`** | **Target (Binary)** | **1 = Interested, 0 = Not Interested** |

---

## 7. Setup & Installation

### Prerequisites
- [Conda](https://docs.conda.io/en/latest/) installed
- Python 3.10
- Git

### Step 1 — Clone the Repository
```bash
git clone https://github.com/your-username/vehicle-insurance-mlops-pipeline.git
cd vehicle-insurance-mlops-pipeline
```

### Step 2 — Create & Activate Conda Environment
```bash
conda create -n vehicle python=3.10 -y
conda activate vehicle
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

> The `-e .` at the bottom of `requirements.txt` installs the `src` package in editable mode using `setup.py`, making all `src.*` imports available.

### Step 4 — Verify Local Package Installation
```bash
pip list | grep src
```
You should see `src` listed as an installed package.

---

## 8. MongoDB Atlas Setup

1. Sign up at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas).
2. Create a new **Project** → **Create Cluster** → Select **M0 (Free)** tier → **Create Deployment**.
3. Set up a **Database User** with a username and password.
4. Go to **Network Access** → Add IP Address → Enter `0.0.0.0/0` (allow from anywhere).
5. Go to **Database** → **Connect** → **Drivers** → Select `Python 3.6 or later` → Copy the connection string.
6. Replace `<password>` in the connection string with your actual password.
7. Open `notebook/mongoDB_demo.ipynb`, select the `vehicle` kernel, and run it to push `data.csv` to MongoDB.
8. Verify: **MongoDB Atlas → Database → Browse Collections** — you should see the data in key-value format.

---

## 9. AWS Setup

### IAM User & Access Keys
1. Log in to **AWS Console** (region: `us-east-1`).
2. Go to **IAM** → **Users** → **Create User** (name: `firstproj`).
3. Attach policy: **AdministratorAccess** → **Create User**.
4. Go to user → **Security Credentials** → **Access Keys** → **Create Access Key** → Select **CLI** → Download CSV.

### S3 Bucket (Model Storage)
1. Go to **S3** → **Create Bucket**.
2. Region: `us-east-1`, Bucket name: `um-mlopsproj`.
3. **Uncheck** "Block all public access" and acknowledge.
4. Hit **Create Bucket**.

### ECR Repository (Docker Image Registry)
1. Go to **ECR** → **Create Repository**.
2. Region: `us-east-1`, Repository name: `vehicleproj`.
3. Copy and save the **URI**.

### EC2 Instance (Deployment Server)
1. Go to **EC2** → **Launch Instance**.
   - Name: `vehicledata-machine`
   - AMI: **Ubuntu Server 24.04 LTS** (Free Tier eligible)
   - Instance type: `t2.medium`
   - Create key pair (name: `proj1key`)
   - Allow HTTP and HTTPS traffic
   - Storage: **30 GB**
2. Launch and connect via **EC2 Instance Connect**.

#### Install Docker on EC2
```bash
# Update packages
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

#### Open Application Port on EC2
1. Go to your instance → **Security** → **Security Groups** → **Edit Inbound Rules**.
2. Add Rule: Type = `Custom TCP`, Port Range = `5000`, Source = `0.0.0.0/0`.
3. Save Rules.

---

## 10. Environment Variables

### Option A — PowerShell (Windows)
```powershell
$env:MONGODB_URL = "mongodb+srv://<username>:<password>@cluster.mongodb.net/?retryWrites=true&w=majority"
$env:AWS_ACCESS_KEY_ID = "your-access-key-id"
$env:AWS_SECRET_ACCESS_KEY = "your-secret-access-key"

# Verify
echo $env:MONGODB_URL
echo $env:AWS_ACCESS_KEY_ID
```

### Option B — Bash / Linux / Mac
```bash
export MONGODB_URL="mongodb+srv://<username>:<password>@cluster.mongodb.net/?retryWrites=true&w=majority"
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"

# Verify
echo $MONGODB_URL
echo $AWS_ACCESS_KEY_ID
```

### Option C — Windows System Environment Variables
1. Search **"Environment Variables"** in Start Menu.
2. Click **"Environment Variables"** → **New** (under User variables).
3. Add:
   - `MONGODB_URL` = your connection string
   - `AWS_ACCESS_KEY_ID` = your key
   - `AWS_SECRET_ACCESS_KEY` = your secret

> **Note:** The `artifact/` directory is listed in `.gitignore` and should never be committed to version control.

---

## 11. Running the Application

### Run the Training Pipeline (via demo.py)
```bash
python demo.py
```

### Start the FastAPI Server
```bash
python app.py
```
Access the app at: [http://localhost:5000](http://localhost:5000)

---

## 12. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Renders the prediction form |
| `POST` | `/` | Accepts form data and returns prediction |
| `GET` | `/train` | Triggers the full training pipeline |

### Prediction Response
- **Response-Yes** → Customer is likely interested in vehicle insurance
- **Response-No** → Customer is likely not interested

---

## 13. CI/CD Pipeline (GitHub Actions + AWS)

The pipeline is defined in `.github/workflows/aws.yaml` and triggers automatically on every push to the `main` branch.

### Pipeline Stages

**Continuous Integration (runs on `ubuntu-latest`):**
1. Checkout source code
2. Configure AWS credentials
3. Login to Amazon ECR
4. Build Docker image → Tag → Push to ECR

**Continuous Deployment (runs on `self-hosted` EC2 runner):**
1. Checkout source code
2. Configure AWS credentials
3. Login to Amazon ECR
4. Pull and run the Docker image on EC2, injecting all secrets as environment variables

### Connect EC2 as a Self-Hosted Runner
1. Go to your GitHub repo → **Settings** → **Actions** → **Runners** → **New self-hosted runner**.
2. Select OS: **Linux**.
3. Run all **Download** commands on your EC2 terminal.
4. Run the first **Configure** command (press Enter for defaults; set runner name as `self-hosted`).
5. Run `./run.sh` to start the runner.
6. Verify: runner status shows **"Idle"** in GitHub.

### Required GitHub Secrets
Go to: **GitHub Repo → Settings → Secrets and Variables → Actions → New Repository Secret**

| Secret Name | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | AWS IAM access key |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM secret key |
| `AWS_DEFAULT_REGION` | `us-east-1` |
| `ECR_REPO` | Your ECR repository name (e.g., `vehicleproj`) |
| `MONGODB_URL` | Your MongoDB Atlas connection string |

---

## 14. Docker

### Build Image Locally
```bash
docker build -t vehicle-insurance-app .
```

### Run Container Locally
```bash
docker run -d \
  -e MONGODB_URL="your-mongodb-url" \
  -e AWS_ACCESS_KEY_ID="your-key" \
  -e AWS_SECRET_ACCESS_KEY="your-secret" \
  -e AWS_DEFAULT_REGION="us-east-1" \
  -p 5000:5000 \
  vehicle-insurance-app
```

Access at: [http://localhost:5000](http://localhost:5000)

---

## 15. Project Flow Summary

```
1.  template.py          → Scaffold project directory structure
2.  setup.py             → Configure local package (src) installation
3.  requirements.txt     → Install all dependencies via pip
4.  MongoDB Atlas        → Create cluster, push dataset via mongoDB_demo.ipynb
5.  notebook/            → EDA & Feature Engineering (exp-notebook.ipynb)
6.  src/logger           → Custom logging module
7.  src/exception        → Custom exception handler
8.  src/constants        → Centralized constants for the entire project
9.  src/configuration    → MongoDB & AWS connection managers
10. src/data_access      → MongoDB → Pandas DataFrame utility
11. src/entity           → Config & artifact dataclasses
12. src/components       → All 6 pipeline components
13. src/pipline          → Training & Prediction pipeline orchestrators
14. app.py               → FastAPI web server (UI + API)
15. Dockerfile           → Containerize the application
16. .github/workflows/   → CI/CD automation via GitHub Actions
17. AWS EC2 + ECR        → Deploy and serve at scale
```

---

## Author

**Umar Faizan** — [mianumarzareen@gmail.com](mailto:mianumarzareen@gmail.com)

---

*Built as part of an end-to-end MLOps learning project.*
