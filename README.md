# 📰 Fake News Detector - MLOps Pipeline

A professional, end-to-end MLOps project that classifies news articles as **REAL** or **FAKE** using a fine-tuned BERT model. This project features a fully automated CI/CD pipeline, centralized logging, secret management, and automated model retraining.

## 🚀 Key Features
- **AI Core:** BERT (Transformers) fine-tuned for sequence classification.
- **Backend:** FastAPI for high-performance prediction and feedback endpoints.
- **Observability:** Centralized logging with the **ELK Stack** (Elasticsearch, Logstash, Kibana).
- **Security:** Secret management via **HashiCorp Vault**.
- **Orchestration:** **Kubernetes** (Minikube) for deployment with HPA and rolling updates.
- **CI/CD:** **Jenkins** pipelines for automated builds, testing, and deployment.
- **Automation:** **Ansible** for configuration management and K8s orchestration.
- **Retraining Loop:** Automated triggering of retraining pipelines based on misclassification thresholds.

## 🏗️ Architecture
1. **Frontend:** React-based UI for users to input news text.
2. **API:** Receives text, gets prediction from BERT, and stores event in SQLite + Logstash.
3. **Feedback Loop:** If a user marks a prediction as wrong, the API tracks it.
4. **Trigger:** Once misclassifications hit a threshold (e.g., 2), the API calls Jenkins.
5. **Retrain Pipeline:** Jenkins trains a new model, builds a Docker image, and patches the K8s cluster.

---

## 🛠️ Setup & Installation

### 1. Prerequisites
- Docker & Docker Compose
- Minikube & Kubectl
- Jenkins (running locally or in Docker)
- Python 3.10+

### 2. Infrastructure (ELK + Vault)
Start the supporting tools using Docker Compose:
```bash
docker compose up -d
```
Access Kibana at `http://localhost:5601`.

### 3. Kubernetes Setup
Start Minikube and apply the manifests:
```bash
minikube start --cpus 4 --memory 6000
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### 4. Running the API Locally (Dev Mode)
```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

---

## 🧪 Testing the Pipeline
1.  **Prediction:** Submit a news snippet to the UI.
2.  **Feedback:** Mark 2 predictions as "Incorrect".
3.  **Automation:** Observe Jenkins automatically starting the `MlOps-retrain` job.
4.  **Live Patch:** Once the job finished, run `kubectl get pods` to see your pods being updated with the new model.

## 📁 Project Structure
- `api/`: FastAPI backend and logic.
- `frontend/`: Static UI files.
- `model/`: BERT training scripts and model weights.
- `k8s/`: Kubernetes manifests (Deployment, Service, HPA).
- `jenkins/`: CI/CD Pipeline definitions (Jenkinsfile).
- `ansible/`: Roles and playbooks for deployment.
- `logstash/`: Configuration for log processing.

---
**Author:** Shruti Verma / svrma13
