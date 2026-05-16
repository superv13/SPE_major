from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from pydantic import BaseModel
import os
import sqlite3
import requests
import logging
import requests
import logging
import hvac
from logstash_async.handler import AsynchronousLogstashHandler
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import socket

# Logstash Configuration (Disabled by default for clean logs)
ENABLE_LOGSTASH = os.environ.get('ENABLE_LOGSTASH', 'false').lower() == 'true'
LOGSTASH_HOST = os.environ.get('LOGSTASH_HOST', 'localhost')
LOGSTASH_PORT = 5000

# Root Logger Config
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if ENABLE_LOGSTASH:
    try:
        from logstash_async.handler import AsynchronousLogstashHandler
        logstash_handler = AsynchronousLogstashHandler(
            host=LOGSTASH_HOST, 
            port=LOGSTASH_PORT, 
            database_path='logstash.db'
        )
        logger.addHandler(logstash_handler)
        logger.info(f"✅ Logstash connected at {LOGSTASH_HOST}:{LOGSTASH_PORT}")
    except Exception as e:
        logger.warning(f"❌ Failed to initialize Logstash: {e}")
else:
    logger.info("ℹ️ Logstash is disabled. Set ENABLE_LOGSTASH=true in .env to enable.")

app = FastAPI()

JENKINS_AUTH = None

def fetch_vault_secrets():
    global JENKINS_AUTH
    logger.info("Attempting to fetch secrets from HashiCorp Vault...")
    vault_url = os.environ.get('VAULT_ADDR', 'http://vault:8200')
    vault_token = os.environ.get('VAULT_TOKEN')  # Should be provided via environment variable
    
    if not vault_token:
        logger.warning("VAULT_TOKEN not set. Vault integration will be skipped.")
        return

    try:
        client = hvac.Client(url=vault_url, token=vault_token)
        if client.is_authenticated():
             # Attempt to read Jenkins credentials from Vault KV engine
             try:
                 read_response = client.secrets.kv.v2.read_secret_version(path='jenkins', mount_point='secret')
                 secrets = read_response['data']['data']
                 JENKINS_AUTH = (secrets['username'], secrets['password'])
                 logger.info("Successfully authenticated and fetched Jenkins credentials from Vault.")
             except Exception as e:
                 logger.error(f"Vault authenticated but failed to read 'jenkins' secret: {e}")
    except Exception as e:
        logger.warning(f"Could not connect to Vault service at {vault_url}: {e}")

fetch_vault_secrets()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "../frontend")), name="static")



MODEL_PATH = os.path.join(BASE_DIR, "../model/bert_fake_news_model")
DB_PATH = os.path.join(BASE_DIR, "../db/fake_news.db")

def init_db():
    db_dir = os.path.dirname(DB_PATH)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    logger.info(f"Initializing database at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS news_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_text TEXT,
            predicted_label TEXT,
            confidence REAL,
            correct_label TEXT,
            feedback_given INTEGER DEFAULT 0,
            is_misclassified INTEGER DEFAULT 0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

model_name_or_path = MODEL_PATH if os.path.exists(MODEL_PATH) else "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
model = BertForSequenceClassification.from_pretrained(model_name_or_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ---------- Request Models ----------

class InputText(BaseModel):
    text: str

class FeedbackInput(BaseModel):
    id: int
    correct_label: str

# ---------- Home ----------

@app.get("/")
def home():
    return FileResponse(os.path.join(BASE_DIR, "../frontend/index.html"))

# ---------- Predict ----------

@app.post("/predict/")
@app.post("/predict")
def predict(data: InputText):

    inputs = tokenizer(
        data.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        pred = torch.argmax(logits, dim=1).item()
        probs = torch.softmax(logits, dim=1)
        confidence = probs[0][pred].item()

    label_map = {0: "FAKE", 1: "REAL"}
    label = label_map[pred]

    # Insert into SQLite
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO news_predictions
        (article_text, predicted_label, confidence)
        VALUES (?, ?, ?)
    """, (data.text, label, confidence))

    row_id = cur.lastrowid

    conn.commit()
    conn.close()

    # Mirror the DB record to ELK as a structured log event
    logger.info(
        "prediction_stored",
        extra={
            "event_type": "prediction",
            "record_id": row_id,
            "article_text": data.text,
            "predicted_label": label,
            "confidence": round(confidence, 4),
        }
    )

    return {
        "id": row_id,
        "prediction": label,
        "confidence": round(confidence, 4)
    }


# ---------- Feedback ----------

@app.post("/feedback/")
@app.post("/feedback")
def feedback(data: FeedbackInput):

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Get predicted label
    cur.execute(
        "SELECT predicted_label FROM news_predictions WHERE id=?",
        (data.id,)
    )

    row = cur.fetchone()

    if not row:
        conn.close()
        return {"error": "Record not found"}

    predicted = row[0]

    misclassified = 1 if predicted != data.correct_label.upper() else 0

    cur.execute("""
        UPDATE news_predictions
        SET correct_label=?,
            feedback_given=1,
            is_misclassified=?
        WHERE id=?
    """, (data.correct_label.upper(), misclassified, data.id))

    # Mirror the feedback update to ELK as a structured log event
    logger.info(
        "feedback_stored",
        extra={
            "event_type": "feedback",
            "record_id": data.id,
            "predicted_label": predicted,
            "correct_label": data.correct_label.upper(),
            "is_misclassified": bool(misclassified),
        }
    )

    if misclassified == 1:
        cur.execute("SELECT COUNT(*) FROM news_predictions WHERE is_misclassified=1")
        count = cur.fetchone()[0]
        if count > 0 and count % 2 == 0:
            logging.warning(f"Misclassification count hit {count}. Triggering Jenkins retraining pipeline!")
            try:
                # Determine credentials source
                username = os.environ.get('JENKINS_USER', 'admin')
                password = None
                
                if JENKINS_AUTH:
                    username, password = JENKINS_AUTH
                    logger.info("Using Jenkins credentials from Vault.")
                else:
                    password = os.environ.get('JENKINS_PASS')
                    if password:
                        logger.info("Using Jenkins credentials from Environment Variables.")
                    else:
                        logger.error("No Jenkins credentials found in Vault or Environment Variables. Trigger aborted.")
                        return

                    auth = (username, password)
                    session = requests.Session()
                    session.auth = auth
                    
                    # Jenkins Config
                    JENKINS_BASE_URL = os.environ.get('JENKINS_URL', 'http://localhost:8080')
                    JOB_NAME = os.environ.get('JENKINS_JOB_NAME', 'MlOps-retrain')

                    # Fetch Crumb for CSRF protection
                    crumb_url = f"{JENKINS_BASE_URL}/crumbIssuer/api/json"
                    crumb_resp = session.get(crumb_url, timeout=5)
                    
                    headers = {}
                    if crumb_resp.status_code == 200:
                        crumb_data = crumb_resp.json()
                        headers = {crumb_data['crumbRequestField']: crumb_data['crumb']}
                    else:
                        logger.warning(f"Could not fetch Jenkins crumb (Status: {crumb_resp.status_code}).")
                    
                    webhook_url = f"{JENKINS_BASE_URL}/job/{JOB_NAME}/build"
                    resp = session.post(webhook_url, headers=headers, timeout=5)
                    
                    if resp.status_code in [200, 201, 202]:
                        logger.info(f"✅ Successfully triggered Jenkins job '{JOB_NAME}' (Status: {resp.status_code}).")
                    else:
                        logger.error(f"❌ Failed to trigger Jenkins. Status: {resp.status_code}, URL: {webhook_url}")
            except Exception as e:
                logger.error(f"Critical error during Jenkins trigger: {e}")

    conn.commit()
    conn.close()

    return {
        "id": data.id,
        "predicted_label": predicted,
        "correct_label": data.correct_label.upper(),
        "is_misclassified": misclassified
    }

app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "../frontend")), name="root_static")