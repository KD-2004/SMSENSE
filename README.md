# SMSense

## AI-Powered SMS Threat Intelligence for Indian SMS, Smishing & Vernacular Phishing

SMSense is a cybersecurity platform designed to analyze SMS messages for **phishing, smishing, scams, social engineering, credential theft, suspicious URLs, and obfuscation indicators**, with particular attention to **Indian English, Hindi, and Roman-Hinglish messages**.

The platform combines:

- Pretrained machine-learning models
- Rule-based behavioral analysis
- URL intelligence
- Sender analysis
- Obfuscation detection
- Semantic similarity search
- FAISS vector retrieval
- Post-hoc LLM explanations
- Persistent analysis history
- Adversarial SMS testing
- REST APIs and system-health monitoring

> **Important:** SMSense's final security decision is produced by an interpretable rule-based evidence-fusion engine. Qwen3 is used for post-hoc explanation only and does not override the final risk score or label.

---

## What is SMSense?

SMSense is an AI-assisted SMS security analysis platform focused on identifying suspicious messages before users interact with them.

Modern phishing and smishing messages frequently use:

- Urgency and account threats
- KYC/account verification themes
- OTP and credential requests
- UPI/PIN/CVV/password requests
- Reward, lottery, cashback, and job scams
- Suspicious or shortened URLs
- Brand impersonation
- Sender/header anomalies
- Intentional spelling changes
- Leetspeak and character substitutions
- Roman-Hinglish and vernacular language
- Small text mutations designed to evade simple keyword filters

SMSense does not rely on a single classifier. Instead, it collects evidence from multiple independent analysis paths and combines that evidence into a transparent score.

---

# Problem

SMS phishing is especially effective when attackers use short, conversational, localized messages.

A scam such as:

```text
KYC update karo warna account block hoga
```

can be transformed into:

```text
KYC upd8 kro warna acc0unt blck hoga
```

A simple keyword filter may struggle to recognize that these two messages express the same threat.

At the same time, legitimate security messages can contain words such as `OTP`, `login`, `account`, and `verify`:

```text
Your OTP for login is 4821. Do not share it with anyone.
```

SMSense therefore attempts to distinguish:

- Security vocabulary from malicious intent
- Normal messages from social-engineering combinations
- Clean text from obfuscated variants
- Model evidence from the final security decision

---

# Solution

SMSense uses a multi-stage evidence architecture:

```text
                         SMS / Message
                              |
                              v
                      Structured Parser
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
          Content          Behavior            URL
          Analysis         Analysis          Analysis
             |                |                |
          HinSpam         Rule Engine       URL Rules
          SilverGuard
             |                |
             +--------+-------+
                      |
                      v
              Sender + Obfuscation
                      |
                      v
             Qwen3 Embedding + FAISS
                      |
                      v
             Evidence Fusion Engine
                      |
                      v
                 Risk Score 0-100
                      |
                      v
          LOW / SUSPICIOUS / HIGH RISK / CRITICAL
                      |
                      v
                   Qwen3 8B
                      |
                      v
              Post-Hoc Explanation
```

The frontend is presentation-only. The FastAPI backend is authoritative for the security decision.

---

# Key Features

## 1. SMS Analysis

Analyze:

- Message body
- Sender ID / SMS header / phone number
- Optional timestamp

The result can include:

- Risk score
- Risk label
- Scam category
- Language
- Raw message
- Normalized message
- Suspicious phrases
- Evidence breakdown
- URL intelligence
- Obfuscation analysis
- Model signals
- Explanation
- Recommended action

---

## 2. Vernacular and Hinglish Support

SMSense recognizes Roman-Hinglish indicators.

Example:

```text
KYC upd8 kro warna account blck hoga
```

can be identified as:

```text
hinglish
```

while:

```text
Your OTP for login is 4821. Do not share it with anyone.
```

can be identified as:

```text
en
```

The current language logic is lightweight and rule-based rather than a dedicated multilingual language model.

---

## 3. Obfuscation Detection

**Obfuscation** means deliberately modifying text so it becomes harder for simple detection systems to recognize.

SMSense checks for patterns such as:

- Leetspeak
- Misspellings
- Transliteration
- Punctuation manipulation
- Spacing manipulation
- Repeated characters
- Concatenated or split patterns
- Spelling normalization

Examples:

```text
upd8      -> update
blck      -> block
acc0unt   -> account
```

Another example:

```text
KYC update karo warna account block hoga
```

becomes:

```text
KYC upd8 kro warna acc0unt blck hoga
```

SMSense preserves both the original and normalized message.

---

## 4. Behavior and Social-Engineering Analysis

The backend detects patterns involving:

- Urgency
- Threats
- KYC
- OTP
- UPI
- PIN
- CVV
- Passwords
- Rewards/prizes
- Authority themes
- Credential requests

It also recognizes combinations such as:

```text
KYC + account threat
KYC + urgency
KYC + URL
OTP + credential request
Reward + payment request
Job offer + fee
```

A single keyword is not automatically treated as proof of phishing.

For example:

```text
OTP
```

alone does not necessarily indicate an attack.

But:

```text
Send me your OTP to verify your account
```

contains an explicit credential-sharing request and becomes much stronger evidence.

---

# Risk Scoring

SMSense uses a transparent evidence-fusion model.

| Evidence Category | Maximum Contribution |
|---|---:|
| Content | 30 |
| Sender | 15 |
| URL | 20 |
| Behavior | 20 |
| Obfuscation | 10 |
| Similarity | 5 |
| **Total** | **100** |

The final number is an **evidence-based risk score**, not a calibrated probability.

For example:

```text
37 / 100
```

does **not** mean:

```text
37% probability of phishing
```

It means the configured evidence sources accumulated 37 points.

Risk labels:

```text
LOW
SUSPICIOUS
HIGH RISK
CRITICAL
```

The exact score thresholds are configurable in the backend.

---

# Example Analysis

## Legitimate OTP

Input:

```text
Your OTP for login is 4821. Do not share it with anyone.
```

A tested backend result was approximately:

```text
Risk Score: 1 / 100
Label: LOW
Category: OTP_NOTICE
Language: en
```

This demonstrates an important SMSense principle:

> The presence of an OTP keyword alone does not make a message malicious.

---

## KYC Phishing

Input:

```text
KYC upd8 kro warna account blck hoga
```

A tested backend result was:

```text
Risk Score: 37 / 100
Label: SUSPICIOUS
Category: KYC_FRAUD
Language: hinglish
```

Example evidence:

```text
Content:       14.9752
Behavior:      10.0
Obfuscation:    7.5
Similarity:     4.5644
Sender:         0
URL:            0
```

Detected transformations:

```text
upd8 -> update
blck -> block
```

Detected behavioral combination:

```text
KYC plus account threat
```

---

## OTP Theft

Input:

```text
Send me your OTP to verify your account
```

A tested backend result was:

```text
Risk Score: 31 / 100
Label: SUSPICIOUS
Category: OTP_THEFT
Language: en
```

Detected signal:

```text
OTP sharing request
```

---

# Detection Pipeline

## Phase 1 — Ingest and Parse

Input:

```text
Message + Sender + Timestamp
```

The parser extracts:

- Raw message
- Sender
- Timestamp
- URLs
- Domains
- Message length

The raw message is preserved exactly.

---

## Phase 2 — Independent Evidence Extraction

### Content

Signals from:

- HinSpam
- SilverGuard

### Behavior

Rule-based social-engineering indicators.

### URL

URL structural analysis.

### Sender

Sender/header analysis.

### Obfuscation

Normalization and text-manipulation analysis.

### Similarity

Qwen3 Embedding + FAISS retrieval.

---

## Phase 3 — Evidence Fusion

The independent signals are converted into the six evidence groups:

```text
Content
Sender
URL
Behavior
Obfuscation
Similarity
```

Each group has a maximum contribution.

---

## Phase 4 — Final Risk Score

The evidence values are combined into a 0–100 score.

---

## Phase 5 — Risk Label

The score is mapped to:

```text
LOW
SUSPICIOUS
HIGH RISK
CRITICAL
```

---

## Phase 6 — Post-Hoc Explanation

Qwen3 8B receives the already-determined result and generates:

- Human-readable summary
- Suspicious phrases
- Recommended user action

Qwen3 cannot change the security score or label.

---

## Phase 7 — Persistence

Completed analyses are stored in SQLite so they can appear in:

- Dashboard
- History
- Statistics
- Feedback workflows

---

# Pretrained Models

## HinSpam

Model:

```text
Keshav0av/HinSpam-spam-detection
```

Purpose:

- Content evidence
- Spam/scam-related text signal

---

## SilverGuard

Model:

```text
tanishqmudaliar/SilverGuard
```

Purpose:

- Additional threat evidence
- Independent model signal

---

## Qwen3 Embedding

Ollama model:

```text
qwen3-embedding:0.6b
```

Purpose:

- Generate message embeddings
- Support semantic similarity search

---

## Qwen3 8B

Ollama model:

```text
qwen3:8b
```

Purpose:

- Generate security explanations
- Explain suspicious phrases
- Recommend safe actions

Qwen3 8B is **not** the source of the final security label.

---

# Semantic Similarity and FAISS

SMSense uses:

```text
Ollama
Qwen3 Embedding 0.6b
FAISS CPU
NumPy
```

The current local demo index contains:

```text
Vectors:   3
Dimension: 1024
Metric:    L2
```

Pipeline:

```text
Message
   |
   v
Qwen3 Embedding
   |
   v
Vector
   |
   v
FAISS
   |
   v
Nearest Scam Examples
```

Similarity is supporting evidence only and is capped at 5 points.

It is not presented as a phishing probability.

---

# URL Intelligence

SMSense can inspect structural URL indicators such as:

- Shortened URLs
- Suspicious URL structure
- IP-address URLs
- Brand/domain mismatch
- Domain information
- Reputation state when available

Example:

```text
URL:        https://bit.ly/example
Domain:     bit.ly
Shortened:  true
Reputation: not_checked
```

The system does not claim that a URL is malicious merely because it has a structural indicator.

---

# Sender Analysis

Sender analysis considers:

- Phone vs. SMS header format
- Sender structure
- Suspicious patterns
- Promotional-looking identifiers
- Sender anomalies

A sender such as:

```text
SBI
```

is not automatically treated as malicious.

Actual anomaly evidence is required.

---

# Qwen3 Explanation Layer

The model is intentionally contained after the security decision:

```text
Evidence
   |
   v
Rule-Based Fusion
   |
   v
Final Score + Label
   |
   v
Qwen3 8B
   |
   v
Explanation Only
```

This means an LLM cannot simply say:

> "This looks safe"

and override the underlying security evidence.

---

# Adversarial Testing

SMSense includes an adversarial-testing workflow.

### Original

```text
KYC update karo warna account block hoga
```

### Obfuscated Variant

```text
KYC upd8 kro warna acc0unt blck hoga
```

The adversarial workflow can compare:

- Score
- Label
- Category
- Language
- Evidence
- Obfuscation indicators

This demonstrates robustness against simple spelling and text mutations.

---

# Technology Stack

## Frontend

- HTML5
- CSS3
- Vanilla JavaScript
- Single-page `index.html`
- Responsive SOC-style interface

## Backend

- Python
- FastAPI
- Uvicorn
- Pydantic

## AI / ML

- HinSpam
- SilverGuard
- Qwen3 8B
- Qwen3 Embedding 0.6B

## Retrieval

- FAISS CPU
- NumPy

## Runtime

- Ollama
- Hugging Face Transformers
- Hugging Face Hub
- ONNX Runtime

## Storage

- SQLite

## API

- REST
- JSON

---

# Project Structure

```text
SMSENSE/
│
├── index.html
│
├── artifacts/
│   └── api-server/
│       ├── app/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── fusion.py
│       │   ├── main.py
│       │   ├── models.py
│       │   ├── normalizer.py
│       │   ├── parser.py
│       │   ├── rules.py
│       │   ├── schemas.py
│       │   ├── service.py
│       │   ├── similarity.py
│       │   └── storage.py
│       │
│       ├── data/
│       ├── tools/
│       │   ├── build_faiss_index.py
│       │   └── validate_messages.py
│       │
│       ├── requirements.txt
│       └── README.md
│
├── screenshots/
│
├── .gitignore
├── .replit
└── package.json
```

---

# API Endpoints

## Health

```http
GET /api/health
```

Returns detailed component health for:

- Ollama
- Qwen3 8B
- Qwen3 Embedding
- HinSpam
- SilverGuard
- FAISS
- SQLite

---

## Lightweight Health

```http
GET /api/healthz
```

Example:

```json
{
  "status": "ok"
}
```

---

## Analyze SMS

```http
POST /api/analyze
```

Example:

```json
{
  "message": "KYC upd8 kro warna account blck hoga",
  "sender": "VM-SBIHELP",
  "timestamp": null
}
```

---

## Adversarial Test

```http
POST /api/adversarial-test
```

---

## History

```http
GET /api/history?limit=50
```

Maximum supported limit:

```text
500
```

---

## History Detail

```http
GET /api/history/{analysis_id}
```

---

## Statistics

```http
GET /api/stats
```

---

## Feedback

```http
POST /api/feedback
```

Example:

```json
{
  "analysis_id": 16,
  "helpful": true,
  "note": "Correctly identified KYC scam"
}
```

---

# Frontend

SMSense uses a single-page:

```text
index.html
```

The UI provides logical views for:

- Dashboard
- Analyze SMS
- Adversarial Test
- History
- URL Intelligence
- Models & Evidence
- System Health
- Feedback

The frontend communicates with the FastAPI backend.

It does **not**:

- Call Ollama directly
- Call Hugging Face directly
- Calculate the phishing score
- Store AI API keys

This keeps security logic centralized in the backend.

---

# Backend

The FastAPI backend is located at:

```text
artifacts/api-server/app/
```

### `main.py`

- FastAPI application
- CORS
- API endpoints
- Health checks
- Error handling

### `parser.py`

- URL extraction
- Domain extraction
- Sender/timestamp preservation
- Message length

### `normalizer.py`

- Normalization
- Leetspeak
- Obfuscation detection
- Transliteration indicators
- Spacing and punctuation analysis

### `rules.py`

- Behavior signals
- Sender analysis
- URL analysis
- Language hint
- Social-engineering combinations

### `fusion.py`

- Evidence caps
- Rule-based fusion
- Final score
- Severity label
- Category selection

### `models.py`

- Hugging Face model adapters
- Ollama client
- Model availability
- Qwen3 explanation

### `similarity.py`

- Embeddings
- FAISS
- Similarity search
- Index availability

### `storage.py`

- SQLite persistence
- History
- Feedback
- Statistics

### `service.py`

Coordinates the analysis pipeline.

### `schemas.py`

Defines API request and response models.

---

# Local Setup

## Requirements

Recommended:

- Python 3.12
- Git
- Ollama
- Sufficient local RAM/CPU for the configured models

---

## Clone

```bash
git clone https://github.com/KD-2004/SMSENSE.git
cd SMSENSE
```

---

## Create Virtual Environment

### Windows

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

---

## Install Backend Dependencies

```bash
cd artifacts/api-server
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

# Ollama Setup

Install Ollama and pull:

```bash
ollama pull qwen3:8b
ollama pull qwen3-embedding:0.6b
```

Verify:

```bash
ollama list
```

---

# Build the FAISS Index

From:

```text
artifacts/api-server/
```

run:

```bash
python tools/build_faiss_index.py
```

The command requires the embedding model to be available through Ollama.

---

# Running SMSense

## Start the Backend

From:

```text
SMSENSE/artifacts/api-server/
```

run:

```powershell
python -m uvicorn app.main:app --reload
```

API:

```text
http://127.0.0.1:8000
```

---

## Verify Backend Health

```powershell
Invoke-RestMethod `
    http://127.0.0.1:8000/api/healthz |
    ConvertTo-Json
```

Expected:

```json
{
  "status": "ok"
}
```

Detailed:

```powershell
Invoke-RestMethod `
    http://127.0.0.1:8000/api/health |
    ConvertTo-Json -Depth 10
```

---

## Run the Frontend

Open or serve:

```text
index.html
```

The frontend currently communicates with:

```text
http://127.0.0.1:8000
```

For a deployed environment, replace the local API URL with the public backend URL.

---

# API Testing

## KYC Scam

```powershell
$body = @{
    message = "KYC upd8 kro warna account blck hoga"
    sender  = "VM-SBIHELP"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/api/analyze" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body |
    ConvertTo-Json -Depth 30
```

---

## Legitimate OTP

```powershell
$body = @{
    message = "Your OTP for login is 4821. Do not share it with anyone."
    sender  = "SBI"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/api/analyze" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body |
    ConvertTo-Json -Depth 30
```

---

# Example Messages

### Suspicious — KYC / Account Threat

```text
KYC upd8 kro warna account blck hoga
```

### Suspicious — OTP Theft

```text
Send me your OTP to verify your account
```

### Suspicious — Hinglish OTP Theft

```text
OTP bhejo abhi warna account block hoga
```

### Suspicious — Job Scam

```text
Congratulations! You are selected for work from home. Pay ₹999 registration fee to start.
```

### Suspicious — Reward Scam

```text
Congratulations! You have won a ₹50,000 cashback reward. Pay ₹499 processing fee to claim your prize.
```

### Benign — OTP Notice

```text
Your OTP for login is 4821. Do not share it with anyone.
```

### Benign — Normal Hinglish

```text
Kal class 10 baje hai bhai.
```

### Benign — Normal Transaction

```text
Your UPI payment of ₹500 to Rahul was successful.
```

---

# Dashboard and History

The Dashboard shows:

- Total analyzed messages
- Low-risk count
- Suspicious count
- High-risk count
- Critical count
- Recent detections
- Risk distribution
- Category distribution

History allows previously analyzed messages to be reopened and inspected again.

---

# System Health

The System Health page reports runtime availability of:

```text
Ollama
Qwen3 8B
Qwen3 Embedding 0.6B
HinSpam
SilverGuard
FAISS
SQLite
```

Overall status can be:

```text
ok
```

or:

```text
degraded
```

Component availability comes from runtime checks rather than placeholder status.

---

# Security Design Principles

## Explainability

The final score is decomposed into visible evidence categories.

## No Black-Box Fusion

SMSense does not use a trained meta-classifier for the final decision. The score comes from explicit evidence rules and category caps.

## LLM Containment

Qwen3 does not override the security decision.

## Fail-Closed Model Handling

When required components are unavailable, the backend can return a model-unavailable error rather than fabricating a security result.

## Separation of Responsibilities

```text
Frontend
    ↓
Input + Display

FastAPI
    ↓
Analysis Orchestration

Models + Rules
    ↓
Evidence

Fusion Engine
    ↓
Final Score + Label

Qwen3
    ↓
Explanation Only
```

## Raw Message Preservation

The original message is retained separately from normalized text.

---

# Limitations

SMSense is a cybersecurity research and prototype system and should not be considered a complete production anti-phishing gateway.

Current limitations include:

- Small local semantic-similarity corpus
- Similarity is supporting evidence, not probability
- URL reputation depends on the intelligence actually available
- `not_checked` does not mean malicious
- Lightweight rule-based language identification
- Pretrained model behavior depends on model training distributions
- Different model signals may disagree
- The final score is not statistically calibrated
- Local inference speed depends on CPU/RAM
- A security label is a signal, not forensic proof

---

# Future Improvements

Potential next steps:

- Larger curated benign/scam dataset
- More Indian-language support
- Improved language identification
- Larger FAISS knowledge base
- URL reputation providers
- Redirect-chain analysis
- Domain-age intelligence
- Formal precision/recall/F1 evaluation
- Held-out validation
- Risk-score calibration
- Model ablation studies
- Production authentication
- Centralized logging
- SIEM integration
- Containerized deployment
- Cloud inference
- Mobile integration
- Streaming SMS analysis

---

# Screenshots

Create a folder at the repository root:

```text
screenshots/
```

Recommended files:

```text
screenshots/
├── dashboard.png
├── analyze-sms.png
├── kyc-analysis.png
├── otp-safe-analysis.png
├── otp-theft-analysis.png
├── adversarial-test.png
├── history.png
├── url-intelligence.png
├── models-evidence.png
└── system-health.png
```

Then use these Markdown entries.

## Dashboard

```markdown
![SMSense Dashboard](screenshots/dashboard.png)
```

## Analyze SMS

```markdown
![SMSense Analyze SMS](screenshots/analyze-sms.png)
```

## KYC Phishing Detection

```markdown
![SMSense KYC Analysis](screenshots/kyc-analysis.png)
```

## Safe OTP Detection

```markdown
![SMSense Safe OTP](screenshots/otp-safe-analysis.png)
```

## OTP Theft Detection

```markdown
![SMSense OTP Theft](screenshots/otp-theft-analysis.png)
```

## Adversarial Testing

```markdown
![SMSense Adversarial Test](screenshots/adversarial-test.png)
```

## History

```markdown
![SMSense History](screenshots/history.png)
```

## URL Intelligence

```markdown
![SMSense URL Intelligence](screenshots/url-intelligence.png)
```

## Models & Evidence

```markdown
![SMSense Models and Evidence](screenshots/models-evidence.png)
```

## System Health

```markdown
![SMSense System Health](screenshots/system-health.png)
```

### Recommended GitHub screenshot order

1. Dashboard
2. Analyze SMS
3. KYC phishing result
4. Safe OTP result
5. OTP theft result
6. Adversarial Test
7. Models & Evidence
8. System Health

This gives visitors a quick story:

```text
Dashboard
   ↓
Analyze
   ↓
Detect
   ↓
Explain
   ↓
Test Robustness
   ↓
Show Architecture
   ↓
Show Runtime Health
```

---

# Demo Flow

## Step 1 — Dashboard

Show:

- Current totals
- Recent detections
- Risk distribution
- Category distribution

## Step 2 — Safe OTP

```text
Your OTP for login is 4821. Do not share it with anyone.
```

Show:

```text
LOW
OTP_NOTICE
```

Explain that OTP terminology alone is not sufficient to classify a message as malicious.

## Step 3 — Vernacular Phishing

```text
KYC upd8 kro warna account blck hoga
```

Show:

```text
SUSPICIOUS
KYC_FRAUD
```

Then show:

- KYC + account threat
- Obfuscation
- Model signal
- Semantic similarity

## Step 4 — OTP Theft

```text
Send me your OTP to verify your account
```

Show:

```text
SUSPICIOUS
OTP_THEFT
```

## Step 5 — Adversarial Test

Compare clean and obfuscated variants.

## Step 6 — Models & Evidence

Explain:

```text
HinSpam
SilverGuard
Behavior Rules
URL Analysis
Obfuscation
Qwen3 Embedding
FAISS
Rule-Based Fusion
Qwen3 Explanation
```

## Step 7 — System Health

Show that the configured components are operational.

---

# Repository

GitHub:

https://github.com/KD-2004/SMSENSE

---

# Team

## Team Lozers

**Project:** SMSense — AI-Powered SMS Threat Intelligence

---

# Disclaimer

SMSense is a cybersecurity research and hackathon prototype intended for defensive analysis and demonstration.

A `SUSPICIOUS`, `HIGH RISK`, or `CRITICAL` result should be treated as a security signal requiring review or verification, not as legal or forensic proof that a message or sender is malicious.

Likewise, a `LOW` result does not guarantee that a message is completely safe.

Always verify sensitive requests through official channels.

---

# License

No license is currently specified for the repository.

Add a license only after selecting the appropriate license for the project.

---

# Acknowledgements

SMSense integrates and builds upon open-source and pretrained technologies including:

- FastAPI
- Hugging Face Transformers
- Hugging Face Hub
- Ollama
- FAISS
- NumPy
- ONNX Runtime
- HinSpam
- SilverGuard
- Qwen3 models

Please review the individual software and model licenses before production or commercial use.
