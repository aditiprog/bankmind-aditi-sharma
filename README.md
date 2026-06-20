# 🏦 BankMind — Customer Subscription Prediction

**BankMind** is a machine learning system that predicts whether a bank customer will subscribe to a term deposit — and explains *why*, using an LLM-powered reasoning layer on top of the prediction.

It combines a classic, well-tested ML pipeline with a modern API layer, making it equally useful as a **data science showcase** and a **deployable backend service**.

---
<img width="1919" height="960" alt="Screenshot 2026-06-20 202211" src="https://github.com/user-attachments/assets/a8e294b5-77f5-4551-8ae6-401489bf1700" />


## ✨ Why BankMind?

- 📊 **End-to-end ML workflow** — from raw data to a served prediction
- 🤖 **Two models compared** — Logistic Regression baseline vs. Random Forest
- ⚡ **Production-ready API** — built with FastAPI, fully documented via Swagger UI
- 🧠 **AI-generated explanations** — Groq LLM turns raw model output into human-readable reasoning
- 🔌 **Simple, clean endpoints** — easy to integrate into any frontend or business workflow

---

## 🔄 System Architecture

```
UCI Bank Marketing Dataset
            │
            ▼
     Data Preprocessing
(Missing values + OneHot Encoding)
            │
            ▼
  Model Training & Evaluation
(Logistic Regression + Random Forest)
            │
            ▼
   Saved Model (.pkl / joblib)
            │
            ▼
      FastAPI Application
            │
   ┌────────┼────────┐
   ▼        ▼         ▼
/health  /predict  /explain
            │         │
            ▼         ▼
      ML Prediction  Groq LLM
            │         │
            └────┬────┘
                 ▼
           JSON Response
```

---

## 🧱 Project Highlights

| Stage | Description |
|---|---|
| **EDA** | Exploratory analysis of customer demographics and campaign data |
| **Modeling** | Logistic Regression (baseline) + Random Forest (primary model) |
| **Evaluation** | Performance comparison across accuracy, precision, recall, F1 |
| **Serving** | FastAPI app exposing prediction and explanation endpoints |
| **Explainability** | Groq LLM converts model predictions into natural-language insights |

---

## 📁 Project Structure

```
BankMind/
│
├── app.py                            # FastAPI app — defines all API routes
├── bank.py                           # Data preprocessing + model training
├── explaination.md                   # Groq LLM explanation logic & prompt design
│
├── bank-full.csv                     # UCI Bank Marketing dataset
├── customer_subscription_model.pkl   # Trained model (Random Forest)
│
├── requirements.txt                  # Python dependencies
├── .env                              # Environment variables (Groq API key)
└── .gitignore                        # Files/folders excluded from git
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd BankMind
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the project root and add your Groq API key:
```env
GROQ_API_KEY=your_api_key_here
```

### 4. Run the API
```bash
python -m uvicorn app:app --reload
```

### 5. Open the interactive docs
```
http://127.0.0.1:8000/docs
```

You now have a live ML API running locally with auto-generated Swagger documentation. 🎉

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check — confirms the API is running |
| `POST` | `/predict` | Returns the model's subscription prediction for a customer |
| `POST` | `/explain` | Returns an AI-generated, plain-language explanation of the prediction |

---

## 🛠️ Tech Stack

**Language & Core**
- Python 3.x

**Data Science & Machine Learning**
- pandas — data loading & manipulation
- NumPy — numerical computing
- scikit-learn — Logistic Regression, Random Forest, preprocessing & evaluation
- joblib — model serialization (saving/loading `.pkl` files)

**API & Backend**
- FastAPI — REST API framework
- Uvicorn — ASGI server
- Pydantic — request/response data validation

**AI / LLM Layer**
- Groq API — LLM-powered natural language explanations

**Tooling**
- python-dotenv — environment variable management
- Git — version control

---
## demo video link 
google drive link- 
- https://drive.google.com/drive/folders/1vuVZAHIyl63RnyqKWEiyHjASyeoyC3RJ


## screenshots
<img width="1919" height="960" alt="Screenshot 2026-06-20 202211" src="https://github.com/user-attachments/assets/99edc393-5df3-4bbf-8f25-17480b83b08d" />
<img width="1919" height="1019" alt="Screenshot 2026-06-20 210219" src="https://github.com/user-attachments/assets/21346b6a-44cb-4950-904a-9f1fc7991ce8" />
<img width="1919" height="995" alt="Screenshot 2026-06-20 210148" src="https://github.com/user-attachments/assets/8bd9d5bc-83ae-4ca0-a971-fbaf76991671" />
<img width="1913" height="1021" alt="Screenshot 2026-06-20 210117" src="https://github.com/user-attachments/assets/fa8b66e1-e6cc-46e5-9284-a94d79e648f7" />



## 📌 Use Cases

- Bank marketing teams prioritizing customer outreach
- Demonstrating an end-to-end ML + LLM pipeline
- Template for similar binary classification + explainability projects

---

## 📄 License

This project is open for learning and demonstration purposes. Feel free to fork, adapt, and build on it.
