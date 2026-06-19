from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Load saved model
with open("customer_subscription_model.pkl", "rb") as file:
    model = pickle.load(file)


app = FastAPI(
    title="Bank Customer Subscription Prediction API"
)


# Customer input schema
class Customer(BaseModel):

    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str



# Health endpoint
@app.get("/health")
def health():

    return {
        "status": "ok",
        "model": "Logistic Regression"
    }



# Prediction endpoint
@app.post("/predict")
@app.post("/explain")
def explain(customer: Customer):

    data = pd.DataFrame(
        [customer.model_dump()]
    )


    probability = model.predict_proba(
        data
    )[0][1]


    prompt = f"""
Customer profile:

- Age: {customer.age}
- Job: {customer.job}
- Balance: {customer.balance}
- Existing loans:
Housing={customer.housing},
Personal={customer.loan}

Model prediction:
{round(probability*100,2)}% chance of subscribing


In 2-3 sentences, explain why this customer would or would not likely
subscribe to a term deposit, and how a relationship manager should
approach the conversation.
"""


    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ],

        temperature=0.3
    )


    explanation = response.choices[0].message.content


    return {

        "probability":
        round(float(probability),2),

        "explanation":
        explanation

    }@app.post("/explain")
def explain(customer: Customer):

    data = pd.DataFrame(
        [customer.model_dump()]
    )


    probability = model.predict_proba(
        data
    )[0][1]


    prompt = f"""
Customer profile:

- Age: {customer.age}
- Job: {customer.job}
- Balance: {customer.balance}
- Existing loans:
Housing={customer.housing},
Personal={customer.loan}

Model prediction:
{round(probability*100,2)}% chance of subscribing


In 2-3 sentences, explain why this customer would or would not likely
subscribe to a term deposit, and how a relationship manager should
approach the conversation.
"""


    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ],

        temperature=0.3
    )


    explanation = response.choices[0].message.content


    return {

        "probability":
        round(float(probability),2),

        "explanation":
        explanation

    }
def predict(customer: Customer):

    customer_data = pd.DataFrame(
        [customer.model_dump()]
    )


    prediction = model.predict(customer_data)[0]


    probability = model.predict_proba(
        customer_data
    )[0][1]


    return {

        "will_subscribe": bool(prediction),

        "probability": round(
            float(probability),
            2
        ),

        "top_factors": [
            "customer profile",
            "balance and financial details",
            "campaign history"
        ]
    }