import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier

##EDA

#load the dataset;
df = pd.read_csv(
    "bank-full.csv",
    sep=";"
)

df.head()

#shape
print(df.shape)

#column info
df.info()

#subscription distribution graph
print(df['y'].value_counts())
df['y'].value_counts().plot(kind='bar')
plt.show()

#percentage calculation
yes_percentage = (df['y'].value_counts()['yes'] / len(df))*100
print(yes_percentage)


# checking for null values: 
print(df.isnull().sum())

# subscription by job: 
print(pd.crosstab(
   df['job'],
   df['y']
))
job_rate = pd.crosstab(
    df['job'],
    df['y'],
    normalize='index'
)
print(job_rate['yes'].sort_values(ascending=False))


#balance
print(df.groupby('y')['balance'].mean())
plt.show()

##preprocessing

#seperate x and y
X = df.drop('y', axis=1)

y = df['y']

#encode 
encoder = LabelEncoder()

y = encoder.fit_transform(y)
#to check:
print(y[:10])

#train/test
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42,stratify=y)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


print("\nTraining target distribution:")
print(pd.Series(y_train).value_counts())


print("\nTesting target distribution:")
print(pd.Series(y_test).value_counts())

#categorial data
categorical_cols = X.select_dtypes(
    include=['object','string']
).columns


numerical_cols = X.select_dtypes(
    exclude=['object','string']
).columns


print("Categorical:")
print(categorical_cols)

print("Numerical:")
print(numerical_cols)


#preprocesssing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            'cat',
            OneHotEncoder(handle_unknown='ignore'),
            categorical_cols
        ),

        (
            'num',
            StandardScaler(),
            numerical_cols
        )
    ]
)
X_train_processed = preprocessor.fit_transform(X_train)

print(X_train_processed.shape)

# Logistic Regression Baseline Model

logistic_model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),

        ('classifier',
         LogisticRegression(
             max_iter=5000,
             class_weight='balanced',
             random_state=42
         ))
    ]
)

# Training
logistic_model.fit(
    X_train,
    y_train
)

# Prediction
log_pred = logistic_model.predict(
    X_test
)


# Probability prediction (useful later)
log_prob = logistic_model.predict_proba(
    X_test
)[:,1]


# Evaluation

print("===== Logistic Regression Results =====")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        log_pred
    )
)


print("Accuracy:",
      accuracy_score(
          y_test,
          log_pred
      ))


print("Precision:",
      precision_score(
          y_test,
          log_pred
      ))


print("Recall:",
      recall_score(
          y_test,
          log_pred
      ))


print("F1 Score:",
      f1_score(
          y_test,
          log_pred
      ))


print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        log_pred
    )
)

#random forest pipeline processor

rf_model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),

        ('classifier',
         RandomForestClassifier(
             n_estimators=50,
             random_state=42,
             class_weight='balanced'
         ))
    ]
)
#train
rf_model.fit(X_train,y_train)

#pridict
rf_pred = rf_model.predict( X_test)


#probability
rf_prob = rf_model.predict_proba( X_test)[:,1]

#evaluation

print("===== Random Forest Results =====")


print(
    classification_report(
        y_test,
        rf_pred
    )
)


print("Accuracy:",
      accuracy_score(
          y_test,
          rf_pred
      ))


print("Precision:",
      precision_score(
          y_test,
          rf_pred
      ))


print("Recall:",
      recall_score(
          y_test,
          rf_pred
      ))


print("F1 Score:",
      f1_score(
          y_test,
          rf_pred
      ))


#compare baseline model with random forest
results = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Random Forest"
        ],

        "Accuracy": [
            accuracy_score(y_test, log_pred),
            accuracy_score(y_test, rf_pred)
        ],

        "Precision": [
            precision_score(y_test, log_pred),
            precision_score(y_test, rf_pred)
        ],

        "Recall": [
            recall_score(y_test, log_pred),
            recall_score(y_test, rf_pred)
        ],

        "F1 Score": [
            f1_score(y_test, log_pred),
            f1_score(y_test, rf_pred)
        ]
    }
)

print(results)

# Predictions from Random Forest

rf_predictions = rf_model.predict(X_test)

rf_probabilities = rf_model.predict_proba(X_test)[:,1]

# Copy test data

sample_results = X_test.copy()


sample_results["Actual"] = y_test

sample_results["Prediction"] = rf_predictions

sample_results["Probability"] = rf_probabilities


sample_results.head()

# Pick 3 YES and 2 NO customers

yes_customers = sample_results[
    sample_results["Prediction"] == 1
].head(3)


no_customers = sample_results[
    sample_results["Prediction"] == 0
].head(2)


five_customers = pd.concat(
    [
        yes_customers,
        no_customers
    ]
)


print(five_customers)

for index, row in five_customers.iterrows():

    print("\n----------------------")
    print("Customer ID:", index)

    print("Age:", row["age"])
    print("Job:", row["job"])
    print("Balance:", row["balance"])
    print("Housing Loan:", row["housing"])
    print("Personal Loan:", row["loan"])

    if row["Prediction"] == 1:
        print("Prediction: YES (Will Subscribe)")
    else:
        print("Prediction: NO (Will Not Subscribe)")


    print(
        "Probability:",
        round(row["Probability"]*100,2),
        "%"
    )

# Get feature names after encoding

feature_names = rf_model.named_steps[
    'preprocessor'
].get_feature_names_out()


# Get importance from Random Forest

importances = rf_model.named_steps[
    'classifier'
].feature_importances_


# Create dataframe

feature_importance = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": importances
    }
)


# Sort highest importance first

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


print(feature_importance.head(10))

import pickle

pickle.dump(
    rf_model,
    open("customer_subscription_model.pkl","wb")
)