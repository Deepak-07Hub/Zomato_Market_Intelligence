import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


@st.cache_data
def load_dataset():
    candidates = [
        "zomato.csv",
        "Zomato Restaurant Dataset.csv",
        "./zomato.csv",
        "./Zomato Restaurant Dataset.csv",
    ]

    for path in candidates:
        try:
            df = pd.read_csv(path)
            return df
        except FileNotFoundError:
            continue

    raise FileNotFoundError("Could not find the Zomato dataset in the project folder.")


@st.cache_resource
def train_best_model():
    df = load_dataset()
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates().copy()

    for col in ["Restaurant Name", "City", "Address", "Locality", "Locality Verbose", "Cuisines", "Currency"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    for col in ["Aggregate rating", "Average Cost for two", "Votes", "Price range"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Aggregate rating" in df.columns:
        df["Aggregate rating"] = df["Aggregate rating"].replace(0, np.nan)

    if "Cuisines" in df.columns:
        df["Cuisines"] = df["Cuisines"].fillna("Unknown").astype(str).str.replace(r"\s+", " ", regex=True)
        df.loc[df["Cuisines"].str.strip().eq(""), "Cuisines"] = "Unknown"

    for col in ["Has Table booking", "Has Online delivery", "Is delivering now", "Switch to order menu"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
            df[col] = df[col].replace({"Yes": "Yes", "No": "No", "Y": "Yes", "N": "No", "Nan": "No"})

    if "Average Cost for two" in df.columns:
        df["Average Cost for two"] = pd.to_numeric(df["Average Cost for two"], errors="coerce")
        df["Average Cost for two"] = df["Average Cost for two"].fillna(df["Average Cost for two"].median())

    if "Cuisines" in df.columns:
        df["Cuisines"] = df["Cuisines"].str.replace(" ,", ",", regex=False).str.replace(", ", ",", regex=False)

    df_model = df.copy()
    df_model["High_Rated"] = (df_model["Aggregate rating"] >= 4.0).astype(int)

    if "Cuisines" in df_model.columns:
        df_model["cuisine_count"] = df_model["Cuisines"].fillna("Unknown").str.split(",").str.len()
        df_model["cuisine_frequency"] = df_model["Cuisines"].map(df_model["Cuisines"].value_counts())

    if "City" in df_model.columns:
        df_model["location_frequency"] = df_model["City"].map(df_model["City"].value_counts())

    if "Average Cost for two" in df_model.columns:
        q1, q2 = np.quantile(df_model["Average Cost for two"].dropna(), [0.33, 0.66])
        df_model["price_category"] = df_model["Average Cost for two"].apply(
            lambda x: "Low" if x <= q1 else "Medium" if x <= q2 else "High"
        ).astype(str)

    if "Votes" in df_model.columns:
        df_model["votes_log"] = np.log1p(df_model["Votes"].fillna(0))

    if "Has Online delivery" in df_model.columns:
        df_model["has_online_order"] = df_model["Has Online delivery"].map({"Yes": 1, "No": 0, "nan": 0})

    if "Has Table booking" in df_model.columns:
        df_model["has_table_booking"] = df_model["Has Table booking"].map({"Yes": 1, "No": 0, "nan": 0})

    supported_feature_cols = [
        "Average Cost for two",
        "Votes",
        "votes_log",
        "cuisine_count",
        "cuisine_frequency",
        "location_frequency",
        "has_online_order",
        "has_table_booking",
        "price_category",
        "City",
    ]
    feature_columns = [col for col in supported_feature_cols if col in df_model.columns]
    target_col = "High_Rated"

    df_model = df_model.dropna(subset=[target_col] + [col for col in feature_columns if col not in ["City", "price_category"]])

    X = df_model[feature_columns].copy()
    y = df_model[target_col].copy()

    for col in X.columns:
        if X[col].isna().any():
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna("Unknown")

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=8, min_samples_leaf=10),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", min_samples_leaf=5),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    model_results = []
    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
        }
        model_results.append(metrics)

    model_results = pd.DataFrame(model_results).sort_values(["F1", "ROC-AUC", "Accuracy"], ascending=False).reset_index(drop=True)
    best_model_name = model_results.iloc[0]["Model"]
    best_model = models[best_model_name]
    best_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", best_model)])
    best_pipeline.fit(X_train, y_train)

    return {
        "df_model": df_model,
        "feature_columns": feature_columns,
        "best_model_name": best_model_name,
        "best_pipeline": best_pipeline,
        "model_results": model_results,
    }


def get_price_category(value, df_model):
    q1, q2 = np.quantile(df_model["Average Cost for two"].dropna(), [0.33, 0.66])
    if value <= q1:
        return "Low"
    if value <= q2:
        return "Medium"
    return "High"


def build_prediction_row(city, cuisine, avg_cost, votes, online_order, table_booking, model_data):
    df_model = model_data["df_model"]
    cuisine_counts = (
        df_model["Cuisines"]
        .fillna("Unknown")
        .str.split(",")
        .explode()
        .str.strip()
        .str.title()
        .value_counts()
        .to_dict()
    )
    city_counts = df_model["City"].value_counts().to_dict()

    sample = pd.DataFrame({
        "City": [city],
        "Cuisines": [cuisine],
        "Average Cost for two": [avg_cost],
        "Votes": [votes],
        "Has Online delivery": [online_order],
        "Has Table booking": [table_booking],
        "votes_log": [np.log1p(votes)],
        "cuisine_count": [1],
        "cuisine_frequency": [cuisine_counts.get(cuisine.strip().title(), 1)],
        "location_frequency": [city_counts.get(city, 1)],
        "has_online_order": [1 if online_order == "Yes" else 0],
        "has_table_booking": [1 if table_booking == "Yes" else 0],
        "price_category": [get_price_category(avg_cost, df_model)],
    })
    return sample


st.set_page_config(page_title="Zomato Rating Predictor", page_icon="🍽️", layout="wide")
st.title("Zomato Restaurant Rating Prediction & Market Intelligence")
st.caption("Interactive business intelligence and restaurant rating prediction using the best model from the notebook.")

model_data = train_best_model()
df_model = model_data["df_model"]
feature_columns = model_data["feature_columns"]
best_model_name = model_data["best_model_name"]
best_pipeline = model_data["best_pipeline"]
model_results = model_data["model_results"]

left_col, right_col = st.columns([1.4, 1])

with left_col:
    with st.form("restaurant_prediction_form"):
        city = st.selectbox("City", sorted(df_model["City"].dropna().unique().tolist()))
        cuisine = st.selectbox(
            "Cuisine",
            sorted(
                df_model["Cuisines"]
                .fillna("Unknown")
                .str.split(",")
                .explode()
                .str.strip()
                .str.title()
                .unique()
                .tolist()
            ),
        )
        avg_cost = st.number_input("Average Cost for Two", min_value=0, max_value=500000, value=500, step=10)
        votes = st.number_input("Votes", min_value=0, max_value=100000, value=100, step=10)
        online_order = st.selectbox("Has Online Delivery", ["Yes", "No"])
        table_booking = st.selectbox("Has Table Booking", ["Yes", "No"])
        submitted = st.form_submit_button("Predict High Rating")

    if submitted:
        sample = build_prediction_row(city, cuisine, avg_cost, votes, online_order, table_booking, model_data)
        sample = sample[feature_columns]
        prob = best_pipeline.predict_proba(sample)[0, 1]
        label = "High Rated" if prob >= 0.5 else "Not High Rated"

        st.subheader("Prediction Result")
        if label == "High Rated":
            st.success(f"Prediction: {label}")
        else:
            st.warning(f"Prediction: {label}")

        st.metric("Probability of High Rating", f"{prob * 100:.1f}%")
        st.write(f"Model used: {best_model_name}")

with right_col:
    st.subheader("Model comparison")
    st.dataframe(model_results, use_container_width=True)

    st.subheader("Market overview")
    st.write(f"Restaurants analyzed: {df_model.shape[0]}")
    st.write(f"Cities covered: {df_model['City'].nunique()}")
    st.write(f"Average rating: {df_model['Aggregate rating'].mean():.2f}")
    st.write(f"Average cost for two: {df_model['Average Cost for two'].mean():.2f}")

    st.subheader("Selected model")
    st.info(best_model_name)


st.markdown("---")

with st.expander("Business insights from the dataset"):
    avg_rating_by_city = df_model.groupby("City")["Aggregate rating"].mean().sort_values(ascending=False).head(10)
    avg_rating_by_cuisine = (
        df_model.assign(cuisine_split=df_model["Cuisines"].str.split(","))
        .explode("cuisine_split")
        .assign(cuisine_split=lambda d: d["cuisine_split"].str.strip().str.title())
        .groupby("cuisine_split")["Aggregate rating"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    st.write("Top cities by average rating:")
    st.dataframe(avg_rating_by_city.reset_index().rename(columns={"index": "City", "Aggregate rating": "Average Rating"}), use_container_width=True)

    st.write("Top cuisines by average rating:")
    st.dataframe(avg_rating_by_cuisine.reset_index().rename(columns={"index": "Cuisine", "Aggregate rating": "Average Rating"}), use_container_width=True)
