"""
app.py

Streamlit app that:
 - connects to MongoDB (secrets.toml)
 - lets you upload a CSV → cleans it → inserts into MongoDB
 - creates dashboards:
     Overview KPIs + filters
     Customer Explorer
     Churn & Risk Analysis
     Support Issue Insights

Run:
  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Customer Revenue & Churn Intelligence", layout="wide")

# -------------------------------------------------------------------
# MONGO CONNECTION
# -------------------------------------------------------------------
@st.cache_resource(ttl=3600)
def get_mongo_client():
    uri = st.secrets["mongo"]["uri"]
    return MongoClient(uri)

@st.cache_data(ttl=300)
def load_data_from_mongo(limit=None):
    client = get_mongo_client()
    db = client[st.secrets["mongo"]["db"]]
    coll = db[st.secrets["mongo"]["coll"]]

    cursor = coll.find({})
    if limit:
        cursor = cursor.limit(limit)
    df = pd.DataFrame(list(cursor))
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])
    return df

# -------------------------------------------------------------------
# DATA CLEANING / FEATURES
# -------------------------------------------------------------------
def clean_column_names(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df

def coerce_and_feature(df):
    df = df.copy()

    # Convert dates
    for c in ["signup_date","last_login_date","order_date","delivery_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Convert numeric
    for c in [
        "quantity","unit_price","price","discount_pct","discount_percent",
        "rating","age","session_duration_sec","pages_viewed"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Compute total_revenue
    if "total_revenue" not in df.columns:
        if {"quantity","unit_price"}.issubset(df.columns):
            if "discount_pct" in df.columns:
                d = df["discount_pct"].fillna(0)
            elif "discount_percent" in df.columns:
                d = df["discount_percent"].fillna(0)
            else:
                d = 0
            df["total_revenue"] = df["quantity"].fillna(0) * df["unit_price"].fillna(0) * (1 - d/100)
        else:
            # Try finding any total-like column
            candidates = [c for c in df.columns if "amount" in c or "total" in c]
            if candidates:
                df["total_revenue"] = pd.to_numeric(df[candidates[0]], errors="coerce")
            else:
                df["total_revenue"] = np.nan

    # Age groups
    if "age" in df.columns:
        bins = [0,17,25,35,50,120]
        labels = ["<18","18-25","26-35","36-50","50+"]
        df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)
        df["age_group"] = df["age_group"].cat.add_categories(["Unknown"]).fillna("Unknown")
    else:
        df["age_group"] = "Unknown"

    # Basic string cleanup
    if "review_text" in df.columns:
        df["review_text"] = df["review_text"].astype(str).str.replace(r"\s+"," ", regex=True)
        df["review_text"] = df["review_text"].str.replace(r"\S+@\S+\.\S+","[redacted_email]", regex=True)

    return df

def aggregate_customers(df):
    if df.empty:
        return pd.DataFrame()

    def mode_or_nan(s):
        s = s.dropna()
        if s.empty:
            return np.nan
        try:
            return s.mode().iloc[0]
        except:
            return s.iloc[0]

    cust = df.groupby("customer_id", dropna=False).agg(
        transactions=("transaction_id","nunique"),
        lifetime_revenue=("total_revenue","sum"),
        avg_order_value=("total_revenue","mean"),
        avg_rating=("rating","mean"),
        last_active=("last_login_date","max"),
        first_signup=("signup_date","min"),
        city=("city", lambda x: mode_or_nan(x)),
        age_group=("age_group", lambda x: mode_or_nan(x))
    ).reset_index()

    cust["lifetime_revenue"] = cust["lifetime_revenue"].fillna(0)
    cust["avg_order_value"] = cust["avg_order_value"].fillna(0)

    return cust

# -------------------------------------------------------------------
# UPLOAD → MONGO
# -------------------------------------------------------------------
def upload_to_mongo(file_bytes):
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    except:
        df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin1", low_memory=False)

    df = clean_column_names(df)
    df = coerce_and_feature(df)

    # Convert datetime to Python datetime
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].apply(lambda x: x.to_pydatetime() if pd.notna(x) else None)

    records = df.where(pd.notnull(df), None).to_dict(orient="records")

    client = get_mongo_client()
    db = client[st.secrets["mongo"]["db"]]
    coll = db[st.secrets["mongo"]["coll"]]

    result = coll.insert_many(records)
    return len(result.inserted_ids)

# -------------------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------------------
st.title("Customer Revenue & Churn Intelligence Dashboard")
st.write("MongoDB-backed analytics app.")

# ---- SIDEBAR ----
st.sidebar.header("Data Controls")

uploaded = st.sidebar.file_uploader("Upload CSV to MongoDB", type=["csv"])

if uploaded and st.sidebar.button("Upload Now"):
    count = upload_to_mongo(uploaded.read())
    st.sidebar.success(f"Uploaded {count} records to Mongo!")
    load_data_from_mongo.clear()  # refresh cache

limit = st.sidebar.number_input("Max rows to load", 100, 200000, 50000)
churn_days = st.sidebar.slider("Churn window (days)", 7, 365, 60)

segmentation = st.sidebar.selectbox(
    "Segment by",
    ["age_group","city","category","marketing_source","channel","gender"],
)

page = st.sidebar.radio(
    "Navigate",
    ["Overview","Customer Explorer","Churn & Risk Analysis","Support Issues"]
)

# ---- LOAD DATA ----
df = load_data_from_mongo(limit=limit)
if df.empty:
    st.warning("No data in MongoDB yet.")
    st.stop()

df = clean_column_names(df)
df = coerce_and_feature(df)

df_customer = aggregate_customers(df)

# Churn flags
ref_date = df["last_login_date"].max() if "last_login_date" in df.columns else datetime.now()
df_customer["days_since_active"] = (ref_date - df_customer["last_active"]).dt.days
df_customer["is_churned"] = df_customer["days_since_active"] > churn_days

# ---- KPIs ----
total_revenue = df["total_revenue"].sum()
avg_clv = df_customer["lifetime_revenue"].mean()
churn_rate = df_customer["is_churned"].mean() * 100

st.sidebar.markdown("### KPIs")
st.sidebar.metric("Total Revenue", f"₹{total_revenue:,.0f}")
st.sidebar.metric("Avg CLV", f"₹{avg_clv:,.0f}")
st.sidebar.metric("Churn Rate", f"{churn_rate:.1f}%")

# -------------------------------------------------------------------
# OVERVIEW PAGE
# -------------------------------------------------------------------
if page == "Overview":
    st.header("Overview Dashboard")
    
    # Filters
    c1, c2 = st.columns(2)
    city_filter = None
    seg_filter = None

    if "city" in df.columns:
        city_filter = c1.multiselect("Filter by City", ["All"] + sorted(df["city"].dropna().unique().tolist()), default=["All"])

    if segmentation in df.columns:
        opts = ["All"] + sorted(df[segmentation].dropna().astype(str).unique().tolist())
        seg_filter = c2.selectbox(f"Filter by {segmentation}", opts)

    filtered = df.copy()
    if city_filter and "All" not in city_filter:
        filtered = filtered[filtered["city"].isin(city_filter)]

    if seg_filter and seg_filter != "All":
        filtered = filtered[filtered[segmentation].astype(str) == seg_filter]

    st.subheader("Revenue Trend")
    if {"order_date","total_revenue"}.issubset(filtered.columns):
        rev_ts = filtered.groupby("order_date")["total_revenue"].sum().reset_index()
        st.line_chart(rev_ts.set_index("order_date"))

    st.subheader("Top Categories by Revenue")
    if "category" in filtered.columns:
        cat = filtered.groupby("category")["total_revenue"].sum().sort_values(ascending=False).head(10)
        st.bar_chart(cat)

# -------------------------------------------------------------------
# CUSTOMER EXPLORER
# -------------------------------------------------------------------
elif page == "Customer Explorer":
    st.header("Customer Explorer")

    q = st.text_input("Search customer_id or review text")
    result = df.copy()
    if q:
        q = q.lower()
        mask = (
            df["customer_id"].astype(str).str.contains(q, case=False, na=False)
            | df.get("review_text","").astype(str).str.contains(q, case=False, na=False)
        )
        result = df[mask]

    st.write(f"Results: {len(result)} rows")
    st.dataframe(result.head(200))

    st.subheader("Select a Customer")
    cust_list = df_customer["customer_id"].dropna().astype(str).unique().tolist()
    selected = st.selectbox("Customer ID", [""] + cust_list)

    if selected:
        tx = df[df["customer_id"].astype(str) == selected].sort_values("order_date")
        st.write("Transactions")
        st.dataframe(tx)

        customer_row = df_customer[df_customer["customer_id"].astype(str) == selected].iloc[0]
        st.metric("Total Spent", f"₹{customer_row['lifetime_revenue']:,.0f}")
        st.metric("Avg Order Value", f"₹{customer_row['avg_order_value']:,.0f}")
        st.metric("Churned", "Yes" if customer_row["is_churned"] else "No")

# -------------------------------------------------------------------
# CHURN & RISK PAGE
# -------------------------------------------------------------------
elif page == "Churn & Risk Analysis":
    st.header("Churn & Risk Analysis")

    st.subheader(f"Churn by {segmentation}")
    if segmentation in df_customer.columns:
        seg = df_customer.groupby(segmentation)["is_churned"].mean()*100
        st.bar_chart(seg)

    st.subheader("High Risk Segments: High CLV + High Churn")
    if segmentation in df_customer.columns:
        seg = df_customer.groupby(segmentation).agg(
            avg_clv=("lifetime_revenue","mean"),
            churn_rate=("is_churned","mean")
        )
        st.dataframe(seg.sort_values("churn_rate", ascending=False))

# -------------------------------------------------------------------
# SUPPORT ISSUES PAGE
# -------------------------------------------------------------------
else:
    st.header("Support Issue Insights")

    if "is_returned" in df.columns:
        st.subheader("Return Rate by Category")
        rr = df.groupby("category")["is_returned"].mean()*100
        st.bar_chart(rr)

    st.subheader("Negative Reviews (rating <= 2)")
    if {"rating","review_text"}.issubset(df.columns):
        bad = df[df["rating"] <= 2][["customer_id","order_date","rating","review_text"]].sort_values("order_date", ascending=False)
        st.dataframe(bad.head(50))

# -------------------------------------------------------------------
# END
# -------------------------------------------------------------------
