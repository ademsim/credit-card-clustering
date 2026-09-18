import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Card Customer Segmentation", page_icon="💳")

with open("cluster_model.pkl", "rb") as f:
    model = pickle.load(f)

SEGMENTS = {
    0: ("Premium Spenders", "Spend the most, have the highest credit limits, and make many purchases."),
    1: ("Frequent Small Shoppers", "Buy often, but in small amounts, mostly with installments."),
    2: ("Cash Advance Users", "Take a lot of cash advances, buy little, and rarely pay in full."),
    3: ("Inactive Customers", "Rarely use the card and have low balances and limits."),
}

PRESETS = {
    "Enter manually": None,
    "Example: Premium Spender": (3551.2, 1.0, 7681.6, 5095.9, 2587.2, 653.6, 0.9, 0.7, 0.8, 0.1, 2, 89, 9696.9, 7288.7, 1970.5, 0.3, 12),
    "Example: Frequent Small Shopper": (894.8, 0.9, 1236.3, 594.0, 642.5, 209.8, 0.9, 0.3, 0.7, 0.0, 1, 22, 4214.0, 1331.3, 633.8, 0.3, 12),
    "Example: Cash Advance User": (4602.4, 1.0, 501.9, 320.2, 181.8, 4521.5, 0.3, 0.1, 0.2, 0.5, 14, 8, 7546.2, 3484.1, 2000.5, 0.0, 11),
    "Example: Inactive Customer": (1013.0, 0.8, 270.3, 210.1, 60.5, 597.2, 0.2, 0.1, 0.1, 0.1, 2, 3, 3278.3, 975.3, 535.6, 0.1, 11),
}

st.title("💳 Credit Card Customer Segmentation")
st.write("Enter a customer's credit card usage data and the model will assign them to a segment.")

choice = st.selectbox("Load an example (optional)", list(PRESETS.keys()))
d = PRESETS[choice] if PRESETS[choice] else (1000.0, 0.9, 500.0, 250.0, 250.0, 0.0, 0.5, 0.2, 0.3, 0.0, 0, 10, 3000.0, 1000.0, 500.0, 0.1, 12)

col1, col2 = st.columns(2)

with col1:
    balance = st.number_input("BALANCE", min_value=0.0, value=float(d[0]), step=100.0)
    balance_frequency = st.slider("BALANCE_FREQUENCY", 0.0, 1.0, float(d[1]), 0.1)
    purchases = st.number_input("PURCHASES", min_value=0.0, value=float(d[2]), step=100.0)
    oneoff_purchases = st.number_input("ONEOFF_PURCHASES", min_value=0.0, value=float(d[3]), step=100.0)
    installments_purchases = st.number_input("INSTALLMENTS_PURCHASES", min_value=0.0, value=float(d[4]), step=100.0)
    cash_advance = st.number_input("CASH_ADVANCE", min_value=0.0, value=float(d[5]), step=100.0)
    purchases_frequency = st.slider("PURCHASES_FREQUENCY", 0.0, 1.0, float(d[6]), 0.1)
    oneoff_purchases_frequency = st.slider("ONEOFF_PURCHASES_FREQUENCY", 0.0, 1.0, float(d[7]), 0.1)
    purchases_installments_frequency = st.slider("PURCHASES_INSTALLMENTS_FREQUENCY", 0.0, 1.0, float(d[8]), 0.1)

with col2:
    cash_advance_frequency = st.slider("CASH_ADVANCE_FREQUENCY", 0.0, 1.0, float(d[9]), 0.1)
    cash_advance_trx = st.number_input("CASH_ADVANCE_TRX", min_value=0, value=int(d[10]), step=1)
    purchases_trx = st.number_input("PURCHASES_TRX", min_value=0, value=int(d[11]), step=1)
    credit_limit = st.number_input("CREDIT_LIMIT", min_value=0.0, value=float(d[12]), step=500.0)
    payments = st.number_input("PAYMENTS", min_value=0.0, value=float(d[13]), step=100.0)
    minimum_payments = st.number_input("MINIMUM_PAYMENTS", min_value=0.0, value=float(d[14]), step=100.0)
    prc_full_payment = st.slider("PRC_FULL_PAYMENT", 0.0, 1.0, float(d[15]), 0.1)
    tenure = st.slider("TENURE (months)", 6, 12, int(d[16]))

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "BALANCE": balance,
        "BALANCE_FREQUENCY": balance_frequency,
        "PURCHASES": purchases,
        "ONEOFF_PURCHASES": oneoff_purchases,
        "INSTALLMENTS_PURCHASES": installments_purchases,
        "CASH_ADVANCE": cash_advance,
        "PURCHASES_FREQUENCY": purchases_frequency,
        "ONEOFF_PURCHASES_FREQUENCY": oneoff_purchases_frequency,
        "PURCHASES_INSTALLMENTS_FREQUENCY": purchases_installments_frequency,
        "CASH_ADVANCE_FREQUENCY": cash_advance_frequency,
        "CASH_ADVANCE_TRX": cash_advance_trx,
        "PURCHASES_TRX": purchases_trx,
        "CREDIT_LIMIT": credit_limit,
        "PAYMENTS": payments,
        "MINIMUM_PAYMENTS": minimum_payments,
        "PRC_FULL_PAYMENT": prc_full_payment,
        "TENURE": tenure,
    }])

    cluster = int(model.predict(input_df)[0])
    name, description = SEGMENTS[cluster]
    st.success(f"Segment: **{name}** (cluster {cluster})")
    st.write(description)
