import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="🛡️  Fraud Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR MODERN UI ---
st.markdown("""
    <style>
    .main { background-color: black; }
    .stMetric {
        background-color: blue;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        transition: all 0.2s ease;
    }
    h1, h2, h3 { color: #1e3a8a; font-family: 'Segoe UI', sans-serif; }
    .stSidebar { background-color: black; border-right: 1px solid #eaeaea; }
    div[data-testid="stExpander"] { border: none; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border-radius: 8px; background: white; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. LOAD DATA & MODEL ---
@st.cache_resource
def load_model():
    # Attempt to load model, return None if missing (Simulated mode)
    try:
        return joblib.load('fraud_xgb_best.pkl')
    except:
        return None

@st.cache_data
def load_data():
    try:
        # Generate synthetic data if file missing for demonstration
        # In production, replace with: pd.read_csv('your_data.csv')
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='H')
        data = {
            'Date': dates,
            'Transaction_Amount': np.random.exponential(scale=200, size=1000),
            'Account_Balance': np.random.uniform(1000, 50000, 1000),
            'Device_Type': np.random.choice(['Mobile', 'Desktop', 'Tablet'], 1000),
            'Location': np.random.choice(['NY', 'CA', 'TX', 'FL', 'IL'], 1000),
            'Fraud_Label': np.random.choice([0, 1], 1000, p=[0.95, 0.05]),
            'Card_Age': np.random.randint(1, 3000, 1000),
            'Previous_Fraud': np.random.choice([0, 1], 1000, p=[0.9, 0.1])
        }
        df = pd.DataFrame(data)
        df['Hour'] = df['Date'].dt.hour
        # Introduce some correlation for realism
        df.loc[df['Transaction_Amount'] > 800, 'Fraud_Label'] = 1
        return df
    except Exception as e:
        st.error(f"Error generating data: {e}")
        return pd.DataFrame()

model = load_model()
df = load_data()

# --- SIDEBAR FILTERS (GLOBAL) ---
st.sidebar.title("🛡️ Sentinel AI")
st.sidebar.caption("Real-time Fraud Intelligence")

# Global Date Filter
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()

with st.sidebar.expander("📅 Global Filters", expanded=True):
    date_range = st.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# Filter Data based on Global Selection
if len(date_range) == 2:
    mask = (df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])
    df_filtered = df.loc[mask]
else:
    df_filtered = df

# Navigation
menu = st.sidebar.radio("Dashboard View:", [
    "🏠 Executive Overview",
    "📊 Deep Dive Analysis",
    "🧠 Live Risk Simulator",
    "⚙️ System Health"
], index=0)

# --- DASHBOARD LOGIC ---

if menu == "🏠 Executive Overview":
    st.title("🏠 Executive Overview")
    st.markdown(f"**Data Range:** {date_range[0] if len(date_range)>0 else ''} to {date_range[1] if len(date_range)>1 else ''}")

    # KPIs
    total_txns = len(df_filtered)
    fraud_txns = df_filtered[df_filtered['Fraud_Label'] == 1]
    fraud_count = len(fraud_txns)
    fraud_rate = (fraud_count / total_txns * 100) if total_txns > 0 else 0
    blocked_amount = fraud_txns['Transaction_Amount'].sum()

    # Dynamic Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{total_txns:,}", delta=f"{np.random.randint(-50, 50)} vs avg")
    c2.metric("Fraud Detected", f"{fraud_count:,}", delta=f"{fraud_count/10:.1f}%", delta_color="inverse")
    c3.metric("Fraud Rate", f"{fraud_rate:.2f}%", delta=f"{0.1:.2f}%", delta_color="inverse")
    c4.metric("Prevented Loss", f"${blocked_amount:,.0f}", delta="Saved", delta_color="normal")

    st.markdown("---")

    # Interactive Charts Row 1
    c_left, c_right = st.columns([2, 1])

    with c_left:
        st.subheader("📈 Fraud Velocity (Timeline)")
        daily_fraud = df_filtered[df_filtered['Fraud_Label']==1].groupby(df_filtered['Date'].dt.date).size().reset_index(name='Attacks')
        fig_line = px.area(daily_fraud, x='Date', y='Attacks',
                           title="Daily Fraud Attempts",
                           color_discrete_sequence=['#FF4B4B'])
        fig_line.update_layout(xaxis_title="", yaxis_title="Number of Attacks", hovermode="x unified")
        st.plotly_chart(fig_line, use_container_width=True)

    with c_right:
        st.subheader("⚠️ Risk by Device")
        dev_counts = df_filtered[df_filtered['Fraud_Label']==1]['Device_Type'].value_counts()

        # --- FIX: Changed px.donut to px.pie with hole parameter ---
        fig_pie = px.pie(dev_counts, values=dev_counts.values, names=dev_counts.index,
                           title="Fraud Source Distribution", hole=0.6,
                           color_discrete_sequence=px.colors.sequential.RdBu)
        # -----------------------------------------------------------

        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

elif menu == "📊 Deep Dive Analysis":
    st.title("📊 Deep Dive Analysis")

    # Advanced Filters in Main Area
    with st.expander("🔍 Advanced Filtering", expanded=False):
        c1, c2, c3 = st.columns(3)
        amt_filter = c1.slider("Min Transaction Amount ($)", 0, 2000, 0)
        loc_filter = c2.multiselect("Locations", df['Location'].unique(), default=df['Location'].unique())
        fraud_only = c3.checkbox("Show Only Fraud", value=True)

    # Apply Local Filters
    view_df = df_filtered[
        (df_filtered['Transaction_Amount'] >= amt_filter) &
        (df_filtered['Location'].isin(loc_filter))
    ]
    if fraud_only:
        view_df = view_df[view_df['Fraud_Label'] == 1]

    # Split View: Chart & Data
    col_chart, col_data = st.columns([1, 1])

    with col_chart:
        st.subheader("📍 High-Risk Geographies")
        loc_stats = view_df.groupby('Location')['Transaction_Amount'].sum().reset_index()
        fig_map = px.bar(loc_stats, x='Location', y='Transaction_Amount',
                         color='Transaction_Amount',
                         title="Total Loss Exposure by Location",
                         color_continuous_scale='Reds')
        st.plotly_chart(fig_map, use_container_width=True)

    with col_data:
        st.subheader("📝 Transaction Ledger")
        st.caption(f"Showing {len(view_df)} flagged transactions")

        # Interactive Table with formatting
        st.dataframe(
            view_df[['Date', 'Transaction_Amount', 'Location', 'Device_Type', 'Fraud_Label']],
            column_config={
                "Transaction_Amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
                "Fraud_Label": st.column_config.CheckboxColumn("Fraud Status"),
            },
            use_container_width=True,
            height=300
        )

    # Scatter Plot for Pattern Recognition
    st.subheader("🔍 Pattern Recognition: Amount vs. Hour")
    fig_scatter = px.scatter(
        view_df, x='Hour', y='Transaction_Amount',
        color='Device_Type', size='Transaction_Amount',
        hover_data=['Location'],
        title="Transaction Clusters (Size = Amount)",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

elif menu == "🧠 Live Risk Simulator":
    st.title("🧠 Live Risk Simulator")
    st.markdown("Test the model in real-time. Adjust parameters to see how the risk score changes.")

    # Layout: Input Panel (Left) & Result Panel (Right)
    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.subheader("Transaction Parameters")
        with st.container(border=True):
            amt = st.slider("Transaction Amount ($)", 0, 5000, 500, help="Higher amounts increase risk.")
            bal = st.number_input("Account Balance ($)", 0, 100000, 15000)
            age = st.slider("Account Age (Days)", 0, 3000, 365)

            c_sub1, c_sub2 = st.columns(2)
            daily_cnt = c_sub1.number_input("Daily Txn Count", 0, 50, 3)
            prev_fraud = c_sub2.selectbox("History of Fraud?", ["No", "Yes"])

            # Button to Trigger Prediction
            analyze_btn = st.button("🛡️ Analyze Risk Level", type="primary", use_container_width=True)

    with col_result:
        st.subheader("Risk Assessment")

        if analyze_btn:
            with st.spinner("Running Neural Scan..."):
                time.sleep(0.8) # UX delay for effect

                # Logic
                pf_val = 1 if prev_fraud == "Yes" else 0

                # Model Prediction (or Heuristic Fallback)
                risk_prob = 0.0
                if model:
                    try:
                        input_vector = np.array([[amt, bal, age, daily_cnt, pf_val]])
                        risk_prob = model.predict_proba(input_vector)[0][1]
                    except:
                        # Fallback calculation if model input shape differs
                        risk_prob = (amt / 5000) * 0.5 + (0.4 if pf_val else 0)
                else:
                    # Pure Heuristic for Demo
                    base_risk = 0.1
                    if amt > 1000: base_risk += 0.3
                    if daily_cnt > 10: base_risk += 0.2
                    if pf_val: base_risk += 0.3
                    risk_prob = min(base_risk, 0.99)

                # Visualization Gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk_prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Fraud Probability (%)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "#d4edda"},
                            {'range': [30, 70], 'color': "#fff3cd"},
                            {'range': [70, 100], 'color': "#f8d7da"}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 85}
                    }
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Dynamic Message
                if risk_prob > 0.7:
                    st.error(f"🚨 **CRITICAL ALERT:** High probability of fraud detected ({risk_prob:.1%}). Immediate verification required.")
                elif risk_prob > 0.3:
                    st.warning(f"⚠️ **WARNING:** Moderate risk detected ({risk_prob:.1%}). Monitor closely.")
                else:
                    st.success(f"✅ **SAFE:** Transaction appears legitimate ({risk_prob:.1%}).")

elif menu == "⚙️ System Health":
    st.title("⚙️ System Diagnostics")

    c1, c2, c3 = st.columns(3)
    c1.info(f"**Model Status:** {'Active ✅' if model else 'Demo Mode ⚠️'}")
    c2.info(f"**Last Data Refresh:** {pd.Timestamp.now().strftime('%H:%M:%S')}")
    c3.info("**API Latency:** 45ms")

    st.subheader("Model Feature Importance")
    # Dummy importance for demo
    imp_df = pd.DataFrame({
        'Feature': ['Amount', 'Frequency', 'Location Risk', 'Device Type', 'Account Age'],
        'Importance': [0.35, 0.25, 0.20, 0.15, 0.05]
    })
    fig_imp = px.bar(imp_df, y='Feature', x='Importance', orientation='h',
                     title="Impact on Fraud Scoring", color='Importance', color_continuous_scale='Blues')
    st.plotly_chart(fig_imp, use_container_width=True)
