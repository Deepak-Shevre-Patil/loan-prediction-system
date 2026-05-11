# Step 1 — Import Libraries

import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF


# Step 2 — Load Trained Model and Scaler

model = joblib.load("models/best_model.pkl")

scaler = joblib.load("models/scaler.pkl")


# Step 3 — App Title

st.title("Loan Approval Prediction System")

st.write(
    "Machine Learning Model for Loan Approval Prediction"
)


# Step 4 — Sidebar Navigation

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Home",
        "Prediction",
        "Dataset Statistics",
        "Model Performance",
        "About Project"
    ]
)


# =========================
# HOME PAGE
# =========================

if page == "Home":

    st.header("Welcome to Loan Approval Prediction System")

    st.write(
        "This project uses Machine Learning and Explainable AI "
        "to predict whether a loan should be approved or rejected."
    )

    st.image(
        "https://images.unsplash.com/photo-1554224155-6726b3ff858f",
        use_container_width=True
    )


# =========================
# PREDICTION PAGE
# =========================

if page == "Prediction":

    # Step 5 — User Input Fields

    no_of_dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=10,
        value=0
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"]
    )

    income_annum = st.number_input(
        "Annual Income (in Lakhs ₹)",
        min_value=0,
        value=0
    )

    loan_amount = st.number_input(
        "Loan Amount (in Lakhs ₹)",
        min_value=0,
        value=0
    )

    loan_term = st.number_input(
        "Loan Term (in Months)",
        min_value=1,
        value=1
    )

    cibil_score = st.number_input(
        "CIBIL Score (Range: 300 to 900)",
        min_value=300,
        max_value=900,
        value=300
    )

    residential_assets_value = st.number_input(
        "Residential Assets Value (in Lakhs ₹)",
        min_value=0,
        value=0
    )

    commercial_assets_value = st.number_input(
        "Commercial Assets Value (in Lakhs ₹)",
        min_value=0,
        value=0
    )

    luxury_assets_value = st.number_input(
        "Luxury Assets Value (in Lakhs ₹)",
        min_value=0,
        value=0
    )

    bank_asset_value = st.number_input(
        "Bank Asset Value (in Lakhs ₹)",
        min_value=0,
        value=0
    )


    # Step 6 — Encode Categorical Variables

    education = 0 if education == "Graduate" else 1

    self_employed = 1 if self_employed == "Yes" else 0


    # Step 7 — Feature Engineering

    loan_income_ratio = (
        loan_amount / income_annum
        if income_annum != 0 else 0
    )

    total_assets = (
        residential_assets_value +
        commercial_assets_value +
        luxury_assets_value +
        bank_asset_value
    )

    emi_risk = (
        loan_amount / loan_term
        if loan_term != 0 else 0
    )


    # Step 8 — Create Input DataFrame

    input_data = pd.DataFrame([[
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value,
        loan_income_ratio,
        total_assets,
        emi_risk
    ]], columns=[
        'no_of_dependents',
        'education',
        'self_employed',
        'income_annum',
        'loan_amount',
        'loan_term',
        'cibil_score',
        'residential_assets_value',
        'commercial_assets_value',
        'luxury_assets_value',
        'bank_asset_value',
        'loan_income_ratio',
        'total_assets',
        'emi_risk'
    ])


    # Step 9 — Scale Input Data

    input_scaled = scaler.transform(input_data)


    # =========================
    # FINANCIAL DASHBOARD
    # =========================

    st.subheader("Financial Dashboard")


    # Row 1

    col1, col2 = st.columns(2)

    with col1:

        fig_income = go.Figure()

        fig_income.add_trace(go.Bar(
            x=["Annual Income", "Loan Amount"],
            y=[income_annum, loan_amount],
            text=[income_annum, loan_amount],
            textposition='auto'
        ))

        fig_income.update_layout(
            title="Income vs Loan Amount",
            height=400
        )

        st.plotly_chart(
            fig_income,
            use_container_width=True
        )


    with col2:

        fig_assets = go.Figure(data=[go.Pie(
            labels=[
                "Residential",
                "Commercial",
                "Luxury",
                "Bank"
            ],
            values=[
                residential_assets_value,
                commercial_assets_value,
                luxury_assets_value,
                bank_asset_value
            ]
        )])

        fig_assets.update_layout(
            title="Asset Distribution",
            height=400
        )

        st.plotly_chart(
            fig_assets,
            use_container_width=True
        )


    # Row 2

    col3, col4, col5 = st.columns(3)


    with col3:

        fig_cibil = go.Figure(go.Indicator(
            mode="gauge+number",
            value=cibil_score,
            title={'text': "CIBIL Score"},
            gauge={
                'axis': {'range': [300, 900]},
                'steps': [
                    {'range': [300, 650], 'color': "red"},
                    {'range': [650, 750], 'color': "yellow"},
                    {'range': [750, 900], 'color': "green"}
                ]
            }
        ))

        fig_cibil.update_layout(height=350)

        st.plotly_chart(
            fig_cibil,
            use_container_width=True
        )


    with col4:

        fig_emi = go.Figure(go.Indicator(
            mode="gauge+number",
            value=emi_risk,
            title={'text': "EMI Risk"},
            gauge={
                'axis': {'range': [0, 500000]},
                'steps': [
                    {'range': [0, 150000], 'color': "green"},
                    {'range': [150000, 300000], 'color': "yellow"},
                    {'range': [300000, 500000], 'color': "red"}
                ]
            }
        ))

        fig_emi.update_layout(height=350)

        st.plotly_chart(
            fig_emi,
            use_container_width=True
        )


    with col5:

        fig_ratio = go.Figure(go.Indicator(
            mode="gauge+number",
            value=loan_income_ratio,
            title={'text': "Loan-Income Ratio"},
            gauge={
                'axis': {'range': [0, 1]},
                'steps': [
                    {'range': [0, 0.4], 'color': "green"},
                    {'range': [0.4, 0.7], 'color': "yellow"},
                    {'range': [0.7, 1], 'color': "red"}
                ]
            }
        ))

        fig_ratio.update_layout(height=350)

        st.plotly_chart(
            fig_ratio,
            use_container_width=True
        )


    # =========================
    # PREDICTION SECTION
    # =========================

    if st.button("Predict Loan Status"):

        prediction = model.predict(input_scaled)

        probability = model.predict_proba(input_scaled)

        approval_prob = probability[0][0] * 100
        rejection_prob = probability[0][1] * 100


        st.subheader("Prediction Probability")

        st.write(f"Approval Chance: {approval_prob:.2f}%")

        st.write(f"Rejection Chance: {rejection_prob:.2f}%")


        st.subheader("Prediction Result")


        # Loan Approved

        if prediction[0] == 0:

            st.success("Loan Approved")

            reasons = []

            if cibil_score >= 700:
                reasons.append("✔ Good CIBIL Score")

            if loan_income_ratio < 0.5:
                reasons.append("✔ Healthy Loan-to-Income Ratio")

            if total_assets > loan_amount:
                reasons.append("✔ Strong Asset Value")

            if emi_risk < 200000:
                reasons.append("✔ EMI Risk is Low")

            st.subheader("Why Approved?")

            for reason in reasons:
                st.write(reason)


        # Loan Rejected

        else:

            st.error("Loan Rejected")

            reasons = []

            if cibil_score < 650:
                reasons.append("❌ Low CIBIL Score")

            if loan_income_ratio > 0.6:
                reasons.append("❌ High Loan-to-Income Ratio")

            if total_assets < loan_amount:
                reasons.append("❌ Low Asset Coverage")

            if emi_risk > 300000:
                reasons.append("❌ High EMI Risk")

            if income_annum < loan_amount:
                reasons.append("❌ Income is Lower than Loan Amount")

            st.subheader("Why Rejected?")

            for reason in reasons:
                st.write(reason)


        # =========================
        # PDF REPORT DOWNLOAD
        # =========================

        pdf = FPDF()

        pdf.add_page()

        pdf.set_font("Arial", size=12)

        pdf.cell(
            200,
            10,
            txt="Loan Prediction Report",
            ln=True
        )

        pdf.cell(
            200,
            10,
            txt=f"Annual Income: {income_annum}",
            ln=True
        )

        pdf.cell(
            200,
            10,
            txt=f"Loan Amount: {loan_amount}",
            ln=True
        )

        pdf.cell(
            200,
            10,
            txt=f"CIBIL Score: {cibil_score}",
            ln=True
        )

        if prediction[0] == 0:
            result_text = "Loan Approved"
        else:
            result_text = "Loan Rejected"

        pdf.cell(
            200,
            10,
            txt=f"Prediction: {result_text}",
            ln=True
        )

        pdf.cell(
            200,
            10,
            txt=f"Approval Chance: {approval_prob:.2f}%",
            ln=True
        )

        pdf.output("loan_report.pdf")

        with open("loan_report.pdf", "rb") as file:

            st.download_button(
                label="Download PDF Report",
                data=file,
                file_name="loan_report.pdf",
                mime="application/pdf"
            )


# =========================
# DATASET STATISTICS PAGE
# =========================

if page == "Dataset Statistics":

    st.header("Dataset Statistics")

    total_applicants = 4269

    approved = 1613

    rejected = 2656

    approval_percentage = (
        approved / total_applicants
    ) * 100

    rejection_percentage = (
        rejected / total_applicants
    ) * 100

    avg_income = 5059123

    avg_loan = 1513340


    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Applicants",
            total_applicants
        )

    with col2:
        st.metric(
            "Approval %",
            f"{approval_percentage:.2f}%"
        )

    with col3:
        st.metric(
            "Rejection %",
            f"{rejection_percentage:.2f}%"
        )


    fig_stats = go.Figure()

    fig_stats.add_trace(go.Bar(
        x=[
            "Average Income",
            "Average Loan Amount"
        ],
        y=[
            avg_income,
            avg_loan
        ]
    ))

    fig_stats.update_layout(
        title="Average Financial Information"
    )

    st.plotly_chart(
        fig_stats,
        use_container_width=True
    )


# =========================
# MODEL PERFORMANCE PAGE
# =========================

if page == "Model Performance":

    st.header("Model Performance Comparison")

    models = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost"
    ]

    accuracy = [
        0.934,
        0.995,
        0.995,
        0.994
    ]

    fig_model = go.Figure()

    fig_model.add_trace(go.Bar(
        x=models,
        y=accuracy,
        text=accuracy,
        textposition='auto'
    ))

    fig_model.update_layout(
        title="Model Accuracy Comparison",
        yaxis_title="Accuracy"
    )

    st.plotly_chart(
        fig_model,
        use_container_width=True
    )


# =========================
# ABOUT PROJECT PAGE
# =========================

if page == "About Project":

    st.header("About Project")

    st.write(
        "This project predicts loan approval "
        "using Machine Learning and Explainable AI."
    )

    st.subheader("Technologies Used")

    st.write("✔ Python")
    st.write("✔ Streamlit")
    st.write("✔ Scikit-learn")
    st.write("✔ XGBoost")
    st.write("✔ SHAP")
    st.write("✔ Plotly")

    st.subheader("Models Used")

    st.write("✔ Logistic Regression")
    st.write("✔ Decision Tree")
    st.write("✔ Random Forest")
    st.write("✔ XGBoost")