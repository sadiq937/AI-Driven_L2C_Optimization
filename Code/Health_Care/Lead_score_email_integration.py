import os
import smtplib
from email.message import EmailMessage

import pandas as pd

# --- CONFIGURATION ---
LEAD_SCORE_PATH = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Data/healthcare_Lead_score.csv"
ORIGINAL_DATA_PATH = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Data/synthetic_healthcare_data.csv"
TO_EMAIL = "totestingemail2025@gmail.com"
FROM_EMAIL = "testemailai2025@gmail.com"  # Replace with your email
EMAIL_PASSWORD = (
    "pito xlcd aaqa xhfe"  # Replace with your app password or email password
)

# --- Step 1: Load data ---
lead_df = pd.read_csv(LEAD_SCORE_PATH)
original_df = pd.read_csv(ORIGINAL_DATA_PATH)

# --- Step 2: Filter by Lead Score > 90 ---
high_leads = lead_df[lead_df["Lead Score"] > 90]
filtered_df = original_df[
    original_df["Patient ID"].isin(high_leads["Patient ID"])
]

# --- Step 3: Save filtered data to Excel ---
filtered_file = "high_lead_score_patients.xlsx"
filtered_df.to_excel(filtered_file, index=False)

# --- Step 4: Email logic ---
msg = EmailMessage()
msg["Subject"] = "High-Value Patient Leads - Lead Score Above 90"
msg["From"] = FROM_EMAIL
msg["To"] = TO_EMAIL

msg.set_content(
    """Dear Team,

Please find attached a detailed report containing patient records with a Lead Score above 90. These individuals have been algorithmically identified as high-potential leads for healthcare services based on our AI-driven scoring system.

The data includes key engagement indicators, demographic insights, referral trends, and other predictive signals that distinguish these patients from the broader population. This report is intended to help prioritize follow-up strategies, enhance conversion rates, and optimize resource allocation.

If you have any questions or would like deeper analytics, feel free to reach out.

Best regards,   
AI-Driven L2O Optimization Team"""
)


# Attach Excel
with open(filtered_file, "rb") as f:
    file_data = f.read()
    msg.add_attachment(
        file_data,
        maintype="application",
        subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filtered_file,
    )

# --- Step 5: Send email via SMTP ---
with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
    smtp.login(FROM_EMAIL, EMAIL_PASSWORD)
    smtp.send_message(msg)

print("Email sent successfully!")
