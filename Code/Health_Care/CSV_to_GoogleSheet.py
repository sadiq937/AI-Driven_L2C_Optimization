import smtplib
from email.message import EmailMessage

import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

# ------------- CONFIGURATION ------------- #

# Local CSV with Lead Score
CSV_PATH = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Data/healthcare_Lead_score.csv"

# Google Sheet name (must already be created)
GOOGLE_SHEET_NAME = "Lead Score data"

# Path to your service account JSON key
CREDENTIALS_FILE = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Code/Health_Care/gspread_creds.json"

# ------------- AUTHENTICATION ------------- #

# Define scope and credentials
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]
creds = ServiceAccountCredentials.from_json_keyfile_name(
    CREDENTIALS_FILE, scope
)
client = gspread.authorize(creds)

# Open the Google Sheet
sheet = client.open(GOOGLE_SHEET_NAME).sheet1

# ------------- LOAD AND PUSH DATA ------------- #

# Load your lead score data
df = pd.read_csv(CSV_PATH)

# Clear existing content
sheet.clear()

# Update the sheet with new data
sheet.update([df.columns.values.tolist()] + df.values.tolist())

print("Google Sheet updated successfully!")


# ---------------- CONFIGURATION ----------------
SENDER_EMAIL = "testemailai2025@gmail.com"  # Your Gmail
SENDER_PASSWORD = "pito xlcd aaqa xhfe"  # Use Gmail App Password
TO_EMAILS = [
    "totestingemail2025@gmail.com",
    "geetnajalikaza@stralynn.com",
    "sl2213@scarletmail.rutgers.edu",
]

SUBJECT = "[Notification] Google Sheet Update: Healthcare Lead Scores"
SHEET_NAME = "Lead Score data"


# ---------------- EMAIL FUNCTION ----------------
def send_notification_email(
    update_reason="An update has occurred on your sheet.",
):
    msg = EmailMessage()
    msg["Subject"] = SUBJECT
    msg["From"] = SENDER_EMAIL
    msg["To"] = ", ".join(TO_EMAILS)

    body = f"""
Hello Team,

This is to inform you that the Google Sheet titled **"{SHEET_NAME}"** was just updated successfully by the automation system.

**Update Reason**: {update_reason}

You can now review the latest data by accessing the shared Google Sheet through your Drive.

If you have any questions or notice discrepancies, please reach out to the operations or analytics team.

Best regards,  
**L2O Automation System**  
    """

    msg.set_content(body)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)
        print("Notification email sent successfully.")
    except Exception as e:
        print("Failed to send email:", e)


# ---------------- EXAMPLE TRIGGER ----------------
if __name__ == "__main__":
    # Triggered after successful sheet update
    send_notification_email(
        update_reason="New lead scores above threshold were inserted."
    )
