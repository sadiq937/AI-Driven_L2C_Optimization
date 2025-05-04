import datetime
import os

import pandas as pd
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Paths
CSV_PATH = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Data/healthcare_Lead_score.csv"
THRESHOLD = 90
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


# Authenticate and create calendar service
def get_calendar_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            os.path.join(os.path.dirname(__file__), "credentials.json"),
            SCOPES,
        )
        creds = flow.run_local_server(prompt="select_account")
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("calendar", "v3", credentials=creds)


# Create a calendar event
def create_calendar_event(service, lead, start_time):
    patient_id = lead.get("Patient ID", "Unknown")
    lead_score = lead.get("Lead Score", 0)
    summary = f"Follow-Up with Lead: {patient_id}"
    description = f"High Lead Score: {lead_score}\nSchedule follow-up call or consultation."

    end_time = start_time + datetime.timedelta(minutes=30)

    event = {
        "summary": summary,
        "description": description,
        "start": {"dateTime": start_time.isoformat() + "Z", "timeZone": "UTC"},
        "end": {"dateTime": end_time.isoformat() + "Z", "timeZone": "UTC"},
    }

    created_event = (
        service.events().insert(calendarId="primary", body=event).execute()
    )
    print(f"Event created: {created_event.get('htmlLink')}")


# Main logic
def main():
    print("Reading lead score data...")
    df = pd.read_csv(CSV_PATH)

    # Filter high score leads and take top 10 only
    high_score_leads = df[df["Lead Score"] >= THRESHOLD].head(10)

    if high_score_leads.empty:
        print("No high score leads found.")
        return

    service = get_calendar_service()
    print("Connected to Google Calendar API")

    # Start scheduling from 10:00 AM tomorrow (UTC)
    base_time = datetime.datetime.utcnow().replace(
        hour=10, minute=0, second=0, microsecond=0
    ) + datetime.timedelta(days=1)

    for i, (_, lead) in enumerate(high_score_leads.iterrows()):
        start_time = base_time + datetime.timedelta(minutes=30 * i)
        create_calendar_event(service, lead, start_time)

    print("All events created for top 10 high scoring leads!")


if __name__ == "__main__":
    main()
