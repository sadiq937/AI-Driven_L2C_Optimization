import numpy as np
import pandas as pd


# --- Generate Synthetic Healthcare Data ---
def generate_full_healthcare_data(num_rows=20000):
    np.random.seed(42)

    def rand_currency(min_val, max_val):
        return f"${np.random.randint(min_val, max_val)}"

    def rand_percent(min_val=10, max_val=100):
        return f"{np.random.randint(min_val, max_val)}%"

    def rand_range(label, min_val, max_val, suffix=""):
        return f"{np.random.randint(min_val, max_val)} {label}{suffix}"

    def rand_id(prefix, start=100, end=9999):
        return f"{prefix}{np.random.randint(start, end)}"

    data = {
        "Patient ID": [rand_id("L", 10000, 99999) for _ in range(num_rows)],
        "Patient/Provider Name": [
            f"John Doe / Dr. {np.random.choice(['Sarah Smith', 'David Johnson', 'Emily Brown'])}"
            for _ in range(num_rows)
        ],
        "Contact Info": [
            f"johndoe{np.random.randint(1000, 9999)}@email.com / +1 234 567 {np.random.randint(1000,9999)}"
            for _ in range(num_rows)
        ],
        "Hospital/Clinic Affiliation": np.random.choice(
            ["St. Mary’s Hospital", "City General", "Greenwood Medical"],
            num_rows,
        ),
        "Specialty (for providers)": np.random.choice(
            ["Cardiology", "Neurology", "Orthopedics", "Pediatrics"], num_rows
        ),
        "Insurance Provider": np.random.choice(
            ["Blue Cross Blue Shield", "Aetna", "Cigna"], num_rows
        ),
        "Healthcare Needs": np.random.choice(
            ["Telemedicine Consultation", "Routine Checkup", "MRI Scan"],
            num_rows,
        ),
        "Chatbot Interaction Data": np.random.choice(
            [
                "Asked about MRI services",
                "Inquired about billing",
                "Requested appointment info",
            ],
            num_rows,
        ),
        "Patient Inquiry Type": np.random.choice(
            ["Appointment Booking", "Treatment Info", "Billing Query"],
            num_rows,
        ),
        "Provider Inquiry Type": np.random.choice(
            ["Equipment Pricing", "Service Availability", "Patient Referral"],
            num_rows,
        ),
        "Website Visit Data": [
            rand_range("visits, Avg. time: ", 1, 5, " min")
            for _ in range(num_rows)
        ],
        "Symptom/Service Interest": np.random.choice(
            ["Headache, MRI", "Fever, Consultation", "Back Pain, Surgery"],
            num_rows,
        ),
        "Insurance Verification Status": np.random.choice(
            ["Verified", "Pending"], num_rows
        ),
        "Historical Patient Data": [
            f"{np.random.randint(1, 10)} past visits, {np.random.choice(['Hypertension', 'Asthma', 'Diabetes'])}"
            for _ in range(num_rows)
        ],
        "Referral Source": np.random.choice(
            ["Doctor Referral", "Digital Ad", "Word of Mouth"], num_rows
        ),
        "Patient Demographics": [
            f"{np.random.choice(['Male', 'Female'])}, {np.random.randint(20, 80)}, {np.random.choice(['California', 'Texas', 'New York'])}"
            for _ in range(num_rows)
        ],
        "Chronic Conditions": np.random.choice(
            [
                "Diabetes, Hypertension",
                "Hypertension, Asthma",
                "No chronic conditions",
            ],
            num_rows,
        ),
        "Prescription History": np.random.choice(
            ["Metformin, Atorvastatin", "Lisinopril", "None"], num_rows
        ),
        "Interaction ID": [
            rand_id("I", 10000, 99999) for _ in range(num_rows)
        ],
        "Call Transcript": np.random.choice(
            [
                "Patient asked about treatment plans",
                "Inquired about medication options",
            ],
            num_rows,
        ),
        "Patient/Provider Feedback": np.random.choice(
            ["Very satisfied with service", "Neutral", "Dissatisfied"],
            num_rows,
        ),
        "Sales Rep ID": [rand_id("S", 100, 999) for _ in range(num_rows)],
        "Healthcare Compliance Concerns": np.random.choice(
            ["Possible HIPAA violation detected", "None detected"], num_rows
        ),
        "Follow-up Recommendation": np.random.choice(
            ["Schedule follow-up appointment", "Prescription refills"],
            num_rows,
        ),
        "Appointment Frequency": np.random.choice(
            ["1 per year", "2 per year", "Quarterly"], num_rows
        ),
        "Claims Submission History": [
            rand_range("claims", 1, 10) for _ in range(num_rows)
        ],
        "Telemedicine Engagement": np.random.choice(
            ["Low", "Medium", "High"], num_rows
        ),
        "Support Requests": [
            rand_range("support tickets", 0, 5) for _ in range(num_rows)
        ],
        "Missed Appointments": [
            rand_range("missed", 0, 3) for _ in range(num_rows)
        ],
        "Provider/Equipment Usage Trends": np.random.choice(
            ["Declining trend", "Stable", "Rising trend"], num_rows
        ),
        "Patient Flow Data": np.random.choice(
            ["High traffic", "Medium", "Low traffic"], num_rows
        ),
        "Referral Volume": [
            rand_range("referrals", 10, 100) for _ in range(num_rows)
        ],
        "Average Treatment Cost": [
            rand_currency(100, 5000) for _ in range(num_rows)
        ],
        "Payer/Provider Network Utilization": [
            rand_percent(50, 100) for _ in range(num_rows)
        ],
        "Prior Authorization Delays": [
            rand_range("days", 1, 15) for _ in range(num_rows)
        ],
        "Prior Diagnoses": np.random.choice(
            ["Diabetes", "Hypertension", "None"], num_rows
        ),
        "Suggested Treatment Plans": np.random.choice(
            ["Diet change, Exercise", "Medication adjustment"], num_rows
        ),
        "Alternative Therapies": np.random.choice(
            ["Acupuncture, Yoga", "Chiropractic"], num_rows
        ),
        "Medical Device/Product Interest": np.random.choice(
            ["Glucose Monitor", "Blood Pressure Monitor"], num_rows
        ),
        "Next Best Action": np.random.choice(
            ["Follow-up consultation", "Revisit for re-evaluation"], num_rows
        ),
        "Service/Product ID": [
            rand_id("MRI", 100, 999) for _ in range(num_rows)
        ],
        "Hospital/Clinic Pricing": [
            rand_currency(500, 2000) for _ in range(num_rows)
        ],
        "Insurance Reimbursement Rates": [
            rand_currency(300, 1500) for _ in range(num_rows)
        ],
        "Government Payer Adjustments": [
            f"{np.random.randint(5, 20)}% Medicare Reduction"
            for _ in range(num_rows)
        ],
        "Tiered Pricing": np.random.choice(
            ["Tier 1", "Tier 2", "Tier 3"], num_rows
        ),
        "Risk-Based Pricing Models": np.random.choice(
            ["High-risk pricing adjustment", "Low-risk pricing"], num_rows
        ),
        "Proposal ID": [rand_id("PR") for _ in range(num_rows)],
        "Order ID": [rand_id("OR") for _ in range(num_rows)],
        "Invoice ID": [rand_id("INV") for _ in range(num_rows)],
        "Transaction ID": [rand_id("TX", 1000, 9999) for _ in range(num_rows)],
        "Payment Due Date": pd.to_datetime(
            np.random.choice(
                pd.date_range("2024-01-01", "2025-01-01"), num_rows
            )
        ),
        "Outstanding Balance": [
            rand_currency(0, 1000) for _ in range(num_rows)
        ],
        "Preferred Communication Channel": np.random.choice(
            ["SMS", "Email", "Phone"], num_rows
        ),
        "Metric ID": [rand_id("M", 100, 9999) for _ in range(num_rows)],
        "Outlier Detection": np.random.choice(
            ["Spike in admissions", "No anomaly detected"], num_rows
        ),
        "Baseline Value": [
            f"{np.random.randint(30, 100)}/day" for _ in range(num_rows)
        ],
        "Alert Status": np.random.choice(["Low", "Medium", "High"], num_rows),
        "Compliance Violation Flags": np.random.choice(
            ["None detected", "HIPAA Breach Risk"], num_rows
        ),
        "Engagement History": [
            f"Visited website {np.random.randint(1, 10)} times"
            for _ in range(num_rows)
        ],
        "Website Interactions": [
            f"Visited {np.random.randint(1, 5)} pages" for _ in range(num_rows)
        ],
        "Lead Source": np.random.choice(
            ["Digital Ad", "Referral", "Organic Search"], num_rows
        ),
        "Conversion Probability": [
            rand_percent(30, 80) for _ in range(num_rows)
        ],
        "Insurance Provider Processing Time": [
            rand_range("days", 1, 10) for _ in range(num_rows)
        ],
        "Provider Referral Trends": np.random.choice(
            ["Stable", "Increasing", "Decreasing"], num_rows
        ),
        "Patient Portal Usage": [
            f"Logged in {np.random.randint(1, 10)} times"
            for _ in range(num_rows)
        ],
        "Threat Intelligence Data": np.random.choice(
            ["No threats detected", "Potential fraud risk"], num_rows
        ),
        "Insurance Coverage": np.random.choice(
            ["Full Coverage", "Partial Coverage", "No Coverage"], num_rows
        ),
        "Duplicate Billing Flags": np.random.choice(
            ["None detected", "Duplicate found"], num_rows
        ),
        "Insurance Type": np.random.choice(
            ["PPO", "HMO", "EPO", "POS", "Indemnity"], num_rows
        ),
        "Upselling Potential": np.random.choice(
            ["Low", "Moderate", "High"], num_rows
        ),
        "Past Fraudulent Claims": np.random.choice(
            ["None", "Detected"], num_rows
        ),
        "Billing History": np.random.choice(
            ["Paid in full", "Outstanding", "Partially paid"], num_rows
        ),
        "Prior Appointment History": [
            f"{np.random.randint(1, 5)} appointments, {np.random.choice(['all attended', 'some missed'])}"
            for _ in range(num_rows)
        ],
        "Telehealth Engagement": np.random.choice(
            ["Low", "Medium", "High"], num_rows
        ),
        "Hospital IT Security Logs": np.random.choice(
            ["No incidents", "Incident detected"], num_rows
        ),
        "Email Content": np.random.choice(
            [
                "Appointment confirmation",
                "Insurance approval",
                "Payment reminder",
            ],
            num_rows,
        ),
        "Hospital Pricing Models": np.random.choice(
            ["Fixed rate", "Variable rates"], num_rows
        ),
        "Insurance Claim Amount": [
            rand_currency(100, 3000) for _ in range(num_rows)
        ],
        "Insurance Payer Data": np.random.choice(
            ["Payer: Blue Cross", "Payer: Aetna", "Payer: Cigna"], num_rows
        ),
        "Compliance Reports": np.random.choice(
            ["No issues found", "Minor violations"], num_rows
        ),
        "Network Vulnerabilities": np.random.choice(
            ["No vulnerabilities", "Low risk detected"], num_rows
        ),
        "Sentiment Analysis": np.random.choice(
            ["Positive", "Neutral", "Negative"], num_rows
        ),
        "Feedback Rating": [
            f"{np.random.randint(1, 6)} stars" for _ in range(num_rows)
        ],
        "Follow-up Response Time": [
            f"{np.random.choice([24, 48, 72])} hours" for _ in range(num_rows)
        ],
        "Treatment Cost Trends": np.random.choice(
            ["Increasing costs for surgeries", "Stable costs"], num_rows
        ),
        "Provider Success Rate": [
            f"{np.random.randint(80, 100)}%" for _ in range(num_rows)
        ],
        "EMR System Performance": np.random.choice(
            ["No system errors", "Minor glitches"], num_rows
        ),
        "Billing Accuracy": np.random.choice(
            ["99% accurate", "100% accurate"], num_rows
        ),
        "Compliance Audit Scores": np.random.choice(
            ["Passed all audits", "Passed with notes"], num_rows
        ),
        "Consultation Bookings": [
            f"{np.random.randint(10, 100)} consultations per month"
            for _ in range(num_rows)
        ],
        "Insurance Approvals": [
            f"{np.random.randint(10, 80)} approvals per month"
            for _ in range(num_rows)
        ],
        "Follow-up Appointments": [
            f"{np.random.randint(5, 50)} follow-ups per month"
            for _ in range(num_rows)
        ],
        "Communication History": np.random.choice(
            ["Email", "Phone call", "In-person"], num_rows
        ),
        "Patient Acquisition Rate": np.random.choice(
            ["10% increase", "Stable growth", "Decline"], num_rows
        ),
        "Marketing Channel Performance": np.random.choice(
            ["Social media 20% conversion", "Email marketing 30% conversion"],
            num_rows,
        ),
        "Referral Success Rate": np.random.choice(
            ["50% conversion rate", "70% conversion rate"], num_rows
        ),
        "Payment History": np.random.choice(
            ["Paid in full", "Pending", "Overdue"], num_rows
        ),
        "Compliance Score": np.random.choice(
            ["100% compliant", "95% compliant"], num_rows
        ),
    }

    return pd.DataFrame(data)


# Generate the data
df_full = generate_full_healthcare_data()

# --- Export to CSV ---
df_full.to_csv("synthetic_healthcare_data.csv", index=False)
print("Data exported to synthetic_healthcare_data.csv")

# --- Export to Excel ---
# df_full.to_excel("synthetic_healthcare_data.xlsx", index=False)
# print("Data exported to synthetic_healthcare_data.xlsx")
