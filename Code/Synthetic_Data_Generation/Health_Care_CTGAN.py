# import ctgan

# from ctgan.synthesizers import CTGANSynthesizer
# import ctgan.synthesizers
import numpy as np
import pandas as pd

# from ctgan.synthesizers.ctgan import CTGANSynthesizer
from ctgan import CTGAN

# from ctgan import CTGANSynthesizer
from sklearn.model_selection import train_test_split


# Sample Data Generation (For illustration purposes, we mimic real-world distribution)
def generate_synthetic_data(num_rows=20000):
    # Categorical columns
    specialties = [
        "Cardiology",
        "Neurology",
        "Orthopedics",
        "General Surgery",
        "Pediatrics",
    ]
    insurance_providers = [
        "Blue Cross Blue Shield",
        "Aetna",
        "Cigna",
        "United Healthcare",
        "Humana",
    ]
    healthcare_needs = [
        "Telemedicine Consultation",
        "Routine Checkup",
        "MRI Scan",
        "Surgery Consultation",
        "Emergency Care",
    ]
    referral_sources = [
        "Doctor Referral",
        "Digital Ad",
        "Word of Mouth",
        "Online Search",
        "Direct Visit",
    ]
    chronic_conditions = [
        "Diabetes, Hypertension",
        "Hypertension, Asthma",
        "No chronic conditions",
    ]
    communication_channels = ["SMS", "Email", "Phone", "Online Portal"]
    action_recommendations = [
        "Follow-up consultation",
        "Schedule test",
        "Revisit for re-evaluation",
        "Prescription refills",
    ]
    insurance_types = ["PPO", "HMO", "EPO", "POS", "Indemnity"]
    provider_success_rates = ["Low", "Moderate", "High"]

    # Numeric columns
    patient_ages = np.random.randint(20, 80, size=num_rows)
    appointment_frequencies = np.random.choice([1, 2, 3, 4], num_rows)
    treatment_costs = np.random.randint(100, 3000, size=num_rows)
    follow_up_response_times = np.random.choice(
        [24, 48, 72], num_rows
    )  # Response time in hours
    claims_processing_times = np.random.choice(
        [1, 2, 3, 4], num_rows
    )  # Days to process claims

    # Additional columns
    referral_volumes = np.random.randint(1, 100, size=num_rows)
    missed_appointments = np.random.randint(
        0, 3, size=num_rows
    )  # Number of missed appointments
    sales_rep_ids = [
        f"S{np.random.randint(100, 999)}" for _ in range(num_rows)
    ]
    payment_due_dates = pd.to_datetime(
        np.random.choice(
            pd.date_range("2024-01-01", "2025-01-01", freq="D"), num_rows
        )
    )

    # Create DataFrame
    data = {
        "Patient ID": [
            f"L{np.random.randint(10000, 99999)}" for _ in range(num_rows)
        ],
        "Patient/Provider Name": [
            f"John Doe / Dr. {np.random.choice(['Sarah Smith', 'David Johnson', 'Emily Brown'])}"
            for _ in range(num_rows)
        ],
        "Contact Info": [
            f"email{np.random.randint(1000, 9999)}@example.com"
            for _ in range(num_rows)
        ],
        "Hospital/Clinic Affiliation": [
            f"Hospital {np.random.choice(['St. Mary\'s', 'City General', 'Greenwood Medical'])}"
            for _ in range(num_rows)
        ],
        "Specialty (for providers)": np.random.choice(specialties, num_rows),
        "Insurance Provider": np.random.choice(insurance_providers, num_rows),
        "Healthcare Needs": np.random.choice(healthcare_needs, num_rows),
        "Chatbot Interaction Data": [
            f"Asked about {np.random.choice(['MRI', 'Appointment', 'Billing'])} services"
            for _ in range(num_rows)
        ],
        "Patient Inquiry Type": np.random.choice(
            ["Appointment Booking", "Treatment Info", "Billing Query"],
            num_rows,
        ),
        "Provider Inquiry Type": np.random.choice(
            ["Equipment Pricing", "Service Availability", "Patient Referral"],
            num_rows,
        ),
        "Website Visit Data": [
            f"{np.random.randint(1, 10)} visits, Avg. time: {np.random.randint(2, 10)} min"
            for _ in range(num_rows)
        ],
        "Symptom/Service Interest": [
            f"{np.random.choice(['Headache', 'Fever', 'Back Pain'])}, {np.random.choice(['MRI', 'Consultation', 'Surgery'])}"
            for _ in range(num_rows)
        ],
        "Insurance Verification Status": np.random.choice(
            ["Verified", "Pending"], num_rows
        ),
        "Historical Patient Data": [
            f"{np.random.randint(1, 10)} past visits, {np.random.choice(['Hypertension', 'Asthma', 'Diabetes'])}"
            for _ in range(num_rows)
        ],
        "Referral Source": np.random.choice(referral_sources, num_rows),
        "Patient Demographics": [
            f"Male, {age}, {np.random.choice(['California', 'New York', 'Texas'])}"
            for age in patient_ages
        ],
        "Chronic Conditions": np.random.choice(chronic_conditions, num_rows),
        "Prescription History": [
            f"{np.random.choice(['Metformin', 'Atorvastatin', 'Lisinopril'])}"
            for _ in range(num_rows)
        ],
        "Interaction ID": [
            f"I{np.random.randint(10000, 99999)}" for _ in range(num_rows)
        ],
        "Call Transcript": [
            f"Patient asked about {np.random.choice(['treatment plans', 'medication options'])}"
            for _ in range(num_rows)
        ],
        "Patient/Provider Feedback": [
            f"{np.random.choice(['Very satisfied', 'Neutral', 'Dissatisfied'])} with service"
            for _ in range(num_rows)
        ],
        "Sales Rep ID": np.random.choice(sales_rep_ids, num_rows),
        "Healthcare Compliance Concerns": [
            f"{np.random.choice(['Possible HIPAA violation', 'None detected'])}"
            for _ in range(num_rows)
        ],
        "Follow-up Recommendation": np.random.choice(
            action_recommendations, num_rows
        ),
        "Appointment Frequency": appointment_frequencies,
        "Claims Submission History": np.random.randint(1, 10, size=num_rows),
        "Telemedicine Engagement": np.random.choice(
            ["Low", "Medium", "High"], num_rows
        ),
        "Support Requests": np.random.randint(0, 5, size=num_rows),
        "Missed Appointments": missed_appointments,
        "Provider/Equipment Usage Trends": np.random.choice(
            ["Declining trend", "Stable", "Rising trend"], num_rows
        ),
        "Patient Flow Data": np.random.choice(
            ["High traffic", "Medium", "Low traffic"], num_rows
        ),
        "Referral Volume": referral_volumes,
        "Average Treatment Cost": treatment_costs,
        "Payer/Provider Network Utilization": np.random.randint(
            50, 100, size=num_rows
        ),
        "Prior Authorization Delays": np.random.randint(0, 10, size=num_rows),
        "Prior Diagnoses": [
            f"{np.random.choice(['Hypertension', 'Diabetes', 'None'])}"
            for _ in range(num_rows)
        ],
        "Suggested Treatment Plans": [
            f"{np.random.choice(['Diet change', 'Exercise', 'Medication adjustment'])}"
            for _ in range(num_rows)
        ],
        "Alternative Therapies": [
            f"{np.random.choice(['Acupuncture', 'Yoga', 'Chiropractic'])}"
            for _ in range(num_rows)
        ],
        "Medical Device/Product Interest": [
            f"{np.random.choice(['Glucose Monitor', 'Blood Pressure Monitor'])}"
            for _ in range(num_rows)
        ],
        "Next Best Action": np.random.choice(action_recommendations, num_rows),
        "Service/Product ID": [
            f"ID{np.random.randint(1000, 9999)}" for _ in range(num_rows)
        ],
        "Hospital/Clinic Pricing": np.random.randint(500, 3000, size=num_rows),
        "Insurance Reimbursement Rates": np.random.randint(
            100, 2500, size=num_rows
        ),
        "Government Payer Adjustments": [
            f"{np.random.randint(5, 20)}% Medicare Reduction"
            for _ in range(num_rows)
        ],
        "Tiered Pricing": np.random.choice(
            ["Tier 1", "Tier 2", "Tier 3"], num_rows
        ),
        "Risk-Based Pricing Models": np.random.choice(
            ["High-risk pricing", "Low-risk pricing"], num_rows
        ),
        "Proposal ID": [
            f"PR{np.random.randint(100, 999)}" for _ in range(num_rows)
        ],
        "Order ID": [
            f"OR{np.random.randint(1000, 9999)}" for _ in range(num_rows)
        ],
        "Invoice ID": [
            f"INV{np.random.randint(1000, 9999)}" for _ in range(num_rows)
        ],
        "Transaction ID": [
            f"TX{np.random.randint(1000, 9999)}" for _ in range(num_rows)
        ],
        "Payment Due Date": payment_due_dates,
        "Outstanding Balance": np.random.randint(0, 1000, size=num_rows),
        "Preferred Communication Channel": np.random.choice(
            communication_channels, num_rows
        ),
        "Metric ID": [
            f"M{np.random.randint(1000, 9999)}" for _ in range(num_rows)
        ],
        "Outlier Detection": [
            f"{np.random.choice(['Spike in admissions', 'No anomaly detected'])}"
            for _ in range(num_rows)
        ],
        "Baseline Value": np.random.randint(30, 100, size=num_rows),
        "Alert Status": np.random.choice(["Low", "Medium", "High"], num_rows),
        "Compliance Violation Flags": [
            f"{np.random.choice(['None detected', 'HIPAA Breach Risk'])}"
            for _ in range(num_rows)
        ],
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
        "Conversion Probability": np.random.randint(30, 80, size=num_rows),
        "Insurance Provider Processing Time": np.random.randint(
            1, 10, size=num_rows
        ),
        "Provider Referral Trends": np.random.choice(
            ["Stable", "Increasing", "Decreasing"], num_rows
        ),
        "Patient Portal Usage": [
            f"Logged in {np.random.randint(1, 10)} times"
            for _ in range(num_rows)
        ],
        "Threat Intelligence Data": [
            f"{np.random.choice(['No threats detected', 'Potential fraud risk'])}"
            for _ in range(num_rows)
        ],
        "Insurance Coverage": np.random.choice(
            ["Full Coverage", "Partial Coverage", "No Coverage"], num_rows
        ),
        "Duplicate Billing Flags": [
            f"{np.random.choice(['None detected', 'Duplicate found'])}"
            for _ in range(num_rows)
        ],
        "Insurance Type": np.random.choice(insurance_types, num_rows),
        "Upselling Potential": np.random.choice(
            ["Low", "Moderate", "High"], num_rows
        ),
        "Past Fraudulent Claims": [
            f"{np.random.choice(['None', 'Detected'])}"
            for _ in range(num_rows)
        ],
        "Billing History": [f"Paid in full" for _ in range(num_rows)],
        "Prior Appointment History": [
            f"{np.random.randint(1, 5)} appointments, {np.random.choice(['all attended', 'some missed'])}"
            for _ in range(num_rows)
        ],
        "Telehealth Engagement": np.random.choice(
            ["Low", "Medium", "High"], num_rows
        ),
        "Hospital IT Security Logs": [
            f"{np.random.choice(['No incidents', 'Incident detected'])}"
            for _ in range(num_rows)
        ],
        "Email Content": [
            f"{np.random.choice(['Appointment confirmation', 'Insurance approval'])}"
            for _ in range(num_rows)
        ],
        "Hospital Pricing Models": np.random.choice(
            ["Fixed rate", "Variable rates"], num_rows
        ),
        "Insurance Claim Amount": np.random.randint(100, 3000, size=num_rows),
        "Insurance Payer Data": [
            f"Payer: {np.random.choice(['Blue Cross', 'Aetna', 'Cigna'])}"
            for _ in range(num_rows)
        ],
        "Compliance Reports": [f"No issues found" for _ in range(num_rows)],
        "Network Vulnerabilities": [
            f"{np.random.choice(['No vulnerabilities', 'Low risk detected'])}"
            for _ in range(num_rows)
        ],
        "Sentiment Analysis": np.random.choice(
            ["Positive", "Neutral", "Negative"], num_rows
        ),
        "Feedback Rating": np.random.choice([1, 2, 3, 4, 5], num_rows),
        "Follow-up Response Time": follow_up_response_times,
        "Treatment Cost Trends": [
            f"{np.random.choice(['Increasing costs', 'Stable costs'])}"
            for _ in range(num_rows)
        ],
        "Provider Success Rate": np.random.choice(
            provider_success_rates, num_rows
        ),
        "EMR System Performance": [
            f"{np.random.choice(['No system errors', 'Minor glitches'])}"
            for _ in range(num_rows)
        ],
        "Billing Accuracy": np.random.choice([99, 100], num_rows),
        "Compliance Audit Scores": np.random.choice(
            [85, 90, 95, 100], num_rows
        ),
        "Consultation Bookings": np.random.randint(30, 100, size=num_rows),
        "Insurance Approvals": np.random.randint(10, 80, size=num_rows),
        "Follow-up Appointments": np.random.randint(5, 50, size=num_rows),
        "Communication History": [
            f"{np.random.choice(['Email', 'Phone call', 'In-person'])}"
            for _ in range(num_rows)
        ],
        "Patient Acquisition Rate": np.random.choice(
            ["10% increase", "Stable growth", "Decline"], num_rows
        ),
        "Marketing Channel Performance": [
            f"{np.random.choice(['Social media 20% conversion', 'Email marketing 30% conversion'])}"
            for _ in range(num_rows)
        ],
        "Referral Success Rate": np.random.randint(50, 90, size=num_rows),
        "Payment History": [f"Paid in full" for _ in range(num_rows)],
        "Compliance Score": np.random.choice([95, 100], num_rows),
    }

    return pd.DataFrame(data)


# Generate sample data
df = generate_synthetic_data(20000)

"""
# Data Preprocessing: Encoding categorical variables into integers
# for column in df.select_dtypes(include=["object"]).columns:
#    df[column] = df[column].astype("category").cat.codes

# --- Step 2: Convert datetime columns to UNIX timestamp ---
for col in df.select_dtypes(include="datetime64[ns]"):
    df[col] = df[col].astype(np.int64) // 10**9


# Split the data into features and target
X = df.drop("Patient ID", axis=1)  # Features (excluding Patient ID)
y = df[
    "Patient ID"
]  # Target column (optional, if you want to track the patient ID)

# Train-test split
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Initialize and train the CTGAN model
# ctgan = CTGANSynthesizer()
# ctgan.fit(X_train)

# Use CTGAN from the main module
synthesizer = ctgan.CTGANSynthesizer()
synthesizer.fit(
    X_train,
    discrete_columns=X_train.select_dtypes(include="int").columns.tolist(),
)

synthesizer = CTGAN(epochs=10)  # Reduce epochs while testing
synthesizer.fit(
    X_train,
    discrete_columns=X_train.select_dtypes(include="int").columns.tolist(),
)

# Generate synthetic data
synthetic_data = synthesizer.sample(20000)

# Convert back categorical variables to their original labels
for column in df.select_dtypes(include=["object"]).columns:
    synthetic_data[column] = (
        synthetic_data[column]
        .astype("category")
        .cat.categories[synthetic_data[column]]
    )

# Add "Patient ID" column
synthetic_data["Patient ID"] = [
    f"L{np.random.randint(10000, 99999)}" for _ in range(20000)
]

# Save the synthetic data to a CSV file
synthetic_data.to_csv("synthetic_healthcare_data.csv", index=False)

print("Synthetic data generated and saved to synthetic_healthcare_data.csv.")
"""

# --- Step 3: Encode categorical variables and save mappings ---
category_mappings = {}
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype("category")
    category_mappings[col] = dict(enumerate(df[col].cat.categories))
    df[col] = df[col].cat.codes

# --- Step 4: Split features/target ---
X = df.drop("Patient ID", axis=1)
y = df["Patient ID"]

X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# --- Step 5: Train CTGAN ---
discrete_columns = [
    col for col in X.columns if X[col].nunique() < 50
]  # Safer selection
synthesizer = CTGAN(epochs=10)
synthesizer.fit(X_train, discrete_columns=discrete_columns)

# --- Step 6: Generate synthetic data ---
synthetic_data = synthesizer.sample(len(X))

# --- Step 7: Reverse-encode categorical variables ---
for col, mapping in category_mappings.items():
    if col in synthetic_data.columns:
        reverse_mapping = {v: k for k, v in mapping.items()}
        synthetic_data[col] = (
            synthetic_data[col].round().astype(int).map(mapping)
        )

# --- Step 8: Add synthetic Patient IDs ---
synthetic_data["Patient ID"] = [
    f"L{np.random.randint(10000, 99999)}" for _ in range(len(synthetic_data))
]

# --- Step 9: Save to CSV ---
synthetic_data.to_csv("synthetic_healthcare_data.csv", index=False)
print("Synthetic data generated and saved to synthetic_healthcare_data.csv.")
