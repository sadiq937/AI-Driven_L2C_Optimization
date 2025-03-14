import pandas as pd
import sdv
from sdv.tabular import CTGAN

# Your data_fields definition (same as your original code)
data_fields = [
    ("Lead ID", "Unique identifier for the lead"),
    ("Client ID", "Unique identifier for an existing customer"),
    ("Client/Organization Name", "Name of the potential customer"),
    ("Contact Info", "Phone number or email of the lead"),
    ("Industry", "Industry of the lead's organization"),
    ("Security Needs", "Cybersecurity services/products needed"),
    ("Lead Source", "How the lead was generated"),
    ("Chatbot Interaction Data", "Details of interaction with chatbot"),
    ("Inquiry Type", "Nature of the inquiry (Phishing, Malware, etc.)"),
    ("Website Visit Data", "Information on lead’s web activity"),
    ("Appointment/Consultation Request", "Whether the lead has requested a meeting"),
    ("Incident Report History", "Past security breaches reported"),
    ("Historical Data on Past Incidents", "Previous cyber incidents reported"),
    ("Security Software Usage History", "Previously used security solutions"),
    ("Previous Security Breach History", "Number and severity of past security breaches"),
    ("Threat Intelligence Feeds", "Data on known threats from intelligence sources"),
    ("Industry Security Standards", "Compliance requirements for the lead’s industry"),
    ("Interaction ID", "Unique ID for a recorded interaction"),
    ("Call/Chat Transcript", "Summary of the conversation"),
    ("Client Feedback", "Feedback received from the lead"),
    ("Sales Rep ID", "Identifier of the assigned sales rep"),
    ("Client Risk Awareness", "Level of awareness regarding cyber threats"),
    ("Recommended Security Solutions", "Suggested cybersecurity products/services"),
    ("Follow-up Recommendation", "Suggested next steps for sales"),
    ("Subscription/License Renewal Frequency", "How often the client renews security services"),
    ("Past Breach Incidents", "Number of security breaches the client has faced"),
    ("Support Request History", "Past customer support tickets"),
    ("Service Downtime History", "Record of security-related downtime"),
    ("Usage Patterns of Security Features", "How often security tools are used"),
    ("Threat Detection Data", "Logs of security threats detected"),
    ("Network Traffic Analysis", "Monitoring of inbound/outbound network traffic"),
    ("Security Incident Volume", "Number of incidents recorded"),
    ("Detection Time", "Time taken to detect a threat"),
    ("IT Security Budget", "Annual budget allocated for cybersecurity"),
    ("Competitor Security Solutions", "Existing security solutions in use by the client"),
    ("Vulnerability Scan Results", "Findings from security scans"),
    ("System Performance Metrics", "Performance stats related to security tools"),
    ("Outlier Detection", "Detection of unusual cybersecurity activity"),
    ("Baseline Value", "Expected normal behavior pattern"),
    ("Alert Status", "Status of detected anomalies"),
    ("Compliance Violation Flags", "Flags raised for compliance violations"),
    ("Incident Severity Level", "Categorization of detected security incidents"),
    ("Service/Product ID", "Unique identifier for cybersecurity service/product"),
    ("Security Service Pricing", "Cost of security service/product"),
    ("Subscription Tiers", "Pricing levels for different service packages"),
    ("Proposal ID", "Unique ID for a sales proposal"),
    ("Service Agreement Terms", "Terms of cybersecurity service contracts"),
    ("Compliance Approvals", "Certifications and approvals"),
    ("Product Catalog", "List of security tools offered"),
    ("Expiry Date", "Validity period of the proposal"),
    ("Order ID", "Unique order identifier"),
    ("Invoice ID", "Unique invoice number"),
    ("Payment Due Date", "Date payment is expected"),
    ("Outstanding Balance", "Amount yet to be paid"),
    ("Preferred Communication Channel", "Preferred medium for notifications"),
    ("Claims Adjudication Status", "Status of insurance/security claim"),
    ("Payment Processing Status", "Status of client payments"),
    ("Revenue Cycle Management Metrics", "Key financial performance indicators"),
    ("Delayed Payment Trends", "History of late payments"),
    ("Email Open Rate", "Percentage of emails opened by the lead/client"),
    ("Chat Interaction Data", "Summary of chatbot or live chat discussions"),
    ("Support Requests", "Number of security-related support tickets"),
    ("Billing Amount", "Amount charged for cybersecurity services"),
    ("Fraudulent Charge Detection", "Flags indicating possible fraudulent transactions"),
    ("IP Address", "Geolocation of login or payment"),
    ("Past Payment Trends", "Historical payment behavior"),
    ("Customer Lifetime Value (CLV)", "Total expected revenue from a customer over time"),
    ("Unusual Login Activity", "Logins from unexpected locations or devices"),
    ("Threat Detection Data", "AI-based threat analytics from security systems"),
    ("Incident Logs", "Records of past security incidents"),
    ("Intrusion Detection Data", "Data from IDS/IPS systems"),
    ("Risk Indicators", "Signs of potential security risks"),
    ("Incident ID", "Unique identifier for a security incident"),
    ("Average Incident Response Time", "Time taken to respond to incidents"),
    ("Threat Score", "Risk level associated with detected threats"),
    ("Security Policy Compliance", "Adherence to security policies"),
    ("Patch Deployment Rate", "Percentage of deployed security patches"),
    ("Incident Recovery Time", "Time taken to recover from an incident"),
    ("Security Patch Deployment Logs", "Records of applied patches"),
    ("Patch Success Rate", "Percentage of successfully applied patches"),
    ("Vulnerability Reduction Metrics", "Impact of patches on security vulnerabilities"),
    ("SOC Efficiency Rating", "Effectiveness of the Security Operations Center"),
    ("Automated Response Rate", "Percentage of automated security responses"),
    ("Compliance Audit Logs", "Logs related to compliance audits"),
    ("Security Policy Score", "Rating of security policies in place"),
    ("Regulatory Adherence Data", "Compliance with regulatory requirements"),
    ("Prior Engagement", "Past interactions with the lead"),
    ("Proposal Acceptance Rate", "Percentage of proposals accepted"),
    ("Industry Vertical", "Industry classification of the lead"),
    ("Alert ID", "Unique identifier for a security alert"),
    ("Client Engagement Rate", "Interaction level of the client"),
    ("Security Alert Open Rate", "Percentage of opened security alerts"),
    ("Compliance Audit Score", "Score received in compliance audits"),
    ("Security Policy Implementation Rate", "Percentage of recommended policies implemented"),
    ("Company Size", "Size of the client’s organization"),
    ("Service Level", "Tier of cybersecurity services subscribed"),
    ("Contract Length", "Duration of the service contract"),
    ("Discount Recommendation", "Suggested discount based on deal size"),
    ("Endpoint Count", "Number of endpoints secured"),
    ("Data Scanned", "Amount of data analyzed for security threats"),
    ("Monthly Service Utilization", "Level of cybersecurity service usage"),
    ("Subscription Renewal Rate", "Rate of subscription renewals"),
    ("Support Ticket Resolution Score", "Percentage of first-call resolution for support tickets"),
]

# Create an empty DataFrame with the defined columns
columns = [col_name for col_name, _ in data_fields]
df = pd.DataFrame(columns=columns)

# Create and train CTGAN model.
model = CTGAN()
model.fit(df) # fit on empty dataframe to just establish the needed column types

# Generate synthetic data
synthetic_data = model.sample(num_rows=20000) # generate 20000 rows

# Save the synthetic data to a CSV file
synthetic_data.to_csv("synthetic_cybersecurity_L2C_data_CTGAN.csv", index=False)

print("Data generation complete (CTGAN). Saved as synthetic_cybersecurity_L2C_data_CTGAN.csv")
