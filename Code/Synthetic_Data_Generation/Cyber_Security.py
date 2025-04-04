import pandas as pd
import random
from faker import Faker

# Initialize Faker instance
fake = Faker()

# Define the columns and sample values based on your provided description
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

# Number of records to generate
num_records = 20000
rows = []

# Generate synthetic data
for _ in range(num_records):
    record = {}
    
    for col_name, _ in data_fields:
        # Add random or fake values for each column based on the type of data expected
        if col_name == "Lead ID":
            record[col_name] = f"LID-{random.randint(10000, 99999)}"
        elif col_name == "Client ID":
            record[col_name] = f"C{random.randint(10000, 99999)}"
        elif col_name == "Client/Organization Name":
            record[col_name] = fake.company()
        elif col_name == "Contact Info":
            record[col_name] = fake.email()
        elif col_name == "Industry":
            record[col_name] = random.choice(["Finance", "Healthcare", "Retail", "Technology", "Manufacturing"])
        elif col_name == "Security Needs":
            record[col_name] = random.choice(["Endpoint Security", "Cloud Security", "SOC Monitoring", "Firewall Management"])
        elif col_name == "Lead Source":
            record[col_name] = random.choice(["Webinar", "Referral", "Digital Campaign", "Cold Email"])
        elif col_name == "Chatbot Interaction Data":
            record[col_name] = random.choice(["Asked about firewall solutions", "Inquired about SIEM deployment", "Requested demo"])
        elif col_name == "Inquiry Type":
            record[col_name] = random.choice(["Phishing", "Ransomware Protection", "Data Breach Response"])
        elif col_name == "Website Visit Data":
            record[col_name] = random.choice(["Visited Cybersecurity Solutions page", "Downloaded Whitepaper", "Checked Pricing"])
        elif col_name == "Appointment/Consultation Request":
            record[col_name] = random.choice(["Yes", "No"])
        elif col_name == "Incident Report History":
            record[col_name] = random.choice(["Phishing attack in Q2 2023", "DDoS attack in 2022", "No incidents reported"])
        elif col_name == "Historical Data on Past Incidents":
            record[col_name] = random.choice(["DDoS attack in 2022", "Malware incident in 2021", "No historical incidents"])
        elif col_name == "Security Software Usage History":
            record[col_name] = random.choice(["SIEM, Antivirus", "Firewall, EDR", "MFA"])
        elif col_name == "Previous Security Breach History":
            record[col_name] = random.choice(["1 Minor Breach", "3 Major Breaches", "No previous breaches"])
        elif col_name == "Threat Intelligence Feeds":
            record[col_name] = random.choice(["Detected botnet activity in region", "Ransomware threat detected"])
        elif col_name == "Industry Security Standards":
            record[col_name] = random.choice(["GDPR", "ISO 27001", "SOC 2"])
        elif col_name == "Interaction ID":
            record[col_name] = f"INT-{random.randint(10000, 99999)}"
        elif col_name == "Call/Chat Transcript":
            record[col_name] = random.choice(["Discussed firewall pricing", "Asked about compliance", "Requested a demo"])
        elif col_name == "Client Feedback":
            record[col_name] = random.choice(["Interested", "Needs more details", "Budget concerns"])
        elif col_name == "Sales Rep ID":
            record[col_name] = f"SR-{random.randint(1000, 9999)}"
        elif col_name == "Client Risk Awareness":
            record[col_name] = random.choice(["High", "Medium", "Low"])
        elif col_name == "Recommended Security Solutions":
            record[col_name] = random.choice(["Next-Gen Firewall", "MDR Service", "EDR Solution"])
        elif col_name == "Follow-up Recommendation":
            record[col_name] = random.choice(["Schedule a security assessment demo", "Send additional materials", "Follow-up in 3 days"])
        elif col_name == "Subscription/License Renewal Frequency":
            record[col_name] = random.choice(["Annually", "Bi-Annually", "Quarterly"])
        elif col_name == "Past Breach Incidents":
            record[col_name] = random.choice(["2 breaches in the past 12 months", "No breaches reported", "3 breaches in the past 2 years"])
        elif col_name == "Support Request History":
            record[col_name] = random.choice(["Raised 5 tickets", "No previous support requests", "10 tickets for malware-related issues"])
        elif col_name == "Service Downtime History":
            record[col_name] = random.choice(["2 hours in the last quarter", "No downtime", "4 hours in the last 6 months"])
        elif col_name == "Usage Patterns of Security Features":
            record[col_name] = random.choice(["Uses SIEM daily, but firewall logs weekly", "Firewall management used occasionally", "Active on EDR every day"])
        elif col_name == "Threat Detection Data":
            record[col_name] = random.choice(["10 unauthorized access attempts", "5 anomalies detected today"])
        elif col_name == "Network Traffic Analysis":
            record[col_name] = random.choice(["500GB data transfer detected overnight", "200GB transferred in last 24 hours"])
        elif col_name == "Security Incident Volume":
            record[col_name] = random.choice(["100 incidents in the last 30 days", "50 incidents reported in the last 7 days"])
        elif col_name == "Detection Time":
            record[col_name] = random.choice(["2 minutes", "5 minutes", "10 minutes"])
        elif col_name == "IT Security Budget":
            record[col_name] = f"${random.randint(100000, 1000000)}"
        elif col_name == "Competitor Security Solutions":
            record[col_name] = random.choice(["Legacy Firewall", "Outdated SIEM", "MDR Service"])
        elif col_name == "Vulnerability Scan Results":
            record[col_name] = random.choice(["10 critical vulnerabilities found", "5 minor vulnerabilities found", "No vulnerabilities detected"])
        elif col_name == "System Performance Metrics":
            record[col_name] = random.choice(["CPU at 70% utilization during scans", "High load on server"])
        elif col_name == "Outlier Detection":
            record[col_name] = random.choice(["Sudden spike in failed logins", "Unusual behavior detected in traffic"])
        elif col_name == "Baseline Value":
            record[col_name] = random.choice(["5 failed logins per day", "100GB data transferred per week"])
        elif col_name == "Alert Status":
            record[col_name] = random.choice(["Investigating", "Resolved", "False positive"])
        elif col_name == "Compliance Violation Flags":
            record[col_name] = random.choice(["Non-compliant third-party integrations detected", "No compliance violations"])
        elif col_name == "Incident Severity Level":
            record[col_name] = random.choice(["High", "Medium", "Low"])
        elif col_name == "Service/Product ID":
            record[col_name] = f"FW-{random.randint(1000, 9999)}"
        elif col_name == "Security Service Pricing":
            record[col_name] = f"${random.randint(5000, 50000)} per year"
        elif col_name == "Subscription Tiers":
            record[col_name] = random.choice(["Basic", "Advanced", "Enterprise"])
        elif col_name == "Proposal ID":
            record[col_name] = f"PROP-{random.randint(10000, 99999)}"
        elif col_name == "Service Agreement Terms":
            record[col_name] = random.choice(["24/7 Monitoring", "Threat Mitigation", "Incident Response"])
        elif col_name == "Compliance Approvals":
            record[col_name] = random.choice(["ISO 27001", "SOC 2", "GDPR"])
        elif col_name == "Product Catalog":
            record[col_name] = random.choice(["SIEM", "EDR", "MDR", "Firewalls"])
        elif col_name == "Expiry Date":
            record[col_name] = fake.date_this_year()
        elif col_name == "Order ID":
            record[col_name] = f"OR-{random.randint(10000, 99999)}"
        elif col_name == "Invoice ID":
            record[col_name] = f"INV-{random.randint(10000, 99999)}"
        elif col_name == "Payment Due Date":
            record[col_name] = fake.date_this_year()
        elif col_name == "Outstanding Balance":
            record[col_name] = f"${random.randint(0, 20000)}"
        elif col_name == "Preferred Communication Channel":
            record[col_name] = random.choice(["Email", "Phone", "SMS"])
        elif col_name == "Claims Adjudication Status":
            record[col_name] = random.choice(["Approved", "Denied", "Pending"])
        elif col_name == "Payment Processing Status":
            record[col_name] = random.choice(["Pending", "Completed", "Failed"])
        elif col_name == "Revenue Cycle Management Metrics":
            record[col_name] = f"{random.randint(70, 100)}% on-time payments"
        elif col_name == "Delayed Payment Trends":
            record[col_name] = random.choice(["3 late payments last year", "No delays", "1 late payment in the past 6 months"])
        elif col_name == "Email Open Rate":
            record[col_name] = f"{random.randint(50, 100)}%"
        elif col_name == "Chat Interaction Data":
            record[col_name] = random.choice(["Inquiry about SIEM deployment", "Requested demo of firewall product"])
        elif col_name == "Support Requests":
            record[col_name] = random.choice(["12 support tickets in the last 6 months", "No support tickets"])
        elif col_name == "Billing Amount":
            record[col_name] = f"${random.randint(5000, 20000)} per month"
        elif col_name == "Fraudulent Charge Detection":
            record[col_name] = random.choice(["Duplicate Invoice Flag Raised", "No fraudulent activities detected"])
        elif col_name == "IP Address":
            record[col_name] = f"192.168.1.{random.randint(1, 255)} (Suspicious Region)"
        elif col_name == "Past Payment Trends":
            record[col_name] = random.choice(["3 late payments last year", "Paid on time every month"])
        elif col_name == "Customer Lifetime Value (CLV)":
            record[col_name] = f"${random.randint(100000, 500000)}"
        elif col_name == "Unusual Login Activity":
            record[col_name] = random.choice(["Multiple failed login attempts", "Login from foreign IP"])
        elif col_name == "Threat Detection Data":
            record[col_name] = random.choice(["10 detected threats", "5 potential risks detected"])
        elif col_name == "Incident Logs":
            record[col_name] = random.choice(["1 critical incident last month", "No incidents"])
        elif col_name == "Intrusion Detection Data":
            record[col_name] = random.choice(["Unauthorized access attempt detected", "No intrusion detected"])
        elif col_name == "Risk Indicators":
            record[col_name] = random.choice(["High", "Moderate", "Low"])
        elif col_name == "Incident ID":
            record[col_name] = f"INC-{random.randint(10000, 99999)}"
        elif col_name == "Average Incident Response Time":
            record[col_name] = f"{random.randint(5, 30)} minutes"
        elif col_name == "Threat Score":
            record[col_name] = random.choice(["High", "Medium", "Low"])
        elif col_name == "Security Policy Compliance":
            record[col_name] = random.choice(["Compliant", "Non-Compliant"])
        elif col_name == "Patch Deployment Rate":
            record[col_name] = f"{random.randint(80, 100)}%"
        elif col_name == "Incident Recovery Time":
            record[col_name] = f"{random.randint(30, 180)} minutes"
        elif col_name == "Security Patch Deployment Logs":
            record[col_name] = f"{random.randint(1, 5)} patches deployed"
        elif col_name == "Patch Success Rate":
            record[col_name] = f"{random.randint(85, 100)}%"
        elif col_name == "Vulnerability Reduction Metrics":
            record[col_name] = f"Reduction of {random.randint(30, 50)}% in vulnerabilities"
        elif col_name == "SOC Efficiency Rating":
            record[col_name] = random.choice(["Excellent", "Good", "Average", "Poor"])
        elif col_name == "Automated Response Rate":
            record[col_name] = f"{random.randint(50, 100)}%"
        elif col_name == "Compliance Audit Logs":
            record[col_name] = random.choice(["Passed audit", "Failed audit"])
        elif col_name == "Security Policy Score":
            record[col_name] = f"{random.randint(60, 100)}%"
        elif col_name == "Regulatory Adherence Data":
            record[col_name] = random.choice(["Fully Adhered", "Non-Compliant"])
        elif col_name == "Prior Engagement":
            record[col_name] = random.choice(["Demo requested", "No prior engagements", "Consultation booked"])
        elif col_name == "Proposal Acceptance Rate":
            record[col_name] = f"{random.randint(30, 100)}%"
        elif col_name == "Industry Vertical":
            record[col_name] = random.choice(["Finance", "Healthcare", "Retail", "Technology", "Manufacturing"])
        elif col_name == "Alert ID":
            record[col_name] = f"ALERT-{random.randint(10000, 99999)}"
        elif col_name == "Client Engagement Rate":
            record[col_name] = f"{random.randint(50, 100)}%"
        elif col_name == "Security Alert Open Rate":
            record[col_name] = f"{random.randint(50, 100)}%"
        elif col_name == "Compliance Audit Score":
            record[col_name] = f"{random.randint(60, 100)}%"
        elif col_name == "Security Policy Implementation Rate":
            record[col_name] = f"{random.randint(50, 100)}%"
        elif col_name == "Company Size":
            record[col_name] = random.choice(["Small", "Medium", "Large"])
        elif col_name == "Service Level":
            record[col_name] = random.choice(["Basic", "Standard", "Premium"])
        elif col_name == "Contract Length":
            record[col_name] = random.choice(["1 Year", "2 Years", "3 Years"])
        elif col_name == "Discount Recommendation":
            record[col_name] = random.choice(["10% Discount", "15% Discount", "No Discount"])
        elif col_name == "Endpoint Count":
            record[col_name] = random.randint(10, 1000)
        elif col_name == "Data Scanned":
            record[col_name] = f"{random.randint(10, 100)}GB"
        elif col_name == "Monthly Service Utilization":
            record[col_name] = f"{random.randint(70, 100)}%"
        elif col_name == "Subscription Renewal Rate":
            record[col_name] = f"{random.randint(70, 100)}%"
        elif col_name == "Support Ticket Resolution Score":
            record[col_name] = f"{random.randint(80, 100)}%"
        
        
    # Append the generated record to the list of rows
    rows.append(record)

# Convert the list of rows into a DataFrame
df = pd.DataFrame(rows)

# Save the DataFrame to a CSV file
df.to_csv("synthetic_cybersecurity_L2C_data.csv", index=False)

print("Data generation complete. Saved as synthetic_cybersecurity_L2C_data.csv")

