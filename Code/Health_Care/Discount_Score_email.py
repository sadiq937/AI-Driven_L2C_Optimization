import mimetypes
import os
import smtplib
from email.message import EmailMessage
from email.utils import make_msgid

import pandas as pd

# Paths
DISCOUNT_PATH = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Data/healthcare_discount_score.csv"
SYNTHETIC_PATH = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Data/synthetic_healthcare_data.csv"
ASSETS_DIR = (
    "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Assets"
)

# Email settings
FROM_EMAIL = "testemailai2025@gmail.com"
EMAIL_PASSWORD = "pito xlcd aaqa xhfe"
TO_EMAIL = "totestingemail2025@gmail.com"  # test recipient

# Load data
discount_df = pd.read_csv(DISCOUNT_PATH)
main_df = pd.read_csv(SYNTHETIC_PATH)
eligible_df = discount_df[discount_df["Discount Score"] >= 40]
merged_df = pd.merge(eligible_df, main_df, on="Patient ID", how="inner")


# Choose image based on discount
def get_discount_image_filename(discount_text):
    if "20%" in discount_text:
        return "20_discount.png"
    elif "10%" in discount_text:
        return "10_discount.png"
    elif "5%" in discount_text:
        return "5_discount.png"
    else:
        return "default_banner.png"


# Email function
def send_discount_email(patient_row, recipient_email):
    name = patient_row.get("First Name", "Customer")
    discount = patient_row["Discount Recommendation"]
    image_file = get_discount_image_filename(discount)
    image_path = os.path.join(ASSETS_DIR, image_file)
    flyer_path = os.path.join(ASSETS_DIR, "Discount_Eligibility_Flyer.pdf")

    image_cid = make_msgid(domain="xyz.com")

    msg = EmailMessage()
    msg["Subject"] = f"🎉 Exclusive {discount} Just for You!"
    msg["From"] = FROM_EMAIL
    msg["To"] = recipient_email

    msg.set_content("You're eligible for a healthcare discount!")  # fallback

    html_content = f"""
    <html>
        <body>
            <p>Dear {name},</p>
            <p style="font-size: 16px;">
                We’re excited to inform you that you're eligible for a <strong>{discount}</strong> based on your engagement and history with our healthcare services.
            </p>
            <p>
                <img src="cid:{image_cid[1:-1]}" alt="Discount Banner" style="width:400px;height:auto;">
            </p>
            <p>
                You can apply this discount to your next appointment. Feel free to reach out if you have questions.
            </p>
            <p>📎 Please find the attached flyer for more details.</p>
            <br>
            <p>Best regards,<br><strong>L2O Healthcare Optimization Team</strong></p>
        </body>
    </html>
    """
    msg.add_alternative(html_content, subtype="html")

    # Attach discount image
    with open(image_path, "rb") as img:
        img_data = img.read()
        msg.get_payload()[1].add_related(
            img_data, "image", "png", cid=image_cid
        )

    # Attach flyer PDF
    with open(flyer_path, "rb") as f:
        file_data = f.read()
        maintype, subtype = mimetypes.guess_type(flyer_path)[0].split("/")
        msg.add_attachment(
            file_data,
            maintype=maintype,
            subtype=subtype,
            filename="Discount_Eligibility_Flyer.pdf",
        )

    # Send email
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(FROM_EMAIL, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print(f"Email sent to {recipient_email} for patient: {name}")
    except Exception as e:
        print(f"Failed to send email to {recipient_email}: {e}")


# Send only 10 test emails to fixed address
test_entries = merged_df.head(10)
for _, row in test_entries.iterrows():
    send_discount_email(row, TO_EMAIL)

print("Sent 10 test discount emails to TO_EMAIL.")
