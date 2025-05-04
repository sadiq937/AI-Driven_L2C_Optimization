import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Code.Health_Care.Engagement_score import (
    calculate_engagement_score,
    extract_portal_usage,
    extract_support_requests,
    map_appointment_freq,
    map_telehealth,
)

# ---------- Test Mapping Functions ----------


def test_map_telehealth():
    assert map_telehealth("High") == 80
    assert map_telehealth("Medium") == 50
    assert map_telehealth("Low") == 20
    assert map_telehealth("Unknown") == 0


def test_map_appointment_freq():
    assert map_appointment_freq("Quarterly") == 80
    assert map_appointment_freq("2 per year") == 40
    assert map_appointment_freq("1 per year") == 20
    assert map_appointment_freq("Unknown") == 0


def test_extract_support_requests():
    assert extract_support_requests("3 requests submitted") == 3
    assert extract_support_requests("1 request") == 1
    assert extract_support_requests("No data") == 0


def test_extract_portal_usage():
    assert extract_portal_usage("Logged in 5 times") == 5
    assert extract_portal_usage("Logged in 10 times") == 10
    assert extract_portal_usage("Invalid string") == 0


# ---------- Test Engagement Score Calculation ----------


def test_calculate_engagement_score_valid():
    df = pd.DataFrame(
        [
            {
                "Telehealth Engagement": "High",
                "Appointment Frequency": "Quarterly",
                "Support Requests": "2 requests",
                "Patient Portal Usage": "Logged in 4 times",
            }
        ]
    )

    result = calculate_engagement_score(df.copy())
    assert "Engagement Score" in result.columns
    assert isinstance(result["Engagement Score"].iloc[0], float)
    assert 0 <= result["Engagement Score"].iloc[0] <= 100


def test_calculate_engagement_score_zero_case():
    df = pd.DataFrame(
        [
            {
                "Telehealth Engagement": "Unknown",
                "Appointment Frequency": "Unknown",
                "Support Requests": "N/A",
                "Patient Portal Usage": "N/A",
            }
        ]
    )
    result = calculate_engagement_score(df.copy())
    assert result["Engagement Score"].iloc[0] == 0.00
