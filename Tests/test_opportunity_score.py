import os
import sys

import pandas as pd
import pytest

# Add root project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Code.Health_Care.Opportunity_Score import (
    calculate_opportunity_score,
    map_demographics,
    map_healthcare_needs,
    map_insurance_provider,
    map_prior_appointment_history,
    map_referral_trends,
)

# ---------------- Test Mapping Functions ----------------


def test_map_demographics_valid():
    assert map_demographics("Male, 30, California") == 45.0
    assert map_demographics("Female, 40, Texas") == 55.0


def test_map_demographics_invalid():
    assert map_demographics("Incomplete data") == 10
    assert map_demographics("") == 10


def test_map_healthcare_needs():
    assert map_healthcare_needs("Telemedicine Consultation") == 80
    assert map_healthcare_needs("MRI Scan") == 90
    assert map_healthcare_needs("Unknown") == 50


def test_map_prior_appointment_history():
    assert map_prior_appointment_history("3 previous visits") == 30
    assert map_prior_appointment_history("0 previous visits") == 0
    assert map_prior_appointment_history("Invalid text") == 0


def test_map_insurance_provider():
    assert map_insurance_provider("Blue Cross Blue Shield") == 90
    assert map_insurance_provider("Aetna") == 80
    assert map_insurance_provider("Unknown Insurance") == 50


def test_map_referral_trends():
    assert map_referral_trends("Increasing") == 90
    assert map_referral_trends("Decreasing") == 50
    assert map_referral_trends("Stable") == 70
    assert map_referral_trends("Other") == 60


# ---------------- Test Score Calculation ----------------


def test_calculate_opportunity_score_single():
    df = pd.DataFrame(
        [
            {
                "Patient Demographics": "Male, 30, California",
                "Healthcare Needs": "MRI Scan",
                "Prior Appointment History": "2 previous visits",
                "Insurance Provider": "Cigna",
                "Provider Referral Trends": "Stable",
            }
        ]
    )
    scores = calculate_opportunity_score(df)
    assert isinstance(scores, list)
    assert len(scores) == 1
    assert 0 <= scores[0] <= 100


def test_opportunity_score_range_multiple():
    df = pd.DataFrame(
        [
            {
                "Patient Demographics": "Male, 30, California",
                "Healthcare Needs": "MRI Scan",
                "Prior Appointment History": "2 previous visits",
                "Insurance Provider": "Cigna",
                "Provider Referral Trends": "Stable",
            },
            {
                "Patient Demographics": "Female, 50, New York",
                "Healthcare Needs": "Routine Checkup",
                "Prior Appointment History": "1 previous visits",
                "Insurance Provider": "Humana",
                "Provider Referral Trends": "Decreasing",
            },
        ]
    )
    scores = calculate_opportunity_score(df)
    assert len(scores) == 2
    assert all(0 <= s <= 100 for s in scores)
    assert scores[0] != scores[1]
