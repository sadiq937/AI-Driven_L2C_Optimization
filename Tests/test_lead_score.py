import os
import sys

import pandas as pd
import pytest

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from Code.Health_Care.Lead_score import (
    calculate_lead_score,
    clean_conversion_probability,
    map_engagement_history,
    map_lead_source,
    map_referral_source,
    map_specialty,
    map_website_interactions,
)

# ---------------- Test mapping functions ----------------


def test_map_specialty():
    assert map_specialty("Cardiology") == 80
    assert map_specialty("Pediatrics") == 40
    assert map_specialty("Unknown") == 0


def test_map_engagement_history():
    assert map_engagement_history("Last 5 days active") == 10
    assert map_engagement_history("Last 3 days active") == 6
    assert map_engagement_history("Invalid text") == 0


def test_map_referral_source():
    assert map_referral_source("Doctor Referral") == 80
    assert map_referral_source("Direct Visit") == 40
    assert map_referral_source("Unknown Source") == 0


def test_map_website_interactions():
    assert map_website_interactions("Visited 4 pages") == 4
    assert map_website_interactions("Clicked 1 link") == 1
    assert map_website_interactions("N/A") == 0


def test_map_lead_source():
    assert map_lead_source("Referral") == 80
    assert map_lead_source("Digital Ad") == 70
    assert map_lead_source("Other") == 0


def test_clean_conversion_probability():
    assert clean_conversion_probability("80%") == 80.0
    assert clean_conversion_probability("35") == 35.0
    assert clean_conversion_probability(None) == 0


# ---------------- Test lead score logic ----------------


def test_calculate_lead_score_simple_case():
    data = pd.DataFrame(
        [
            {
                "Specialty": "Cardiology",
                "Engagement History": "Last 5 days active",
                "Referral Source": "Doctor Referral",
                "Website Interactions": "Visited 4 pages",
                "Lead Source": "Referral",
                "Conversion Probability": "90%",
            }
        ]
    )

    scores = calculate_lead_score(data)
    assert isinstance(scores, list)
    assert len(scores) == 1
    assert 0 <= scores[0] <= 100


def test_lead_score_range_multiple():
    data = pd.DataFrame(
        [
            {
                "Specialty": "Cardiology",
                "Engagement History": "Last 5 days active",
                "Referral Source": "Doctor Referral",
                "Website Interactions": "Visited 4 pages",
                "Lead Source": "Referral",
                "Conversion Probability": "90%",
            },
            {
                "Specialty": "Pediatrics",
                "Engagement History": "Last 1 days active",
                "Referral Source": "Direct Visit",
                "Website Interactions": "Visited 1 pages",
                "Lead Source": "Digital Ad",
                "Conversion Probability": "10%",
            },
        ]
    )
    scores = calculate_lead_score(data)
    assert all(0 <= score <= 100 for score in scores)
    assert scores[0] > scores[1]  # First one should be higher lead score
