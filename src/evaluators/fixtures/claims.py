MOTOR_VEHICLE_CLAIM_INPUT = (
    "Car accident on the highway, minor injuries reported, non-life-threatening."
)

PROPERTY_CLAIM_INPUT = (
    "Water damage in the basement due to a burst pipe, urgent repair required."
)

PUBLIC_LIABILITY_CLAIM_INPUT = "Slip and fall incident at a public park, minor injuries, requires urgent attention."

OTHER_CLAIM_INPUT = "Lost luggage during travel, replacement of essential items needed."

TEST_CLAIMS = [
    {
        "input": MOTOR_VEHICLE_CLAIM_INPUT,
        "expected_claim_type": "motor_vehicle",
        "expected_urgency": "high",
        "expected_policy_findings": "Motor vehicle claims require a police report if damage exceeds $2500, and the standard excess is $650. Coverage includes third-party property damage.",
        "expected_recommendation": "A police report is required if the damage exceeds $2500.",
    },
    {
        "input": PROPERTY_CLAIM_INPUT,
        "expected_claim_type": "property",
        "expected_urgency": "high",
        "expected_policy_findings": "Property claims require photos and repair quotes within 30 days.",
        "expected_recommendation": "Request submission of photos and repair quotes within the 30-day window to proceed with the claim.",
    },
    {
        "input": PUBLIC_LIABILITY_CLAIM_INPUT,
        "expected_claim_type": "public_liability",
        "expected_urgency": "high",
        "expected_policy_findings": "Public liability claims require an incident report and witness statements within 14 days.",
        "expected_recommendation": "Ensure that the incident report and witness statements are submitted within the 14-day timeframe to proceed with the claim.",
    },
    {
        "input": OTHER_CLAIM_INPUT,
        "expected_claim_type": "other",
        "expected_urgency": "medium",
        "expected_policy_findings": "Other claims do not fall under standard categories and require manual review by a senior assessor.",
        "expected_recommendation": "Manual review by a senior assessor is required.",
    },
]
