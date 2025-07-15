from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import List

router = APIRouter()

# ========== SOW Estimator (SOW Internal Estimation tab) ==========

class SOWInput(BaseModel):
    ecc_version: float
    ewm_version: float
    enhancements: int
    test_cases: str
    customer_rating: str
    corrections: float
    configuration: float

def map_to_output_value(category: str, value):
    if category == "enhancements":
        if value <= 15:
            return "Low"
        elif value <= 40:
            return "Medium"
        elif value <= 100:
            return "High"
        elif value <= 130:
            return "Very high"
        else:
            return "Extremely High"
    elif category == "ecc_version":
        if value < 3.7:
            return "Bad"
        elif value <= 4.1:
            return "Average"
        else:
            return "Good"
    elif category == "ewm_version":
        return "Average" if value <= 2.1 else "Good"
    elif category == "test_cases":
        return value.capitalize()
    elif category == "customer_rating":
        return value.capitalize()
    return value

def get_ratio(category: str, output_value: str):
    ratios = {
        ("Number of enhancement", "Low"): 1,
        ("Number of enhancement", "Medium"): 1.2,
        ("Number of enhancement", "High"): 1.3,
        ("Number of enhancement", "Very high"): 1.5,
        ("Number of enhancement", "Extremely High"): 2,
        ("Test cases", "Yes"): 1,
        ("Test cases", "No"): 1.5,
        ("Customer Rating", "Bad"): 1.5,
        ("Customer Rating", "Average"): 1.2,
        ("Customer Rating", "Good"): 1,
        ("ShipERP Version ECC", "Bad"): 1.5,
        ("ShipERP Version ECC", "Average"): 1.2,
        ("ShipERP Version ECC", "Good"): 1,
        ("ShipERP Version EWM", "Average"): 1.2,
        ("ShipERP Version EWM", "Good"): 1,
    }
    return ratios.get((category, output_value), 1)

@router.post("/sow-estimate")
async def estimate_sow(request: Request, input_data: SOWInput):
    enhancement_level = map_to_output_value("enhancements", input_data.enhancements)
    ecc_level = map_to_output_value("ecc_version", input_data.ecc_version)
    ewm_level = map_to_output_value("ewm_version", input_data.ewm_version)
    test_case_level = map_to_output_value("test_cases", input_data.test_cases)
    rating_level = map_to_output_value("customer_rating", input_data.customer_rating)

    enhancement_ratio = get_ratio("Number of enhancement", enhancement_level)
    ecc_ratio = get_ratio("ShipERP Version ECC", ecc_level)
    ewm_ratio = get_ratio("ShipERP Version EWM", ewm_level)
    test_case_ratio = get_ratio("Test cases", test_case_level)
    rating_ratio = get_ratio("Customer Rating", rating_level)

    ratio_sum = sum([ecc_ratio, ewm_ratio, enhancement_ratio, test_case_ratio, rating_ratio])
    ratio_from = ratio_sum / 4.5
    ratio_to = ratio_sum / 3

    b14 = input_data.corrections
    b15 = input_data.configuration

    b25 = b14 * ratio_from
    b26 = b15 * ratio_from
    b27 = b15 * 0.3 * ratio_from
    b28 = (b14 + b15) * 0.4 * ratio_from
    b29 = (b14 + b15) * 0.4 * ratio_from
    b30 = sum([b25, b26, b27, b28, b29]) * 0.2
    b_total = sum([b25, b26, b27, b28, b29, b30])

    c25 = b14 * ratio_to
    c26 = b15 * ratio_to
    c27 = b15 * 0.3 * ratio_to
    c28 = (b14 + b15) * 0.4 * ratio_to
    c29 = (b14 + b15) * 0.4 * ratio_to
    c30 = sum([c25, c26, c27, c28, c29]) * 0.2
    c_total = sum([c25, c26, c27, c28, c29, c30])

    return {
        "from": round(b_total),
        "to": round(c_total),
        "details": {
            "Development Corrections&Patch Application": [round(b25), round(c25)],
            "Configuration": [round(b26), round(c26)],
            "Unit Test": [round(b27), round(c27)],
            "SIT & UAT Support": [round(b28), round(c28)],
            "Go Live & Hypercare": [round(b29), round(c29)],
            "PM hours": [round(b30), round(c30)]
        }
    }


# ========== New Carrier Estimator (New Carrier tab) ==========

class NewCarrierEstimateRequest(BaseModel):
    carrierName: str
    sapVersion: str
    abapVersion: str
    zEnhancements: int
    onlineOrOffline: str
    features: List[str]
    systemUsed: List[str]
    shipmentScreens: List[str]
    serpcarUsage: str
    shipFrom: List[str]
    shipTo: List[str]
    shipToVolume: str
    shiperpVersion: str
    shipmentScreenString: str

@router.post("/estimate/new_carrier")
async def estimate_new_carrier(data: NewCarrierEstimateRequest):
    line_19 = 60 if len(data.features) == 0 else 40

    if data.onlineOrOffline.lower() == "online":
        line_20 = 0 if data.zEnhancements == 0 else 40  # Adjust value if needed
    else:
        line_20 = 0

    line_21 = -16 if data.serpcarUsage.lower() == "yes" else 0

    shipto_map = {
        "less than 10": 0,
        "between 10 and 50": 16,
        "more than 50": 32,
        "i'm not sure": 8
    }
    line_22 = shipto_map.get(data.shipToVolume.strip().lower(), 0)

    line_23 = 0 if data.onlineOrOffline.lower() == "online" else -32

    line_24 = 16 if len(data.shipFrom) > 3 else 0

    version_map = {
        "above 4.5": 0,
        "between 4.0 and 4.5": 0,
        "between 3.6 and 3.9": 8,
        "lower than 3.6": 12
    }
    line_26 = version_map.get(data.shiperpVersion.strip().lower(), 0)

    comma_count = data.shipmentScreenString.count(",")
    line_28 = 8 if comma_count >= 1 else 0

    total_effort = sum([
        line_19, line_20, line_21, line_22, line_23,
        line_24, 0, line_26, 0, line_28
    ])

    return {
        "total_effort": max(0, round(total_effort)),
        "details": {
            "E19_Features": line_19,
            "E20_Enhancements_Online": line_20,
            "E21_SERPCAR": line_21,
            "E22_ShipToVolume": line_22,
            "E23_OnlineImpact": line_23,
            "E24_ShipFromCount": line_24,
            "E26_ShipERPVersion": line_26,
            "E28_ShipmentScreensCount": line_28
        },
        "input_snapshot": data.dict()
    }
