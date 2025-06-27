from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()

class EstimatorInput(BaseModel):
    customer_name: str
    ecc_version: float
    ewm_version: float
    enhancement_count: int
    test_cases: str
    customer_rating: str
    dev_corrections: float
    configuration: float


def map_ratio_enhancement(count):
    if count <= 15:
        return 1
    elif count <= 40:
        return 1.2
    elif count <= 100:
        return 1.3
    elif count <= 130:
        return 1.5
    else:
        return 2


def map_ratio_version(value, threshold):
    return 1 if value <= threshold else 1.2


def map_rating_multiplier(rating):
    return {
        "Bad": 1.5,
        "Average": 1.2,
        "Good": 1
    }.get(rating, 1)


def map_test_case_multiplier(value):
    return 1.5 if value.strip().lower() == "no" else 1


@router.post("/sow-estimate")
async def sow_estimate(input: EstimatorInput):
    # Mapping logic
    c4 = None  # Not used
    c5 = map_ratio_version(input.ecc_version, 4.1)
    c6 = map_ratio_version(input.ewm_version, 2.1)
    c7 = map_ratio_enhancement(input.enhancement_count)
    c8 = map_test_case_multiplier(input.test_cases)
    c9 = map_rating_multiplier(input.customer_rating)

    multiplier = c5 * c6 * c7 * c8 * c9

    result_rows = [
        {
            "task": "Development Corrections&Patch Application",
            "from_": input.dev_corrections * 1.5 * multiplier,
            "to": input.dev_corrections * 2.5 * multiplier
        },
        {
            "task": "Configuration",
            "from_": input.configuration * 1.5 * multiplier,
            "to": input.configuration * 2.1 * multiplier
        },
        {
            "task": "Unit Test",
            "from_": 3,
            "to": 5
        },
        {
            "task": "SIT & UAT Support",
            "from_": 7,
            "to": 10
        },
        {
            "task": "Go Live & Hypercare",
            "from_": 7,
            "to": 10
        },
        {
            "task": "PM hours",
            "from_": 7,
            "to": 10
        },
    ]

    total_from = sum(r["from_"] for r in result_rows)
    total_to = sum(r["to"] for r in result_rows)

    return {
        "calculated": {
            "C4": input.customer_name,
            "C5": c5,
            "C6": c6,
            "C7": c7,
            "C8": c8,
            "C9": c9
        },
        "result_rows": result_rows,
        "finalResult": f"{round(total_from)} to {round(total_to)} hours"
    }
