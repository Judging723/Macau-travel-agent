from eval.run_plan_eval import evaluate_plan


def make_case() -> dict:
    return {
        "trip": {
            "start_date": "2026-10-01",
            "end_date": "2026-10-01",
            "budget": 1000,
        },
        "expected": {
            "max_activities_per_day": 4,
            "required_item_types": ["activity", "food"],
            "required_concepts": [["澳门博物馆"]],
            "forbidden_terms": ["上海", "南京"],
        },
    }


def make_response() -> dict:
    return {
        "trip": {
            "itinerary": [
                {
                    "day_number": 1,
                    "travel_date": "2026-10-01",
                    "title": "澳门历史城区",
                    "items": [
                        {
                            "type": "activity",
                            "description": "参观澳门博物馆",
                            "cost": 100,
                        },
                        {
                            "type": "food",
                            "description": "品尝澳门本地美食",
                            "cost": 50,
                        },
                    ],
                }
            ],
        },
        "summary": "澳门博物馆文化一日游",
        "budget_analysis": ("预计总费用为150元，总预算为1000元，剩余850元。"),
        "total_cost": 150,
    }


def test_evaluate_plan_accepts_consistent_budget_fields() -> None:
    checks = evaluate_plan(make_response(), make_case())

    assert checks["summary_cost"] is True
    assert checks["budget_text"] is True


def test_evaluate_plan_rejects_conflicting_budget_fields() -> None:
    response = make_response()
    response["summary"] = "澳门一日游，预计花费800元"
    response["budget_analysis"] = "预计总费用为800元。"

    checks = evaluate_plan(response, make_case())

    assert checks["summary_cost"] is False
    assert checks["budget_text"] is False
