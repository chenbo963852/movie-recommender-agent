import json
from pathlib import Path

import requests


BASE_URL = "http://127.0.0.1:8000"
OUTPUT_PATH = Path("data/processed/api_smoke_test_results.json")


def test_get_search():
    url = f"{BASE_URL}/search"

    params = {
        "query": "mind bending science fiction thriller",
        "genre": "Science Fiction,Thriller",
        "min_vote_average": 7.0,
        "min_vote_count": 500,
        "top_k": 5,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    return response.json()


def test_post_structured_recommend():
    url = f"{BASE_URL}/recommend/structured"

    payload = {
        "query": "mind bending science fiction thriller",
        "genre": "Science Fiction,Thriller",
        "min_vote_average": 7.0,
        "min_vote_count": 500,
        "top_k": 5,
    }

    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()

    return response.json()


def test_post_agent_recommend():
    url = f"{BASE_URL}/agent/recommend"

    payload = {
        "message": "推荐5部2010年之后的高评分的科幻惊悚片，不要恐怖片，至少500人评分"
    }

    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()

    return response.json()


def test_post_agent_user_recommend():
    url = f"{BASE_URL}/agent/user-recommend"

    payload = {
        "user_id": 45811,
        "message": "推荐5部2010年之后的高评分科幻片，不要恐怖片，至少300人评分"
    }

    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()

    return response.json()


def summarize_result(name: str, result: dict):
    results = result.get("results", [])

    titles = [
        item.get("title")
        for item in results
    ]

    print("=" * 80)
    print(f"Test: {name}")
    print(f"Result count: {len(results)}")
    print(f"Titles: {titles}")

    if "agent_response" in result:
        print(f"Agent response: {result['agent_response']}")

    if "parsed_intent" in result:
        print(f"Parsed intent: {result['parsed_intent']}")

    if "attempts" in result:
        print(f"Attempts: {result['attempts']}")


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    tests = {
        "search": test_get_search,
        "structured_recommend": test_post_structured_recommend,
        "agent_recommend": test_post_agent_recommend,
        "agent_user_recommend": test_post_agent_user_recommend,
    }

    all_results = {}

    for name, func in tests.items():
        try:
            result = func()
            all_results[name] = {
                "success": True,
                "response": result,
            }
            summarize_result(name, result)

        except Exception as e:
            all_results[name] = {
                "success": False,
                "error": str(e),
            }

            print("=" * 80)
            print(f"Test: {name}")
            print(f"FAILED: {e}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"Saved smoke test results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
