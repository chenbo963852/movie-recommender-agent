import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService
from app.services.bm25_service import BM25Service
from app.services.movie_rating_stats_service import MovieRatingStatsService
from app.services.retrieval_service import RetrievalService
from app.services.intent_parser_service import IntentParserService
from app.services.agent_recommendation_service import AgentRecommendationService
from app.services.llm_service import LLMService
from config import llm_config


TEST_CASES_PATH = PROJECT_ROOT / "data/eval/agent_test_cases.json"
RESULTS_DIR = PROJECT_ROOT / "data/eval/results"

OUTPUT_SUMMARY = RESULTS_DIR / "agent_eval_summary.json"
OUTPUT_DETAILS = RESULTS_DIR / "agent_eval_details.json"
OUTPUT_COMPARISON = RESULTS_DIR / "agent_eval_comparison.json"  # LLM vs Rule 对比


def normalize_list(value):
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]

    return []


def list_match(predicted, expected):
    predicted_set = set(normalize_list(predicted))
    expected_set = set(normalize_list(expected))
    return predicted_set == expected_set


def scalar_match(predicted, expected):
    if expected is None:
        return predicted is None

    if predicted is None:
        return False

    try:
        if isinstance(expected, float):
            return float(predicted) == float(expected)

        if isinstance(expected, int):
            return int(predicted) == int(expected)
    except Exception:
        return False

    return predicted == expected


def movie_has_genres(movie, required_genres):
    required = normalize_list(required_genres)

    if not required:
        return True

    genres_text = movie.get("genres") or ""
    genres_text = genres_text.lower()

    return all(
        genre.lower() in genres_text
        for genre in required
    )


def movie_excludes_genres(movie, excluded_genres):
    excluded = normalize_list(excluded_genres)

    if not excluded:
        return True

    genres_text = movie.get("genres") or ""
    genres_text = genres_text.lower()

    return not any(
        genre.lower() in genres_text
        for genre in excluded
    )


def movie_satisfies_year(movie, year_from, year_to):
    year = movie.get("year")

    if year is None:
        return year_from is None and year_to is None

    try:
        year = int(year)
    except Exception:
        return year_from is None and year_to is None

    if year_from is not None and year < year_from:
        return False

    if year_to is not None and year > year_to:
        return False

    return True


def movie_satisfies_quality(movie, min_vote_average, min_vote_count):
    if min_vote_average is not None:
        vote_average = movie.get("vote_average")

        try:
            vote_average = float(vote_average)
        except Exception:
            return False

        if vote_average < float(min_vote_average):
            return False

    if min_vote_count is not None:
        vote_count = movie.get("vote_count")

        try:
            vote_count = int(vote_count)
        except Exception:
            return False

        if vote_count < int(min_vote_count):
            return False

    return True


def result_constraint_satisfaction(movie, expected):
    return (
        movie_has_genres(movie, expected.get("genre"))
        and movie_excludes_genres(movie, expected.get("exclude_genre"))
        and movie_satisfies_year(
            movie,
            expected.get("year_from"),
            expected.get("year_to"),
        )
        and movie_satisfies_quality(
            movie,
            expected.get("min_vote_average"),
            expected.get("min_vote_count"),
        )
    )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print("Loading services...")

    embedding_service = EmbeddingService()
    qdrant_service = QdrantService()

    bm25_service = BM25Service()
    if not bm25_service.load_cache_if_exists():
        raise RuntimeError("BM25 cache not found. Please rebuild BM25 index first.")

    movie_rating_stats_service = MovieRatingStatsService()
    movie_rating_stats_service.load_stats()

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        qdrant_service=qdrant_service,
        bm25_service=bm25_service,
        movie_rating_stats_service=movie_rating_stats_service,
    )

    intent_parser_service = IntentParserService()

    # 初始化 LLM 服务
    llm_service = LLMService(backend=llm_config.backend, config=llm_config.to_dict())

    # 规则 Agent（保持原有逻辑）
    rule_agent_service = AgentRecommendationService(
        llm_service=llm_service,
        retrieval_service=retrieval_service,
        intent_parser_service=intent_parser_service,
    )

    # LLM Agent（使用 Agent 循环 + RAG）
    llm_agent_service = AgentRecommendationService(
        llm_service=llm_service,
        retrieval_service=retrieval_service,
        intent_parser_service=intent_parser_service,
    )

    total = len(test_cases)

    field_names = [
        "genre", "exclude_genre", "year_from", "year_to",
        "min_vote_average", "min_vote_count", "top_k",
    ]

    def evaluate_one_agent(agent_service, agent_label: str, use_llm: bool):
        """评估单个 Agent。"""
        field_correct = {field: 0 for field in field_names}
        exact_param_correct = 0
        non_empty_count = 0
        fallback_used_count = 0
        fallback_success_count = 0
        total_result_count = 0
        satisfied_result_count = 0
        rag_generated_count = 0

        details = []

        for case in test_cases:
            case_id = case["id"]
            prompt = case["prompt"]
            expected = case["expected"]

            print(f"[{agent_label}] Evaluating case {case_id}: {prompt}")

            if use_llm:
                try:
                    response = agent_service.recommend_llm(prompt)
                except Exception as e:
                    print(f"  LLM agent failed: {e}, falling back to rule")
                    response = agent_service.recommend(prompt)
            else:
                response = agent_service.recommend(prompt)

            parsed = response.get("parsed_intent") or {}
            results = response.get("results") or []

            # 检查 RAG
            if response.get("rag_generated"):
                rag_generated_count += 1

            field_result = {}
            for field in field_names:
                if field in ["genre", "exclude_genre"]:
                    ok = list_match(parsed.get(field), expected.get(field))
                else:
                    ok = scalar_match(parsed.get(field), expected.get(field))
                field_result[field] = ok
                if ok:
                    field_correct[field] += 1

            if all(field_result.values()):
                exact_param_correct += 1

            if results:
                non_empty_count += 1

            fallback_used = bool(response.get("fallback_used"))
            if fallback_used:
                fallback_used_count += 1
            if fallback_used and results:
                fallback_success_count += 1

            result_checks = []
            for movie in results:
                ok = result_constraint_satisfaction(movie, expected)
                result_checks.append({
                    "id": movie.get("id"),
                    "title": movie.get("title"),
                    "genres": movie.get("genres"),
                    "year": movie.get("year"),
                    "vote_average": movie.get("vote_average"),
                    "vote_count": movie.get("vote_count"),
                    "satisfies_constraints": ok,
                    "agent_strategy": movie.get("agent_strategy"),
                })
                total_result_count += 1
                if ok:
                    satisfied_result_count += 1

            details.append({
                "id": case_id,
                "prompt": prompt,
                "expected": expected,
                "parsed_intent": parsed,
                "field_correct": field_result,
                "exact_param_match": all(field_result.values()),
                "fallback_used": fallback_used,
                "rag_generated": response.get("rag_generated", False),
                "llm_driven_loop": response.get("llm_driven_loop", False),
                "result_count": len(results),
                "constraint_satisfaction_count": sum(
                    1 for item in result_checks if item["satisfies_constraints"]
                ),
                "results": result_checks,
            })

        return {
            "agent_label": agent_label,
            "test_case_count": total,
            "exact_param_accuracy": round(exact_param_correct / total, 4),
            "field_accuracy": {
                field: round(field_correct[field] / total, 4) for field in field_names
            },
            "non_empty_result_rate": round(non_empty_count / total, 4),
            "fallback_used_rate": round(fallback_used_count / total, 4),
            "fallback_success_rate": (
                round(fallback_success_count / fallback_used_count, 4)
                if fallback_used_count else 0.0
            ),
            "constraint_satisfaction_rate": (
                round(satisfied_result_count / total_result_count, 4)
                if total_result_count else 0.0
            ),
            "rag_generated_rate": round(rag_generated_count / total, 4),
            "details": details,
        }

    # ─── 评估规则 Agent ───
    print("\n=== Evaluating Rule Agent ===\n")
    rule_summary = evaluate_one_agent(rule_agent_service, "rule_agent", use_llm=False)

    # ─── 评估 LLM Agent ───
    print("\n=== Evaluating LLM Agent ===\n")
    llm_summary = evaluate_one_agent(llm_agent_service, "llm_agent", use_llm=True)

    # ─── 对比 ───
    comparison = {
        "rule_agent": {k: v for k, v in rule_summary.items() if k != "details"},
        "llm_agent": {k: v for k, v in llm_summary.items() if k != "details"},
        "delta": {
            "exact_param_accuracy": round(
                llm_summary["exact_param_accuracy"] - rule_summary["exact_param_accuracy"], 4
            ),
            "non_empty_result_rate": round(
                llm_summary["non_empty_result_rate"] - rule_summary["non_empty_result_rate"], 4
            ),
            "constraint_satisfaction_rate": round(
                llm_summary["constraint_satisfaction_rate"] - rule_summary["constraint_satisfaction_rate"], 4
            ),
            "rag_enabled": llm_summary["rag_generated_rate"] > 0,
        },
    }

    # ─── 保存 ───
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(rule_summary, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_DETAILS, "w", encoding="utf-8") as f:
        json.dump(rule_summary["details"], f, ensure_ascii=False, indent=2)

    with open(OUTPUT_COMPARISON, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print("\n=== Rule Agent ===")
    print(json.dumps({k: v for k, v in rule_summary.items() if k != "details"}, ensure_ascii=False, indent=2))
    print("\n=== LLM Agent ===")
    print(json.dumps({k: v for k, v in llm_summary.items() if k != "details"}, ensure_ascii=False, indent=2))
    print("\n=== Comparison (LLM - Rule) ===")
    print(json.dumps(comparison["delta"], ensure_ascii=False, indent=2))

    print(f"\nSaved to: {OUTPUT_SUMMARY}, {OUTPUT_DETAILS}, {OUTPUT_COMPARISON}")


if __name__ == "__main__":
    main()
