from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_PATH = PROJECT_ROOT / "data/eval/results/main_recommender_eval_results.csv"


def main():
    df = pd.read_csv(RESULTS_PATH)

    print("Total users:", len(df))
    print("Hit users:", int((df["hit_rate_at_10"] > 0).sum()))
    print("Hit rate:", round((df["hit_rate_at_10"] > 0).mean(), 4))

    print("\nRecommended count distribution:")
    print(df["recommended_count"].value_counts().sort_index())

    print("\nUsers with fewer than 10 recommendations:")
    fewer_than_10 = df[df["recommended_count"] < 10]
    print(len(fewer_than_10))

    print("\nAverage recommended_count:")
    print(round(df["recommended_count"].mean(), 4))

    print("\nSeen violation total:")
    print(int(df["seen_violation_count"].sum()))

    print("\nNDCG summary:")
    print(df["ndcg_at_10"].describe())

    print("\nRecall summary:")
    print(df["recall_at_10"].describe())


if __name__ == "__main__":
    main()
