"""
Локальная оценка решения по метрикам хакатона.

Использование:
    uv run python evaluate.py                        # оценить solution.py на val-выборке
    uv run python evaluate.py --sample 500           # быстрая оценка на 500 строках
    uv run python evaluate.py --data other.csv       # оценить на другом файле
"""
import argparse
import sys
import io
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, fbeta_score
from sklearn.model_selection import train_test_split

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def token_f1(pred: str, true: str) -> float:
    p_tok = set(str(pred or "").lower().split())
    t_tok = set(str(true or "").lower().split())
    if not t_tok:
        return 1.0 if not p_tok else 0.0
    if not p_tok:
        return 0.0
    common = p_tok & t_tok
    prec = len(common) / len(p_tok)
    rec  = len(common) / len(t_tok)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def compute_metrics(df_true: pd.DataFrame, df_pred: pd.DataFrame) -> dict:
    # TypeQuery F2 — по всем строкам
    tq_f2 = fbeta_score(
        df_true["TypeQuery"], df_pred["TypeQuery"],
        beta=2, zero_division=0,
    )

    # ContentType macro F1 — только GT TypeQuery=1
    mask = df_true["TypeQuery"] == 1
    ct_f1 = f1_score(
        df_true.loc[mask, "ContentType"].fillna(""),
        df_pred.loc[mask, "ContentType"].fillna(""),
        average="macro", zero_division=0,
    )

    # Title token F1 — только GT TypeQuery=1
    title_scores = [
        token_f1(p, t)
        for p, t in zip(
            df_pred.loc[mask, "Title"].fillna(""),
            df_true.loc[mask, "Title"].fillna(""),
        )
    ]
    title_f1 = float(np.mean(title_scores)) if title_scores else 0.0

    combined = 0.35 * tq_f2 + 0.30 * ct_f1 + 0.35 * title_f1
    return {
        "typequery_f2": round(tq_f2, 4),
        "contenttype_macro_f1": round(ct_f1, 4),
        "title_token_f1": round(title_f1, 4),
        "combined_score": round(combined, 4),
        "n_total": len(df_true),
        "n_video": int(mask.sum()),
    }


def print_metrics(metrics: dict, elapsed: float) -> None:
    print("\n" + "=" * 50)
    print("  РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 50)
    print(f"  Строк всего:          {metrics['n_total']}")
    print(f"  Строк TypeQuery=1:    {metrics['n_video']}")
    print("-" * 50)
    print(f"  TypeQuery F2:         {metrics['typequery_f2']:.4f}  (вес 0.35)")
    print(f"  ContentType macro F1: {metrics['contenttype_macro_f1']:.4f}  (вес 0.30)")
    print(f"  Title token F1:       {metrics['title_token_f1']:.4f}  (вес 0.35)")
    print("-" * 50)
    print(f"  COMBINED SCORE:       {metrics['combined_score']:.4f}")
    print(f"  Время:                {elapsed:.1f}с")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="train.csv")
    parser.add_argument("--sample", type=int, default=None,
                        help="Оценить на N строках (по умолчанию — весь val)")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    from solution import PredictionModel

    print(f"Загружаем данные: {args.data}")
    df = pd.read_csv(args.data)
    df["ContentType"] = df["ContentType"].fillna("")
    df["Title"]       = df["Title"].fillna("")

    # Выделяем val-выборку (20%) — те же строки что при обучении
    _, df_val = train_test_split(df, test_size=0.2, random_state=args.seed,
                                 stratify=df["TypeQuery"])
    df_val = df_val.reset_index(drop=True)

    if args.sample:
        df_val = df_val.sample(min(args.sample, len(df_val)),
                               random_state=args.seed).reset_index(drop=True)

    print(f"Оцениваем на {len(df_val)} строках...")

    model = PredictionModel()
    results = []
    batch = model.batch_size
    t0 = time.time()

    for start in range(0, len(df_val), batch):
        chunk = df_val.iloc[start:start + batch][["QueryText"]].copy()
        pred  = model.predict(chunk)
        results.append(pred)
        done = min(start + batch, len(df_val))
        print(f"  {done}/{len(df_val)}...", end="\r")

    df_pred = pd.concat(results, ignore_index=True)
    elapsed = time.time() - t0

    metrics = compute_metrics(df_val, df_pred)
    print_metrics(metrics, elapsed)


if __name__ == "__main__":
    main()
