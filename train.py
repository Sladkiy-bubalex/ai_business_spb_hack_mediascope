"""
Скрипт обучения локальной ML-модели.

Использование:
    uv run python train.py                        # полное обучение
    uv run python train.py --data new_data.csv    # обучение на новых данных
    uv run python train.py --incremental          # дообучение на новых данных

Генерирует: model.pkl, titles_dict.json
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


VALID_CT = {"фильм", "сериал", "мультфильм", "мультсериал", "прочее"}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["QueryText"] = df["QueryText"].fillna("").str.lower().str.strip()
    df["TypeQuery"] = df["TypeQuery"].fillna(0).astype(int)
    df["ContentType"] = df["ContentType"].fillna("").str.strip().str.lower()
    df["Title"] = df["Title"].fillna("").str.strip().str.lower()
    # Нормализуем ContentType
    df.loc[~df["ContentType"].isin(VALID_CT), "ContentType"] = ""
    return df


def build_pipeline() -> Pipeline:
    """TF-IDF (char + word n-grams) + LogisticRegression."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=100_000,
            sublinear_tf=True,
            min_df=1,
        )),
        ("clf", LogisticRegression(
            C=5.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])


def token_f1(pred: str, true: str) -> float:
    p_tok = set(str(pred or "").split())
    t_tok = set(str(true or "").split())
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


def evaluate(df_true: pd.DataFrame, df_pred: pd.DataFrame) -> dict:
    tq_f2 = fbeta_score(df_true["TypeQuery"], df_pred["TypeQuery"], beta=2, zero_division=0)
    mask = df_true["TypeQuery"] == 1
    ct_f1 = f1_score(
        df_true.loc[mask, "ContentType"].fillna(""),
        df_pred.loc[mask, "ContentType"].fillna(""),
        average="macro", zero_division=0,
    )
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
    }


def build_titles_dict(df: pd.DataFrame, output: str = "titles_dict.json") -> None:
    titles = (
        df.loc[df["Title"] != "", "Title"]
        .dropna()
        .str.lower()
        .str.strip()
        .unique()
        .tolist()
    )
    titles = sorted(set(t for t in titles if t))
    with open(output, "w", encoding="utf-8") as f:
        json.dump(titles, f, ensure_ascii=False, indent=2)
    print(f"titles_dict.json: {len(titles)} уникальных тайтлов -> {output}")


def train(data_path: str = "train.csv", output: str = "model.pkl",
          incremental: bool = False) -> None:
    print(f"Загружаем данные: {data_path}")
    df = load_data(data_path)
    print(f"  Строк: {len(df)}, TypeQuery=1: {df['TypeQuery'].sum()}")

    # Если инкрементальный режим — дозагружаем старые данные
    if incremental and Path("train.csv").exists() and data_path != "train.csv":
        old = load_data("train.csv")
        df = pd.concat([old, df], ignore_index=True).drop_duplicates("QueryText")
        print(f"  После merge с базовыми данными: {len(df)} строк")

    # Разбивка train/val
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42,
                                        stratify=df["TypeQuery"])

    # --- TypeQuery classifier ---
    print("\nОбучаем TypeQuery классификатор...")
    type_model = build_pipeline()
    type_model.fit(df_train["QueryText"], df_train["TypeQuery"])

    y_pred_tq = type_model.predict(df_val["QueryText"])
    tq_f2 = fbeta_score(df_val["TypeQuery"], y_pred_tq, beta=2, zero_division=0)
    print(f"  TypeQuery F2 (val): {tq_f2:.4f}")

    # --- ContentType classifier (только TypeQuery=1) ---
    print("\nОбучаем ContentType классификатор...")
    mask_train = df_train["TypeQuery"] == 1
    ct_model = build_pipeline()
    ct_model.fit(df_train.loc[mask_train, "QueryText"],
                 df_train.loc[mask_train, "ContentType"])

    mask_val = df_val["TypeQuery"] == 1
    y_pred_ct = ct_model.predict(df_val.loc[mask_val, "QueryText"])
    ct_f1 = f1_score(df_val.loc[mask_val, "ContentType"], y_pred_ct,
                     average="macro", zero_division=0)
    print(f"  ContentType macro F1 (val): {ct_f1:.4f}")

    # --- Сохраняем модель ---
    model_data = {
        "type_model": type_model,
        "ct_model": ct_model,
    }
    with open(output, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\nМодель сохранена: {output}")

    # --- Строим словарь тайтлов ---
    build_titles_dict(df)

    # --- Итоговая оценка (TypeQuery-only, без Title т.к. ML не извлекает Title) ---
    df_pred_val = df_val[["QueryText"]].copy()
    df_pred_val["TypeQuery"] = y_pred_tq
    df_pred_val["ContentType"] = ""
    df_pred_val.loc[mask_val, "ContentType"] = y_pred_ct
    df_pred_val["Title"] = ""  # ML не извлекает Title
    print("\n=== Метрики на val (без Title — ML не извлекает названия) ===")
    metrics = evaluate(df_val.reset_index(drop=True),
                       df_pred_val.reset_index(drop=True))
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="train.csv")
    parser.add_argument("--output", default="model.pkl")
    parser.add_argument("--incremental", action="store_true",
                        help="Дообучить на новых данных поверх train.csv")
    args = parser.parse_args()
    train(args.data, args.output, args.incremental)
