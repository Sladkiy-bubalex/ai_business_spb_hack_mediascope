import pandas as pd


class PredictionModel:
    batch_size: int = 10

    def __init__(self) -> None:
        pass

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Получает батч строк с колонкой QueryText, возвращает DataFrame
        со столбцами QueryText, TypeQuery, Title, ContentType."""
        out = df[["QueryText"]].copy()
        out["TypeQuery"] = 0
        out["Title"] = ""
        out["ContentType"] = "other"
        return out
