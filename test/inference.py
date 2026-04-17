import pandas as pd

from solution import PredictionModel

train_path = 'train.csv'

try:
    df_train = pd.read_csv(train_path)
except Exception as e:
    print(f"Ошибка загрузки: {e}")

model = PredictionModel()

predictions_df = model.predict(df_train)
print(predictions_df.head(5))