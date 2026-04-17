# Mediascope — AI Business SPB Hackathon 2026

Классифицируйте поисковые запросы по трём атрибутам:

- **TypeQuery** (бинарный) — относится ли запрос к профессиональному видеоконтенту
- **ContentType** — `фильм`, `сериал`, `мультфильм`, `мультсериал`, `прочее` или пусто
- **Title** — нормализованное название франшизы или пусто

Тренировочные данные содержат пользовательские запросы с опечатками, транслитерацией и обобщёнными формулировками. Тестовые данные командам недоступны — вы отправляете код, который запускается в серверной песочнице.

**Платформа:** https://app.ai-business-spb.ru

## Метрика

```
typequery_f2         = F_beta(y_true, y_pred, beta=2)                   # по всем строкам
contenttype_macro_f1 = macro F1 по 6 классам                            # только GT TypeQuery=1
title_token_f1       = среднее token-level F1 на bag-of-words           # только GT TypeQuery=1
combined_score       = 0.35 * typequery_f2 + 0.30 * contenttype_macro_f1 + 0.35 * title_token_f1
```

`combined_score` — ключ лидерборда.

## Быстрый старт

```bash
uv sync
cp .env.example .env
# впишите API_KEY из личного кабинета на app.ai-business-spb.ru
```

## Загрузка данных

### Через скрипт

```bash
uv run scripts/download_data.py
```

### Вручную через API

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
     https://data.ai-business-spb.ru/data/mediascope/train.csv \
     -o data/train.csv
```

Или через Python:

```python
import requests

headers = {"X-API-Key": "YOUR_API_KEY"}
r = requests.get("https://data.ai-business-spb.ru/data/mediascope/train.csv", headers=headers)
with open("data/train.csv", "wb") as f:
    f.write(r.content)
```

## Формат данных

`train.csv` содержит столбцы: `QueryText`, `TypeQuery`, `Title`, `ContentType`.

## Интерфейс решения

В корне архива должен быть `solution.py` с классом `PredictionModel`:

```python
class PredictionModel:
    batch_size: int = 10  # опционально, по умолчанию 10

    def __init__(self) -> None:
        # вызывается один раз при старте
        ...

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        # df — батч из batch_size строк (последний может быть меньше)
        # df содержит столбец QueryText
        # вернуть DataFrame со столбцами: QueryText, TypeQuery (int 0/1), Title (str), ContentType (str)
        ...
```

Песочница вызывает `predict()` порциями по `batch_size` строк. Если вам нужна параллельная обработка (например, конкурентные запросы к LLM API) — реализуйте её внутри `predict()`.

## Запуск бейзлайна локально

```bash
uv run python -c "
import pandas as pd
from solution import PredictionModel
df = pd.read_csv('data/train.csv').head(20)
model = PredictionModel()
print(model.predict(df[['QueryText']]))
"
```

Базовое решение в `solution.py` предсказывает `TypeQuery=0`, `Title=""`, `ContentType="other"` для всех строк — тривиальный старт.

## Отправка решения

### Через веб-интерфейс

Зайдите на https://app.ai-business-spb.ru, перейдите в раздел «Отправка» и загрузите архив.

### Через скрипт

```bash
uv run scripts/submit.py
```

### Вручную через API

```bash
curl -X POST \
     -H "X-API-Key: YOUR_API_KEY" \
     -F "file=@bundle.zip" \
     https://app.ai-business-spb.ru/api/mediascope/submissions
```

## Формат submission-архива

Архив (`bundle.zip`) должен содержать:

- `solution.py` в корне с классом `PredictionModel` (см. выше)
- Любые вспомогательные Python-модули и веса моделей рядом с `solution.py`
- `data/`, `.venv/`, `.git/`, `notebooks/`, `scripts/` автоматически исключаются

Лимиты песочницы: 6 CPU, 48 GB RAM, 1 × NVIDIA RTX 4090 (24 GB VRAM), 10 мин, сетевой доступ только к разрешённым LLM API (Yandex Cloud, OpenAI, Anthropic).

## Формат сдачи

```
Формат сдачи:
– Презентации должны открываться по ссылке
– Код загружен в публичный Git-репозиторий и открывается по ссылке (коммиты после дедлайна не принимаются)
– В ReadMe — минимальная документация: структура кода, зависимости, инструкция по деплою
– По кейсам с лидербордом — загружены и выбраны итоговые решения
```

## Активация промокода Яндекс

```
Активация промокода Яндекс:
Для активации промокода необходимо:

1) Перейти по ссылке https://center.yandex.cloud/
2) Перейти в раздел Billing
3) Нажать кнопку Активировать промокод
4) Активировать промокод

+ будет ссылка на инструкцию по активации биллинг аккаунта.
```
