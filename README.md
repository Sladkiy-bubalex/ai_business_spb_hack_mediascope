# Решение команды ДКАРТ — AI Business SPB Hack 2026

**Трек:** Mediascope — классификация поисковых запросов

---

## Задача

По поисковому запросу пользователя определить три поля:

| Поле | Описание | Значения |
|------|----------|---------|
| **TypeQuery** | Ищет ли пользователь видеоконтент | `0` / `1` |
| **ContentType** | Тип контента | `фильм` / `сериал` / `мультфильм` / `мультсериал` / `прочее` / _пусто_ |
| **Title** | Название тайтла (название франшизы) | строка |

**Метрика:** `0.35 × TypeQuery_F2 + 0.30 × ContentType_macroF1 + 0.35 × Title_tokenF1`

**Результат на тестовой выборке:** `0.7206`

---

## Зависимости

**Python:** 3.9+

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
joblib>=1.3
rapidfuzz>=3.0
openai>=1.0
```

Установка:

```bash
pip install pandas numpy scikit-learn joblib rapidfuzz openai
```

---

## Структура архива

```
mediahack-8/
├── solution.py          # Основной файл — классы TextEnsemble и PredictionModel
├── llm_config.json      # Конфигурация API Yandex AI Studio
├── README.md            # Этот файл
└── models/
    ├── ens_type.pkl      # Обученный ансамбль для предсказания TypeQuery
    ├── ens_content.pkl   # Обученный ансамбль для предсказания ContentType
    ├── le_content.pkl    # LabelEncoder для классов ContentType
    ├── noise_words.pkl   # Список стоп-слов для извлечения тайтла
    └── titles_dict.json  # Словарь тайтлов (~35 000 записей из KinoPoisk + train)
```

---

## Как запустить

```python
from solution import PredictionModel
import pandas as pd

# Инициализация — загружает модели, словарь и LLM-клиент
model = PredictionModel("models")

# Входные данные — DataFrame с колонкой QueryText
df = pd.DataFrame({
    "QueryText": [
        "гарри поттер смотреть онлайн",
        "погода в москве",
        "наруто 5 сезон все серии",
        "garri potter smotret online",
    ]
})

# Предсказание
result = model.predict(df)
print(result)
# Колонки: QueryText, TypeQuery, Title, ContentType
```

---

## Внутренние составляющие

### 1. TextEnsemble (ML-ансамбль)

Три TF-IDF модели с разными конфигурациями, усреднение вероятностей:

| № | Векторизация | Классификатор |
|---|-------------|---------------|
| 1 | word n-gram (1,2), max 50 000 фич | LogisticRegression, C=1.5, balanced |
| 2 | word n-gram (2,4), max 30 000 фич | SGDClassifier, log_loss, balanced |
| 3 | char_wb n-gram (3,5), max 40 000 фич | LinearSVC в CalibratedClassifierCV |

Используется дважды: для **TypeQuery** (бинарная классификация с порогом) и **ContentType** (6 классов).

### 2. RapidFuzz — поиск тайтла в словаре

Последовательный поиск по словарю KinoPoisk (~35 000 алиасов):

1. Точное совпадение после нормализации (ё→е, удаление шума, стоп-фраз, годов)
2. `token_sort_ratio ≥ 72` — устойчив к перестановкам слов
3. `token_set_ratio ≥ 78` — ловит вхождения (минимум 2 слова в алиасе)
4. **Fallback:** longest group — разбивает запрос на группы по стоп-словам, берёт наибольшую; также извлекает текст из кавычек («...» / "...")

### 3. KinoPoisk override (без доп. запросов)

Корректировка ContentType на основе типа из словаря:
- ML говорит `фильм/сериал`, KP говорит `мультфильм/мультсериал` → берём KP
- ML говорит `null/NaN`, KP знает тип → берём KP

### 4. YandexGPT 5 Pro — LLM routing

LLM вызывается в двух случаях:

| Условие | Что делает LLM |
|---------|----------------|
| Fuzzy не нашёл тайтл в словаре | Извлекает Title + ContentType |
| ML уверенность (max proba) < 65% | Уточняет ContentType |

Запросы объединяются в батчи по 15, обрабатываются в 5 параллельных потоков.

**Что умеет LLM:**
- Транслитерация: `"garri potter"` → `"гарри поттер"`
- Опечатки: `"антниме наруто"` → `"наруто"`, мультсериал
- Реалити-шоу: `"взвешенные люди"` → прочее
- Нормализация франшизы: `"мстители финал"` → `"мстители"`
- Маппинг нестандартных типов: реалити-шоу / документальный / концерт → прочее; аниме → мультсериал

---

## Прогресс метрик

| Версия | TypeQuery F2 | ContentType F1 | Title F1 | Итог |
|--------|-------------|----------------|----------|------|
| ML ансамбль только | 0.9645 | 0.5069 | 0.6208 | 0.7069 |
| + LLM (yandexgpt-lite) для Title | 0.9645 | 0.5060 | 0.6434 | 0.7145 |
| + YandexGPT 5 Pro + confidence routing | 0.9645 | 0.5172 | 0.6509 | **0.7206** |
