"""
Клиент Yandex AI Studio (OpenAI-compatible API).

Endpoint: https://ai.api.cloud.yandex.net/v1
Модель:   gpt-oss-120b/latest
"""
import os
import re
import time
import json

import openai

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_VALID_CT   = {"фильм", "сериал", "мультфильм", "мультсериал"}
_MAX_TOKENS = 4000
_TIMEOUT    = 90

SYSTEM_PROMPT = """Ты — эксперт-классификатор поисковых запросов для системы Mediascope.

## Задача
Для каждого запроса в нумерованном списке определи 4 параметра и верни JSON-массив.

## Шаг 1: TypeQuery
ВАЖНО: при любом сомнении ставь TypeQuery=1 — пропуск видеозапроса хуже, чем лишнее срабатывание.

TypeQuery=1 если запрос содержит ХОТЯ БЫ ОДИН признак:
- Ключевые слова: смотреть, онлайн, скачать, hd, 4k, трейлер, озвучка, субтитры, бесплатно
- Структура контента: сезон, серия, эпизод, часть, глава, выпуск
- Жанры/форматы: аниме, дорама, документальный, мультфильм, мультсериал, короткометражка
- Название фильма/сериала/мультфильма с любым уточнителем (год, номер части, качество)
- Транслитерированные названия (garri potter, igra prestolov, vedьmak)
- Запросы "название + год" (мстители 2019, дюна 2021)
- Имя актёра, режиссёра или персонажа из кино/сериала (брэд питт, чан гык сок, джонни депп, киану ривз, хён бин)
- Корейские, японские, китайские имена и названия — вероятно дорама или аниме (TypeQuery=1, ContentType="сериал" или "мультсериал")

TypeQuery=0 ТОЛЬКО если запрос явно о другом:
- Транспорт и маршруты (троллейбус, расписание электричек, автобус)
- Погода и прогнозы
- Спорт (результаты матчей, расписание игр, трансляции спортивных событий)
- Новости и события
- Рецепты и кулинария
- Карты и навигация
- Курсы валют и финансы

## Шаг 2: ContentType (только если TypeQuery=1, иначе "")
- "фильм" — полнометражный, разовый просмотр, живые актёры, не серийный
- "сериал" — многосерийный, сезоны, живые актёры
- "мультфильм" — анимация полнометражная, не серийная (включая аниме-фильм)
- "мультсериал" — анимация серийная, сезоны (включая аниме-сериал)
- "" — тип неизвестен или запрос слишком общий

Подсказки для различения:
- слова "сезон"/"серия" + анимация/аниме → мультсериал
- одиночное название мультфильма без сезонов → мультфильм
- аниме без указания серийности → мультсериал (аниме по умолчанию серийное)

## Шаг 3: Title (только если TypeQuery=1, иначе "")
Извлеки название ФРАНШИЗЫ (не конкретной части):
- Убери: год, номер части/сезона/серии, слова смотреть/онлайн/скачать/hd/full/бесплатно/качество
- "гарри поттер и узник азкабана" → "гарри поттер" (франшиза, не конкретная часть)
- "мстители финал" → "мстители"
- "холодное сердце 2" → "холодное сердце"
- "кот в сапогах последнее желание" → "кот в сапогах"
- Транслит переводи в кириллицу: "garri potter" → "гарри поттер", "the witcher" → "ведьмак"
- Если название неизвестно или запрос общий → ""
- Всегда нижний регистр

## Примеры (few-shot)

### Не видеоконтент (TypeQuery=0)
"10 троллейбус ижевск" → {"TypeQuery":0,"ContentType":"","Title":"","confidence":0.99}
"расписание электричек москва" → {"TypeQuery":0,"ContentType":"","Title":"","confidence":0.99}
"лига чемпионов расписание матчей" → {"TypeQuery":0,"ContentType":"","Title":"","confidence":0.97}
"курс доллара сегодня цб" → {"TypeQuery":0,"ContentType":"","Title":"","confidence":0.99}
"погода в москве на неделю" → {"TypeQuery":0,"ContentType":"","Title":"","confidence":0.99}

### Фильм (TypeQuery=1, ContentType="фильм")
"гарри поттер и узник азкабана full hd" → {"TypeQuery":1,"ContentType":"фильм","Title":"гарри поттер","confidence":0.96}
"garri potter i uznik azkabana" → {"TypeQuery":1,"ContentType":"фильм","Title":"гарри поттер","confidence":0.91}
"мстители финал смотреть онлайн" → {"TypeQuery":1,"ContentType":"фильм","Title":"мстители","confidence":0.96}
"дюна часть вторая 2024 hd" → {"TypeQuery":1,"ContentType":"фильм","Title":"дюна","confidence":0.96}
"брат 2 смотреть" → {"TypeQuery":1,"ContentType":"фильм","Title":"брат","confidence":0.95}
"интерстеллар онлайн бесплатно" → {"TypeQuery":1,"ContentType":"фильм","Title":"интерстеллар","confidence":0.97}
"джонни депп фильмы" → {"TypeQuery":1,"ContentType":"фильм","Title":"джонни депп","confidence":0.95}
"смотреть фильмы 2025 онлайн" → {"TypeQuery":1,"ContentType":"фильм","Title":"","confidence":0.93}

### Сериал (TypeQuery=1, ContentType="сериал")
"1 сезон тьмы" → {"TypeQuery":1,"ContentType":"сериал","Title":"тьма","confidence":0.97}
"igra prestolov 8 sezon" → {"TypeQuery":1,"ContentType":"сериал","Title":"игра престолов","confidence":0.93}
"the witcher season 2" → {"TypeQuery":1,"ContentType":"сериал","Title":"ведьмак","confidence":0.90}
"дом дракона смотреть онлайн" → {"TypeQuery":1,"ContentType":"сериал","Title":"дом дракона","confidence":0.96}
"во все тяжкие все сезоны" → {"TypeQuery":1,"ContentType":"сериал","Title":"во все тяжкие","confidence":0.97}
"чан гык сок" → {"TypeQuery":1,"ContentType":"сериал","Title":"чан гык сок","confidence":0.90}
"хён бин" → {"TypeQuery":1,"ContentType":"сериал","Title":"хён бин","confidence":0.90}

### Мультфильм (TypeQuery=1, ContentType="мультфильм") — анимация, не серийная
"мультфильм холодное сердце 2 онлайн" → {"TypeQuery":1,"ContentType":"мультфильм","Title":"холодное сердце","confidence":0.95}
"унесённые призраками мультфильм" → {"TypeQuery":1,"ContentType":"мультфильм","Title":"унесённые призраками","confidence":0.95}
"кот в сапогах последнее желание" → {"TypeQuery":1,"ContentType":"мультфильм","Title":"кот в сапогах","confidence":0.94}
"король лев смотреть дисней" → {"TypeQuery":1,"ContentType":"мультфильм","Title":"король лев","confidence":0.96}
"как приручить дракона 3 онлайн" → {"TypeQuery":1,"ContentType":"мультфильм","Title":"как приручить дракона","confidence":0.94}
"мультик три богатыря" → {"TypeQuery":1,"ContentType":"мультфильм","Title":"три богатыря","confidence":0.95}
"иван царевич и серый волк смотреть" → {"TypeQuery":1,"ContentType":"мультфильм","Title":"иван царевич и серый волк","confidence":0.94}
"кунг фу панда 4 2024" → {"TypeQuery":1,"ContentType":"мультфильм","Title":"кунг фу панда","confidence":0.95}
"миньоны онлайн бесплатно" → {"TypeQuery":1,"ContentType":"мультфильм","Title":"миньоны","confidence":0.95}
"человек паук через вселенные" → {"TypeQuery":1,"ContentType":"мультфильм","Title":"человек паук","confidence":0.93}
"судзумэ закрывающая двери" → {"TypeQuery":1,"ContentType":"мультфильм","Title":"судзумэ закрывающая двери","confidence":0.92}
"элементарно пиксар смотреть" → {"TypeQuery":1,"ContentType":"мультфильм","Title":"элементарно","confidence":0.95}

### Мультсериал (TypeQuery=1, ContentType="мультсериал") — серийная анимация и аниме
"губка боб 3 сезон онлайн" → {"TypeQuery":1,"ContentType":"мультсериал","Title":"губка боб","confidence":0.95}
"наруто 5 сезон все серии" → {"TypeQuery":1,"ContentType":"мультсериал","Title":"наруто","confidence":0.98}
"атака титанов финальный сезон" → {"TypeQuery":1,"ContentType":"мультсериал","Title":"атака титанов","confidence":0.97}
"истребитель демонов 3 сезон" → {"TypeQuery":1,"ContentType":"мультсериал","Title":"истребитель демонов","confidence":0.97}
"магическая битва смотреть аниме" → {"TypeQuery":1,"ContentType":"мультсериал","Title":"магическая битва","confidence":0.97}
"ван пис все серии онлайн" → {"TypeQuery":1,"ContentType":"мультсериал","Title":"ван пис","confidence":0.98}
"блич тысячелетняя кровавая война" → {"TypeQuery":1,"ContentType":"мультсериал","Title":"блич","confidence":0.96}
"невероятные приключения джоджо" → {"TypeQuery":1,"ContentType":"мультсериал","Title":"невероятные приключения джоджо","confidence":0.95}
"леди баг и супер кот 5 сезон" → {"TypeQuery":1,"ContentType":"мультсериал","Title":"леди баг и супер кот","confidence":0.96}
"семь смертных грехов аниме" → {"TypeQuery":1,"ContentType":"мультсериал","Title":"семь смертных грехов","confidence":0.96}
"боевой континент смотреть" → {"TypeQuery":1,"ContentType":"мультсериал","Title":"боевой континент","confidence":0.94}
"my hero academia season 4" → {"TypeQuery":1,"ContentType":"мультсериал","Title":"моя геройская академия","confidence":0.93}

Ответь ТОЛЬКО валидным JSON-массивом, ровно столько элементов сколько запросов:
[{"TypeQuery":int,"ContentType":str,"Title":str,"confidence":float},...]"""


class YandexLLMClient:
    """Клиент Yandex AI Studio (OpenAI-compatible) для классификации поисковых запросов."""

    def __init__(
        self,
        api_key: str | None = None,
        folder_id: str | None = None,
        model: str = "gpt-oss-120b/latest",
    ) -> None:
        self.api_key   = api_key   or os.environ.get("YANDEX_API_KEY")
        self.folder_id = folder_id or os.environ.get("YANDEX_FOLDER_ID")
        if not self.api_key:
            raise ValueError("YANDEX_API_KEY not set. Add it to .env or pass as argument.")
        if not self.folder_id:
            raise ValueError("YANDEX_FOLDER_ID not set. Add it to .env or pass as argument.")
        self.model_uri = f"gpt://{self.folder_id}/{model}"
        self._client   = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://ai.api.cloud.yandex.net/v1",
            default_headers={"x-folder-id": self.folder_id},
            timeout=_TIMEOUT,
        )

    # ------------------------------------------------------------------

    def _request(self, queries: list[str], attempt: int = 0) -> list[dict]:
        numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(queries))
        try:
            response = self._client.chat.completions.create(
                model=self.model_uri,
                temperature=0.0,
                max_tokens=_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Классифицируй запросы:\n{numbered}"},
                ],
            )
        except openai.RateLimitError:
            if attempt < 4:
                time.sleep(2 ** attempt)
                return self._request(queries, attempt + 1)
            raise

        text = response.choices[0].message.content.strip()
        try:
            results = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON array in LLM response: {text[:200]}")
            results = json.loads(match.group())

        if len(results) != len(queries):
            raise ValueError(f"Expected {len(queries)} results, got {len(results)}")
        return results

    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(r: dict) -> tuple[int, str, str, float]:
        tq    = int(r.get("TypeQuery", 0))
        ct    = str(r.get("ContentType") or "").strip().lower()
        title = str(r.get("Title")       or "").strip().lower()
        conf  = float(r.get("confidence", 0.8))
        if tq not in (0, 1):
            tq = 0
        if ct not in _VALID_CT:
            ct = ""
        if tq == 0:
            ct    = ""
            title = ""
        return tq, ct, title, conf

    # ------------------------------------------------------------------

    def classify(self, queries: list[str]) -> list[tuple[int, str, str, float]]:
        """Классифицирует список запросов.

        Returns list of (TypeQuery, ContentType, Title, confidence).
        On failure returns safe defaults (0, "", "", 0.5).
        """
        try:
            raw = self._request(queries)
            return [self._normalize(r) for r in raw]
        except Exception as e:
            print(f"[YandexLLM] chunk failed ({e}), using defaults")
            return [(0, "", "", 0.5)] * len(queries)
