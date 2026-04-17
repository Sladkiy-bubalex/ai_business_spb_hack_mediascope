import os
import re
import json
import requests
import pandas as pd
from dotenv import load_dotenv
from dataclasses import dataclass
from enum import Enum
from marmel_grammar import MarmelGrammar
from typing import Optional, Tuple, Dict, Any

load_dotenv()

# ============================================================================
# МОДЕЛИ ДАННЫХ
# ============================================================================


class ContentType(Enum):
    FILM = "фильм"
    SERIAL = "сериал"
    CARTOON = "мультфильм"
    CARTOON_SERIAL = "мультсериал"
    OTHER = "прочее"
    EMPTY = ""


@dataclass
class QueryAnalysis:
    original_query: str
    normalized_query: str
    type_query: bool
    content_type: ContentType
    franchise_title: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "type_query": self.type_query,
            "content_type": self.content_type.value,
            "franchise_title": self.franchise_title,
        }


# ============================================================================
# ЭТАП 1: НОРМАЛИЗАЦИЯ ТЕКСТА
# ============================================================================


class TextNormalizer:
    """
    Нормализация запроса: транслитерация, исправление опечаток, приведение к нормальной форме.
    Если библиотека marmel-grammar недоступна, используется упрощенная реализация.
    """

    _grammar = MarmelGrammar()

    @classmethod
    def normalize(cls, query: str) -> str:
        """Основной метод нормализации текста запроса."""
        if not query:
            return ""

        normalized = query.strip()

        # 1. Транслитерация с латиницы на кириллицу
        normalized = cls._transliterate(normalized)

        # 3. Удаление лишних пробелов
        normalized = re.sub(r"\s+", " ", normalized)

        return normalized

    @classmethod
    def _transliterate(cls, text: str) -> str:
        """Упрощенная транслитерация. Для production лучше использовать marmel-grammar."""
        # Проверяем, есть ли латинские символы
        if re.search(r"[a-zA-Z]", text):
            query = cls._grammar.transliterate_to_russian(text)
            return query.lower()

        return text.lower()


# ============================================================================
# ЭТАП 2: ЛОКАЛЬНЫЕ ПРАВИЛА И МАРКЕРЫ
# ============================================================================


class LocalRuleEngine:
    """
    Определение параметров по регулярным выражениям и словарям маркеров.
    Позволяет избежать вызова API для очевидных случаев.
    """

    # Маркеры для разных типов контента
    SERIAL_MARKERS = [
        r"сезон\s*\d+",
        r"серия\s*\d+",
        r"все\s+серии",
        r"сериал",
        r"все\s+сезоны",
        r"\d+\s+сезон",
    ]

    CARTOON_MARKERS = [
        r"мультфильм",
        r"мультик",
        r"мульт",
        r"анимаци",
        r"animation",
        r"pixar",
        r"dreamworks",
        r"disney",
        r"дисней",
        r"пиксар",
    ]

    CARTOON_SERIAL_MARKERS = [r"мультсериал", r"анимационный\s+сериал"]

    FILM_MARKERS = [r"фильм", r"кино", r"кинолента", r"блокбастер"]

    @classmethod
    def analyze(cls, query: str) -> Optional[Tuple[bool, ContentType, Optional[str]]]:
        """
        Возвращает (type_query, content_type, franchise_title) или None,
        если правила не смогли определить.
        """
        query_lower = query.lower()

        # 1. Определение типа контента
        content_type = cls._detect_content_type(query_lower)

        # 2. Определение, относится ли к профессиональному контенту
        type_query = content_type != ContentType.EMPTY

        # 3. Извлечение франшизы (упрощенно)
        franchise = cls._extract_franchise(query)

        if content_type != ContentType.EMPTY:
            return (type_query, content_type, franchise)

        return None

    @classmethod
    def _detect_content_type(cls, query_lower: str) -> ContentType:
        """Определение типа контента по маркерам."""
        # Проверяем мультсериалы (более специфичный случай)
        for marker in cls.CARTOON_SERIAL_MARKERS:
            if re.search(marker, query_lower):
                return ContentType.CARTOON_SERIAL

        # Проверяем сериалы
        for marker in cls.SERIAL_MARKERS:
            if re.search(marker, query_lower):
                # Если есть мульт-маркеры, то это мультсериал
                for cartoon_marker in cls.CARTOON_MARKERS:
                    if re.search(cartoon_marker, query_lower):
                        return ContentType.CARTOON_SERIAL
                return ContentType.SERIAL

        # Проверяем мультфильмы
        for marker in cls.CARTOON_MARKERS:
            if re.search(marker, query_lower):
                return ContentType.CARTOON

        # Проверяем фильмы
        for marker in cls.FILM_MARKERS:
            if re.search(marker, query_lower):
                return ContentType.FILM

        return ContentType.EMPTY

    @classmethod
    def _extract_franchise(cls, query: str) -> Optional[str]:
        """
        Извлечение названия франшизы из запроса.
        Пример: "Гарри Поттер: Узник Аскабана" -> "Гарри Поттер"
        """
        # Паттерн: "Название: Подзаголовок" или "Название. Подзаголовок"
        patterns = [
            r"^([^:]+):",
            r"^([^.]+)\.",
            r"^(.+?)\s+фильм",
            r"^(.+?)\s+\d+$",  # "Название 2" -> "Название"
        ]

        for pattern in patterns:
            match = re.search(pattern, query.strip())
            if match:
                franchise = match.group(1).strip()
                # Проверяем, что название достаточно длинное
                if len(franchise) > 2:
                    return franchise

        return None


# ============================================================================
# КЛИЕНТ ДЛЯ KINOPOISK API UNOFFICIAL
# ============================================================================


class KinopoiskClient:
    """
    Клиент для работы с Kinopoisk API Unofficial
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.base_url = "https://kinopoiskapiunofficial.tech/api/v2.1"
        self.session.headers.update(
            {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        )

    def search(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Поиск фильма/сериала по ключевому слову.
        Возвращает нормализованные данные или None.
        """
        print(f"   🔍 Поиск через Kinopoisk API: '{query}'")

        try:
            params = {"keyword": query, "page": 1}

            response = self.session.get(
                f"{self.base_url}/films/search-by-keyword", params=params, timeout=10
            )
            response.raise_for_status()
            data = response.json()

            # Проверяем наличие результатов
            if data.get("films") and len(data["films"]) > 0:
                # Берем самый релевантный результат
                best_match = data["films"][0]
                return self._normalize_response(best_match)
            else:
                print(f"   ⚠️ Ничего не найдено для '{query}'")

        except requests.exceptions.RequestException as e:
            print(f"   ❌ Ошибка при обращении к Kinopoisk API: {e}")
        except json.JSONDecodeError:
            print(f"   ❌ Ошибка: Не удалось декодировать JSON ответ.")

        return None

    def _normalize_response(self, raw_data: Dict) -> Dict[str, Any]:
        """Приводит данные Kinopoisk API к единому формату для вашего пайплайна."""

        film_id = raw_data.get("filmId")
        title_ru = raw_data.get("nameRu", "")
        title_en = raw_data.get("nameEn", "")
        year = raw_data.get("year", "")

        # Определяем тип контента (FILM или SERIAL)
        # В базовом поиске тип может быть не указан, но мы можем попробовать
        # определить его по дополнительным данным или по названию
        content_type_str = raw_data.get("type")

        # Проверяем, является ли мультфильмом (упрощенно)
        genres = raw_data.get("genres", [])
        is_animation = any(genre.get("genre") == "мультфильм" for genre in genres)

        # Получаем детальную информацию для определения типа и франшизы
        details = self._get_film_details(film_id) if film_id else None

        if details:
            data = details.get("data")
            # Уточняем тип по детальной информации
            if data.get("serial") or data.get("type") == "TV_SERIES":
                content_type_str = "TV_SERIES"
            elif data.get("type") == "MINI_SERIES":
                content_type_str = "MINI_SERIES"
            elif data.get("type") == "FILM":
                content_type_str = "FILM"

            # Уточняем жанр (анимация)
            if not is_animation:
                detail_genres = data.get("genres", [])
                is_animation = any(
                    g.get("genre") == "мультфильм" for g in detail_genres
                )

            # Пытаемся определить франшизу
            franchise = self._extract_franchise(title_ru)
        else:
            franchise = None

        # Формируем ответ
        return {
            "found": True,
            "media_type": (
                "tv" if content_type_str in ("TV_SERIES", "MINI_SERIES") else "movie"
            ),
            "is_animation": is_animation,
            "title": title_ru or title_en,
            "original_title": title_en,
            "franchise": franchise,
            "kinopoisk_id": film_id,
            "year": year,
            "source": "kinopoisk",
        }

    def _get_film_details(self, film_id: int) -> Optional[Dict]:
        """Получение детальной информации о фильме/сериале."""
        url = f"{self.base_url}/films/{film_id}"

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"   ⚠️ Не удалось получить детали фильма {film_id}: {e}")
            return None

    def _extract_franchise(self, title: str) -> Optional[str]:
        """Упрощенное извлечение названия франшизы."""
        if not title:
            return None

        # Разделяем по двоеточию или тире
        for separator in [":", " – ", " - "]:
            if separator in title:
                return title.split(separator)[0].strip()

        # Паттерн: "Название 2" -> "Название"
        match = re.search(r"^(.+?)\s+\d+$", title)
        if match:
            return match.group(1).strip()

        return None


# ============================================================================
# МАППИНГ В CONTENTTYPE
# ============================================================================


class KinopoiskMapper:
    """Преобразование данных Kinopoisk API в ContentType."""

    @classmethod
    def map_to_analysis(
        cls, normalized_query: str, api_result: Dict
    ) -> "QueryAnalysis":
        """Преобразование результата API в QueryAnalysis."""

        # Определяем ContentType
        if api_result["media_type"] == "movie":
            if api_result.get("is_animation", False):
                content_type = ContentType.CARTOON
            else:
                content_type = ContentType.FILM
        else:  # tv
            if api_result.get("is_animation", False):
                content_type = ContentType.CARTOON_SERIAL
            else:
                content_type = ContentType.SERIAL

        return QueryAnalysis(
            original_query="",  # Будет заполнено в пайплайне
            normalized_query=normalized_query,
            type_query=True,  # Если нашли в Kinopoisk, то это профессиональный контент
            content_type=content_type,
            franchise_title=api_result.get("franchise"),
        )


# ============================================================================
# ГЛАВНЫЙ КОНВЕЙЕР ОБРАБОТКИ
# ============================================================================


class QueryPipeline:
    """
    Основной конвейер обработки запроса.
    Последовательность: Нормализация -> Локальные правила -> Кинопоиск API -> Fallback на модель.
    """

    def __init__(self, kinopoisk_api_key: str):
        self.kinopoisk_client = KinopoiskClient(kinopoisk_api_key)
        self.local_model = None

    def process(self, query: str) -> QueryAnalysis:
        """
        Основной метод обработки запроса.
        Возвращает полный анализ запроса.
        """
        print(f"\n🔍 Обработка запроса: '{query}'")

        # Шаг 1: Нормализация
        normalized = TextNormalizer.normalize(query)
        print(f"   📝 Нормализовано: '{normalized}'")

        # Шаг 2: Локальные правила
        rule_result = LocalRuleEngine.analyze(normalized)
        if rule_result:
            type_query, content_type, franchise = rule_result
            print(
                f"   ✅ Определено правилами: type_query={type_query}, content_type={content_type.value}"
            )
            return QueryAnalysis(
                original_query=query,
                normalized_query=normalized,
                type_query=type_query,
                content_type=content_type,
                franchise_title=franchise,
            )

        # Шаг 3: Запрос к Kinopoisk API
        api_result = self.kinopoisk_client.search(normalized)
        if api_result:
            print(
                f"   ✅ Найдено в Kinopoisk: {api_result['title']} ({api_result['media_type']})"
            )
            analysis = KinopoiskMapper.map_to_analysis(normalized, api_result)
            analysis.original_query = query
            return analysis

        # Шаг 4: Fallback на модель
        return self._fallback_to_model(query, normalized)

    def _fallback_to_model(
        self, original_query: str, normalized_query: str
    ) -> QueryAnalysis:
        """
        Заглушка для вызова вашей локальной модели.
        Здесь будет интеграция с моделью, которую пишет другой разработчик.
        """
        # TODO: Заменить на реальный вызов модели
        # result = self.local_model.predict(normalized_query)

        # Временная заглушка
        return QueryAnalysis(
            original_query=original_query,
            normalized_query=normalized_query,
            type_query=False,
            content_type=ContentType.EMPTY,
            franchise_title=None,
        )

    def set_local_model(self, model):
        """Установка модели для fallback-обработки."""
        self.local_model = model


# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ И ТЕСТИРОВАНИЕ
# ============================================================================


def test_pipeline():
    """Тестирование конвейера на разных запросах."""

    pipeline = QueryPipeline(kinopoisk_api_key=os.getenv("KINOPOISK_API_KEY"))

    print("=" * 70)
    print("ТЕСТИРОВАНИЕ КОНВЕЙЕРА ОБРАБОТКИ ПОИСКОВЫХ ЗАПРОСОВ")
    print("=" * 70)

    df = pd.read_csv("train.csv").head(10)

    for _, row in df.iterrows():
        result = pipeline.process(row["QueryText"])
        print(f"\n📊 Результат для '{row["QueryText"]}':")
        print(f"   {json.dumps(result.to_dict(), ensure_ascii=False, indent=2)}")
        print("-" * 50)


if __name__ == "__main__":
    test_pipeline()
