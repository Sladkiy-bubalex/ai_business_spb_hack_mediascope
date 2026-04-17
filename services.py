import os
import re
import json
import requests
import pandas as pd
from dotenv import load_dotenv
from dataclasses import dataclass
from enum import Enum
from loguru import logger
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
# УТИЛИТЫ
# ============================================================================


def extract_franchise(title: str) -> Optional[str]:
    """
    Извлечение названия франшизы из полного названия.
    """
    if not title:
        return None

    patterns = [
        r'^([^:]+):',           # До двоеточия
        r'^([^.]+)\.',           # До точки
        r'^(.+?)\s+фильм',       # До слова "фильм"
        r'^(.+?)\s+\d+$',        # До номера части
        r'^(.+?)\s+—',           # До тире
        r'^(.+?)\s+-',           # До дефиса
    ]

    matches = [re.search(pattern, title.strip()) for pattern in patterns if re.search(pattern, title.strip())]
    if matches:
        return matches[0].group(1).strip()
    
    return None


# ============================================================================
# ЭТАП 2: ЛОКАЛЬНЫЕ ПРАВИЛА И МАРКЕРЫ
# ============================================================================


class UnifiedQueryAnalyzer:
    """
    Единый анализатор запросов с предварительной очисткой и запроса к API Кинопоиска.

    Алгоритм:
    1. Очищаем запрос от типовых фраз ("смотреть онлайн", "скачать" и т.д.)
    2. Ищем маркеры типа контента и запоминаем их
    3. Удаляем найденные маркеры из запроса
    4. Отправляем очищенный запрос в API Кинопоиска
    5. Комбинируем результат API с найденным типом контента
    """

    # Стоп-фразы, которые нужно удалять из запроса
    STOP_PHRASES = [
        r"смотреть\s+онлайн(?:\s+бесплатно)?(?:\s+в\s+хорошем\s+качестве)?",
        r"скачать(?:\s+бесплатно)?(?:\s+торрент)?(?:\s+без\s+торрента)?",
        r"все\s+серии",
        r"все\s+сезоны",
        r"все\s+части",
        r"\d+\s+(?:серия|серии|серию|серий)",
        r"\d+\s+(?:сезон|сезона|сезоне|сезонов)",
        r"(?:серия|серии|серию|серий)\s+\d+",
        r"(?:сезон|сезона|сезоне|сезонов)\s+\d+",
        r"\d+\s+(?:эпизод|эпизода|эпизодов)",
        r"(?:эпизод|эпизода|эпизодов)\s+\d+",
        r"в\s+хорошем\s+качестве",
        r"hd\s*качество",
        r"hd\s*rip",
        r"бесплатно",
        r"на\s+русском",
        r"с\s+субтитрами",
        r"полный\s+фильм",
        r"полная\s+версия",
    ]

    # Маркеры для определения типа контента (с приоритетом)
    CONTENT_MARKERS = [
        # Мультсериалы (наивысший приоритет)
        (ContentType.CARTOON_SERIAL, [r"мультсериал", r"анимационный\s+сериал"]),
        
        # Мультфильмы
        (ContentType.CARTOON, [
            r"мультфильм", r"мультик", r"мульт\b", r"анимаци",
            r"pixar", r"dreamworks", r"disney", r"дисней", r"пиксар"
        ]),
        
        # Сериалы
        (ContentType.SERIAL, [
            r"сериал", r"сезон\s*\d+", r"серия\s*\d+", 
            r"\d+\s+сезон", r"тв\s*сериал"
        ]),
        
        # Фильмы
        (ContentType.FILM, [
            r"фильм", r"кино(?!поиск)", r"кинолента", r"блокбастер",
            r"трейлер", r"премьера"
        ]),
    ]

    def __init__(self, kinopoisk_api_key: Optional[str] = None):
        """
        Инициализация анализатора.
        
        Args:
            kinopoisk_api_key: API ключ для Кинопоиска. Если None, API не используется.
        """
        self.kinopoisk_client = KinopoiskClient(kinopoisk_api_key) if kinopoisk_api_key else None
        self.stop_patterns = [re.compile(p, re.IGNORECASE) for p in self.STOP_PHRASES]
        self.marker_patterns = [
            (ct, [re.compile(m, re.IGNORECASE) for m in markers])
            for ct, markers in self.CONTENT_MARKERS
        ]

    def analyze(self, query: str) -> QueryAnalysis:
        """
        Основной метод анализа запроса.
        
        Args:
            query: Исходный поисковый запрос
            
        Returns:
            QueryAnalysis с результатами анализа
        """
        logger.debug(f"Анализ запроса: '{query}'")
        
        # Шаг 1: Поиск маркеров типа контента и их удаление
        content_type, cleaned_without_markers = self._extract_and_remove_markers(query)
        logger.debug(f"Определен тип: {content_type.value}")
        logger.debug(f"Ключевые слова для API: '{cleaned_without_markers}'")

        # Шаг 2: Очистка запроса от стоп-фраз
        cleaned = self._clean_query(cleaned_without_markers)
        logger.debug(f"После очистки: '{cleaned}'")

        # Шаг 3: Запрос к API Кинопоиска
        api_result = None
        franchise_title = None
        
        if self.kinopoisk_client and cleaned:
            api_result = self.kinopoisk_client.search(cleaned)
            
            if api_result and api_result.get("found"):
                logger.debug(f"Найдено в API: {api_result['title']} ({api_result['year']})")
                
                # Если тип контента не был определен по маркерам, определяем по API
                if content_type == ContentType.EMPTY:
                    content_type = self._determine_type_from_api(api_result)
        
        # Шаг 4: Определяем, относится ли запрос к профессиональному контенту
        type_query = (
            content_type != ContentType.EMPTY or 
            (api_result and api_result.get("found", False))
        )
        
        return QueryAnalysis(
            original_query=query,
            normalized_query=cleaned,
            type_query=type_query,
            content_type=content_type,
            franchise_title=franchise_title
        )

    def _clean_query(self, query: str) -> str:
        """
        Очистка запроса от типовых фраз.
        """
        cleaned = query.strip()
        
        # Удаляем стоп-фразы
        for pattern in self.stop_patterns:
            cleaned = pattern.sub('', cleaned)
        
        # Удаляем лишние пробелы и знаки препинания
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'[,.!?;:]$', '', cleaned)
        
        return cleaned.strip()

    def _extract_and_remove_markers(self, query: str) -> Tuple[ContentType, str]:
        """
        Извлекает тип контента по маркерам и удаляет их из запроса.
        
        Returns:
            Tuple[ContentType, str]: (определенный тип, очищенный запрос)
        """
        content_type = ContentType.EMPTY
        cleaned_query = query
        
        # Проверяем маркеры в порядке приоритета (используем прекомпилированные паттерны)
        for ct, patterns in self.marker_patterns:
            for pattern in patterns:
                if pattern.search(cleaned_query):
                    content_type = ct
                    cleaned_query = pattern.sub('', cleaned_query)
                    break
            if content_type != ContentType.EMPTY:
                break
        
        # Дополнительная очистка после удаления маркеров
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
        
        return content_type, cleaned_query

    def _determine_type_from_api(self, api_result: Dict) -> ContentType:
        """
        Определяет тип контента по результату API.
        """
        media_type = api_result.get("media_type", "")
        is_animation = api_result.get("is_animation", False)
        
        if media_type == "tv":
            return ContentType.CARTOON_SERIAL if is_animation else ContentType.SERIAL
        else:  # movie
            return ContentType.CARTOON if is_animation else ContentType.FILM


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
        self._search_cache: Dict[str, Dict[str, Any]] = {}
        self._details_cache: Dict[int, Dict] = {}

    def search(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Поиск фильма/сериала по ключевому слову.
        Возвращает нормализованные данные или None.
        """
        logger.debug(f"Поиск через Kinopoisk API: '{query}'")

        # Проверяем кэш
        if query in self._search_cache:
            logger.debug(f"Результат из кэша для '{query}'")
            return self._search_cache[query]

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
                result = self._normalize_response(best_match)
                self._search_cache[query] = result
                return result
            else:
                logger.warning(f"Ничего не найдено для '{query}'")

        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при обращении к Kinopoisk API: {e}")
        except json.JSONDecodeError:
            logger.error("Не удалось декодировать JSON ответ.")

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

        franchise = extract_franchise(title_ru) if title_ru else None

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
        if film_id in self._details_cache:
            return self._details_cache[film_id]

        url = f"{self.base_url}/films/{film_id}"

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            result = response.json()
            self._details_cache[film_id] = result
            return result
        except requests.exceptions.RequestException as e:
            logger.warning(f"Не удалось получить детали фильма {film_id}: {e}")
            return None


# ============================================================================
# ГЛАВНЫЙ КОНВЕЙЕР ОБРАБОТКИ
# ============================================================================


class QueryPipeline:
    """
    Основной конвейер обработки запроса.
    Последовательность: Нормализация -> Локальные правила -> Кинопоиск API -> Fallback на модель.
    """

    def __init__(self, kinopoisk_api_key: str):
        self.analyzer = UnifiedQueryAnalyzer(kinopoisk_api_key)
        self.local_model = None
    
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

    def _normalize_text(self, text: str) -> str:
        """Нормализация текста (сохраняет двоеточия и тире для извлечения франшиз)"""
        text = text.lower()
        text = text.replace("ё", "е")
        text = re.sub(r"\r\n|\n", " ", text)
        text = re.sub(r"[^\w\s:–\-]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def process(self, query: str) -> QueryAnalysis:
        """
        Основной метод обработки запроса.
        Возвращает полный анализ запроса.
        """
        logger.info(f"🔍 Обработка запроса: '{query}'")

        # Шаг 1: Нормализация
        normalized = self._normalize_text(query)
        logger.info(f"📝 Нормализовано: '{normalized}'")

        # Шаг 2: Локальные правила
        rule_result = self.analyzer.analyze(normalized)
        if rule_result.type_query:
            return rule_result

        # Шаг 3: Fallback на модель
        return self._fallback_to_model(query, normalized)

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
        print(f"\n📊 Результат для '{row['QueryText']}':")
        print(f"   {json.dumps(result.to_dict(), ensure_ascii=False, indent=2)}")
        print("-" * 50)


if __name__ == "__main__":
    test_pipeline()
