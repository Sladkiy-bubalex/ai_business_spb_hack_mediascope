"""
Rule-based фильтр для быстрой классификации запросов.
Покрывает очевидные случаи без обращения к ML или LLM API.
"""
import enum
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Паттерны → TypeQuery=0 (не видеоконтент)
_STOP = [
    r'\bтроллейбус\b', r'\bтрамвай\b', r'\bметро\b(?!\s+фильм|\s+сериал)',
    r'\bрасписани\w+', r'\bмаршрут\b(?!\s+\w+\s+(?:фильм|сериал))',
    r'\bпогод\w+', r'\bтемператур\w+',
    r'\bкурс\s+валют', r'\bдоллар\b', r'\bевро\b(?!\s+фильм)',
    r'\bбирж\w+', r'\bакци\w+(?!\s+фильм)',
    r'\bновост\w+', r'\bполитик\w+',
    r'\bрецепт\w+', r'\bкулинар\w+',
    r'\bнавигатор\b', r'\bяндекс\.карт', r'\bгугл\s+карт',
    r'\bфутбол\b(?!\s+(?:фильм|мультфильм))',
    r'\bхоккей\b', r'\bбаскетбол\b', r'\bтеннис\b',
    r'\bработ\w+\s+вакансии', r'\bвакансии?\b',
    r'\bкредит\w+', r'\bипотек\w+',
]

# Паттерны → TypeQuery=1 (видеоконтент)
_TRIGGER = [
    r'\bсмотрет\w+', r'\bпосмотрет\w+',
    r'\bонлайн\b', r'\bскачат\w+',
    r'\bторрент\b', r'\bhd\b', r'\b4k\b', r'\bfull\s*hd\b',
    r'\bhdrezka\b', r'\blostfilm\b', r'\bkinogo\b', r'\bлордфильм\b',
    r'\bсезон\b', r'\bсери[яи]\b', r'\bэпизод\b',
    r'\bтрейлер\b',
    r'\bаниме\b', r'\banime\b',
    r'\bдорам[аы]\b',
    r'\bмультик\w*\b', r'\bмульт\b',
    r'\bдисней\b', r'\bпиксар\b', r'\bdisney\b', r'\bpixar\b',
]

# Сигналы типа контента (порядок важен — от специфичного к общему)
_CT_SIGNALS = [
    ('мультсериал', [r'\bаниме\b', r'\banime\b', r'\bмультсериал\b']),
    ('мультфильм',  [r'\bмультфильм\b', r'\bмульт\b', r'\bмультик\w*\b',
                     r'\bдисней\b', r'\bпиксар\b', r'\bdisney\b', r'\bpixar\b', r'\bdreamworks\b']),
    ('сериал',      [r'\bсериал\b', r'\bсезон\b', r'\bсери[яи]\b', r'\bдорам[аы]\b']),
    ('фильм',       [r'\bфильм\b', r'\bкинофильм\b']),
]

# Компилируем заранее
_STOP_RE     = [re.compile(p, re.IGNORECASE) for p in _STOP]
_TRIGGER_RE  = [re.compile(p, re.IGNORECASE) for p in _TRIGGER]
_CT_RE       = [(ct, [re.compile(p, re.IGNORECASE) for p in pats])
                for ct, pats in _CT_SIGNALS]


# ==============================================================================
# ТИПЫ КОНТЕНТА И РЕЗУЛЬТАТ АНАЛИЗА
# ==============================================================================


class ContentType(enum.Enum):
    EMPTY = ""
    FILM = "фильм"
    SERIAL = "сериал"
    CARTOON = "мультфильм"
    CARTOON_SERIAL = "мультсериал"


@dataclass
class QueryAnalysis:
    original_query: str
    normalized_query: str
    type_query: bool
    content_type: ContentType
    franchise_title: Optional[str] = None


# ==============================================================================
# КЛИЕНТ ДЛЯ KINOPOISK API UNOFFICIAL
# ==============================================================================


class KinopoiskClient:
    """Клиент для работы с Kinopoisk API Unofficial."""

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
        """Поиск фильма/сериала по ключевому слову."""
        logger.debug(f"Поиск через Kinopoisk API: '{query}'")

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

            if data.get("films") and len(data["films"]) > 0:
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
        """Приводит данные Kinopoisk API к единому формату."""
        film_id = raw_data.get("filmId")
        title_ru = raw_data.get("nameRu", "")
        title_en = raw_data.get("nameEn", "")
        year = raw_data.get("year", "")

        content_type_str = raw_data.get("type")

        genres = raw_data.get("genres", [])
        is_animation = any(genre.get("genre") == "мультфильм" for genre in genres)

        details = self._get_film_details(film_id) if film_id else None

        if details:
            data = details.get("data", {})
            if data.get("serial") or data.get("type") == "TV_SERIES":
                content_type_str = "TV_SERIES"
            elif data.get("type") == "MINI_SERIES":
                content_type_str = "MINI_SERIES"
            elif data.get("type") == "FILM":
                content_type_str = "FILM"

            if not is_animation:
                detail_genres = data.get("genres", [])
                is_animation = any(
                    g.get("genre") == "мультфильм" for g in detail_genres
                )

        return {
            "found": True,
            "media_type": (
                "tv" if content_type_str in ("TV_SERIES", "MINI_SERIES") else "movie"
            ),
            "is_animation": is_animation,
            "title": title_ru or title_en,
            "original_title": title_en,
            "franchise": None,
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


# ==============================================================================
# ЕДИНЫЙ АНАЛИЗАТОР ЗАПРОСОВ
# ==============================================================================


class UnifiedQueryAnalyzer:
    """
    Единый анализатор запросов с предварительной очисткой и запросом к API Кинопоиска.

    Алгоритм:
    1. Очищаем запрос от типовых фраз ("смотреть онлайн", "скачать" и т.д.)
    2. Ищем маркеры типа контента и запоминаем их
    3. Удаляем найденные маркеры из запроса
    4. Отправляем очищенный запрос в API Кинопоиска
    5. Комбинируем результат API с найденным типом контента
    """

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

    CONTENT_MARKERS = [
        (ContentType.CARTOON_SERIAL, [r"мультсериал", r"анимационный\s+сериал"]),
        (ContentType.CARTOON, [
            r"мультфильм", r"мультик", r"мульт\b", r"анимаци",
            r"pixar", r"dreamworks", r"disney", r"дисней", r"пиксар",
        ]),
        (ContentType.SERIAL, [
            r"сериал", r"сезон\s*\d+", r"серия\s*\d+",
            r"\d+\s+сезон", r"тв\s*сериал",
        ]),
        (ContentType.FILM, [
            r"фильм", r"кино(?!поиск)", r"кинолента", r"блокбастер",
            r"трейлер", r"премьера",
        ]),
    ]

    def __init__(self, kinopoisk_api_key: Optional[str] = None):
        self.kinopoisk_client = KinopoiskClient(kinopoisk_api_key) if kinopoisk_api_key else None
        self.stop_patterns = [re.compile(p, re.IGNORECASE) for p in self.STOP_PHRASES]
        self.marker_patterns = [
            (ct, [re.compile(m, re.IGNORECASE) for m in markers])
            for ct, markers in self.CONTENT_MARKERS
        ]

    def analyze(self, query: str) -> QueryAnalysis:
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
            franchise_title=franchise_title,
        )

    def _clean_query(self, query: str) -> str:
        cleaned = query.strip()
        for pattern in self.stop_patterns:
            cleaned = pattern.sub('', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'[,.!?;:]$', '', cleaned)
        return cleaned.strip()

    def _extract_and_remove_markers(self, query: str) -> Tuple[ContentType, str]:
        content_type = ContentType.EMPTY
        cleaned_query = query

        for ct, patterns in self.marker_patterns:
            for pattern in patterns:
                if pattern.search(cleaned_query):
                    content_type = ct
                    cleaned_query = pattern.sub('', cleaned_query)
                    break
            if content_type != ContentType.EMPTY:
                break

        cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
        return content_type, cleaned_query

    def _determine_type_from_api(self, api_result: Dict) -> ContentType:
        media_type = api_result.get("media_type", "")
        is_animation = api_result.get("is_animation", False)

        if media_type == "tv":
            return ContentType.CARTOON_SERIAL if is_animation else ContentType.SERIAL
        else:
            return ContentType.CARTOON if is_animation else ContentType.FILM


# ============================================================================
# SINGLETON АНАЛИЗАТОРА
# ============================================================================

_analyzer: "UnifiedQueryAnalyzer | None" = None


def _get_analyzer() -> "UnifiedQueryAnalyzer | None":
    global _analyzer
    if _analyzer is None:
        api_key = os.environ.get("KINOPOISK_API_KEY")
        _analyzer = UnifiedQueryAnalyzer(api_key)
    return _analyzer


def apply_rules(query: str) -> tuple[int | None, str, float]:
    """
    Возвращает (TypeQuery, ContentType, confidence).
    TypeQuery=None означает «не уверен — передай в ML».

    Использует UnifiedQueryAnalyzer для предварительной очистки запроса,
    определения маркеров типа контента и запроса к API Кинопоиска.
    """
    q = str(query).lower()

    # Явный TypeQuery=0
    for pat in _STOP_RE:
        if pat.search(q):
            return 0, '', 0.97

    # Предобработка через UnifiedQueryAnalyzer (очистка + маркеры + API Кинопоиска)
    analyzer = _get_analyzer()
    analysis = analyzer.analyze(query) if analyzer else None

    # Ищем триггеры в исходном запросе
    hits = [p for p in _TRIGGER_RE if p.search(q)]
    if not hits:
        # Нет триггеров — но анализатор мог подтвердить видеоконтент через API/маркеры
        if analysis and analysis.type_query:
            ct = analysis.content_type.value if analysis.content_type != ContentType.EMPTY else ''
            return 1, ct, 0.80
        return None, '', 0.0  # не уверен

    # Определяем ContentType (из правил)
    content_type = ''
    for ct, pats in _CT_RE:
        if any(p.search(q) for p in pats):
            content_type = ct
            break

    # Если тип не определён правилами, берём из анализатора
    if not content_type and analysis and analysis.content_type != ContentType.EMPTY:
        content_type = analysis.content_type.value

    confidence = min(0.97, 0.75 + 0.04 * len(hits))
    return 1, content_type, confidence
