"""
Confidence Router — маршрутизация запросов в Yandex AI Studio.

Принимает запросы с низким confidence от Rule-based или ML-модели
и перенаправляет их в YandexLLMClient для классификации.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

from yandex_llm import YandexLLMClient

CONFIDENCE_THRESHOLD = 0.85
MAX_WORKERS          = 5
LLM_CHUNK            = 10

_llm: YandexLLMClient | None = None


def _get_llm() -> YandexLLMClient | None:
    global _llm
    if _llm is None:
        try:
            _llm = YandexLLMClient()
        except ValueError as e:
            print(f"[Router] LLM unavailable: {e}")
            return None
    return _llm


def needs_llm(confidence: float) -> bool:
    """Возвращает True если запрос нужно направить в AI Studio."""
    return confidence < CONFIDENCE_THRESHOLD


def route(queries: list[str]) -> list[tuple[int, str, str, float]]:
    """Отправляет батч запросов в Yandex AI Studio.

    Returns list of (TypeQuery, ContentType, Title, confidence).
    """
    chunks = [
        queries[i: i + LLM_CHUNK]
        for i in range(0, len(queries), LLM_CHUNK)
    ]
    chunk_results: dict[int, list] = {}

    llm = _get_llm()
    if llm is None:
        return [(0, "", "", 0.5)] * len(queries)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(llm.classify, chunk): cidx
            for cidx, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            cidx = futures[future]
            chunk_results[cidx] = future.result()

    flat = []
    for cidx in range(len(chunks)):
        flat.extend(chunk_results[cidx])
    return flat
