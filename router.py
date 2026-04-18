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

_llm = YandexLLMClient()


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

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_llm.classify, chunk): cidx
            for cidx, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            cidx = futures[future]
            chunk_results[cidx] = future.result()

    flat = []
    for cidx in range(len(chunks)):
        flat.extend(chunk_results[cidx])
    return flat
