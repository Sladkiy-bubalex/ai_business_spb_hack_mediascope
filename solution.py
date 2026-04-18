import gc
import json
import os
import pickle
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from rules import apply_rules
from router import needs_llm, route

CONFIDENCE_THRESHOLD = 0.85
TYPE_THRESHOLD       = 0.35   # F2 штрафует FN вдвое — порог ниже 0.5
SORT_THRESHOLD       = 72     # rapidfuzz token_sort_ratio (1-й проход, ловит перестановки)
SET_THRESHOLD        = 78     # rapidfuzz token_set_ratio (2-й проход, ловит подмножества)
MATCH_THRESHOLD      = 78     # общий порог принятия fuzzy-match
DICT_CT_THRESHOLD    = 90     # порог переопределения ContentType из словаря
VALID_CT = {"фильм", "сериал", "мультфильм", "мультсериал"}

# Generic-слова, которые не могут быть настоящим тайтлом (для fallback)
_GENERIC_TITLE_WORDS = frozenset({
    "фильм", "фильмы", "фильма", "фильмов",
    "сериал", "сериалы", "сериала", "сериалов",
    "мультфильм", "мультфильмы", "мультик", "мультики",
    "мультсериал", "мультсериалы",
    "кино", "аниме", "дорама", "дорамы",
    "год", "года", "году", "годов",
    "новый", "новая", "новое", "новые", "новинка", "новинки",
    "топ", "лучший", "лучшая", "лучшее", "лучшие",
    "онлайн", "смотреть", "скачать",
    "сезон", "сезона", "сезоны", "серия", "серии", "серий",
})

_DIR = os.path.dirname(os.path.abspath(__file__))

# Phrase-level очистка (убирает составные выражения до пословного _JUNK)
_PHRASE_STOP = re.compile(
    r'смотреть\s+онлайн(?:\s+бесплатно)?(?:\s+в\s+хорошем\s+качестве)?'
    r'|скачать(?:\s+бесплатно)?(?:\s+торрент)?(?:\s+без\s+торрента)?'
    r'|все\s+(?:серии|сезоны|части)'
    r'|\d+\s+(?:серия|серии|серию|серий|сезон|сезона|сезоне|сезонов|эпизод|эпизода|эпизодов)'
    r'|(?:серия|серии|серию|серий|сезон|сезона|сезоне|сезонов|эпизод|эпизода|эпизодов)\s+\d+'
    r'|в\s+хорошем\s+качестве'
    r'|hd\s*(?:качество|rip)'
    r'|полный\s+фильм|полная\s+версия'
    r'|на\s+русском|с\s+субтитрами',
    re.IGNORECASE,
)

_JUNK = re.compile(
    r'\b(смотреть|онлайн|бесплатно|скачать|торрент|hd|1080|720|480|4k|'
    r'сезон|серия|серий|эпизод|s\d+e\d+|s\d+|е\d+|ep\d+|\d+\s*серия|\d+\s*сезон|'
    r'субтитры|дублированный|дубляж|перевод|rus|eng|ru|'
    r'новый|новинка|все|полный|полностью|'
    r'фильм|кино|сериал|мультфильм|мультсериал|мультик|аниме|дорама|шоу|'
    r'на русском|в хорошем качестве|хорошее качество|без регистрации)\b',
    re.IGNORECASE,
)
_YEAR = re.compile(r'\b(19|20)\d{2}\b')
_SPC  = re.compile(r'\s+')


def _load(filename):
    path = os.path.join(_DIR, "models", filename)
    with open(path, "rb") as f:
        return pickle.load(f)


def _preprocess_for_title(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r'[«»""\'`]', '', t)
    t = _PHRASE_STOP.sub('', t)   # phrase-level первым
    t = _YEAR.sub('', t)
    t = _JUNK.sub('', t)
    return _SPC.sub(' ', t).strip()


class PredictionModel:
    batch_size: int = 1024

    def __init__(self) -> None:
        self.tfidf_type    = _load("tfidf_type.pkl")
        self.model_type    = _load("model_type.pkl")
        self.tfidf_content = _load("tfidf_content.pkl")
        self.model_content = _load("model_content.pkl")
        self.le_content    = _load("le_content.pkl")
        self.stop_words    = _load("stop_words_search.pkl")

        # Словарь тайтлов с нечётким поиском
        self._lookup: dict[str, tuple[str, str, str, bool]] = {}
        self._aliases: list[str] = []
        dict_path = os.path.join(_DIR, "models", "titles_dict.json")
        if os.path.exists(dict_path):
            with open(dict_path, encoding="utf-8") as f:
                data = json.load(f)
            entries = data.get("titles", data) if isinstance(data, dict) else data
            for entry in entries:
                if isinstance(entry, str):
                    # старый формат: просто список строк
                    if entry not in self._lookup:
                        self._lookup[entry] = (entry, "", "", False)
                else:
                    canonical = entry["canonical"]
                    ct        = entry.get("content_type", "")
                    year      = entry.get("year", "")
                    kp        = entry.get("kp_source", False)
                    for alias in entry.get("aliases", [canonical]):
                        if alias not in self._lookup:
                            self._lookup[alias] = (canonical, ct, year, kp)
            self._aliases = list(self._lookup.keys())
            print(f"Словарь тайтлов: {len(self._aliases)} алиасов")

    # ------------------------------------------------------------------

    def _clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zа-яё0-9\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _match_title(self, query: str) -> tuple[str, str, float]:
        title, ct, score, _ = self._match_title_full(query)
        return title, ct, score

    def _match_title_full(self, query: str) -> tuple[str, str, float, bool]:
        """Двухуровневый fuzzy-поиск. Возвращает (title, content_type, score, kp_source)."""
        clean = _preprocess_for_title(query)
        if not clean or len(clean) < 2:
            return "", "", 0.0, False
        if clean in self._lookup:
            canonical, ct, _, kp = self._lookup[clean]
            return canonical, ct, 100.0, kp

        result = process.extractOne(
            clean, self._aliases,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=SORT_THRESHOLD,
        )
        if result is not None:
            best_alias, score, _ = result
            canonical, ct, _, kp = self._lookup[best_alias]
            return canonical, ct, float(score), kp

        result = process.extractOne(
            clean, self._aliases,
            scorer=fuzz.token_set_ratio,
            score_cutoff=SET_THRESHOLD,
        )
        if result is None:
            return "", "", 0.0, False
        best_alias, score, _ = result
        canonical, ct, _, kp = self._lookup[best_alias]
        return canonical, ct, float(score), kp

    def _fallback_title(self, query: str):
        """Стоп-слова + generic-фильтр. Возвращает nan если остались только мусорные слова."""
        if not isinstance(query, str) or pd.isna(query):
            return np.nan
        words = query.split()
        filtered = [
            w for w in words
            if w not in self.stop_words
            and w not in _GENERIC_TITLE_WORDS
            and len(w) > 1
            and not w.isdigit()
        ]
        title = " ".join(filtered[:5])  # >5 слов — скорее шум, чем реальный тайтл
        return title if len(title) >= 2 else np.nan

    def _get_title_and_ct(self, queries_raw: list[str], queries_clean: list[str]) -> tuple[list, list]:
        """Title + ContentType для списка TypeQuery=1 запросов."""
        n = len(queries_raw)
        titles = [np.nan] * n
        cts    = [np.nan] * n

        # ContentType — ML (базовое предсказание)
        X_ct      = self.tfidf_content.transform(pd.Series(queries_clean))
        ct_enc    = self.model_content.predict(X_ct)
        ct_labels = self.le_content.inverse_transform(ct_enc)
        for k in range(n):
            cts[k] = ct_labels[k]

        # Title — словарь с fuzzy, иначе стоп-слова.
        # CT override:
        #   - train.csv записи (kp_source=False): CT точный — всегда перекрываем ML
        #   - KP записи (kp_source=True): CT только фильм/сериал — доверяем только для animated
        ANIMATED = {"мультфильм", "мультсериал"}
        for k, (raw, _) in enumerate(zip(queries_raw, queries_clean)):
            if self._aliases:
                title, dict_ct, score, kp_src = self._match_title_full(raw)
                if score >= MATCH_THRESHOLD:
                    titles[k] = title
                    if dict_ct in VALID_CT:
                        if not kp_src:
                            cts[k] = dict_ct  # train.csv — точный CT
                        elif dict_ct in ANIMATED:
                            cts[k] = dict_ct  # KP — только для animated
                    continue
            titles[k] = self._fallback_title(self._clean_text(raw))

        return titles, cts

    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        queries = df["QueryText"].tolist()
        n = len(queries)

        type_out  = [0]      * n
        ct_out    = [np.nan] * n
        title_out = [np.nan] * n
        llm_done: set[int] = set()   # запросы, обработанные LLM (их ответ не переписываем)

        # ------------------------------------------------------------------
        # Шаг 1: Rule-based фильтр
        # ------------------------------------------------------------------
        need_ml = []
        for i, q in enumerate(queries):
            tq, ct, conf = apply_rules(str(q))
            if tq is not None and not needs_llm(conf):
                type_out[i] = tq
                ct_out[i]   = ct if ct else np.nan
            else:
                need_ml.append(i)

        # ------------------------------------------------------------------
        # Шаг 1.5: Словарь-бустер TypeQuery
        # Для rules-неуверенных запросов — fuzzy match по словарю.
        # Если найден известный тайтл → TypeQuery=1, используем ct+title из словаря.
        # ------------------------------------------------------------------
        # Для таких запросов принудительно ставим TypeQuery=1 и title из словаря.
        # ContentType НЕ заполняем из словаря — оставляем на ML в шаге 4.
        need_ml_final = []
        dict_boosted = []
        if self._aliases:
            for i in need_ml:
                title, _, score = self._match_title(queries[i])
                if score >= MATCH_THRESHOLD:
                    type_out[i]  = 1
                    title_out[i] = title
                    dict_boosted.append(i)
                else:
                    need_ml_final.append(i)
        else:
            need_ml_final = need_ml

        # ------------------------------------------------------------------
        # Шаг 2: ML — порог 0.35 (F2 favours recall)
        # ------------------------------------------------------------------
        need_llm_idx = []
        ml_fallback  = {}

        if need_ml_final:
            ml_raw     = [queries[i] for i in need_ml_final]
            ml_cleaned = [self._clean_text(q) for q in ml_raw]
            cleaned_s  = pd.Series(ml_cleaned)

            X_type     = self.tfidf_type.transform(cleaned_s)
            proba_type = self.model_type.predict_proba(X_type)[:, 1]  # P(tq=1)
            preds      = (proba_type >= TYPE_THRESHOLD).astype(int)

            high_conf_pos, low_conf = [], []
            for j, (tq, p1) in enumerate(zip(preds, proba_type)):
                i    = need_ml_final[j]
                conf = float(p1) if tq == 1 else float(1 - p1)

                if needs_llm(conf):
                    need_llm_idx.append(i)
                    ml_fallback[i] = (int(tq), j)
                    low_conf.append(j)
                else:
                    type_out[i] = int(tq)
                    if tq == 1:
                        high_conf_pos.append((i, j))

            # CT + Title для высокоуверенных tq=1
            if high_conf_pos:
                hc_raw     = [ml_raw[j]     for _, j in high_conf_pos]
                hc_cleaned = [ml_cleaned[j] for _, j in high_conf_pos]
                titles, cts = self._get_title_and_ct(hc_raw, hc_cleaned)
                for k, (i, _) in enumerate(high_conf_pos):
                    ct_out[i]    = cts[k]
                    title_out[i] = titles[k]

            # CT + Title для LLM-bound запросов (ML fallback на случай отказа LLM)
            lc_tq1 = [(need_ml_final[lj], lj) for lj in low_conf
                      if ml_fallback[need_ml_final[lj]][0] == 1]
            if lc_tq1:
                lc_raw     = [ml_raw[j]     for _, j in lc_tq1]
                lc_cleaned = [ml_cleaned[j] for _, j in lc_tq1]
                titles, cts = self._get_title_and_ct(lc_raw, lc_cleaned)
                for k, (i, _) in enumerate(lc_tq1):
                    ml_fallback[i] = (1, cts[k], titles[k])
            for i in need_llm_idx:
                if isinstance(ml_fallback[i][1], int):
                    tq_fb = ml_fallback[i][0]
                    ml_fallback[i] = (tq_fb, np.nan, np.nan)

        # ------------------------------------------------------------------
        # Шаг 3: Yandex AI Studio — низкий confidence
        # ------------------------------------------------------------------
        if need_llm_idx:
            llm_queries = [queries[i] for i in need_llm_idx]
            llm_results = route(llm_queries)
            for i, (tq, ct, title, conf) in zip(need_llm_idx, llm_results):
                if conf <= 0.5:  # LLM упал — ML fallback
                    tq_ml, ct_ml, title_ml = ml_fallback[i]
                    type_out[i]  = tq_ml
                    ct_out[i]    = ct_ml    if not (isinstance(ct_ml,    float) and np.isnan(ct_ml))    else np.nan
                    title_out[i] = title_ml if not (isinstance(title_ml, float) and np.isnan(title_ml)) else np.nan
                else:
                    type_out[i]  = tq
                    ct_out[i]    = ct    if ct    else np.nan
                    title_out[i] = title if title else np.nan
                    llm_done.add(i)  # LLM дал ответ — уважаем его, не перезаписываем в step 4

        # ------------------------------------------------------------------
        # Шаг 4: Дозаполнение CT+Title для tq=1 без предсказания (НЕ трогаем LLM-результаты)
        # ------------------------------------------------------------------
        missing = [
            i for i, tq in enumerate(type_out)
            if tq == 1 and i not in llm_done and (
                (isinstance(ct_out[i],    float) and np.isnan(ct_out[i]))    or
                (isinstance(title_out[i], float) and np.isnan(title_out[i]))
            )
        ]
        if missing:
            miss_raw     = [queries[i] for i in missing]
            miss_cleaned = [self._clean_text(q) for q in miss_raw]
            titles, cts  = self._get_title_and_ct(miss_raw, miss_cleaned)
            for k, i in enumerate(missing):
                if isinstance(ct_out[i], float) and np.isnan(ct_out[i]):
                    ct_out[i] = cts[k]
                if isinstance(title_out[i], float) and np.isnan(title_out[i]):
                    title_out[i] = titles[k]

        # ------------------------------------------------------------------
        gc.collect()

        out = df[["QueryText"]].copy()
        out["TypeQuery"]   = type_out
        out["Title"]       = title_out
        out["ContentType"] = ct_out
        return out


# ---------------------------------------------------------------------------
# Module-level helpers для test_pipeline.py / test_200.py
# ---------------------------------------------------------------------------

_MODEL: "PredictionModel | None" = None


def _get_model() -> PredictionModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = PredictionModel()
    return _MODEL


def _ml_predict(queries: list[str]) -> list[tuple[int, str, float]]:
    """Возвращает [(TypeQuery, ContentType, confidence), ...] для каждого запроса."""
    model   = _get_model()
    cleaned = [model._clean_text(q) for q in queries]
    X_type  = model.tfidf_type.transform(pd.Series(cleaned))
    proba   = model.model_type.predict_proba(X_type)[:, 1]
    preds   = (proba >= TYPE_THRESHOLD).astype(int)

    results = []
    for j, (tq, p1) in enumerate(zip(preds, proba)):
        conf = float(p1) if tq == 1 else float(1 - p1)
        ct   = ""
        if tq == 1:
            X_ct  = model.tfidf_content.transform(pd.Series([cleaned[j]]))
            ct    = model.le_content.inverse_transform(model.model_content.predict(X_ct))[0]
        results.append((int(tq), ct, conf))
    return results


def _get_title(query: str) -> str:
    """Извлекает тайтл из запроса: сначала словарь, потом fallback по стоп-словам."""
    model = _get_model()
    if model._aliases:
        title, _, score = model._match_title(query)
        if score >= MATCH_THRESHOLD:
            return title
    result = model._fallback_title(model._clean_text(query))
    return result if isinstance(result, str) else ""
