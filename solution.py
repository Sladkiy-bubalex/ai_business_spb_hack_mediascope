import gc
import json
import os
import re
import sys
import joblib
import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    from rapidfuzz import fuzz, process
    load_dotenv()
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    print("⚠️ rapidfuzz not installed. Fuzzy title matching will fall back to rule-based extraction.")

from rules import apply_rules
from router import needs_llm, route

sys.modules['__main__'] = sys.modules[__name__]

_DIR = os.path.dirname(os.path.abspath(__file__))


class TextEnsemble:
    """Ансамблевый классификатор — должен совпадать с классом при обучении."""
    def __init__(self, n_models=3):
        self.vecs = []
        self.clfs = []
        self.n_models = n_models

    def fit(self, X_texts, y):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression, SGDClassifier
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV

        vec_configs = [
            {'max_features': 50000, 'ngram_range': (1, 2), 'sublinear_tf': True, 'min_df': 2, 'max_df': 0.95, 'smooth_idf': True},
            {'max_features': 30000, 'ngram_range': (2, 4), 'sublinear_tf': True, 'min_df': 2, 'max_df': 0.95, 'smooth_idf': True},
            {'max_features': 40000, 'ngram_range': (3, 5), 'analyzer': 'char_wb', 'sublinear_tf': True, 'min_df': 2, 'max_df': 0.95, 'smooth_idf': True},
        ]
        clf_configs = [
            LogisticRegression(C=1.5, max_iter=1500, class_weight='balanced', solver='lbfgs', n_jobs=-1),
            SGDClassifier(loss='log_loss', max_iter=1500, class_weight='balanced', random_state=42, n_jobs=-1),
            CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced', max_iter=3000, random_state=42), cv=3, method='isotonic'),
        ]
        for i in range(min(self.n_models, len(vec_configs), len(clf_configs))):
            v = TfidfVectorizer(**vec_configs[i])
            X_vec = v.fit_transform(X_texts)
            clf = clf_configs[i]
            clf.fit(X_vec, y)
            self.vecs.append(v)
            self.clfs.append(clf)
        return self

    def predict_proba(self, X_texts):
        probs = np.zeros((len(X_texts), len(self.clfs[0].classes_)))
        for v, c in zip(self.vecs, self.clfs):
            probs += c.predict_proba(v.transform(X_texts))
        return probs / len(self.clfs)

    def predict(self, X_texts):
        return np.argmax(self.predict_proba(X_texts), axis=1)

CONFIDENCE_THRESHOLD = 0.85
SORT_THRESHOLD       = 72
SET_THRESHOLD        = 78
MATCH_THRESHOLD      = 78
VALID_CT = {"фильм", "сериал", "мультфильм", "мультсериал"}

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


def _preprocess_for_title(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r'[«»""\'`]', '', t)
    t = _PHRASE_STOP.sub('', t)
    t = _YEAR.sub('', t)
    t = _JUNK.sub('', t)
    return _SPC.sub(' ', t).strip()


class PredictionModel:
    batch_size: int = 1024

    def __init__(self) -> None:
        models_dir = os.path.join(_DIR, "models")

        # Ансамблевые ML-модели
        self.ens_type, self.thresh = joblib.load(os.path.join(models_dir, "ens_type.pkl"))
        self.ens_content           = joblib.load(os.path.join(models_dir, "ens_content.pkl"))
        self.le_content            = joblib.load(os.path.join(models_dir, "le_content.pkl"))
        self.noise                 = set(joblib.load(os.path.join(models_dir, "noise_words.pkl")))

        # Словарь тайтлов
        self._lookup: dict[str, tuple[str, str, str, bool]] = {}
        self._aliases: list[str] = []
        dict_path = os.path.join(models_dir, "titles_dict.json")
        if os.path.exists(dict_path):
            with open(dict_path, encoding="utf-8") as f:
                data = json.load(f)
            entries = data.get("titles", data) if isinstance(data, dict) else data
            for entry in entries:
                if isinstance(entry, str):
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
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _clean_text(self, text) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zа-яё0-9\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _match_title_full(self, query: str) -> tuple[str, str, float, bool]:
        """Двухуровневый fuzzy-поиск. Возвращает (title, content_type, score, kp_source)."""
        clean = _preprocess_for_title(query)
        if not clean or len(clean) < 2:
            return "", "", 0.0, False
        if clean in self._lookup:
            canonical, ct, _, kp = self._lookup[clean]
            return canonical, ct, 100.0, kp

        if HAS_RAPIDFUZZ and self._aliases:
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
            if result is not None:
                best_alias, score, _ = result
                canonical, ct, _, kp = self._lookup[best_alias]
                return canonical, ct, float(score), kp

        return "", "", 0.0, False

    def _match_title(self, query: str) -> tuple[str, str, float]:
        title, ct, score, _ = self._match_title_full(query)
        return title, ct, score

    def _fallback_title(self, query: str):
        if not isinstance(query, str) or pd.isna(query):
            return np.nan
        words = query.lower().split()
        stop_set = self.noise | _GENERIC_TITLE_WORDS
        filtered = [
            w for w in words
            if w not in stop_set and len(w) > 1 and not w.isdigit()
        ]
        title = " ".join(filtered[:5])
        return title if len(title) >= 2 else np.nan

    def _get_title_and_ct(self, queries_raw: list[str], queries_clean: list[str]) -> tuple[list, list]:
        """Title + ContentType для списка TypeQuery=1 запросов."""
        n = len(queries_raw)
        titles = [np.nan] * n
        cts    = [np.nan] * n

        # ContentType через ансамблевую модель
        pred_codes = self.ens_content.predict(pd.Series(queries_clean))
        ct_labels  = self.le_content.inverse_transform(pred_codes)
        for k in range(n):
            cts[k] = ct_labels[k]

        # Title: словарь + fuzzy, потом fallback; CT override из словаря
        ANIMATED = {"мультфильм", "мультсериал"}
        for k, (raw, _) in enumerate(zip(queries_raw, queries_clean)):
            if self._aliases:
                title, dict_ct, score, kp_src = self._match_title_full(raw)
                if score >= MATCH_THRESHOLD:
                    titles[k] = title
                    if dict_ct in VALID_CT:
                        if not kp_src:
                            cts[k] = dict_ct
                        elif dict_ct in ANIMATED:
                            cts[k] = dict_ct
                    continue
            titles[k] = self._fallback_title(self._clean_text(raw))

        return titles, cts

    # ------------------------------------------------------------------
    # Основной predict
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        queries = df["QueryText"].tolist()
        n = len(queries)

        type_out  = [0]      * n
        ct_out    = [np.nan] * n
        title_out = [np.nan] * n
        llm_done: set[int] = set()

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
        # Если найден известный тайтл → TypeQuery=1, title из словаря.
        # ------------------------------------------------------------------
        need_ml_final = []
        if self._aliases:
            for i in need_ml:
                title, _, score = self._match_title(queries[i])
                if score >= MATCH_THRESHOLD:
                    type_out[i]  = 1
                    title_out[i] = title
                else:
                    need_ml_final.append(i)
        else:
            need_ml_final = need_ml

        # ------------------------------------------------------------------
        # Шаг 2: ML-ансамбль
        # ------------------------------------------------------------------
        need_llm_idx = []
        ml_fallback: dict[int, tuple] = {}

        if need_ml_final:
            ml_raw     = [queries[i] for i in need_ml_final]
            ml_cleaned = [self._clean_text(q) for q in ml_raw]
            cleaned_s  = pd.Series(ml_cleaned)

            probs_type = self.ens_type.predict_proba(cleaned_s)
            proba_type = probs_type[:, 1]  # P(TypeQuery=1)
            preds      = (proba_type >= self.thresh).astype(int)

            high_conf_pos = []
            low_conf_j    = []

            for j, (tq, p1) in enumerate(zip(preds, proba_type)):
                i    = need_ml_final[j]
                conf = float(p1) if tq == 1 else float(1 - p1)

                if needs_llm(conf):
                    need_llm_idx.append(i)
                    ml_fallback[i] = (int(tq), j)
                    low_conf_j.append(j)
                else:
                    type_out[i] = int(tq)
                    if tq == 1:
                        high_conf_pos.append((i, j))

            # CT + Title для высокоуверенных TypeQuery=1
            if high_conf_pos:
                hc_raw     = [ml_raw[j]     for _, j in high_conf_pos]
                hc_cleaned = [ml_cleaned[j] for _, j in high_conf_pos]
                titles, cts = self._get_title_and_ct(hc_raw, hc_cleaned)
                for k, (i, _) in enumerate(high_conf_pos):
                    ct_out[i]    = cts[k]
                    title_out[i] = titles[k]

            # ML-fallback для LLM-bound запросов (на случай отказа LLM)
            lc_tq1 = [
                (need_ml_final[lj], lj) for lj in low_conf_j
                if ml_fallback[need_ml_final[lj]][0] == 1
            ]
            if lc_tq1:
                lc_raw     = [ml_raw[j]     for _, j in lc_tq1]
                lc_cleaned = [ml_cleaned[j] for _, j in lc_tq1]
                titles, cts = self._get_title_and_ct(lc_raw, lc_cleaned)
                for k, (i, _) in enumerate(lc_tq1):
                    ml_fallback[i] = (1, cts[k], titles[k])

            # Нормализуем fallback для TypeQuery=0 случаев
            for i in need_llm_idx:
                if len(ml_fallback[i]) == 2:
                    tq_fb = ml_fallback[i][0]
                    ml_fallback[i] = (tq_fb, np.nan, np.nan)

        # ------------------------------------------------------------------
        # Шаг 3: Yandex AI Studio — низкий confidence
        # ------------------------------------------------------------------
        if need_llm_idx:
            llm_queries = [queries[i] for i in need_llm_idx]
            llm_results = route(llm_queries)
            for i, (tq, ct, title, conf) in zip(need_llm_idx, llm_results):
                if conf <= 0.5:  # LLM недоступен → ML fallback
                    tq_ml, ct_ml, title_ml = ml_fallback[i]
                    type_out[i]  = tq_ml
                    ct_out[i]    = ct_ml    if not (isinstance(ct_ml,    float) and np.isnan(ct_ml))    else np.nan
                    title_out[i] = title_ml if not (isinstance(title_ml, float) and np.isnan(title_ml)) else np.nan
                else:
                    type_out[i]  = tq
                    ct_out[i]    = ct    if ct    else np.nan
                    title_out[i] = title if title else np.nan
                    llm_done.add(i)

        # ------------------------------------------------------------------
        # Шаг 4: Дозаполнение CT+Title для TypeQuery=1 без предсказания
        # (LLM-результаты не трогаем)
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
    model    = _get_model()
    cleaned  = pd.Series([model._clean_text(q) for q in queries])
    probs    = model.ens_type.predict_proba(cleaned)
    proba_1  = probs[:, 1]
    preds    = (proba_1 >= model.thresh).astype(int)

    results = []
    for j, (tq, p1) in enumerate(zip(preds, proba_1)):
        conf = float(p1) if tq == 1 else float(1 - p1)
        ct   = ""
        if tq == 1:
            pred_code = model.ens_content.predict(pd.Series([cleaned.iloc[j]]))
            ct        = model.le_content.inverse_transform(pred_code)[0]
        results.append((int(tq), ct, conf))
    return results


def _get_title(query: str) -> str:
    """Извлекает тайтл из запроса: сначала словарь, потом fallback."""
    model = _get_model()
    if model._aliases:
        title, _, score = model._match_title(query)
        if score >= MATCH_THRESHOLD:
            return title
    result = model._fallback_title(model._clean_text(query))
    return result if isinstance(result, str) else ""
