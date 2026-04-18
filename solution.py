import os
import re
import json
import gc
import sys
import joblib
import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz, process
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    print("⚠️ rapidfuzz not installed. Fuzzy title matching will fall back to rule-based extraction.")

try:
    import openai as _openai_mod
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Fix для joblib при запуске в некоторых окружениях
sys.modules['__main__'] = sys.modules[__name__]

# ===========================================================================
# Ансамбль классификаторов
# ===========================================================================
class TextEnsemble:
    def __init__(self, n_models=3):
        self.vecs = []
        self.clfs = []
        self.n_models = n_models

    def fit(self, X_texts, y):
        vec_configs = [
            {'max_features': 50000, 'ngram_range': (1, 2), 'sublinear_tf': True, 'min_df': 2, 'max_df': 0.95, 'smooth_idf': True},
            {'max_features': 30000, 'ngram_range': (2, 4), 'sublinear_tf': True, 'min_df': 2, 'max_df': 0.95, 'smooth_idf': True},
            {'max_features': 40000, 'ngram_range': (3, 5), 'analyzer': 'char_wb', 'sublinear_tf': True, 'min_df': 2, 'max_df': 0.95, 'smooth_idf': True}
        ]
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression, SGDClassifier
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV

        clf_configs = [
            LogisticRegression(C=1.5, max_iter=1500, class_weight='balanced', solver='lbfgs', n_jobs=-1),
            SGDClassifier(loss='log_loss', max_iter=1500, class_weight='balanced', random_state=42, n_jobs=-1),
            CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced', max_iter=3000, random_state=42), cv=3, method='isotonic')
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


# ===========================================================================
# Конфигурация fuzzy-матчинга
# ===========================================================================
SORT_THRESHOLD  = 72
SET_THRESHOLD   = 78
MATCH_THRESHOLD = 78

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
    """Специфичная очистка для fuzzy-поиска тайтлов."""
    t = text.lower().strip().replace('ё', 'е')   # нормализация ё→е
    t = re.sub(r'[«»""\'`]', '', t)
    t = _PHRASE_STOP.sub('', t)
    t = _YEAR.sub('', t)
    t = _JUNK.sub('', t)
    return _SPC.sub(' ', t).strip()


# ===========================================================================
# LLM system prompt (Yandex AI Studio)
# ===========================================================================
_LLM_SYSTEM = """Ты — эксперт-классификатор поисковых запросов для системы Mediascope.

## Задача
Для каждого запроса определи 3 параметра и верни JSON-массив.

## ContentType
- "фильм" — полнометражный, живые актёры, не серийный
- "сериал" — многосерийный, сезоны, живые актёры (включая дорамы)
- "мультфильм" — анимация полнометражная, не серийная
- "мультсериал" — анимация серийная (включая аниме-сериал)
- "" — тип неизвестен или запрос слишком общий

## Title
Извлеки название ФРАНШИЗЫ (без года, номера части/сезона, служебных слов):
- "гарри поттер и узник азкабана" → "гарри поттер"
- "мстители финал" → "мстители"
- "холодное сердце 2" → "холодное сердце"
- Транслит → кириллица: "garri potter" → "гарри поттер", "the witcher" → "ведьмак"
- Если название неизвестно → ""
- Всегда нижний регистр

## Примеры
"garri potter smotret online" → {"ContentType":"фильм","Title":"гарри поттер"}
"igra prestolov 8 sezon" → {"ContentType":"сериал","Title":"игра престолов"}
"наруто 5 сезон все серии" → {"ContentType":"мультсериал","Title":"наруто"}
"мультфильм холодное сердце 2 онлайн" → {"ContentType":"мультфильм","Title":"холодное сердце"}
"ужасы смотреть онлайн" → {"ContentType":"","Title":""}
"дом 2 дневной эфир" → {"ContentType":"прочее","Title":"дом 2"}
"взвешенные люди" → {"ContentType":"прочее","Title":"взвешенные люди"}
"my hero academia season 4" → {"ContentType":"мультсериал","Title":"моя геройская академия"}
"пацанки 5 сезон" → {"ContentType":"прочее","Title":"пацанки"}
"the witcher season 2" → {"ContentType":"сериал","Title":"ведьмак"}
"интерстеллар онлайн" → {"ContentType":"фильм","Title":"интерстеллар"}
"канал мир смотреть" → {"ContentType":"","Title":""}

Ответь ТОЛЬКО JSON-массивом, ровно столько элементов сколько запросов:
[{"ContentType":str,"Title":str},...]"""

_LLM_VALID_CT = {"фильм", "сериал", "мультфильм", "мультсериал", "прочее", ""}
_LLM_CHUNK    = 15
_LLM_WORKERS  = 5


# ===========================================================================
# Основной класс модели
# ===========================================================================
class PredictionModel:
    def __init__(self, model_dir: str = "models") -> None:
        # Робастное разрешение пути: если относительный путь не найден,
        # используем папку models/ рядом с этим файлом
        if not os.path.isabs(model_dir) and not os.path.isdir(model_dir):
            here = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.join(here, model_dir)
            if os.path.isdir(candidate):
                model_dir = candidate
        self.dir = model_dir

        # 1. Загрузка ML-моделей
        self.ens_type, self.thresh = joblib.load(os.path.join(self.dir, 'ens_type.pkl'))
        self.ens_content           = joblib.load(os.path.join(self.dir, 'ens_content.pkl'))
        self.le_content            = joblib.load(os.path.join(self.dir, 'le_content.pkl'))
        self.noise                 = set(joblib.load(os.path.join(self.dir, 'noise_words.pkl')))

        # 2. Загрузка словаря тайтлов
        self._lookup: dict[str, tuple] = {}
        self._aliases: list[str] = []
        dict_path = os.path.join(self.dir, "titles_dict.json")
        if os.path.exists(dict_path):
            with open(dict_path, encoding="utf-8") as f:
                data = json.load(f)
            entries = data.get("titles", data) if isinstance(data, dict) else data
            for entry in entries:
                if isinstance(entry, str):
                    key = entry.replace('ё', 'е')
                    if key not in self._lookup:
                        self._lookup[key] = (entry, "", "", False)
                else:
                    canonical = entry["canonical"]
                    ct        = entry.get("content_type", "")
                    year      = entry.get("year", "")
                    kp        = entry.get("kp_source", False)
                    for alias in entry.get("aliases", [canonical]):
                        key = alias.replace('ё', 'е')
                        if key not in self._lookup:
                            self._lookup[key] = (canonical, ct, year, kp)
            self._aliases = list(self._lookup.keys())
            print(f"✅ Словарь тайтлов: {len(self._aliases)} алиасов")

        # 3. Инициализация Yandex LLM (опционально)
        self._llm = None
        self._llm_model = None
        here = os.path.dirname(os.path.abspath(__file__))
        llm_cfg_path = os.path.join(here, "llm_config.json")
        if HAS_OPENAI and os.path.exists(llm_cfg_path):
            try:
                with open(llm_cfg_path, encoding="utf-8") as f:
                    cfg = json.load(f)
                self._llm = _openai_mod.OpenAI(
                    api_key=cfg["api_key"],
                    base_url="https://ai.api.cloud.yandex.net/v1",
                    default_headers={"x-folder-id": cfg["folder_id"]},
                    timeout=90,
                )
                self._llm_model = f"gpt://{cfg['folder_id']}/{cfg['model']}"
                print(f"✅ YandexLLM: {cfg['model']}")
            except Exception as e:
                print(f"⚠️ LLM init failed: {e}")

    # ------------------------------------------------------------------
    # LLM classify
    # ------------------------------------------------------------------
    def _llm_chunk(self, queries: list[str]) -> list[tuple[str, str]]:
        """Классифицирует батч запросов через LLM. Возвращает [(ContentType, Title)]."""
        default = [("", "")] * len(queries)
        if not self._llm:
            return default
        numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(queries))
        try:
            resp = self._llm.chat.completions.create(
                model=self._llm_model,
                temperature=0.0,
                max_tokens=1500,
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM},
                    {"role": "user",   "content": f"Классифицируй запросы:\n{numbered}"},
                ],
            )
            text = resp.choices[0].message.content.strip()
            # Извлекаем JSON array
            import re as _re
            m = _re.search(r'\[.*\]', text, _re.DOTALL)
            if not m:
                return default
            results = json.loads(m.group())
            if not isinstance(results, list) or len(results) != len(queries):
                return default
            out = []
            # Маппинг нестандартных типов от LLM → наши классы
            _LLM_REMAP = {
                'реалити-шоу': 'прочее', 'реалити шоу': 'прочее', 'реалити': 'прочее',
                'тв-шоу': 'прочее', 'тв шоу': 'прочее', 'ток-шоу': 'прочее',
                'аниме': 'мультсериал', 'аниме-сериал': 'мультсериал',
                'аниме-фильм': 'мультфильм', 'мультфильм/аниме': 'мультфильм',
                'документальный': 'прочее', 'документальный фильм': 'прочее',
                'концерт': 'прочее', 'стендап': 'прочее', 'стенд-ап': 'прочее',
                'короткометражка': 'фильм',
            }
            for r in results:
                ct    = str(r.get("ContentType") or "").strip().lower()
                title = str(r.get("Title")       or "").strip().lower()
                ct = _LLM_REMAP.get(ct, ct)
                if ct not in _LLM_VALID_CT:
                    ct = ""
                out.append((ct, title))
            return out
        except Exception as e:
            print(f"[LLM] chunk failed: {e}")
            return default

    def _llm_classify(self, queries: list[str]) -> list[tuple[str, str]]:
        """Параллельная LLM-классификация батчей по _LLM_CHUNK запросов."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        chunks = [queries[i:i+_LLM_CHUNK] for i in range(0, len(queries), _LLM_CHUNK)]
        results: dict[int, list] = {}
        with ThreadPoolExecutor(max_workers=_LLM_WORKERS) as ex:
            futures = {ex.submit(self._llm_chunk, chunk): idx
                       for idx, chunk in enumerate(chunks)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        flat = []
        for i in range(len(chunks)):
            flat.extend(results[i])
        return flat

    # ------------------------------------------------------------------
    # Fuzzy-поиск + Fallback
    # ------------------------------------------------------------------
    def _match_title_fuzzy(self, query: str) -> tuple[str, str, bool]:
        """Двухуровневый fuzzy-поиск. Возвращает (title, kp_content_type, from_dict)."""
        clean = _preprocess_for_title(query)
        if not clean or len(clean) < 2:
            return np.nan, "", False

        # Точное совпадение
        if clean in self._lookup:
            entry = self._lookup[clean]
            return entry[0], entry[1], True

        # Fuzzy Stage 1: token_sort_ratio
        if HAS_RAPIDFUZZ and self._aliases:
            result = process.extractOne(
                clean, self._aliases,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=SORT_THRESHOLD,
            )
            if result:
                entry = self._lookup[result[0]]
                return entry[0], entry[1], True

            # Fuzzy Stage 2: token_set_ratio
            # Требуем ≥2 слов в алиасе, чтобы избежать false-positive
            # (например, фильм "8" матчит любой запрос с цифрой 8)
            result = process.extractOne(
                clean, self._aliases,
                scorer=fuzz.token_set_ratio,
                score_cutoff=SET_THRESHOLD,
            )
            if result and len(result[0].split()) >= 2:
                entry = self._lookup[result[0]]
                return entry[0], entry[1], True

        # Fallback (from_dict=False → кандидат для LLM)
        return self._fallback_title(query), "", False

    def _fallback_title(self, query: str) -> str:
        if not isinstance(query, str) or pd.isna(query):
            return np.nan
        q = query.strip().lower()

        # 1. Попробуем взять текст в кавычках
        quotes = re.findall(r'["«]([^"»]+)["»]', q)
        if quotes:
            return quotes[0].strip()

        # 2. Убираем числа, маркеры серий/сезонов, качество
        q = re.sub(r'\b\d{3,4}\b', ' ', q)
        q = re.sub(r'(?:s\d{2}e\d{2}|с\d{1,3}\s?е\d{1,3}|серия\s\d+|сезон\s\d+|эпизод\s\d+|выпуск\s\d+)', ' ', q)
        q = re.sub(r'\b(?:hd|fhd|720p|1080p|4k|camrip|bdrip|webdl|webrip|ts|tc|scr|dvdrip|avi|mkv|mp4|mov)\b', ' ', q)
        q = re.sub(r'[^a-zа-яё0-9\s\-]', ' ', q)

        words = q.split()
        stop_set = self.noise | _GENERIC_TITLE_WORDS

        # 3. Группируем слова, разбивая на стоп-словах → берём наибольшую группу
        groups: list[list[str]] = []
        cur: list[str] = []
        for w in words:
            if w in stop_set or w.isdigit() or len(w) <= 1:
                if cur:
                    groups.append(cur)
                    cur = []
            else:
                cur.append(w)
        if cur:
            groups.append(cur)

        if not groups:
            return np.nan

        best = max(groups, key=len)
        title = " ".join(best).strip()
        return title if len(title) >= 2 else np.nan

    def _clean_aggressive(self, text: str) -> str:
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'[^a-zа-яё0-9\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[["QueryText"]].copy()
        cleaned_agg = df['QueryText'].apply(self._clean_aggressive)

        # 1. TypeQuery
        probs = self.ens_type.predict_proba(cleaned_agg)
        out["TypeQuery"] = (probs[:, 1] >= self.thresh).astype(int)

        out["ContentType"] = pd.Series(np.nan, index=out.index, dtype=object)
        out["Title"]       = pd.Series(np.nan, index=out.index, dtype=object)

        # 2. ContentType + Title для TypeQuery == 1
        mask = out["TypeQuery"] == 1
        if mask.any():
            pos_idx = out[mask].index
            pos_q_agg = cleaned_agg.loc[pos_idx]

            ct_proba   = self.ens_content.predict_proba(pos_q_agg)
            pred_codes = np.argmax(ct_proba, axis=1)
            ct_conf    = ct_proba.max(axis=1)          # уверенность ML [0..1]
            ml_ct = self.le_content.inverse_transform(pred_codes)
            out.loc[pos_idx, "ContentType"] = ml_ct

            # Fuzzy: получаем тайтл и content_type из словаря (KP)
            raw_queries = df.loc[pos_idx, "QueryText"].tolist()
            match_results = [self._match_title_fuzzy(q) for q in raw_queries]
            titles    = [r[0] for r in match_results]
            kp_types  = [r[1] for r in match_results]
            from_dict = [r[2] for r in match_results]
            out.loc[pos_idx, "Title"] = titles

            # Корректировка ContentType по KP-словарю:
            _KP_TYPES = {'фильм', 'сериал', 'мультфильм', 'мультсериал'}
            for i, (idx, kp_ct) in enumerate(zip(pos_idx, kp_types)):
                if kp_ct not in _KP_TYPES:
                    continue
                ml_ct = out.loc[idx, "ContentType"]
                # 1. Анимация: KP надёжнее ML (фильм/сериал → мультфильм/мультсериал)
                if kp_ct in {'мультфильм', 'мультсериал'} and ml_ct in {'фильм', 'сериал'}:
                    out.loc[idx, "ContentType"] = kp_ct
                # 2. ML предсказал null/NaN, но KP знает тип → доверяем KP
                elif pd.isna(ml_ct):
                    out.loc[idx, "ContentType"] = kp_ct

            # LLM: для запросов где fuzzy не нашёл тайтл ИЛИ ML не уверен в ContentType
            # ML confidence threshold: < 0.65 → отправляем в LLM
            _CT_CONF_THR = 0.65
            if self._llm is not None:
                need_llm_title = set(i for i, fd in enumerate(from_dict) if not fd)
                need_llm_ct    = set(i for i, conf in enumerate(ct_conf) if conf < _CT_CONF_THR)
                need_llm       = sorted(need_llm_title | need_llm_ct)

                if need_llm:
                    llm_queries = [raw_queries[i] for i in need_llm]
                    llm_results = self._llm_classify(llm_queries)
                    for i, (llm_ct, llm_title) in zip(need_llm, llm_results):
                        idx = pos_idx[i]
                        # Title: обновляем только если fuzzy не нашёл (fallback)
                        if i in need_llm_title and llm_title:
                            out.loc[idx, "Title"] = llm_title
                        # ContentType: используем LLM для неуверенных предсказаний
                        if i in need_llm_ct and llm_ct:
                            out.loc[idx, "ContentType"] = llm_ct

        gc.collect()
        return out[["QueryText", "TypeQuery", "Title", "ContentType"]]
