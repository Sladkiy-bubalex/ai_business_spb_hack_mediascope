"""
Rule-based фильтр для быстрой классификации запросов.
Покрывает очевидные случаи без обращения к ML или LLM API.
"""
import re

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


def apply_rules(query: str) -> tuple[int | None, str, float]:
    """
    Возвращает (TypeQuery, ContentType, confidence).
    TypeQuery=None означает «не уверен — передай в ML».
    """
    q = str(query).lower()

    # Явный TypeQuery=0
    for pat in _STOP_RE:
        if pat.search(q):
            return 0, '', 0.97

    # Ищем триггеры
    hits = [p for p in _TRIGGER_RE if p.search(q)]
    if not hits:
        return None, '', 0.0  # не уверен

    # Определяем ContentType
    content_type = ''
    for ct, pats in _CT_RE:
        if any(p.search(q) for p in pats):
            content_type = ct
            break

    confidence = min(0.97, 0.75 + 0.04 * len(hits))
    return 1, content_type, confidence
