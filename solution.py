import os
import gc
import sys
import os, re, joblib, numpy as np, pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score, f1_score
from sklearn.calibration import CalibratedClassifierCV

sys.modules['__main__'] = sys.modules[__name__]

class TextEnsemble:
    """Ансамбль из трёх различных TF-IDF + классификаторов"""
    def __init__(self, n_models=3):
        self.vecs = []
        self.clfs = []
        self.n_models = n_models

    def fit(self, X_texts, y):
        # три разные конфигурации векторизаторов
        vec_configs = [
            {'max_features': 50000, 'ngram_range': (1, 2), 'sublinear_tf': True, 'min_df': 2, 'max_df': 0.95, 'smooth_idf': True},
            {'max_features': 30000, 'ngram_range': (2, 4), 'sublinear_tf': True, 'min_df': 2, 'max_df': 0.95, 'smooth_idf': True},
            {'max_features': 40000, 'ngram_range': (3, 5), 'analyzer': 'char_wb', 'sublinear_tf': True, 'min_df': 2, 'max_df': 0.95, 'smooth_idf': True}
        ]
        # соответствующие классификаторы
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

class PredictionModel:
    def __init__(self) -> None:
        os.chdir('../solution')
        self.dir = 'models'
        self.ens_type, self.thresh = joblib.load(os.path.join(self.dir, 'ens_type.pkl'))
        self.ens_content = joblib.load(os.path.join(self.dir, 'ens_content.pkl'))
        self.le_content = joblib.load(os.path.join(self.dir, 'le_content.pkl'))
        self.noise = joblib.load(os.path.join(self.dir, 'noise_words.pkl'))

    def _clean_aggressive(self, text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'[^a-zа-яё0-9\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _clean_title(self, text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'[^a-zа-яё0-9\s\-]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _extract_title(self, query):
        if not isinstance(query, str) or pd.isna(query): return np.nan
        q = query.strip().lower()
        
        quotes = re.findall(r'["«]([^"»]+)["»]', q)
        if quotes: return quotes[0].strip()
        
        q = re.sub(r'\b\d{3,4}\b', ' ', q)
        q = re.sub(r'(?:s\d{2}e\d{2}|с\d{1,3}\s?е\d{1,3}|серия\s\d+|сезон\s\d+|эпизод\s\d+|выпуск\s\d+)', ' ', q)
        q = re.sub(r'\b(?:hd|fhd|720p|1080p|4k|camrip|bdrip|webdl|webrip|ts|tc|scr|dvdrip|avi|mkv|mp4|mov|mp3|flac)\b', ' ', q)
        q = re.sub(r'[^a-zа-яё0-9\s\-]', ' ', q)
        
        words = q.split()
        valid = [w for w in words if w not in self.noise and not w.isdigit() and len(w) > 1]
        if not valid: return np.nan
        
        groups, cur = [], []
        for w in words:
            if w in self.noise or w.isdigit() or len(w) <= 1:
                if cur: groups.append(cur); cur = []
            else: cur.append(w)
        if cur: groups.append(cur)
        if not groups: return np.nan
        
        best = max(groups, key=len)
        title = " ".join(best).strip()
        return title if len(title) >= 2 else np.nan

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[["QueryText"]].copy()
        cleaned_agg = df['QueryText'].apply(self._clean_aggressive)
        
        probs = self.ens_type.predict_proba(cleaned_agg)
        out["TypeQuery"] = (probs[:, 1] >= self.thresh).astype(int)
        
        out["ContentType"] = pd.Series(np.nan, index=out.index, dtype=object)
        out["Title"] = pd.Series(np.nan, index=out.index, dtype=object)
        
        mask = out["TypeQuery"] == 1
        if mask.any():
            pos_idx = out[mask].index
            pos_q_agg = cleaned_agg.loc[pos_idx]
            pred_codes = self.ens_content.predict(pos_q_agg)
            pred_ct = self.le_content.inverse_transform(pred_codes)
            out.loc[pos_idx, "ContentType"] = pred_ct
            
            pos_q_title = df.loc[pos_idx, "QueryText"].apply(self._clean_title)
            for idx, q in zip(pos_idx, pos_q_title):
                out.loc[idx, "Title"] = self._extract_title(q)
                
        gc.collect()
        return out[["QueryText", "TypeQuery", "Title", "ContentType"]]