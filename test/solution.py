import os
import re
import pickle
import numpy as np
import pandas as pd
import gc

class PredictionModel:
    batch_size: int = 1024
    
    def __init__(self) -> None:
        self.model_dir = 'models'
        
        with open(os.path.join(self.model_dir, 'tfidf_type.pkl'), 'rb') as f:
            self.tfidf_type = pickle.load(f)
        with open(os.path.join(self.model_dir, 'model_type.pkl'), 'rb') as f:
            self.model_type = pickle.load(f)
        
        with open(os.path.join(self.model_dir, 'tfidf_content.pkl'), 'rb') as f:
            self.tfidf_content = pickle.load(f)
        with open(os.path.join(self.model_dir, 'model_content.pkl'), 'rb') as f:
            self.model_content = pickle.load(f)
        with open(os.path.join(self.model_dir, 'le_content.pkl'), 'rb') as f:
            self.le_content = pickle.load(f)
        
        with open(os.path.join(self.model_dir, 'stop_words_search.pkl'), 'rb') as f:
            self.stop_words_search = pickle.load(f)

    def _clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zа-яё0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _extract_title(self, query):
        if not isinstance(query, str) or pd.isna(query):
            return np.nan
        
        words = query.split()
        filtered_words = [w for w in words if w not in self.stop_words_search and len(w) > 1]
        
        if not filtered_words:
            return np.nan
            
        title = " ".join(filtered_words)
        
        if len(title) < 2:
            return np.nan
            
        return title

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[["QueryText"]].copy()
        
        cleaned_queries = df['QueryText'].apply(self._clean_text)
        
        X_type = self.tfidf_type.transform(cleaned_queries)
        pred_type = self.model_type.predict(X_type)
        
        out["TypeQuery"] = pred_type
        
        out["Title"] = pd.Series([np.nan] * len(out), dtype="object")
        out["ContentType"] = pd.Series([np.nan] * len(out), dtype="object")
        
        idx_positive = out[out["TypeQuery"] == 1].index
        
        if len(idx_positive) > 0:
            positive_queries = cleaned_queries.loc[idx_positive]
            
            X_content = self.tfidf_content.transform(positive_queries)
            pred_content_enc = self.model_content.predict(X_content)
            pred_content_labels = self.le_content.inverse_transform(pred_content_enc)
            
            out.loc[idx_positive, "ContentType"] = pred_content_labels
            
            titles = positive_queries.apply(self._extract_title)
            out.loc[idx_positive, "Title"] = titles
            
        gc.collect()
        
        return out