"""
analyzer.py — Tri-Layer Hybrid Forensic Authorship Agent
Combines Word TF-IDF (Vocabulary), Char-WB TF-IDF (Syntax), and Bit-Level Profile Intersection (Micro).
"""

import re
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize

    def _ensure_nltk():
        for res in ["stopwords", "punkt", "punkt_tab"]:
            try: nltk.download(res, quiet=True)
            except: pass
    _ensure_nltk()
    _HAS_NLTK = True
except:
    def sent_tokenize(text): return re.split(r"(?<=[.!?])\s+", text.strip())
    def word_tokenize(text): return re.findall(r"\b[a-zA-Z']+\b", text)

def extract_features(text: str) -> dict:
    """Human-readable linguistic statistics for the UI."""
    words = word_tokenize(text)
    words_alpha = [w for w in words if re.match(r"^[a-zA-Z]+$", w)]
    sentences = sent_tokenize(text)

    word_count = len(words_alpha)
    sentence_count = max(len(sentences), 1)
    unique_words = len(set(w.lower() for w in words_alpha))

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": round(sum(len(w) for w in words_alpha) / max(word_count, 1), 2),
        "avg_sentence_length": round(word_count / sentence_count, 2),
        "unique_words": unique_words,
        "vocabulary_richness": round(unique_words / max(word_count, 1), 3),
        "punctuation_density": round(len(re.findall(r"[,;:!?]", text)) / max(word_count, 1), 3),
    }

def preprocess(text: str) -> str:
    """Lowercase and strip punctuation for the Vectorizers."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if len(t) > 1 or t in ["a", "i"]]
    return " ".join(tokens)


def get_hybrid_tfidf_similarities(known_texts: list, test_text: str) -> tuple:
    """Calculates both Word-level and Character-level TF-IDF similarities."""
    author_profile = " ".join(preprocess(t) for t in known_texts)
    test_processed = preprocess(test_text)

    if not author_profile.strip() or not test_processed.strip(): return 0.0, 0.0

    try:
        word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        w_matrix = word_vec.fit_transform([author_profile, test_processed])
        word_sim = float(cosine_similarity(w_matrix[0:1], w_matrix[1:2])[0][0])
    except: word_sim = 0.0

    try:
        char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=1, sublinear_tf=True)
        c_matrix = char_vec.fit_transform([author_profile, test_processed])
        char_sim = float(cosine_similarity(c_matrix[0:1], c_matrix[1:2])[0][0])
    except: char_sim = 0.0

    return word_sim, char_sim


def get_bit_profile(text: str, n: int = 11) -> dict:
    """Extracts normalized 11-bit sequences from raw text encoding."""
    binary_str = ''.join(format(byte, '08b') for byte in text.encode('utf-8'))
    if len(binary_str) < n: return {}
    
    ngrams = [binary_str[i:i+n] for i in range(len(binary_str) - n + 1)]
    counts = Counter(ngrams)
    total = len(ngrams)
    return {k: v / total for k, v in counts.items()}

def profile_intersection(prof1: dict, prof2: dict) -> float:
    """Calculates Exact Histogram Intersection. Much stricter than Euclidean."""
    keys = set(prof1.keys()).union(set(prof2.keys()))
    return sum(min(prof1.get(k, 0.0), prof2.get(k, 0.0)) for k in keys)



def classify(similarity: float):
    if similarity >= 0.55:
        return ("Same Author", "High", "green",
                "Forensic match. Vocabulary, syntax, and bit-level encoding habits align strongly.")
    elif similarity >= 0.40:
        return ("Possibly Same", "Medium", "yellow",
                "Moderate stylometric overlap. Shared structural patterns, but differing vocabulary or topics.")
    elif similarity >= 0.25:
        return ("Possibly Not Same", "Low", "orange",
                "Weak forensic footprint. Minor structural similarities exist, but core bit-level and syntax markers diverge.")
    else:
        return ("Different Author", "High", "red",
                "Authorship rejected. Significant divergence across vocabulary, syntax, and binary footprint.")

def analyze(known_texts: list, test_text: str) -> dict:
    test_features = extract_features(test_text)
    known_features_list = [extract_features(t) for t in known_texts]
    avg_known = {k: round(sum(f[k] for f in known_features_list)/len(known_features_list), 2) for k in test_features.keys()}

    word_sim, char_sim = get_hybrid_tfidf_similarities(known_texts, test_text)
    

    macro_tfidf_sim = (word_sim * 0.30) + (char_sim * 0.70)

    combined_known = " ".join(known_texts)
    known_bit_prof = get_bit_profile(combined_known, 11)
    test_bit_prof = get_bit_profile(test_text, 11)
    
    raw_bit_sim = profile_intersection(known_bit_prof, test_bit_prof)
    

    scaled_bit_sim = max(0.0, min(1.0, (raw_bit_sim - 0.60) / 0.30))


    final_similarity = (macro_tfidf_sim * 0.60) + (scaled_bit_sim * 0.40)
    
    label, confidence, color, explanation = classify(final_similarity)

    return {
        "label": label,
        "confidence": confidence,
        "color": color,
        "explanation": explanation,
        "similarity_score": round(final_similarity, 4), 
        "tfidf_score": round(macro_tfidf_sim, 4),        # UI: TF-IDF Vector (Combined Word/Char)
        "structural_score": round(scaled_bit_sim, 4),    # UI: Structural (Bit-Level)
        "sample_count": len(known_texts),               
        "known_features": avg_known,
        "test_features": test_features,
    }