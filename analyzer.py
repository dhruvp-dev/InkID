"""
analyzer.py
===========
Two fully independent pipelines:

  1. verify_authorship(known_texts, test_text)
     → TF-IDF char_wb (3-4) + word (1-2) dual-vectorizer approach
     → thresholds calibrated for char-level cosine score range

  2. detect_ai(text)
     → Entropy + sentence-length variance + avg sentence length
     → Honest probability with explicit uncertainty band
     → Labeled "experimental" — no magic calibration numbers from foreign datasets

  3. analyze(known_texts, test_text)   ← the combined public API used by main.py
     → Runs both pipelines, returns merged result dict
"""

import re
import math
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── NLTK (optional) ────────────────────────────────────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords as _nltk_sw
    from nltk.tokenize import sent_tokenize as _sent_tok
    from nltk.tokenize import word_tokenize as _word_tok

    def _ensure():
        for r in ["stopwords", "punkt"]:
            try:
                nltk.data.find(
                    f"tokenizers/{r}" if "punkt" in r else f"corpora/{r}"
                )
            except LookupError:
                nltk.download(r, quiet=True)

    _ensure()
    STOP_WORDS = set(_nltk_sw.words("english"))

    def sent_tokenize(t): return _sent_tok(t)
    def word_tokenize(t): return _word_tok(t)

except ImportError:
    STOP_WORDS = {
        "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
        "yourself","yourselves","he","him","his","himself","she","her","hers",
        "herself","it","its","itself","they","them","their","theirs","themselves",
        "what","which","who","whom","this","that","these","those","am","is","are",
        "was","were","be","been","being","have","has","had","having","do","does",
        "did","doing","a","an","the","and","but","if","or","because","as","until",
        "while","of","at","by","for","with","about","against","between","into",
        "through","during","before","after","above","below","to","from","up","down",
        "in","out","on","off","over","under","again","further","then","once","here",
        "there","when","where","why","how","all","both","each","few","more","most",
        "other","some","such","no","nor","not","only","own","same","so","than",
        "too","very","s","t","can","will","just","don","should","now","d","ll",
        "m","o","re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn",
        "haven","isn","ma","mightn","mustn","needn","shan","shouldn","wasn","weren",
        "won","wouldn",
    }

    def sent_tokenize(text):
        return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]

    def word_tokenize(text):
        return re.findall(r"\b[a-zA-Z']+\b", text)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _alpha_words(text: str) -> list:
    """Return only alphabetic tokens from text."""
    return [w for w in word_tokenize(text) if re.match(r"^[a-zA-Z]+$", w)]


def _preprocess_word(text: str) -> str:
    """Lowercase, remove punctuation, strip stopwords — for word-level TF-IDF."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


def _preprocess_char(text: str) -> str:
    """Light normalisation only — char n-grams need punctuation and spaces."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_features(text: str) -> dict:
    """Compute human-readable linguistic statistics for UI display."""
    words = _alpha_words(text)
    sentences = sent_tokenize(text)
    wc = len(words)
    sc = max(len(sentences), 1)

    return {
        "word_count":          wc,
        "sentence_count":      sc,
        "avg_word_length":     round(sum(len(w) for w in words) / max(wc, 1), 2),
        "avg_sentence_length": round(wc / sc, 2),
        "unique_words":        len(set(w.lower() for w in words)),
        "lexical_diversity":   round(len(set(w.lower() for w in words)) / max(wc, 1), 3),
        "punct_density":       round(len(re.findall(r"[,;:!?]", text)) / max(wc, 1), 3),
        "paragraph_count":     max(len([p for p in text.split("\n") if p.strip()]), 1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — AUTHORSHIP VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def _dual_similarity(known_texts: list, test_text: str) -> tuple:
    """
    Compute two independent cosine similarity scores and return both.

    Score A — char_wb TF-IDF (3-4 grams)
        Captures spelling habits, morphology, punctuation style.
        Char-level cosine scores sit naturally in the 0.5–0.95 range
        for same-domain English text, so thresholds must be char-calibrated.

    Score B — word TF-IDF (1-2 grams) with stopword removal
        Captures vocabulary preference and topic fingerprint.
        Word-level cosine scores sit naturally in the 0.05–0.40 range.

    Both are normalized to [0, 1] before blending.
    Returns (char_sim, word_sim, blended_sim).
    """
    author_char = " ".join(_preprocess_char(t) for t in known_texts)
    test_char   = _preprocess_char(test_text)
    author_word = " ".join(_preprocess_word(t) for t in known_texts)
    test_word   = _preprocess_word(test_text)

    char_sim = 0.0
    word_sim = 0.0

    # ── Char TF-IDF ──────────────────────────────────────────────────────
    try:
        vec_c = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 4),
            sublinear_tf=True,
            min_df=1,
        )
        mc = vec_c.fit_transform([author_char, test_char])
        raw_c = float(cosine_similarity(mc[0:1], mc[1:2])[0][0])
        # Char cosine range: ~0.15 (unrelated English texts) to ~0.85 (near-copy)
        # Map to [0, 1] with empirically derived floor/ceil
        char_sim = float(np.clip((raw_c - 0.15) / (0.85 - 0.15), 0.0, 1.0))
    except Exception:
        char_sim = 0.0

    # ── Word TF-IDF ───────────────────────────────────────────────────────
    if author_word.strip() and test_word.strip():
        try:
            vec_w = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
            )
            mw = vec_w.fit_transform([author_word, test_word])
            raw_w = float(cosine_similarity(mw[0:1], mw[1:2])[0][0])
            # Word cosine range: ~0.00–0.35 for typical prose; map to [0, 1]
            
            word_sim = float(np.clip(raw_w / 0.35, 0.0, 1.0))
        except Exception:
            word_sim = 0.0

    # ── Blend: char carries more weight (style > topic) ───────────────────
    # 60% char (writing style) + 40% word (vocabulary/topic)
    blended = (char_sim * 0.60) + (word_sim * 0.40)
    return round(char_sim, 4), round(word_sim, 4), round(blended, 4)


def _classify_authorship(blended: float) -> tuple:
    """
    Map blended [0,1] score → (label, confidence, explanation).

    Thresholds (on normalised blended score):
      >= 0.55  → Same Author        — High
      >= 0.30  → Possibly Same      — Medium
      >= 0.12  → Possibly Different — Low
      <  0.12  → Different Author   — High
    """
    if blended >= 0.55:
        return (
            "Same Author", "High",
            "The linguistic fingerprint of the test text closely matches the known "
            "author samples. Character-level patterns, vocabulary choices, and "
            "stylistic markers are strongly consistent across all documents.",
        )
    elif blended >= 0.30:
        return (
            "Possibly Same", "Medium",
            "There is a meaningful stylistic overlap between the test text and the "
            "known samples, but it falls short of a confident match. The author may "
            "have varied their style, or the documents cover different topics. "
            "Providing more or longer samples will improve accuracy.",
        )
    elif blended >= 0.12:
        return (
            "Possibly Different", "Low",
            "The test text shares some surface-level patterns with the known samples "
            "but diverges meaningfully in vocabulary and character-level style. "
            "This could indicate a different author, a heavily edited document, or "
            "a different genre from the known samples.",
        )
    else:
        return (
            "Different Author", "High",
            "The linguistic fingerprint differs significantly from the known samples. "
            "Vocabulary distribution, phrase patterns, and character-level style all "
            "point toward a different author.",
        )


def verify_authorship(known_texts: list, test_text: str) -> dict:
    """
    Public authorship verification API.
    Returns scores and classification — no AI detection mixed in.
    """
    char_sim, word_sim, blended = _dual_similarity(known_texts, test_text)
    label, confidence, explanation = _classify_authorship(blended)

    return {
        "blended_score": blended,
        "char_score":    char_sim,
        "word_score":    word_sim,
        "label":         label,
        "confidence":    confidence,
        "explanation":   explanation,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — AI TEXT DETECTION  (separate, honest about limitations)
# ═══════════════════════════════════════════════════════════════════════════
#
# Methodology:
#   Three signals that are known to differ between human and LLM text:
#
#   (A) Shannon word entropy
#       Human writing tends to be more locally repetitive (lower entropy
#       within a passage) while LLM text is often more "uniformly diverse".
#       HIGH entropy → nudges toward AI.
#
#   (B) Sentence-length variance (burstiness)
#       Humans write burstier sentences (mix of very short and very long).
#       LLMs produce more uniform sentence lengths.
#       LOW variance → nudges toward AI.
#
#   (C) Average sentence length
#       LLMs tend to produce slightly longer, well-formed sentences on average.
#       HIGH avg length → nudges toward AI.
#
#   Important limitations:
#   - These are heuristics, not ground truth.
#   - A skilled human writer or a paraphrased LLM output can fool all three.
#   - Short texts (< 80 words) produce unreliable results.
#   - Result is displayed as a range ("likely", "possibly") not a precise %.
#
# No external calibration dataset is used. Thresholds are derived from
# first principles (information theory + known stylometric research).
# ─────────────────────────────────────────────────────────────────────────

def _word_entropy(words: list) -> float:
    """Shannon entropy of word unigram distribution."""
    if not words:
        return 0.0
    counts = Counter(w.lower() for w in words)
    total = len(words)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _sentence_variance(sentences: list) -> float:
    """Variance of per-sentence word counts."""
    if len(sentences) < 2:
        return 0.0
    lens = [len(word_tokenize(s)) for s in sentences]
    return float(np.var(lens))


def detect_ai(text: str) -> dict:
    """
    Estimate the likelihood that `text` was written by an AI.

    Returns:
        ai_probability  : float [0.0, 1.0]   raw blended score
        ai_label        : str                 "Likely Human" / "Uncertain" / "Possibly AI"
        ai_confidence   : str                 "Low" for short texts, else "Medium"
        signals         : dict                individual signal values for transparency
    """
    words     = _alpha_words(text)
    sentences = sent_tokenize(text)
    wc        = len(words)
    sc        = max(len(sentences), 1)

    # Not enough text — return uncertain
    if wc < 50:
        return {
            "ai_probability": 0.5,
            "ai_label":       "Uncertain",
            "ai_confidence":  "Low (text too short — need ≥ 50 words)",
            "signals": {
                "word_entropy":       None,
                "sentence_variance":  None,
                "avg_sentence_length": round(wc / sc, 1),
            },
        }

    entropy  = _word_entropy(words)
    variance = _sentence_variance(sentences)
    avg_len  = wc / sc

    # ── Signal A: Entropy ──────────────────────────────────────────────────
    # Human prose: ~3.5–5.5 | LLM prose: ~5.0–7.0
    # Map to [0=human, 1=AI]:  below 4.0 → 0, above 6.5 → 1
    ent_score = float(np.clip((entropy - 4.0) / (6.5 - 4.0), 0.0, 1.0))

    # ── Signal B: Sentence variance ────────────────────────────────────────
    # Human: high variance (bursty) → low AI score
    # LLM:   low variance (uniform) → high AI score
    # Map: variance > 120 → human; variance < 10 → AI
    # Inverted: high variance = more human
    var_score = float(np.clip(1.0 - (variance - 10.0) / (120.0 - 10.0), 0.0, 1.0))

    # ── Signal C: Avg sentence length ──────────────────────────────────────
    # Human: ~12–18 words | LLM: ~18–28 words
    # Map: <12 → 0, >28 → 1
    len_score = float(np.clip((avg_len - 12.0) / (28.0 - 12.0), 0.0, 1.0))

    # ── Blend (entropy=50%, variance=30%, length=20%) ─────────────────────
    ai_prob = (ent_score * 0.50) + (var_score * 0.30) + (len_score * 0.20)
    ai_prob = float(round(np.clip(ai_prob, 0.0, 1.0), 4))

    # ── Label ──────────────────────────────────────────────────────────────
    if ai_prob >= 0.65:
        label = "Possibly AI-Generated"
    elif ai_prob >= 0.40:
        label = "Uncertain"
    else:
        label = "Likely Human-Written"

    confidence = "Medium" if wc >= 60 else "Low (insufficient text — need ≥60 words)"

    return {
        "ai_probability": ai_prob,
        "ai_label":       label,
        "ai_confidence":  confidence,
        "signals": {
            "word_entropy":        round(entropy, 3),
            "sentence_variance":   round(variance, 2),
            "avg_sentence_length": round(avg_len, 1),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — COMBINED PUBLIC API  (used by main.py)
# ═══════════════════════════════════════════════════════════════════════════

def analyze(known_texts: list, test_text: str) -> dict:
    """
    Run both pipelines independently and return a merged result dict.
    Authorship result and AI detection result are kept in separate keys
    so the UI can display them without conflation.
    """
    # Features for display
    known_features_list = [extract_features(t) for t in known_texts]
    avg_known = {
        k: round(sum(f[k] for f in known_features_list) / len(known_features_list), 2)
        for k in known_features_list[0]
    }
    test_features = extract_features(test_text)

    # Pipeline 1: authorship
    auth = verify_authorship(known_texts, test_text)

    # Pipeline 2: AI detection (on the test text only)
    ai   = detect_ai(test_text)

    return {
        # ── Authorship ──────────────────────────────────────────────────
        "similarity":     auth["blended_score"],
        "similarity_pct": round(auth["blended_score"] * 100, 1),
        "char_score":     auth["char_score"],
        "word_score":     auth["word_score"],
        "label":          auth["label"],
        "confidence":     auth["confidence"],
        "explanation":    auth["explanation"],
        # ── AI Detection ────────────────────────────────────────────────
        "ai_probability":     ai["ai_probability"],
        "ai_probability_pct": round(ai["ai_probability"] * 100, 1),
        "ai_label":           ai["ai_label"],
        "ai_confidence":      ai["ai_confidence"],
        "ai_signals":         ai["signals"],
        # ── Display features ────────────────────────────────────────────
        "known_features": avg_known,
        "test_features":  test_features,
    }