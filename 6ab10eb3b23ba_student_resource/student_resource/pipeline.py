#!/usr/bin/env python3
"""
================================================================================
Business Entity Resolution Pipeline — Amazon ML Challenge 2026
================================================================================

HOW TO RUN
----------
End-to-end (all stages):
    python pipeline.py

Single stage (e.g., blocking only):
    python pipeline.py --stage 2

Kaggle Notebook (2x T4 GPU):
    1. Upload this repo as a Kaggle dataset.
    2. Enable 2x T4 GPU in notebook settings.
    3. Install deps:  !pip install -q -r requirements.txt
    4. Run:  !python pipeline.py
    Adjust --data-dir if files are under /kaggle/input/.

AWS g4dn.xlarge (single T4):
    1. Launch g4dn.xlarge with Deep Learning AMI.
    2. pip install -r requirements.txt
    3. python pipeline.py --data-dir ./dataset --output-dir ./output

EXPECTED RUNTIME (2x T4, full dataset ~2.2M S1, ~10M S2+S3)
------------------------------------------------------------
    Stage 0 — Train/Val Split:                ~1 min
    Stage 1 — Normalization:                  ~3–8 min
    Stage 2 — Blocking / Candidate Gen:       ~30–60 min  (LaBSE encode dominates)
    Stage 3 — Feature Engineering:            ~10–20 min
    Stage 4 — Matching Model (LightGBM):      ~3–8 min
    Stage 5 — Inference on Test:              ~40–70 min
    Stage 6 — Post-processing / Validation:   ~1 min
    TOTAL:                                    ~90–170 min

DISK CACHE (skip on rerun)
--------------------------
    cache/train_val_split.pkl           — Stage 0 indices
    cache/normalized_*.pkl              — Stage 1 normalized DataFrames
    cache/tfidf_*                       — Stage 2 TF-IDF models + matrices
    cache/labse_embeddings_*.npy        — Stage 2 LaBSE embeddings
    cache/labse_id_map_*.pkl            — Stage 2 id→index maps
    cache/hnsw_index_*.bin              — Stage 2 HNSW index
    cache/bm25_model_*.pkl              — Stage 2 BM25 model
    cache/candidates_*.pkl              — Stage 2 candidates
    cache/features_*.pkl                — Stage 3 feature matrix
    cache/lgb_model.pkl                 — Stage 4 trained LightGBM
    config.json                         — Stage 4 tuned threshold

LICENSE CHECK
-------------
    LaBSE:           Apache 2.0 — 471M params ✓
    LightGBM:        MIT        — N/A         ✓
    cross-encoder/ms-marco-MiniLM-L-6-v2: Apache 2.0 — ~22M params ✓
================================================================================
"""

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import jellyfish
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from rapidfuzz import fuzz
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Global paths (overrideable via CLI)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
UTILS_DIR = os.path.join(BASE_DIR, "utils")

STAGE_TIMES = {}


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


# ===========================================================================
# SUFFIX / ADDRESS MAPS  (Terrorizer-inspired legal suffix canonicalization)
# ===========================================================================

SUFFIX_MAP = {
    "pvt": "private", "ltd": "limited", "llc": "limited liability company",
    "llp": "limited liability partnership", "corp": "corporation",
    "inc": "incorporated", "co": "company", "intl": "international",
    "mfg": "manufacturing", "svc": "services", "svcs": "services",
    "tech": "technologies", "techs": "technologies", "sys": "systems",
    "grp": "group", "mgmt": "management", "assoc": "associates",
    "assn": "association", "dept": "department", "natl": "national",
    "ent": "enterprises", "enterp": "enterprises", "hldg": "holdings",
    "hldgs": "holdings", "invst": "investments", "fin": "financial",
    "&": "and", "w/": "with",
}

ADDR_MAP = {
    "st": "street", "rd": "road", "ave": "avenue", "blvd": "boulevard",
    "apt": "apartment", "dr": "drive", "ln": "lane", "hwy": "highway",
    "nr": "near", "opp": "opposite", "adj": "adjacent",
    "no": "", "no.": "", "#": "",
    "p.o.": "po", "p.o": "po",
    "mkt": "market", "nagar": "nagar", "ng": "nagar",
}

# Pre-compiled regex patterns
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_SPACE = re.compile(r"\s+")
_LEADING_ZEROS = re.compile(r"\b0+(\d+)\b")
_DIGITS_RE = re.compile(r"\d+")


# ===========================================================================
# STAGE 1 — NORMALIZATION  (vectorized with pandas .apply)
# ===========================================================================

def _expand_tokens(text: str, token_map: dict) -> str:
    """Token-by-token expansion using a mapping dict."""
    tokens = text.split()
    expanded = []
    for tok in tokens:
        tok_clean = tok.rstrip(".,;:!?")
        if tok_clean in token_map:
            replacement = token_map[tok_clean]
            if replacement:
                expanded.append(replacement)
        else:
            expanded.append(tok)
    return " ".join(expanded)


def normalize_name_series(series: pd.Series) -> pd.Series:
    """Vectorized name normalization (Smash/Terrorizer-inspired)."""
    s = series.fillna("").astype(str).str.strip().str.lower()
    # Token-by-token suffix expansion
    s = s.apply(lambda x: _expand_tokens(x, SUFFIX_MAP) if x else "")
    # Remove punctuation except spaces
    s = s.str.replace(_PUNCT_RE, " ", regex=True)
    # Collapse whitespace
    s = s.str.replace(_MULTI_SPACE, " ", regex=True).str.strip()
    return s


def normalize_address_series(series: pd.Series) -> pd.Series:
    """Vectorized address normalization."""
    s = series.fillna("").astype(str).str.strip().str.lower()
    # Token-by-token address expansion
    s = s.apply(lambda x: _expand_tokens(x, ADDR_MAP) if x else "")
    # Strip leading zeros from standalone numbers
    s = s.str.replace(_LEADING_ZEROS, r"\1", regex=True)
    # Remove punctuation except spaces
    s = s.str.replace(_PUNCT_RE, " ", regex=True)
    s = s.str.replace(_MULTI_SPACE, " ", regex=True).str.strip()
    return s


def normalize_dataframe(df: pd.DataFrame, desc: str = "") -> pd.DataFrame:
    """Add normalized columns to a DataFrame."""
    n = len(df)
    print(f"    Normalizing {n:,} records ({desc})...")
    df = df.copy()
    df["norm_name"] = normalize_name_series(df["business_name"])
    df["norm_addr"] = normalize_address_series(df["business_address"])
    df["norm_country"] = df["country"].fillna("").astype(str).str.strip()
    print(f"    Done normalizing {desc}.")
    return df


# ===========================================================================
# STAGE 0 — TRAIN / VAL SPLIT
# ===========================================================================

def run_stage_0():
    """Stratified 80/20 split preserving singleton ratio."""
    t0 = time.time()
    print("=" * 50 + " STAGE 0: TRAIN/VAL SPLIT " + "=" * 50)

    gt_path = os.path.join(DATA_DIR, "train", "train_ground_truth.tsv")
    gt = pd.read_csv(gt_path, sep="\t", dtype=str)
    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")
    gt["has_match"] = (gt["matched_entity_ids"].str.strip() != "").astype(int)

    train_gt, val_gt = train_test_split(
        gt, test_size=0.2, stratify=gt["has_match"], random_state=42
    )
    train_gt = train_gt.reset_index(drop=True)
    val_gt = val_gt.reset_index(drop=True)

    train_singleton_pct = (train_gt["has_match"] == 0).mean() * 100
    val_singleton_pct = (val_gt["has_match"] == 0).mean() * 100

    print(f"    Train size: {len(train_gt):,}")
    print(f"    Val size:   {len(val_gt):,}")
    print(f"    Train singleton %: {train_singleton_pct:.2f}%")
    print(f"    Val singleton %:   {val_singleton_pct:.2f}%")

    split_data = {
        "train_s1_ids": set(train_gt["source1_entity_id"].tolist()),
        "val_s1_ids": set(val_gt["source1_entity_id"].tolist()),
        "train_gt": train_gt,
        "val_gt": val_gt,
    }
    joblib.dump(split_data, os.path.join(CACHE_DIR, "train_val_split.pkl"))
    print(f"    Saved split to cache/train_val_split.pkl")

    STAGE_TIMES["stage_0"] = time.time() - t0
    print(f"    Stage 0 time: {STAGE_TIMES['stage_0']:.1f}s")
    return split_data


# ===========================================================================
# STAGE 1 — NORMALIZATION
# ===========================================================================

def load_source_data(split: str = "train"):
    """Load S1, S2, S3 TSV files for a given split."""
    s1 = pd.read_csv(os.path.join(DATA_DIR, split, f"{split}_source1.tsv"),
                      sep="\t", dtype=str, na_values=[], keep_default_na=False)
    s2 = pd.read_csv(os.path.join(DATA_DIR, split, f"{split}_source2.tsv"),
                      sep="\t", dtype=str, na_values=[], keep_default_na=False)
    s3 = pd.read_csv(os.path.join(DATA_DIR, split, f"{split}_source3.tsv"),
                      sep="\t", dtype=str, na_values=[], keep_default_na=False)
    print(f"    Loaded {split}: S1={len(s1):,}, S2={len(s2):,}, S3={len(s3):,}")
    return s1, s2, s3


def run_stage_1(split: str = "train"):
    """Normalize all source data."""
    t0 = time.time()
    print("=" * 50 + f" STAGE 1: NORMALIZATION ({split}) " + "=" * 50)

    cache_file = os.path.join(CACHE_DIR, f"normalized_{split}.pkl")
    if os.path.exists(cache_file):
        print(f"    Loading cached normalized data from {cache_file}")
        data = joblib.load(cache_file)
        STAGE_TIMES[f"stage_1_{split}"] = time.time() - t0
        print(f"    Stage 1 ({split}) time: {STAGE_TIMES[f'stage_1_{split}']:.1f}s")
        return data

    s1, s2, s3 = load_source_data(split)
    s1 = normalize_dataframe(s1, f"S1-{split}")
    s2 = normalize_dataframe(s2, f"S2-{split}")
    s3 = normalize_dataframe(s3, f"S3-{split}")

    # Print sanity check: 3 before/after pairs
    print("\n    === Normalization Samples (S1) ===")
    for i in range(min(3, len(s1))):
        row = s1.iloc[i]
        print(f"    [{i}] Name:    '{row['business_name']}' -> '{row['norm_name']}'")
        print(f"         Address: '{row['business_address']}' -> '{row['norm_addr']}'")

    data = {"s1": s1, "s2": s2, "s3": s3}
    joblib.dump(data, cache_file, compress=3)
    print(f"    Saved normalized data to {cache_file}")

    STAGE_TIMES[f"stage_1_{split}"] = time.time() - t0
    print(f"    Stage 1 ({split}) time: {STAGE_TIMES[f'stage_1_{split}']:.1f}s")
    return data


# ===========================================================================
# STAGE 2 — BLOCKING / CANDIDATE GENERATION
# ===========================================================================

def build_tfidf_blocker(texts, ids, cache_prefix):
    """Build and cache TF-IDF char n-gram vectorizer + matrix.

    NOTE: This TF-IDF is kept for Stage 3 feature computation (tfidf_cosine
    features) but is NO LONGER used as a blocker. Blocker A is now MinHash LSH.
    """
    vec_path = os.path.join(CACHE_DIR, f"tfidf_{cache_prefix}_vectorizer.pkl")
    mat_path = os.path.join(CACHE_DIR, f"tfidf_{cache_prefix}_matrix.npz")
    ids_path = os.path.join(CACHE_DIR, f"tfidf_{cache_prefix}_ids.pkl")

    if os.path.exists(vec_path) and os.path.exists(mat_path):
        print(f"    Loading cached TF-IDF ({cache_prefix})...")
        vectorizer = joblib.load(vec_path)
        matrix = sparse.load_npz(mat_path)
        id_list = joblib.load(ids_path)
        return vectorizer, matrix, id_list

    print(f"    Fitting TF-IDF ({cache_prefix}) on {len(texts):,} records...")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4),
        min_df=2,  # min_df=2 to control vocab size on 10M records
        sublinear_tf=True, dtype=np.float32,
        max_features=500_000,  # cap vocab to control memory
    )
    matrix = vectorizer.fit_transform(texts)

    joblib.dump(vectorizer, vec_path)
    sparse.save_npz(mat_path, matrix)
    joblib.dump(ids, ids_path)
    print(f"    TF-IDF ({cache_prefix}): vocab={len(vectorizer.vocabulary_):,}, "
          f"matrix={matrix.shape}")
    return vectorizer, matrix, ids


# ── BLOCKER A: MinHash LSH on character trigrams ────────────────────────
# Replaces: TF-IDF char n-gram cosine (caused 540hr runtime)
# Why MinHash LSH: same character-level fuzzy recall, sublinear query time,
# no dense matrix anywhere. Uses band hashing, not dot products.
# datasketch: MIT license ✓

MINHASH_NUM_PERM = 128      # accuracy/speed tradeoff sweet spot
MINHASH_THRESHOLD = 0.15    # low = high recall; tune on val if needed


def _char_trigrams(s: str) -> set:
    """Character trigrams of a normalized string."""
    s = s.strip()
    if len(s) < 3:
        return {s}
    return {s[i:i+3] for i in range(len(s) - 2)}


def _make_minhash(text: str, num_perm: int = MINHASH_NUM_PERM) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for tg in _char_trigrams(text):
        m.update(tg.encode('utf8'))
    return m


def build_blocker_a(records_s2s3: pd.DataFrame,
                    cache_path: str = None) -> tuple:
    """
    Build MinHash LSH index over all S2+S3 norm_names.
    Returns (lsh, minhashes_dict).

    Build time: ~60-120s for 10M records.
    Query time: ~1-5ms per S1 entity.
    RAM: ~2GB. No dense matrix anywhere.

    Loads from cache if cache_path exists — skip rebuild on reruns.
    """
    if cache_path is None:
        cache_path = os.path.join(CACHE_DIR, "blocker_a.pkl")

    if os.path.exists(cache_path):
        print(f"    [Blocker A] Loading from cache: {cache_path}")
        return joblib.load(cache_path)

    print(f"    [Blocker A] Building MinHash LSH "
          f"(threshold={MINHASH_THRESHOLD}, num_perm={MINHASH_NUM_PERM}) ...")

    lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=MINHASH_NUM_PERM)
    minhashes = {}

    for i, (_, row) in enumerate(records_s2s3.iterrows()):
        eid = row['entity_id']
        m = _make_minhash(row['norm_name'])
        try:
            lsh.insert(eid, m)
        except ValueError:
            # duplicate key — skip (shouldn't happen with unique entity_ids)
            pass
        minhashes[eid] = m
        if i % 500_000 == 0 and i > 0:
            print(f"      indexed {i:,} records ...")

    os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".", exist_ok=True)
    joblib.dump((lsh, minhashes), cache_path, compress=3)
    print(f"    [Blocker A] Done. Indexed {len(minhashes):,} records. Saved to {cache_path}")
    return lsh, minhashes


def query_blocker_a(norm_name_s1: str, lsh: MinHashLSH) -> set:
    """Query MinHash LSH for a single S1 norm_name."""
    m = _make_minhash(norm_name_s1)
    return set(lsh.query(m))


# ── BLOCKER D: Token Inverted Index ─────────────────────────────────────
# Catches: exact important-word matches BM25 misses due to IDF weighting
# Example: "Reliance" appears in many records → BM25 downweights it.
#          But sharing "Reliance" is still strong blocking signal.
# Build time: ~3-10 seconds. Query time: <1ms. RAM: <100MB.

# Tokens too common across ALL businesses to carry identity signal.
# Keep this list minimal — over-filtering kills recall.
_BSTOP = {
    'the', 'and', 'of', 'a', 'an', 'in', 'for', 'at', 'by', 'to',
    'company', 'limited', 'private', 'corporation', 'incorporated',
    'services', 'group', 'international', 'national',
    # deliberately NOT including: technologies, systems, solutions,
    # industries, enterprises — these carry domain identity signal
}


def build_blocker_d(records_s2s3: pd.DataFrame,
                    min_token_len: int = 4) -> dict:
    """
    Build token → set[entity_id] inverted index over S2+S3 norm_names.
    Only indexes tokens of length >= min_token_len not in _BSTOP.
    """
    print("    [Blocker D] Building token inverted index ...")
    index = defaultdict(set)
    for _, row in records_s2s3.iterrows():
        for token in row['norm_name'].split():
            if len(token) >= min_token_len and token not in _BSTOP:
                index[token].add(row['entity_id'])
    print(f"    [Blocker D] Done. Unique index tokens: {len(index):,}")
    return dict(index)


def query_blocker_d(norm_name_s1: str,
                    index: dict,
                    min_shared: int = 1,
                    min_token_len: int = 4) -> set:
    """
    Return all S2+S3 entity_ids sharing >= min_shared content tokens
    with the S1 name.
    min_shared=1 → maximum recall (correct for business names)
    """
    tokens = {t for t in norm_name_s1.split()
              if len(t) >= min_token_len and t not in _BSTOP}
    if not tokens:
        return set()
    hits = defaultdict(int)
    for token in tokens:
        for eid in index.get(token, set()):
            hits[eid] += 1
    return {eid for eid, cnt in hits.items() if cnt >= min_shared}


def encode_labse_batched(entity_ids, texts, cache_name, batch_size=512):
    """Encode texts with LaBSE in batches and cache to disk."""
    emb_path = os.path.join(CACHE_DIR, f"labse_embeddings_{cache_name}.npy")
    ids_path = os.path.join(CACHE_DIR, f"labse_id_map_{cache_name}.pkl")

    if os.path.exists(emb_path) and os.path.exists(ids_path):
        print(f"    Loading cached LaBSE embeddings ({cache_name})...")
        embeddings = np.load(emb_path, mmap_mode="r")
        id_map = joblib.load(ids_path)
        return embeddings, id_map

    print(f"    Encoding {len(texts):,} texts with LaBSE ({cache_name})...")
    from sentence_transformers import SentenceTransformer
    # LaBSE: Apache 2.0 license, 471M parameters — within 8B cap ✓
    device = "cuda" if _has_cuda() else "cpu"
    model = SentenceTransformer("sentence-transformers/LaBSE", device=device)

    # Encode in batches and write to memory-mapped file for large datasets
    dim = 768
    n = len(texts)
    fp = np.memmap(emb_path, dtype=np.float32, mode="w+", shape=(n, dim))

    for start in tqdm(range(0, n, batch_size), desc=f"    LaBSE {cache_name}"):
        end = min(start + batch_size, n)
        batch_embs = model.encode(
            texts[start:end], batch_size=batch_size,
            show_progress_bar=False, convert_to_numpy=True,
            normalize_embeddings=True
        )
        fp[start:end] = batch_embs

    fp.flush()
    del fp, model
    gc.collect()

    id_map = {eid: idx for idx, eid in enumerate(entity_ids)}
    joblib.dump(id_map, ids_path)

    embeddings = np.load(emb_path, mmap_mode="r")
    print(f"    LaBSE ({cache_name}): shape=({n}, {dim})")
    return embeddings, id_map


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def build_hnsw_index(embeddings, cache_name="s2s3"):
    """Build HNSW index on embeddings."""
    import hnswlib
    idx_path = os.path.join(CACHE_DIR, f"hnsw_index_{cache_name}.bin")

    n, dim = embeddings.shape
    if os.path.exists(idx_path):
        print(f"    Loading cached HNSW index ({cache_name})...")
        index = hnswlib.Index(space="cosine", dim=dim)
        index.load_index(idx_path, max_elements=n)
        index.set_ef(50)
        return index

    print(f"    Building HNSW index on {n:,} vectors (dim={dim})...")
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=n, ef_construction=200, M=16)

    # Add in batches to control memory
    batch = 100_000
    for start in tqdm(range(0, n, batch), desc="    HNSW add"):
        end = min(start + batch, n)
        # Read from memmap in chunks
        chunk = np.array(embeddings[start:end])
        index.add_items(chunk, np.arange(start, end))

    index.set_ef(50)
    index.save_index(idx_path)
    print(f"    HNSW index saved ({cache_name})")
    return index


def build_bm25_model(corpus_tokens, cache_name="s2s3"):
    """Build BM25 model on tokenized corpus.

    BM25Okapi on 10M+ docs can use 20–40 GB RAM. If memory is insufficient,
    returns None and the pipeline falls back to 2-blocker mode (HNSW + MinHash).
    """
    bm25_path = os.path.join(CACHE_DIR, f"bm25_model_{cache_name}.pkl")
    if os.path.exists(bm25_path):
        print(f"    Loading cached BM25 ({cache_name})...")
        try:
            return joblib.load(bm25_path)
        except MemoryError:
            print("    WARNING: Not enough memory to load BM25 cache. Skipping Blocker C.")
            return None

    print(f"    Building BM25 on {len(corpus_tokens):,} documents...")
    try:
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi(corpus_tokens)
        joblib.dump(bm25, bm25_path, compress=3)
        return bm25
    except MemoryError:
        print("    WARNING: Not enough memory to build BM25. Skipping Blocker C.")
        print("    Pipeline will use 2 blockers (MinHash + HNSW) instead of 4.")
        return None


def run_hnsw_blocking(s1_embeddings, hnsw_index, top_k=25, batch_size=10000):
    """HNSW nearest neighbor blocking."""
    n = s1_embeddings.shape[0]
    all_labels = []
    all_distances = []
    k = min(top_k, hnsw_index.get_current_count())
    for start in tqdm(range(0, n, batch_size), desc="    Blocker B (HNSW)"):
        end = min(start + batch_size, n)
        chunk = np.array(s1_embeddings[start:end])
        labels, distances = hnsw_index.knn_query(chunk, k=k)
        all_labels.append(labels)
        all_distances.append(distances)
    return np.vstack(all_labels), np.vstack(all_distances)


def run_bm25_blocking(s1_queries, bm25, s2s3_ids, top_k=15):
    """BM25 blocking on combined name+address."""
    results = {}
    for i, query_tokens in enumerate(tqdm(s1_queries, desc="    Blocker C (BM25)")):
        if not query_tokens:
            results[i] = []
            continue
        scores = bm25.get_scores(query_tokens)
        if len(scores) == 0:
            results[i] = []
            continue
        k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])][::-1]
        cands = [(s2s3_ids[j], float(scores[j])) for j in top_indices if scores[j] > 0]
        results[i] = cands
    return results


def union_and_cap_candidates(s1_ids, s1_names, lsh, blocker_b_labels,
                              blocker_b_dists, blocker_c, s2s3_ids,
                              token_index, cap=60):
    """Union all 4 blocker results and cap per S1 entity.

    Blockers:
      A — MinHash LSH (char trigram fuzzy)
      B — LaBSE HNSW (semantic, multilingual)
      C — BM25 (ranked word match)
      D — Token Inverted Index (exact content words)

    Priority order when capping: B > A > C > D
    """
    candidates = {}
    bm25_scores_cache = {}

    for i, s1_id in enumerate(tqdm(s1_ids, desc="    Union candidates")):
        norm_name = s1_names[i] if i < len(s1_names) else ""

        # ── Blocker B: LaBSE HNSW (semantic) ─────────────────────────
        b_cands = set()
        if i < len(blocker_b_labels):
            for idx in blocker_b_labels[i]:
                if idx < len(s2s3_ids):
                    b_cands.add(s2s3_ids[idx])

        # ── Blocker A: MinHash LSH (char trigram fuzzy) ──────────────
        a_cands = query_blocker_a(norm_name, lsh) if lsh is not None else set()

        # ── Blocker C: BM25 (ranked word match) ──────────────────────
        c_cands_scored = blocker_c.get(i, [])
        c_cands = set(c[0] for c in c_cands_scored)

        # Store BM25 scores for feature 14
        for cid, score in c_cands_scored:
            bm25_scores_cache[(s1_id, cid)] = score

        # ── Blocker D: Token inverted index (exact content words) ────
        d_cands = query_blocker_d(norm_name, token_index) if token_index else set()

        # Union of all 4 blockers
        all_cands = b_cands | a_cands | c_cands | d_cands

        # Cap at limit with priority-ordered fill
        if len(all_cands) > cap:
            # Priority: B (semantic) > A (fuzzy char) > C (ranked word) > D (exact word)
            kept = set(b_cands)                       # always keep all HNSW
            for pool in [a_cands, c_cands, d_cands]:
                if len(kept) >= cap:
                    break
                remaining = cap - len(kept)
                kept |= set(list(pool - kept)[:remaining])
            all_cands = kept

        candidates[s1_id] = list(all_cands)

    return candidates, bm25_scores_cache


def write_candidate_pairs(candidates, s1_ids_ordered):
    """Write candidate_pairs.tsv."""
    cand_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    print(f"    Writing {cand_path}...")
    with open(cand_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_ids_ordered:
            cands = candidates.get(s1_id, [])
            cands = [c for c in cands if c.startswith(("S2-", "S3-"))]
            cands = list(dict.fromkeys(cands))
            f.write(f"{s1_id}\t{','.join(cands)}\n")


def run_stage_2(data, split_data=None, mode="train"):
    """Run all four blockers and union results.

    Blockers:
      A — MinHash LSH on character trigrams (fuzzy name match)
      B — LaBSE HNSW (semantic, multilingual embedding match)
      C — BM25 (ranked word match on name+address)
      D — Token Inverted Index (exact content-word match)
    """
    t0 = time.time()
    print("=" * 50 + f" STAGE 2: BLOCKING ({mode}) " + "=" * 50)

    cache_file = os.path.join(CACHE_DIR, f"candidates_{mode}.pkl")
    if os.path.exists(cache_file):
        print(f"    Loading cached candidates from {cache_file}")
        result = joblib.load(cache_file)
        STAGE_TIMES[f"stage_2_{mode}"] = time.time() - t0
        print(f"    Stage 2 ({mode}) time: {STAGE_TIMES[f'stage_2_{mode}']:.1f}s (cached)")
        return result

    s1, s2, s3 = data["s1"], data["s2"], data["s3"]

    # Combine S2 + S3
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    s2s3_ids = s2s3["entity_id"].values.tolist()
    s2s3_names = s2s3["norm_name"].values.tolist()
    s2s3_addrs = s2s3["norm_addr"].values.tolist()
    s2s3_combined = [f"{n} {a}" for n, a in zip(s2s3_names, s2s3_addrs)]

    s1_ids = s1["entity_id"].values.tolist()
    s1_names = s1["norm_name"].values.tolist()
    s1_addrs = s1["norm_addr"].values.tolist()
    s1_combined = [f"{n} {a}" for n, a in zip(s1_names, s1_addrs)]

    print(f"    S1 entities: {len(s1_ids):,}")
    print(f"    S2+S3 entities: {len(s2s3_ids):,}")

    # ---- Build TF-IDF on name (for Stage 3 feature reuse ONLY, NOT for blocking) ----
    tfidf_name_vec, tfidf_name_mat, tfidf_name_ids = build_tfidf_blocker(
        s2s3_names, s2s3_ids, cache_prefix=f"name_{mode}"
    )

    # ---- Also build TF-IDF on address (for feature reuse in Stage 3) ----
    tfidf_addr_vec, tfidf_addr_mat, tfidf_addr_ids = build_tfidf_blocker(
        s2s3_addrs, s2s3_ids, cache_prefix=f"addr_{mode}"
    )

    # ---- Blocker A: MinHash LSH on char trigrams ----
    print("\n    Building Blocker A (MinHash LSH on char trigrams) ...")
    t_a = time.time()
    lsh, minhashes = build_blocker_a(
        s2s3, cache_path=os.path.join(CACHE_DIR, f"blocker_a_{mode}.pkl")
    )
    print(f"    Blocker A ready in {time.time()-t_a:.1f}s")

    # ---- Blocker D: Token Inverted Index ----
    print("\n    Building Blocker D (token inverted index) ...")
    t_d = time.time()
    token_index = build_blocker_d(s2s3)
    print(f"    Blocker D ready in {time.time()-t_d:.1f}s")

    # ---- Blocker B: LaBSE + HNSW ----
    s2s3_embs, s2s3_id_map = encode_labse_batched(
        s2s3_ids, s2s3_names, cache_name=f"s2s3_{mode}"
    )
    hnsw_index = build_hnsw_index(s2s3_embs, cache_name=f"s2s3_{mode}")

    s1_embs, s1_id_map = encode_labse_batched(
        s1_ids, s1_names, cache_name=f"s1_{mode}"
    )
    b_labels, b_dists = run_hnsw_blocking(s1_embs, hnsw_index, top_k=25)

    # ---- Blocker C: BM25 ----
    corpus_tokens = [doc.split() for doc in s2s3_combined]
    bm25 = build_bm25_model(corpus_tokens, cache_name=f"s2s3_{mode}")
    if bm25 is not None:
        s1_query_tokens = [doc.split() for doc in s1_combined]
        blocker_c = run_bm25_blocking(s1_query_tokens, bm25, s2s3_ids, top_k=15)
        del s1_query_tokens
    else:
        blocker_c = {}  # empty — pipeline degrades gracefully
        print("    Blocker C skipped (BM25 unavailable)")

    # Free memory
    del corpus_tokens, bm25
    gc.collect()

    # ---- Union and cap (all 4 blockers) ----
    candidates, bm25_scores = union_and_cap_candidates(
        s1_ids, s1_names, lsh, b_labels, b_dists, blocker_c, s2s3_ids,
        token_index, cap=60
    )

    # Free blocker memory
    del lsh, minhashes, token_index
    gc.collect()

    result = {
        "candidates": candidates,
        "bm25_scores": bm25_scores,
        "tfidf_name_prefix": f"name_{mode}",
        "tfidf_addr_prefix": f"addr_{mode}",
        "s2s3_ids": s2s3_ids,
    }

    joblib.dump(result, cache_file, compress=3)
    print(f"    Saved candidates to {cache_file}")

    # ---- Report ----
    total_pairs = sum(len(v) for v in candidates.values())
    mean_cands = total_pairs / max(len(candidates), 1)
    print(f"\n    Total (S1, candidate) pairs: {total_pairs:,}")
    print(f"    Mean candidates per S1 entity: {mean_cands:.1f}")

    # Blocking recall on val set
    if mode == "train" and split_data is not None:
        val_gt = split_data["val_gt"]
        val_s1_ids = split_data["val_s1_ids"]
        total_true = 0
        found_true = 0
        for _, row in val_gt.iterrows():
            s1_id = row["source1_entity_id"]
            true_matches = row["matched_entity_ids"]
            if not true_matches or not str(true_matches).strip():
                continue
            true_ids = set(str(true_matches).split(","))
            cand_set = set(candidates.get(s1_id, []))
            total_true += len(true_ids)
            found_true += len(true_ids & cand_set)
        blocking_recall = found_true / max(total_true, 1)
        print(f"    Blocking recall on val: {blocking_recall:.4f} ({found_true:,}/{total_true:,})")
        print(f"    TARGET: blocking recall > 0.88")

    # Write candidate_pairs.tsv for test mode
    if mode == "test":
        write_candidate_pairs(candidates, s1_ids)

    STAGE_TIMES[f"stage_2_{mode}"] = time.time() - t0
    print(f"    Stage 2 ({mode}) time: {STAGE_TIMES[f'stage_2_{mode}']:.1f}s")
    return result


# ===========================================================================
# STAGE 3 — FEATURE ENGINEERING
# ===========================================================================

def _char_ngrams(s, n=3):
    """Character n-gram set."""
    if len(s) < n:
        return set()
    return set(s[i:i + n] for i in range(len(s) - n + 1))


def _jaccard(set_a, set_b):
    """Jaccard similarity."""
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union_size = len(set_a | set_b)
    return inter / union_size if union_size > 0 else 0.0


FEATURE_NAMES = [
    "jaro_winkler_name",         # 1
    "token_set_ratio_name",      # 2
    "token_sort_ratio_name",     # 3
    "partial_ratio_name",        # 4
    "tfidf_cosine_name",         # 5
    "char_3gram_jaccard_name",   # 6
    "common_token_ratio_name",   # 7
    "length_ratio_name",         # 8
    "token_set_ratio_address",   # 9
    "char_3gram_jaccard_address",# 10
    "common_number_match",       # 11
    "address_tfidf_cosine",      # 12
    "country_match",             # 13
    "bm25_score_normalized",     # 14
    "embedding_cosine_labse",    # 15
]


def compute_features_batch(pairs_data):
    """Compute 15 features for a list of (s1_data, cand_data, extra) tuples.

    Each element of pairs_data is a dict with keys:
        nn_s1, nn_cand, na_s1, na_cand, country_s1, country_cand,
        tfidf_name_cos, tfidf_addr_cos, labse_cos, bm25_norm
    """
    features = np.zeros((len(pairs_data), 15), dtype=np.float32)

    for i, d in enumerate(pairs_data):
        nn_s1 = d["nn_s1"]
        nn_cand = d["nn_cand"]
        na_s1 = d["na_s1"]
        na_cand = d["na_cand"]
        both_names = bool(nn_s1 and nn_cand)
        addr_empty = (not na_s1) or (not na_cand)

        # 1. jaro_winkler_name
        features[i, 0] = jellyfish.jaro_winkler_similarity(nn_s1, nn_cand) if both_names else 0.0
        # 2. token_set_ratio_name
        features[i, 1] = fuzz.token_set_ratio(nn_s1, nn_cand) / 100.0 if both_names else 0.0
        # 3. token_sort_ratio_name
        features[i, 2] = fuzz.token_sort_ratio(nn_s1, nn_cand) / 100.0 if both_names else 0.0
        # 4. partial_ratio_name
        features[i, 3] = fuzz.partial_ratio(nn_s1, nn_cand) / 100.0 if both_names else 0.0
        # 5. tfidf_cosine_name (pre-computed)
        features[i, 4] = d["tfidf_name_cos"]
        # 6. char_3gram_jaccard_name
        features[i, 5] = _jaccard(_char_ngrams(nn_s1, 3), _char_ngrams(nn_cand, 3))
        # 7. common_token_ratio_name
        tokens_s1 = set(nn_s1.split()) if nn_s1 else set()
        tokens_cand = set(nn_cand.split()) if nn_cand else set()
        features[i, 6] = _jaccard(tokens_s1, tokens_cand)
        # 8. length_ratio_name
        ls1 = len(nn_s1) if nn_s1 else 0
        lc = len(nn_cand) if nn_cand else 0
        features[i, 7] = min(ls1, lc) / (max(ls1, lc) + 1e-9)
        # 9. token_set_ratio_address
        features[i, 8] = 0.5 if addr_empty else fuzz.token_set_ratio(na_s1, na_cand) / 100.0
        # 10. char_3gram_jaccard_address
        features[i, 9] = 0.5 if addr_empty else _jaccard(_char_ngrams(na_s1, 3), _char_ngrams(na_cand, 3))
        # 11. common_number_match
        nums_s1 = set(_DIGITS_RE.findall(na_s1)) if na_s1 else set()
        nums_cand = set(_DIGITS_RE.findall(na_cand)) if na_cand else set()
        if not nums_s1 or not nums_cand:
            features[i, 10] = 0.5
        elif nums_s1 & nums_cand:
            features[i, 10] = 1.0
        else:
            features[i, 10] = 0.0
        # 12. address_tfidf_cosine (pre-computed)
        features[i, 11] = 0.5 if addr_empty else d["tfidf_addr_cos"]
        # 13. country_match
        c_s1 = d["country_s1"].strip().lower() if d["country_s1"] else ""
        c_cand = d["country_cand"].strip().lower() if d["country_cand"] else ""
        features[i, 12] = 1.0 if c_s1 and c_cand and c_s1 == c_cand else 0.0
        # 14. bm25_score_normalized (pre-computed)
        features[i, 13] = d["bm25_norm"]
        # 15. embedding_cosine_labse (pre-computed)
        features[i, 14] = d["labse_cos"]

    return features


def run_stage_3(data, blocking_result, split_data=None, mode="train"):
    """Build feature matrix for all (S1, candidate) pairs."""
    t0 = time.time()
    print("=" * 50 + f" STAGE 3: FEATURE ENGINEERING ({mode}) " + "=" * 50)

    cache_file = os.path.join(CACHE_DIR, f"features_{mode}.pkl")
    if os.path.exists(cache_file):
        print(f"    Loading cached features from {cache_file}")
        feat_data = joblib.load(cache_file)
        STAGE_TIMES[f"stage_3_{mode}"] = time.time() - t0
        print(f"    Stage 3 ({mode}) time: {STAGE_TIMES[f'stage_3_{mode}']:.1f}s (cached)")
        return feat_data

    s1, s2, s3 = data["s1"], data["s2"], data["s3"]
    candidates = blocking_result["candidates"]
    bm25_scores = blocking_result["bm25_scores"]

    # Build lookup dicts
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    print(f"    Building lookups for {len(s2s3):,} S2+S3 records...")
    s2s3_name_dict = dict(zip(s2s3["entity_id"], s2s3["norm_name"]))
    s2s3_addr_dict = dict(zip(s2s3["entity_id"], s2s3["norm_addr"]))
    s2s3_country_dict = dict(zip(s2s3["entity_id"], s2s3["norm_country"]))
    s1_name_dict = dict(zip(s1["entity_id"], s1["norm_name"]))
    s1_addr_dict = dict(zip(s1["entity_id"], s1["norm_addr"]))
    s1_country_dict = dict(zip(s1["entity_id"], s1["norm_country"]))
    del s2s3
    gc.collect()

    # Load TF-IDF vectorizers and matrices
    tfidf_name_prefix = blocking_result["tfidf_name_prefix"]
    tfidf_addr_prefix = blocking_result["tfidf_addr_prefix"]
    tfidf_name_vec = joblib.load(os.path.join(CACHE_DIR, f"tfidf_{tfidf_name_prefix}_vectorizer.pkl"))
    tfidf_name_mat = sparse.load_npz(os.path.join(CACHE_DIR, f"tfidf_{tfidf_name_prefix}_matrix.npz"))
    tfidf_name_ids = joblib.load(os.path.join(CACHE_DIR, f"tfidf_{tfidf_name_prefix}_ids.pkl"))
    tfidf_addr_vec = joblib.load(os.path.join(CACHE_DIR, f"tfidf_{tfidf_addr_prefix}_vectorizer.pkl"))
    tfidf_addr_mat = sparse.load_npz(os.path.join(CACHE_DIR, f"tfidf_{tfidf_addr_prefix}_matrix.npz"))
    tfidf_addr_ids = joblib.load(os.path.join(CACHE_DIR, f"tfidf_{tfidf_addr_prefix}_ids.pkl"))

    name_id2idx = {eid: idx for idx, eid in enumerate(tfidf_name_ids)}
    addr_id2idx = {eid: idx for idx, eid in enumerate(tfidf_addr_ids)}

    # Load LaBSE embeddings
    s1_embs = np.load(os.path.join(CACHE_DIR, f"labse_embeddings_s1_{mode}.npy"), mmap_mode="r")
    s1_id_map = joblib.load(os.path.join(CACHE_DIR, f"labse_id_map_s1_{mode}.pkl"))
    s2s3_embs = np.load(os.path.join(CACHE_DIR, f"labse_embeddings_s2s3_{mode}.npy"), mmap_mode="r")
    s2s3_id_map = joblib.load(os.path.join(CACHE_DIR, f"labse_id_map_s2s3_{mode}.pkl"))

    # Build ground truth labels (train mode)
    gt_dict = {}
    if mode == "train" and split_data is not None:
        full_gt = pd.concat([split_data["train_gt"], split_data["val_gt"]], ignore_index=True)
        for _, row in full_gt.iterrows():
            s1_id = row["source1_entity_id"]
            matches = str(row["matched_entity_ids"]).strip()
            gt_dict[s1_id] = set(matches.split(",")) if matches else set()

    # Compute features in chunks per S1 entity
    all_features = []
    all_labels = []
    all_pairs = []

    total_pairs = sum(len(v) for v in candidates.values())
    print(f"    Computing features for {total_pairs:,} pairs...")

    processed = 0
    for s1_id in tqdm(candidates.keys(), desc="    Features"):
        cand_ids = candidates[s1_id]
        if not cand_ids:
            continue

        nn_s1 = s1_name_dict.get(s1_id, "")
        na_s1 = s1_addr_dict.get(s1_id, "")
        country_s1 = s1_country_dict.get(s1_id, "")

        # S1 TF-IDF vectors (once per S1 entity)
        s1_tfidf_name_vec = tfidf_name_vec.transform([nn_s1])
        s1_tfidf_addr_vec = tfidf_addr_vec.transform([na_s1])

        # S1 LaBSE embedding
        s1_emb_idx = s1_id_map.get(s1_id)
        s1_labse = np.array(s1_embs[s1_emb_idx]) if s1_emb_idx is not None else None

        # Max BM25 score for normalization
        max_bm25 = max(
            (bm25_scores.get((s1_id, cid), 0.0) for cid in cand_ids),
            default=0.0
        )

        batch_data = []
        for cand_id in cand_ids:
            nn_cand = s2s3_name_dict.get(cand_id, "")
            na_cand = s2s3_addr_dict.get(cand_id, "")
            country_cand = s2s3_country_dict.get(cand_id, "")

            # TF-IDF cosine name
            cand_name_idx = name_id2idx.get(cand_id)
            tfidf_name_cos = 0.0
            if cand_name_idx is not None:
                sim = sk_cosine_similarity(
                    s1_tfidf_name_vec, tfidf_name_mat[cand_name_idx:cand_name_idx + 1]
                )
                tfidf_name_cos = float(sim[0, 0])

            # TF-IDF cosine address
            tfidf_addr_cos = 0.0
            if na_s1 and na_cand:
                cand_addr_idx = addr_id2idx.get(cand_id)
                if cand_addr_idx is not None:
                    sim = sk_cosine_similarity(
                        s1_tfidf_addr_vec, tfidf_addr_mat[cand_addr_idx:cand_addr_idx + 1]
                    )
                    tfidf_addr_cos = float(sim[0, 0])

            # LaBSE cosine
            labse_cos = 0.0
            if s1_labse is not None:
                cand_emb_idx = s2s3_id_map.get(cand_id)
                if cand_emb_idx is not None:
                    cand_labse = np.array(s2s3_embs[cand_emb_idx])
                    labse_cos = float(np.dot(s1_labse, cand_labse))

            # BM25 normalized
            bm25_val = bm25_scores.get((s1_id, cand_id), 0.0)
            bm25_norm = bm25_val / (max_bm25 + 1e-9) if max_bm25 > 0 else 0.0

            batch_data.append({
                "nn_s1": nn_s1, "nn_cand": nn_cand,
                "na_s1": na_s1, "na_cand": na_cand,
                "country_s1": country_s1, "country_cand": country_cand,
                "tfidf_name_cos": tfidf_name_cos,
                "tfidf_addr_cos": tfidf_addr_cos,
                "labse_cos": labse_cos,
                "bm25_norm": bm25_norm,
            })

            all_pairs.append((s1_id, cand_id))
            if mode == "train":
                true_matches = gt_dict.get(s1_id, set())
                all_labels.append(1 if cand_id in true_matches else 0)

        feats = compute_features_batch(batch_data)
        all_features.append(feats)
        processed += len(cand_ids)

    X = np.vstack(all_features) if all_features else np.zeros((0, 15), dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32) if all_labels else np.array([])

    feat_data = {
        "X": X, "y": y, "pairs": all_pairs,
        "feature_names": FEATURE_NAMES
    }
    joblib.dump(feat_data, cache_file, compress=3)

    print(f"\n    Feature matrix shape: {X.shape}")
    if len(y) > 0:
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        print(f"    Positive pairs: {pos:,}")
        print(f"    Negative pairs: {neg:,}")
        print(f"    Ratio (neg/pos): {neg / max(pos, 1):.1f}:1")

    STAGE_TIMES[f"stage_3_{mode}"] = time.time() - t0
    print(f"    Stage 3 ({mode}) time: {STAGE_TIMES[f'stage_3_{mode}']:.1f}s")
    return feat_data


# ===========================================================================
# STAGE 4 — MATCHING MODEL
# ===========================================================================

def compute_f05(precision, recall):
    """F0.5 score."""
    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def macro_f05_on_val(probs, threshold, pairs, val_gt, val_s1_ids):
    """Compute macro F0.5 across all val S1 entities, including singletons."""
    pred_matches = defaultdict(set)
    for i, (s1_id, cand_id) in enumerate(pairs):
        if s1_id in val_s1_ids and probs[i] >= threshold:
            pred_matches[s1_id].add(cand_id)

    gt_dict = {}
    for _, row in val_gt.iterrows():
        s1_id = row["source1_entity_id"]
        matches = str(row["matched_entity_ids"]).strip()
        gt_dict[s1_id] = set(matches.split(",")) if matches else set()

    f05_scores = []
    for s1_id in val_s1_ids:
        true_set = gt_dict.get(s1_id, set())
        pred_set = pred_matches.get(s1_id, set())

        if not true_set and not pred_set:
            f05_scores.append(1.0)  # correct singleton
        elif not true_set and pred_set:
            f05_scores.append(0.0)  # false positive on singleton
        elif true_set and not pred_set:
            f05_scores.append(0.0)  # missed all
        else:
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f05_scores.append(compute_f05(p, r))

    return np.mean(f05_scores) if f05_scores else 0.0


def run_stage_4(feat_data, split_data):
    """Train LightGBM classifier and tune threshold for F0.5."""
    t0 = time.time()
    print("=" * 50 + " STAGE 4: MATCHING MODEL " + "=" * 50)

    X = feat_data["X"]
    y = feat_data["y"]
    pairs = feat_data["pairs"]

    train_s1_ids = split_data["train_s1_ids"]
    val_s1_ids = split_data["val_s1_ids"]
    val_gt = split_data["val_gt"]

    # Split by S1 entity membership
    train_mask = np.array([p[0] in train_s1_ids for p in pairs])
    val_mask = np.array([p[0] in val_s1_ids for p in pairs])

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    val_pairs = [p for p, m in zip(pairs, val_mask) if m]

    print(f"    Train samples: {len(X_train):,}  (pos={int(y_train.sum()):,}, neg={int((y_train == 0).sum()):,})")
    print(f"    Val samples:   {len(X_val):,}  (pos={int(y_val.sum()):,}, neg={int((y_val == 0).sum()):,})")

    class_imbalance = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"    Class imbalance (neg/pos): {class_imbalance:.1f}")

    # LightGBM: MIT license ✓
    lgb_params = {
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": class_imbalance,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }

    print("    Training LightGBM...")
    model = lgb.LGBMClassifier(**lgb_params)

    # Custom eval function: sample-level F0.5 (fast proxy for early stopping)
    def lgb_f05_eval(y_true_cb, y_pred_cb):
        preds_binary = (y_pred_cb > 0.5).astype(int)
        tp = int(((preds_binary == 1) & (y_true_cb == 1)).sum())
        fp = int(((preds_binary == 1) & (y_true_cb == 0)).sum())
        fn = int(((preds_binary == 0) & (y_true_cb == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f05 = compute_f05(p, r)
        return "f05", f05, True

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=lgb_f05_eval,
        callbacks=[
            lgb.early_stopping(50, verbose=True),
            lgb.log_evaluation(100),
        ],
    )

    print(f"    Best iteration: {model.best_iteration_}")

    # Feature importances
    importances = model.feature_importances_
    feat_imp = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
    print("\n    Feature importances (top 10):")
    for name, imp in feat_imp[:10]:
        print(f"      {name}: {imp}")

    # Save model
    model_path = os.path.join(CACHE_DIR, "lgb_model.pkl")
    joblib.dump(model, model_path)
    print(f"    Model saved to {model_path}")

    # ---- Threshold tuning ----
    print("\n    Tuning threshold for macro F0.5 on validation set...")
    val_probs = model.predict_proba(X_val)[:, 1]

    best_t = 0.5
    best_f05 = 0.0
    best_p = 0.0
    best_r = 0.0

    for t in np.arange(0.10, 0.91, 0.01):
        f05 = macro_f05_on_val(val_probs, t, val_pairs, val_gt, val_s1_ids)
        if f05 > best_f05:
            best_f05 = f05
            best_t = float(round(t, 2))
            # Overall P, R at this threshold
            preds = (val_probs >= t).astype(int)
            tp = int(((preds == 1) & (y_val == 1)).sum())
            fp = int(((preds == 1) & (y_val == 0)).sum())
            fn = int(((preds == 0) & (y_val == 1)).sum())
            best_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            best_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print(f"\n    Optimal threshold t*: {best_t:.2f}")
    print(f"    Val Precision:       {best_p:.4f}")
    print(f"    Val Recall:          {best_r:.4f}")
    print(f"    Val macro F0.5:      {best_f05:.4f}")

    config = {
        "threshold": best_t,
        "val_precision": float(best_p),
        "val_recall": float(best_r),
        "val_f05": float(best_f05),
        "best_iteration": int(model.best_iteration_),
        "use_cross_encoder": False,
        "cross_encoder_uncertain_low": 0.3,
        "cross_encoder_uncertain_high": 0.7,
        "cross_encoder_weight_lgb": 0.6,
        "cross_encoder_weight_ce": 0.4,
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"    Config saved to {CONFIG_PATH}")

    STAGE_TIMES["stage_4"] = time.time() - t0
    print(f"    Stage 4 time: {STAGE_TIMES['stage_4']:.1f}s")
    return model, config


# ===========================================================================
# STAGE 4B — CROSS-ENCODER RE-RANKING (OPTIONAL)
# ===========================================================================

def run_cross_encoder_reranking(probs, config, pairs, data_test):
    """Optional cross-encoder re-ranking on uncertain predictions."""
    if not config.get("use_cross_encoder", False):
        print("    Skipping cross-encoder re-ranking (disabled in config)")
        return probs

    t0 = time.time()
    print("=" * 50 + " STAGE 4B: CROSS-ENCODER RE-RANKING " + "=" * 50)

    from sentence_transformers import CrossEncoder
    # cross-encoder/ms-marco-MiniLM-L-6-v2: Apache 2.0, ~22M params ✓
    device = "cuda" if _has_cuda() else "cpu"
    ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

    s1, s2, s3 = data_test["s1"], data_test["s2"], data_test["s3"]
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    name_dict = dict(zip(
        list(s1["entity_id"]) + list(s2s3["entity_id"]),
        list(s1["norm_name"]) + list(s2s3["norm_name"])
    ))
    addr_dict = dict(zip(
        list(s1["entity_id"]) + list(s2s3["entity_id"]),
        list(s1["norm_addr"]) + list(s2s3["norm_addr"])
    ))
    del s2s3
    gc.collect()

    low = config.get("cross_encoder_uncertain_low", 0.3)
    high = config.get("cross_encoder_uncertain_high", 0.7)
    w_lgb = config.get("cross_encoder_weight_lgb", 0.6)
    w_ce = config.get("cross_encoder_weight_ce", 0.4)

    uncertain_mask = (probs >= low) & (probs <= high)
    uncertain_indices = np.where(uncertain_mask)[0]
    print(f"    Uncertain pairs ({low}–{high}): {len(uncertain_indices):,}")

    if len(uncertain_indices) == 0:
        STAGE_TIMES["stage_4b"] = time.time() - t0
        return probs

    ce_inputs = []
    for idx in uncertain_indices:
        s1_id, cand_id = pairs[idx]
        nn_s1 = name_dict.get(s1_id, "")
        na_s1 = addr_dict.get(s1_id, "")
        nn_cand = name_dict.get(cand_id, "")
        na_cand = addr_dict.get(cand_id, "")
        text_a = f"business name: {nn_s1} address: {na_s1}"
        text_b = f"business name: {nn_cand} address: {na_cand}"
        ce_inputs.append((text_a, text_b))

    print(f"    Running cross-encoder on {len(ce_inputs):,} pairs...")
    ce_scores = ce_model.predict(ce_inputs, batch_size=256, show_progress_bar=True)
    from scipy.special import expit
    ce_scores = expit(ce_scores)

    final_probs = probs.copy()
    for i, idx in enumerate(uncertain_indices):
        final_probs[idx] = w_lgb * probs[idx] + w_ce * ce_scores[i]

    STAGE_TIMES["stage_4b"] = time.time() - t0
    print(f"    Stage 4B time: {STAGE_TIMES['stage_4b']:.1f}s")
    return final_probs


# ===========================================================================
# STAGE 5 — INFERENCE ON TEST SET
# ===========================================================================

def run_stage_5(feat_data_test, model, config, data_test):
    """Apply model to test set and generate predictions."""
    t0 = time.time()
    print("=" * 50 + " STAGE 5: INFERENCE ON TEST SET " + "=" * 50)

    X_test = feat_data_test["X"]
    pairs = feat_data_test["pairs"]
    threshold = config["threshold"]

    print(f"    Test pairs: {len(X_test):,}")
    print(f"    Threshold t*: {threshold:.2f}")

    # Predict probabilities
    probs = model.predict_proba(X_test)[:, 1]

    # Optional cross-encoder re-ranking
    probs = run_cross_encoder_reranking(probs, config, pairs, data_test)

    # Apply threshold
    predictions = defaultdict(set)
    for i, (s1_id, cand_id) in enumerate(pairs):
        if probs[i] >= threshold:
            predictions[s1_id].add(cand_id)

    # Ensure ALL S1 entities appear
    s1_ids = data_test["s1"]["entity_id"].tolist()
    singletons = 0
    matched = 0
    for s1_id in s1_ids:
        if s1_id not in predictions or not predictions[s1_id]:
            predictions[s1_id] = set()
            singletons += 1
        else:
            matched += 1

    print(f"    Singletons predicted: {singletons:,}")
    print(f"    Matched entities predicted: {matched:,}")

    STAGE_TIMES["stage_5"] = time.time() - t0
    print(f"    Stage 5 time: {STAGE_TIMES['stage_5']:.1f}s")
    return predictions, s1_ids


# ===========================================================================
# STAGE 6 — POST-PROCESSING AND OUTPUT
# ===========================================================================

def run_stage_6(predictions, s1_ids, data_test):
    """Write output files and validate."""
    t0 = time.time()
    print("=" * 50 + " STAGE 6: POST-PROCESSING AND OUTPUT " + "=" * 50)

    # Valid S2/S3 IDs in test
    s2_ids = set(data_test["s2"]["entity_id"].tolist())
    s3_ids = set(data_test["s3"]["entity_id"].tolist())
    valid_ids = s2_ids | s3_ids

    # Write matching_results.tsv
    match_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    print(f"    Writing {match_path}...")
    with open(match_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_ids:
            match_set = predictions.get(s1_id, set())
            # Guard: remove S1 IDs and invalid IDs
            match_set = {m for m in match_set if not m.startswith("S1-") and m in valid_ids}
            # Deduplicate assertion
            match_list = sorted(match_set)
            assert len(match_list) == len(set(match_list)), f"Duplicate IDs for {s1_id}"
            f.write(f"{s1_id}\t{','.join(match_list)}\n")

    # Validate
    print("\n    Running validation...")
    validator_path = os.path.join(UTILS_DIR, "validate_submission.py")
    test_dir = os.path.join(DATA_DIR, "test")
    cand_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

    try:
        result = subprocess.run(
            [sys.executable, validator_path,
             "--matching", match_path,
             "--candidate", cand_path,
             "--test-dir", test_dir],
            capture_output=True, text=True, timeout=300
        )
        print(result.stdout)
        if result.returncode != 0:
            print("    VALIDATION FAILED:")
            print(result.stderr)
        else:
            print("    ✓ VALIDATION PASSED")
    except FileNotFoundError:
        print(f"    WARNING: Validator not found at {validator_path}")
    except subprocess.TimeoutExpired:
        print("    WARNING: Validation timed out")

    STAGE_TIMES["stage_6"] = time.time() - t0
    print(f"    Stage 6 time: {STAGE_TIMES['stage_6']:.1f}s")


# ===========================================================================
# MAIN ORCHESTRATOR
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Business Entity Resolution Pipeline — Amazon ML Challenge 2026"
    )
    parser.add_argument("--stage", type=int, default=None,
                        help="Run only this stage (0-6). Default: run all.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override dataset directory path")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory path")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Override cache directory path")
    args = parser.parse_args()

    global DATA_DIR, OUTPUT_DIR, CACHE_DIR, CONFIG_PATH, UTILS_DIR
    if args.data_dir:
        DATA_DIR = args.data_dir
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    if args.cache_dir:
        CACHE_DIR = args.cache_dir

    ensure_dirs()

    total_start = time.time()
    run_all = args.stage is None
    target = args.stage

    # ===== STAGE 0 =====
    if run_all or target == 0:
        split_data = run_stage_0()
    else:
        sp = os.path.join(CACHE_DIR, "train_val_split.pkl")
        if os.path.exists(sp):
            split_data = joblib.load(sp)
        else:
            print("ERROR: Stage 0 cache not found. Run --stage 0 first.")
            return

    # ===== STAGE 1 (train) =====
    need_train_data = run_all or target in (1, 2, 3, 4)
    if need_train_data:
        data_train = run_stage_1(split="train")
    else:
        cp = os.path.join(CACHE_DIR, "normalized_train.pkl")
        data_train = joblib.load(cp) if os.path.exists(cp) else None

    # ===== STAGE 2 (train) =====
    if run_all or target == 2:
        blocking_train = run_stage_2(data_train, split_data=split_data, mode="train")
    else:
        cp = os.path.join(CACHE_DIR, "candidates_train.pkl")
        blocking_train = joblib.load(cp) if os.path.exists(cp) else None

    # ===== STAGE 3 (train) =====
    if run_all or target == 3:
        feat_train = run_stage_3(data_train, blocking_train, split_data=split_data, mode="train")
    else:
        cp = os.path.join(CACHE_DIR, "features_train.pkl")
        feat_train = joblib.load(cp) if os.path.exists(cp) else None

    # ===== STAGE 4 =====
    if run_all or target == 4:
        model, config = run_stage_4(feat_train, split_data)
    else:
        mp = os.path.join(CACHE_DIR, "lgb_model.pkl")
        model = joblib.load(mp) if os.path.exists(mp) else None
        config = json.load(open(CONFIG_PATH)) if os.path.exists(CONFIG_PATH) else {"threshold": 0.5}

    # ===== STAGE 1 (test) =====
    need_test_data = run_all or target in (1, 5)
    if need_test_data:
        data_test = run_stage_1(split="test")
    else:
        cp = os.path.join(CACHE_DIR, "normalized_test.pkl")
        data_test = joblib.load(cp) if os.path.exists(cp) else None

    # ===== STAGE 2 (test) =====
    if run_all or target == 5:
        blocking_test = run_stage_2(data_test, mode="test")
    else:
        cp = os.path.join(CACHE_DIR, "candidates_test.pkl")
        blocking_test = joblib.load(cp) if os.path.exists(cp) else None

    # ===== STAGE 3 (test) =====
    if run_all or target == 5:
        feat_test = run_stage_3(data_test, blocking_test, mode="test")
    else:
        cp = os.path.join(CACHE_DIR, "features_test.pkl")
        feat_test = joblib.load(cp) if os.path.exists(cp) else None

    # ===== STAGE 5 =====
    if run_all or target == 5:
        predictions, s1_ids = run_stage_5(feat_test, model, config, data_test)
    else:
        predictions, s1_ids = None, None

    # ===== STAGE 6 =====
    if run_all or target == 6:
        if predictions is not None and data_test is not None:
            run_stage_6(predictions, s1_ids, data_test)
        else:
            print("ERROR: No predictions available. Run --stage 5 first.")

    # ===== TIMING SUMMARY =====
    total_time = time.time() - total_start
    print("\n" + "=" * 50 + " TIMING SUMMARY " + "=" * 50)
    for stage, t in sorted(STAGE_TIMES.items()):
        print(f"    {stage}: {t:.1f}s ({t / 60:.1f}m)")
    print(f"    TOTAL: {total_time:.1f}s ({total_time / 60:.1f}m)")


if __name__ == "__main__":
    main()
