import gzip
import json
import math
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from nltk.stem import SnowballStemmer
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_FILE = os.path.join(BASE_DIR, "nfcorpus_corpus.json.gz")

stemmer = SnowballStemmer("english")
STOPWORDS = frozenset(['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'around', 'as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both', 'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 'cry', 'de', 'describe', 'detail', 'do', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen', 'fifty', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into', 'is', 'it', 'its', 'itself', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 'same', 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system', 'take', 'ten', 'than', 'that', 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thick', 'thin', 'third', 'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 'we', 'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves'])
CLINICAL_SINGLE_LETTERS = frozenset({"b", "c", "d", "e", "k"})


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"'(?:s|re|ve|ll|d|m)\b", "", text)
    text = re.sub(r"n't\b", " not", text)
    return text


def tokenize_and_filter(text: str) -> list[str]:
    norm = normalize_text(text)
    raw_tokens = re.findall(r"[a-z0-9]+", norm)
    filtered = []
    for tok in raw_tokens:
        if tok in STOPWORDS:
            continue
        if len(tok) < 2 and not tok.isdigit() and tok not in CLINICAL_SINGLE_LETTERS:
            continue
        filtered.append(tok)
    return filtered


def preprocess(text: str) -> list[str]:
    return [stemmer.stem(tok) for tok in tokenize_and_filter(text)]


t0 = time.perf_counter()
with gzip.open(CORPUS_FILE, "rt", encoding="utf-8") as f:
    _records = json.load(f)

doc_ids = [r["_id"] for r in _records]
doc_titles = [r["title"] for r in _records]
doc_raw_texts = [r["text"] for r in _records]
del _records

NUM_DOCS = len(doc_ids)
inverted_index = defaultdict(list)
doc_term_freqs = []
doc_lengths = [0.0] * NUM_DOCS

for idx in range(NUM_DOCS):
    full_text = f"{doc_titles[idx]} {doc_raw_texts[idx]}".strip()
    tokens = preprocess(full_text)
    doc_lengths[idx] = float(len(tokens))
    tf_counts = Counter(tokens)
    doc_term_freqs.append(tf_counts)
    for term, tf in tf_counts.items():
        inverted_index[term].append((idx, tf))

inverted_index = dict(inverted_index)
avgdl = sum(doc_lengths) / NUM_DOCS
vocabulary = {
    term: {"term_id": tid, "df": len(inverted_index[term])}
    for tid, term in enumerate(sorted(inverted_index.keys()))
}

idf_tfidf = {
    term: math.log((NUM_DOCS + 1.0) / (info["df"] + 1.0)) + 1.0
    for term, info in vocabulary.items()
}
doc_tfidf_norms = [1.0] * NUM_DOCS
for idx, tf_counts in enumerate(doc_term_freqs):
    sq_sum = sum(((1.0 + math.log(cnt)) * idf_tfidf[term]) ** 2 for term, cnt in tf_counts.items())
    if sq_sum > 0.0:
        doc_tfidf_norms[idx] = math.sqrt(sq_sum)
del doc_term_freqs

BM25_K1 = 1.5
BM25_B = 0.75
idf_bm25 = {
    term: math.log(1.0 + (NUM_DOCS - info["df"] + 0.5) / (info["df"] + 0.5))
    for term, info in vocabulary.items()
}
BUILD_SECONDS = round(time.perf_counter() - t0, 3)


def build_snippet(raw_text: str, query: str, max_chars: int = 240) -> str:
    clean_text = re.sub(r"\s+", " ", raw_text).strip()
    if not clean_text or len(clean_text) <= max_chars:
        return clean_text
    query_terms = [t for t in re.findall(r"[a-z0-9]+", normalize_text(query)) if t not in STOPWORDS]
    lower_text = clean_text.lower()
    match_pos = -1
    for term in query_terms:
        pos = lower_text.find(term)
        if pos != -1:
            match_pos = pos
            break
    if match_pos == -1 or match_pos < max_chars // 2:
        return clean_text[:max_chars].rsplit(" ", 1)[0] + "..."
    start = max(0, match_pos - max_chars // 3)
    end = min(len(clean_text), start + max_chars)
    snippet = clean_text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet.split(" ", 1)[-1]
    if end < len(clean_text):
        snippet = snippet.rsplit(" ", 1)[0] + "..."
    return snippet


def search_tfidf(query: str, top_k: int = 10) -> list[dict]:
    if top_k <= 0:
        return []
    q_tokens = preprocess(query)
    if not q_tokens:
        return []
    q_tf = Counter(q_tokens)
    q_weights = {t: (1.0 + math.log(c)) * idf_tfidf[t] for t, c in q_tf.items() if t in inverted_index}
    if not q_weights:
        return []
    q_norm = math.sqrt(sum(w * w for w in q_weights.values()))
    dot_products = defaultdict(float)
    for term, qw in q_weights.items():
        idf_t = idf_tfidf[term]
        for doc_idx, d_cnt in inverted_index[term]:
            dot_products[doc_idx] += ((1.0 + math.log(d_cnt)) * idf_t) * qw
    scored = sorted(
        ((dot / (doc_tfidf_norms[doc_idx] * q_norm), doc_idx) for doc_idx, dot in dot_products.items()),
        key=lambda item: (-item[0], doc_ids[item[1]]),
    )
    return [
        {
            "rank": rank,
            "document_id": doc_ids[doc_idx],
            "score": round(float(score), 6),
            "title": doc_titles[doc_idx],
            "snippet": build_snippet(doc_raw_texts[doc_idx], query),
        }
        for rank, (score, doc_idx) in enumerate(scored[:top_k], start=1)
    ]


def search_bm25(query: str, top_k: int = 10, k1: float = BM25_K1, b: float = BM25_B) -> list[dict]:
    if top_k <= 0:
        return []
    q_tokens = preprocess(query)
    if not q_tokens:
        return []
    q_tf = Counter(q_tokens)
    scores = defaultdict(float)
    for term, q_cnt in q_tf.items():
        if term not in inverted_index:
            continue
        idf_t = idf_bm25[term]
        for doc_idx, d_cnt in inverted_index[term]:
            dl = doc_lengths[doc_idx]
            tf_norm = (d_cnt * (k1 + 1.0)) / (d_cnt + k1 * (1.0 - b + b * (dl / avgdl)))
            scores[doc_idx] += q_cnt * idf_t * tf_norm
    scored = sorted(
        ((score, doc_idx) for doc_idx, score in scores.items()),
        key=lambda item: (-item[0], doc_ids[item[1]]),
    )
    return [
        {
            "rank": rank,
            "document_id": doc_ids[doc_idx],
            "score": round(float(score), 6),
            "title": doc_titles[doc_idx],
            "snippet": build_snippet(doc_raw_texts[doc_idx], query),
        }
        for rank, (score, doc_idx) in enumerate(scored[:top_k], start=1)
    ]


class SearchRequest(BaseModel):
    query: str = Field(..., description="Texto de la consulta")
    model: str = Field(default="bm25", description="Modelo: 'bm25' o 'tfidf'")
    top_k: int = Field(default=10, ge=1, le=500, description="Número máximo de resultados")


class SearchResultItem(BaseModel):
    rank: int
    document_id: str
    score: float
    title: str
    snippet: str


class SearchResponse(BaseModel):
    model: str
    results: list[SearchResultItem]


app = FastAPI(
    title="NFCorpus Information Retrieval API",
    description="Servicio web de recuperación de información (TF-IDF y BM25) sobre BeIR/nfcorpus.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "NFCorpus Information Retrieval API",
        "status": "ok",
        "endpoints": {
            "health": "GET /health",
            "search": "POST /search",
            "docs": "GET /docs",
        },
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "corpus": "BeIR/nfcorpus",
        "num_documents": NUM_DOCS,
        "vocabulary_size": len(vocabulary),
        "models": ["tfidf", "bm25"],
    }


@app.post("/search", response_model=SearchResponse)
def search_endpoint(payload: SearchRequest):
    normalized_model = payload.model.strip().lower().replace("-", "")
    if normalized_model not in {"bm25", "tfidf"}:
        raise HTTPException(status_code=400, detail="El campo 'model' debe ser 'bm25' o 'tfidf'.")
    results = (
        search_bm25(payload.query, top_k=payload.top_k)
        if normalized_model == "bm25"
        else search_tfidf(payload.query, top_k=payload.top_k)
    )
    return SearchResponse(model=normalized_model, results=results)
