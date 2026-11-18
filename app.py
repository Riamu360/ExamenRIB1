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
from fastapi.responses import HTMLResponse
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


INDEX_HTML = '<!DOCTYPE html>\n<html lang="es">\n<head>\n  <meta charset="UTF-8" />\n  <meta name="viewport" content="width=device-width, initial-scale=1.0" />\n  <title>Buscador NFCorpus — TF-IDF &amp; BM25</title>\n  <style>\n    :root {\n      --bg: #0f172a;\n      --card: #1e293b;\n      --card-hover: #263349;\n      --border: #334155;\n      --text: #f1f5f9;\n      --muted: #94a3b8;\n      --accent: #38bdf8;\n      --accent-2: #fb923c;\n      --ok: #22c55e;\n    }\n    * { box-sizing: border-box; }\n    body {\n      margin: 0;\n      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;\n      background: var(--bg);\n      color: var(--text);\n      line-height: 1.5;\n    }\n    .container { max-width: 960px; margin: 0 auto; padding: 28px 18px 48px; }\n    header {\n      display: flex; flex-wrap: wrap; justify-content: space-between;\n      align-items: center; gap: 12px; margin-bottom: 22px;\n    }\n    h1 { margin: 0; font-size: 1.55rem; font-weight: 700; }\n    .subtitle { color: var(--muted); font-size: 0.92rem; margin-top: 4px; }\n    .badge {\n      display: inline-flex; align-items: center; gap: 8px;\n      padding: 6px 12px; border-radius: 999px;\n      background: rgba(34, 197, 94, 0.12); border: 1px solid rgba(34, 197, 94, 0.35);\n      color: #86efac; font-size: 0.85rem; font-weight: 500;\n    }\n    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--ok); }\n    .search-box {\n      background: var(--card); border: 1px solid var(--border);\n      border-radius: 14px; padding: 18px; margin-bottom: 18px;\n    }\n    .row { display: flex; flex-wrap: wrap; gap: 10px; }\n    input[type="text"] {\n      flex: 1 1 320px; padding: 11px 14px; border-radius: 10px;\n      border: 1px solid var(--border); background: #0b1220; color: var(--text);\n      font-size: 0.98rem; outline: none;\n    }\n    input[type="text"]:focus { border-color: var(--accent); }\n    select, button {\n      padding: 10px 14px; border-radius: 10px; border: 1px solid var(--border);\n      background: #0b1220; color: var(--text); font-size: 0.95rem;\n    }\n    button.primary {\n      background: var(--accent); color: #082f49; border: none;\n      font-weight: 700; cursor: pointer; padding: 10px 20px;\n    }\n    button.primary:hover { filter: brightness(1.08); }\n    .meta-bar {\n      display: flex; justify-content: space-between; align-items: center;\n      color: var(--muted); font-size: 0.88rem; margin: 10px 2px 14px;\n    }\n    .grid-compare { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }\n    @media (max-width: 780px) { .grid-compare { grid-template-columns: 1fr; } }\n    .col-title {\n      font-size: 1rem; font-weight: 700; margin: 0 0 10px;\n      padding-bottom: 6px; border-bottom: 1px solid var(--border);\n    }\n    .result-card {\n      background: var(--card); border: 1px solid var(--border);\n      border-radius: 12px; padding: 14px 16px; margin-bottom: 12px;\n      transition: background 0.15s ease;\n    }\n    .result-card:hover { background: var(--card-hover); }\n    .result-top {\n      display: flex; flex-wrap: wrap; justify-content: space-between;\n      align-items: center; gap: 8px; margin-bottom: 6px;\n    }\n    .rank-id { display: flex; align-items: center; gap: 8px; font-size: 0.84rem; }\n    .rank-pill {\n      background: rgba(56, 189, 248, 0.16); color: var(--accent);\n      padding: 2px 8px; border-radius: 6px; font-weight: 700;\n    }\n    .doc-id {\n      font-family: ui-monospace, monospace; background: #0b1220;\n      padding: 2px 8px; border-radius: 6px; color: #cbd5e1; border: 1px solid var(--border);\n    }\n    .score { font-family: ui-monospace, monospace; font-size: 0.85rem; color: var(--accent-2); font-weight: 600; }\n    .doc-title { font-weight: 600; font-size: 1.02rem; margin: 4px 0 6px; color: #f8fafc; }\n    .doc-snippet { margin: 0; color: #cbd5e1; font-size: 0.9rem; }\n    footer {\n      margin-top: 28px; padding-top: 14px; border-top: 1px solid var(--border);\n      display: flex; flex-wrap: wrap; gap: 16px; font-size: 0.86rem; color: var(--muted);\n    }\n    a { color: var(--accent); text-decoration: none; }\n    a:hover { text-decoration: underline; }\n  </style>\n</head>\n<body>\n  <div class="container">\n    <header>\n      <div>\n        <h1>Buscador NFCorpus (TF-IDF &amp; BM25)</h1>\n        <div class="subtitle">Servicio de Recuperación de Información sobre BeIR/nfcorpus</div>\n      </div>\n      <div class="badge" id="healthBadge"><span class="dot"></span> Conectando...</div>\n    </header>\n\n    <div class="search-box">\n      <form id="searchForm" class="row">\n        <input id="queryInput" type="text" placeholder="Escribe tu consulta de búsqueda..." required />\n        <select id="modelSelect" title="Modelo">\n          <option value="bm25">BM25</option>\n          <option value="tfidf">TF-IDF (Coseno)</option>\n          <option value="compare">Comparar ambos</option>\n        </select>\n        <select id="topKSelect" title="Top-K">\n          <option value="5">Top 5</option>\n          <option value="10" selected>Top 10</option>\n          <option value="20">Top 20</option>\n        </select>\n        <button type="submit" class="primary">Buscar</button>\n      </form>\n    </div>\n\n    <div class="meta-bar" id="metaBar">Ingresa una consulta para recuperar documentos de NFCorpus.</div>\n    <div id="resultsContainer"></div>\n\n    <footer>\n      <span>Endpoints REST disponibles:</span>\n      <a href="/health" target="_blank">GET /health</a>\n      <a href="/docs" target="_blank">Swagger UI (/docs)</a>\n      <span>POST /search (JSON)</span>\n    </footer>\n  </div>\n\n  <script>\n    function esc(str) {\n      return String(str ?? "")\n        .replace(/&/g, "&amp;")\n        .replace(/</g, "&lt;")\n        .replace(/>/g, "&gt;")\n        .replace(/"/g, "&quot;");\n    }\n\n    async function loadHealth() {\n      try {\n        const res = await fetch("/health");\n        const data = await res.json();\n        document.getElementById("healthBadge").innerHTML =\n          `<span class="dot"></span> ${esc(data.corpus)} · ${data.num_documents} docs · ${data.vocabulary_size} términos`;\n      } catch (e) {\n        document.getElementById("healthBadge").textContent = "Sin conexión a /health";\n      }\n    }\n\n    function renderList(results) {\n      if (!results || results.length === 0) {\n        return `<div class="result-card">No se encontraron documentos coincidentes para esta consulta.</div>`;\n      }\n      return results.map(item => `\n        <div class="result-card">\n          <div class="result-top">\n            <div class="rank-id">\n              <span class="rank-pill">#${item.rank}</span>\n              <span class="doc-id">${esc(item.document_id)}</span>\n            </div>\n            <span class="score">score: ${Number(item.score).toFixed(4)}</span>\n          </div>\n          <div class="doc-title">${esc(item.title || "(Sin título)")}</div>\n          <p class="doc-snippet">${esc(item.snippet)}</p>\n        </div>\n      `).join("");\n    }\n\n    async function callSearch(query, model, topK) {\n      const res = await fetch("/search", {\n        method: "POST",\n        headers: { "Content-Type": "application/json" },\n        body: JSON.stringify({ query, model, top_k: topK })\n      });\n      if (!res.ok) throw new Error("HTTP " + res.status);\n      return await res.json();\n    }\n\n    async function performSearch() {\n      const query = document.getElementById("queryInput").value.trim();\n      if (!query) return;\n      const mode = document.getElementById("modelSelect").value;\n      const topK = Number(document.getElementById("topKSelect").value);\n      const container = document.getElementById("resultsContainer");\n      const meta = document.getElementById("metaBar");\n      meta.textContent = "Buscando...";\n      const t0 = performance.now();\n\n      try {\n        if (mode === "compare") {\n          const [bm, tf] = await Promise.all([\n            callSearch(query, "bm25", topK),\n            callSearch(query, "tfidf", topK)\n          ]);\n          const ms = Math.round(performance.now() - t0);\n          meta.textContent = `Comparación BM25 vs TF-IDF · Top ${topK} · ${ms} ms`;\n          container.innerHTML = `\n            <div class="grid-compare">\n              <div>\n                <h2 class="col-title">BM25 (${bm.results.length} resultados)</h2>\n                ${renderList(bm.results)}\n              </div>\n              <div>\n                <h2 class="col-title">TF-IDF Coseno (${tf.results.length} resultados)</h2>\n                ${renderList(tf.results)}\n              </div>\n            </div>\n          `;\n        } else {\n          const data = await callSearch(query, mode, topK);\n          const ms = Math.round(performance.now() - t0);\n          meta.textContent = `Modelo: ${data.model.toUpperCase()} · ${data.results.length} resultados · ${ms} ms`;\n          container.innerHTML = renderList(data.results);\n        }\n      } catch (err) {\n        meta.textContent = "Error al consultar POST /search: " + err.message;\n      }\n    }\n\n    document.getElementById("searchForm").addEventListener("submit", (e) => {\n      e.preventDefault();\n      performSearch();\n    });\n\n    loadHealth();\n  </script>\n</body>\n</html>'

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


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return HTMLResponse(content=INDEX_HTML)


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
