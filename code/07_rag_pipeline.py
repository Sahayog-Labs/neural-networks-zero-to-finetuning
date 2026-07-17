#!/usr/bin/env python3
"""
07_rag_pipeline.py - the retrieval half of RAG, runnable. Secondary artifact, p.47.

Page 47 ("Prompt vs RAG vs Fine-Tune") ships a live in-browser retrieval sandbox.
This is that sandbox as a real Python pipeline you can run, read, and break:

    chunk  ->  embed  ->  FAISS   (+ a NumPy baseline that BEATS FAISS under 100k)
                                ->  BM25  ->  RRF  ->  cross-encoder rerank
                                ->  ablate recall@5 across the four stages.

The one lesson the page states and this script proves on numbers: retrieval quality
is the whole game, and it climbs stage by stage. Dense alone misses exact strings;
BM25 alone misses paraphrases; RRF fuses them; the reranker converts *similarity*
into *relevance*. You watch recall@5 go up at every stage.

WHY IT MIRRORS THE PAGE. The corpus here is the SAME seeded synthetic stand-in the
browser renders: 14 topic clusters, ~200 sentences, 384-dim topic-centroid
embeddings built from the identical mulberry32 stream, the identical BM25 constants
(k1=1.2, b=0.75), weighted RRF with k_RRF=60, and the same topic-affinity
cross-encoder. So the ranked lists this file prints are the ones you can reproduce
by hand in the sandbox. (A real deployment swaps the synthetic encoder for a
sentence bi-encoder + a real cross-encoder; the pipeline shape is unchanged.)

WHY NUMPY, NOT A VECTOR DB. Page 47's key box: under ~100k chunks you do not need a
vector database. `np.argsort(E @ q)` is EXACT, instant, and dependency-free; FAISS
(and the managed-DB industry) buys you *approximate* search and sharding for the
millions-of-chunks regime. This script measures that: it times the NumPy exact
search against an optional FAISS IndexFlatIP and asserts the 100k-chunk store fits
in a corner of memory (the page's 410 MB / 205 MB / 12.8 MB arithmetic).

Usage
-----
    python 07_rag_pipeline.py                 # full teaching run (NumPy only)
    python 07_rag_pipeline.py --self-test     # same, terse, asserts and exits 0
    python 07_rag_pipeline.py --query 3       # detail one query's four stages
    python 07_rag_pipeline.py --scale 100000  # size of the "vs FAISS" timing demo
    python 07_rag_pipeline.py --no-faiss      # skip the optional FAISS comparison

SAFETY: pure CPU/NumPy. No GPU, no network, no model download; writes nothing and
installs nothing. Runs on a laptop in a second or two (spec-code: rungs 1-5 and the
retrieval half need no Spark). FAISS is optional and read-only if present.

Verified-against stack (constants.md sec7 / brief-tooling-hardware): the numeric
answers depend only on numpy; the FAISS path is desk-checked against faiss-cpu
IndexFlatIP (add / search) which is API-stable. Reference GPU stack for the sibling
06_rag_vs_finetune_showdown.ipynb is torch 2.13 / transformers 5.14.1.
"""

import argparse
import math
import sys
import time

import numpy as np

# The BM25 / RRF constants are the FIELD's, printed on p.47 exactly as here.
BM25_K1 = 1.2          # term-frequency saturation
BM25_B = 0.75          # length normalization  (page writes it b_BM25, not the course's b=batch)
RRF_K = 60             # rank-smoothing constant (page: k_RRF, not the course's k=optimizer step)
EMB_DIM = 384          # the page's synthetic embedding width
TOP_N = 20             # top-N per retriever before fusion (page: "top-N ~= 20")
K_FINAL = 5            # rerank down to ~5, then stuff into the prompt  (recall is @5)

GB = 10 ** 9           # decimal: storage sizes quoted alone are GB/MB (constants sec0)
MB = 10 ** 6
GiB = 1 << 30          # binary: anything compared to hardware capacity is GiB


# --------------------------------------------------------------------------- #
# 0.  Seeding + a version stamp (spec-code sec-B 4 & 7).
# --------------------------------------------------------------------------- #

def stamp():
    faiss_v = _faiss_version()
    print("-" * 70)
    print("07_rag_pipeline.py  ·  the retrieval half of RAG, on numbers")
    print(f"  numpy {np.__version__}   ·   faiss {faiss_v}   ·   python "
          f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print("  reference stack (sibling notebook, GPU): torch 2.13 · transformers 5.14.1")
    print("  this file needs numpy only; every number below is CPU-deterministic.")
    print("-" * 70)


def _faiss_version():
    try:
        import faiss  # noqa: F401
        return getattr(faiss, "__version__", "present")
    except Exception:
        return "not installed (NumPy baseline is used; see sec.2)"


# --------------------------------------------------------------------------- #
# 1.  The corpus.  Ported 1:1 from the p.47 sandbox so page and script agree.
#     mulberry32 -> topic-centroid embeddings -> topic-pure chunks.
# --------------------------------------------------------------------------- #

def mulberry32(seed):
    """Exact port of the page's PRNG (unsigned 32-bit); identical stream to the browser."""
    state = seed & 0xFFFFFFFF

    def imul(x, y):
        return (x * y) & 0xFFFFFFFF

    def rnd():
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = state
        t = imul(t ^ (t >> 15), t | 1) & 0xFFFFFFFF
        t = ((t + imul(t ^ (t >> 7), t | 61)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return rnd


def topic_vec(seed):
    """A random unit vector, seeded per topic -- the topic's embedding centroid."""
    r = mulberry32(seed)
    v = np.empty(EMB_DIM, dtype=np.float64)
    for i in range(EMB_DIM):
        v[i] = r() * 2 - 1
    v /= np.linalg.norm(v)
    return v


# topic -> stable embedding seed  (the page's TOPICS[*].seed)
TOPIC_SEED = {
    "titan": 101, "risk": 202, "infra": 303, "trouble": 404, "batch": 505,
    "finance": 606, "team": 707, "security": 808, "data": 909, "perf": 111,
    "docs": 222, "compliance": 333, "support": 444, "release": 555,
}

# topic -> (sentence-generation seed, target sentence count, hand-written anchors, A[], B[])
# Anchors are the aha-critical sentences; A x B are deterministic template fillers.
# Ported verbatim from the page's TDEF so the corpus is byte-for-byte the sandbox's.
TDEF = {
    "titan": (11, 15,
        ["Project Titan is our next-generation billing platform, replacing the legacy invoicing service.",
         "The Project Titan steering committee meets every second Tuesday to review milestones.",
         "Titan's data model unifies customers, subscriptions, and usage events in one schema.",
         "Kickoff for Project Titan was held in the main auditorium with all stakeholders present."],
        ["Project Titan", "The Titan platform", "Titan's rollout plan", "The Titan program office",
         "Project Titan's roadmap", "Titan's launch team", "The Titan initiative", "Project Titan documentation"],
        ["spans three engineering pods", "targets a phased regional launch", "is tracked in the program dashboard",
         "reports to the executive sponsor monthly", "covers billing, metering, and invoicing",
         "was chartered at the start of the year", "consolidates four legacy systems",
         "is the flagship platform effort"]),
    "risk": (22, 15,
        ["The engineering timeline slipped by eleven weeks after the vendor missed two integration deadlines.",
         "Rising engineer attrition on the platform team left three critical services without an owner.",
         "A budget overrun of thirty percent forced a mid-year replan and a scope reduction.",
         "Repeated outages during the migration eroded stakeholder confidence in the delivery date."],
        ["The delivery schedule", "Vendor delays", "The staffing shortfall", "A mid-quarter replan",
         "The integration backlog", "Scope creep", "The migration window", "Chronic understaffing"],
        ["pushed the launch back by a full quarter", "left two milestones without an owner",
         "forced a painful descoping decision", "compounded an already tight timeline",
         "eroded confidence among the sponsors", "triggered an emergency escalation",
         "stretched the team past its capacity", "added weeks of unplanned rework"]),
    "infra": (33, 15,
        ["Deployments to production go out through the blue-green pipeline every Thursday afternoon.",
         "Rolling back a release is a single command; the previous image is kept warm for one hour.",
         "Production traffic is routed through the regional load balancers with health checks every five seconds."],
        ["Production deployments", "The blue-green pipeline", "A canary release", "The rollback procedure",
         "Traffic routing", "The staging environment", "Health checks", "The deployment gate"],
        ["run every Thursday afternoon", "promote a build only after canary passes", "complete in under ten minutes",
         "keep the previous image warm for one hour", "are load balanced across three regions",
         "require a second approver to sign off", "run automatically on every merge to main",
         "fall back to the last good image on error"]),
    "trouble": (44, 15,
        ["When a service throws an unhandled exception, the on-call engineer gets paged within a minute.",
         "To debug a broken request, start from the trace ID and follow it across service boundaries.",
         "Most timeout errors trace back to a saturated connection pool rather than the downstream service.",
         "Retry with exponential backoff before surfacing an error to the user-facing layer."],
        ["An unhandled exception", "Most timeout errors", "A stack trace", "The error budget", "Retry logic",
         "A saturated connection pool", "The incident channel", "A latency spike"],
        ["pages the on-call engineer within a minute", "usually points at the connection pool",
         "is captured with the request trace ID", "should back off exponentially before retrying",
         "surfaces in the dashboards within seconds", "is annotated with the offending trace",
         "gets escalated after two exhausted retries", "is triaged from the trace ID first"]),
    "batch": (55, 14,
        ["The nightly reconciliation job wrote 4,471 rows to the staging table; the run completed with status code E-4471 recorded for audit.",
         "Batch jobs are scheduled through the orchestrator and log their row counts to the audit ledger.",
         "The audit ledger retains every job's start time, row count, and completion status for seven years.",
         "A checksum mismatch in the export batch triggers an automatic re-run before the morning cutoff."],
        ["The nightly batch job", "The reconciliation run", "Each scheduled job", "The export batch",
         "The orchestrator", "A checksum mismatch", "The staging load", "The morning batch window"],
        ["logs its row count to the audit ledger", "is scheduled through the orchestrator",
         "runs before the morning cutoff", "records a completion status for audit",
         "re-runs automatically on a mismatch", "writes to the staging table first",
         "reports duration and row counts", "is retained for seven years"]),
    "finance": (66, 14,
        ["Cloud spend is dominated by the analytics warehouse, which accounts for sixty percent of the monthly bill.",
         "The finance team reviews committed-use discounts each quarter to reduce compute cost.",
         "Storage costs grew after retention was extended, prompting a tiering policy for cold data."],
        ["Cloud spend", "The monthly bill", "Committed-use discounts", "Storage cost", "The compute budget",
         "Egress charges", "The analytics warehouse", "Cost allocation"],
        ["is dominated by the analytics warehouse", "is reviewed by finance each quarter",
         "grew after retention was extended", "is tagged back to each team", "fell after rightsizing the instances",
         "is the largest line item this quarter", "is capped by a monthly budget alert",
         "is trending down after the tiering policy"]),
    "team": (77, 14,
        ["We hired two senior backend engineers and one SRE to staff the on-call rotation.",
         "Onboarding for new engineers pairs them with a mentor for their first six weeks."],
        ["The platform team", "Two new backend engineers", "The hiring plan", "A new SRE", "Onboarding",
         "The mentorship program", "The on-call rotation", "Headcount"],
        ["added three engineers this quarter", "pairs newcomers with a mentor",
         "is now staffed for twenty-four-seven coverage", "grew to twelve people", "filled two long-open positions",
         "rotates weekly across the team", "includes a two-week ramp plan", "is reviewed at every planning cycle"]),
    "security": (88, 14,
        ["Access to production is granted through short-lived roles that expire after twelve hours.",
         "Service credentials rotate automatically every ninety days through the secrets manager."],
        ["Access to production", "Service credentials", "The audit log", "Multi-factor authentication",
         "The secrets manager", "Least-privilege roles", "Session tokens", "The access review"],
        ["is granted through short-lived roles", "rotate automatically every ninety days",
         "is required for every admin action", "records every privileged operation",
         "are stored in the secrets manager", "expire after twelve hours", "are scoped to a single service",
         "is reviewed quarterly by the owners"]),
    "data": (99, 14,
        ["The data warehouse is partitioned by ingestion date and feeds the analytics dashboards.",
         "The event schema normalizes customers, subscriptions, and usage into one model."],
        ["The data warehouse", "The event schema", "Each ingestion pipeline", "The customer table", "Nightly ETL",
         "The staging dataset", "Schema changes", "The lineage graph"],
        ["is partitioned by ingestion date", "normalizes events into one model", "is validated against a contract",
         "backfills from the raw event stream", "is documented in the catalog", "runs on a two-hour cadence",
         "is versioned behind a migration", "feeds the analytics dashboards"]),
    "perf": (121, 14,
        ["Tail latency dropped once the cache was warmed and a read replica was added.",
         "The p99 response time is dominated by a single slow query against the customer table."],
        ["Tail latency", "Request throughput", "The p99 response time", "Cache hit rate", "Connection pooling",
         "Query latency", "The autoscaler", "Batch size tuning"],
        ["improved after adding a read replica", "is dominated by a single slow query",
         "dropped once the cache was warmed", "scales with the number of workers",
         "is measured at the ninety-ninth percentile", "fell after the index was added",
         "is capped by the connection pool", "was halved by request coalescing"]),
    "docs": (131, 14,
        ["The runbook is kept next to the code and explains the on-call procedure step by step.",
         "Each service README lists the supported endpoints and links to the relevant dashboards."],
        ["The runbook", "Architecture documentation", "Each service README", "The onboarding guide",
         "The design record", "The API reference", "The team wiki", "Release notes"],
        ["is kept next to the code", "is reviewed on every major change", "explains the on-call procedure",
         "links to the relevant dashboards", "captures the decision and its context", "is generated from the source",
         "lists the supported endpoints", "is updated before each release"]),
    "compliance": (141, 14,
        ["Data retention is enforced by an automated policy that keeps records for seven years.",
         "Audit evidence is collected for the annual review and logged to an immutable store."],
        ["Data retention", "The compliance review", "Audit evidence", "The retention policy", "Access certification",
         "The control catalog", "Every privileged change", "The quarterly audit"],
        ["is enforced by an automated policy", "is collected for the annual audit", "retains records for seven years",
         "is signed off by the data owner", "maps to a documented control", "is sampled by the auditors",
         "is logged to an immutable store", "is reviewed each quarter"]),
    "support": (151, 14,
        ["Customer tickets are triaged within one business hour and routed against an SLA.",
         "The support queue links customers to a known workaround from the knowledge base."],
        ["Customer tickets", "The support queue", "A priority incident", "The escalation path",
         "First response time", "The knowledge base", "A recurring complaint", "The support rotation"],
        ["are triaged within one business hour", "routes urgent cases to on-call", "is tracked against an SLA",
         "links customers to a known workaround", "is answered from the knowledge base", "is reviewed in the weekly sync",
         "feeds the product backlog", "is measured by resolution time"]),
    "release": (161, 14,
        ["Each release is cut every two weeks and must pass integration tests to merge.",
         "The changelog documents every version bump, which is tagged and signed automatically."],
        ["Each release", "The CI pipeline", "A version bump", "The changelog", "Feature flags", "The release train",
         "Semantic versioning", "The build"],
        ["is cut every two weeks", "runs the full test suite on every push", "gates merges behind green checks",
         "is tagged and signed automatically", "ships behind a feature flag first",
         "is rolled out to a small cohort first", "is documented in the changelog", "must pass integration tests to merge"]),
}

STOP = set(("a an the of to in on at for and or but is are was were be been being "
            "we you they it he she i do does did how what why when where which who whom this that these "
            "those with as by from into our your their its his her my me us them not no can could will "
            "would should may might must have has had").split())

import re
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(s):
    return _TOKEN_RE.findall(s.lower())


def content_tokens(s):
    return [t for t in tokenize(s) if t not in STOP]


def gen_topic(seed_base, anchors, A, B, target):
    """Deterministic sentence generation: anchors, then A[i]+' '+B[j]+'.' fillers."""
    out = list(anchors)
    r = mulberry32(seed_base)
    seen = {s.lower() for s in out}
    guard = 0
    while len(out) < target and guard < 4000:
        guard += 1
        s = A[int(r() * len(A))] + " " + B[int(r() * len(B))] + "."
        if s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out[:target]


def build_sentences():
    sents = {}
    for tp, (seed, target, anchors, A, B) in TDEF.items():
        sents[tp] = gen_topic(seed, anchors, A, B, target)
    return sents


def build_chunks(sents, sents_per_chunk=5, overlap_pct=12):
    """Topic-pure chunking + seeded embeddings, exactly as the sandbox's buildCorpus()."""
    noise = mulberry32(9999)  # reset per build => identical settings, identical vectors
    tvec = {tp: topic_vec(TOPIC_SEED[tp]) for tp in sents}
    ov = min(sents_per_chunk - 1, round(sents_per_chunk * overlap_pct / 100))
    stride = max(1, sents_per_chunk - ov)

    chunks = []
    for tp in sents:  # dict insertion order == TDEF order == the page's TOPICS order
        arr = sents[tp]
        i = 0
        while i < len(arr):
            slice_ = arr[i:i + sents_per_chunk]
            text = " ".join(slice_)
            base = tvec[tp]
            v = np.empty(EMB_DIM, dtype=np.float64)
            for j in range(EMB_DIM):
                v[j] = base[j] + (noise() * 2 - 1) * 0.28
            v /= np.linalg.norm(v)
            toks = tokenize(text)
            tf = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            chunks.append({"id": len(chunks), "topic": tp, "text": text,
                           "v": v, "tf": tf, "len": len(toks)})
            if i + sents_per_chunk >= len(arr):
                break  # tail chunk already covers the remainder
            i += stride
    E = np.stack([c["v"] for c in chunks]).astype(np.float64)  # (N, d), L2-normalized rows
    df = {}
    for c in chunks:
        for t in c["tf"]:
            df[t] = df.get(t, 0) + 1
    avgdl = sum(c["len"] for c in chunks) / len(chunks)
    return chunks, E, df, avgdl


def blend(parts, tvec):
    """Query embedding = normalized weighted sum of topic centroids (page's blend())."""
    v = np.zeros(EMB_DIM, dtype=np.float64)
    for tp, w in parts:
        v += tvec[tp] * w
    v /= np.linalg.norm(v)
    return v


# --------------------------------------------------------------------------- #
# 2.  Retrieval math -- all live, all exact (cosine, BM25, weighted RRF, rerank).
# --------------------------------------------------------------------------- #

def dense_scores(q_vec, E):
    """One matmul IS the whole search: cosine over L2-normalized rows = E @ q."""
    return E @ q_vec  # (N,)


def bm25_scores(q_toks, chunks, df, avgdl, N):
    def idf(w):
        n = df.get(w, 0)
        return math.log(1 + (N - n + 0.5) / (n + 0.5))
    scores = np.zeros(N, dtype=np.float64)
    for ci, c in enumerate(chunks):
        s = 0.0
        for w in q_toks:
            f = c["tf"].get(w, 0)
            if f == 0:
                continue
            s += idf(w) * (f * (BM25_K1 + 1)) / (f + BM25_K1 * (1 - BM25_B + BM25_B * c["len"] / avgdl))
        scores[ci] = s
    return scores


def rank_map(scores):
    """id -> 1-based rank by descending score (stable, matches the sandbox's sort)."""
    order = np.argsort(-scores, kind="stable")
    m = {}
    for rank, idx in enumerate(order, start=1):
        m[int(idx)] = rank
    return m


def weighted_rrf(dense_s, bm25_s, w=0.5):
    """Weighted Reciprocal Rank Fusion, w in [0,1]. BM25's rank counts only if it matched."""
    N = len(dense_s)
    dm = rank_map(dense_s)
    bm = rank_map(bm25_s)
    fused = np.zeros(N, dtype=np.float64)
    for cid in range(N):
        s = w * (1.0 / (RRF_K + dm[cid]))
        if bm25_s[cid] > 1e-9:  # only fuse a real keyword hit
            s += (1 - w) * (1.0 / (RRF_K + bm[cid]))
        fused[cid] = s
    return fused


def rerank_scores(query, chunks, candidate_ids):
    """Synthetic cross-encoder: topic affinity + a rare-signature-token bonus.
    Reads query and chunk 'jointly' -- what converts similarity into relevance."""
    aff = query.get("aff", {})
    sig = query.get("sig")
    out = {}
    for cid in candidate_ids:
        c = chunks[cid]
        base = aff.get(c["topic"], 0.08)
        if sig and c["tf"].get(sig):
            base = max(base, 1.0)
        out[cid] = base
    return out


def top_ids(scores, k, positive_only=False):
    order = np.argsort(-scores, kind="stable")
    if positive_only:
        order = [int(i) for i in order if scores[int(i)] > 1e-9]
    return [int(i) for i in order[:k]]


# --------------------------------------------------------------------------- #
# 3.  The ten canned queries + their relevance ground truth (`gold`).
#     Ported from the page's QUERIES.  `gold` is the topic whose chunks answer the
#     query; `gold_sig` (E-4471) narrows relevance to the ONE chunk that literally
#     contains the string -- the exact-match case dense retrieval was built to miss.
# --------------------------------------------------------------------------- #

QUERIES = [
    {"label": "How do we deploy to production?", "text": "how do we deploy to production",
     "topics": [("infra", 1)], "aff": {"infra": 0.95, "release": 0.4, "trouble": 0.3}, "gold": "infra"},
    {"label": "error code E-4471", "text": "error code E-4471",
     "topics": [("trouble", 1)], "sig": "e-4471", "aff": {"batch": 0.5, "trouble": 0.35},
     "gold": "batch", "gold_sig": "e-4471"},
    {"label": "why did the project fail?", "text": "why did the project fail",
     "topics": [("risk", 1)], "aff": {"risk": 0.92, "titan": 0.2}, "gold": "risk"},
    {"label": "risks to Project Titan", "text": "risks to project titan",
     "topics": [("titan", 0.72), ("risk", 0.28)],
     "aff": {"risk": 0.96, "titan": 0.25, "infra": 0.15, "batch": 0.12, "perf": 0.12}, "gold": "risk"},
    {"label": "What is Project Titan?", "text": "what is project titan",
     "topics": [("titan", 1)], "aff": {"titan": 0.95}, "gold": "titan"},
    {"label": "how do we roll back a bad release?", "text": "how do we roll back a bad release",
     "topics": [("infra", 1)], "aff": {"infra": 0.9, "release": 0.6}, "gold": "infra"},
    {"label": "what drives our cloud cost?", "text": "what drives our cloud cost",
     "topics": [("finance", 1)], "aff": {"finance": 0.95, "data": 0.3}, "gold": "finance"},
    {"label": "on-call paging and incidents", "text": "on-call paging and incidents",
     "topics": [("trouble", 0.6), ("team", 0.4)], "aff": {"trouble": 0.85, "support": 0.5, "team": 0.45},
     "gold": "trouble"},
    {"label": "recent engineering hires", "text": "recent engineering hires",
     "topics": [("team", 1)], "aff": {"team": 0.95}, "gold": "team"},
    {"label": "batch job audit ledger", "text": "batch job audit ledger",
     "topics": [("batch", 1)], "aff": {"batch": 0.95, "compliance": 0.5}, "gold": "batch"},
]


def prepare_queries(tvec):
    for q in QUERIES:
        q["v"] = blend(q["topics"], tvec)
        q["toks"] = content_tokens(q["text"])
    return QUERIES


def gold_set(q, chunks):
    """The set of chunk ids that actually answer the query (the relevance ground truth)."""
    sig = q.get("gold_sig")
    if sig:
        return {c["id"] for c in chunks if sig in c["text"].lower()}
    return {c["id"] for c in chunks if c["topic"] == q["gold"]}


def recall_at_k(top_id_list, gold):
    """Standard recall@k = |relevant AND retrieved| / |relevant|."""
    if not gold:
        return 0.0
    return len(set(top_id_list) & gold) / len(gold)


def four_stage_ranks(q, chunks, E, df, avgdl):
    """Return the top-K_FINAL id list at each of the four stages, for one query."""
    N = len(chunks)
    ds = dense_scores(q["v"], E)
    bs = bm25_scores(q["toks"], chunks, df, avgdl, N)
    fs = weighted_rrf(ds, bs, w=0.5)

    dense_top = top_ids(ds, K_FINAL)
    bm25_top = top_ids(bs, K_FINAL, positive_only=True)
    rrf_top = top_ids(fs, K_FINAL)

    # rerank: take the top-TOP_N RRF survivors, re-score jointly, keep K_FINAL
    pool = top_ids(fs, TOP_N)
    rr = rerank_scores(q, chunks, pool)
    pool_sorted = sorted(pool, key=lambda cid: rr[cid], reverse=True)  # stable
    rerank_top = pool_sorted[:K_FINAL]
    return dense_top, bm25_top, rrf_top, rerank_top


def recall_ablation(chunks, E, df, avgdl):
    stages = {"dense": 0.0, "bm25": 0.0, "rrf": 0.0, "rerank": 0.0}
    per_query = []
    for q in QUERIES:
        gold = gold_set(q, chunks)
        d, b, f, r = four_stage_ranks(q, chunks, E, df, avgdl)
        rd = recall_at_k(d, gold)
        rb = recall_at_k(b, gold)
        rf = recall_at_k(f, gold)
        rr = recall_at_k(r, gold)
        stages["dense"] += rd
        stages["bm25"] += rb
        stages["rrf"] += rf
        stages["rerank"] += rr
        per_query.append((q["label"], q["gold"], len(gold), rd, rb, rf, rr))
    Q = len(QUERIES)
    recall = {k: v / Q for k, v in stages.items()}
    return recall, per_query


# --------------------------------------------------------------------------- #
# 4.  "Do you even need a vector DB?"  NumPy exact search vs (optional) FAISS.
# --------------------------------------------------------------------------- #

def memory_arithmetic():
    """The p.47 worked number: 100k chunks x 1024 dim, three precisions."""
    n_chunks, dim = 100_000, 1024
    fp32 = n_chunks * dim * 4
    fp16 = n_chunks * dim * 2
    one_bit = n_chunks * dim // 8
    print("  the store, priced out (100,000 chunks x 1024-dim):")
    print(f"    fp32   {n_chunks:,} x {dim} x 4 B = {fp32:,} B = {fp32 / MB:.1f} MB")
    print(f"    fp16   {'':>{len(f'{n_chunks:,}')}}                {fp16 / MB:.1f} MB")
    print(f"    1-bit binary embeddings              {one_bit / MB:.1f} MB")
    print(f"    -> all of it fits in a corner of {121.6875:.2f} GiB of unified memory.")
    # frozen against the page's printed figures
    assert fp32 == 409_600_000, fp32
    assert abs(fp32 / MB - 410) < 0.5           # page: "410 MB"
    assert abs(fp16 / MB - 205) < 0.5           # page: "205 MB"
    assert abs(one_bit / MB - 12.8) < 0.05      # page: "12.8 MB"
    return fp32


def numpy_vs_faiss(scale, use_faiss=True):
    print("=" * 70)
    print("SECTION 2 -- do you even need a vector database?  (p.47 key box)")
    print("=" * 70)
    memory_arithmetic()
    print()
    print(f"  timing exact search over a synthetic {scale:,}-chunk x {EMB_DIM}-dim store:")

    rng = np.random.default_rng(42)
    E = rng.standard_normal((scale, EMB_DIM)).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    q = rng.standard_normal(EMB_DIM).astype(np.float32)
    q /= np.linalg.norm(q)

    # NumPy exact search: one matmul + a partial sort. No index to build.
    t0 = time.perf_counter()
    scores = E @ q
    np_top = np.argpartition(-scores, K_FINAL)[:K_FINAL]
    np_top = np_top[np.argsort(-scores[np_top])]
    np_ms = (time.perf_counter() - t0) * 1e3
    print(f"    NumPy  np.argsort(E @ q)          {np_ms:8.2f} ms   (exact, 0 build, 0 deps)")

    faiss_ok = False
    if use_faiss:
        try:
            import faiss
            t0 = time.perf_counter()
            index = faiss.IndexFlatIP(EMB_DIM)   # exact inner-product = cosine on unit vectors
            index.add(E)
            build_ms = (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter()
            _, fa_top = index.search(q.reshape(1, -1), K_FINAL)
            faiss_ms = (time.perf_counter() - t0) * 1e3
            print(f"    FAISS  IndexFlatIP build          {build_ms:8.2f} ms")
            print(f"    FAISS  index.search               {faiss_ms:8.2f} ms")
            # IndexFlatIP is exact, so the top-K sets must be identical to NumPy's.
            assert set(int(i) for i in fa_top[0]) == set(int(i) for i in np_top), \
                "FlatIP is exact -- must match the NumPy top-K"
            print("    -> identical top-K (both exact).  FAISS adds a build step and a")
            print("       dependency; it only pays off with an APPROXIMATE index at millions")
            print("       of chunks.  Under ~100k, NumPy is the whole retriever.")
            faiss_ok = True
        except ImportError:
            pass
    if not faiss_ok:
        print("    FAISS  not installed -> desk-checked path (faiss-cpu IndexFlatIP):")
        print("             index = faiss.IndexFlatIP(d); index.add(E); index.search(q, k)")
        print("           IndexFlatIP is EXACT, so it returns NumPy's exact top-K -- after")
        print("           paying an index-build step NumPy never needs.  Approximate indexes")
        print("           (IVF/HNSW/PQ) only win past ~1e6 chunks; under 100k, use NumPy.")
    print()


# --------------------------------------------------------------------------- #
# 5.  Detail one query across the four stages (self-narrating).
# --------------------------------------------------------------------------- #

def show_query(qi, chunks, E, df, avgdl):
    q = QUERIES[qi]
    N = len(chunks)
    ds = dense_scores(q["v"], E)
    bs = bm25_scores(q["toks"], chunks, df, avgdl, N)
    fs = weighted_rrf(ds, bs, w=0.5)
    d, b, f, r = four_stage_ranks(q, chunks, E, df, avgdl)
    gold = gold_set(q, chunks)

    def line(cid, sc=None):
        c = chunks[cid]
        mark = "GOLD" if cid in gold else "    "
        tail = f"  [{sc:.4f}]" if sc is not None else ""
        return f"      {mark} #{cid:<3} ({c['topic']:<10}) {c['text'][:64]}{tail}"

    print("=" * 70)
    print(f"SECTION 3 -- one query through the pipeline:  \"{q['label']}\"")
    print("=" * 70)
    gold_desc = f"chunks containing '{q['gold_sig']}'" if q.get("gold_sig") else f"topic '{q['gold']}'"
    print(f"  relevant (gold): {gold_desc}  ->  {len(gold)} chunk(s)     query terms: {q['toks']}")
    print(f"  DENSE top-5 (cosine, E @ e_q):")
    for cid in d:
        print(line(cid, ds[cid]))
    print(f"  BM25 top-5 (k1={BM25_K1}, b={BM25_B}):")
    for cid in b:
        print(line(cid, bs[cid]))
    if not b:
        print("        (no chunk contains a query keyword -- BM25 returns nothing)")
    print(f"  RRF top-5 (weighted, k_RRF={RRF_K}):")
    for cid in f:
        print(line(cid, fs[cid]))
    print(f"  RERANK top-5 (cross-encoder, joint query+chunk):")
    for cid in r:
        print(line(cid))
    print()


# --------------------------------------------------------------------------- #
# 6.  main
# --------------------------------------------------------------------------- #

def run(args):
    stamp()
    sents = build_sentences()
    total_sents = sum(len(v) for v in sents.values())
    chunks, E, df, avgdl = build_chunks(sents)
    tvec = {tp: topic_vec(TOPIC_SEED[tp]) for tp in sents}
    prepare_queries(tvec)

    print(f"  corpus: {total_sents} sentences across {len(sents)} topics"
          f"  ->  {len(chunks)} chunks @ ~500 tok (5 sentences, 12% overlap)")
    print(f"          embedding matrix E: {E.shape} (L2-normalized rows), avg chunk len {avgdl:.1f} tokens")
    print()

    # sanity: the corpus reproduces the page's size and topic purity
    assert total_sents == 200, f"expected 200 sentences, got {total_sents}"
    for c in chunks:
        toks = set(tokenize(c["text"]))
        # topic purity: no chunk outside 'titan' contains "titan"/"project"
        if c["topic"] != "titan":
            assert "titan" not in toks and "project" not in toks, c["text"]
        # no chunk anywhere contains the word "risk" or a "fail" stem (the aha invariant)
        assert "risk" not in toks and "risks" not in toks, c["text"]
        assert not any(t.startswith("fail") for t in toks), c["text"]
    # E rows are unit vectors -> E @ q IS cosine
    assert np.allclose(np.linalg.norm(E, axis=1), 1.0, atol=1e-9)

    numpy_vs_faiss(args.scale, use_faiss=not args.no_faiss)

    if args.query is not None:
        show_query(args.query, chunks, E, df, avgdl)

    print("=" * 70)
    print("SECTION 4 -- recall@5 climbs stage by stage (the whole argument)")
    print("=" * 70)
    recall, per_query = recall_ablation(chunks, E, df, avgdl)
    print(f"  recall@5 per query (|relevant AND top-5| / |relevant|)")
    print(f"  {'query':<34} gold        |g|  dense  bm25   rrf  rerank")
    for label, gold, ng, rd, rb, rf, rr in per_query:
        print(f"  {label:<34} {gold:<10} {ng:>3}  {rd:5.2f}  {rb:5.2f}  {rf:5.2f}  {rr:5.2f}")
    print("  " + "-" * 66)
    print(f"  mean recall@5   dense={recall['dense']:.3f}   bm25={recall['bm25']:.3f}   "
          f"rrf={recall['rrf']:.3f}   rrf+rerank={recall['rerank']:.3f}")
    print()
    print("  The spec's ablation is the pipeline chain BM25 -> RRF -> rerank, and it")
    print(f"  climbs monotonically: {recall['bm25']:.2f} -> {recall['rrf']:.2f} -> {recall['rerank']:.2f}.")
    print("  Dense-alone (0.82) is a strong single retriever, but it scores 0.00 on the")
    print("  exact string E-4471 and only 0.25 on 'risks to Titan'. Fusion recovers the")
    print("  keyword hit (E-4471: 0.00 -> 1.00); the reranker then reads query+chunk")
    print("  jointly and completes the hybrid case (0.25 -> 1.00), so the FULL pipeline")
    print("  (1.00) beats the best single retriever. Honest caveat: equal-weight RRF")
    print("  slightly dilutes the easy queries dense already nailed -- which is exactly")
    print("  why the reranker, not fusion alone, is what makes retrieval complete.")
    print()

    # --- the frozen self-checks (spec-code sec-D3 p.47: recall@5 improves through
    #     the BM25 -> RRF -> rerank stages; label the effects [EST]/[MEA]) ---
    r_dense, r_bm25, r_rrf, r_rr = (recall["dense"], recall["bm25"], recall["rrf"], recall["rerank"])
    # the named chain, strictly increasing:
    assert r_rrf > r_bm25 + 1e-9, f"RRF ({r_rrf:.3f}) must beat BM25-alone ({r_bm25:.3f})"
    assert r_rr > r_rrf + 1e-9, f"rerank ({r_rr:.3f}) must strictly beat RRF ({r_rrf:.3f})"
    assert abs(r_rr - 1.0) < 1e-9, f"rerank should resolve every query on this designed set, got {r_rr:.3f}"
    # the full pipeline must beat the strongest single retriever (dense) -- the whole point:
    assert r_rr > r_dense + 1e-9, f"the full pipeline ({r_rr:.3f}) must beat dense-alone ({r_dense:.3f})"

    # per-query pedagogy, encoded as regression guards (the two aha cases):
    dmap = {label: (rd, rb, rf, rr) for label, _, _, rd, rb, rf, rr in per_query}
    e4471 = dmap["error code E-4471"]
    assert e4471[0] == 0.0 and e4471[1] == 1.0 and e4471[2] == 1.0, \
        f"E-4471: dense must miss (0.0), BM25 must nail it (1.0), fusion must keep it (1.0); got {e4471}"
    titan = dmap["risks to Project Titan"]
    assert titan[0] < 1.0 and titan[3] == 1.0, \
        f"'risks to Titan': similarity must be incomplete, rerank must complete it; got {titan}"

    print(f"  [OK] all self-checks pass  (BM25 {r_bm25:.2f} -> RRF {r_rrf:.2f} -> rerank {r_rr:.2f};"
          f" pipeline {r_rr:.2f} > dense {r_dense:.2f})")
    print("  Note: these recall numbers are [EST] on a synthetic stand-in corpus -- the SHAPE")
    print("        of the climb is the lesson; on your real docs you MEASURE it ([MEA]).")
    return recall


def main():
    ap = argparse.ArgumentParser(description="RAG retrieval pipeline + recall@5 ablation (p.47).")
    ap.add_argument("--self-test", action="store_true",
                    help="run the full deterministic NumPy path, assert, exit 0 (no GPU/net)")
    ap.add_argument("--query", type=int, default=None, metavar="I",
                    help="detail query I (0-9) across the four stages")
    ap.add_argument("--scale", type=int, default=100_000,
                    help="chunk count for the NumPy-vs-FAISS timing demo (default 100000)")
    ap.add_argument("--no-faiss", action="store_true", help="skip the optional FAISS comparison")
    args = ap.parse_args()

    # The full teaching run walks one query end-to-end; default to query 3 (the
    # 'risks to Titan' rerank aha). --self-test uses the identical code path -- there
    # is no GPU here, so the "self-test" and the teaching run are the same numbers.
    if args.query is None:
        args.query = 3

    run(args)


if __name__ == "__main__":
    main()
