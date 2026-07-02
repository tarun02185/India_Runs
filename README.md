# Redrob AI Candidate Ranking System

> A 7-stage pipeline for ranking 100K AI/ML job candidates against a structured job description, built for the India Runs Data & AI Challenge.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Pre-compute embeddings + FAISS index (one-time, ~15min)
python scripts/precompute.py

# Step 2: Run the full pipeline (hybrid retrieval: BM25 + Dense, ~23s)
python rank.py

# Quick test on 50 sample candidates
python rank.py --sample
```

**Output:** `output/submission.csv` — top-100 ranked candidates with scores and reasoning.

---

## Architecture

```
LOAD (10s) → JD_EXTRACT (0s) → BM25+DENSE RETRIEVE (5s) → FEATURE SCORE (0.2s) → RANK → REASON → OUTPUT
```

### Stage-by-Stage

| Stage | What | 
|-------|------|
| **1. Load** | Parse 100K JSONL candidates | 
| **2. JD Extract** | Build `JDRequirements` structured object from job description | 
| **3. Retrieve** | BM25 + Dense (sentence-transformers/FAISS) with RRF fusion |
| **4. Score** | 6-component feature scoring with sigmoid gate |
| **5. Rank** | Sort by composite score, select top-100 |
| **6. Reason** | Generate fact-based, profile-specific reasoning |
| **7. Output** | Write submission CSV + validate |

### Key Design Decisions

1. **Structured JD extraction** — The JD contains 14 explicit, scorable requirements. A `JDRequirements` data object is computed once and drives every retrieval query and scoring decision.

2. **Steep sigmoid gate** (not hard threshold) — Instead of a binary technical gate that permanently eliminates candidates on a single feature error, a steep sigmoid smoothly penalizes low-relevance candidates while preserving recoverability.

3. **Skills as corroborative evidence** — Skills are NOT trusted at face value. They are cross-validated against career descriptions, endorsements, and usage duration. Keyword stuffing (AI skills + no technical career) triggers severe penalties.

4. **Honeypot detection** — 8 rule-based anomaly checks catch impossible profiles (salary min > max, signup after last-active, expert skills with 0 months, etc.).

5. **Dual retrieval** — BM25 catches keyword matches; dense retrieval catches "plain-language Tier 5" candidates who describe building recommendation systems without using ML jargon.

---

## Scoring Weights

| Component | Weight | What it measures |
|-----------|--------|------------------|
| Technical Relevance | 55% | Career description match, title alignment, hard-req skills |
| Career Structure | 15% | Product company, experience fit, tenure, recency |
| Behavioral | 15% | Engagement, response rate, platform activity |
| Logistical | 10% | Location fit, notice period, work mode |
| Misc | 5% | GitHub score, skill trust |

### Modifiers (multiplicative)
- **Sigmoid gate**: Smoothly zeroes out candidates with near-zero technical relevance
- **Keyword stuffing penalty**: Penalizes claimed AI skills without career evidence
- **Honeypot penalty**: Zeroes out flagged anomalous profiles

---

## Project Structure

```
India_Runs/
├── rank.py                          # Main entry point
├── requirements.txt                 # Dependencies
├── submission_metadata.yaml         # Competition metadata
│
├── src/
│   ├── config.py                    # All paths, parameters, weights
│   ├── jd_requirements.py           # Structured JD extraction
│   ├── data_loader.py               # JSONL/JSON/GZ loading
│   ├── honeypot_detector.py         # 8-rule anomaly detection
│   ├── feature_pipeline.py          # Feature orchestrator + scorer
│   │
│   ├── features/
│   │   ├── text_features.py         # Template detection, career relevance
│   │   ├── role_features.py         # Title, experience, company type
│   │   ├── skill_features.py        # Skill trust, stuffing detection
│   │   ├── behavioral_features.py   # Engagement, platform activity
│   │   └── logistical_features.py   # Location, notice, work mode
│   │
│   ├── retrieval/
│   │   ├── bm25_retriever.py        # Rank-BM25 retrieval
│   │   ├── dense_retriever.py       # Sentence-transformers + FAISS
│   │   └── hybrid_retriever.py      # Reciprocal Rank Fusion
│   │
│   └── output/
│       ├── reason_generator.py      # Fact-based reasoning
│       └── submission_generator.py  # CSV output + validator
│
├── scripts/
│   └── precompute.py                # Offline embedding generation
│
├── artifacts/                       # Pre-computed indexes (gitignored)
└── output/                          # Submission CSV
```

---

## Performance on Full Dataset (100K Candidates)

| Metric | Value |
|--------|-------|
| Total runtime | 23s |
| Candidates loaded | 100,000 |
| BM25 retrieved | 500 |
| Honeypots in top-100 | **0** |
| Score range (top-100) | 0.9485 — 0.8718 |
| Rank #1 | Senior ML Engineer at Zomato (7.2yr, India) |

---

## Constraints Compliance

| Constraint | Status |
|------------|--------|
| CPU only, no GPU | ✅ |
| 16GB RAM | ✅ (~2GB peak) |
| 5-minute runtime | ✅ |
| No network during ranking | ✅ |
| Top-100 output | ✅ |
| Honeypot rate < 10% | ✅ |
| Monotonic scores | ✅ |
| Unique candidate IDs | ✅ |
| Fact-based reasoning | ✅ |

---

## Dependencies

- `numpy` — Array operations
- `rank-bm25` — BM25 retrieval
- `python-dateutil` — Date parsing
- `sentence-transformers` — Dense embeddings (optional, for hybrid retrieval)
- `faiss-cpu` — Vector search (optional, for hybrid retrieval)
- `tqdm` — Progress bars
