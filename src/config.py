"""
Configuration module for the Redrob AI Candidate Ranking System.

Centralizes all paths, parameters, thresholds, and scoring weights.
Every tunable value lives here — nothing is hardcoded in business logic.
"""

from pathlib import Path
from dataclasses import dataclass, field

# ─── Paths ───────────────────────────────────────────────────────────────────

# Project root: parent of the src/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directory
DATA_DIR = PROJECT_ROOT / "data"

CANDIDATES_JSONL = DATA_DIR / "candidates.jsonl"
SAMPLE_CANDIDATES = DATA_DIR / "sample_candidates.json"

# Pre-computed artifacts
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "embeddings.npy"
FAISS_INDEX_PATH = ARTIFACTS_DIR / "faiss_index.bin"
BM25_INDEX_PATH = ARTIFACTS_DIR / "bm25_index.pkl"
CANDIDATE_IDS_PATH = ARTIFACTS_DIR / "candidate_ids.pkl"

# Output
OUTPUT_DIR = PROJECT_ROOT / "output"
SUBMISSION_CSV = OUTPUT_DIR / "submission.csv"

# ─── Reproducibility ─────────────────────────────────────────────────────────

RANDOM_SEED = 42

# ─── Pipeline Parameters ─────────────────────────────────────────────────────

# How many candidates to output
TOP_K = 100

# Coarse filter: how aggressively to exclude non-technical candidates
# This is a HIGH-RECALL filter — we'd rather keep noise than lose a real candidate
COARSE_FILTER_MIN_EXPERIENCE_YEARS = 1.5
COARSE_FILTER_MAX_EXPERIENCE_YEARS = 25.0

# Retrieval: how many candidates to retrieve from each method
BM25_RETRIEVAL_DEPTH = 1000
DENSE_RETRIEVAL_DEPTH = 1000
# After RRF fusion, keep the top N for feature scoring
HYBRID_RETRIEVAL_DEPTH = 500

# Reciprocal Rank Fusion parameter (standard value from literature)
RRF_K = 60

# ─── Honeypot Detection ──────────────────────────────────────────────────────

# Anomaly score threshold: candidates scoring >= this are flagged as honeypots
HONEYPOT_SCORE_THRESHOLD = 3.0

# ─── Scoring Weights ─────────────────────────────────────────────────────────
# These are the weights for the final composite score.
# All component scores are normalized to [0, 1] BEFORE weighting.

@dataclass
class ScoringWeights:
    """Weights for the final composite score computation."""
    technical_relevance: float = 0.55     # Career desc match + validated skills + title
    career_structure: float = 0.15        # Product co, tenure, experience range
    behavioral: float = 0.15             # Response rate, activity, availability
    logistical: float = 0.10             # Location, notice, work mode
    misc: float = 0.05                   # GitHub, certifications, verification

SCORING_WEIGHTS = ScoringWeights()

# ─── Sigmoid Gate Parameters ─────────────────────────────────────────────────
# Instead of a hard gate, we use a steep sigmoid to smoothly penalize
# candidates with low technical relevance.
# sigmoid(steepness * (x - threshold))
# At threshold: multiplier = 0.5
# Below threshold: multiplier drops rapidly toward 0
# Above threshold: multiplier rises rapidly toward 1

SIGMOID_GATE_STEEPNESS = 20.0
SIGMOID_GATE_THRESHOLD = 0.15

# ─── Embedding Model ─────────────────────────────────────────────────────────

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
EMBEDDING_BATCH_SIZE = 256

# ─── Consulting Firms (for career type detection) ────────────────────────────

CONSULTING_FIRMS = frozenset({
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mindtree", "mphasis", "l&t infotech",
    "ltimindtree", "persistent systems", "zensar", "hexaware",
    "cyient", "sonata software", "niit technologies", "coforge",
})

# ─── Template Detection ──────────────────────────────────────────────────────

# Signature phrases that indicate a templated (non-technical) summary
TEMPLATE_SIGNATURES = [
    "Lately I've been curious about how AI tools could augment my work",
    "I've experimented with ChatGPT and a few other tools for productivity",
    "Open to roles where I can apply my domain expertise alongside emerging AI",
    "I've built and led teams, owned KPIs, and driven business outcomes",
]

# ─── Non-Technical Titles (for coarse filter) ────────────────────────────────
# These titles, combined with non-technical career descriptions,
# indicate a noise candidate.

NON_TECHNICAL_TITLES = frozenset({
    "hr manager", "human resources", "marketing manager", "accountant",
    "content writer", "graphic designer", "civil engineer",
    "mechanical engineer", "customer support", "sales manager",
    "operations manager", "brand manager", "financial analyst",
    "supply chain", "procurement", "real estate", "teacher",
    "professor", "nurse", "doctor", "lawyer", "legal",
    "receptionist", "administrative", "executive assistant",
})

# ─── Technical Titles (for role relevance scoring) ───────────────────────────

STRONG_TECHNICAL_TITLES = frozenset({
    "ai engineer", "ml engineer", "machine learning engineer",
    "senior ai engineer", "senior ml engineer",
    "senior machine learning engineer", "data scientist",
    "senior data scientist", "research engineer",
    "applied scientist", "nlp engineer", "search engineer",
    "ranking engineer", "recommendation engineer",
    "staff engineer", "principal engineer",
})

MODERATE_TECHNICAL_TITLES = frozenset({
    "software engineer", "senior software engineer",
    "backend engineer", "data engineer", "senior data engineer",
    "platform engineer", "infrastructure engineer",
    "full stack engineer", "python developer",
    "java developer", ".net developer", "devops engineer",
    "cloud engineer", "systems engineer",
})
