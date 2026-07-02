"""
Structured Job Description Requirements.

This module extracts and encodes the JD's requirements into a machine-usable
data object. Every scoring decision in the pipeline traces back to a specific
field in this object.

Built from: job_description.md (lines referenced in comments)
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class JDRequirements:
    """
    Structured extraction of all scorable requirements from the JD.

    This object is computed ONCE at startup and passed to every scoring
    module. It serves as the single source of truth for what the JD
    requires, prefers, and disqualifies.
    """

    # ── Hard Requirements (JD lines 49-53: "Things you absolutely need") ─────
    # These are MUST-HAVE. Absence of all of them is a strong negative signal.
    hard_requirements: tuple[str, ...] = (
        "production_embeddings_retrieval",   # L50: production experience with embeddings-based retrieval
        "vector_db_or_hybrid_search",        # L51: vector DBs or hybrid search infra
        "strong_python",                     # L52: strong Python, code quality
        "ranking_evaluation_frameworks",     # L53: NDCG, MRR, MAP, A/B testing
    )

    # ── Soft Preferences (JD lines 55-59: "Things we'd like but won't reject") ─
    soft_preferences: tuple[str, ...] = (
        "llm_finetuning",                    # L56: LoRA, QLoRA, PEFT
        "learning_to_rank_models",           # L57: XGBoost-based or neural LTR
        "hrtech_marketplace_exposure",       # L58: HR-tech, recruiting tech
        "distributed_systems",               # L59: large-scale inference optimization
        "opensource_contributions",           # L60: open-source in AI/ML space
    )

    # ── Disqualifiers (JD lines 40-42, 64-69: explicit rejection criteria) ───
    disqualifiers: tuple[str, ...] = (
        "pure_research_no_production",       # L40: career in pure research without deployment
        "only_recent_llm_experience",        # L41: AI experience only post-ChatGPT, <12 months
        "not_writing_code_18mo",             # L42: moved to architecture/tech-lead, no recent code
        "title_chaser",                      # L65: switching every 1.5 years for title bumps
        "framework_enthusiast_only",         # L66: LangChain tutorials, not systems thinking
        "consulting_only_career",            # L67: entire career at TCS/Infosys/Wipro etc.
        "cv_speech_robotics_only",           # L68: primary expertise not NLP/IR
        "closed_source_only_5yr",            # L69: no external validation for 5+ years
    )

    # ── Ideal Candidate Profile (JD lines 88-95: "How to read between the lines") ─
    experience_years_ideal_min: float = 5.0          # L37, L89
    experience_years_ideal_max: float = 9.0          # L37
    experience_years_hard_min: float = 2.0           # L37: "range, not requirement"
    experience_years_hard_max: float = 20.0          # generous upper bound
    applied_ml_years_min: float = 4.0                # L89: 4-5 in applied ML/AI
    shipped_ranking_search_recsys: bool = True        # L90
    product_company_required: bool = True             # L89: "not pure services"
    location_preferences: tuple[str, ...] = (         # L74, L92
        "pune", "noida", "hyderabad", "mumbai",
        "delhi", "ncr", "gurgaon", "gurugram",
        "bangalore", "bengaluru",
    )
    platform_active: bool = True                      # L93

    # ── BM25 Query Terms ─────────────────────────────────────────────────────
    # These are the terms used to query the BM25 index over career descriptions.
    # Derived from hard requirements + plain-language Tier 5 vocabulary.
    # Organized by priority: terms from hard requirements first, then
    # plain-language equivalents that catch Tier 5 candidates BM25 might miss.

    retrieval_query_primary: tuple[str, ...] = (
        # From hard requirements (L50-53)
        "embeddings", "retrieval", "ranking", "search",
        "vector database", "FAISS", "Pinecone", "Weaviate",
        "Qdrant", "Milvus", "Elasticsearch", "OpenSearch",
        "NDCG", "MRR", "evaluation", "A/B testing",
        "sentence-transformers", "hybrid search",
        # Core ML/AI terms
        "machine learning", "deep learning", "NLP",
        "transformer", "fine-tuning", "model training",
        "neural network", "PyTorch", "TensorFlow",
    )

    retrieval_query_secondary: tuple[str, ...] = (
        # Plain-language Tier 5 descriptions (JD L102)
        "recommendation system", "recommendation engine",
        "search system", "search engine", "matching system",
        "what users see", "candidate ranking",
        "deployed to users", "production", "shipped",
        "end-to-end", "real users", "at scale",
        "information retrieval", "relevance",
        "personalization", "content ranking",
    )

    # ── Career Description Positive Signals ──────────────────────────────────
    # Keywords in career descriptions that indicate relevant experience.
    # Split into strong (direct match to JD requirements) and moderate.

    career_strong_positive: tuple[str, ...] = (
        # Direct matches to "shipped ranking/search/recommendation"
        "ranking system", "ranking model", "ranking pipeline",
        "search system", "search engine", "search infrastructure",
        "recommendation system", "recommendation engine", "recommender",
        "retrieval system", "retrieval pipeline", "retrieval quality",
        "embedding", "embeddings", "vector search", "vector database",
        "hybrid search", "semantic search",
        "NDCG", "MRR", "MAP", "precision", "recall",
        "A/B test", "online evaluation", "offline evaluation",
        "deployed to production", "shipped to users",
        "candidate matching", "talent matching",
    )

    career_moderate_positive: tuple[str, ...] = (
        # Adjacent technical experience
        "machine learning", "ML model", "ML pipeline",
        "deep learning", "neural network", "NLP",
        "text classification", "named entity", "NER",
        "transformer", "BERT", "GPT", "fine-tuning", "fine-tune",
        "PyTorch", "TensorFlow", "scikit-learn", "XGBoost",
        "data pipeline", "feature engineering", "feature store",
        "model serving", "model deployment", "inference",
        "Python", "Spark", "Airflow",
    )

    # ── Career Description Negative Signals ──────────────────────────────────
    # Keywords in career descriptions that indicate NON-relevant experience.

    career_negative: tuple[str, ...] = (
        "brand design", "creative direction", "packaging design",
        "accounting", "financial reporting", "audit",
        "sales quota", "ARR quota", "enterprise sales",
        "mechanical engineering", "CAD", "SolidWorks", "ANSYS",
        "civil engineering", "structural", "construction",
        "customer support", "support tickets", "tier-1 and tier-2 tickets",
        "marketing campaign", "SEO", "content writing",
        "content creation", "social media",
        "HR management", "recruitment process",
        "supply chain", "logistics", "procurement",
    )

    # ── Skills That Map to Hard Requirements ─────────────────────────────────

    skills_hard_req_embeddings: tuple[str, ...] = (
        "sentence-transformers", "embeddings", "BGE", "E5",
        "OpenAI embeddings", "word2vec", "fasttext",
        "SBERT", "embedding models",
    )

    skills_hard_req_vector_db: tuple[str, ...] = (
        "FAISS", "Pinecone", "Weaviate", "Qdrant", "Milvus",
        "Elasticsearch", "OpenSearch", "Chroma", "pgvector",
        "vector database", "vector search", "hybrid search",
    )

    skills_hard_req_python: tuple[str, ...] = (
        "Python",
    )

    skills_hard_req_eval: tuple[str, ...] = (
        "NDCG", "MRR", "MAP", "BM25", "evaluation",
        "A/B testing", "ranking evaluation",
        "learning to rank", "information retrieval",
    )

    skills_soft_pref_llm: tuple[str, ...] = (
        "Fine-tuning LLMs", "LoRA", "QLoRA", "PEFT",
        "LLM", "GPT", "instruction tuning",
        "Hugging Face Transformers",
    )

    skills_soft_pref_ltr: tuple[str, ...] = (
        "XGBoost", "LightGBM", "CatBoost",
        "learning to rank", "ranking model",
    )

    # ── Work Mode Compatibility ──────────────────────────────────────────────
    # JD L74: "Hybrid — flexible cadence"
    compatible_work_modes: tuple[str, ...] = ("hybrid", "flexible", "remote")

    # ── Notice Period ────────────────────────────────────────────────────────
    # JD L75: "sub-30-day notice... can buy out up to 30 days"
    notice_period_ideal_max: int = 30       # ideal
    notice_period_acceptable_max: int = 60  # still in scope, bar higher
    notice_period_concern: int = 90         # significant concern


# Singleton instance — import this wherever needed
JD = JDRequirements()
