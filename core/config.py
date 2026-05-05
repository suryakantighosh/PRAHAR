"""
prahar/core/config.py
Central configuration — reads from .env via python-dotenv
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# ── Database ─────────────────────────────────────────────────
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://prahar:prahar_secret@localhost:5432/prahar"
)

# ── Redis ────────────────────────────────────────────────────
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ── Celery ───────────────────────────────────────────────────
CELERY_BROKER_URL: str = os.getenv(
    "CELERY_BROKER_URL",
    "amqp://prahar:prahar_secret@localhost:5672//"
)
CELERY_RESULT_BACKEND: str = os.getenv(
    "CELERY_RESULT_BACKEND",
    "redis://localhost:6379/0"
)

# ── Neo4j ────────────────────────────────────────────────────
NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "prahar_neo4j")

# ── Ollama ───────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3:8b")

# ── Free API keys ────────────────────────────────────────────
SHODAN_API_KEY: str = os.getenv("SHODAN_API_KEY", "")
VIRUSTOTAL_API_KEY: str = os.getenv("VIRUSTOTAL_API_KEY", "")
ABUSEIPDB_API_KEY: str = os.getenv("ABUSEIPDB_API_KEY", "")
GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")

# ── PRAHAR settings ──────────────────────────────────────────
PRAHAR_ENV: str = os.getenv("PRAHAR_ENV", "development")
LOG_LEVEL: str = os.getenv("PRAHAR_LOG_LEVEL", "INFO")
CASE_TIMEOUT: int = int(os.getenv("PRAHAR_CASE_TIMEOUT_SECONDS", "90"))
CPIF_THRESHOLD: float = float(os.getenv("PRAHAR_CPIF_THRESHOLD", "0.72"))
