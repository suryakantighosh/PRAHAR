# PRAHAR v2

PRAHAR is an evidence-driven intelligence platform for transforming fragmented public-source data into a structured, explainable investigation workflow. It combines asynchronous ingestion, identity resolution, graph-based evidence linking, natural language processing, behavioral scoring, and analyst-ready reporting into a single architecture designed for open-source intelligence, risk assessment, and investigative analysis.

The project is intentionally modular: each stage of the workflow is isolated as a capability, allowing the system to evolve from a research prototype into a production-grade decision-support platform.

---

## Why this project exists

Modern investigations often begin with scattered signals: usernames, domains, emails, phone numbers, photos, public records, and social footprints. These signals rarely live in one place and are often inconsistent, incomplete, or difficult to verify.

PRAHAR exists to reduce that friction by providing a repeatable pipeline that:

- ingests data from multiple public and semi-public sources,
- consolidates identities across platforms,
- builds an evidence graph for relationships and contradictions,
- scores confidence using layered analytics,
- and produces structured, auditable intelligence outputs.

This makes the platform useful for analysts, investigators, researchers, and teams that need defensible, traceable, and explainable workflows rather than opaque black-box predictions.

---

## Problem statement

Investigative work is frequently slowed by:

- fragmented source collection,
- manual correlation across platforms,
- limited provenance and auditability,
- inconsistent confidence assessment,
- and weak handoff from raw signals to usable insight.

PRAHAR addresses that gap by combining ingestion, enrichment, graph analytics, scoring, and reporting in a single platform that can be deployed locally or in a containerized environment.

---

## Product overview

PRAHAR is organized around a multi-stage intelligence pipeline:

1. Seed intake from domains, usernames, names, emails, phones, or images.
2. Evidence collection from public-source connectors and web-based enrichment services.
3. Identity fusion and consolidation using platform-level signals.
4. Entity extraction and relationship analysis through NLP and graph storage.
5. Confidence scoring using layered models such as AMCE, SIF, and TBS.
6. Delivery of analyst-facing artifacts such as PDF briefs and STIX exports.

This design emphasizes transparency, provenance, and practical usability for human operators.

---

## Architecture at a glance

PRAHAR is built as a layered system with the following responsibilities:

- Ingestion layer: acquires raw evidence from external services and public data sources.
- Identity layer: resolves and fuses identity fragments across platforms.
- Knowledge layer: persists structured records and evidence relationships in PostgreSQL and Neo4j.
- Intelligence layer: applies NLP, facial embedding analysis, stylometric fingerprinting, temporal behavior analysis, and confidence scoring.
- Analyst layer: exposes dashboards, APIs, and report-generation workflows for downstream use.

The system is designed around asynchronous processing, modular services, and a shared persistence model so that each stage can be developed and improved independently.

---

## Key capabilities

### Evidence ingestion
- Domain, username, person, and metadata-based ingestion workflows.
- Async connector orchestration with rate-limit awareness and safe error handling.
- Raw-data persistence with audit hashes and provenance-friendly metadata.

### Identity resolution
- Cross-platform identity signal collection.
- Fusion of fragments into consolidated identities.
- Support for breach, phone, and platform profile enrichment.

### Knowledge graph and entity analysis
- Graph-based modeling of evidence relationships and contradictions.
- NLP-based entity extraction and normalization.
- Structured storage for entities, aliases, and evidence relationships.

### Confidence scoring
- Multi-layer adaptive scoring using AMCE.
- Behavioral and stylometric enrichment through SIF and TBS.
- Feedback-driven optimization for ongoing scoring refinement.

### Analyst outputs
- Dashboard endpoints for operational visibility.
- PDF intelligence briefs and STIX export generation.
- Traceable provenance chains for auditability.

---

## Technology stack

PRAHAR is implemented primarily in Python and uses a modern async, service-oriented architecture.

### Core platform
- Python 3.9+
- FastAPI for API exposure
- SQLAlchemy for relational persistence
- PostgreSQL with pgvector support
- Redis for caching and queue-like coordination
- RabbitMQ for asynchronous task coordination
- Neo4j for graph storage and relationship analysis

### Intelligence and data processing
- Async HTTP clients and scraping-oriented tooling
- Playwright for browser-based public record retrieval
- spaCy, transformers, numpy, scipy, scikit-learn for NLP and ML workflows
- OpenCV, Pillow, and deepface-style computer vision capabilities
- ReportLab for PDF report generation
- STIX 2.1 support for structured intelligence exchange

---

## Repository structure

```text
prahar/
  api/                  # FastAPI application entry points
  core/                 # configuration, database, shared runtime settings
  models/               # SQLAlchemy ORM models for persistence
  modules/              # domain-specific capability modules
    c01_ingestion/      # evidence ingestion
    c02_identity/       # identity fusion and resolution
    c03_face/           # face embedding and matching workflows
    c04_records/        # public record retrieval
    c05_nlp/            # entity extraction and NLP enrichment
    c06_graph/          # graph persistence and schema
    c07_amce/           # confidence scoring engine
    c08_brief/          # report generation and provenance logic
    c09_dashboard/      # API and aggregation layer for analyst workflows
    c10_sif/            # stylometric fingerprinting
    c11_tbs/            # temporal behavior scoring
    c12_optimizer/      # feedback-driven weight optimization
  tests/                # unit and integration test suites
infra/                 # infrastructure assets and database initialization
Dockerfile             # application container definition
docker-compose.yml     # local services stack
requirements.txt       # Python dependency manifest
```

---

## Installation

### Prerequisites

- Python 3.9+
- Docker and Docker Compose (recommended for local infrastructure)
- A local or remote PostgreSQL, Redis, RabbitMQ, and Neo4j environment, or the provided Docker stack

### Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
playwright install chromium
```

### Environment setup

```bash
cp env.example .env
```

Review the variables in the environment file and adjust service endpoints, credentials, and optional API keys before starting the platform.

---

## Quick start

### Option A: Docker-based local deployment

```bash
docker-compose up -d
```

This starts the supporting services for PostgreSQL, Redis, RabbitMQ, Neo4j, and the API application container.

### Option B: Local development runtime

```bash
uvicorn prahar.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API documentation is available at:

- http://localhost:8000/docs
- http://localhost:8000/redoc

---

## Configuration

Configuration is driven through environment variables and the shared configuration module. The repository includes an environment template in [env.example](env.example) to support local setup and deployment planning.

### Core environment variables

- POSTGRES_DB / POSTGRES_USER / POSTGRES_PASSWORD
- DATABASE_URL
- REDIS_URL
- CELERY_BROKER_URL / CELERY_RESULT_BACKEND
- NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD
- OLLAMA_BASE_URL / OLLAMA_MODEL
- PRAHAR_ENV / PRAHAR_LOG_LEVEL / PRAHAR_CASE_TIMEOUT_SECONDS / PRAHAR_CPIF_THRESHOLD

Optional free-tier enrichments can be configured through the same environment file if available in your deployment environment.

---

## Running the platform

### API

The FastAPI service exposes a set of investigation-oriented endpoints for ingestion, resolution, scoring, and dashboard views.

### Background processing

The platform is designed to support asynchronous execution for time-consuming tasks such as enrichment and scoring. Background worker patterns are wired through the modular service architecture and the queue/backing infrastructure.

### Verification

```bash
python smoke_test.py
```

This smoke test checks the availability of the major runtime dependencies and helps confirm the environment is ready before further use.

---

## Docker deployment

The repository includes a containerized deployment model that provisions the core infrastructure services required by the platform:

- PostgreSQL with pgvector support
- Redis
- RabbitMQ
- Neo4j
- An API container for the application runtime

This is the recommended path for local development and controlled evaluation environments.

---

## API overview

The API surface is organized around investigation workflows:

- POST /api/v1/ingest/domain
- POST /api/v1/ingest/username
- POST /api/v1/resolve/username
- POST /api/v1/score/case
- GET /api/v1/dashboard/health
- GET /api/v1/dashboard/stats
- GET /api/v1/dashboard/cases
- GET /api/v1/dashboard/graph
- GET /api/v1/dashboard/quotas

These routes provide both operational health and analyst-facing reporting capabilities.

---

## Workflow

A typical investigation flow looks like this:

1. Seed an entity or subject using a domain, username, email, phone, or image.
2. Ingest raw evidence from the relevant connectors.
3. Persist and normalize the records into the shared datastore.
4. Resolve identity fragments across available platforms.
5. Enrich the case with NLP, graph relationships, and behavioral signals.
6. Score the subject using AMCE and associated supporting models.
7. Publish a summarised intelligence brief or STIX bundle for analyst review.

This workflow emphasizes repeatability, observability, and evidence linkage rather than ad hoc analysis.

---

## Security model

PRAHAR is built with an investigation-oriented security posture rather than a consumer SaaS posture. The current implementation emphasizes:

- environment-based secret management,
- service-level configuration isolation,
- cryptographic provenance for report generation,
- and traceable evidence persistence.

For production deployment, the following hardening areas should be reviewed:

- authentication and authorization for API access,
- network segmentation between internal services,
- secrets rotation and vault integration,
- stricter CORS and ingress control,
- and immutable audit logging for operational events.

---

## Performance and scalability

The architecture is designed for asynchronous execution and modular scaling:

- concurrent ingestion tasks reduce wall-clock latency for multi-source collection,
- CPU-heavy ML and NLP stages are isolated from the event loop,
- graph and vector-backed storage support relationship and similarity workloads,
- and the API layer remains stateless and suitable for container-based deployment.

The design is suitable for moderate-scale operational use and can be extended with additional worker pools, queue scaling, and dedicated inference services as demand grows.

---

## Roadmap

The current repository reflects a strong foundation for an investigative intelligence platform. The next phase of maturity would focus on:

- production-grade authentication and RBAC,
- authenticated analyst workspaces and case management,
- durable job orchestration and retries,
- richer graph analytics and visualization,
- expanded connectors and enrichment providers,
- and more advanced model evaluation and tuning workflows.

---

## Screenshots and visual artifacts

This repository is structured to support rich analyst experiences, including:

- dashboard health and pipeline status views,
- case detail and evidence summaries,
- graph-based relationship views,
- confidence score breakdowns,
- and generated intelligence briefs.

These artifacts are well suited for future documentation expansion and public demos.

---

## Architecture section

PRAHAR combines several specialized subsystems into a cohesive intelligence workflow:

- C-01 ingestion collects and stores raw evidence.
- C-02 identity resolution consolidates fragments into identity representations.
- C-03 and C-04 enrich the case with face and public-record signals.
- C-05 and C-06 turn evidence into entities and graph relationships.
- C-07, C-10, C-11, and C-12 produce confidence scoring and optimization feedback.
- C-08 and C-09 deliver analyst-facing reporting and dashboard access.

The platform is designed not just for data collection, but for producing defensible, human-readable intelligence from noisy and fragmented sources.

---

## Contributing

Contributions are welcome. For substantial changes, please open an issue first to discuss the direction of the work and its impact on the platform architecture.

Suggested contribution areas include:

- connector reliability and source management,
- scoring methodology improvements,
- dashboard and reporting enhancements,
- infrastructure hardening and deployment automation,
- and documentation quality.

---

## License

This repository does not currently include a declared public license file. Before publishing the repository publicly, add a standard open-source license such as MIT or Apache-2.0 and align it with your intended distribution model.

---

## Acknowledgements

PRAHAR builds on a broad ecosystem of open-source software for web access, data processing, graph analytics, NLP, and reporting. The project is intended as an applied research and engineering platform that values transparency, modularity, and operational usefulness.
