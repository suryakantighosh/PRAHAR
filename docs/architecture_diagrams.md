# Architecture diagram specifications for PRAHAR

This document defines publication-quality diagram specifications for the public-facing documentation package.

## 1. Overall system architecture

### Purpose
Explain how ingestion, identity resolution, graph analytics, scoring, and reporting integrate into one platform.

### Why it matters
This is the most important visual for understanding the repository as a system rather than as a set of isolated modules.

### Recommended size
2200 × 1400 px

### Aspect ratio
16:10

### Visual style
Modern enterprise architecture, clean white background, high contrast, subtle gradients, minimal clutter.

### Layout
Top row: external data sources and analyst actors. Middle row: core platform services. Bottom row: storage and output artifacts.

### Colour palette
- Background: #F8FAFC
- Primary navy: #0F172A
- Accent red: #DC2626
- Accent blue: #2563EB
- Accent green: #16A34A
- Neutral gray: #64748B

### Typography
Use a sans-serif font such as Inter or Segoe UI. Titles in bold, labels in regular weight.

### Legend
Include a small legend for input sources, core services, storage systems, and outputs.

### Grid layout
Five columns, three rows, with balanced spacing and clear grouping.

### Background
White with very light gray panel framing for the main architecture block.

### Boxes
Rounded rectangles with subtle shadow and thin borders.

### Connections
Arrows flow left-to-right and top-to-bottom. Use solid arrows for primary flow and dashed arrows for feedback or enrichment.

### Component descriptions
- Data sources: domains, usernames, emails, phones, faces, public records
- Ingestion engine: orchestrates evidence collection
- Identity engine: fuses fragments into consolidated identities
- Graph store: Neo4j evidence relationships
- NLP / analytics: entity extraction, behavioral scoring, stylometry
- Scoring engine: AMCE and related confidence models
- Reporting layer: PDFs, STIX, dashboard APIs

### Placement
- Left: seed inputs and data sources
- Center: ingestion and intelligence modules
- Right: storage and analyst outputs

### Footer
Add a compact footer: “PRAHAR v2 — evidence-driven intelligence platform”

---

## 2. Investigation workflow / agent lifecycle

### Purpose
Show how an investigation case progresses across modules from intake to reporting.

### Why it matters
This diagram helps explain the product narrative and the lifecycle of a case.

### Recommended size
2200 × 1600 px

### Aspect ratio
4:3

### Visual style
Flowchart with sequential stages and small callout annotations.

### Layout
Horizontal pipeline with 7 numbered stages.

### Boxes
Each stage should be a rounded rectangle with a small icon and a short subtitle.

### Connections
Arrows go from stage to stage, with optional feedback loops to scoring and optimization.

### Component descriptions
1. Seed intake
2. Evidence collection
3. Raw persistence
4. Identity fusion
5. NLP and graph enrichment
6. Scoring and risk evaluation
7. Report generation

### Placement
Stages aligned horizontally, with a feedback loop from scoring back to analyst review and optimization.

---

## 3. Data flow and provenance model

### Purpose
Illustrate how raw evidence becomes a provenance-linked intelligence artifact.

### Why it matters
This is important because provenance and auditability are central to the system’s value proposition.

### Recommended size
2200 × 1400 px

### Aspect ratio
16:10

### Visual style
Clean data flow diagram with layered blocks and hash-chain annotations.

### Layout
Top: source records. Middle: processing modules. Bottom: immutable outputs and audit artifacts.

### Boxes
Include a small hash icon or anchor symbol near provenance nodes.

### Connections
Use labeled arrows: raw data → normalized record → entity graph → scored identity → report artifact.

### Annotations
Include a note: “Each output is linked by SHA-256 provenance chain.”

---

## 4. Deployment architecture

### Purpose
Show how the platform fits into a container-based deployment with supporting infrastructure services.

### Why it matters
This helps readers understand how to run the system locally or in a dev/test environment.

### Recommended size
2200 × 1600 px

### Aspect ratio
4:3

### Visual style
Cloud-native infrastructure diagram with Docker and service containers.

### Layout
Left: user/API. Middle: containerized services. Right: data stores.

### Boxes
Use Docker-style container shapes where appropriate, with service names.

### Containers
- API container
- PostgreSQL
- Redis
- RabbitMQ
- Neo4j
- Ollama

### Connections
Arrows from API to each dependency with labels such as persistence, caching, queue, graph, and inference.

---

## Combined image prompt

Create a polished, enterprise-style software architecture diagram for a platform called PRAHAR v2. The image should look suitable for a GitHub README, technical whitepaper, and architecture presentation. Use a modern clean style with a light gray background, subtle shadows, rounded rectangles, thin borders, and a professional blue-red-green palette. The composition should show a left-to-right workflow from external intelligence inputs through ingestion, identity fusion, analytics, scoring, and report generation. Include labeled boxes for data sources such as domains, usernames, emails, phones, faces, and public records; a central orchestration layer for ingestion and intelligence modules; a graph and relational storage layer; and a right-side reporting layer with dashboard APIs, PDF briefs, and STIX outputs. Include subtle connection arrows, a small legend, and a footer reading “PRAHAR v2 — evidence-driven intelligence platform.” Keep the diagram uncluttered, highly legible, and visually balanced, with enough space for labels and annotations. The layout should feel like a mature enterprise architecture diagram rather than a generic flowchart.
