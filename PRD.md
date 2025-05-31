Product Requirements Document (PRD)

## **1. Executive Summary**

Build a vertically-integrated, 100-camera global network that extracts entropy from live CCTV feeds, validates randomness, computes multi-metric “coherence” analytics, correlates anomalies with geo-located news/sports/events and renders everything on a real-time 3-D globe. Single-VPS baseline must hit <100 ms end-to-end latency and cost <€120 / month, while retaining a clear path to horizontal scale (K3s).

---

## **2. Goals & Success Metrics**

| **ID** | **Goal** | **KPI / Target** |
| --- | --- | --- |
| G-1 | Generate cryptographically-secure TRNG stream | ≥ 256 b / node / sec, NIST SP-800-22 pass-rate ≥ 99 % |
| G-2 | Real-time coherence analytics | CEC, PNS, FCI, MFR refresh ≤ 5 s |
| G-3 | Dashboard UX | Globe FPS ≥ 55 on M1/M3 laptop at 1080 p |
| G-4 | Correlate spikes with events | Relevant news/event returned ≤ 30 s after spike |
| G-5 | Research experiments | One-click experiment with rolling p-value & auto-stop |

---

## **3. Scope**

### **In-Scope**

1. **Edge Entropy Extraction** from RTSP/HTTP cameras.
2. **Quality-Assurance Pipeline** (sub-second NIST subset, ApEn, χ², auto-corr).
3. **GPU-accelerated Coherence Metrics** (CEC, PNS, FCI, MFR, RCC).
4. **Event Correlation Service** pulling **NewsAPI**, **API-Sports**, **PredictHQ**, enriched via local Llama-3.
5. **FastAPI / GraphQL Gateway** with JWT auth.
6. **Next.js + Three.js/WebGPU** dashboard & experiment console.
7. **CI/CD, monitoring, IaC** for single-VPS deployment.

### **Out-of-Scope (v1)**

*Mobile app, SMS notifications, >1000 nodes*, advanced AI prediction models.

---

## **4. Personas & Use Cases**

| **Persona** | **Primary Tasks** |
| --- | --- |
| Researcher | Launch global meditation experiment; export CSV for publication |
| Casual Visitor | Watch globe heat-map, drill down to city-level metrics |
| Ops Engineer | Monitor node health, add / revoke cameras |
| Data Scientist | Query raw entropy, run custom stats notebooks |

---

## **5. Functional Requirements**

| **Req-ID** | **Description** | **Acceptance Criteria** |
| --- | --- | --- |
| FR-1 | Edge agent pulls frames (1–30 FPS), outputs SHA-3 512-bit samples | Hash digest arrives via NATS topic entropy.raw.<id> ≤ 100 ms after frame |
| FR-2 | QA service scores each sample (0-100) | Node flagged when rolling score < 70 for 10 s |
| FR-3 | Analyzer computes metrics every 5 s sliding window | Metrics row written to TimescaleDB and UI receives WS delta |
| FR-4 | Spike detector (Z > 3) emits event to Event-Corr | Correlation service returns ≥ 1 headline or sports event with geo match confidence ≥ 0.7 |
| FR-5 | Experiment manager schedules, tracks Wald SPRT | Auto-stop & persist summary when p < 0.01 or timeout |
| FR-6 | Dashboard globe shows node spheres size ∝ quality, color ∝ coherence | Refresh ≤ 5 s; FPS ≥ 55 |

---

## **6. Non-Functional Requirements**

| **Category** | **Target** |
| --- | --- |
| **Perf.** | <100 ms camera➔metric; end-to-end CPU <70 % on 8vCPU |
| **Reliability** | VPS uptime ≥ 99.9 %; automatic node failover |
| **Security** | mTLS edge→ingest, JWT auth, LUKS disk crypto |
| **Scalability** | Linear scale to 300 nodes by adding worker pods |
| **Cost** | OPEX ≤ €120/mo single VPS |
| **Compliance** | GDPR: only hashes + coarse geo stored |

---

## **7. System Architecture Overview**

Mermaid:

```
flowchart TD
    subgraph Edge
        CAM[Camera] --> AGENT(Rust Entropy Agent)
    end
    AGENT -- Protobuf:NATS --> INGEST[Ingest-GW]
    INGEST --> CORE[Entropy-Core]
    CORE --> QA[QA Engine]
    QA -- good --> ANALYZER[GPU Coherence Analyzer]
    QA -- Timescale --> TSDB[(TimescaleDB)]
    ANALYZER --> TSDB
    ANALYZER -- WS ⇄ --> API[FastAPI/GraphQL]
    API --> UI[Next.js Globe]
    ANALYZER -- spike --> CORR[Event Correlator]
    CORR -- events --> TSDB
    API --> UI
    EXPERIMENT[Experiment Manager] -. query .-> TSDB
    UI -- mutate --> API
```

(High-level only; detailed component call graph delivered earlier.)

---

## **8. Technology Stack (locked)**

| **Layer** | **Tech** |
| --- | --- |
| Edge | Rust 1.78, tokio, opencv-rust, sha3, prost |
| Messaging | NATS JetStream 3.x |
| Core Libs | Rust + PyO3 bindings |
| GPU | CUDA 12, CuPy 13, Numba 0.60 |
| API | FastAPI 0.111 + Ariadne GraphQL |
| Auth | Keycloak 24 / JWT |
| DB | TimescaleDB 2.15 (PostgreSQL 16 ext) |
| Cache | Redis 7 |
| Front-end | Next.js 15, React 19, Three.js 0.163, WebGPU |
| DevOps | Docker, Helm, k3s, GitHub Actions, Prometheus + Grafana |

---

## **9. Detailed Component Specs**

### **9.1 Entropy Agent (**

### **services/entropy_agent**

### **)**

*Language*: Rust

*Entry* : main.rs

*Key structs*

| **Struct** | **Purpose** |
| --- | --- |
| FrameGrabber | Async RTSP pull, yields Mat |
| EntropyCalc | Implements Atmospheric, Photon, Thermal, Chaos formulas (SIMD) |
| VonNeumann | Debias, returns Vec<u8> |
| Publisher | Wraps NATS, sends EntropySample protobuf |

*Config (agent.toml)*

```
node_id       = 42
rtsp_url      = "rtsp://username:pwd@host/stream"
fps_min       = 1
fps_max       = 30
tls_cert_path = "certs/node.pem"
```

Unit tests: cargo test -- --nocapture.

### **9.2 Ingest Gateway (**

### **services/ingest_gw**

### **)**

*Axum* async web-socket & NATS producer.

Routes:

| **Method** | **Path** | **Purpose** |
| --- | --- | --- |
| WS | /v1/entropy | Mutual-TLS stream from agents |
| GET | /healthz | Prometheus metric scrape |

### **9.3 Entropy-Core (**

### **services/entropy_core**

### **)**

Rust lib crate → compiled as libentropy_core.so, imported by Python harness for batch.

Public FFI:

```
#[no_mangle] pub extern "C" fn merge_entropy(buf_ptr:*const u8, len:usize,
                                             out_ptr:*mut u8) -> usize;
```

Produces 512-bit digest per input vector.

### **9.4 QA Engine (**

### **services/qa_engine**

### **)**

Rust; consumes entropy.merged subject, scores, writes node_quality hypertable:

```
CREATE TABLE node_quality(
  ts TIMESTAMPTZ,
  node_id BIGINT,
  quality FLOAT
);
```

Publishes entropy.good for values ≥ 70.

### **9.5 Coherence Analyzer (**

### **services/coherence_analyzer**

### **)**

Python, entry main.py.

Loop every 5 s:

```
buf = fetch_timescale("SELECT * FROM entropy_good WHERE ts>now()-'5s'")
metrics = gpu.compute_all(buf)  # cupy kernels
write_timescale(metrics_df, "coherence_metrics")
pub_ws(metrics_delta_json)
```

CUDA kernels in libs/cuda_kernels/.

### **9.6 Event Correlator (**

### **services/event_correlator**

### **)**

Subscribes spike.detected → queries News/Sports APIs (keys via Vault).

LLM annotator:

```
from ctransformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-3-8B-Instruct")
```

Caches response in Redis (TTL 1 h).

### **9.7 Experiment Manager (**

### **services/experiment_mgr**

### **)**

Celery worker (Redis broker). Task run_experiment(id):

1. Poll metric.
2. Compute Wald SPRT p-value.
3. Update experiments table.
4. When finished, publish experiment.done.<id>.

### **9.8 API Gateway (**

### **api/gateway**

### **)**

| **Type** | **Route** | **Payload** | **Notes** |
| --- | --- | --- | --- |
| GraphQL | /graphql | subscription metricsDelta | auth header Bearer |
| REST POST | /experiments | JSON | create experiment |
| REST GET | /events?lat&lon&dt | JSON | news/sports list |

OpenAPI auto-gen in CI.

### **9.9 Front-end (**

### **frontend/dashboard**

### **)**

Next.js /app/globe/page.tsx renders <Globe /> using three-globe.

State from useGraphQLSubscription.

CSS via Tailwind (already configured).

---

## **10. External API Keys (store in Vault)**

| **Provider** | **Env Var** | **Notes** |
| --- | --- | --- |
| NewsAPI v3 | NEWS_API_KEY | 1 000 req / day |
| API-Sports | SPORTS_API_KEY | football, basketball |
| PredictHQ | PREDICTHQ_TOKEN | city events |
| OpenWeather (optional) | WX_KEY | future weather correlation |

---

## **11. Dev & Deployment Workflow**

1. **Clone** → ./scripts/dev-bootstrap.sh (installs Rust, Python, Node).
2. **Local stack**: docker compose up (Timescale, NATS, Redis, Keycloak, UI).
3. **Unit tests**:
    - Rust: cargo test
    - Python: pytest -q
    - UI: pnpm test
4. **Lint/format**: pre-commit run --all-files.
5. **Build images**: act -j build.
6. **Push to registry** via GitHub Action on main.
7. **Deploy**: Watchtower auto-pulls latest tag on VPS.
8. **Helm** charts in deploy/helm/ for k3s; helm upgrade --install gcp ./deploy/helm.

---

## **12. Testing Strategy**

| **Level** | **Tool** | **Coverage Target** |
| --- | --- | --- |
| Unit | Rust cargo + Python pytest | ≥ 85 % lines |
| Integration | docker compose -f test.yml | ingest⇄QA⇄analyzer happy-path |
| E2E | Playwright | UI metrics appear in <6 s |
| Performance | Locust (WS) | sustain 200 clients, <500 ms p95 |
| Security | Trivy scan images, Snyk dependency audit |  |

---

## **13. Monitoring & Alerting**

*Prometheus* exporters: NATS, Timescale, Rust apps (opentelemetry-prom).

Alerts (Grafana OnCall):

| **Alert** | **Threshold** |
| --- | --- |
| CPU > 90 % 1 min | critical |
| End-to-end latency > 150 ms | warn |
| Node flagged count > 5 | info |
| Disk free < 20 GB | critical |

---

## **14. Risks & Mitigations**

| **Risk** | **Likelihood** | **Impact** | **Mitigation** |
| --- | --- | --- | --- |
| CCTV feed outages | High | Medium | Auto-throttle; redundant nodes |
| API provider rate-limits | Medium | Medium | Redis cache, exponential back-off |
| GPU memory leak | Low | High | CuPy pooled allocator, nightly restart job |
| Single-VPS failure | Medium | High | Daily pg_dump to B2, documented restore runbook |

---

## **15. Open Questions**

1. Which 100 public cameras meet licensing for data redistribution?
2. Do we store raw frames for future audit (GDPR)?
3. Exact set of cultural indices for MFR weights – source dataset?

---

## **16. Appendix – Directory Skeleton (ready for**

## **Cline**

## **)**

```
gcp-trng
├── cmd/
│   ├── ingest-gw/
│   ├── entropy-agent/
│   └── coherence-cli/
├── services/
│   ├── entropy_agent/
│   ├── ingest_gw/
│   ├── entropy_core/
│   ├── qa_engine/
│   ├── coherence_analyzer/
│   ├── experiment_mgr/
│   └── event_correlator/
├── libs/
│   ├── proto/
│   ├── cuda_kernels/
│   └── common/
├── api/
│   └── gateway/
├── frontend/
│   └── dashboard/
├── deploy/
│   ├── docker/
│   ├── helm/
│   └── k8s/
├── infra/
│   ├── terraform/
│   └── ansible/
├── scripts/
├── docs/
└── tests/
```

---

### **✅ Hand-off Checklist for**

### **Cline**

- Clone repo & generate initial Cargo/Poetry/PNPM workspaces.
- Scaffold protobuf messages under libs/proto.
- Implement Entropy Agent skeleton & unit tests.
- Stand-up local NATS + Timescale via compose.
- Flesh out each service module following *Component Specs* §9.
- Pass unit & integration test suite.
- Produce Docker images tagged 0.1.0.
- Deploy to staging VPS (vps-01).
- Confirm KPIs G-1…G-4 hit.

*This PRD plus implementation notes constitute the complete contract for first delivery.*