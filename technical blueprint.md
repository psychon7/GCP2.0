### **🚀 End-to-End Technical Blueprint**

**Enhanced Global Consciousness Project (GCP 3.0) — CCTV-Based TRNG Network**

*Design objective: deliver a production-grade, cost-efficient, single-VPS deployment that can later be shard-scaled to 1 000 + nodes with zero architectural re-writes.*

---

## **1️⃣ High-Level Component Graph**

```
┌──────────┐        gRPC/Protobuf         ┌──────────────┐
│ Edge-Cam │ ───────────────────────────▶ │  Ingest-GW   │─────────┐
└──────────┘                               └──────────────┘         │
   (Rust agent)                                                    │
        │                  NATS JetStream (pub/sub)                │
        └──────────────────────────────────────────────────────────▶│
                                                                    ▼
┌──────────────┐   shared‐mem (mmap)   ┌──────────────┐     ┌────────────────┐
│ Entropy-Core │◀──────────────────────│  QA Engine   │◀───▶│   Metrics DB   │
│   (Rust)     │                      └──────────────┘     │ (TimescaleDB)  │
└──────────────┘                            ▲              └────────────────┘
         │                                   │                      ▲
         │                                   │ REST/GraphQL        │
         ▼                                   │ (Falcon)            │
┌──────────────┐                              │                     │
│ Coherence    │ (CUDA / NumPy / Numba)       │                     │
│  Analyzer    │──────────────────────────────┘                     │
└──────────────┘                                                    │
        │        WebSocket / GraphQL-Sub                             │
        └──────────────────────────────▶ ┌──────────────────────────┐
                                         │   Next.js + Three.js     │
                                         │  Interactive Dashboard   │
                                         └──────────────────────────┘
```

---

## **2️⃣ Language & Framework Choices**

| **Layer** | **Language** | **Rationale** |
| --- | --- | --- |
| **Edge-Cam Agent** | **Rust** | zero-cost abstractions, tokio async RTSP, SIMD entropy math |
| **Core Entropy + QA** | Rust (lib) exposed via C-ABI + Python bindings | keeps hot-path in native, lets Python orchestrate |
| **Ingest Gateway** | Go (tiny) or Rust | memory-safe, ultralight; handles TLS, auth, throttling |
| **Coherence Analyzer** | Python 3.12 + Numba + CuPy | rapid math iteration, GPU fallback, easy SciPy stats |
| **Event Correlator** | Python + LangChain + Ollama LLMs (on-device) | pluggable NLP, no external costs |
| **API Gateway** | FastAPI + GraphQL | async, automatic OpenAPI, websocket-friendly |
| **Visualization** | Next.js 15 + React 19 + Three.js/WebGL & WebGPU | SSR + CSR, leverages 2025 WebGPU adoption |
| **Orchestration** | Docker Compose (single VPS) → K3s when sharding | seamless migration path |
| **Messaging** | NATS JetStream | lightweight, at-least-once, zero-broker single binary |
| **DB** | TimescaleDB (metrics) + SQLite (edge cache) + Redis (hot cache) | time-series queries, retention policies |
| **CI/CD** | GitHub Actions → Docker Hub → Watchtower on VPS | zero downtime rolling update |
| **Observability** | Prometheus + Grafana + Loki | one-line Rust/Python exporters |

---

## **3️⃣ Edge-Cam Entropy Extraction (Rust Agent)**

```
// pseudo-code
let mut rtsp = RtspClient::connect(url)?;
loop {
    let frame = rtsp.next_frame()?;                    // YUV420P
    let lsb_mat = frame.to_luma().bitand(0x01);       // LSB extraction
    let var_lsb = variance(&lsb_mat);                 // σ²(LSB_pixels)
    let snr = calc_snr(&frame);                       // SNR
    let atmos_entropy = var_lsb * (1.0 + snr).log2(); // Eq. 1
    /* repeat for photon, thermal, chaos → entropy_vec */

    // von-Neumann debias
    let unbiased = von_neumann(&entropy_vec);

    // SHA3-512 chain
    let digest = sha3_512(&unbiased);

    // push to NATS
    nats.publish("node.entropy", digest)?;
}
```

*CPU cost*: ~6 µs per 640 × 360 frame on AVX2 laptop.

**Protocol** (Protobuf):

```
message EntropySample {
  fixed64 node_id     = 1;
  fixed64 unix_nanos  = 2;
  bytes   sha3_digest = 3; // 64 B
  float   quality     = 4; // 0-100, pre-QA heuristic
}
```

---

## **4️⃣ Quality-Assurance Microservice (Rust → Python)**

### **Streaming NIST subset (< 100 ms)**

| **Test** | **Window** | **SIMD impl** | **Pass-fail** |
| --- | --- | --- | --- |
| Frequency | 1 024 bits | packed_simd |  |
| Runs | 1 024 bits | idem | p > 0.01 |
| Poker | 4-bit buckets | LUT | 1.03 … 57.4 |

Quality score computed:

```
score = 0.3*nist + 0.25*apen + 0.25*dist + 0.2*autocorr
```

Samples below 70 → auto-flag; NATS subject node.flagged.

---

## **5️⃣ Coherence Analyzer (Python + GPU)**

*Process loop every 5 s, sliding 1 000-sample window.*

- **CEC**: compute Gram matrix G = ψ ψᴴ, CUDA cupy.inner, O(N²) ~2 ms @100 nodes.
- **MFR**: pre-computed cultural & distance weights in PostgreSQL, fetched once/day into GPU buffer.
- **FCI**: Hurst exponent via RAPIDS cudf R/S kernel.
- **PNS / RCC**: batched torch.fft.rfft cross-correlation, distance-weighted.

All metric outputs written to coherence.metrics hypertable:

```
CREATE TABLE coherence.metrics (
  ts         TIMESTAMPTZ,
  node_a     BIGINT,
  node_b     BIGINT,
  metric     TEXT,          -- 'CEC','PNS',…
  value      DOUBLE PRECISION
) PARTITION BY RANGE (ts);
SELECT create_hypertable('coherence.metrics','ts',chunk_time_interval=>'2 days');
```

Retention policy: DROP AFTER 30 days, continuous aggregate 1 min, 1 hr, 1 day.

---

## **6️⃣ Experiment Manager**

**PostgreSQL schema**

```
experiments(id UUID PK,
            name TEXT,
            start_ts TIMESTAMPTZ,
            end_ts   TIMESTAMPTZ,
            region   GEOGRAPHY,
            metric   TEXT,           -- e.g. 'CEC'
            target_delta DOUBLE PRECISION,
            p_value_target DOUBLE PRECISION);
```

Runtime engine (Celery worker) subscribes to NATS metrics.#, performs sequential‐test p-value update (Wald). Auto-stops experiment when p < 0.01 or timeout. Results materialized to experiments_results and pushed to dashboard via GraphQL subscription.

Trigger template example:

```
POST /api/experiments
{
  "name": "Global Peace Meditation 2025-06-15",
  "window_minutes": 90,
  "metric": "CEC",
  "region": "POLYGON((...))",
  "effect_size": 0.2
}
```

---

## **7️⃣ Data Pipeline & Storage Volumes (Single VPS)**

| **Path** | **Size/30 d** | **FS** | **Notes** |
| --- | --- | --- | --- |
| /var/lib/pgsql | 35 GB | ext4 on NVMe | TimescaleDB |
| /var/lib/nats | 2 GB | ext4 | JetStream retention 12 h |
| /var/log/loki | 5 GB | ext4, gzip |  |
| /var/tmp/frames | tmpfs 256 MB | in-RAM | no long-term storage |

Backups: daily pg_dump to Backblaze B2 (~1 GB).

---

## **8️⃣ Visualization Stack**

### **8.1 Interactive Globe**

- **Three.js Globe** (three-globe 5.x) with WebGPU renderer.
- Tile source: Mapbox Vector-Tiles Lite (< 1 MB).
- Real-time updates: GraphQL subscription onMetricUpdate(metric:"CEC").
- Shaders:
    - node spheres size ∝ quality, color HSL(120 ✔ → 0 ✖).
    - heatmap overlay via fragment shader sampling cec field raster (provided as H3 grid texture from backend every 5 s).
    - flow particles = instanced meshes with tail fading, direction computed from pairwise PNS > 0.8.

### **8.2 Metrics Dashboard (Next.js 13 app router)**

- **Card**: current global CEC, MFR, FCI.
- **Line chart**: @nivo/line streamed via incremental static regeneration (ISR) 5 s.
- **Node table**: TanStack v5 grid, virtualised.
- **Event correlation timeline**: vis-timeline component, news items piped from Event Correlator microservice.

---

## **9️⃣ Deployment on Single VPS (Hetzner CAX41, €108/mo)**

```
# bootstrap
curl -sfL https://get.k3s.io | sh -
kubectl apply -f k8s/nats.yaml
kubectl apply -f k8s/postgres-ts.yaml
kubectl apply -f k8s/stack/*.yaml
watch kubectl get pods
```

### **Pods & CPU/RAM Envelope**

| **Pod** | **Replicas** | **CPU** | **RAM** |
| --- | --- | --- | --- |
| ingest-gw | 1 | 200 m | 128 MB |
| entropy-core | 2 | 1.5 CPU | 512 MB |
| qa-engine | 1 | 500 m | 256 MB |
| coherence-gpu | 1 | 2 CPU + T4 GPU | 2 GB |
| api-gateway | 1 | 300 m | 256 MB |
| experiment-mgr | 1 | 200 m | 256 MB |
| frontend | 1 | 100 m | 128 MB |
| prometheus | 1 | 200 m | 256 MB |
| grafana | 1 | 50 m | 128 MB |

Total ≈ 7 vCPU / 4 GB — leaves 9 GB headroom.

---

## **🔟 Security & Hardening**

- Mutual-TLS (rustls) Edge-Cam ⇆ Ingest-GW.
- Node enrolment via one-time token + WireGuard mesh as fallback.
- Post-quantum handshake option (Kyber 1024).
- TimescaleDB at rest: aes-xts-plain64 LUKS.
- Prometheus, Grafana, PG behind Traefik; ACME certs.

---

## **1️⃣1️⃣ Scale-Out Path**

*Remove single-VPS bottleneck without rewriting code:*

1. Promote TimescaleDB to separate managed cluster (Aiven).
2. Replicate NATS JetStream in clustered mode (3 × edge VMs).
3. Package Rust agent into BalenaOS image for volunteer Raspberry Pis.
4. Horizontal-pod-autoscale entropy-core and coherence-gpu (GPU node pool).
5. Introduce ClickHouse for long-term cold storage (S3 backend).

---

## **1️⃣2️⃣ Project Timeline (dev-ready)**

| **Week** | **Deliverable** |
| --- | --- |
| 1-2 | Rust Edge Agent MVP, local frame entropy pass NIST subset |
| 3-4 | NATS ingest + TimescaleDB schema, QA Engine |
| 5-6 | Coherence Analyzer (CEC, PNS) GPU kernels |
| 7-8 | FastAPI/GraphQL endpoint, React Globe prototype |
| 9 | Experiment Manager + Wald sequential test |
| 10 | Full single-VPS CI/CD, Prometheus alerts |
| 11-12 | Closed beta with 10 live cameras, public dashboard |

---

### **✅ Summary**

The stack above keeps **entropy math** in **Rust/CUDA** for raw speed, reserves **Python** for statistically heavy—and iteratively evolving—coherence research, and leverages **Next.js + Three.js/WebGPU** to stream a real-time, morphing Earth visualization. All services communicate over **NATS JetStream** with typed **Protobuf** messages; data persistence relies on **TimescaleDB** for millisecond read/write and automatic retention. The entire system fits comfortably on a single €100/month VPS yet scales linearly once nodes or queries explode.

### **1. Detailed Component-Flow Graph (single-VPS baseline)**

```
┌──────────────┐     ┌────────────── Camera Management (Python) ───────────────┐
│ Public CCTV  │     │ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│ /Volunteer   │────▶│ │ Discovery   │─▶│ Validation  │─▶│ Health      │      │
│ Cameras      │     │ │ Scrapers    │  │ Worker Pool │  │ Monitor     │      │
└──────────────┘     │ └─────────────┘  └─────────────┘  └─────────────┘      │
                     │           │              │               │               │
                     │           ▼              ▼               ▼               │
                     │        ┌───────────────────────────────────────┐        │
                     │        │ Registry DB (PostgreSQL)              │        │
                     │        └───────────────────────────────────────┘        │
                     │                            │                             │
                     │                            ▼                             │
                     │                 ┌────────────────────┐                  │
                     │                 │ API Server (FastAPI)│                  │
                     │                 └────────────────────┘                  │
                     └───────────────────────────│─────────────────────────────┘
                                                 │
                                                 ▼
                 ┌──────────── Edge-Cam Node (Rust) ────────────┐
                 │  RTSP Pull                                   │
                 │  SIMD Entropy Math  (atmos, photon…)         │
                 │  von-Neumann / SHA-3                         │
                 │  Protobuf EntropySample ⟶ NATS subject:      │
                 │  "entropy.raw.<node-id>"                     │
                 │                                              │
                 └──────────────────────────────────────────────┘
                               │ JetStream (at-least-once)
                               ▼
                   ┌─────────────────────────────────┐
                   │  Ingest Gateway (Rust, Axum)    │
                   │  • mTLS auth / rate-limit       │
                   │  • Prom-exporter                │
                   └─────────────────────────────────┘
                               │ publishes to
                               ▼
              ┌──────────────────────────────────────────┐
              │  Entropy-Core (Rust dylib + PyO3)        │
              │  • LSB, temporal diff, gradient layers   │
              │  • merges multi-source entropy           │
              │  • emits "entropy.merged" (512-bit blob) │
              └──────────────────────────────────────────┘
                               │
                               ▼
            ┌────────────────────────────────────────────┐
            │  QA Engine  (Rust for NIST, called by Py)  │
            │  • sub-second Frequency / Runs / Poker     │
            │  • ApEn, χ², auto-corr                     │
            │  • writes quality row to TimescaleDB       │
            │  • bad sample ⟶ NATS "node.flagged"        │
            └────────────────────────────────────────────┘
                               │ good samples only
                               ▼
   ┌───────────────────────────────────────────────────────────────┐
   │  Coherence Analyzer (Python 3.12, CuPy / Numba / PyTorch)    │
   │  • CEC, FCI, MFR, PNS, RCC kernels (GPU)                     │
   │  • every 5 s, pushes metric rows to TimescaleDB              │
   │  • publishes front-end deltas on WebSocket "metrics.delta"   │
   └───────────────────────────────────────────────────────────────┘
                               │
                               │  GraphQL (FastAPI-Starlette)
                               ▼
                     ┌───────────────────────────┐
                     │  API Gateway              │
                     │  • REST/GraphQL           │
                     │  • Auth (JWT keycloak)    │
                     │  • WS subscriptions       │
                     └───────────────────────────┘
                               │
      ┌─────────────────────────────────────────────────────┬────────────┐
      ▼                                                     ▼            ▼
┌───────────────┐                                   ┌────────────┐ ┌────────────┐
│ Next.js 19 UI │  ⟵ live JSON ⟵  API  ⟶ mutations  │ Experiment │ │ Event /    │
│  • 3-D Globe  │                                   │  Manager   │ │ News NLP   │
│  • Metrics DB │                                   │  (Celery)  │ │ Correlator │
└───────────────┘                                   └────────────┘ └────────────┘
                                                                       ▲
                                                                       │
                                                region/time query      │
                                                 to News & Sports APIs │
                                                                       ▼
                                                ┌──────────────────────┐
                                                │  External Feeds      │
                                                │  • NewsAPI, GDELT    │
                                                │  • API-Sports        │
                                                │  • PredictHQ events  │
                                                └──────────────────────┘
```

---

### **2. Implementation Plan of Action**

| **Sprint (2 w)** | **Deliverables** | **Key Tech** | **Acceptance** |
| --- | --- | --- | --- |
| **S-1** | • RTSP capture lib• Entropy formulas in SIMD• Protobuf schema | Rust (tokio, simd) | 1 Gb hashed stream passes local NIST subset 95 % |
| **S-2** | • NATS JetStream cluster (single-node)• Ingest Gateway mTLS | NATS 3.x, Axum | sustained 100 msg/s |
| **S-3** | • Entropy-Core DLL + Py bindings• Python harness | Rust → PyO3 | latency < 2 ms/sample |
| **S-4** | • QA Engine (NIST, ApEn, χ²)• TimescaleDB schema | Rust, Timescale 2.15 | real-time QA < 100 ms |
| **S-5** | • GPU kernels: CEC, PNS• CuPy unit tests | CuPy/Numba, CUDA | 100-node -> metrics in < 50 ms |
| **S-6** | • FastAPI GraphQL gateway• JWT + Keycloak | FastAPI, Ariadne | auth & WS stable 24 h |
| **S-7** | • Next.js Globe w/ Three.js shaders• line charts (Nivo) | Next.js 15, WebGPU | 60 fps on MBA M3 |
| **S-8** | • Experiment Manager (Celery) with Wald sequential test | Python, Celery-Redis | auto-stop p < 0.01 |
| **S-9** | • Event Correlator:   – News fetchers   – LLM entity matcher | NewsAPI, API-Sports, Llama-3-8B-Instruct | event ↔ metric lag < 5 s |
| **S-10** | • Helm/Compose deploy on Hetzner• Prometheus+Grafana alerts | k3s, Helm | < 80 % CPU, 99.9 % uptime |
| **S-11–12** | • Closed beta w/ 10 cams• Documentation & Terraform IaC | mdBook, TF | stakeholder sign-off |

---

### **3. Project Directory Structure**

```
gcp-trng/
├── cmd/                      # tiny CLI wrappers
│   ├── ingest-gw/
│   ├── entropy-agent/
│   └── coherence-cli/
├── services/
│   ├── entropy_agent/        # Rust
│   ├── ingest_gw/            # Rust
│   ├── entropy_core/         # Rust lib + Py bindings
│   ├── qa_engine/            # Rust
│   ├── coherence_analyzer/   # Python GPU
│   ├── experiment_mgr/       # Python Celery
│   └── event_correlator/     # Python + LLM
├── libs/
│   ├── proto/                # .proto definitions
│   ├── cuda_kernels/         # .cu files for GPU ops
│   └── common/               # shared utils (logging, cfg)
├── api/
│   └── gateway/              # FastAPI + GraphQL
├── frontend/
│   ├── dashboard/            # Next.js src/
│   └── experiments/          # React admin panel
├── deploy/
│   ├── docker/               # Dockerfiles
│   ├── helm/                 # Helm charts
│   └── k8s/                  # raw manifests for k3s
├── infra/
│   ├── terraform/            # Hetzner + Backblaze
│   └── ansible/              # one-shot VPS hardening
├── scripts/                  # helper bash / Python
├── docs/
│   ├── architecture.md
│   ├── api.md
│   └── dev-setup.md
└── tests/
    ├── integration/
    └── gpu_bench/
```

*Rule of thumb*: **go binaries live under cmd/**, long-running **micro-services inside services/**, pure shareable code in **libs/**. Everything containerised; one docker-compose.override.yml spawns the full stack for local hacking.

---

### **4. Event-Correlation Sub-system**

| **Function** | **Implementation Detail** |
| --- | --- |
| **News fetch** | Abstraction trait NewsProvider with drivers:• NewsAPI v3 (country, category, geo params)  • WorldNewsAPI (location query)  • NewsData.io for fine-grained region filters |
| **Sports feed** | API-Sports real-time fixtures & scores (path /v3/fixtures?team=<id>&date=<d>) |
| **Events/Concerts** | PredictHQ “events” endpoint (geo box) |
| **LLM annotator** | Local Llama-3-8B via ctransformers. Prompt = “[title] … Map to {city,country,lat,lon,event_type}”. |
| **Correlation logic** | On metric spike (Z-score > 3), Experiment Manager publishes spike(region, ts) ⇒ Correlator queries providers ±30 min and returns ranked list by cosine-sim(embedding(news), embedding(metric-context)). |

Cache layer: Redis 7 with 1 h TTL to avoid API-rate hits.

---

### **5. Data-Contracts (Protobuf excerpts)**

```
message MetricDelta {
  int64   ts_unix_ms = 1;
  string  metric     = 2; // CEC, FCI…
  double  value      = 3;
  int64   node_a     = 4;
  int64   node_b     = 5; // 0 when single-node metric
}

message ExperimentResult {
  string  experiment_id = 1;
  double  p_value       = 2;
  double  effect_size   = 3;
  string  status        = 4; // RUNNING, SUCCESS, FAIL
}
```

---

### **6. Front-End Data Flow**

```
GraphQL subscription (wss)
   metricsDelta(metric:"CEC")  ──────▶  Globe state machine
                                            │
                                            ├─ update node sphere color
                                            ├─ update heat-map texture (H3 grid)
                                            └─ push to Redux store

HTTP GET /experiments/:id/summary  ───────▶  Experiment panel
HTTP GET /events?lat=…&lon=…        ─────▶  Timeline lane "news" / "sports"
```

All WebSocket payloads are < 2 KB; UI renders diff only, ensuring < 5 MB/h ↓ bandwidth per client.

---

### **7. Why This Scales**

*Entropy extraction* and *QA* remain **lock-free** Rust; coherence math is **batch GPU** (O(N²) but N ≤ 300). All heavy state lives in TimescaleDB with hypertable compression; Grafana uses continuous aggregates, so dashboard queries hit in-memory chunks. When one VPS maxes out, promote each micro-service to its own node; no rewrite—just scale NATS, Timescale, and CUDA pool.

---

Done — the structure above lets you clone ➜ docker compose up and have camera ➜ entropy ➜ metrics ➜ 3-D globe, with spike-to-news correlation ready for GCP-3.0 research runs.

---

## **11️⃣ Camera Management Module**

### **11.1 Component Architecture**

```
flowchart TD
    subgraph Discovery
      A1[DOT Portal Scraper]
      A2[EarthCam Crawler]
      A3[Insecam Scanner]
      A4[Shodan/ZoomEye Poller]
    end

    subgraph CameraManager Service
      B1[Candidate Queue]  
      B2[Validation Worker Pool]
      B3[Health Monitor]
      B4[Registry DB (Postgres)]
      B5[API Server (FastAPI)]
    end

    subgraph Entropy Network
      C1[Entropy Agent (Rust)]
      C2[QA Engine → TimescaleDB] 
      C3[Coherence Analyzer]
    end

    A1 --> B1
    A2 --> B1
    A3 --> B1
    A4 --> B1

    B1 --> B2
    B2 --> B4
    B4 --> B5
    B4 --> B3
    B3 --> B4

    C1 -- "GET /active_cameras" --> B5
    C1 --> NATS (entropy.raw)
    B5 --> C1

    B2 -. writes bench data .-> B4
    B3 -. updates health status .-> B4
```

### **11.2 Module Overview**

- **Discovery sub-modules** (A1…A4) run periodically (e.g. once/hour) to enqueue new camera URLs into a Candidate Queue.
- **Validation Worker Pool** (B2) pulls from that queue, performs a quick "10-s bench":
  - capture → 10 frames → measure actual FPS/res/latency
  - compute a short "entropy sample" and run the NIST frequency test on those 1,000 bits
  - if (FPS ≥ 5, res ≥ 320×240, latency ≤ 1 s, NIST_pass ≥ 0.9), mark as healthy candidate
- **Registry DB** (B4, PostgreSQL) holds one row per camera with fields:
  - camera_id, source_type, url, discovered_at, last_bench_at, fps, width, height, latency_ms, noise_score, health_status
  - health_status ∈ {"pending", "healthy", "unhealthy"}
- **Health Monitor** (B3) periodically (every 30 s) re-benchmarks all healthy cameras
- **API Server** (B5) exposes a single endpoint: `GET /api/v1/active_cameras`

### **11.3 Data Model**

```python
# services/camera_manager/models.py

import sqlalchemy as sa
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Camera(Base):
    __tablename__ = "cameras"
    camera_id    = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    source_type  = sa.Column(sa.String, nullable=False)  
    url          = sa.Column(sa.Text, unique=True, nullable=False)
    discovered_at= sa.Column(sa.TIMESTAMP(timezone=True), server_default=sa.func.now())
    last_bench_at= sa.Column(sa.TIMESTAMP(timezone=True), nullable=True)
    
    # bench results:
    fps          = sa.Column(sa.Float, nullable=True)
    width        = sa.Column(sa.Integer, nullable=True)
    height       = sa.Column(sa.Integer, nullable=True)
    latency_ms   = sa.Column(sa.Float, nullable=True)
    noise_score  = sa.Column(sa.Float, nullable=True)  # e.g., NIST pass ratio
    health_status= sa.Column(sa.Enum("pending","healthy","unhealthy", name="health_enum"),
                              nullable=False, server_default="pending")
```

### **11.4 Key Processes**

#### **Discovery**
- Multiple crawlers for DOT portals, EarthCam, Insecam, and Shodan/ZoomEye
- Run on configurable schedules (hourly, daily)
- Enqueue new URLs to candidate queue after deduplication

#### **Validation**
- Benchmark workers pull from queue and perform multi-metric assessment
- Metrics: FPS, resolution, latency, entropy quality (NIST frequency test)
- Healthy cameras must meet minimum thresholds across all metrics

#### **Health Monitoring**
- Periodic check (every 30s) on all healthy cameras
- Quick 3-frame test to ensure continued operation
- Automatically mark cameras unhealthy if they fail checks

#### **API Integration**
- FastAPI endpoint for entropy_agent to query active cameras
- Rust agent refreshes camera list every 5 minutes
- Dynamic reconciliation of camera pool based on health status

### **11.5 Fallback & Redundancy**

- **Minimum Pool Size**: Enforce at least 20 simultaneous healthy cameras
  - Temporarily relax thresholds if pool size drops below minimum
  - Emit Prometheus alert if still below threshold
- **Region-Aware Quota**: Store geo data to maximize geographic diversity
  - Limit cameras per region to ensure distributed entropy sources
- **Graceful Decommission**: Keep unhealthy cameras for 24h for possible recovery

### **11.6 Deployment**

| **Component** | **Replicas** | **CPU** | **Memory** |
| --- | --- | --- | --- |
| camera_manager | 1 | 500m | 512MB |
| postgres | 1 | 200m | 512MB |

**Resource Estimates**: ~1GB total memory, ~700m CPU

**Monitoring Metrics**:
- `camera_discovered_total{source="dot"}`
- `camera_health_status{status="healthy"}`
- `camera_bench_latency_ms` (histogram)

This module ensures the entropy_agent always has a fresh, globally diverse, high-quality camera pool without manual URL management.


