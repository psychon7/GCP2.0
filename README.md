<div align="center">
  <h1>🌍 Enhanced Global Consciousness Project (GCP 2.0)</h1>
  <p><strong>Fostering global awareness and collective intelligence through technology</strong></p>
  
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Contributors Welcome](https://img.shields.io/badge/contributors-welcome-brightgreen.svg)](CONTRIBUTING.md)
  [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
  [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
  [![Node.js 16+](https://img.shields.io/badge/node.js-16+-green.svg)](https://nodejs.org/)
  [![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
  
  <br>
  
  <img src="https://img.shields.io/github/stars/psychon7/GCP2.0?style=social" alt="GitHub stars">
  <img src="https://img.shields.io/github/forks/psychon7/GCP2.0?style=social" alt="GitHub forks">
  <img src="https://img.shields.io/github/watchers/psychon7/GCP2.0?style=social" alt="GitHub watchers">
  
</div>

## 🌍 Vision

The Enhanced Global Consciousness Project (GCP 2.0) is an ambitious open-source initiative aimed at fostering global awareness, interconnectedness, and collective intelligence through technology. Our mission is to create tools and platforms that help humanity understand our shared challenges and work together toward sustainable solutions.

## 📖 Overview

This project builds upon the concept of global consciousness - the idea that human awareness can be measured and enhanced through collective participation and technological innovation. GCP 2.0 represents the next evolution of consciousness research, combining cutting-edge technology with human-centered design to create meaningful impact.

### Key Objectives

- 🧠 **Consciousness Measurement**: Generate cryptographically-secure TRNG stream (≥ 256 b/node/sec)
- 🌐 **Global Connectivity**: Build a vertically-integrated, 100-camera global network with real-time coherence analytics
- 📊 **Data-Driven Insights**: Compute multi-metric coherence analytics (CEC, PNS, FCI, MFR, RCC) with 5s refresh
- 🤝 **Collaborative Solutions**: Correlate spikes with geo-located news/sports/events in under 30s
- 🔬 **Research Advancement**: Provide one-click experiment setup with rolling p-value & auto-stop

## 🏗️ Technical Architecture

The technical architecture is designed for a production-grade, cost-efficient, single-VPS deployment that can later be shard-scaled to 1,000+ nodes with zero architectural re-writes. It focuses on a CCTV-Based True Random Number Generator (TRNG) Network with end-to-end latency under 100ms and monthly operating costs under €120.

For a comprehensive understanding, please refer to the [Detailed Technical Blueprint](technical%20blueprint.md).

### High-Level Component Graph

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

### Language & Framework Choices

| **Layer**          | **Language**                                 | **Rationale**                                              |
|--------------------|----------------------------------------------|------------------------------------------------------------|
| **Edge-Cam Agent** | **Rust 1.78**                                | zero-cost abstractions, tokio async RTSP, SIMD entropy math|
| **Core Entropy + QA**| Rust (lib) exposed via C-ABI + Python bindings | keeps hot-path in native, lets Python orchestrate          |
| **Ingest Gateway** | Go or Rust                                   | memory-safe, ultralight; handles TLS, auth, throttling     |
| **Coherence Analyzer**| Python 3.12 + Numba 0.60 + CuPy 13            | rapid math iteration, GPU fallback, easy SciPy stats       |
| **Event Correlator**| Python + LangChain + Ollama (Llama-3-8B)     | pluggable NLP, no external costs                           |
| **API Gateway**    | FastAPI 0.111 + Ariadne GraphQL              | async, automatic OpenAPI, websocket-friendly               |
| **Visualization**  | Next.js 15 + React 19 + Three.js 0.163 + WebGPU | SSR + CSR, leverages 2025 WebGPU adoption                |
| **Orchestration**  | Docker Compose (single VPS) → K3s when sharding| seamless migration path                                    |
| **Messaging**      | NATS JetStream 3.x                           | lightweight, at-least-once, zero-broker single binary      |
| **DB**             | TimescaleDB 2.15 (PostgreSQL 16) + SQLite + Redis 7 | time-series queries, retention policies                    |
| **CI/CD**          | GitHub Actions → Docker Hub → Watchtower on VPS| zero downtime rolling update                               |
| **Observability**  | Prometheus + Grafana + Loki                  | one-line Rust/Python exporters                             |

### Key Architectural Highlights

*   **Entropy Extraction**: Edge cameras (Rust agents) perform entropy math (SIMD, LSB extraction) and push data via NATS.
*   **Quality Assurance**: A Rust/Python microservice performs NIST-subset tests and other quality checks.
*   **Coherence Analysis**: Python with Numba/CuPy for GPU-accelerated computation of coherence metrics (CEC, MFR, FCI, PNS, RCC).
*   **Data Pipeline**: NATS JetStream for messaging, TimescaleDB for metrics, SQLite for edge cache, and Redis for hot caching.
*   **Deployment**: Initially on a single VPS (e.g., Hetzner CAX41) using Docker Compose, with a clear path to K3s for sharding.
*   **Visualization**: Next.js with Three.js/WebGL (and future WebGPU) for an interactive 3D globe and metrics dashboard.
*   **Security**: Includes mTLS, node enrolment tokens, WireGuard, and data-at-rest encryption.
*   **Scalability**: Designed to scale out by promoting services to dedicated clusters/nodes (e.g., managed TimescaleDB, clustered NATS, GPU node pools).

For more details on specific components like the Edge-Cam Agent, QA Engine, Coherence Analyzer, Experiment Manager, detailed data flow, deployment specifics, and the full scale-out path, please consult the [Technical Blueprint](technical%20blueprint.md).

## 🚀 Getting Started

### Prerequisites

- Python 3.12 or higher
- Node.js 16 or higher
- Rust 1.78 or higher
- Docker and Docker Compose
- Git
- CUDA 12 (for GPU acceleration)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/psychon7/GCP2.0.git
   cd GCP2.0
   ```

2. **Set up the backend**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up the frontend**
   ```bash
   cd frontend
   npm install
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

6. **Access the application**
   - 🌐 **Frontend**: http://localhost:3000
   - 🔧 **Backend API**: http://localhost:8000
   - 📚 **API Documentation**: http://localhost:8000/docs
   - 📊 **Admin Panel**: http://localhost:8000/admin
   - 📈 **Monitoring**: http://localhost:9090 (Prometheus)

### 📊 Usage Examples

#### Basic API Usage

```python
import requests

# Get coherence metrics data
response = requests.get('http://localhost:8000/api/v1/metrics/coherence/global')
data = response.json()

print(f"Current CEC level: {data['cec']}")
print(f"Field Coherence Index: {data['fci']}")
print(f"Active nodes: {data['active_nodes']}")
```

#### Real-time Data Streaming

```javascript
// GraphQL subscription for real-time updates
const client = new GraphQLClient('http://localhost:8000/graphql');

const subscription = client.subscribe({
  query: `subscription {
    onMetricUpdate(metric: "CEC") {
      timestamp
      value
      nodes {
        id
        location
        quality
      }
    }
  }`
});

subscription.subscribe(({ data }) => {
  console.log('Real-time coherence data:', data.onMetricUpdate);
  updateGlobeVisualization(data.onMetricUpdate);
});
```

#### Data Visualization

```python
from gcp2 import ConsciousnessAnalyzer

# Initialize analyzer
analyzer = ConsciousnessAnalyzer()

# Generate consciousness map
map_data = analyzer.generate_global_map()
analyzer.visualize(map_data, output='consciousness_map.html')
```

## ✨ Features

<table>
<tr>
<td width="50%">

### 🚀 Current Features
- ✅ **Entropy Extraction**: Edge cameras extract entropy with Rust agents
- ✅ **Quality Assurance**: NIST subset tests and other quality checks
- ✅ **Coherence Analysis**: GPU-accelerated computation of coherence metrics (CEC, MFR, FCI, PNS, RCC)
- ✅ **Interactive Globe**: Three.js/WebGL/WebGPU with real-time node visualization
- ✅ **Event Correlation**: Correlating entropy spikes with global events
- ✅ **GraphQL API**: Comprehensive API with subscription support
- ✅ **Docker Support**: Containerized single-VPS deployment
- ✅ **Experiment Console**: One-click setup for global experiments

</td>
<td width="50%">

### 🔮 Planned Features
- 🔄 **Horizontal Scaling**: K3s deployment for 1,000+ node support
- 📱 **Mobile Applications**: iOS and Android native apps for experiment monitoring
- 🌐 **Additional Entropy Sources**: Expanding beyond CCTV to other sources
- ⛓️ **Blockchain Verification**: Decentralized data integrity for research results
- 🌍 **Multi-language Support**: Global dashboard accessibility
- 🔊 **Advanced Event Detection**: Enhanced correlation with global events
- 📊 **AR/VR Interfaces**: Immersive consciousness experiences
- 🤖 **On-device LLMs**: Enhanced Llama model integration

</td>
</tr>
</table>


## 🤝 Contributing

We welcome contributions from developers, researchers, designers, and consciousness enthusiasts worldwide! Please read our [Contributing Guidelines](CONTRIBUTING.md) before getting started.

### 🎯 Ways to Contribute

<table>
<tr>
<td width="33%">

#### 💻 **Development**
- 🐛 Bug fixes and issue resolution
- ✨ New feature implementation
- ⚡ Performance optimizations
- 🔧 Code refactoring and cleanup
- 🧪 Test coverage improvements

</td>
<td width="33%">

#### 📚 **Documentation**
- 📖 API documentation
- 🎓 Tutorial creation
- 🌍 Translation and localization
- 📝 Blog posts and articles
- 🎥 Video tutorials

</td>
<td width="33%">

#### 🔬 **Research**
- 📊 Data analysis and insights
- 🧠 Consciousness research
- 🤖 AI/ML model development
- 📈 Statistical analysis
- 📋 Research paper reviews

</td>
</tr>
</table>


### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Ensure all tests pass (`npm test` and `pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### 📋 Code Standards

#### Python Guidelines
```python
# Follow PEP 8 style guide
# Use type hints
def calculate_consciousness_level(data: List[Dict]) -> float:
    """Calculate global consciousness level from participant data.
    
    Args:
        data: List of participant consciousness measurements
        
    Returns:
        Normalized consciousness level (0.0 to 1.0)
    """
    pass
```

#### JavaScript Guidelines
```javascript
// Use ESLint + Prettier
// Prefer const/let over var
const calculateConsciousnessLevel = (data) => {
  // Use meaningful variable names
  const normalizedData = data.map(item => normalizeValue(item));
  return normalizedData.reduce((sum, value) => sum + value, 0) / data.length;
};
```

#### Commit Message Format
```
type(scope): description

Types: feat, fix, docs, style, refactor, test, chore
Example: feat(api): add consciousness prediction endpoint
```

### 📜 Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) to understand our community standards.

**Core Values**: Respect • Inclusivity • Collaboration • Scientific Integrity • Open Innovation

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original Global Consciousness Project researchers
- Open source community contributors
- Scientific advisors and research partners
- All participants in consciousness research

---

<div align="center">
  <strong>Together, we can enhance global consciousness and create a better world for all.</strong>
  <br><br>
  <a href="#enhanced-global-consciousness-project-gcp-20">Back to Top</a>
</div>