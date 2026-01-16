# Ring Attractor Engine

**Ring Attractor Engine - Persistent State Engine**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/qquartsco-svg/ring-attractor-engine)
[![Status](https://img.shields.io/badge/status-commercial%20ready-green.svg)](https://github.com/qquartsco-svg/ring-attractor-engine)

**Korean**: [README.md](README.md)

---

## 🎯 What It Does

**Ring Attractor Engine** is a minimal unit state memory engine that maintains continuous states stably even after input disappears.

**Core Features**:
- **State Retention**: Maintains state without continuous input
- **Continuous Representation**: Represents continuous states, not discrete values
- **Stability**: Robust against small noise
- **Self-Sustaining Dynamics**: Operates without external input

**This engine is an independent minimal unit component.**

---

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

Or install in development mode:

```bash
pip install -e .
```

### Basic Usage

```python
from hippo_memory.ring_engine import RingAttractorEngine

# Initialize Ring Attractor Engine
engine = RingAttractorEngine(size=15, config="case2")

# Inject phase
engine.inject(direction_idx=5, strength=0.8)
engine.run(duration_ms=2.5)

# Maintain state after input removal
engine.release_input()
state = engine.run(duration_ms=150.0)

print(f"State sustained: {state.sustained}")
print(f"Active neurons: {state.active_count}")
print(f"Bump center: {state.center:.2f}")
```

---

## 📁 Project Structure

```
ring-attractor-engine/
├── hippo_memory/              # Core engine modules
│   ├── ring_engine.py         # Ring Attractor Engine (core)
│   ├── ring_engine_config.py  # Configuration
│   ├── state_types.py         # State type definitions
│   └── application_domains.py # Various domain configurations
├── examples/                  # Executable demo scripts
│   ├── run_ring.py            # Basic state retention demo
│   ├── run_predictive_drift.py # Predictive control demo
│   └── ring_attractor_config.py # Configuration file
├── tests/                     # Test suite
│   └── test_ring_engine.py    # Engine tests
├── docs/                      # Technical documentation
├── README.md                  # This file (Korean)
├── README_EN.md               # English version
├── LICENSE                    # MIT License
├── setup.py                   # Package configuration
├── requirements.txt           # Dependencies (neurons-engine included)
├── BLOCKCHAIN_HASH_RECORD.md  # Blockchain hash record
├── GPG_SIGNING_GUIDE.md       # GPG signing guide
├── REVENUE_SHARING.md         # Code reuse revenue sharing principles
└── CHANGELOG.md               # Changelog
```

---

## 🎯 Key Features

### 1. State Retention
- Maintains phase/orientation without continuous input
- Self-sustaining dynamics (Ring Attractor)
- Drift control and stability

### 2. Predictive Control
- Future phase prediction
- Proactive correction
- Stability improvement

### 3. Various Application Domains
- **Marine**: Propulsion shaft control
- **Vehicle**: Steering angle stabilization
- **Aircraft**: Attitude control
- **Spacecraft**: Satellite attitude control

---

## 📊 Verified Performance

### Core Metrics
- **State Retention**: Maintains for 150ms+ after input removal
- **Stability**: Long-term stability verified
- **Drift Control**: Prediction-based drift control
- **Disturbance Recovery**: State recovery after external disturbance

### Test Results
- **Tests Passed**: Core functionality verified
- **Test Coverage**: Core functionality verified

---

## 🔬 Technical Background

### Ring Attractor Engine
**This is the minimal component engine of this project.**

- **Location**: `hippo_memory/ring_engine.py`
- **Class**: `RingAttractorEngine`
- **Biological Inspiration**: Hippocampal CA3 region
- **Mathematical Model**: Continuous attractor dynamics
- **State Variables**: Phase, velocity, acceleration
- **Topology**: Mexican-hat (excitatory/inhibitory)
- **Features**: Phase memory, self-sustaining dynamics, drift control

**Usage Example**:
```python
from hippo_memory.ring_engine import RingAttractorEngine

# Initialize Ring Attractor Engine
engine = RingAttractorEngine(size=15, config="case2")

# Inject phase
engine.inject(direction_idx=5, strength=0.8)
engine.run(duration_ms=2.5)

# Maintain state after input removal
engine.release_input()
state = engine.run(duration_ms=150.0)
```

---

## 📚 Documentation

### User Guide
- `README.md` (Korean)
- `README_EN.md` (English)

### Technical Documentation
- `docs/` - Detailed technical documentation

### Examples
- `examples/` - Usage example code

---

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Tests
```bash
pytest tests/test_ring_engine.py -v
```

---

## 💰 Revenue Sharing

For code reuse revenue sharing principles, see `REVENUE_SHARING.md`.

---

## 🔐 Blockchain Hash Record

This project uses blockchain hash records to prove:
- Public release: Technology is publicly available (no patents)
- Integrity: Files are unchanged (SHA-256 hashes)
- Precedence: Technical precedence can be proven

**Hash Record**: See `BLOCKCHAIN_HASH_RECORD.md`

---

## 📝 License

**MIT License** - See `LICENSE` file for details

This technology is publicly available (no patents) and can be used as:
- Research/education: Free use
- Commercial use: See `REVENUE_SHARING.md`

---

## 🎯 Application Domains

### 1. Marine
- **Application**: Propulsion shaft control
- **Status**: Demo ready

### 2. Vehicle
- **Application**: Steering angle stabilization
- **Status**: Demo ready

### 3. Aircraft
- **Application**: Attitude control, rotor synchronization
- **Status**: Demo ready

### 4. Spacecraft
- **Application**: Attitude control, reaction wheel control
- **Status**: Demo ready

---

## 🔗 Related Repositories

### Dependencies
- [neurons-engine](https://github.com/qquartsco-svg/neurons-engine) - Neurons Engine (used by this engine)

### Extended Products
- [orbit-stabilizer-sdk](https://github.com/qquartsco-svg/orbit-stabilizer-sdk) - OrbitStabilizer SDK (uses this engine)
- [marine-propulsion-engine](https://github.com/qquartsco-svg/marine-propulsion-engine) - Marine Propulsion Engine (uses SDK)

---

## 📞 Contact

**GitHub Issues**: [Repository Issues](https://github.com/qquartsco-svg/ring-attractor-engine/issues)

---

**Last Updated**: 2026-01-17  
**Version**: v1.0.0  
**Status**: Commercial Ready ✅

