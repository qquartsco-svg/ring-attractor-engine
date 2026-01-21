# Ring Attractor Engine

**링어트랙트 엔진 - 연속 상태 유지 엔진**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/qquartsco-svg/ring-attractor-engine)
[![Status](https://img.shields.io/badge/status-commercial%20ready-green.svg)](https://github.com/qquartsco-svg/ring-attractor-engine)

**English**: [README_EN.md](README_EN.md)

---

## 🎯 무엇을 하는가

**Ring Attractor Engine**은 입력이 사라진 이후에도 연속적인 상태를 안정적으로 유지하는 최소 단위의 상태 메모리 엔진입니다.

**핵심 기능**:
- **상태 유지**: 입력이 없어도 상태를 유지
- **연속 표현**: 이산적 값이 아닌 연속적인 상태 표현
- **안정성**: 작은 노이즈에 강건함
- **자가 지속 동역학**: 외부 입력 없이도 동작 유지

**이 엔진은 독립적으로 사용할 수 있는 최소 단위 부품입니다.**

---

## 🚀 빠른 시작

### 설치

```bash
pip install -r requirements.txt
```

또는 개발 모드로 설치:

```bash
pip install -e .
```

### 기본 사용법

```python
from hippo_memory.ring_engine import RingAttractorEngine

# Ring Attractor Engine 초기화
engine = RingAttractorEngine(size=15, config="case2")

# 위상 주입
engine.inject(direction_idx=5, strength=0.8)
engine.run(duration_ms=2.5)

# 입력 제거 후 상태 유지
engine.release_input()
state = engine.run(duration_ms=150.0)

print(f"상태 유지: {state.sustained}")
print(f"활성 뉴런 수: {state.active_count}")
print(f"범프 중심: {state.center:.2f}")
```

---

## 📁 프로젝트 구조

```
ring-attractor-engine/
├── hippo_memory/              # 핵심 엔진 모듈
│   ├── ring_engine.py         # Ring Attractor Engine (핵심)
│   ├── ring_engine_config.py  # 설정
│   ├── state_types.py         # 상태 타입 정의
│   └── application_domains.py # 다양한 도메인 설정
├── examples/                  # 실행 가능한 데모 스크립트
│   ├── run_ring.py            # 기본 상태 유지 데모
│   ├── run_predictive_drift.py # 예측 제어 데모
│   └── ring_attractor_config.py # 설정 파일
├── tests/                     # 테스트 스위트
│   └── test_ring_engine.py    # 엔진 테스트
├── docs/                      # 기술 문서
├── README.md                  # 이 파일 (한국어 - 메인)
├── README_EN.md               # 영어 버전
├── LICENSE                    # MIT 라이선스
├── setup.py                   # 패키지 설정
├── requirements.txt           # 의존성 (neurons-engine 포함)
├── BLOCKCHAIN_HASH_RECORD.md  # 블록체인 해시 기록
├── GPG_SIGNING_GUIDE.md       # GPG 서명 가이드
├── REVENUE_SHARING.md         # 코드 재사용 수익 분배 원칙
└── CHANGELOG.md               # 변경 이력
```

---

## 🎯 주요 기능

### 1. 상태 유지 (State Retention)
- 연속 입력 없이 위상/방향 유지
- 자가 지속 동역학 (Ring Attractor)
- 드리프트 제어 및 안정성

### 2. 예측 제어 (Predictive Control)
- 미래 위상 예측
- 선제적 보정
- 안정성 향상

### 3. 다양한 응용 도메인
- **선박**: 추진축 제어
- **차량**: 조향각 안정화
- **항공**: 자세 제어
- **우주선**: 위성 자세 제어

---

## 📊 검증된 성능

### 핵심 지표
- **상태 유지**: 입력 제거 후 150ms 이상 유지
- **안정성**: 장기간 안정성 검증 완료
- **드리프트 제어**: 예측 기반 드리프트 제어
- **외란 복구**: 외부 외란 후 상태 복구

### 테스트 결과
- **테스트 통과**: 핵심 기능 검증 완료
- **테스트 커버리지**: 핵심 기능 검증 완료

---

## 🔬 기술 배경

### Ring Attractor Engine
**이것이 이 프로젝트의 최소 부품 엔진입니다.**

- **위치**: `hippo_memory/ring_engine.py`
- **클래스**: `RingAttractorEngine`
- **생물학적 영감**: 해마 CA3 영역
- **수학적 모델**: 연속 어트랙터 동역학
- **상태 변수**: 위상, 속도, 가속도
- **토폴로지**: Mexican-hat (흥분/억제)
- **기능**: 위상 기억, 자가 지속 동역학, 드리프트 제어

**사용 예시**:
```python
from hippo_memory.ring_engine import RingAttractorEngine

# Ring Attractor Engine 초기화
engine = RingAttractorEngine(size=15, config="case2")

# 위상 주입
engine.inject(direction_idx=5, strength=0.8)
engine.run(duration_ms=2.5)

# 입력 제거 후 상태 유지
engine.release_input()
state = engine.run(duration_ms=150.0)
```

---

## 📚 문서

### 사용 가이드
- `README.md` (한국어 - 메인)
- `README_EN.md` (영어)

### 기술 문서
- `docs/` - 상세 기술 문서

### 예제
- `examples/` - 사용 예제 코드

---

## 🧪 테스트

### 모든 테스트 실행
```bash
pytest tests/ -v
```

### 특정 테스트 실행
```bash
pytest tests/test_ring_engine.py -v
```

---

## 💰 코드 재사용 수익 분배

코드 재사용으로 수익이 발생할 경우 분배 원칙은 `REVENUE_SHARING.md`를 참조하세요.

---

## 🔐 블록체인 해시 기록

이 프로젝트는 블록체인 해시 기록을 사용하여:
- 공개 발매 증명
- 파일 무결성 보장
- 기술적 선행 기술 증명

**해시 기록**: `BLOCKCHAIN_HASH_RECORD.md` 참조

---

## 📝 라이선스

**MIT 라이선스** - 자세한 내용은 `LICENSE` 파일 참조

이 기술은 공개적으로 사용 가능하며 (특허 없음) 다음과 같이 사용할 수 있습니다:
- 연구/교육용 자유 사용
- 상업적 사용시 `REVENUE_SHARING.md` 참조

---

## 🎯 응용 도메인

### 1. 선박
- **응용**: 추진축 제어
- **상태**: 데모 준비 완료

### 2. 차량
- **응용**: 조향각 안정화
- **상태**: 데모 준비 완료

### 3. 항공
- **응용**: 자세 제어, 로터 동기화
- **상태**: 데모 준비 완료

### 4. 우주선
- **응용**: 자세 제어, 반작용 휠 제어
- **상태**: 데모 준비 완료

---

## 🔗 관련 레포지토리

### 의존성
- [neurons-engine](https://github.com/qquartsco-svg/neurons-engine) - 뉴런 엔진 (이 엔진이 사용)

### 확장 제품
- [orbit-stabilizer-sdk](https://github.com/qquartsco-svg/orbit-stabilizer-sdk) - OrbitStabilizer SDK (이 엔진 사용)
- [marine-propulsion-engine](https://github.com/qquartsco-svg/marine-propulsion-engine) - 선박 추진축 엔진 (SDK 사용)

---

## 📞 문의

**GitHub Issues**: [레포지토리 Issues](https://github.com/qquartsco-svg/ring-attractor-engine/issues)

---

**Last Updated**: 2026-01-17  
**Version**: v1.0.0  
**Status**: 상용화 준비 완료 ✅

---

## 🧠 Memory Layers (Concept Alignment)

This project uses the same **memory-layer taxonomy** as the Brain Atlas:

- **L0** Ring Attractor → **Neural Intrinsic Memory** (local attractor dynamics)
- **L1** Grid Engine → **Spatial State Representation**
- **L2** Hippo Memory → **Contextual / Place Memory**
- **L3** Cerebellum → **Motor Pattern Optimizer**

See: `~/Desktop/Brain_Atlas/CONCEPTS_MEMORY_LAYERS.md`
