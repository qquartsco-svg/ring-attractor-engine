"""
HippoMemory Engine - I/O Contracts (고수준 인터페이스)

입력/출력을 고수준으로 추상화하여 범용 모듈화의 기반을 마련합니다.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# ======================================================================
# 입력 계약 (Input Contracts)
# ======================================================================

@dataclass
class MemoryEvent:
    """
    기억 이벤트 객체 (고수준 입력)
    
    이벤트 타입에 따라 학습/회상/재생으로 분기
    """
    event_type: str        # 'pattern', 'cue', 'replay'
    pattern_id: str        # 패턴 식별자
    timestamp: float       # 이벤트 시간 [ms]
    strength: float        # 이벤트 강도 (0.0 ~ 1.0)
    context: str           # 맥락 식별자
    metadata: Dict[str, Any] = field(default_factory=dict)  # 추가 메타데이터


@dataclass
class MemoryPattern:
    """
    기억 패턴 객체 (고수준 입력)
    
    학습할 패턴 정보를 담은 객체
    """
    pattern_id: str        # 패턴 식별자
    feature_vector: Optional[List[float]] = None  # 특징 벡터 (선택적)
    context: str = "default"  # 맥락 식별자
    importance: float = 0.5  # 중요도 (0.0 ~ 1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)  # 추가 메타데이터


@dataclass
class MemoryContext:
    """
    기억 맥락 객체 (고수준 입력)
    
    맥락 정보 및 연관 패턴 정보
    """
    context_id: str        # 맥락 식별자
    associated_patterns: List[str] = field(default_factory=list)  # 연관된 패턴 리스트
    relevance_map: Dict[str, float] = field(default_factory=dict)  # 패턴별 관련도 맵 (0.0 ~ 1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)  # 추가 메타데이터


# ======================================================================
# 출력 계약 (Output Contracts)
# ======================================================================

@dataclass
class RecallResult:
    """
    회상 결과 객체 (고수준 출력)
    
    recall() 메서드의 고수준 결과
    """
    success: bool              # 회상 성공 여부
    recalled_patterns: List[str] = field(default_factory=list)  # 회상된 패턴 리스트
    relevance_scores: Dict[str, float] = field(default_factory=dict)  # 패턴별 관련도 점수 (0.0 ~ 1.0)
    novelty_score: float = 0.0  # 새로움 점수 (0.0 ~ 1.0, 높을수록 새로운 것)
    recall_confidence: float = 0.0  # 회상 신뢰도 (0.0 ~ 1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)  # 추가 메타데이터


@dataclass
class TrainingResult:
    """
    학습 결과 객체 (고수준 출력)
    
    train() 메서드의 고수준 결과
    """
    pattern_id: str            # 학습한 패턴 식별자
    training_success: bool     # 학습 성공 여부
    memory_strength: float     # 기억 강도 (0.0 ~ 1.0)
    weight_change: float       # 가중치 변화량
    learning_efficiency: float = 0.0  # 학습 효율 (0.0 ~ 1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)  # 추가 메타데이터


@dataclass
class ConsolidationResult:
    """
    정합 결과 객체 (고수준 출력)
    
    sleep() 메서드의 고수준 결과
    """
    consolidation_success: bool    # 정합 성공 여부
    consolidated_patterns: List[str] = field(default_factory=list)  # 정합된 패턴 리스트
    memory_stability: float = 0.0  # 기억 안정도 (0.0 ~ 1.0)
    weight_strengthening: Dict[str, float] = field(default_factory=dict)  # 패턴별 가중치 강화 정도
    metadata: Dict[str, Any] = field(default_factory=dict)  # 추가 메타데이터


@dataclass
class MemoryStatus:
    """
    메모리 상태 객체 (고수준 출력)
    
    get_results() 메서드의 고수준 결과
    """
    total_memories: int        # 총 기억 수
    total_connections: int     # 총 연결 수
    avg_memory_strength: float  # 평균 기억 강도 (0.0 ~ 1.0)
    memory_capacity: float     # 메모리 용량 (0.0 ~ 1.0)
    homeostasis_status: str    # 항상성 상태 ('stable', 'excited', 'depressed')
    metadata: Dict[str, Any] = field(default_factory=dict)  # 추가 메타데이터

