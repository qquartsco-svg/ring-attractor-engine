"""
State Taxonomy for Ring Attractor Engine

이 엔진이 유지할 수 있는 상태의 분류체계를 정의합니다.

핵심 질문: "이 엔진은 '무엇을 유지하는 엔진'인가?"

이 분류를 통해:
- "밥을 먹고 있다"
- "회전 중이다"
- "항로를 유지 중이다"

이 모든 것이 같은 엔진으로 가능함을 증명합니다.
"""

from enum import Enum
from typing import Dict, Optional
from dataclasses import dataclass


class StateType(Enum):
    """
    Ring Attractor가 유지할 수 있는 상태 타입
    
    이 분류는 "기억 엔진"에서 "상태 엔진"으로의 정체성 전환을 의미합니다.
    """
    POSITION = "position"           # 위치 (공간적 위치)
    ORIENTATION = "orientation"     # 방향 (각도, 방위)
    PHASE = "phase"                 # 회전 위상 (0~2π)
    ACTION_ONGOING = "action_ongoing"  # 행동 중 상태 (진행 중인 작업)
    TASK_CONTEXT = "task_context"   # 작업 맥락 (상황 인식)


@dataclass
class StateTypeConfig:
    """
    각 상태 타입별 엔진 설정값
    
    같은 엔진이지만, 상태 타입에 따라 최적화된 파라미터가 다를 수 있습니다.
    """
    state_type: StateType
    description: str
    default_size: int = 15
    default_config: str = "case2"
    
    # 상태 타입별 특성
    requires_drift: bool = False      # Drift가 필요한가?
    requires_stability: bool = True   # 안정성이 중요한가?
    noise_tolerance: float = 0.1     # 노이즈 허용도 (0.0~1.0)
    
    # 물리적 의미
    physical_meaning: str = ""        # 물리적 해석
    example_use_case: str = ""        # 사용 예시


# 상태 타입별 설정 사전 정의
STATE_TYPE_CONFIGS: Dict[StateType, StateTypeConfig] = {
    StateType.POSITION: StateTypeConfig(
        state_type=StateType.POSITION,
        description="공간적 위치를 유지",
        default_size=15,
        default_config="case2",
        requires_drift=False,
        requires_stability=True,
        noise_tolerance=0.1,
        physical_meaning="로봇의 현재 위치, GPS 좌표",
        example_use_case="자율 주행 로봇의 위치 추적"
    ),
    
    StateType.ORIENTATION: StateTypeConfig(
        state_type=StateType.ORIENTATION,
        description="방향/각도를 유지",
        default_size=15,
        default_config="case2",
        requires_drift=False,
        requires_stability=True,
        noise_tolerance=0.15,
        physical_meaning="선박의 방위각, 드론의 자세",
        example_use_case="자율 선박의 항로 유지"
    ),
    
    StateType.PHASE: StateTypeConfig(
        state_type=StateType.PHASE,
        description="회전 위상을 유지",
        default_size=15,
        default_config="case2",
        requires_drift=True,  # 회전은 자연스러운 drift 필요
        requires_stability=True,
        noise_tolerance=0.2,
        physical_meaning="회전체의 위상각, 진동의 위상",
        example_use_case="플라이휠, 반작용 휠의 회전 상태"
    ),
    
    StateType.ACTION_ONGOING: StateTypeConfig(
        state_type=StateType.ACTION_ONGOING,
        description="진행 중인 행동 상태를 유지",
        default_size=15,
        default_config="case2",
        requires_drift=False,
        requires_stability=True,
        noise_tolerance=0.3,  # 행동은 외란에 더 강건해야 함
        physical_meaning="'밥을 먹고 있다', '회전 중이다'",
        example_use_case="작업 중단 후 복원, 치매 방지 시스템"
    ),
    
    StateType.TASK_CONTEXT: StateTypeConfig(
        state_type=StateType.TASK_CONTEXT,
        description="작업 맥락/상황을 유지",
        default_size=15,
        default_config="case2",
        requires_drift=False,
        requires_stability=True,
        noise_tolerance=0.25,
        physical_meaning="현재 작업 상황, 인지적 맥락",
        example_use_case="작업 전환 시 맥락 유지, 멀티태스킹"
    ),
}


def get_state_type_config(state_type: StateType) -> StateTypeConfig:
    """
    상태 타입별 설정 반환
    
    Args:
        state_type: 상태 타입
        
    Returns:
        StateTypeConfig: 해당 타입의 설정
    """
    return STATE_TYPE_CONFIGS.get(state_type, STATE_TYPE_CONFIGS[StateType.PHASE])


def get_state_type_by_name(name: str) -> Optional[StateType]:
    """
    이름으로 상태 타입 찾기
    
    Args:
        name: 상태 타입 이름 (대소문자 무시)
        
    Returns:
        StateType 또는 None
    """
    name_lower = name.lower()
    for state_type in StateType:
        if state_type.value.lower() == name_lower or state_type.name.lower() == name_lower:
            return state_type
    return None

