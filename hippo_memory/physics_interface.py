"""
Physics System Interface

물리 시스템과의 인터페이스를 정의합니다.

이 엔진이 물리계에 무엇을 출력하는가를 명확히 정의합니다.
모터, 자기부상 베어링, 추진기, 조향 시스템 등 어디든 붙일 수 있는 중립 인터페이스입니다.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PhysicsOutput:
    """
    물리 시스템 출력 (중립 인터페이스)
    
    이 엔진이 물리계에 제공하는 출력값입니다.
    하드웨어는 이 값들을 보고 제어를 수행합니다.
    
    Attributes:
        desired_phase: 목표 위상 (0 ~ size-1, 또는 각도)
        phase_velocity: 위상 속도 (neuron/ms 또는 rad/ms)
        stability_score: 안정성 점수 (0.0 ~ 1.0)
        deviation_error: 편차 오차 (neuron 단위, 목표와의 차이)
        orbit_stability: 궤도 안정성 (0.0 ~ 1.0)
    """
    desired_phase: float          # 목표 위상
    phase_velocity: float         # 위상 속도
    stability_score: float        # 안정성 점수
    deviation_error: float         # 편차 오차
    orbit_stability: float         # 궤도 안정성


def ring_state_to_physics_output(
    state,
    target_phase: Optional[float] = None
) -> PhysicsOutput:
    """
    RingState를 PhysicsOutput으로 변환
    
    Args:
        state: RingState 객체
        target_phase: 목표 위상 (None이면 현재 center 사용)
    
    Returns:
        PhysicsOutput: 물리 시스템 출력
    """
    desired_phase = target_phase if target_phase is not None else state.center
    
    # 위상 속도 계산 (drift를 속도로 변환)
    # drift는 neuron 단위이므로, 이를 시간 단위로 나누면 속도
    # 기본 dt = 0.1ms이므로, drift를 0.1로 나누면 neuron/ms
    phase_velocity = state.drift / 0.1 if state.drift > 0 else 0.0
    
    # 편차 오차 계산 (목표와의 차이)
    deviation_error = abs(state.center - desired_phase)
    
    return PhysicsOutput(
        desired_phase=desired_phase,
        phase_velocity=phase_velocity,
        stability_score=state.stability,
        deviation_error=deviation_error,
        orbit_stability=state.orbit_stability
    )

