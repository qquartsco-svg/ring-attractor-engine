"""
Wear Minimization Engine

마모 최소화 엔진으로의 재정의.

핵심 개념:
❌ 마모를 없앤다
⭕ 마모가 최소가 되는 회전 반경을 유지한다

즉:
- 힘 제어 ❌
- 토크 극대화 ❌
- 궤도 유지 ⭕
- 편차 복원 ⭕

이 순간부터 이 엔진은:
- "동력 엔진"이 아니라
- "회전 상태 보존 엔진"
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class OrbitControl:
    """
    궤도 제어 출력
    
    마모 최소화를 위한 궤도 유지 및 편차 복원 명령입니다.
    
    Attributes:
        maintain_orbit: 궤도 유지 명령 (True = 현재 궤도 유지)
        restore_deviation: 편차 복원 명령 (True = 편차 복원 필요)
        deviation_magnitude: 편차 크기 (neuron 단위)
        correction_direction: 보정 방향 (0 ~ size-1, 또는 None)
        wear_risk: 마모 위험도 (0.0 ~ 1.0, 높을수록 마모 위험)
    """
    maintain_orbit: bool          # 궤도 유지 명령
    restore_deviation: bool        # 편차 복원 명령
    deviation_magnitude: float     # 편차 크기
    correction_direction: Optional[int]  # 보정 방향
    wear_risk: float               # 마모 위험도


def calculate_wear_risk(
    deviation_error: float,
    orbit_stability: float,
    drift: float,
    max_deviation: float = 2.0
) -> float:
    """
    마모 위험도 계산
    
    편차가 크고 안정성이 낮을수록 마모 위험이 높습니다.
    
    Args:
        deviation_error: 편차 오차
        orbit_stability: 궤도 안정성
        drift: 드리프트 거리
        max_deviation: 최대 허용 편차
    
    Returns:
        Wear risk (0.0 ~ 1.0)
    """
    # 편차 위험도 (편차가 클수록 위험)
    deviation_risk = min(1.0, deviation_error / max_deviation)
    
    # 안정성 위험도 (안정성이 낮을수록 위험)
    stability_risk = 1.0 - orbit_stability
    
    # 드리프트 위험도 (드리프트가 클수록 위험)
    drift_risk = min(1.0, drift / 2.0)
    
    # 종합 위험도 (가중 평균)
    wear_risk = (
        deviation_risk * 0.5 +      # 편차가 가장 중요
        stability_risk * 0.3 +      # 안정성
        drift_risk * 0.2             # 드리프트
    )
    
    return min(1.0, wear_risk)


def calculate_orbit_control(
    current_phase: float,
    target_phase: float,
    deviation_error: float,
    orbit_stability: float,
    drift: float,
    size: int,
    deviation_threshold: float = 1.0
) -> OrbitControl:
    """
    궤도 제어 명령 계산
    
    마모 최소화를 위한 궤도 유지 및 편차 복원 명령을 생성합니다.
    
    Args:
        current_phase: 현재 위상
        target_phase: 목표 위상
        deviation_error: 편차 오차
        orbit_stability: 궤도 안정성
        drift: 드리프트 거리
        size: Ring 크기
        deviation_threshold: 편차 임계값
    
    Returns:
        OrbitControl: 궤도 제어 명령
    """
    # 편차 복원 필요 여부
    restore_deviation = deviation_error > deviation_threshold
    
    # 보정 방향 계산
    correction_direction = None
    if restore_deviation:
        # 목표 방향으로 보정
        if target_phase > current_phase:
            if target_phase - current_phase < size / 2:
                correction_direction = int((current_phase + 1) % size)
            else:
                correction_direction = int((current_phase - 1) % size)
        else:
            if current_phase - target_phase < size / 2:
                correction_direction = int((current_phase - 1) % size)
            else:
                correction_direction = int((current_phase + 1) % size)
    
    # 마모 위험도 계산
    wear_risk = calculate_wear_risk(deviation_error, orbit_stability, drift)
    
    return OrbitControl(
        maintain_orbit=True,  # 항상 궤도 유지
        restore_deviation=restore_deviation,
        deviation_magnitude=deviation_error,
        correction_direction=correction_direction,
        wear_risk=wear_risk
    )

