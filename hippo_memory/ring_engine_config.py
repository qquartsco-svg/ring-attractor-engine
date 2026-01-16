"""
Ring Engine 설정 파일

하드코딩된 값들을 중앙 집중식으로 관리합니다.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RingEngineConfig:
    """
    Ring Engine 설정
    
    모든 하드코딩된 값들을 이 클래스로 관리합니다.
    """
    # 시간 설정
    dt: float = 0.1  # 시간 간격 [ms]
    
    # 예측 설정
    prediction_horizon_ms: float = 100.0  # 예측 시간 간격 [ms]
    prediction_interval_ms: float = 10.0  # 예측 간격 [ms]
    
    # 보정 설정
    max_correction_force: float = 0.2  # 최대 보정 힘
    disturbance_threshold: float = 0.1  # 외란 임계값
    
    # 방향성 편향 설정
    directional_bias_strength: float = 0.1  # 방향성 편향 강도
    
    # Baseline 전압 설정
    baseline_V: float = -55.0  # 기본 막전위 [mV]
    initial_bias_mV: float = 2.0  # 초기 편향 [mV]
    
    # 안정성 점수 가중치
    stability_active_weight: float = 0.5
    stability_width_weight: float = 0.3
    stability_drift_weight: float = 0.2
    
    # 궤도 안정성 점수 가중치
    orbit_active_weight: float = 0.2
    orbit_width_weight: float = 0.3
    orbit_drift_weight: float = 0.2
    orbit_center_weight: float = 0.1
    
    # Controlled Drift 설정
    drift_step_duration_ms: float = 10.0  # 드리프트 단계 시간 [ms]
    drift_strength_coefficient: float = 0.5  # 드리프트 강도 계수
    max_drift_strength: float = 0.1  # 최대 드리프트 강도
    
    # 예측 신뢰도 설정
    min_history_points: int = 5  # 최소 이력 데이터 포인트
    max_history_points: int = 20  # 최대 이력 데이터 포인트 (신뢰도 1.0 달성)
    confidence_base: float = 0.3  # 신뢰도 기본값
    max_history_size: int = 100  # 최대 이력 데이터 크기
    
    # 속도 계산 설정
    velocity_dt: float = 0.1  # 속도 계산용 dt [ms]
    
    # 외란 복구 설정
    default_disturbance_duration_ms: float = 5.0  # 기본 외란 지속 시간 [ms]
    default_recovery_time_ms: float = 100.0  # 기본 복구 시간 [ms]
    default_disturbance_strength: float = 0.2  # 기본 외란 강도
    
    # 보정 적용 시간
    correction_apply_duration_ms: float = 10.0  # 보정 적용 시간 [ms]
    
    # 위상 차이 임계값
    phase_diff_threshold: float = 0.1  # 위상 차이 임계값


# 기본 설정 인스턴스
DEFAULT_CONFIG = RingEngineConfig()


def get_config(config_name: Optional[str] = None) -> RingEngineConfig:
    """
    설정 가져오기
    
    Args:
        config_name: 설정 이름 (None이면 기본 설정)
    
    Returns:
        RingEngineConfig: 설정 객체
    """
    if config_name is None:
        return DEFAULT_CONFIG
    
    # 향후 다른 설정 추가 가능
    configs = {
        "default": DEFAULT_CONFIG,
        "fast": RingEngineConfig(
            dt=0.05,
            prediction_horizon_ms=50.0,
            prediction_interval_ms=5.0
        ),
        "precise": RingEngineConfig(
            dt=0.05,
            prediction_horizon_ms=200.0,
            prediction_interval_ms=20.0,
            max_correction_force=0.15
        )
    }
    
    return configs.get(config_name, DEFAULT_CONFIG)

