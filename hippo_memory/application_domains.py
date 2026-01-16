"""
Application Domains

적용 도메인 정의 (STEP 6)

가장 현실적인 순서:
1. 자율 선박 (항로·방향 유지)
2. 우주선 자세 제어
3. 정밀 회전체 (플라이휠, 반작용 휠)
4. 생체/인지 모델

자동차는 지금 단계에선 아님
(마찰·열·법규가 너무 많음)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional
from .state_types import StateType


class ApplicationDomain(Enum):
    """
    적용 도메인
    
    이 엔진이 사용될 수 있는 실제 응용 분야입니다.
    
    확장 전략:
    1. 기본용 (BASIC) - 범용 설정
    2. 차량용 (VEHICLE) - 자율주행 차량
    3. 선박용 (AUTONOMOUS_VESSEL) - 자율 선박
    4. 비행용 (AIRCRAFT) - 항공기/드론
    5. 우주선용 (SPACECRAFT_ATTITUDE) - 우주선 자세 제어
    """
    BASIC = "basic"                              # 기본용 (범용)
    VEHICLE = "vehicle"                          # 차량용 (자율주행)
    AUTONOMOUS_VESSEL = "autonomous_vessel"      # 자율 선박
    AIRCRAFT = "aircraft"                        # 비행용 (항공기/드론)
    SPACECRAFT_ATTITUDE = "spacecraft_attitude"  # 우주선 자세 제어
    PRECISION_ROTOR = "precision_rotor"          # 정밀 회전체
    BIOLOGICAL_MODEL = "biological_model"         # 생체/인지 모델


@dataclass
class DomainConfig:
    """
    도메인별 설정
    
    각 도메인에 최적화된 엔진 설정입니다.
    """
    domain: ApplicationDomain
    description: str
    state_type: StateType
    size: int = 15
    config: str = "case2"
    
    # 도메인별 특성
    requires_precision: bool = True      # 정밀도 요구
    requires_stability: bool = True      # 안정성 요구
    allows_drift: bool = False           # 드리프트 허용
    noise_tolerance: float = 0.1         # 노이즈 허용도
    
    # 물리적 특성
    physical_unit: str = ""              # 물리 단위
    typical_range: str = ""              # 일반적인 범위
    example_application: str = ""        # 사용 예시


# 도메인별 설정 사전 정의
DOMAIN_CONFIGS: Dict[ApplicationDomain, DomainConfig] = {
    ApplicationDomain.BASIC: DomainConfig(
        domain=ApplicationDomain.BASIC,
        description="기본용 (범용 설정)",
        state_type=StateType.POSITION,
        size=15,
        config="case2",
        requires_precision=True,
        requires_stability=True,
        allows_drift=True,
        noise_tolerance=0.1,
        physical_unit="arbitrary",
        typical_range="0-360 degrees",
        example_application="연구 및 개발, 개념 증명, 범용 회전 제어"
    ),
    
    ApplicationDomain.VEHICLE: DomainConfig(
        domain=ApplicationDomain.VEHICLE,
        description="차량용 (자율주행)",
        state_type=StateType.ORIENTATION,
        size=15,
        config="case2",
        requires_precision=True,
        requires_stability=True,
        allows_drift=False,
        noise_tolerance=0.05,
        physical_unit="degrees",
        typical_range="0-360 degrees",
        example_application="자율주행 차량 조향각 유지, 바퀴 회전 제어, 차선 유지"
    ),
    
    ApplicationDomain.AUTONOMOUS_VESSEL: DomainConfig(
        domain=ApplicationDomain.AUTONOMOUS_VESSEL,
        description="자율 선박 항로·방향 유지",
        state_type=StateType.ORIENTATION,
        size=15,
        config="case2",
        requires_precision=True,
        requires_stability=True,
        allows_drift=False,
        noise_tolerance=0.15,
        physical_unit="degrees",
        typical_range="0-360 degrees",
        example_application="자율 선박의 항로 유지, GPS 신호 끊김 시 방향 기억"
    ),
    
    ApplicationDomain.AIRCRAFT: DomainConfig(
        domain=ApplicationDomain.AIRCRAFT,
        description="비행용 (항공기/드론)",
        state_type=StateType.ORIENTATION,
        size=15,
        config="case2",
        requires_precision=True,
        requires_stability=True,
        allows_drift=False,
        noise_tolerance=0.02,
        physical_unit="degrees",
        typical_range="0-360 degrees",
        example_application="드론 자세 제어, 항공기 터빈 회전 제어, 프로펠러 동기화"
    ),
    
    ApplicationDomain.SPACECRAFT_ATTITUDE: DomainConfig(
        domain=ApplicationDomain.SPACECRAFT_ATTITUDE,
        description="우주선 자세 제어",
        state_type=StateType.ORIENTATION,
        size=15,
        config="case2",
        requires_precision=True,
        requires_stability=True,
        allows_drift=True,  # 우주선은 천천히 회전 가능
        noise_tolerance=0.1,
        physical_unit="radians",
        typical_range="0-2π radians",
        example_application="우주선 자세 유지, 반작용 휠 제어"
    ),
    
    ApplicationDomain.PRECISION_ROTOR: DomainConfig(
        domain=ApplicationDomain.PRECISION_ROTOR,
        description="정밀 회전체 (플라이휠, 반작용 휠)",
        state_type=StateType.PHASE,
        size=15,
        config="case2",
        requires_precision=True,
        requires_stability=True,
        allows_drift=True,  # 회전은 자연스러운 drift
        noise_tolerance=0.2,
        physical_unit="radians",
        typical_range="0-2π radians",
        example_application="플라이휠 회전 상태 유지, 반작용 휠 제어"
    ),
    
    ApplicationDomain.BIOLOGICAL_MODEL: DomainConfig(
        domain=ApplicationDomain.BIOLOGICAL_MODEL,
        description="생체/인지 모델",
        state_type=StateType.ACTION_ONGOING,
        size=15,
        config="case2",
        requires_precision=False,
        requires_stability=True,
        allows_drift=False,
        noise_tolerance=0.3,
        physical_unit="arbitrary",
        typical_range="0-1 normalized",
        example_application="작업 중 상태 유지, 치매 방지 시스템"
    ),
}


def get_domain_config(domain: ApplicationDomain) -> DomainConfig:
    """
    도메인별 설정 반환
    
    Args:
        domain: 적용 도메인
        
    Returns:
        DomainConfig: 해당 도메인의 설정
    """
    return DOMAIN_CONFIGS.get(domain, DOMAIN_CONFIGS[ApplicationDomain.BASIC])


def get_domain_by_name(name: str) -> Optional[ApplicationDomain]:
    """
    이름으로 도메인 찾기
    
    Args:
        name: 도메인 이름 (대소문자 무시)
        
    Returns:
        ApplicationDomain 또는 None
    """
    name_lower = name.lower()
    for domain in ApplicationDomain:
        if domain.value.lower() == name_lower or domain.name.lower() == name_lower:
            return domain
    return None

