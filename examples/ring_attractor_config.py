"""
Ring Attractor 실험 설정 (재사용 가능한 파라미터 세트)

사용법:
    from ring_attractor_config import get_case_params
    
    params = get_case_params('case2')
    # params.recurrent_base_weight, params.w_inh_base 등 사용
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class RingAttractorParams:
    """Ring Attractor 형성을 위한 파라미터 세트"""
    # Recurrent connection parameters
    recurrent_base_weight: float  # 기본 흥분 가중치 (E)
    w_inh_base: float  # 기본 억제 가중치 (I)
    sigma: float  # Gaussian 표준편차 (bump 폭 조절)
    r_exc: int  # 흥분 거리 임계값 (거리 <= r_exc: E, 거리 > r_exc: I)
    
    # Neuron parameters
    baseline_V: float  # Baseline 전압 [mV]
    
    # Cue parameters
    cue_duration: float  # Cue 지속 시간 [ms] (end - start)
    cue_start: float = 1.0  # Cue 시작 시간 [ms]
    
    # Directional bias (방향성 편향) - V4.4.3
    directional_bias_enabled: bool = False  # 방향성 편향 활성화 여부
    directional_bias_strength: float = 0.1  # 방향성 편향 강도 (0.0~1.0)
    directional_bias_direction: int = 1  # 방향 (1: 시계방향, -1: 반시계방향)
    
    # Description
    description: str = ""
    
    def get_cue_end(self) -> float:
        """Cue 종료 시간 계산"""
        return self.cue_start + self.cue_duration


# 케이스별 파라미터 정의
CASE_PARAMS: Dict[str, RingAttractorParams] = {
    'case1': RingAttractorParams(
        recurrent_base_weight=0.30,
        w_inh_base=0.18,
        sigma=2.0,
        r_exc=3,
        baseline_V=-54.0,
        cue_duration=2.0,
        description="케이스 1: 보수적 접근 (안정성 우선)"
    ),
    
    'case2': RingAttractorParams(
        recurrent_base_weight=0.28,  # 유지
        w_inh_base=0.68,  # ⚠️ 조정: 0.65 → 0.68 (BAT 10/15 → 5~7 목표)
        sigma=2.0,  # 유지
        r_exc=2,  # 유지 (활성화 수 감소에 효과적)
        # 목표: 평균 활성 ≤ 7, drift ≤ 2.0 neuron
        # 현재: 활성화 평균 6.7/15 (BAT 10/15), Drift: 평균 3.80 (개별 1.27 관측)
        baseline_V=-55.0,
        cue_duration=1.5,
        description="케이스 2: 균형 접근 (권장) + 억제 추가 강화 (w_inh_base 0.68)"
    ),
    
    'case2_refined': RingAttractorParams(
        recurrent_base_weight=0.35,
        w_inh_base=0.22,  # 0.20 → 0.22 (억제 약간 강화)
        sigma=1.8,  # 2.0 → 1.8 (더 좁은 bump)
        r_exc=3,
        baseline_V=-55.0,
        cue_duration=1.5,
        directional_bias_enabled=False,  # 먼저 bump width만 조정
        description="케이스 2 개선: Bump width 조정 (sigma=1.8, w_inh_base=0.22)"
    ),
    
    'case2_tuned': RingAttractorParams(
        recurrent_base_weight=0.40,  # 1️⃣ 0.35 → 0.40 (14% 증가: 이웃 뉴런 임계 도달 보장)
        w_inh_base=0.22,  # 유지
        sigma=1.8,  # 유지
        r_exc=3,
        baseline_V=-55.0,  # 유지 (전역 변경 금지)
        cue_duration=1.5,
        directional_bias_enabled=False,
        description="케이스 2 튜닝: Recurrent Weight 상향 (0.35→0.40) + Temporal Summation 강화 (tau_syn 15→20ms)"
    ),
    
    'case2_directional': RingAttractorParams(
        recurrent_base_weight=0.35,
        w_inh_base=0.22,
        sigma=1.8,
        r_exc=3,
        baseline_V=-55.0,
        cue_duration=1.5,
        directional_bias_enabled=True,  # 방향성 편향 활성화
        directional_bias_strength=0.15,  # 15% 비대칭
        directional_bias_direction=1,  # 시계방향
        description="케이스 2 + 방향성: Bump width 조정 + 방향성 편향 (비대칭 bump)"
    ),
    
    'case3_step1_directional': RingAttractorParams(
        recurrent_base_weight=0.40,  # case2_tuned와 동일
        w_inh_base=0.22,  # 유지
        sigma=1.8,  # 유지
        r_exc=3,  # 유지 (STEP 2에서 변경)
        baseline_V=-55.0,  # 유지
        cue_duration=1.5,
        directional_bias_enabled=True,  # 🥇 STEP 1: 방향성 편향 활성화
        directional_bias_strength=0.30,  # 🥇 STEP 1: 0.15 → 0.30 (2배 강화)
        directional_bias_direction=1,  # 시계방향
        description="STEP 1: 방향성 편향 강화 (0.15→0.30) - Symmetry Breaking"
    ),
    
    'case3_step2_rexc': RingAttractorParams(
        recurrent_base_weight=0.40,  # 유지
        w_inh_base=0.22,  # 유지
        sigma=1.8,  # 유지
        r_exc=2,  # 🥈 STEP 2: 3 → 2 (흥분 범위 축소)
        baseline_V=-55.0,  # 유지
        cue_duration=1.5,
        directional_bias_enabled=True,  # 유지
        directional_bias_strength=0.30,  # 유지
        directional_bias_direction=1,  # 시계방향
        description="STEP 2: r_exc 감소 (3→2) - 더 국소적인 bump 형성"
    ),
    
    'case3_step3_inhibition': RingAttractorParams(
        recurrent_base_weight=0.40,  # 유지
        w_inh_base=0.30,  # 🥉 STEP 3: 0.22 → 0.30 (억제 강화)
        sigma=1.8,  # 유지
        r_exc=2,  # STEP 2 결과 유지
        baseline_V=-55.0,  # 유지
        cue_duration=1.5,
        directional_bias_enabled=True,  # 유지
        directional_bias_strength=0.30,  # 유지
        directional_bias_direction=1,  # 시계방향
        description="STEP 3: 억제 강화 (w_inh_base 0.22→0.30) - 양쪽 끝 억제"
    ),
    
    'case3_step4_combined': RingAttractorParams(
        recurrent_base_weight=0.40,  # 유지
        w_inh_base=0.35,  # 추가 억제 강화 (0.30 → 0.35)
        sigma=1.5,  # 더 좁은 bump (1.8 → 1.5)
        r_exc=3,  # r_exc 복원 (2 → 3, 더 넓은 흥분 범위)
        baseline_V=-55.0,  # 유지
        cue_duration=1.5,
        directional_bias_enabled=True,  # 유지
        directional_bias_strength=0.40,  # 방향성 편향 극대화 (0.30 → 0.40)
        directional_bias_direction=1,  # 시계방향
        description="STEP 4: 조합 접근 (w_inh_base 0.35, sigma 1.5, r_exc 3, bias 0.40)"
    ),
    
    'case3_step5_aggressive': RingAttractorParams(
        recurrent_base_weight=0.45,  # 흥분 약간 증가 (0.40 → 0.45)
        w_inh_base=0.35,  # 강한 억제 유지
        sigma=1.5,  # 좁은 bump 유지
        r_exc=3,  # 유지
        baseline_V=-55.0,  # 유지
        cue_duration=1.5,
        directional_bias_enabled=True,  # 유지
        directional_bias_strength=0.50,  # 방향성 편향 극대화 (0.40 → 0.50)
        directional_bias_direction=1,  # 시계방향
        description="STEP 5: 공격적 접근 (recurrent 0.45, bias 0.50) - Symmetry Breaking 강화"
    ),
    
    'case3_final': RingAttractorParams(
        recurrent_base_weight=0.42,  # 균형 조정
        w_inh_base=0.32,  # 강한 억제
        sigma=1.6,  # 중간 bump 폭
        r_exc=3,  # 유지
        baseline_V=-55.0,  # 유지
        cue_duration=1.5,
        directional_bias_enabled=True,  # 필수
        directional_bias_strength=0.60,  # 극대화 (0.50 → 0.60)
        directional_bias_direction=1,  # 시계방향
        description="최종 시도: 극대화된 방향성 편향 (bias 0.60) + 강한 억제 (w_inh_base 0.32)"
    ),
    
    'case3_optimized': RingAttractorParams(
        recurrent_base_weight=0.38,  # 약간 감소 (12/15 → 5~7 목표)
        w_inh_base=0.38,  # 억제 추가 강화 (0.32 → 0.38)
        sigma=1.5,  # 더 좁은 bump
        r_exc=3,  # 유지
        baseline_V=-55.0,  # 유지
        cue_duration=1.5,
        directional_bias_enabled=True,  # 필수
        directional_bias_strength=0.60,  # 유지 (반대 방향 차단)
        directional_bias_direction=1,  # 시계방향
        description="최적화: 반대 방향 차단 + 억제 강화 (w_inh_base 0.38) - 12/15 → 5~7 목표"
    ),
    
    'case3_balanced': RingAttractorParams(
        recurrent_base_weight=0.30,  # 더 감소 (12/15 → 5~7 목표)
        w_inh_base=0.40,  # 억제 극대화 (0.38 → 0.40)
        sigma=1.4,  # 더 좁은 bump (1.5 → 1.4)
        r_exc=2,  # 흥분 범위 축소 (3 → 2)
        baseline_V=-55.0,  # 유지
        cue_duration=1.5,
        directional_bias_enabled=True,  # 필수
        directional_bias_strength=0.60,  # 유지 (반대 방향 차단)
        directional_bias_direction=1,  # 시계방향
        description="균형: 반대 방향 차단 + 억제 극대화 (w_inh_base 0.40) + r_exc 2 - 5~7 목표"
    ),
    
    'case3_target': RingAttractorParams(
        recurrent_base_weight=0.25,  # 더 감소 (9/15 → 5~7 목표)
        w_inh_base=0.42,  # 억제 추가 강화 (0.40 → 0.42)
        sigma=1.3,  # 더 좁은 bump (1.4 → 1.3)
        r_exc=2,  # 유지
        baseline_V=-55.0,  # 유지
        cue_duration=1.5,
        directional_bias_enabled=True,  # 필수
        directional_bias_strength=0.60,  # 유지 (반대 방향 차단)
        directional_bias_direction=1,  # 시계방향
        description="목표: 9/15 → 5~7 (recurrent 0.25, w_inh_base 0.42, sigma 1.3)"
    ),
    
    'case3_final_attempt': RingAttractorParams(
        recurrent_base_weight=0.28,  # 약간 증가 (너무 줄이면 전파 안 됨)
        w_inh_base=0.45,  # 억제 극대화 (0.42 → 0.45)
        sigma=1.2,  # 더 좁은 bump (1.3 → 1.2)
        r_exc=1,  # 흥분 범위 최소화 (2 → 1)
        baseline_V=-55.0,  # 유지
        cue_duration=1.5,
        directional_bias_enabled=True,  # 필수
        directional_bias_strength=0.60,  # 유지 (반대 방향 차단)
        directional_bias_direction=1,  # 시계방향
        description="최종 시도: r_exc 1 (최소 흥분 범위) + 억제 극대화 (w_inh_base 0.45) - 5~7 목표"
    ),
    
    'case3': RingAttractorParams(
        recurrent_base_weight=0.25,
        w_inh_base=0.25,
        sigma=1.8,
        r_exc=2,
        baseline_V=-56.0,
        cue_duration=1.0,
        description="케이스 3: 공격적 접근 (강한 억제)"
    ),
    
    # 커스텀 케이스 예시
    'custom': RingAttractorParams(
        recurrent_base_weight=0.35,
        w_inh_base=0.20,
        sigma=2.0,
        r_exc=3,
        baseline_V=-55.0,
        cue_duration=1.5,
        description="커스텀 파라미터"
    )
}


def get_case_params(case_name: str = 'case2') -> RingAttractorParams:
    """
    케이스별 파라미터 가져오기
    
    Parameters
    ----------
    case_name : str
        케이스 이름 ('case1', 'case2', 'case3', 'custom')
    
    Returns
    -------
    RingAttractorParams
        파라미터 객체
    
    Raises
    ------
    ValueError
        존재하지 않는 케이스 이름인 경우
    """
    if case_name not in CASE_PARAMS:
        available = ', '.join(CASE_PARAMS.keys())
        raise ValueError(f"Unknown case: '{case_name}'. Available: {available}")
    
    return CASE_PARAMS[case_name]


def list_available_cases() -> Dict[str, str]:
    """사용 가능한 케이스 목록 반환"""
    return {name: params.description for name, params in CASE_PARAMS.items()}


if __name__ == '__main__':
    # 테스트: 사용 가능한 케이스 목록 출력
    print("사용 가능한 케이스:")
    for name, desc in list_available_cases().items():
        params = get_case_params(name)
        print(f"  {name}: {desc}")
        print(f"    - recurrent_base_weight: {params.recurrent_base_weight}")
        print(f"    - w_inh_base: {params.w_inh_base}")
        print(f"    - baseline_V: {params.baseline_V} mV")
        print(f"    - cue_duration: {params.cue_duration} ms")
        print()

