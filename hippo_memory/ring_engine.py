"""
Ring Attractor Engine

Persistent State Engine: 입력이 사라진 이후에도 연속적인 상태를 안정적으로 유지하는 최소 단위의 상태 메모리 엔진.

핵심 기능:
- 상태 유지: 입력이 없어도 상태를 유지
- 연속 표현: 이산적 값이 아닌 연속적인 상태 표현
- 안정성: 작은 노이즈에 강건함

이 엔진은 독립적으로 사용할 수 있는 최소 단위 부품입니다.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import sys
import os

# 프로젝트 루트 경로 추가
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

# examples 디렉토리도 경로에 추가 (ring_attractor_config.py를 찾기 위해)
examples_dir = os.path.join(base_dir, 'examples')
sys.path.insert(0, examples_dir)

# 내부 모듈 (같은 패키지 내)
from .engine import HippoMemoryV4System

# 설정 파일 (examples 디렉토리에서)
try:
    import sys
    import os
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples')
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    from ring_attractor_config import get_case_params
except ImportError:
    # fallback: 직접 파라미터 정의
    get_case_params = None


@dataclass
class RingState:
    """
    Ring Attractor 상태 (제품 계약)
    
    모든 필드는 항상 유효한 값을 가집니다.
    
    Attributes:
        center: 중심 위치 (0 ~ size-1), 활성화 없으면 0.0
        width: 범프 너비 (standard deviation), 활성화 없으면 0.0
        active_count: 활성 뉴런 수 (0 ~ size)
        stability: 안정성 점수 (0.0 ~ 1.0)
        drift: 드리프트 거리 (neuron 단위), 이전 step과의 차이
        sustained: 상태 유지 여부 (active_count > 0)
    """
    center: float
    width: float
    active_count: int
    stability: float
    drift: float
    sustained: bool
    orbit_stability: float = 1.0  # 궤도 안정성 점수 (0.0 ~ 1.0)


class RingAttractorEngine:
    """
    Ring Attractor Engine
    
    Persistent State Engine: 입력이 사라진 이후에도 연속적인 상태를 안정적으로 유지하는 최소 단위의 상태 메모리 엔진.
    
    이 엔진은 독립적으로 사용할 수 있는 최소 단위 부품입니다.
    내부 구현 세부사항은 캡슐화되어 있으며, 물리적 법칙만 노출됩니다.
    
    사용 예:
        engine = RingAttractorEngine(size=15, config="case2")
        engine.inject(direction_idx=5, strength=0.8)
        state1 = engine.run(duration_ms=2.5)
        engine.release_input()
        state2 = engine.run(duration_ms=150.0)
        print(f"State retained: {state2.sustained}")
    """
    
    def __init__(
        self,
        size: int = 15,
        config: str = "stable",
        seed: Optional[int] = None,
        debug: bool = False,
        state_type: Optional['StateType'] = None,
        application_domain: Optional['ApplicationDomain'] = None
    ):
        """
        엔진 초기화
        
        Args:
            size: Ring 크기 (뉴런 수)
            config: 설정 ("stable", "balanced", "aggressive")
            seed: 랜덤 시드 (재현성 보장)
            debug: 디버그 모드
            state_type: 상태 타입 (StateType) - None이면 기본값 사용
            application_domain: 적용 도메인 (ApplicationDomain) - None이면 기본값 사용
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 적용 도메인 설정 (도메인이 지정되면 자동으로 state_type 설정)
        if application_domain is not None:
            try:
                from .application_domains import get_domain_config
            except ImportError:
                from application_domains import get_domain_config
            
            domain_config = get_domain_config(application_domain)
            if state_type is None:
                state_type = domain_config.state_type
            if size == 15:  # 기본값이면 도메인 설정 사용
                size = domain_config.size
            if config == "stable":  # 기본값이면 도메인 설정 사용
                config = domain_config.config
            
            self.application_domain = application_domain
            self.domain_config = domain_config
        else:
            self.application_domain = None
            self.domain_config = None
        
        # 상태 타입 설정
        if state_type is None:
            # StateType import (순환 참조 방지)
            try:
                from .state_types import StateType as ST
                state_type = ST.PHASE  # 기본값: 회전 위상
            except ImportError:
                try:
                    from state_types import StateType as ST
                    state_type = ST.PHASE
                except ImportError:
                    state_type = None
        
        if state_type is not None:
            try:
                from .state_types import get_state_type_config
            except ImportError:
                from state_types import get_state_type_config
            
            self.state_type = state_type
            try:
                self.state_config = get_state_type_config(state_type)
            except:
                self.state_config = None
        else:
            self.state_type = None
            self.state_config = None
        
        self.size = size
        self.config = config
        self.debug = debug
        
        # 내부 상태 (생물학적 세부사항은 숨김)
        self._initialize_internal_state()
        
        # 설정 로드
        self._load_config(config)
    
    def _initialize_internal_state(self):
        """
        내부 상태 초기화 (생물학적 구현 세부사항)
        
        Ring Attractor 엔진의 내부 네트워크를 구성합니다:
        - CA3 뉴런 생성 (size 개수만큼)
        - Recurrent 연결 생성 (Mexican-hat topology)
        - Homeostasis 초기화
        """
        self._dt = 0.1
        self._current_time = 0.0
        
        # HippoMemoryV4System 생성 (내부 네트워크 관리)
        self._system = HippoMemoryV4System(dt=self._dt, learn=False)  # Ring Attractor는 학습 불필요
        
        # CA3 뉴런 생성 (size 개수만큼)
        # 이름: "ring_0", "ring_1", ..., "ring_{size-1}"
        self._neuron_names = []
        for i in range(self.size):
            neuron_name = f"ring_{i}"
            self._system.add_ca3_neuron(neuron_name)
            self._neuron_names.append(neuron_name)
        
        # baseline_V는 _load_config()에서 설정됨
        
        # 이전 center 저장 (drift 계산용)
        self._previous_center = None
        
        # Phase 1 spike count 백업 (Phase 2 측정용)
        self._phase1_spike_counts = {}
        
        # Predictive Drift Control 관련 상태
        self._phase_history = []  # 위상 이력
        self._velocity_history = []  # 속도 이력
        self._prediction_horizon = 100.0  # 예측 시간 간격 (ms)
        self._disturbance_threshold = 1.0  # 외란 임계값
        self._max_correction_force = 0.2  # 최대 보정 힘
        
        if self.debug:
            print(f"[DEBUG] Initialized {self.size} ring neurons")
    
    def _load_config(self, config: str):
        """
        설정 로드 및 네트워크 구성
        
        ring_attractor_config.py에서 설정을 로드하고,
        Recurrent 연결을 생성합니다 (Mexican-hat topology).
        """
        # 설정 로드
        if get_case_params is not None:
            try:
                params = get_case_params(config)
                self._recurrent_weight = params.recurrent_base_weight
                self._inhibition_base = params.w_inh_base
                self._sigma = params.sigma
                self._r_exc = params.r_exc
                self._baseline_V = params.baseline_V
                self._cue_duration = params.cue_duration
                self._directional_bias_enabled = params.directional_bias_enabled
                self._directional_bias_strength = params.directional_bias_strength
                self._directional_bias_direction = params.directional_bias_direction
            except (ValueError, AttributeError):
                # fallback: 기본값 사용
                self._recurrent_weight = 0.28
                self._inhibition_base = 0.68
                self._sigma = 2.0
                self._r_exc = 2
                self._baseline_V = -55.0
                self._cue_duration = 1.5
                self._directional_bias_enabled = False
                self._directional_bias_strength = 0.1
                self._directional_bias_direction = 1
        else:
            # fallback: 기본값
            if config == "stable":
                self._recurrent_weight = 0.28
                self._inhibition_base = 0.68
                self._sigma = 2.0
                self._r_exc = 2
                self._baseline_V = -55.0
            elif config == "balanced":
                self._recurrent_weight = 0.35
                self._inhibition_base = 0.50
                self._sigma = 2.0
                self._r_exc = 3
                self._baseline_V = -55.0
            elif config == "aggressive":
                self._recurrent_weight = 0.40
                self._inhibition_base = 0.75
                self._sigma = 1.5
                self._r_exc = 2
                self._baseline_V = -55.0
            else:
                raise ValueError(f"Unknown config: {config}")
            self._cue_duration = 1.5
            self._directional_bias_enabled = False
            self._directional_bias_strength = 0.1
            self._directional_bias_direction = 1
        
        # CA3 뉴런에 baseline_V 적용
        for neuron_name in self._neuron_names:
            neuron = self._system.ca3_neurons.get(neuron_name)
            if neuron:
                neuron.baseline_V = self._baseline_V
                neuron.soma.V = self._baseline_V
        
        # CA3 → CA3 Recurrent Connections 생성 (Mexican-hat topology)
        self._create_recurrent_connections()
        
        if self.debug:
            print(f"[DEBUG] Config loaded: {config}")
            print(f"[DEBUG]   recurrent_weight={self._recurrent_weight}, inhibition={self._inhibition_base}")
    
    def _create_recurrent_connections(self):
        """
        CA3 → CA3 Recurrent 연결 생성 (Mexican-hat topology)
        
        가까운 이웃: 흥분 (E)
        원거리: 억제 (I)
        """
        n_neurons = self.size
        
        for i, source in enumerate(self._neuron_names):
            for j, target in enumerate(self._neuron_names):
                if i != j:  # 자기 자신 제외
                    # 링 거리 계산 (원형 인덱스: 0과 size-1이 인접)
                    ring_distance = min(abs(i - j), n_neurons - abs(i - j))
                    
                    # 방향성 계산 (시계방향/반시계방향 구분)
                    direct_dist = j - i
                    if direct_dist > n_neurons // 2:
                        is_clockwise = False
                    elif direct_dist < -n_neurons // 2:
                        is_clockwise = True
                    else:
                        is_clockwise = (direct_dist > 0)
                    
                    if ring_distance <= self._r_exc:
                        # 가까운 이웃: 흥분 (Gaussian) + 정규화
                        gaussian_weight = self._recurrent_weight * np.exp(
                            -(ring_distance**2) / (2 * self._sigma**2)
                        )
                        normalized_weight = gaussian_weight / n_neurons
                        
                        # 방향성 편향 적용 (선택적)
                        if self._directional_bias_enabled:
                            if (is_clockwise and self._directional_bias_direction > 0) or \
                               (not is_clockwise and self._directional_bias_direction < 0):
                                bias_factor = 1.0 + self._directional_bias_strength
                            else:
                                bias_factor = max(0.1, 1.0 - self._directional_bias_strength * 2.0)
                            normalized_weight *= bias_factor
                            
                            # 극단적 억제 (선택적)
                            if self._directional_bias_strength > 0.55:
                                if not ((is_clockwise and self._directional_bias_direction > 0) or \
                                       (not is_clockwise and self._directional_bias_direction < 0)):
                                    continue  # 반대 방향 연결 차단
                        
                        final_weight = normalized_weight
                        synapse_type = "excitatory"
                        
                        # Baseline 전위 국소적 조정 (재귀 연결 타겟 뉴런에만 +1.5mV)
                        target_neuron = self._system.ca3_neurons.get(target)
                        if target_neuron and hasattr(target_neuron, 'baseline_V'):
                            if not hasattr(target_neuron, '_recurrent_baseline_bias_applied'):
                                target_neuron.baseline_V = target_neuron.baseline_V + 1.5
                                target_neuron._recurrent_baseline_bias_applied = True
                    else:
                        # 원거리: 억제 (전역 억제)
                        final_weight = self._inhibition_base
                        synapse_type = "inhibitory"
                    
                    # 연결 생성
                    self._system.add_connection(
                        source, target,
                        weight=final_weight,
                        delay=1.5,
                        synapse_type=synapse_type
                    )
        
        if self.debug:
            num_connections = n_neurons * (n_neurons - 1)
            print(f"[DEBUG] Created {num_connections} recurrent connections")
    
    def inject(self, direction_idx: int, strength: float = 0.8):
        """
        외부 자극 주입 (입력 시작)
        
        입력을 주입한 후, run()을 호출하여 실행해야 합니다.
        
        Args:
            direction_idx: 방향 인덱스 (0 ~ size-1)
            strength: 입력 강도 (0.0 ~ 1.0, 기본값: 0.8)
        
        Returns:
            None (내부 상태 업데이트)
        """
        if direction_idx < 0 or direction_idx >= self.size:
            raise ValueError(f"direction_idx must be in [0, {self.size-1}]")
        
        if strength < 0.0 or strength > 1.0:
            raise ValueError("strength must be in [0.0, 1.0]")
        
        # Cue 뉴런 선택
        cue_neuron_name = self._neuron_names[direction_idx]
        cue_neuron = self._system.ca3_neurons.get(cue_neuron_name)
        
        if not cue_neuron:
            raise ValueError(f"Neuron {cue_neuron_name} not found")
        
        # I_ext 값 계산 (strength를 실제 전류 값으로 변환)
        # strength 0.0~1.0 → I_ext 0.0~200.0 pA
        I_ext_value = strength * 200.0
        
        # I_ext 딕셔너리로 관리 (run()에서 사용)
        if not hasattr(self, '_I_ext_dict'):
            self._I_ext_dict = {}
        self._I_ext_dict[cue_neuron_name] = I_ext_value
        
        # 초기 조건 비대칭성 (선택적, directional_bias가 활성화된 경우)
        if self._directional_bias_enabled and hasattr(cue_neuron, 'soma'):
            cue_neuron.soma.V = self._baseline_V + 2.0  # +2.0mV 초기 편향
        
        # 입력 시작 시점 기록
        self._input_start_time = self._current_time
        self._input_direction = direction_idx
        self._input_strength = strength
        
        if self.debug:
            print(f"[DEBUG] Inject: direction={direction_idx}, strength={strength}, I_ext={I_ext_value:.1f}pA")
    
    def release_input(self):
        """
        외부 자극 제거 (입력 종료)
        
        inject()로 주입한 입력을 제거합니다.
        """
        # 모든 뉴런의 I_ext 제거
        for neuron_name in self._neuron_names:
            neuron = self._system.ca3_neurons.get(neuron_name)
            if neuron and hasattr(neuron, 'I_ext'):
                neuron.I_ext = 0.0
        
        # I_ext 딕셔너리도 제거
        if hasattr(self, '_I_ext_dict'):
            self._I_ext_dict = {}
        
        # Phase 1 종료 직후 spike count 백업 (Phase 2 측정을 위해)
        self._phase1_spike_counts = {
            name: neuron.wake_spike_count
            for name, neuron in self._system.ca3_neurons.items()
            if name in self._neuron_names
        }
        
        if self.debug:
            print(f"[DEBUG] Release input")
    
    def input(self, direction_idx: int, strength: float = 0.8, duration: Optional[float] = None):
        """
        외부 자극 입력 (레거시 메서드, 호환성 유지)
        
        ⚠️ Deprecated: inject() + run() + release_input() 사용을 권장합니다.
        
        Args:
            direction_idx: 방향 인덱스 (0 ~ size-1)
            strength: 입력 강도 (0.0 ~ 1.0, 기본값: 0.8)
            duration: 입력 지속 시간 [ms] (None이면 설정값 사용)
        
        Returns:
            phase_result (dict)
        """
        if duration is None:
            duration = self._cue_duration
        
        # inject + run + release_input
        self.inject(direction_idx, strength)
        result = self.run(duration)
        self.release_input()
        
        return result
    
    def step(self) -> RingState:
        """
        한 스텝 전진 (1 tick, dt만큼)
        
        내부에서 dt (기본값: 0.1ms)만큼 한 번만 전진합니다.
        여러 스텝을 실행하려면 run()을 사용하세요.
        
        Returns:
            RingState: 현재 상태 (center, width, stability, drift 등)
        """
        dt = self._dt
        
        # Phase 2 시작 전 설정: I_ext 제거 및 Homeostasis 관리
        for neuron_name in self._neuron_names:
            neuron = self._system.ca3_neurons.get(neuron_name)
            if not neuron:
                continue
            
            # I_ext 제거
            if hasattr(neuron, 'I_ext'):
                neuron.I_ext = 0.0
            
            # Homeostasis 완화 (선택적, Phase 2에서 bump 유지 허용)
            if hasattr(neuron, 'spike_count_window'):
                # Phase 2 시작 시 Homeostasis 상태 완전 리셋
                neuron.spike_count_window = 0
                if hasattr(neuron, 'dynamic_threshold_penalty'):
                    neuron.dynamic_threshold_penalty = 0.0
                if hasattr(neuron, 'soma') and hasattr(neuron, 'base_spike_thresh'):
                    neuron.soma.spike_thresh = neuron.base_spike_thresh
                
                # spike_budget는 정상 작동하도록 유지 (완화 최소화)
                original_budget = getattr(neuron, 'spike_budget', 5)
                neuron.spike_budget = original_budget + 0  # 완화 제거 (정상 Homeostasis)
            
            # V 리셋 (포화 상태 해제, 선택적)
            if hasattr(neuron, 'baseline_V') and hasattr(neuron, 'soma'):
                if neuron.soma.V > neuron.baseline_V:
                    neuron.soma.V = neuron.baseline_V
        
        # Phase 2 시작 전 bump_center 측정 (drift 계산용)
        phase2_start_center = self._calculate_bump_center(self._phase1_spike_counts)
        
        # Phase 2 실행: 입력 제거 후 유지
        I_ext_dict_empty = {}  # 모든 입력 제거
        phase_result = self._system.run_phase(dt, I_ext_dict_empty)
        
        # Phase 2 종료 후 spike count 수집
        phase2_spike_counts = {
            name: neuron.wake_spike_count
            for name, neuron in self._system.ca3_neurons.items()
            if name in self._neuron_names
        }
        
        # Phase 2 동안 추가된 spike만 계산 (Phase 1 이후 증가분)
        phase2_additional_spikes = {
            name: phase2_spike_counts.get(name, 0) - self._phase1_spike_counts.get(name, 0)
            for name in self._neuron_names
        }
        
        # Phase 2 종료 후 bump_center 측정
        phase2_end_center = self._calculate_bump_center(phase2_additional_spikes)
        
        # Drift 계산
        drift_distance = 0.0
        if phase2_start_center is not None and phase2_end_center is not None:
            drift_distance = abs(phase2_end_center - phase2_start_center)
        
        # 현재 상태 계산
        state = self._calculate_state(phase2_additional_spikes, drift_distance)
        
        # 이전 center 업데이트
        self._previous_center = state.center
        
        # 현재 시간 업데이트
        self._current_time += dt
        
        # Predictive Drift Control: 이력 데이터 수집 (매 step마다)
        current_center = state.center
        current_drift = state.drift
        current_velocity = current_drift / 0.1 if current_drift > 0 else 0.0
        self._phase_history.append(current_center)
        self._velocity_history.append(current_velocity)
        
        # 이력 데이터 크기 제한 (최근 100개만 유지)
        if len(self._phase_history) > 100:
            self._phase_history.pop(0)
        if len(self._velocity_history) > 100:
            self._velocity_history.pop(0)
        
        if self.debug:
            print(f"[DEBUG] Step: time={self._current_time:.2f}ms, drift={drift_distance:.2f}, center={state.center:.2f}")
        
        return state
    
    def run(self, duration_ms: Optional[float] = None) -> RingState:
        """
        지정된 시간만큼 실행 (내부 루프)
        
        내부에서 step()을 반복 호출하여 duration_ms만큼 실행합니다.
        사용자는 내부 루프를 신뢰할 수 있습니다.
        
        Args:
            duration_ms: 실행 시간 [ms] (None이면 내부 default 사용)
        
        Returns:
            RingState: 최종 상태
        """
        if duration_ms is None:
            duration_ms = self._cue_duration
        num_steps = int(duration_ms / self._dt)
        
        if self.debug:
            print(f"[DEBUG] Run: {duration_ms:.1f}ms ({num_steps} steps)")
        
        # I_ext_dict 설정 (run_phase 형식)
        I_ext_dict = {}
        if hasattr(self, '_I_ext_dict') and self._I_ext_dict:
            for neuron_name, I_ext_value in self._I_ext_dict.items():
                neuron = self._system.ca3_neurons.get(neuron_name)
                if neuron:
                    I_ext_dict[neuron_name] = {
                        'value': I_ext_value,
                        'start': self._current_time,
                        'end': self._current_time + duration_ms
                    }
        
        # Phase 2 시작 전 설정 (입력이 없는 경우)
        if not I_ext_dict:
            # 입력 제거 후 유지 모드
            for neuron_name in self._neuron_names:
                neuron = self._system.ca3_neurons.get(neuron_name)
                if not neuron:
                    continue
                
                # I_ext 제거
                if hasattr(neuron, 'I_ext'):
                    neuron.I_ext = 0.0
                
                # Homeostasis 완화 (선택적)
                if hasattr(neuron, 'spike_count_window'):
                    neuron.spike_count_window = 0
                    if hasattr(neuron, 'dynamic_threshold_penalty'):
                        neuron.dynamic_threshold_penalty = 0.0
                    if hasattr(neuron, 'soma') and hasattr(neuron, 'base_spike_thresh'):
                        neuron.soma.spike_thresh = neuron.base_spike_thresh
                    original_budget = getattr(neuron, 'spike_budget', 5)
                    neuron.spike_budget = original_budget + 0
                
                # V 리셋 (포화 상태 해제, 선택적)
                if hasattr(neuron, 'baseline_V') and hasattr(neuron, 'soma'):
                    if neuron.soma.V > neuron.baseline_V:
                        neuron.soma.V = neuron.baseline_V
            
            # Phase 2 시작 전 bump_center 측정 (drift 계산용)
            phase2_start_center = self._calculate_bump_center(self._phase1_spike_counts)
        else:
            # Phase 1 (입력 중)
            phase2_start_center = None
        
        # run_phase 호출
        phase_result = self._system.run_phase(duration_ms, I_ext_dict)
        
        # Phase 종료 후 spike count 수집
        current_spike_counts = {
            name: neuron.wake_spike_count
            for name, neuron in self._system.ca3_neurons.items()
            if name in self._neuron_names
        }
        
        # Phase 2인 경우 추가분 계산
        if not I_ext_dict and hasattr(self, '_phase1_spike_counts'):
            phase2_additional_spikes = {
                name: current_spike_counts.get(name, 0) - self._phase1_spike_counts.get(name, 0)
                for name in self._neuron_names
            }
            phase2_end_center = self._calculate_bump_center(phase2_additional_spikes)
            
            # Drift 계산
            drift_distance = 0.0
            if phase2_start_center is not None and phase2_end_center is not None:
                drift_distance = abs(phase2_end_center - phase2_start_center)
            
            state = self._calculate_state(phase2_additional_spikes, drift_distance)
        else:
            # Phase 1인 경우 전체 spike count 사용
            drift_distance = 0.0
            state = self._calculate_state(current_spike_counts, drift_distance)
        
        # 이전 center 업데이트
        self._previous_center = state.center
        
        # 현재 시간 업데이트
        self._current_time += duration_ms
        
        return state
    
    def _calculate_bump_center(self, spike_counts: Dict[str, int]) -> Optional[float]:
        """
        Bump center 계산 (weighted mean)
        
        Args:
            spike_counts: 뉴런 이름 -> spike count 딕셔너리
        
        Returns:
            Bump center (0 ~ size-1), 또는 None (활성화 없음)
        """
        # Activation vector 생성
        activation_vector = [
            spike_counts.get(f"ring_{i}", 0) for i in range(self.size)
        ]
        total_activation = sum(activation_vector)
        
        if total_activation == 0:
            return None
        
        # Weighted mean 계산
        bump_center = sum(i * activation_vector[i] for i in range(self.size)) / total_activation
        return bump_center
    
    def _calculate_state(self, spike_counts: Dict[str, int], drift: float) -> RingState:
        """
        RingState 계산
        
        Args:
            spike_counts: 뉴런 이름 -> spike count 딕셔너리
            drift: 드리프트 거리
        
        Returns:
            RingState 객체
        """
        # Activation vector 생성
        activation_vector = [
            spike_counts.get(f"ring_{i}", 0) for i in range(self.size)
        ]
        total_activation = sum(activation_vector)
        active_count = sum(1 for count in activation_vector if count > 0)
        
        if total_activation == 0:
            return RingState(
                center=0.0,
                width=0.0,
                active_count=0,
                stability=0.0,
                drift=drift,
                sustained=False,
                orbit_stability=0.0  # 활성화 없으면 궤도 안정성 0
            )
        
        # Bump center 계산
        bump_center = sum(i * activation_vector[i] for i in range(self.size)) / total_activation
        
        # Bump width 계산 (standard deviation)
        bump_variance = sum(
            (i - bump_center)**2 * activation_vector[i]
            for i in range(self.size)
        ) / total_activation
        bump_width = np.sqrt(bump_variance) if bump_variance > 0 else 0.0
        
        # Stability 계산 (간단한 휴리스틱)
        # 활성화 수가 5~7개이고 width가 적절하면 안정적
        stability = 0.0
        if 5 <= active_count <= 7:
            stability += 0.5
        if 1.0 <= bump_width <= 4.5:
            stability += 0.3
        if active_count > 0:
            stability += 0.2
        
        # Sustained 판정 (활성화가 유지되고 있는지)
        sustained = active_count > 0
        
        # Orbit Stability Score 계산 (0.0 ~ 1.0)
        orbit_stability = self._calculate_orbit_stability(
            bump_center, bump_width, active_count, drift, sustained
        )
        
        return RingState(
            center=bump_center,
            width=bump_width,
            active_count=active_count,
            stability=min(1.0, stability),
            drift=drift,
            sustained=sustained,
            orbit_stability=orbit_stability
        )
    
    def get_state(self) -> RingState:
        """
        현재 상태 반환
        
        Returns:
            RingState: 현재 상태 (center, width, stability 등)
        """
        # 현재 spike count 수집 (전체)
        current_spike_counts = {
            name: neuron.wake_spike_count
            for name, neuron in self._system.ca3_neurons.items()
            if name in self._neuron_names
        }
        
        # Phase 1 이후 증가분 계산 (Phase 2 동안의 활성화)
        phase2_additional_spikes = {
            name: current_spike_counts.get(name, 0) - self._phase1_spike_counts.get(name, 0)
            for name in self._neuron_names
        }
        
        # Phase 2가 실행되었는지 확인 (phase2_additional_spikes가 모두 0이면 Phase 1 직후)
        phase2_executed = any(count > 0 for count in phase2_additional_spikes.values())
        
        # Phase 1 직후면 전체 spike count 사용, Phase 2 이후면 추가분 사용
        if phase2_executed:
            # Phase 2 이후: 추가분만 사용
            spike_counts_for_state = phase2_additional_spikes
        else:
            # Phase 1 직후: 전체 spike count 사용
            spike_counts_for_state = current_spike_counts
        
        # Drift 계산 (이전 center와 비교)
        current_center = self._calculate_bump_center(spike_counts_for_state)
        drift = 0.0
        if self._previous_center is not None and current_center is not None:
            drift = abs(current_center - self._previous_center)
        
        # 상태 계산
        state = self._calculate_state(spike_counts_for_state, drift)
        
        return state
    
    def reset(self):
        """상태 리셋"""
        self._current_time = 0.0
        self._previous_center = None
        
        # 뉴런 상태 리셋
        for neuron_name in self._neuron_names:
            neuron = self._system.ca3_neurons.get(neuron_name)
            if neuron:
                neuron.soma.V = self._baseline_V
                if hasattr(neuron, 'I_ext'):
                    neuron.I_ext = 0.0
                if hasattr(neuron, 'I_syn_accumulated'):
                    neuron.I_syn_accumulated = 0.0
                if hasattr(neuron, 'recurrent_state'):
                    neuron.recurrent_state = 0.0
                if hasattr(neuron, 'spike_count_window'):
                    neuron.spike_count_window = 0
                if hasattr(neuron, 'dynamic_threshold_penalty'):
                    neuron.dynamic_threshold_penalty = 0.0
                if hasattr(neuron, 'soma'):
                    neuron.soma.spike_thresh = getattr(neuron, 'base_spike_thresh', neuron.soma.spike_thresh)
        
        if self.debug:
            print("[DEBUG] Engine reset")
    
    def _calculate_orbit_stability(
        self,
        center: float,
        width: float,
        active_count: int,
        drift: float,
        sustained: bool
    ) -> float:
        """
        Orbit Stability Score 계산 (0.0 ~ 1.0)
        
        궤도 신뢰도: 중심 위치의 안정성, 범프의 일관성, 드리프트 크기 등을 종합
        
        Args:
            center: 범프 중심
            width: 범프 너비
            active_count: 활성 뉴런 수
            drift: 드리프트 거리
            sustained: 상태 유지 여부
        
        Returns:
            Orbit stability score (0.0 ~ 1.0)
        """
        if not sustained or active_count == 0:
            return 0.0
        
        score = 0.0
        
        # 1. 활성화 수 적절성 (40%)
        if 5 <= active_count <= 7:
            score += 0.4
        elif 3 <= active_count <= 9:
            score += 0.2
        
        # 2. 범프 너비 적절성 (30%)
        if 1.0 <= width <= 3.0:
            score += 0.3
        elif 0.5 <= width <= 4.5:
            score += 0.15
        
        # 3. 드리프트 작음 (20%)
        # 작은 drift는 정상, 큰 drift는 불안정
        if drift < 0.5:
            score += 0.2
        elif drift < 1.0:
            score += 0.1
        elif drift < 2.0:
            score += 0.05
        
        # 4. 이전 중심과의 일관성 (10%)
        # 이전 center와 비교 (있는 경우)
        if hasattr(self, '_previous_center') and self._previous_center is not None:
            center_diff = abs(center - self._previous_center)
            if center_diff < 1.0:
                score += 0.1
            elif center_diff < 2.0:
                score += 0.05
        
        return min(1.0, score)
    
    def apply_disturbance(
        self,
        direction_idx: int,
        strength: float,
        duration_ms: float = 5.0
    ) -> RingState:
        """
        외란 주입 (Disturbance Recovery 테스트용)
        
        Phase 2 도중 강제로 외란을 주입하고 복구 성능을 측정합니다.
        
        Args:
            direction_idx: 외란 방향 (0 ~ size-1)
            strength: 외란 강도 (0.0 ~ 1.0)
            duration_ms: 외란 지속 시간 [ms]
        
        Returns:
            RingState: 외란 주입 후 복구 상태
        """
        # 외란 주입 전 상태 저장
        pre_disturbance_state = self.get_state()
        pre_disturbance_center = pre_disturbance_state.center
        
        # 외란 주입
        self.inject(direction_idx=direction_idx, strength=strength)
        post_disturbance_state = self.run(duration_ms=duration_ms)
        self.release_input()
        
        # 복구 시간 측정 (외란 제거 후 원래 위치로 복귀)
        recovery_state = self.run(duration_ms=50.0)  # 복구 시간
        
        # 복구 성능 계산
        center_shift = abs(post_disturbance_state.center - pre_disturbance_center)
        recovery_distance = abs(recovery_state.center - pre_disturbance_center)
        
        if self.debug:
            print(f"[DEBUG] Disturbance: center_shift={center_shift:.2f}, "
                  f"recovery_distance={recovery_distance:.2f}")
        
        return recovery_state
    
    def measure_recovery_performance(
        self,
        disturbance_strength: float = 0.2,
        recovery_time_ms: float = 100.0
    ) -> Dict[str, float]:
        """
        복구 성능 측정
        
        외란 주입 후 복구 성능을 정량적으로 측정합니다.
        
        Args:
            disturbance_strength: 외란 강도
            recovery_time_ms: 복구 확인 시간 [ms]
        
        Returns:
            Dict: 복구 성능 지표
                - center_shift: 외란으로 인한 중심 이동
                - recovery_distance: 복구 후 원래 위치와의 거리
                - recovery_ratio: 복구 비율 (0.0 = 완전 복구, 1.0 = 복구 실패)
                - orbit_stability_after: 복구 후 궤도 안정성
                - sustained_after: 복구 후 상태 유지 여부
        """
        # 현재 상태 저장
        initial_state = self.get_state()
        initial_center = initial_state.center
        
        # 반대 방향에 외란 주입
        disturbance_direction = (int(initial_center) + 7) % self.size
        
        # 외란 주입
        self.inject(direction_idx=disturbance_direction, strength=disturbance_strength)
        disturbed_state = self.run(duration_ms=5.0)
        self.release_input()
        
        # 복구 확인
        recovered_state = self.run(duration_ms=recovery_time_ms)
        
        # 성능 지표 계산
        center_shift = abs(disturbed_state.center - initial_center)
        recovery_distance = abs(recovered_state.center - initial_center)
        
        # 복구 비율 (0.0 = 완전 복구, 1.0 = 복구 실패)
        if center_shift > 0:
            recovery_ratio = recovery_distance / center_shift
        else:
            recovery_ratio = 0.0
        
        return {
            'center_shift': center_shift,
            'recovery_distance': recovery_distance,
            'recovery_ratio': recovery_ratio,
            'orbit_stability_after': recovered_state.orbit_stability,
            'sustained_after': recovered_state.sustained
        }
    
    def run_with_drift(
        self,
        velocity: float,
        duration_ms: float
    ) -> RingState:
        """
        Controlled Drift 실행
        
        상태가 유지되면서 천천히 이동하는 기능입니다.
        이는 "의도된 이동"으로, 외란에 의한 drift가 아닌 제어된 drift입니다.
        
        핵심 질문: "상태가 유지되면서, 천천히 이동할 수 있는가?"
        
        이게 되면:
        - 자율 선박 항로 유지
        - 회전축 미세 보정
        - 로터 중심선 유지
        - 작업 중 상태 전이
        
        모두 가능해집니다.
        
        Args:
            velocity: 이동 속도 (neuron/ms, 양수=시계방향, 음수=반시계방향)
            duration_ms: 실행 시간 [ms]
        
        Returns:
            RingState: 이동 후 상태
        """
        # 현재 상태 저장
        initial_state = self.get_state()
        initial_center = initial_state.center
        
        # 목표 위상 계산 (속도 * 시간)
        target_phase = (initial_center + velocity * duration_ms) % self.size
        
        # 작은 단계로 나누어 이동 (안정성 유지)
        num_steps = max(1, int(duration_ms / 10.0))  # 10ms 단위로 나눔
        step_duration = duration_ms / num_steps
        step_velocity = velocity / num_steps
        
        for i in range(num_steps):
            # 각 단계에서 약간의 방향성 입력 주입 (제어된 drift)
            # 현재 중심에서 목표 방향으로 약한 입력
            current_center = self.get_state().center
            
            # 목표 방향 계산
            if target_phase > current_center:
                if target_phase - current_center < self.size / 2:
                    direction = int((current_center + 1) % self.size)
                else:
                    direction = int((current_center - 1) % self.size)
            else:
                if current_center - target_phase < self.size / 2:
                    direction = int((current_center - 1) % self.size)
                else:
                    direction = int((current_center + 1) % self.size)
            
            # 약한 방향성 입력 (제어된 drift)
            drift_strength = min(0.1, abs(step_velocity) * 0.5)  # 속도에 비례한 강도
            if drift_strength > 0.01:
                self.inject(direction_idx=direction, strength=drift_strength)
                self.run(duration_ms=step_duration)
                self.release_input()
            else:
                self.run(duration_ms=step_duration)
        
        # 최종 상태
        final_state = self.get_state()
        
        if self.debug:
            actual_drift = abs(final_state.center - initial_center)
            print(f"[DEBUG] Controlled Drift: target={target_phase:.2f}, "
                  f"actual={final_state.center:.2f}, drift={actual_drift:.2f}")
        
        return final_state
    
    def get_physics_output(
        self,
        target_phase: Optional[float] = None
    ) -> 'PhysicsOutput':
        """
        물리 시스템 출력 반환
        
        이 엔진이 물리계에 제공하는 출력값을 반환합니다.
        하드웨어는 이 값들을 보고 제어를 수행합니다.
        
        Args:
            target_phase: 목표 위상 (None이면 현재 center 사용)
        
        Returns:
            PhysicsOutput: 물리 시스템 출력
                - desired_phase: 목표 위상
                - phase_velocity: 위상 속도
                - stability_score: 안정성 점수
                - deviation_error: 편차 오차
                - orbit_stability: 궤도 안정성
        """
        try:
            from .physics_interface import PhysicsOutput, ring_state_to_physics_output
        except ImportError:
            from physics_interface import PhysicsOutput, ring_state_to_physics_output
        
        state = self.get_state()
        return ring_state_to_physics_output(state, target_phase)
    
    def predict_future_phase(
        self,
        prediction_horizon_ms: float = 100.0
    ) -> Dict[str, float]:
        """
        미래 위상 예측
        
        Look-ahead Logic: 현재 Phase Velocity와 Acceleration을 바탕으로
        링 어트랙터의 범프가 1-step 뒤에 가야 할 '이상적 좌표'를 미리 계산합니다.
        
        수학적 모델:
        ------------
        1. 위상 속도 계산:
        
            v(t) = drift(t) / dt
        
        여기서:
            drift(t): 현재 드리프트 거리 (neuron 단위)
            dt: 시간 간격 (기본값: 0.1ms)
        
        2. 위상 가속도 추정:
        
            a(t) = (v(t) - v(t-Δt)) / Δt
        
        이력 데이터 기반:
            if len(velocity_history) >= 2:
                a(t) = (velocity_history[-1] - velocity_history[-2]) / dt
        
        3. 미래 위상 예측:
        
            φ_predicted(t + Δt) = φ(t) + v(t) · Δt + 0.5 · a(t) · Δt²
        
        여기서:
            φ(t): 현재 위상
            v(t): 위상 속도
            a(t): 위상 가속도
            Δt: 예측 시간 간격 (prediction_horizon_ms)
        
        Ring 구조 고려:
            φ_predicted = φ_predicted mod N
        
        여기서:
            N: Ring 크기 (뉴런 수)
        
        4. 미래 속도 예측:
        
            v_predicted(t + Δt) = v(t) + a(t) · Δt
        
        5. 예측 신뢰도:
        
            confidence = {
                0.0                                    if N_history < 5
                min(1.0, (N_history - 5) / 20.0 + 0.3)  otherwise
            }
        
        여기서:
            N_history: 이력 데이터 포인트 수
            최소 5개 이상의 이력 데이터가 있어야 신뢰도 계산
            20개 이상이면 신뢰도 1.0
        
        개념:
        -----
        - Predictive Control: 미래 상태를 예측하여 선제적 제어
        - Look-ahead: 현재 상태와 속도, 가속도를 기반으로 미래 예측
        - Confidence: 이력 데이터가 많을수록 예측 신뢰도 높음
        
        Args:
            prediction_horizon_ms: 예측 시간 간격 [ms] (기본값: 100.0)
        
        Returns:
            Dict containing:
                - predicted_phase: 예측된 미래 위상 (0 ~ size-1)
                - predicted_velocity: 예측된 미래 속도 (neuron/ms)
                - predicted_disturbance: 예측된 외란 크기 (현재는 0.0)
                - confidence: 예측 신뢰도 (0.0 ~ 1.0)
        """
        current_state = self.get_state()
        current_phase = current_state.center
        current_drift = current_state.drift
        
        # 위상 속도 계산 (drift를 속도로 변환)
        # drift는 neuron 단위이므로, 이를 시간 단위로 나누면 속도
        # 기본 dt = 0.1ms이므로, drift를 0.1로 나누면 neuron/ms
        current_velocity = current_drift / 0.1 if current_drift > 0 else 0.0
        
        # 위상 가속도 추정 (이력 데이터 기반)
        acceleration = 0.0
        if len(self._velocity_history) >= 2:
            dt = self._dt  # 0.1ms
            recent_velocities = self._velocity_history[-2:]
            acceleration = (recent_velocities[-1] - recent_velocities[0]) / dt
        
        # 미래 위상 예측
        # φ_predicted(t+Δt) = φ(t) + v(t) * Δt + 0.5 * a(t) * Δt²
        dt_prediction = prediction_horizon_ms
        predicted_phase = (
            current_phase +
            current_velocity * dt_prediction +
            0.5 * acceleration * dt_prediction * dt_prediction
        )
        
        # Ring 구조에 맞게 모듈러 연산
        predicted_phase = predicted_phase % self.size
        
        # 미래 속도 예측
        # v_predicted(t+Δt) = v(t) + a(t) * Δt
        predicted_velocity = current_velocity + acceleration * dt_prediction
        
        # 예측 신뢰도 계산 (이력 데이터가 많을수록 신뢰도 높음)
        # 최소 5개 이상의 이력 데이터가 있어야 신뢰도 계산
        if len(self._phase_history) >= 5:
            confidence = min(1.0, (len(self._phase_history) - 5) / 20.0 + 0.3)
        else:
            confidence = 0.0
        
        # 이력 데이터 업데이트
        self._phase_history.append(current_phase)
        self._velocity_history.append(current_velocity)
        
        # 이력 데이터 크기 제한 (최근 100개만 유지)
        if len(self._phase_history) > 100:
            self._phase_history.pop(0)
        if len(self._velocity_history) > 100:
            self._velocity_history.pop(0)
        
        return {
            'predicted_phase': predicted_phase,
            'predicted_velocity': predicted_velocity,
            'predicted_disturbance': 0.0,  # 목표 위상이 없으면 0
            'confidence': confidence
        }
    
    def apply_predictive_correction(
        self,
        target_phase: Optional[float] = None,
        prediction_horizon_ms: float = 100.0
    ) -> RingState:
        """
        선제적 보정 적용
        
        Pre-emptive Inhibition: 미래 위상에서 예상되는 외란(Disturbance)이나
        궤도 이탈 징후를 발견하면, 실제 하우징에 닿기 전에 반대 방향의
        억제 신호를 먼저 생성합니다.
        
        Args:
            target_phase: 목표 위상 (None이면 현재 center 사용)
            prediction_horizon_ms: 예측 시간 간격 [ms]
        
        Returns:
            RingState: 보정 후 상태
        """
        current_state = self.get_state()
        
        if target_phase is None:
            target_phase = current_state.center
        
        # 미래 위상 예측
        prediction = self.predict_future_phase(prediction_horizon_ms)
        predicted_phase = prediction['predicted_phase']
        
        # 외란 예측
        # disturbance_predicted = |φ_predicted(t+Δt) - φ_target(t+Δt)|
        # 목표 위상도 미래로 예측 (현재 목표가 유지된다고 가정)
        predicted_target = target_phase
        predicted_disturbance = abs(predicted_phase - predicted_target)
        
        # Ring 구조 고려 (최단 거리)
        if predicted_disturbance > self.size / 2:
            predicted_disturbance = self.size - predicted_disturbance
        
        # 선제적 보정 강도 계산
        # correction_strength = min(1.0, disturbance_predicted / threshold)
        correction_strength = min(1.0, predicted_disturbance / self._disturbance_threshold)
        
        # 선제적 억제 방향 계산
        # correction_direction = sign(φ_target(t+Δt) - φ_predicted(t+Δt))
        phase_diff = predicted_target - predicted_phase
        if phase_diff > self.size / 2:
            phase_diff = phase_diff - self.size
        elif phase_diff < -self.size / 2:
            phase_diff = phase_diff + self.size
        
        if abs(phase_diff) < 0.1:
            correction_direction = 0
        else:
            correction_direction = 1 if phase_diff > 0 else -1
        
        # 보정이 필요한 경우
        if correction_strength > 0.1 and correction_direction != 0:
            # 보정 방향으로 약한 입력 주입
            correction_strength_actual = correction_strength * self._max_correction_force
            
            # 보정 방향 계산 (neuron 인덱스)
            current_center_idx = int(current_state.center) % self.size
            if correction_direction > 0:
                correction_idx = (current_center_idx + 1) % self.size
            else:
                correction_idx = (current_center_idx - 1) % self.size
            
            # 선제적 보정 적용
            self.inject(direction_idx=correction_idx, strength=correction_strength_actual)
            
            # 짧은 시간 실행 (보정 적용)
            self.run(duration_ms=10.0)
            
            # 입력 제거
            self.release_input()
            
            if self.debug:
                print(f"[DEBUG] Predictive Correction: "
                      f"disturbance={predicted_disturbance:.2f}, "
                      f"strength={correction_strength:.2f}, "
                      f"direction={correction_direction}")
        
        # 최종 상태 반환
        return self.get_state()
    
    def run_with_predictive_control(
        self,
        duration_ms: float,
        target_phase: Optional[float] = None,
        prediction_horizon_ms: float = 100.0
    ) -> RingState:
        """
        예측 제어를 포함한 실행
        
        Predictive Drift Control을 포함하여 실행합니다.
        각 단계에서 미래 위상을 예측하고, 필요시 선제적 보정을 적용합니다.
        
        Args:
            duration_ms: 실행 시간 [ms]
            target_phase: 목표 위상 (None이면 현재 center 사용)
            prediction_horizon_ms: 예측 시간 간격 [ms]
        
        Returns:
            RingState: 최종 상태
        """
        if target_phase is None:
            initial_state = self.get_state()
            target_phase = initial_state.center
        
        # 예측 제어 주기 (예: 10ms마다 예측 및 보정)
        prediction_interval = 10.0  # ms
        num_predictions = max(1, int(duration_ms / prediction_interval))
        interval_duration = duration_ms / num_predictions
        
        for i in range(num_predictions):
            # 예측 및 선제적 보정
            self.apply_predictive_correction(
                target_phase=target_phase,
                prediction_horizon_ms=prediction_horizon_ms
            )
            
            # 나머지 시간 실행
            if interval_duration > 0:
                self.run(duration_ms=interval_duration)
        
        return self.get_state()
    
    def get_orbit_control(
        self,
        target_phase: Optional[float] = None
    ) -> 'OrbitControl':
        """
        궤도 제어 명령 반환 (마모 최소화 엔진)
        
        이 엔진은 "동력 엔진"이 아니라 "회전 상태 보존 엔진"입니다.
        
        핵심 개념:
        - 힘 제어 ❌
        - 토크 극대화 ❌
        - 궤도 유지 ⭕
        - 편차 복원 ⭕
        
        Args:
            target_phase: 목표 위상 (None이면 현재 center 사용)
        
        Returns:
            OrbitControl: 궤도 제어 명령
                - maintain_orbit: 궤도 유지 명령
                - restore_deviation: 편차 복원 명령
                - deviation_magnitude: 편차 크기
                - correction_direction: 보정 방향
                - wear_risk: 마모 위험도
        """
        try:
            from .wear_minimization import OrbitControl, calculate_orbit_control
        except ImportError:
            from wear_minimization import OrbitControl, calculate_orbit_control
        
        state = self.get_state()
        current_phase = state.center
        
        if target_phase is None:
            target_phase = current_phase
        
        physics_output = self.get_physics_output(target_phase)
        
        return calculate_orbit_control(
            current_phase=current_phase,
            target_phase=target_phase,
            deviation_error=physics_output.deviation_error,
            orbit_stability=physics_output.orbit_stability,
            drift=state.drift,
            size=self.size
        )
    
    def to_dict(self) -> Dict:
        """상태를 dict로 변환"""
        state = self.get_state()
        return {
            "center": state.center,
            "width": state.width,
            "active_count": state.active_count,
            "stability": state.stability,
            "drift": state.drift,
            "sustained": state.sustained,
            "size": self.size,
            "config": self.config
        }

