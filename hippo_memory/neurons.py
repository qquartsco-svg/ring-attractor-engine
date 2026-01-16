"""
Hippocampus Memory Engine - Neurons Module

Neuron classes for hippocampal circuit simulation
"""

import numpy as np
from typing import Optional
from v4_contracts.spike_event import SpikeEvent
from v4_engines.hh_soma_quick_v4 import HHSomaQuickV4
from .config import CONFIG

# 호환성을 위해 별칭 사용
HHSomaV4 = HHSomaQuickV4

class DGNeuronV4:
    """
    [Dentate Gyrus: Pattern Separation through Sparse Coding] (V4)
    
    V4 네트워킹 구조를 사용하는 DG 뉴런
    """
    def __init__(self, name, activation_threshold=0.8):
        self.name = name
        # HHSomaV4 사용
        cfg = CONFIG["HH"].copy()
        self.soma = HHSomaV4(cfg, neuron_id=name, axon_delay=2.0)
        self.activation_threshold = activation_threshold
        self.S, self.PTP = 0.0, 1.0
    
    def step(self, dt, t, I_ext=0.0, ATP=100.0):
        """V4 스타일 step (전역 시간 t 필수) - NeuronNetwork 호환"""
        # 역치 이상일 때만 활성화
        if I_ext > self.activation_threshold * 300.0:
            result = self.soma.step(dt, t, I_ext + self.soma.I_syn, ATP)
        else:
            result = self.soma.step(dt, t, self.soma.I_syn, ATP)  # 억제
        
        self.soma.I_syn = 0.0  # 사용 후 초기화
        
        sp = result.get("spiking", False) or (self.soma.V > self.soma.spike_thresh)
        
        if sp:
            self.S = min(1.0, self.S + 0.3)
            self.PTP = min(2.0, self.PTP + 0.05)
        else:
            self.S = max(0.0, self.S - 0.01)
            self.PTP = max(1.0, self.PTP - 0.001)
        
        return sp, self.S, self.PTP
    
    def handle_event(self, event: SpikeEvent):
        """V4 이벤트 처리"""
        self.soma.handle_event(event)
    
    def emit_spike(self, t=None):
        """V4 스파이크 이벤트 생성 (t는 호환성을 위해 받지만 사용 안 함)"""
        return self.soma.emit_spike()

class CA3NeuronV4:
    """
    [CA3: Associative Memory with Recurrent Connections] (V4.2)
    
    ⭐ V4.2 업그레이드:
    - Baseline Depolarization: 막전위 -60mV 유지
    - Temporal Summation: I_syn 누적 메커니즘
    
    ⭐ CA3 Recurrent Memory:
    - CA3 → CA3 재귀 연결을 통한 패턴 완성(Pattern Completion)
    - 부분 패턴에서 전체 패턴 복원 가능
    - Attractor dynamics (재귀 연결을 통한 안정적 패턴 유지)
    
    V4 네트워킹 구조를 사용하는 CA3 뉴런
    """
    def __init__(self, name):
        self.name = name
        cfg = CONFIG["HH"].copy()
        # ⭐ 최적화: spike_thresh를 -34mV로 조정 (발열 해결 최종)
        # -35mV: CA3_0이 여전히 146 spikes (발열 발생)
        # -34mV: 적정 spike (목표 3~10/뉴런) → 발열 없음
        cfg["spike_thresh"] = -34.0
        self.soma = HHSomaV4(cfg, neuron_id=name, axon_delay=2.0)
        self.base_spike_thresh = -34.0  # ⭐ V4.3: 기본 임계값 저장 (Spike Budget용)
        self.S, self.PTP = 0.0, 1.0
        self.wake_spike_count = 0
        # ⭐ CA3 Recurrent Memory: 재귀 연결을 통한 패턴 완성 추적
        self.recurrent_activation_count = 0  # 재귀 연결로 인한 활성화 횟수
        
        # ⭐ V4.2: Baseline Depolarization
        # ⭐ 케이스 2: baseline_V -55.0mV (saturation 감소)
        # -50mV: 쉽게 발화 → Global Saturation
        # -55mV: 발화 임계 조절 → Ring Attractor 형성 가능
        self.baseline_V = -55.0  # -50.0 → -55.0 (케이스 2: saturation 감소)
        self.soma.V = self.baseline_V  # 초기값 설정
        
        # ⭐ V4.2: Temporal Summation (I_syn 누적)
        self.I_syn_accumulated = 0.0  # 누적된 시냅스 전류
        # ⭐ 2️⃣ 파라미터 튜닝: 15.0ms → 20.0ms (재귀 입력 시간적 누적 강화)
        self.tau_syn = 20.0  # 시냅스 전류 감쇠 시간 상수 [ms]
        
        # ⭐ V4.4: Ring Attractor 관성 (느리게 감쇠되는 상태 변수)
        # 목적: Pattern Completion → Ring Attractor 전환
        # 역할: "한 번 돌기 시작하면 계속 도는 힘, 관성, 팽이"
        # - spike 발생 시 증가, 매우 느리게 감쇠 (tau_recurrent = 200ms)
        # - 이전 상태가 다음 상태를 밀어주는 변수
        # ⚠️ 주의: 재귀 연결로 인한 활성화일 때만 증가 (Global Saturation 방지)
        self.recurrent_state = 0.0  # Ring Attractor 관성 상태 (0.0 ~ 1.0)
        self.tau_recurrent = 200.0  # 관성 감쇠 시간 상수 [ms] (매우 느림)
        self.recurrent_gain = 0.08  # spike 발생 시 관성 증가량 (0.15 → 0.08: Global Saturation 방지)
        
        # ⭐ V4.3: Spike Budget (Homeostasis) - 과흥분 억제 메커니즘 (발열 해결)
        # ⭐ 엔진 승격: Homeostasis 기본값 고정 (V4.3 기준)
        # 🔴 Step 2: Homeostasis 2단계 분리 (Ring Attractor를 위한 조정)
        # Fast (Spike): 짧은 불응기 정도만 보장 (개별 스파이크 형태 유지)
        # Slow (Budget): 수백 ms 단위 평균 발화율 감시 (전체 시스템 과열 방지)
        self.spike_budget = 5  # ⚠️ 수정: Homeostasis 강화 (15 → 8 → 5: Global Saturation 방지)
        self.spike_count_window = 0  # 현재 window 내 spike 수
        self.dynamic_threshold_penalty = 0.0  # 동적 임계값 증가 (mV)
        self.penalty_increase_rate = 2.5  # ⚠️ 수정: Homeostasis 강화 (1.2 → 2.5: Global Saturation 방지)
        self.window_start_time = 0.0  # 현재 window 시작 시간 [ms]
        self.window_duration = 80.0  # ⚠️ 최종 조정: 60.0 → 80.0ms (penalty 빈도 감소, drift 부드러운 이동)
        
        # ⭐ 외부 전류 주입 (I_ext) - inject_current()에서 설정됨
        self.I_ext = 0.0
    
    def step(self, dt, t, I_ext=0.0, ATP=100.0):
        """
        V4.3 스타일 step (Temporal Summation + Spike Budget 포함)
        
        ⭐ V4.2 업그레이드:
        - Temporal Summation: I_syn 누적 및 감쇠
        - Baseline Depolarization: 막전위 -50mV 유지
        
        ⭐ V4.3 업그레이드 (Homeostasis):
        - Spike Budget: window 내 max spike 수 제한
        - Dynamic Threshold: 초과 시 임계값 동적 상승
        
        ⭐ CA3 Recurrent Memory:
        - 재귀 연결로 인한 I_syn은 자동으로 포함됨
        - 부분 패턴 입력 → 재귀 연결 → 전체 패턴 완성
        """
        # ⭐ V4.3: Window 리셋 (60ms마다)
        if (t - self.window_start_time) >= self.window_duration:
            self.spike_count_window = 0
            self.dynamic_threshold_penalty = 0.0
            self.soma.spike_thresh = self.base_spike_thresh
            self.window_start_time = t
        
        # ⭐ V4.2: Temporal Summation
        # 1. 새로운 시냅스 입력을 누적
        # ⭐ STEP 3: I_syn 값을 저장 (spike 발생 시 recurrent_state 증가에 사용)
        current_I_syn = self.soma.I_syn  # 저장 (초기화 전에)
        self.I_syn_accumulated += current_I_syn
        
        # 2. 기존 누적 전류 감쇠 (exponential decay)
        decay_factor = np.exp(-dt / self.tau_syn)
        self.I_syn_accumulated *= decay_factor
        
        # ⭐ V4.4: Ring Attractor 관성 감쇠 (매우 느리게)
        # 목적: 이전 상태가 다음 상태를 밀어주는 힘 유지
        recurrent_decay_factor = np.exp(-dt / self.tau_recurrent)
        self.recurrent_state *= recurrent_decay_factor
        
        # 3. 누적된 전류를 사용
        # ⭐ 외부 전류: 파라미터 I_ext 또는 self.I_ext 사용 (inject_current()에서 설정)
        # ⭐ NeuronNetwork.tick()에서 I_ext를 전달하지 않으므로 self.I_ext 사용
        effective_I_ext = self.I_ext if I_ext == 0.0 else I_ext
        # ⭐ STEP 3: Ring Attractor 관성을 I_total에 추가 (관성 = 이전 상태가 다음 상태를 밀어주는 힘)
        # recurrent_state는 0.0~1.0 범위이므로, 적절한 스케일링 필요
        # ⚠️ STEP 3: Phase 2에서 bump 유지를 위해 스케일링 증가 (40.0 → 60.0)
        # Phase 2에서 I_ext=0일 때 recurrent_current + I_syn_accumulated로 임계값 도달
        # V=-53.5mV에서 -34.0mV까지 상승하려면 약 20mV 전류 필요
        # recurrent_state=0.8일 때 recurrent_current=48.0이면 충분
        # ⭐ STEP 2-B 골든레인지 탐색: Pure Ring Attractor 구현
        # 목표: recurrent_current는 bump 유지만 담당, noise가 drift 주도
        # ⚠️ 골든레인지 탐색 중: 파라미터 조정 가능
        # 테스트 조합:
        #   1. recurrent=20.0, noise=0.10 (보수적)
        #   2. recurrent=25.0, noise=0.10 (현재 recurrent, 작은 noise)
        #   3. recurrent=25.0, noise=0.15 (현재 설정)
        #   4. recurrent=30.0, noise=0.10 (recurrent 증가)
        #   5. recurrent=30.0, noise=0.15 (recurrent 증가, 현재 noise)
        # ⚠️ 골든레인지 탐색: 조합별 테스트
        # 조합 1: recurrent=20.0, noise=0.10 (보수적)
        # 조합 2: recurrent=25.0, noise=0.10 (현재)
        # 조합 3: recurrent=25.0, noise=0.15
        # 조합 4: recurrent=30.0, noise=0.10
        # ⚠️ 골든레인지 탐색 완료: 최적 조합 결정
        # 조합 1: recurrent=20.0, noise=0.10 → 평균 9.3/15 (과다)
        # 조합 2: recurrent=18.0, noise=0.10 → 평균 9.0/15 (과다)
        # 조합 3: recurrent=18.0, noise=0.08 → 평균 7.7/15 ✅ **최적**
        # 조합 4: recurrent=15.0, noise=0.08 → 평균 9.7/15 (악화, bump 유지 실패)
        # 결론: recurrent=18.0, noise=0.08이 가장 균형잡힌 설정
        # ⚠️ 최종 단계: Drift 속도 감소 (6.5 → 0.5~2.0 neuron)
        # 현재 상태: Ring Attractor 작동 중, drift 크기 조정 필요
        # 목표: drift ≤ 2.0 neuron / 150ms
        # 조정: recurrent_current 스케일 감소로 drift inertia 유지하면서 속도만 감소
        RECURRENT_SCALE = 4.8  # 최적값 복구 (4.5는 활성화 악화)
        NOISE_STD = 0.0018  # 최적값 복구 (0.0015는 활성화 악화)
        # ⚠️ 최종 최적 조합 (튜닝 결과):
        # 활성화: 평균 5.0/15 ✅ (목표 달성, 완벽!)
        # Drift: 평균 4.64~6.00 neuron (목표 0.5~2.0보다 큼)
        # 개별 drift: 1.27, 3.42 neuron 관측됨 (변동성 있음)
        # RECURRENT_SCALE < 4.8 또는 NOISE_STD < 0.0018: 활성화 악화
        
        recurrent_current = self.recurrent_state * RECURRENT_SCALE  # 관성을 전류로 변환
        I_total = effective_I_ext + self.I_syn_accumulated + recurrent_current
        
        # ⭐ STEP 2-B: Noise 기반 Diffusion (Ring Attractor drift)
        # 작은 zero-mean Gaussian noise를 주입하여 자연스러운 bump drift 유도
        # 노이즈는 bump 형태를 유지하면서 위치만 천천히 이동시킴
        # ⚠️ 외부 입력이 없을 때만 노이즈 적용 (Phase 2에서 drift 발생)
        if effective_I_ext <= 0.0:  # 외부 입력이 없을 때만
            # ⚠️ 골든레인지 탐색: 노이즈 강도 조정 가능
            # 목표: spike 수 거의 일정, 활성 뉴런 수 고정 (5~7), center만 천천히 이동
            # σ < threshold: noise가 threshold를 직접 넘기지 않고, 미세한 전위 변화만 유도
            # 이 미세한 변화가 시간에 누적되어 연속적인 drift 발생
            drift_noise = np.random.normal(0.0, NOISE_STD)  # zero-mean Gaussian noise (stochastic diffusion)
            I_total += drift_noise
        
        # ⭐ 디버깅: DOG/BAT cue 버그 해결 (I_ext 전달 확인)
        if (self.name.startswith('CA3_DOG_0') or self.name.startswith('CA3_BAT_0')) and effective_I_ext > 0:
            if t < 10.0:  # 처음 10ms만 출력
                print(f"[DEBUG] step: {self.name} t={t:.2f}ms, I_ext={self.I_ext:.1f}, effective_I_ext={effective_I_ext:.1f}, I_total={I_total:.1f}, V={self.soma.V:.2f}mV, thresh={self.soma.spike_thresh:.2f}mV")
        
        # ⭐ 디버깅: I_ext 전달 확인 (CA3_CAT_0만) - 주석 처리 (sp 변수 정의 전 사용 방지)
        # if self.name.startswith('CA3_CAT_0') and effective_I_ext > 0:
        #     if t < 10.0:  # 처음 10ms만 출력
        #         print(f"[DEBUG] step: {self.name} t={t:.2f}ms, I_ext={self.I_ext:.1f}, effective_I_ext={effective_I_ext:.1f}, I_total={I_total:.1f}, V={self.soma.V:.2f}mV")
        
        # ⭐ V4.2: Baseline Depolarization 유지
        # 막전위가 너무 낮아지면 baseline으로 복원
        # ⚠️ 수정: baseline_V보다 낮으면 즉시 복원 (이전: baseline_V-5.0)
        # ⚠️ 중요: step() 시작 전에 V를 확인하여 baseline_V보다 낮으면 복원
        # ⚠️ STEP 1: I_ext가 있을 때는 baseline_V 복원을 건너뛰기 (cue 버그 해결)
        # ⚠️ 수정: effective_I_ext > 0이면 baseline_V 복원 건너뛰기
        if self.soma.V < self.baseline_V and effective_I_ext <= 0.0:
            self.soma.V = self.baseline_V
        
        # ⭐ V4.3: Spike Budget 체크 (엔진 승격: Homeostasis 기본값 고정)
        # Budget 초과 시 임계값 동적으로 상승 (억제) + 예방적 억제
        # ⚠️ 수정: Budget 초과 시 더 강력한 억제
        if self.spike_count_window >= self.spike_budget:
            # Budget 초과: 임계값을 더 가파르게 상승
            self.dynamic_threshold_penalty += self.penalty_increase_rate * dt  # dt 고려
        effective_thresh = self.soma.spike_thresh + self.dynamic_threshold_penalty
        
        # ⚠️ 수정: effective_thresh를 soma에 임시 적용 (Homeostasis 작동 보장)
        original_thresh = self.soma.spike_thresh
        self.soma.spike_thresh = effective_thresh
        
        result = self.soma.step(dt, t, I_total, ATP)
        
        # ⚠️ 수정: step() 후 원래 임계값 복원 (다음 스텝에서 다시 계산)
        self.soma.spike_thresh = original_thresh
        
        # ⭐ V4.2: Baseline Depolarization 유지 (step() 후에도 확인)
        # HHSomaQuick.step()에서 V가 낮아질 수 있으므로 다시 확인
        # ⚠️ STEP 1: I_ext가 있을 때는 baseline_V 복원을 건너뛰기 (cue 버그 해결)
        # ⚠️ 수정: effective_I_ext > 0이면 baseline_V 복원 건너뛰기
        if self.soma.V < self.baseline_V and effective_I_ext <= 0.0:
            self.soma.V = self.baseline_V
            # ⚠️ STEP 3: recurrent_state 증가는 spike 발생 시 처리 (위로 이동)
        
        # ⭐ V4.2: I_syn은 Temporal Summation에서 사용했으므로 초기화
        self.soma.I_syn = 0.0
        
        # ⭐ V4.3: Spike 체크 (동적 임계값 사용)
        sp = result.get("spiking", False) or (self.soma.V > effective_thresh)
        
        # ⭐ 발열 해결: Budget 초과 시 즉시 발화 차단 (spike 발생 전에 체크)
        # Budget 초과 시 spike 발생 자체를 막음 (발열 해결 핵심)
        if sp and self.spike_count_window >= self.spike_budget:
            sp = False  # Budget 초과 시 즉시 발화 차단
        
        if sp:
            # Budget 미초과 시에만 spike 처리
            self.spike_count_window += 1
            
            # 🔴 Step 2: Homeostasis 2단계 분리 (천천히 억제)
            # Bump 형성 전 집단 spike 허용, 이후 천천히 억제
            if self.spike_count_window == int(self.spike_budget * 0.3) + 1:  # 30% 도달 시
                # ✅ 증가 속도 1/10 감소 (12.0 → 1.2)
                if self.dynamic_threshold_penalty < self.penalty_increase_rate:
                    self.dynamic_threshold_penalty = self.penalty_increase_rate  # += 대신 = (한 번만 설정, 누적 방지)
                    self.soma.spike_thresh = self.base_spike_thresh + self.dynamic_threshold_penalty
                    self.soma.V -= 8.0
            
            # 🔴 Step 2: Budget 초과 시 강력한 억제 (추가 안전장치 - 발생하지 않아야 함)
            # ✅ 증가 속도 감소 (25.0 → 2.5)
            if self.spike_count_window == self.spike_budget + 1:  # budget 초과 시점에만 (한 번만)
                self.dynamic_threshold_penalty = self.penalty_increase_rate * 2.0  # += 대신 = (누적 방지)
                self.soma.spike_thresh = self.base_spike_thresh + self.dynamic_threshold_penalty
                self.soma.V -= 30.0
            
            # ⭐ STEP 3: Spike 발생 시 recurrent_state 증가 (Ring Attractor 관성 축적)
            # Phase 1에서 관성을 쌓아서 Phase 2에서 사용할 수 있도록
            # ⚠️ STEP 3: 임시 실험 - 모든 spike에 대해 관성 증가 (재귀 연결 여부와 관계없이)
            # 목적: Phase 1에서 관성을 쌓아서 Phase 2에서 사용
            # TODO: 나중에 재귀 연결로 인한 활성화일 때만 증가하도록 수정
            self.recurrent_state = min(1.0, self.recurrent_state + self.recurrent_gain)
            # 재귀 연결 추적 (디버깅용)
            if current_I_syn > 0.1:
                self.recurrent_activation_count += 1
            
            self.S = min(1.0, self.S + 0.3)
            self.PTP = min(2.0, self.PTP + 0.05)
            self.wake_spike_count += 1
            # ⚠️ V4.4 수정: 관성 증가는 재귀 연결로 인한 활성화일 때만 (위에서 처리)
        else:
            self.S = max(0.0, self.S - 0.01)
            self.PTP = max(1.0, self.PTP - 0.001)
        
        return sp, self.S, self.PTP
    
    def handle_event(self, event: SpikeEvent):
        """V4 이벤트 처리"""
        self.soma.handle_event(event)
    
    def emit_spike(self, t=None):
        """V4 스파이크 이벤트 생성 (t는 호환성을 위해 받지만 사용 안 함)"""
        return self.soma.emit_spike()

class CA1TimeCellV4:
    """
    [CA1 Time Cells: Temporal Sequence Encoding] (V4)
    
    ⚠️  한계: trigger() 메서드가 정의되어 있으나 호출 경로가 없음
    - CA3 → CA1 연결은 있으나, CA3 스파이크 시 trigger()를 호출하는 로직 없음
    - 현재는 I_syn으로만 입력받고, trigger_time이 None이면 delay 발화 로직이 작동 안 함
    - "CA1 Time working" 단정 불가
    """
    def __init__(self, delay_ms, name):
        self.delay_ms = delay_ms
        self.name = name
        cfg = CONFIG["HH"].copy()
        self.soma = HHSomaV4(cfg, neuron_id=name, axon_delay=2.0)
        self.trigger_time = None  # ⚠️  trigger() 호출 경로 없어서 항상 None일 가능성
        self.S, self.PTP = 0.0, 1.0
    
    def trigger(self, t):
        """
        CA3에서 신호 받으면 타이머 시작
        
        ⭐ Phase 2: handle_event()에서 자동 호출됨
        - CA3 스파이크 시 trigger_time 설정
        - delay 발화 로직 작동 가능
        """
        if self.trigger_time is None:
            self.trigger_time = t
    
    def step(self, dt, t, I_ext=0.0, ATP=100.0):
        """시간이 되면 자동 발화"""
        if self.trigger_time is not None:
            elapsed = t - self.trigger_time
            if abs(elapsed - self.delay_ms) < 2.0:
                I_ext += 200.0
        
        result = self.soma.step(dt, t, I_ext + self.soma.I_syn, ATP)
        self.soma.I_syn = 0.0
        
        sp = result.get("spiking", False) or (self.soma.V > self.soma.spike_thresh)
        
        if sp:
            self.S = min(1.0, self.S + 0.3)
            self.PTP = min(2.0, self.PTP + 0.05)
        else:
            self.S = max(0.0, self.S - 0.01)
            self.PTP = max(1.0, self.PTP - 0.001)
        
        return sp, self.S, self.PTP
    
    def handle_event(self, event: SpikeEvent):
        """
        V4 이벤트 처리
        
        ⭐ Phase 2: CA3 스파이크 시 자동 trigger
        - CA3에서 온 이벤트면 trigger_time 설정
        - delay 발화 로직 작동 가능하게 함
        """
        self.soma.handle_event(event)
        # ⭐ Phase 2: CA3 스파이크 시 자동 trigger
        if event.source_id.startswith('CA3_'):
            self.trigger(event.t)
    
    def emit_spike(self, t=None):
        """V4 스파이크 이벤트 생성 (t는 호환성을 위해 받지만 사용 안 함)"""
        return self.soma.emit_spike()

class CA1NoveltyDetectorV4:
    """
    [CA1 Novelty Detection: Comparator Function] (V4)
    """
    def __init__(self, name):
        self.name = name
        cfg = CONFIG["HH"].copy()
        self.soma = HHSomaV4(cfg, neuron_id=name, axon_delay=2.0)
        self.expected_patterns = []
        self.novelty_threshold = 0.5
        self.S, self.PTP = 0.0, 1.0
    
    def learn_pattern(self, pattern_name):
        """패턴 학습"""
        if pattern_name not in self.expected_patterns:
            self.expected_patterns.append(pattern_name)
    
    def compute_novelty(self, pattern_name):
        """새로움 점수"""
        if pattern_name in self.expected_patterns:
            return 0.0
        else:
            return 1.0
    
    def step(self, dt, t, I_ext=0.0, ATP=100.0, pattern_name=None):
        """Novelty에 비례하여 발화 (NeuronNetwork 호환)"""
        # pattern_name이 없으면 기본값 사용 (나중에 외부에서 설정 가능)
        if pattern_name is None:
            pattern_name = getattr(self, '_current_pattern', 'UNKNOWN')
        
        novelty_score = self.compute_novelty(pattern_name)
        
        if novelty_score > self.novelty_threshold:
            I_ext += 200.0 * novelty_score
        
        result = self.soma.step(dt, t, I_ext + self.soma.I_syn, ATP)
        self.soma.I_syn = 0.0
        
        sp = result.get("spiking", False) or (self.soma.V > self.soma.spike_thresh)
        
        if sp:
            self.S = min(1.0, self.S + 0.3)
            self.PTP = min(2.0, self.PTP + 0.05)
        else:
            self.S = max(0.0, self.S - 0.01)
            self.PTP = max(1.0, self.PTP - 0.001)
        
        return sp, novelty_score
    
    def set_pattern(self, pattern_name):
        """현재 패턴 설정 (외부에서 호출)"""
        self._current_pattern = pattern_name
    
    def handle_event(self, event: SpikeEvent):
        """V4 이벤트 처리"""
        self.soma.handle_event(event)
    
    def emit_spike(self, t=None):
        """V4 스파이크 이벤트 생성 (t는 호환성을 위해 받지만 사용 안 함)"""
        return self.soma.emit_spike()

class SubiculumGateV4:
    """
    [Subiculum: Context-Dependent Output Gating] (V4)
    
    ⚠️  한계: step() 메서드가 없음
    - NeuronNetwork.tick()은 모든 뉴런의 step()을 호출하는 구조
    - step()이 없으면 실행되지 않거나 예외 발생 가능
    - 현재는 데이터 구조(맥락 관련성 계산)로만 존재, 회로 요소로 작동 안 함
    - "Subiculum working" 단정 불가
    """
    def __init__(self, name):
        self.name = name
        cfg = CONFIG["HH"].copy()
        self.soma = HHSomaV4(cfg, neuron_id=name, axon_delay=2.0)
        self.context_memory = {}
        self.current_context = None
        self.S, self.PTP = 0.0, 1.0
    
    def set_context(self, context):
        """맥락 설정"""
        self.current_context = context
    
    def learn_context_association(self, context, word):
        """맥락-단어 연관 학습"""
        if context not in self.context_memory:
            self.context_memory[context] = []
        if word not in self.context_memory[context]:
            self.context_memory[context].append(word)
    
    def compute_relevance(self, word):
        """맥락 관련성"""
        if self.current_context is None:
            return 0.5
        
        if self.current_context in self.context_memory:
            relevant_words = self.context_memory[self.current_context]
            if word in relevant_words:
                return 1.0
            else:
                return 0.0
        
        return 0.5
    
    def gate(self, word, ca_input):
        """출력 게이팅"""
        relevance = self.compute_relevance(word)
        return ca_input * relevance
    
    def step(self, dt, t, I_ext=0.0, ATP=100.0):
        """
        V4 스타일 step (NeuronNetwork 호환)
        
        ⭐ Phase 2: Subiculum은 게이팅 로직만 수행
        - 실제 발화는 하지 않음 (게이팅 로직만)
        """
        # 맥락 관련성 계산 (게이팅 로직)
        # 실제 발화는 하지 않고 게이팅 상태만 갱신
        result = self.soma.step(dt, t, I_ext + self.soma.I_syn, ATP)
        self.soma.I_syn = 0.0
        
        # 발화는 하지 않음 (게이팅만)
        return False, self.S, self.PTP  # (spike=False, S, PTP)
    
    def handle_event(self, event: SpikeEvent):
        """V4 이벤트 처리"""
        self.soma.handle_event(event)
    
    def emit_spike(self, t=None):
        """V4 스파이크 이벤트 생성 (t는 호환성을 위해 받지만 사용 안 함)"""
        return self.soma.emit_spike()
