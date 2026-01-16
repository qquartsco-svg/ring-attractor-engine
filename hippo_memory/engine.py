"""
Hippocampus Memory Engine - Main Engine Module

Hippocampus memory system engine for pattern learning and recall
"""

from typing import Dict, Any, Optional
from v4_networking.neuron_network import NeuronNetwork
from .neurons import (
    DGNeuronV4,
    CA3NeuronV4,
    CA1TimeCellV4,
    CA1NoveltyDetectorV4,
    SubiculumGateV4
)
from .spatial_neurons import CA3PlaceCellV4, PlaceField
from .contracts import (
    MemoryEvent,
    MemoryPattern,
    MemoryContext,
    RecallResult,
    TrainingResult,
    ConsolidationResult,
    MemoryStatus
)

class HippoMemoryV4System:
    """
    V4 네트워킹 구조를 사용한 히포 메모리 시스템
    
    V4 계약:
    - NeuronNetwork 사용
    - EventQueue 사용
    - 전역 시간 t 기반 실행
    - 기억 형성 기능 (V4 승격)
    
    ⭐ 엔진 인터페이스 (Engine Interface):
    - 결과 계약: 모든 실행 함수가 구조화된 dict 반환
    - 학습 스위치: self.learn = False/True (STDP 활성화/비활성화)
    """
    def __init__(self, dt=0.1, learn=True):
        self.dt = dt
        self.t = 0.0  # 전역 시간 [ms] ⭐ V4
        self.network = NeuronNetwork()
        
        # ⭐ 엔진 인터페이스: 학습 스위치 (STDP 활성화/비활성화)
        self.learn = learn  # True: STDP 활성화, False: STDP 비활성화 (기억 형성 차단)
        
        # 뉴런 딕셔너리
        self.dg_neurons = {}
        self.ca3_neurons = {}
        self.ca1_time_cells = {}
        self.ca1_novelty = None
        self.subiculum_gates = {}
        
        # ⭐ V4 Memory Formation: 시냅스 가중치는 Synapse 내부에 있음
        # 외부 dict 제거 - 기억은 엔진 내부에만 존재
        
        # ⭐ 엔진 인터페이스: 결과 저장소
        self._results = {}  # 최종 결과 저장
    
    def add_dg_neuron(self, name, activation_threshold=0.8):
        """DG 뉴런 추가"""
        neuron = DGNeuronV4(name, activation_threshold)
        self.network.add_neuron(name, neuron)
        self.dg_neurons[name] = neuron
        return neuron
    
    def add_ca3_neuron(self, name, place_field=None):
        """CA3 뉴런 추가
        
        Args:
            name: 뉴런 이름
            place_field: PlaceField 객체 (None이면 일반 CA3 뉴런)
        """
        if place_field is not None:
            # Place Cell로 생성
            neuron = CA3PlaceCellV4(name, place_field)
        else:
            # 일반 CA3 뉴런
            neuron = CA3NeuronV4(name)
        
        self.network.add_neuron(name, neuron)
        self.ca3_neurons[name] = neuron
        return neuron
    
    def add_ca1_time_cell(self, name, delay_ms):
        """CA1 Time Cell 추가"""
        neuron = CA1TimeCellV4(delay_ms, name)
        self.network.add_neuron(name, neuron)
        self.ca1_time_cells[name] = neuron
        return neuron
    
    def add_ca1_novelty(self, name):
        """CA1 Novelty Detector 추가"""
        neuron = CA1NoveltyDetectorV4(name)
        self.network.add_neuron(name, neuron)
        self.ca1_novelty = neuron
        return neuron
    
    def add_subiculum_gate(self, name):
        """Subiculum Gate 추가"""
        neuron = SubiculumGateV4(name)
        self.network.add_neuron(name, neuron)
        self.subiculum_gates[name] = neuron
        return neuron
    
    def add_connection(self, source_id, target_id, weight=1.0, delay=2.0, synapse_type="excitatory"):
        """시냅스 연결 추가 (V4) - weight는 Synapse 내부에 저장됨
        
        ⭐ 엔진 인터페이스: 학습 스위치 전달
        - system.learn 값이 Synapse.learn_enabled에 전달됨
        - synapse_type: "excitatory" | "inhibitory" (V4.4 억제 시냅스 지원)
        """
        self.network.add_connection(source_id, target_id, weight, delay, synapse_type=synapse_type, learn_enabled=self.learn)
        # ⭐ 외부 dict 제거 - weight는 Synapse 객체 내부에 있음
    
    def inject_current(self, neuron_id, current, duration_ms=10.0):
        """외부 전류 주입 (V4)"""
        # ⭐ STEP 1: CA3 True Recall cue 버그 수정
        # network.get_neuron()이 실패할 수 있으므로 ca3_neurons에서 직접 찾기
        neuron = None
        
        # 우선순위 1: ca3_neurons에서 직접 찾기 (True Recall cue 버그 해결)
        if neuron_id in self.ca3_neurons:
            neuron = self.ca3_neurons[neuron_id]
        # 우선순위 2: network에서 찾기
        elif self.network:
            neuron = self.network.get_neuron(neuron_id)
        
        if neuron:
            # ⭐ CA3NeuronV4 등은 I_ext 속성 사용
            if hasattr(neuron, 'I_ext'):
                neuron.I_ext = current
                # ⭐ 디버깅: BAT/DOG cue 버그 해결
                if 'BAT' in neuron_id or 'DOG' in neuron_id:
                    print(f"[DEBUG] inject_current: {neuron_id} I_ext={current} set (t={self.t:.2f}ms)")
            # ⭐ 기존 방식: soma.I_syn에 추가 (하위 호환성)
            elif hasattr(neuron, 'soma'):
                neuron.soma.I_syn += current
        else:
            # ⭐ 디버깅: neuron을 찾지 못한 경우
            print(f"[WARNING] inject_current: neuron '{neuron_id}' not found in ca3_neurons or network!")
            print(f"  Available ca3_neurons keys: {list(self.ca3_neurons.keys())[:5]}...")
    
    def inject_spatial_input(self, x, y, spatial_input_strength=50.0):
        """
        공간 좌표로 Place Cell 활성화 (V4.5 Spatial Memory)
        
        모든 CA3 Place Cell에 공간 입력 주입
        
        Args:
            x: 현재 X 좌표
            y: 현재 Y 좌표
            spatial_input_strength: 공간 입력 강도 스케일링
        """
        for neuron_id, neuron in self.ca3_neurons.items():
            if isinstance(neuron, CA3PlaceCellV4):
                # Place Cell인 경우 공간 입력 계산
                neuron.spatial_input_strength = spatial_input_strength
                neuron.compute_spatial_input(x, y)
            elif hasattr(neuron, 'place_field'):
                # place_field 속성이 있는 경우 (하위 호환성)
                activation = neuron.place_field.activation(x, y)
                if hasattr(neuron, 'spatial_input'):
                    neuron.spatial_input = activation * spatial_input_strength
    
    def run_step(self, dt=None):
        """한 스텝 실행 (V4)"""
        if dt is None:
            dt = self.dt
        
        # V4 계약: tick(dt, t)
        result = self.network.tick(dt, self.t)
        
        # 시간 업데이트
        self.t += dt
        
        return result
    
    def run_phase(self, T_max, I_ext_dict=None):
        """
        페이즈 실행 (Wake/Sleep/Recall)
        
        ⭐ 엔진 인터페이스: 결과 계약 고정
        - dict 반환 (구조화된 결과)
        
        Returns
        -------
        dict
            {
                'phase_type': 'wake' | 'sleep' | 'recall',
                'duration': T_max,
                'spike_counts': {...},
                'events_processed': int,
                'homeostasis': {...},
                'weights': {...}
            }
        """
        phase_start_t = self.t
        total_spikes = {'DG': 0, 'CA3': 0, 'CA1_Time': 0, 'CA1_Novelty': 0, 'Subiculum': 0, 'total': 0}
        total_events = 0
        weight_sum = 0.0
        weight_count = 0
        weight_max = 0.0
        weight_min = 50.0
        
        # ⭐ run_phase() 시작 시 CA3 뉴런의 V를 baseline_V로 강제 설정
        # soft_reset() 후 첫 step() 전에 V가 낮아질 수 있으므로 보장
        for neuron in self.ca3_neurons.values():
            baseline = neuron.baseline_V if hasattr(neuron, 'baseline_V') else -50.0
            if neuron.soma.V < baseline:
                neuron.soma.V = baseline
        
        steps = int(T_max / self.dt)
        
        for k in range(steps):
            current_time = phase_start_t + k * self.dt
            # 외부 자극 주입
            if I_ext_dict:
                for neuron_id, current in I_ext_dict.items():
                    start_time = current.get('start', 0)  # ⭐ 상대 시간 (phase_start_t 기준)
                    end_time = current.get('end', T_max)  # ⭐ 상대 시간
                    # ⭐ 현재 시간이 시작 시간 이후이고 종료 시간 이전이면 I_ext 주입
                    if start_time <= current_time < end_time:
                        self.inject_current(neuron_id, current.get('value', 0.0))
                    elif current_time >= end_time:
                        # ⭐ 시간 범위 밖이면 I_ext를 0으로 리셋
                        # ⭐ STEP 1: ca3_neurons에서 직접 찾기 (True Recall cue 버그 해결)
                        neuron = None
                        if neuron_id in self.ca3_neurons:
                            neuron = self.ca3_neurons[neuron_id]
                        elif self.network:
                            neuron = self.network.get_neuron(neuron_id)
                        if neuron and hasattr(neuron, 'I_ext'):
                            neuron.I_ext = 0.0
            
            # V4 스텝 실행
            result = self.run_step()
            
            # 계층별 spike 계측
            if isinstance(result, dict):
                layer_spikes = result.get('layer_spikes', {})
                total_spikes['DG'] += layer_spikes.get('DG', 0)
                total_spikes['CA3'] += layer_spikes.get('CA3', 0)
                total_spikes['CA1_Time'] += layer_spikes.get('CA1_Time', 0)
                total_spikes['CA1_Novelty'] += layer_spikes.get('CA1_Novelty', 0)
                total_spikes['Subiculum'] += layer_spikes.get('Subiculum', 0)
                total_spikes['total'] += result.get('spikes_emitted', 0)
                total_events += result.get('events_processed', 0)
        
        # 가중치 통계
        for synapse in self.network.get_all_synapses():
            weight_sum += synapse.weight
            weight_count += 1
            weight_max = max(weight_max, synapse.weight)
            weight_min = min(weight_min, synapse.weight)
        
        avg_weight = weight_sum / weight_count if weight_count > 0 else 1.0
        total_neurons = len(self.dg_neurons) + len(self.ca3_neurons) + len(self.ca1_time_cells) + (1 if self.ca1_novelty else 0) + len(self.subiculum_gates)
        avg_spikes_per_neuron = total_spikes['total'] / total_neurons if total_neurons > 0 else 0.0
        
        # ⭐ 엔진 인터페이스: 구조화된 dict 반환
        return {
            'phase_type': 'unknown',
            'duration': T_max,
            'start_time': phase_start_t,
            'end_time': self.t,
            'spike_counts': total_spikes,
            'events_processed': total_events,
            'homeostasis': {
                'budget_exceeded': 0,  # V4.2에는 Homeostasis 없음
                'avg_spikes_per_neuron': avg_spikes_per_neuron,
                'total_neurons': total_neurons
            },
            'weights': {
                'avg_weight': avg_weight,
                'max_weight': weight_max,
                'min_weight': weight_min,
                'total_synapses': weight_count
            }
        }
    
    def get_synapse_weight(self, source_id, target_id):
        """시냅스 가중치 조회 (V4 기억 형성 확인) - Synapse 내부에서 가져옴"""
        # ⭐ V4 Memory Formation: Synapse 객체에서 직접 가져옴 (public API 사용)
        synapse = self.network.get_synapse(source_id, target_id)
        if synapse:
            return synapse.weight
        return 1.0
    
    def recall(self, cue_word: str, top_n: int = 3):
        """
        기억 검색 (V4) - MemoryRank 적용
        
        ⚠️  한계: 실제 "회상(Recall)"이 아니라 "문자열 매칭 + 시냅스 가중치 평균" 기반 인덱싱
        - 실제 회상: cue 입력 → 네트워크 다이내믹스 활성화/전파/경쟁
        - 현재 구현: 뉴런 ID 파싱 → 문자열 매칭 → 정적 weight 조회
        
        ⭐ 엔진 인터페이스: 결과 계약 고정
        - dict 반환 (구조화된 결과)
        
        Args:
            cue_word: 검색할 단어 (부분 매칭 가능)
            top_n: 반환할 결과 개수
            
        Returns:
            dict
            {
                'cue_word': str,
                'recall_success': bool,
                'results': [
                    {'word': str, 'score': float, 'importance': float}
                ],
                'spike_counts': {...},
                'activated_neurons': {...}
            }
        """
        # 1. 기본 유사도 계산 (단어 매칭)
        scores = {}
        word_to_neurons = {}
        for neuron_id in self.network.get_all_neuron_ids():
            parts = neuron_id.split('_')
            if len(parts) >= 2:
                word = parts[1]
                if word not in word_to_neurons:
                    word_to_neurons[word] = []
                word_to_neurons[word].append(neuron_id)
        
        # 단어 매칭 점수 계산
        for word_id, neuron_ids in word_to_neurons.items():
            if cue_word.upper() in word_id.upper() or word_id.upper() in cue_word.upper():
                total_weight = 0.0
                count = 0
                for neuron_id in neuron_ids:
                    if neuron_id.startswith('CA3_'):
                        incoming = self.network.get_incoming_synapses(neuron_id)
                        for synapse in incoming:
                            if synapse.source_id.startswith('DG_'):
                                total_weight += synapse.weight
                                count += 1
                if count > 0:
                    scores[word_id] = total_weight / count
        
        if not scores:
            return {
                'cue_word': cue_word,
                'recall_success': False,
                'results': [],
                'spike_counts': {'DG': 0, 'CA3': 0, 'CA1_Time': 0, 'CA1_Novelty': 0, 'total': 0},
                'activated_neurons': {'CA3': 0, 'total': 0}
            }
        
        # 2. MemoryRank 부스트 적용
        try:
            from v4_networking.memory_rank import apply_memory_rank_v4
            sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            boosted_results = apply_memory_rank_v4(self, sorted_results, boost_factor=1.5)
            final_results = boosted_results[:top_n]
        except Exception as e:
            print(f"⚠️  MemoryRank 적용 실패: {e}")
            sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            final_results = sorted_results[:top_n]
        
        # 3. 결과를 구조화된 dict로 변환
        recall_results = []
        for word_id, score in final_results:
            importance = self.get_memory_rank(word_id)
            recall_results.append({
                'word': word_id,
                'score': float(score),
                'importance': float(importance)
            })
        
        # 4. Spike counts 및 활성화된 뉴런 수 집계
        spike_counts = {'DG': 0, 'CA3': 0, 'CA1_Time': 0, 'CA1_Novelty': 0, 'total': 0}
        activated_ca3 = 0
        
        for neuron_name, neuron in self.ca3_neurons.items():
            spike_count = getattr(neuron, 'wake_spike_count', 0)
            spike_counts['CA3'] += spike_count
            if spike_count > 0:
                activated_ca3 += 1
        
        for neuron_name, neuron in self.dg_neurons.items():
            spike_counts['DG'] += getattr(neuron, 'wake_spike_count', 0)
        
        for neuron_name, neuron in self.ca1_time_cells.items():
            spike_counts['CA1_Time'] += getattr(neuron, 'wake_spike_count', 0)
        
        if self.ca1_novelty:
            spike_counts['CA1_Novelty'] += getattr(self.ca1_novelty, 'wake_spike_count', 0)
        
        spike_counts['total'] = sum(spike_counts.values())
        total_neurons = len(self.dg_neurons) + len(self.ca3_neurons) + len(self.ca1_time_cells) + (1 if self.ca1_novelty else 0)
        
        # ⭐ 엔진 인터페이스: 구조화된 dict 반환
        return {
            'cue_word': cue_word,
            'recall_success': len(recall_results) > 0,
            'results': recall_results,
            'spike_counts': spike_counts,
            'activated_neurons': {
                'CA3': activated_ca3,
                'total': total_neurons
            }
        }
    
    def get_memory_rank(self, word_id: str) -> float:
        """
        특정 메모리의 PageRank 중요도 점수 조회 (V4)
        
        Args:
            word_id: 단어 ID
            
        Returns:
            Importance score (0.0 ~ 1.0)
        """
        try:
            from v4_networking.memory_rank import MemoryRankV4
            ranker = MemoryRankV4(self)
            return ranker.get_score(word_id, default=0.5)
        except Exception as e:
            print(f"⚠️  MemoryRank 조회 실패: {e}")
            return 0.5
    
    def reset_all(self):
        """
        모든 뉴런 초기화 (V4 Memory Formation)
        
        ⭐ 중요: 시냅스 가중치(Synapse.weight)는 보존됨
        - reset_all은 뉴런 상태(V, m, h, n)만 초기화
        - 학습된 가중치는 유지되어 다음 테스트에 영향
        - 이것이 "비가역성(Hysteresis)"의 핵심
        """
        for neuron in self.dg_neurons.values():
            neuron.soma.V = -70.0
            neuron.soma.m = 0.05
            neuron.soma.h = 0.60
            neuron.soma.n = 0.32
            neuron.S = 0.0
            neuron.PTP = 1.0
            if hasattr(neuron, 'trigger_time'):
                neuron.trigger_time = None
        
        for neuron in self.ca3_neurons.values():
            neuron.soma.V = -70.0
            neuron.soma.m = 0.05
            neuron.soma.h = 0.60
            neuron.soma.n = 0.32
            neuron.S = 0.0
            neuron.PTP = 1.0
            neuron.wake_spike_count = 0
            neuron.recurrent_activation_count = 0  # ⭐ CA3 Recurrent Memory 초기화
            # ⭐ V4.3: Spike Budget 초기화 (엔진 승격: Homeostasis 기본값 고정)
            if hasattr(neuron, 'spike_count_window'):
                neuron.spike_count_window = 0
                neuron.dynamic_threshold_penalty = 0.0
                neuron.soma.spike_thresh = neuron.base_spike_thresh if hasattr(neuron, 'base_spike_thresh') else -34.0
                neuron.window_start_time = 0.0
        
        for neuron in self.ca1_time_cells.values():
            neuron.soma.V = -70.0
            neuron.soma.m = 0.05
            neuron.soma.h = 0.60
            neuron.soma.n = 0.32
            neuron.S = 0.0
            neuron.PTP = 1.0
            neuron.trigger_time = None
        
        if self.ca1_novelty:
            self.ca1_novelty.soma.V = -70.0
            self.ca1_novelty.soma.m = 0.05
            self.ca1_novelty.soma.h = 0.60
            self.ca1_novelty.soma.n = 0.32
            self.ca1_novelty.S = 0.0
            self.ca1_novelty.PTP = 1.0
        
        self.t = 0.0
    
    def soft_reset(self):
        """
        ⭐ V4.2/4.3: Soft Reset (부분적 상태 유지)
        
        완전 초기화 대신 부분적 상태 유지:
        - CA3 뉴런: 막전위를 -50mV로 유지 (Baseline Depolarization)
        - 게이트 변수(m, h, n)는 유지
        - Temporal Summation은 초기화 (I_syn_accumulated = 0)
        - ⭐ V4.3: Spike Budget 초기화 (새 window 시작) - 엔진 승격: Homeostasis 기본값 고정
        
        효과: 흥분성 상태 일부 유지 → 작은 입력에도 반응 가능
        """
        for neuron in self.dg_neurons.values():
            # DG는 완전 초기화 유지
            neuron.soma.V = -70.0
            neuron.soma.m = 0.05
            neuron.soma.h = 0.60
            neuron.soma.n = 0.32
            neuron.S = 0.0
            neuron.PTP = 1.0
            if hasattr(neuron, 'trigger_time'):
                neuron.trigger_time = None
        
        for neuron in self.ca3_neurons.values():
            # ⭐ V4.2: CA3는 Baseline Depolarization 유지
            baseline = neuron.baseline_V if hasattr(neuron, 'baseline_V') else -50.0
            # ⭐ V를 강제로 baseline_V로 설정 (이전 테스트의 잔여 상태 제거)
            neuron.soma.V = baseline
            # ⭐ V4.4: Ring Attractor 관성 초기화 (새 테스트 시작 시)
            if hasattr(neuron, 'recurrent_state'):
                neuron.recurrent_state = 0.0
            # ⭐ STEP 1: I_ext 초기화 (True Recall cue 버그 해결)
            if hasattr(neuron, 'I_ext'):
                neuron.I_ext = 0.0
            # ⭐ 게이트 변수는 유지 (m, h, n 변경 안 함)
            neuron.S = 0.0
            neuron.PTP = 1.0
            neuron.wake_spike_count = 0
            neuron.recurrent_activation_count = 0
            
            # ⭐ V4.2: Temporal Summation 초기화 (잔류 에너지 강제 소거)
            # ⚠️ 중요: 이전 테스트의 잔류 에너지가 다음 테스트를 오염시키지 않도록 완전 초기화
            if hasattr(neuron, 'I_syn_accumulated'):
                neuron.I_syn_accumulated = 0.0  # 완전 제로
            if hasattr(neuron.soma, 'I_syn_accumulated'):
                neuron.soma.I_syn_accumulated = 0.0  # 완전 제로
            if hasattr(neuron.soma, 'I_syn'):
                neuron.soma.I_syn = 0.0  # 완전 제로
            # ⭐ V4.4: Ring Attractor 관성도 초기화 (잔류 관성 제거)
            if hasattr(neuron, 'recurrent_state'):
                neuron.recurrent_state = 0.0  # 관성 완전 제로
            
            # ⭐ I_ext 초기화 (새 테스트 시작)
            # ⚠️ 주의: run_phase()에서 I_ext_dict를 통해 다시 설정되므로 여기서 리셋하지 않음
            # 하지만 이전 테스트의 잔여 자극을 제거하기 위해 초기화
            if hasattr(neuron, 'I_ext'):
                neuron.I_ext = 0.0
            
            # ⭐ Baseline Depolarization 강제 적용 (첫 step() 전에 V 보장)
            # 이전 테스트의 잔여 상태로 인해 V가 낮아질 수 있으므로 강제로 baseline_V로 설정
            if neuron.soma.V < baseline:
                neuron.soma.V = baseline
            
            # ⭐ V4.3: Spike Budget 초기화 (새 window 시작) - 엔진 승격: Homeostasis 기본값 고정
            if hasattr(neuron, 'spike_count_window'):
                neuron.spike_count_window = 0
                neuron.dynamic_threshold_penalty = 0.0
                neuron.soma.spike_thresh = neuron.base_spike_thresh if hasattr(neuron, 'base_spike_thresh') else -34.0
                neuron.window_start_time = 0.0
        
        for neuron in self.ca1_time_cells.values():
            neuron.soma.V = -70.0
            neuron.soma.m = 0.05
            neuron.soma.h = 0.60
            neuron.soma.n = 0.32
            neuron.S = 0.0
            neuron.PTP = 1.0
            neuron.trigger_time = None
        
        if self.ca1_novelty:
            self.ca1_novelty.soma.V = -70.0
            self.ca1_novelty.soma.m = 0.05
            self.ca1_novelty.soma.h = 0.60
            self.ca1_novelty.soma.n = 0.32
            self.ca1_novelty.S = 0.0
            self.ca1_novelty.PTP = 1.0
        
        self.t = 0.0
    
    # =========================================================
    # ⭐ 엔진 인터페이스: 학습/회상 분리 API
    # =========================================================
    
    def train(self, word: str, train_count: int = 1, I_ext: float = 350.0, 
              T_train: float = 10.0, context: str = "default") -> dict:
        """엔진 인터페이스: 학습 단계 (Wake Phase) - self.learn 스위치 제어"""
        dg_name = f"DG_{word}_0"
        ca3_name = f"CA3_{word}_0"
        weight_before = self.get_synapse_weight(dg_name, ca3_name)
        synapse = self.network.get_synapse(dg_name, ca3_name)
        pre_count_before = synapse.on_pre_spike_count if synapse else 0
        post_count_before = synapse.on_post_spike_count if synapse else 0
        
        for i in range(train_count):
            self.reset_all()
            I_ext_dict = {}
            for j in range(10):  # DG 10개 (102개 모델)
                dg_id = f"DG_{word}_{j}"
                I_ext_dict[dg_id] = {'value': I_ext, 'start': 1.0, 'end': 3.0}
            phase_result = self.run_phase(T_train, I_ext_dict)
            phase_result['phase_type'] = 'train'
        
        weight_after = self.get_synapse_weight(dg_name, ca3_name)
        pre_count_after = synapse.on_pre_spike_count if synapse else 0
        post_count_after = synapse.on_post_spike_count if synapse else 0
        
        return {
            'word': word,
            'train_count': train_count,
            'learning_enabled': self.learn,
            'spike_counts': phase_result.get('spike_counts', {}),
            'weights_before': weight_before,
            'weights_after': weight_after,
            'weight_change': weight_after - weight_before,
            'stdp_calls': {
                'on_pre_spike': pre_count_after - pre_count_before,
                'on_post_spike': post_count_after - post_count_before
            },
            'homeostasis': phase_result.get('homeostasis', {})
        }
    
    def sleep(self, num_theta_cycles: int = 8, replay_probability: float = 0.3) -> dict:
        """엔진 인터페이스: 수면 단계 (Sleep/Consolidation Phase)"""
        weights_before = {}
        for synapse in self.network.get_all_synapses():
            if synapse.source_id.startswith('DG_') and synapse.target_id.startswith('CA3_'):
                weights_before[synapse.source_id] = synapse.weight
        
        replay_counts = {}
        T_sleep = 5.0
        phase_result = self.run_phase(T_sleep)
        phase_result['phase_type'] = 'sleep'
        
        weights_after = {}
        for synapse in self.network.get_all_synapses():
            if synapse.source_id.startswith('DG_') and synapse.target_id.startswith('CA3_'):
                weights_after[synapse.source_id] = synapse.weight
        
        avg_before = sum(weights_before.values()) / len(weights_before) if weights_before else 1.0
        avg_after = sum(weights_after.values()) / len(weights_after) if weights_after else 1.0
        
        return {
            'sleep_cycles': num_theta_cycles,
            'replay_counts': replay_counts,
            'spike_counts': phase_result.get('spike_counts', {}),
            'weights_before': avg_before,
            'weights_after': avg_after,
            'weight_change': avg_after - avg_before,
            'homeostasis': phase_result.get('homeostasis', {})
        }
    
    def get_results(self) -> dict:
        """엔진 인터페이스: 현재 시스템 상태 조회"""
        total_neurons = len(self.dg_neurons) + len(self.ca3_neurons) + len(self.ca1_time_cells) + (1 if self.ca1_novelty else 0) + len(self.subiculum_gates)
        total_connections = sum(len(self.network.get_outgoing_synapses(nid)) 
                               for nid in self.network.get_all_neuron_ids())
        all_synapses = self.network.get_all_synapses()
        weights = [s.weight for s in all_synapses] if all_synapses else [1.0]
        
        return {
            'network': {
                'neurons': total_neurons,
                'connections': total_connections,
                'dg_neurons': len(self.dg_neurons),
                'ca3_neurons': len(self.ca3_neurons),
                'ca1_time_cells': len(self.ca1_time_cells),
                'ca1_novelty': 1 if self.ca1_novelty else 0,
                'subiculum_gates': len(self.subiculum_gates)
            },
            'weights': {
                'avg_weight': sum(weights) / len(weights),
                'max_weight': max(weights),
                'min_weight': min(weights),
                'total_synapses': len(all_synapses)
            },
            'homeostasis': {
                'spike_budget': 10,  # V4.2에는 Homeostasis 없음
                'budget_exceeded_neurons': 0,
                'avg_spikes_per_neuron': 0.0
            },
            'learning': {
                'enabled': self.learn,
                'stdp_active': self.learn
            },
            'results': self._results
        }
