"""
RingAttractorEngine 테스트

시뮬레이션 결과로 핵심 기능 검증
"""

import pytest
import numpy as np
from hippo_memory.ring_engine import RingAttractorEngine
from hippo_memory.state_types import StateType


class TestRingAttractorEngine:
    """RingAttractorEngine 기본 기능 테스트"""
    
    def test_engine_initialization(self, engine_config):
        """엔진 초기화 테스트"""
        engine = RingAttractorEngine(**engine_config)
        
        assert engine is not None
        assert engine.size == 15
        assert engine.config == "case2"
    
    def test_state_injection_and_retention(self, engine_config):
        """상태 주입 및 유지 테스트 (핵심 기능)"""
        engine = RingAttractorEngine(**engine_config)
        
        # Phase 1: 상태 주입
        engine.inject(direction_idx=5, strength=0.8)
        state1 = engine.run(duration_ms=2.5)
        engine.release_input()
        
        # Phase 2: 입력 제거 후 유지
        state2 = engine.run(duration_ms=150.0)
        
        # 검증: 상태 유지 확인
        assert state2.sustained, "상태가 유지되어야 함"
        assert state2.active_count > 0, "활성 뉴런이 있어야 함"
        assert 5 <= state2.active_count <= 7, f"활성 뉴런 수가 적절해야 함 (현재: {state2.active_count})"
    
    def test_state_center_stability(self, engine_config):
        """상태 중심 안정성 테스트"""
        engine = RingAttractorEngine(**engine_config)
        
        # 상태 주입
        engine.inject(direction_idx=5, strength=0.8)
        engine.run(duration_ms=2.5)
        engine.release_input()
        
        # 여러 스텝에서 중심 위치 확인
        centers = []
        for _ in range(10):
            state = engine.step()
            centers.append(state.center)
        
        # 검증: 중심 위치가 안정적이어야 함
        center_std = np.std(centers)
        assert center_std < 2.0, f"중심 위치가 너무 불안정함 (std: {center_std})"
    
    def test_orbit_stability_score(self, engine_config):
        """궤도 안정성 점수 테스트"""
        engine = RingAttractorEngine(**engine_config)
        
        # 상태 주입 및 유지
        engine.inject(direction_idx=5, strength=0.8)
        engine.run(duration_ms=2.5)
        engine.release_input()
        state = engine.run(duration_ms=150.0)
        
        # 검증: 궤도 안정성 점수가 적절해야 함
        assert 0.0 <= state.orbit_stability <= 1.0, "궤도 안정성 점수는 0~1 사이여야 함"
        assert state.orbit_stability > 0.5, f"궤도 안정성 점수가 너무 낮음 (현재: {state.orbit_stability})"
    
    def test_controlled_drift(self, engine_config):
        """제어된 이동 테스트"""
        engine = RingAttractorEngine(**engine_config)
        
        # 초기 상태 형성
        engine.inject(direction_idx=5, strength=0.8)
        engine.run(duration_ms=2.5)
        engine.release_input()
        
        # Controlled Drift 실행
        initial_state = engine.get_state()
        state = engine.run_with_drift(velocity=0.01, duration_ms=100.0)
        
        # 검증: 상태가 유지되면서 이동했는지
        assert state.sustained, "상태가 유지되어야 함"
        assert abs(state.center - initial_state.center) > 0, "중심 위치가 이동했어야 함"
    
    def test_disturbance_recovery(self, engine_config):
        """외란 후 복구 테스트"""
        engine = RingAttractorEngine(**engine_config)
        
        # 초기 상태 형성
        engine.inject(direction_idx=5, strength=0.8)
        engine.run(duration_ms=2.5)
        engine.release_input()
        initial_state = engine.run(duration_ms=50.0)
        
        # 외란 주입 (다른 방향으로 강한 입력)
        engine.inject(direction_idx=6, strength=1.0)  # 외란 시뮬레이션 (최대 강도)
        disturbed_state = engine.run(duration_ms=10.0)
        engine.release_input()
        
        # 복구 확인
        recovered_state = engine.run(duration_ms=100.0)
        
        # 검증: 복구 후 안정성이 개선되거나 유지되어야 함
        # (외란이 작으면 안정성이 거의 변하지 않을 수 있음)
        assert recovered_state.orbit_stability >= disturbed_state.orbit_stability * 0.8, \
            f"복구 후 안정성이 크게 악화되지 않아야 함 (disturbed: {disturbed_state.orbit_stability}, recovered: {recovered_state.orbit_stability})"
    
    def test_state_type_configuration(self, engine_config):
        """상태 타입 설정 테스트"""
        for state_type in StateType:
            engine = RingAttractorEngine(
                size=15,
                config="case2",
                state_type=state_type,
                seed=42
            )
            
            assert engine.state_type == state_type, f"상태 타입이 올바르게 설정되어야 함: {state_type}"
    
    def test_deterministic_behavior(self, engine_config):
        """결정론적 동작 테스트 (시드 고정)"""
        # 동일한 시드로 두 번 실행
        engine1 = RingAttractorEngine(**engine_config)
        engine2 = RingAttractorEngine(**engine_config)
        
        # 동일한 입력
        engine1.inject(direction_idx=5, strength=0.8)
        engine2.inject(direction_idx=5, strength=0.8)
        
        state1 = engine1.run(duration_ms=10.0)
        state2 = engine2.run(duration_ms=10.0)
        
        # 검증: 동일한 결과 (시드 고정 시)
        assert state1.active_count == state2.active_count, \
            "시드 고정 시 동일한 결과가 나와야 함"
        assert abs(state1.center - state2.center) < 0.1, \
            "시드 고정 시 중심 위치가 거의 동일해야 함"


class TestRingEngineSimulation:
    """시뮬레이션 결과 검증 테스트"""
    
    def test_wear_risk_reduction_simulation(self, engine_config):
        """마모 위험도 감소 시뮬레이션 테스트"""
        engine = RingAttractorEngine(**engine_config)
        
        # 초기 상태
        engine.inject(direction_idx=5, strength=0.8)
        engine.run(duration_ms=2.5)
        engine.release_input()
        
        # 장기간 시뮬레이션
        wear_risks = []
        for _ in range(100):
            state = engine.step()
            orbit_control = engine.get_orbit_control(target_phase=5.0)
            wear_risks.append(orbit_control.wear_risk)
        
        # 검증: 마모 위험도가 적절한 범위 내에 있어야 함
        avg_wear_risk = np.mean(wear_risks)
        assert 0.0 <= avg_wear_risk <= 1.0, "마모 위험도는 0~1 사이여야 함"
        assert avg_wear_risk < 0.5, f"마모 위험도가 너무 높음 (평균: {avg_wear_risk})"
    
    def test_long_term_stability(self, engine_config):
        """장기간 안정성 테스트"""
        engine = RingAttractorEngine(**engine_config)
        
        # 초기 상태
        engine.inject(direction_idx=5, strength=0.8)
        engine.run(duration_ms=2.5)
        engine.release_input()
        
        # 장기간 실행 (1000 스텝)
        stability_scores = []
        for _ in range(1000):
            state = engine.step()
            stability_scores.append(state.orbit_stability)
        
        # 검증: 장기간 안정성 유지
        final_stability = stability_scores[-1]
        assert final_stability > 0.3, f"장기간 실행 후에도 안정성이 유지되어야 함 (현재: {final_stability})"
        
        # 검증: 안정성 변동이 크지 않아야 함
        stability_std = np.std(stability_scores)
        assert stability_std < 0.3, f"안정성 변동이 너무 큼 (std: {stability_std})"

