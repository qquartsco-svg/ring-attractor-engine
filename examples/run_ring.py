#!/usr/bin/env python3
"""
Ring Attractor Engine - 데모

이 데모는 엔진의 핵심 기능을 보여줍니다:
1. 상태를 주입 (Input ON)
2. 입력을 제거 (Input OFF)
3. 상태가 유지되는지 확인 (State Retention)

사용법:
    python run_ring.py
    python run_ring.py --case case2
    python run_ring.py --case case2 --seed 42
    python run_ring.py --case case2 --debug
"""

import sys
import os
import argparse

# 프로젝트 루트 경로 추가
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from hippo_memory.ring_engine import RingAttractorEngine

def main():
    parser = argparse.ArgumentParser(
        description='Ring Attractor Engine - 입력 제거 후 상태 유지 테스트',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python run_ring.py
  python run_ring.py --case case2
  python run_ring.py --case case2 --seed 42
  python run_ring.py --case case2 --debug
  python run_ring.py --direction 5 --strength 0.8
        """
    )
    parser.add_argument(
        '--case', '-c',
        type=str,
        default='case2',
        help='Ring Attractor 케이스 선택 (기본값: case2)'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=None,
        help='랜덤 시드 (재현성 보장)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='디버그 모드 활성화'
    )
    parser.add_argument(
        '--direction',
        type=int,
        default=5,
        help='입력 방향 인덱스 (0 ~ size-1, 기본값: 5)'
    )
    parser.add_argument(
        '--strength',
        type=float,
        default=0.8,
        help='입력 강도 (0.0 ~ 1.0, 기본값: 0.8)'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=15,
        help='Ring 크기 (뉴런 수, 기본값: 15)'
    )
    parser.add_argument(
        '--cue-duration',
        type=float,
        default=None,
        help='Cue 지속 시간 [ms] (None이면 설정값 사용)'
    )
    parser.add_argument(
        '--maintain-duration',
        type=float,
        default=150.0,
        help='상태 유지 시간 [ms] (기본값: 150.0)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Ring Attractor Engine - State Retention Demo")
    print("=" * 70)
    print("Mode: STATE RETENTION ENGINE (STEP 1)")
    print("이 데모는 입력을 제거해도 상태가 유지되는지 확인합니다.")
    print("=" * 70)
    print("Note: This demo does NOT show navigation or drift.")
    print("      It demonstrates pure state retention.")
    print("=" * 70)
    print(f"Config: {args.case}")
    print(f"Size: {args.size}")
    print(f"Direction: {args.direction}")
    print(f"Strength: {args.strength}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print("=" * 70)
    
    # 엔진 초기화
    engine = RingAttractorEngine(
        size=args.size,
        config=args.case,
        seed=args.seed,
        debug=args.debug
    )
    
    # Phase 1: Input ON - 상태 주입
    print(f"\n📍 Phase 1: Input ON - 상태 주입 (direction={args.direction}, strength={args.strength})")
    
    # 새로운 API 사용: inject + run + release_input
    # 방법 A (베스트): None이면 엔진이 내부 default 사용 (블랙박스 유지)
    engine.inject(direction_idx=args.direction, strength=args.strength)
    state1 = engine.run(duration_ms=args.cue_duration)  # None이면 엔진 내부 default 사용
    engine.release_input()
    
    print(f"\n✅ Phase 1 완료:")
    print(f"   - Center: {state1.center:.2f}")
    print(f"   - Width: {state1.width:.2f}")
    print(f"   - Active: {state1.active_count}/{args.size}")
    print(f"   - Stability: {state1.stability:.2f}")
    
    # Phase 2: Input OFF - 입력 제거 후 상태 유지 확인
    print(f"\n📍 Phase 2: Input OFF - 입력 제거 후 상태 유지 확인 ({args.maintain_duration:.1f}ms)")
    state2 = engine.run(duration_ms=args.maintain_duration)
    
    # Phase 2 결과
    print(f"\n✅ Phase 2 완료:")
    print(f"   - Center: {state2.center:.2f}")
    print(f"   - Width: {state2.width:.2f}")
    print(f"   - Active: {state2.active_count}/{args.size}")
    print(f"   - Drift: {state2.drift:.2f} neurons")
    print(f"   - Stability: {state2.stability:.2f}")
    print(f"   - Sustained: {'YES' if state2.sustained else 'NO'}")
    
    # 최종 판정
    print(f"\n" + "=" * 70)
    print("📊 State Retention 판정")
    print("=" * 70)
    print("이 엔진은 입력을 제거해도 상태를 잃지 않는다는 것을 보여줍니다.")
    print("=" * 70)
    
    # Ring Attractor 성공 조건
    is_local_activation = 5 <= state2.active_count <= 7
    is_asymmetric = 1.0 <= state2.width <= 4.5
    is_sustained = state2.sustained
    
    if is_local_activation and is_asymmetric and is_sustained:
        verdict = "✅ SUCCESS"
        print(f"{verdict}: Ring Attractor 형성 성공")
        print(f"   - Local activation: {state2.active_count}/{args.size} ✅")
        print(f"   - Bump width: {state2.width:.2f} neurons ✅")
        print(f"   - Sustained: {'YES' if state2.sustained else 'NO'} ✅")
        if state2.drift > 0.1:
            print(f"   - Drift: {state2.drift:.2f} neurons ✅")
    elif state2.active_count == args.size:
        verdict = "⚠️  GLOBAL SATURATION"
        print(f"{verdict}: 모든 뉴런 활성화 (Ring Attractor 아님)")
    elif state2.active_count == 0:
        verdict = "❌ FAILED"
        print(f"{verdict}: 활성화 없음")
    else:
        verdict = "⚠️  PARTIAL"
        print(f"{verdict}: 부분 성공")
        print(f"   - Active: {state2.active_count}/{args.size}")
        print(f"   - Width: {state2.width:.2f}")
        print(f"   - Sustained: {'YES' if state2.sustained else 'NO'}")
    
    print("=" * 70)
    
    # 결과 요약 (10줄 이내)
    print(f"\n📋 결과 요약:")
    print(f"   Active: {state2.active_count}/{args.size}")
    print(f"   Center: {state2.center:.2f}")
    print(f"   Width: {state2.width:.2f}")
    print(f"   Drift: {state2.drift:.2f} neurons")
    print(f"   Sustained: {'YES' if state2.sustained else 'NO'}")
    print(f"   Stability: {state2.stability:.2f}")
    print(f"   Verdict: {verdict}")
    
    return 0 if verdict == "✅ SUCCESS" else 1

if __name__ == "__main__":
    sys.exit(main())

