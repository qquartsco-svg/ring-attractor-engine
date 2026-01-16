"""
Hippocampus Memory Engine - Cognitive Map Module

Cognitive Map: 공간 맵 형성 및 저장
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class SpatialMemory:
    """
    Spatial Memory: 방문한 위치 및 Place Cell 활성화 패턴 저장
    
    생물학적 배경:
    - 해마의 공간 기억 저장
    - Place Cell 활성화 패턴 = 공간 위치 표현
    - STDP를 통한 공간-공간 연결 강화
    """
    def __init__(self):
        """Spatial Memory 초기화"""
        # 방문한 위치 기록
        self.visited_locations: List[Tuple[float, float]] = []
        
        # 위치별 Place Cell 활성화 패턴
        # key: (x, y) 위치, value: {neuron_id: activation}
        self.location_patterns: Dict[Tuple[float, float], Dict[str, float]] = {}
        
        # 위치 간 연결 강도 (STDP 기반)
        # key: ((x1, y1), (x2, y2)), value: connection_strength
        self.spatial_connections: Dict[Tuple[Tuple[float, float], Tuple[float, float]], float] = defaultdict(float)
        
        # 위치별 방문 횟수
        self.visit_counts: Dict[Tuple[float, float], int] = defaultdict(int)
    
    def record_location(self, x: float, y: float, 
                      place_cell_activations: Dict[str, float]):
        """
        위치 및 Place Cell 활성화 패턴 기록
        
        Args:
            x: X 좌표
            y: Y 좌표
            place_cell_activations: {neuron_id: activation} 딕셔너리
        """
        location = (x, y)
        
        # 방문 위치 기록
        if location not in self.visited_locations:
            self.visited_locations.append(location)
        
        # Place Cell 활성화 패턴 저장
        self.location_patterns[location] = place_cell_activations.copy()
        
        # 방문 횟수 증가
        self.visit_counts[location] += 1
    
    def get_pattern(self, x: float, y: float, 
                   threshold: float = 0.1) -> Optional[Dict[str, float]]:
        """
        주어진 위치의 Place Cell 활성화 패턴 반환
        
        Args:
            x: X 좌표
            y: Y 좌표
            threshold: 활성화 임계값 (이하 제외)
        
        Returns:
            {neuron_id: activation} 딕셔너리 또는 None
        """
        location = (x, y)
        
        if location in self.location_patterns:
            pattern = self.location_patterns[location]
            # 임계값 이상만 반환
            return {k: v for k, v in pattern.items() if v >= threshold}
        
        return None
    
    def find_nearest_location(self, x: float, y: float, 
                            max_distance: float = 1.0) -> Optional[Tuple[float, float]]:
        """
        주어진 위치에서 가장 가까운 방문 위치 찾기
        
        Args:
            x: X 좌표
            y: Y 좌표
            max_distance: 최대 거리
        
        Returns:
            가장 가까운 위치 (x, y) 또는 None
        """
        if not self.visited_locations:
            return None
        
        min_distance = float('inf')
        nearest_location = None
        
        for loc_x, loc_y in self.visited_locations:
            distance = np.sqrt((x - loc_x)**2 + (y - loc_y)**2)
            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                nearest_location = (loc_x, loc_y)
        
        return nearest_location
    
    def strengthen_connection(self, from_location: Tuple[float, float],
                             to_location: Tuple[float, float],
                             strength: float = 0.1):
        """
        위치 간 연결 강화 (STDP 기반)
        
        Args:
            from_location: 출발 위치 (x, y)
            to_location: 도착 위치 (x, y)
            strength: 연결 강도 증가량
        """
        connection = (from_location, to_location)
        self.spatial_connections[connection] += strength
        # 상한선 설정 (선택적)
        self.spatial_connections[connection] = min(1.0, self.spatial_connections[connection])
    
    def get_connection_strength(self, from_location: Tuple[float, float],
                               to_location: Tuple[float, float]) -> float:
        """위치 간 연결 강도 반환"""
        connection = (from_location, to_location)
        return self.spatial_connections.get(connection, 0.0)
    
    def get_visited_locations(self) -> List[Tuple[float, float]]:
        """방문한 모든 위치 반환"""
        return self.visited_locations.copy()
    
    def get_visit_count(self, x: float, y: float) -> int:
        """주어진 위치의 방문 횟수 반환"""
        location = (x, y)
        return self.visit_counts.get(location, 0)


class CognitiveMap:
    """
    Cognitive Map: 공간 맵 형성 및 내비게이션 지원
    
    생물학적 배경:
    - 해마의 인지 지도 (Cognitive Map)
    - 공간 맵 형성 및 저장
    - 경로 계획 및 내비게이션
    """
    def __init__(self):
        """Cognitive Map 초기화"""
        self.spatial_memory = SpatialMemory()
        
        # 목표 위치 (내비게이션용)
        self.goal_location: Optional[Tuple[float, float]] = None
    
    def set_goal(self, x: float, y: float):
        """목표 위치 설정"""
        self.goal_location = (x, y)
    
    def plan_path(self, start_x: float, start_y: float,
                 goal_x: float, goal_y: float,
                 max_steps: int = 100) -> List[Tuple[float, float]]:
        """
        시작 위치에서 목표 위치까지의 경로 계획
        
        간단한 그리디 알고리즘 사용 (향후 개선 가능)
        
        Args:
            start_x: 시작 X 좌표
            start_y: 시작 Y 좌표
            goal_x: 목표 X 좌표
            goal_y: 목표 Y 좌표
            max_steps: 최대 경로 길이
        
        Returns:
            경로 (위치 리스트)
        """
        path = [(start_x, start_y)]
        current_x, current_y = start_x, start_y
        
        for step in range(max_steps):
            # 목표에 도달했는지 확인
            distance_to_goal = np.sqrt((current_x - goal_x)**2 + (current_y - goal_y)**2)
            if distance_to_goal < 0.1:  # 임계값
                break
            
            # 가장 가까운 방문 위치 찾기
            nearest = self.spatial_memory.find_nearest_location(current_x, current_y, max_distance=0.5)
            
            if nearest is None:
                # 방문한 위치가 없으면 목표 방향으로 이동
                dx = goal_x - current_x
                dy = goal_y - current_y
                norm = np.sqrt(dx**2 + dy**2)
                if norm > 0:
                    current_x += dx / norm * 0.1
                    current_y += dy / norm * 0.1
                else:
                    break
            else:
                # 방문한 위치로 이동
                current_x, current_y = nearest
            
            path.append((current_x, current_y))
        
        return path
    
    def recall_spatial_pattern(self, x: float, y: float,
                              threshold: float = 0.1) -> Optional[Dict[str, float]]:
        """
        주어진 위치의 공간 패턴 회상
        
        Pattern Completion: 부분 위치 입력 → 전체 공간 맵 복원
        
        Args:
            x: X 좌표
            y: Y 좌표
            threshold: 활성화 임계값
        
        Returns:
            Place Cell 활성화 패턴 또는 None
        """
        # 가장 가까운 방문 위치 찾기
        nearest = self.spatial_memory.find_nearest_location(x, y, max_distance=0.5)
        
        if nearest is not None:
            return self.spatial_memory.get_pattern(nearest[0], nearest[1], threshold)
        
        return None
    
    def get_spatial_map_summary(self) -> dict:
        """공간 맵 요약 정보 반환"""
        return {
            'visited_locations_count': len(self.spatial_memory.visited_locations),
            'spatial_connections_count': len(self.spatial_memory.spatial_connections),
            'goal_location': self.goal_location
        }

