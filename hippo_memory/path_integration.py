"""
Hippocampus Memory Engine - Path Integration Module

Path Integration: 자기 중심적 위치 추적
"""

import numpy as np
from typing import Tuple, Optional


class PathIntegrator:
    """
    Path Integration: 속도 벡터를 사용한 자기 중심적 위치 추적
    
    생물학적 배경:
    - 내후각피질(Entorhinal Cortex)의 Grid Cells
    - 해마의 Place Cells와 연동
    - 자기 중심적 위치 추적 (egocentric to allocentric)
    """
    def __init__(self, initial_x: float = 0.0, initial_y: float = 0.0):
        """
        Args:
            initial_x: 초기 X 좌표
            initial_y: 초기 Y 좌표
        """
        self.x = initial_x
        self.y = initial_y
        self.initial_x = initial_x
        self.initial_y = initial_y
        
        # 속도 추적 (디버깅용)
        self.vx = 0.0
        self.vy = 0.0
    
    def update(self, vx: float, vy: float, dt: float) -> Tuple[float, float]:
        """
        속도 벡터로 위치 업데이트
        
        Args:
            vx: X 방향 속도
            vy: Y 방향 속도
            dt: 시간 간격 [ms]
        
        Returns:
            업데이트된 위치 (x, y)
        """
        # 위치 업데이트: (x, y) = (x, y) + (vx, vy) * dt
        self.x += vx * dt
        self.y += vy * dt
        
        # 속도 저장 (디버깅용)
        self.vx = vx
        self.vy = vy
        
        return self.x, self.y
    
    def reset(self, x: Optional[float] = None, y: Optional[float] = None):
        """
        위치 리셋
        
        Args:
            x: 리셋할 X 좌표 (None이면 초기값)
            y: 리셋할 Y 좌표 (None이면 초기값)
        """
        if x is not None:
            self.x = x
        else:
            self.x = self.initial_x
        
        if y is not None:
            self.y = y
        else:
            self.y = self.initial_y
        
        self.vx = 0.0
        self.vy = 0.0
    
    def get_position(self) -> Tuple[float, float]:
        """현재 위치 반환"""
        return self.x, self.y
    
    def get_velocity(self) -> Tuple[float, float]:
        """현재 속도 반환"""
        return self.vx, self.vy
    
    def distance_from_origin(self) -> float:
        """원점으로부터의 거리"""
        return np.sqrt(self.x**2 + self.y**2)
    
    def distance_from(self, x: float, y: float) -> float:
        """주어진 위치로부터의 거리"""
        return np.sqrt((self.x - x)**2 + (self.y - y)**2)

