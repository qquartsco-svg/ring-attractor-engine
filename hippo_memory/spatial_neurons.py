"""
Hippocampus Memory Engine - Spatial Neurons Module

Place Cells and spatial encoding for hippocampal spatial memory
"""

import numpy as np
from typing import Tuple, Optional
from .neurons import CA3NeuronV4


class PlaceField:
    """
    Place Field: 공간 위치에 대한 선호 반응 영역
    
    Gaussian tuning curve를 사용하여 특정 공간 위치에 선호 반응
    """
    def __init__(self, center_x: float, center_y: float, sigma: float = 0.5):
        """
        Args:
            center_x: Place field 중심 X 좌표
            center_y: Place field 중심 Y 좌표
            sigma: Place field 크기 (Gaussian 표준편차)
        """
        self.center_x = center_x
        self.center_y = center_y
        self.sigma = sigma
    
    def activation(self, x: float, y: float) -> float:
        """
        주어진 위치에서 Place Field 활성화 계산
        
        Gaussian tuning curve: f(x, y) = exp(-((x-x0)² + (y-y0)²) / (2σ²))
        
        Args:
            x: 현재 X 좌표
            y: 현재 Y 좌표
        
        Returns:
            활성화 값 (0.0 ~ 1.0)
        """
        distance_sq = (x - self.center_x)**2 + (y - self.center_y)**2
        activation = np.exp(-distance_sq / (2 * self.sigma**2))
        return float(activation)
    
    def distance(self, x: float, y: float) -> float:
        """주어진 위치에서 Place Field 중심까지의 거리"""
        return np.sqrt((x - self.center_x)**2 + (y - self.center_y)**2)


class CA3PlaceCellV4(CA3NeuronV4):
    """
    [CA3 Place Cell: 공간 위치 인코딩 + Ring Attractor]
    
    CA3NeuronV4를 확장하여 공간 위치 인코딩 기능 추가
    
    특징:
    - Place Field를 통한 공간 위치 선호 반응
    - Ring Attractor 동역학 유지
    - 공간 입력 + 재귀 연결 → Place Cell 활성화
    """
    def __init__(self, name: str, place_field: Optional[PlaceField] = None):
        """
        Args:
            name: 뉴런 이름
            place_field: Place Field (None이면 자동 생성)
        """
        super().__init__(name)
        
        # Place Field 설정
        if place_field is None:
            # 기본값: 이름에서 위치 추출 시도, 실패 시 (0, 0)
            self.place_field = PlaceField(0.0, 0.0, sigma=0.5)
        else:
            self.place_field = place_field
        
        # 공간 입력 관련 변수
        self.spatial_input = 0.0  # 공간 입력 전류 [µA/cm²]
        self.spatial_input_strength = 50.0  # 공간 입력 강도 스케일링
    
    def compute_spatial_input(self, x: float, y: float) -> float:
        """
        주어진 공간 위치에서 Place Field 활성화 계산
        
        Args:
            x: 현재 X 좌표
            y: 현재 Y 좌표
        
        Returns:
            공간 입력 전류 [µA/cm²]
        """
        activation = self.place_field.activation(x, y)
        self.spatial_input = activation * self.spatial_input_strength
        return self.spatial_input
    
    def step(self, dt: float, t: float, I_ext: float = 0.0, ATP: float = 100.0, 
             spatial_position: Optional[Tuple[float, float]] = None) -> dict:
        """
        V4 스타일 step with spatial input
        
        Args:
            dt: 시간 간격 [ms]
            t: 현재 시간 [ms]
            I_ext: 외부 전류 [µA/cm²]
            ATP: ATP 수준 [0, 100]
            spatial_position: 공간 위치 (x, y) 튜플 (None이면 공간 입력 없음)
        
        Returns:
            step 결과 딕셔너리
        """
        # 공간 위치가 주어지면 공간 입력 계산
        if spatial_position is not None:
            x, y = spatial_position
            spatial_input = self.compute_spatial_input(x, y)
        else:
            spatial_input = self.spatial_input  # 이전 값 유지
        
        # 공간 입력을 외부 전류에 추가
        I_total_ext = I_ext + spatial_input
        
        # 부모 클래스의 step 호출 (공간 입력 포함)
        return super().step(dt, t, I_total_ext, ATP)
    
    def get_place_field_info(self) -> dict:
        """Place Field 정보 반환"""
        return {
            'center_x': self.place_field.center_x,
            'center_y': self.place_field.center_y,
            'sigma': self.place_field.sigma,
            'spatial_input': self.spatial_input
        }


def create_place_fields_circular(n_neurons: int, radius: float = 1.0, 
                                 sigma: float = 0.5) -> list:
    """
    원형으로 배치된 Place Fields 생성
    
    CA3 뉴런들을 원형으로 배치하여 공간 맵 형성
    
    Args:
        n_neurons: 뉴런 수
        radius: 배치 반경
        sigma: Place field 크기
    
    Returns:
        PlaceField 리스트
    """
    place_fields = []
    
    for i in range(n_neurons):
        # 원형 배치: 각도 균등 분할
        angle = 2 * np.pi * i / n_neurons
        center_x = radius * np.cos(angle)
        center_y = radius * np.sin(angle)
        
        place_field = PlaceField(center_x, center_y, sigma=sigma)
        place_fields.append(place_field)
    
    return place_fields


def create_place_fields_grid(n_x: int, n_y: int, 
                             x_range: Tuple[float, float] = (-1.0, 1.0),
                             y_range: Tuple[float, float] = (-1.0, 1.0),
                             sigma: float = 0.5) -> list:
    """
    격자로 배치된 Place Fields 생성
    
    Args:
        n_x: X 방향 격자 수
        n_y: Y 방향 격자 수
        x_range: X 범위 (min, max)
        y_range: Y 범위 (min, max)
        sigma: Place field 크기
    
    Returns:
        PlaceField 리스트
    """
    place_fields = []
    
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    x_step = (x_max - x_min) / (n_x - 1) if n_x > 1 else 0.0
    y_step = (y_max - y_min) / (n_y - 1) if n_y > 1 else 0.0
    
    for i in range(n_y):
        for j in range(n_x):
            center_x = x_min + j * x_step
            center_y = y_min + i * y_step
            place_field = PlaceField(center_x, center_y, sigma=sigma)
            place_fields.append(place_field)
    
    return place_fields

