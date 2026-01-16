"""
Hippocampus Memory Engine

A biologically plausible hippocampal memory model for pattern learning and recall
"""

from .engine import HippoMemoryV4System
from .config import CONFIG, HH_CONFIG
from .contracts import (
    MemoryEvent,
    MemoryPattern,
    MemoryContext,
    RecallResult,
    TrainingResult,
    ConsolidationResult,
    MemoryStatus
)
# V4.5 Spatial Memory modules
from .spatial_neurons import (
    PlaceField,
    CA3PlaceCellV4,
    create_place_fields_circular,
    create_place_fields_grid
)
from .path_integration import PathIntegrator
from .cognitive_map import CognitiveMap, SpatialMemory

__version__ = '1.0.0'
__all__ = [
    'HippoMemoryV4System',
    'CONFIG',
    'HH_CONFIG',
    'MemoryEvent',
    'MemoryPattern',
    'MemoryContext',
    'RecallResult',
    'TrainingResult',
    'ConsolidationResult',
    'MemoryStatus',
    # V4.5 Spatial Memory
    'PlaceField',
    'CA3PlaceCellV4',
    'create_place_fields_circular',
    'create_place_fields_grid',
    'PathIntegrator',
    'CognitiveMap',
    'SpatialMemory'
]
