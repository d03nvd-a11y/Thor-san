"""
Human-like Binocular Vision System

Mimics biological vision processing:
- Independent monocular processing (retinal processing)
- Feature-based correspondence matching (binocular neurons in V1/V2)
- Temporal depth fusion (visual memory)
- Visual attention mechanism (bottom-up + top-down)
"""

from .binocular_vision import BinocularVision, BinocularConfig
from .correspondence_matcher import CorrespondenceMatcher
from .temporal_fusion import TemporalFusion
from .visual_attention import VisualAttention

__all__ = [
    'BinocularVision',
    'BinocularConfig',
    'CorrespondenceMatcher',
    'TemporalFusion',
    'VisualAttention'
]
