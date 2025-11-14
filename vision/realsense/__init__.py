"""
RealSense D435i Vision System

Simple, practical wrapper for Intel RealSense D435i depth camera.
Provides RGB-D streams, point clouds, and IMU data.

NO overcomplicated feature matching or calibration needed!
Hardware does all the heavy lifting.
"""

from .camera import RealSenseCamera, RealSenseConfig
from .processing import DepthProcessor

__all__ = ['RealSenseCamera', 'RealSenseConfig', 'DepthProcessor']
