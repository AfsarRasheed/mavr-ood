"""
Text-Guided Detection Package
Multi-agent text-guided object detection with step-by-step visualization.

Usage:
    from src.text_guided import run_text_guided_pipeline
    from src.text_guided.scene_agent import scene_understanding
    from src.text_guided.attribute_agent import attribute_matching_agent
"""

from src.text_guided.pipeline import run_text_guided_pipeline
from src.text_guided.scene_agent import scene_understanding
from src.text_guided.attribute_agent import attribute_matching_agent
from src.text_guided.query_parser import parse_query, spatial_filter
from src.text_guided.reasoning_agent import reasoning_agent
