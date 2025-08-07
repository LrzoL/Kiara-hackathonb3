"""Common data models used across different analyzers."""

from typing import Any, Dict, List
from pydantic import BaseModel


class LanguageInfo(BaseModel):
    """Language detection information."""
    
    primary_language: str
    confidence: float
    frameworks: List[str] = []
    package_managers: List[str] = []
    build_tools: List[str] = []
    testing_frameworks: List[str] = []


class ArchitectureInfo(BaseModel):
    """Architecture analysis information."""
    
    project_type: str
    architectural_patterns: List[str] = []
    main_components: List[str] = []
    data_flow: str = ""
    external_dependencies: List[str] = []
    deployment_info: Dict[str, Any] = {}


class VisualInsight(BaseModel):
    """Visual analysis insight from images."""
    
    image_path: str
    image_type: str
    description: str
    technical_relevance: float
    extracted_components: List[str] = []
    architectural_info: Dict[str, Any] = {}


class ProjectAnalysis(BaseModel):
    """Complete project analysis result."""
    
    language: LanguageInfo
    architecture: ArchitectureInfo
    description: str
    visual_insights: List[VisualInsight] = []
    installation_steps: List[str] = []
    usage_examples: List[str] = []
    project_structure: str = ""