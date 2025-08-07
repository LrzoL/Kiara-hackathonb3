"""
4 Distinct README Templates for Optimized Documentation Generation
Each template generates unique documentation optimized for token limits
"""

from typing import Dict, Any
from enum import Enum

class TemplateType(str, Enum):
    MINIMAL = "minimal"
    EMOJI_RICH = "emoji_rich"
    MODERN = "modern"
    TECHNICAL_DEEP = "technical_deep"

class ReadmeTemplates:
    """4 distinctly different README templates optimized for Groq LangChain integration"""
    
    @staticmethod
    def get_template_prompt(template_type: TemplateType, project_data: Dict[str, Any]) -> str:
        """Get the appropriate template prompt based on type and project data"""
        
        templates = {
            TemplateType.MINIMAL: ReadmeTemplates._get_minimal_template(),
            TemplateType.EMOJI_RICH: ReadmeTemplates._get_emoji_rich_template(), 
            TemplateType.MODERN: ReadmeTemplates._get_modern_template(),
            TemplateType.TECHNICAL_DEEP: ReadmeTemplates._get_technical_deep_template()
        }
        
        return templates[template_type]
    
    @staticmethod
    def auto_select_template(project_data: Dict[str, Any]) -> TemplateType:
        """Automatically select the best template based on project characteristics"""
        
        # Extract language and project type from analysis
        language = project_data.get('language', {}).get('primary_language', '').lower()
        project_type = project_data.get('architecture', {}).get('project_type', '').lower()
        frameworks = project_data.get('language', {}).get('frameworks', [])
        
        # Data Science projects - redirect to technical deep
        if any(fw.lower() in ['jupyter', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'matplotlib'] for fw in frameworks):
            return TemplateType.TECHNICAL_DEEP
        
        if language in ['python', 'r'] and any(keyword in project_type for keyword in ['data', 'ml', 'ai', 'analytics']):
            return TemplateType.TECHNICAL_DEEP
        
        # Technical/Enterprise projects
        if project_type in ['library', 'framework', 'api', 'microservice'] or language in ['java', 'c++', 'rust', 'go']:
            return TemplateType.TECHNICAL_DEEP
        
        # Web projects - modern template
        if project_type in ['web_application', 'webapp'] or language in ['javascript', 'typescript']:
            return TemplateType.MODERN
        
        # Default to emoji rich for appealing presentation
        return TemplateType.EMOJI_RICH
    
    @staticmethod
    def _get_minimal_template() -> str:
        """MINIMAL: Clean, professional README without emojis"""
        return """Create a professional README.md for this project.

STRUCTURE:
1. # Project Title
2. ## Overview - purpose and benefits
3. ## Features - key features
4. ## Installation - setup steps
5. ## Usage - examples
6. ## Contributing

Style: Clean, professional, no emojis. Be concise.

PROJECT DATA: {project_data}"""

    @staticmethod
    def _get_emoji_rich_template() -> str:
        """EMOJI_RICH: Fun README with emojis and engaging tone"""
        return """Create an engaging README.md with emojis for this project.

STRUCTURE:
🎯 # Project Title
🌟 ## Overview - what it does
✨ ## Features - key features
📦 ## Installation - setup steps  
🚀 ## Usage - examples
🤝 ## Contributing

Style: Use emojis, be friendly and visually appealing.

PROJECT DATA: {project_data}"""

    @staticmethod
    def _get_modern_template() -> str:
        """MODERN: Contemporary GitHub-style README with badges and tables"""
        return """Create a modern GitHub-style README.md.

STRUCTURE:
# Project Title
[![badges](shields.io-style-badges)]

## Overview
Project value proposition

## Features
| Feature | Status |
|---------|--------|
| Core | ✅ |

## Installation
Installation with code blocks

## Usage
Examples with syntax highlighting

## Contributing
Development workflow

Style: Use badges, tables, modern markdown.

PROJECT DATA: {project_data}"""

    @staticmethod
    def _get_technical_deep_template() -> str:
        """TECHNICAL_DEEP: Enterprise-grade technical documentation"""
        return """Create comprehensive technical documentation.

STRUCTURE:
# Project Title
## Executive Summary
## Architecture
- Components and design
- Integration points

## Technical Specs
- Requirements
- Dependencies

## Installation & Deployment
- Prerequisites
- Installation steps
- Configuration

## API Documentation
- Endpoints
- Authentication
- Examples

## Development
- Environment setup
- Build process

## Troubleshooting
- Common issues

Style: Formal, detailed, enterprise-focused.

PROJECT DATA: {project_data}"""

# Available template types for external use
AVAILABLE_TEMPLATES = [
    TemplateType.MINIMAL,
    TemplateType.EMOJI_RICH,
    TemplateType.MODERN,
    TemplateType.TECHNICAL_DEEP
]

# Utility functions for external use
def get_template_by_type(template_type: str, project_data: Dict[str, Any]) -> str:
    """Get template prompt by string type"""
    if template_type not in [t.value for t in AVAILABLE_TEMPLATES]:
        template_type = TemplateType.EMOJI_RICH.value
    
    template_enum = TemplateType(template_type)
    return ReadmeTemplates.get_template_prompt(template_enum, project_data)

def auto_select_template_type(project_data: Dict[str, Any]) -> str:
    """Auto-select template and return as string"""
    selected_template = ReadmeTemplates.auto_select_template(project_data)
    return selected_template.value

def list_available_templates() -> list:
    """List all available template types"""
    return [template.value for template in AVAILABLE_TEMPLATES]