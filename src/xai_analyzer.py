"""xAI Grok 4 analyzer for high-output documentation generation with 131K tokens support."""

import json
import logging
import time
from typing import Any, Dict, List, Optional

import oci
from pydantic import BaseModel

from .config import Settings, get_settings
from .readme_templates import get_template_by_type, auto_select_template_type, AVAILABLE_TEMPLATES
from .models import LanguageInfo, ArchitectureInfo, VisualInsight, ProjectAnalysis

logger = logging.getLogger(__name__)


class XAIGrokAnalysisError(Exception):
    """xAI Grok 4 analysis error."""
    pass


class XAIGrokAnalyzer:
    """xAI Grok 4 analyzer for high-output documentation generation with 131K tokens support."""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        
        # xAI Grok 4 Configuration with 131K output tokens
        self.compartment_id = self.settings.xai_grok_compartment_id or self.settings.oci_compartment_id
        self.config_profile = self.settings.xai_grok_config_profile
        self.model_id = self.settings.xai_grok_model_id
        self.endpoint = self.settings.xai_grok_endpoint
        self.max_tokens = self.settings.xai_grok_max_tokens  # 131,000 tokens max
        self.temperature = self.settings.xai_grok_temperature
        self.frequency_penalty = self.settings.xai_grok_frequency_penalty
        self.presence_penalty = self.settings.xai_grok_presence_penalty
        self.top_p = self.settings.xai_grok_top_p
        self.top_k = self.settings.xai_grok_top_k
        
        # Initialize OCI client for xAI Grok 4
        try:
            config = oci.config.from_file('~/.oci/config', self.config_profile)
            self.generative_ai_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                config=config,
                service_endpoint=self.endpoint,
                retry_strategy=oci.retry.NoneRetryStrategy(),
                timeout=(10, 240)
            )
            logger.info("xAI Grok 4 client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize xAI Grok 4 client: {e}")
            raise XAIGrokAnalysisError(f"xAI Grok 4 client initialization failed: {e}")
    
    async def _call_xai_grok_model(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Call xAI Grok 4 model with 131K token output support."""
        try:
            # Check prompt length (xAI Grok 4 has ~131K token input limit)
            # Rough estimation: 1 token ≈ 4 characters
            max_prompt_chars = 131000 * 4  # ~524,000 characters max
            if len(prompt) > max_prompt_chars:
                logger.warning(f"Prompt too long ({len(prompt)} chars), truncating to {max_prompt_chars}")
                prompt = prompt[:max_prompt_chars] + "\n\n[Note: Content truncated due to token limit. Please summarize the key points.]"
            
            # Build the message content
            content = oci.generative_ai_inference.models.TextContent()
            content.text = prompt
            
            message = oci.generative_ai_inference.models.Message()
            message.role = "USER"
            message.content = [content]
            
            # Create chat request with xAI Grok 4 compatible parameters
            chat_request = oci.generative_ai_inference.models.GenericChatRequest()
            chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
            chat_request.messages = [message]
            chat_request.max_tokens = max_tokens or self.max_tokens  # Up to 131,000 tokens
            chat_request.temperature = self.temperature
            # Note: xAI Grok 4 doesn't support presence_penalty and frequency_penalty
            # chat_request.frequency_penalty = self.frequency_penalty
            # chat_request.presence_penalty = self.presence_penalty
            chat_request.top_p = self.top_p
            chat_request.top_k = self.top_k
            
            # Create chat details
            chat_detail = oci.generative_ai_inference.models.ChatDetails()
            chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
                model_id=self.model_id
            )
            chat_detail.chat_request = chat_request
            chat_detail.compartment_id = self.compartment_id
            
            # Send request
            logger.info(f"Sending request to xAI Grok 4 model: {self.model_id}")
            logger.info(f"Max tokens: {chat_request.max_tokens}")
            
            chat_response = self.generative_ai_client.chat(chat_detail)
            
            # Extract response
            response_text = chat_response.data.chat_response.choices[0].message.content[0].text
            logger.info(f"xAI Grok 4 response received: {len(response_text)} characters")
            
            return response_text
            
        except oci.exceptions.ServiceError as e:
            logger.error(f"xAI Grok 4 service error: {e.status} {e.message}")
            raise XAIGrokAnalysisError(f"xAI Grok 4 API error: {e.status} {e.message}")
        except Exception as e:
            logger.error(f"xAI Grok 4 call failed: {e}")
            raise XAIGrokAnalysisError(f"xAI Grok 4 call failed: {e}")
    
    async def analyze_project(self, project_data: Dict[str, Any]) -> ProjectAnalysis:
        """Analyze project data using xAI Grok 4."""
        try:
            # Create basic analysis prompt
            analysis_prompt = f"""
            Analyze this project data and provide structured analysis:
            
            Project Data: {project_data}
            
            Provide analysis in this JSON format:
            {{
                "language": {{
                    "primary_language": "detected language",
                    "confidence": 0.9,
                    "frameworks": ["list of frameworks"],
                    "package_managers": ["package managers"],
                    "build_tools": ["build tools"],
                    "testing_frameworks": ["test frameworks"]
                }},
                "architecture": {{
                    "project_type": "type of project",
                    "architectural_patterns": ["patterns"],
                    "main_components": ["components"],
                    "data_flow": "description",
                    "external_dependencies": ["dependencies"],
                    "deployment_info": {{}}
                }},
                "description": "Project description",
                "installation_steps": ["installation steps"],
                "usage_examples": ["usage examples"],
                "project_structure": "structure description"
            }}
            """
            
            # Call xAI Grok 4
            response = await self._call_xai_grok_model(analysis_prompt, max_tokens=4000)
            
            # Parse response
            import json
            try:
                analysis_data = json.loads(response)
                return ProjectAnalysis(
                    language=LanguageInfo(**analysis_data["language"]),
                    architecture=ArchitectureInfo(**analysis_data["architecture"]),
                    description=analysis_data["description"],
                    installation_steps=analysis_data.get("installation_steps", []),
                    usage_examples=analysis_data.get("usage_examples", []),
                    project_structure=analysis_data.get("project_structure", "")
                )
            except (json.JSONDecodeError, KeyError):
                # Fallback basic analysis
                return self._create_basic_analysis(project_data)
                
        except Exception as e:
            logger.error(f"xAI Grok 4 analysis failed: {e}")
            return self._create_basic_analysis(project_data)
    
    def _create_basic_analysis(self, project_data: Dict[str, Any]) -> ProjectAnalysis:
        """Create basic analysis when xAI Grok 4 fails."""
        repo_info = project_data.get('repository', {})
        
        return ProjectAnalysis(
            language=LanguageInfo(
                primary_language="Unknown",
                confidence=0.5,
                frameworks=[],
                package_managers=[],
                build_tools=[],
                testing_frameworks=[]
            ),
            architecture=ArchitectureInfo(
                project_type="Unknown",
                architectural_patterns=[],
                main_components=[],
                data_flow="",
                external_dependencies=[],
                deployment_info={}
            ),
            description=repo_info.get('description', 'A software project'),
            installation_steps=[],
            usage_examples=[],
            project_structure=""
        )

    async def generate_comprehensive_readme(
        self,
        project_data: Dict[str, Any],
        template_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive README using xAI Grok 4 with 131K token output."""
        
        logger.info(f"Generating comprehensive README with xAI Grok 4 and template: {template_type or 'auto-selected'}")
        
        # Auto-select template if not specified
        if not template_type:
            template_type = auto_select_template_type(project_data)
            logger.info(f"Auto-selected template: {template_type}")
        
        # Get template prompt
        template_prompt = get_template_by_type(template_type, project_data)
        
        return await self._generate_with_xai_grok(template_prompt, project_data, template_type)
    
    async def _generate_with_xai_grok(
        self,
        template_prompt: str,
        project_data: Dict[str, Any],
        template_type: str
    ) -> Dict[str, Any]:
        """Generate comprehensive README using xAI Grok 4 with 131K output tokens."""
        
        try:
            start_time = time.time()
            
            # Prepare detailed project data for comprehensive analysis
            # With 131K tokens, we can include much more detail
            detailed_data = self._prepare_detailed_project_data(project_data)
            
            # Create comprehensive prompt for high-output generation
            comprehensive_prompt = self._create_comprehensive_prompt(
                template_prompt, 
                detailed_data, 
                template_type
            )
            
            # Log prompt info
            prompt_length = len(comprehensive_prompt)
            logger.info(f"Comprehensive prompt length: {prompt_length} characters")
            
            # With 131K tokens, we can handle much larger prompts
            # Rough estimate: 4 chars = 1 token, so ~500K chars for full input+output
            if prompt_length > 400000:  # Leave room for 131K output tokens
                logger.warning(f"Prompt very long ({prompt_length} chars), may need truncation...")
                # Still allow large prompts due to high token limit
            
            # Generate comprehensive documentation with xAI Grok 4
            logger.info(f"Generating comprehensive documentation using xAI Grok 4")
            response = await self._call_xai_grok_model(comprehensive_prompt, max_tokens=self.max_tokens)
            
            generation_time = time.time() - start_time
            
            # Validate response
            if not response:
                raise Exception("No response from xAI Grok 4 model")
            
            return {
                "success": True,
                "template_type": template_type,
                "model_used": self.model_id,
                "method": "xai_grok_4",
                "documentation": response,
                "tokens_used": len(response.split()),  # Approximate
                "max_tokens_available": self.max_tokens,
                "template_prompt_length": len(template_prompt),
                "generation_time": generation_time,
                "endpoint": self.endpoint,
                "comprehensive_mode": True
            }
            
        except Exception as e:
            logger.error(f"xAI Grok 4 generation failed: {e}")
            # Fallback to basic template generation
            return await self._generate_basic_fallback(template_prompt, project_data, template_type)
    
    def _prepare_detailed_project_data(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare detailed project data for comprehensive analysis with 131K tokens."""
        
        detailed_data = project_data.copy()
        
        # Add comprehensive analysis sections
        detailed_data["comprehensive_analysis"] = {
            "enable_detailed_code_analysis": True,
            "include_architecture_diagrams": True,
            "generate_api_documentation": True,
            "include_deployment_guides": True,
            "add_troubleshooting_section": True,
            "include_performance_considerations": True,
            "add_security_best_practices": True,
            "generate_testing_strategies": True,
            "include_contribution_guidelines": True,
            "add_advanced_usage_examples": True,
            "include_migration_guides": True,
            "generate_changelog_template": True
        }
        
        return detailed_data
    
    def _create_comprehensive_prompt(
        self, 
        template_prompt: str, 
        detailed_data: Dict[str, Any], 
        template_type: str
    ) -> str:
        """Create comprehensive prompt for xAI Grok 4 with 131K output support."""
        
        comprehensive_prompt = f"""
You are an expert technical writer with access to xAI Grok 4's 131,000 token output capability.
Generate a comprehensive, professional README.md that leverages this high output limit.

INSTRUCTIONS FOR COMPREHENSIVE DOCUMENTATION:
- Generate detailed, thorough documentation (target: 20,000+ words)
- Include extensive code examples and use cases
- Add comprehensive API documentation if applicable
- Include detailed installation guides for multiple platforms
- Add troubleshooting section with common issues and solutions
- Include performance optimization tips
- Add security best practices and considerations
- Generate comprehensive testing strategies
- Include detailed contribution guidelines
- Add advanced usage examples and tutorials
- Include deployment guides for different environments
- Add monitoring and logging recommendations

PROJECT DATA:
{json.dumps(detailed_data, indent=2)}

TEMPLATE TYPE: {template_type}

BASE TEMPLATE:
{template_prompt}

COMPREHENSIVE REQUIREMENTS:
1. **Executive Summary** (200-300 words)
2. **Detailed Feature Overview** (1000+ words)
3. **Complete Installation Guide** (800+ words)
   - Prerequisites for all platforms
   - Step-by-step installation
   - Docker setup if applicable
   - Cloud deployment options
4. **Comprehensive API Documentation** (2000+ words if applicable)
5. **Advanced Usage Examples** (1500+ words)
6. **Architecture Deep Dive** (1000+ words)
7. **Performance Considerations** (500+ words)
8. **Security Best Practices** (500+ words)
9. **Testing Strategies** (800+ words)
10. **Troubleshooting Guide** (1000+ words)
11. **Deployment Guide** (800+ words)
12. **Monitoring & Logging** (400+ words)
13. **Contributing Guidelines** (600+ words)
14. **Migration Guides** (if applicable)
15. **Changelog Template**
16. **Advanced Configuration**
17. **Integration Examples**
18. **Best Practices Summary**

Generate a comprehensive, professional README that fully utilizes the 131K token limit for maximum value.
"""
        
        return comprehensive_prompt
    
    async def _generate_basic_fallback(
        self,
        template_prompt: str,
        project_data: Dict[str, Any],
        template_type: str
    ) -> Dict[str, Any]:
        """Generate basic README when comprehensive generation fails."""
        
        logger.warning("Using basic fallback mode for xAI Grok 4")
        
        # Get basic project info
        repo_info = project_data.get('repository', {})
        project_name = repo_info.get('name', 'Project')
        description = repo_info.get('description', 'A software project')
        
        # Generate basic README
        basic_readme = f"""# {project_name}

{description}

## Overview

This project was analyzed using xAI Grok 4 but encountered an issue during comprehensive generation.

## Installation

```bash
# Basic installation steps would go here
# Please refer to the project files for specific requirements
```

## Usage

```bash
# Basic usage examples would go here
```

## Contributing

Please refer to the project's contribution guidelines.

## License

Please refer to the project's license file.

---
*Generated by Kiara with xAI Grok 4 (fallback mode)*
"""
        
        return {
            "success": True,
            "template_type": template_type,
            "model_used": "xai_grok_4_fallback",
            "method": "fallback_generation",
            "documentation": basic_readme,
            "tokens_used": len(basic_readme.split()),
            "fallback_mode": True,
            "endpoint": self.endpoint
        }


# Helper function for backward compatibility
async def generate_comprehensive_documentation(
    project_data: Dict[str, Any],
    template_type: Optional[str] = None,
    settings: Optional[Settings] = None
) -> Dict[str, Any]:
    """Generate comprehensive documentation using xAI Grok 4."""
    
    analyzer = XAIGrokAnalyzer(settings)
    return await analyzer.generate_comprehensive_readme(project_data, template_type)