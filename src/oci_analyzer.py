"""OCI AI analyzer for multimodal project analysis with Oracle Cloud Integration."""

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


class OCIAnalysisError(Exception):
    """OCI analysis error."""
    pass


class OCIAnalyzer:
    """Oracle Cloud Infrastructure AI analyzer for multimodal project analysis."""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        
        # OCI Configuration
        self.compartment_id = self.settings.oci.compartment_id
        self.config_profile = self.settings.oci.config_profile
        self.model_id = self.settings.oci.model_id
        self.endpoint = self.settings.oci.endpoint
        self.max_tokens = self.settings.oci.max_tokens  # 4000 tokens max
        self.temperature = self.settings.oci.temperature
        
        # Initialize OCI client
        try:
            config = oci.config.from_file('~/.oci/config', self.config_profile)
            self.generative_ai_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                config=config,
                service_endpoint=self.endpoint,
                retry_strategy=oci.retry.NoneRetryStrategy(),
                timeout=(10, 240)
            )
            logger.info("OCI Generative AI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OCI client: {e}")
            raise OCIAnalysisError(f"OCI client initialization failed: {e}")
    
    async def _call_oci_model(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Call OCI Generative AI model."""
        try:
            # Build the message content
            user_message_content = oci.generative_ai_inference.models.TextContent(text=prompt)
            user_message = oci.generative_ai_inference.models.Message(
                role="USER",
                content=[user_message_content]
            )
            
            # Create chat request with specified limits
            chat_request = oci.generative_ai_inference.models.GenericChatRequest(
                messages=[user_message],
                max_tokens=max_tokens or self.max_tokens,
                temperature=self.temperature,
                frequency_penalty=0,
                presence_penalty=0,
                top_p=0.75
            )
            
            # Create chat details
            chat_detail = oci.generative_ai_inference.models.ChatDetails(
                compartment_id=self.compartment_id,
                serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(model_id=self.model_id),
                chat_request=chat_request
            )
            
            # Send request
            logger.info(f"Sending request to OCI model: {self.model_id}")
            chat_response = self.generative_ai_client.chat(chat_detail)
            
            # Extract response
            response_text = chat_response.data.chat_response.choices[0].message.content[0].text
            logger.info(f"OCI response received: {len(response_text)} characters")
            
            return response_text
            
        except oci.exceptions.ServiceError as e:
            logger.error(f"OCI service error: {e.status} {e.message}")
            raise OCIAnalysisError(f"OCI API error: {e.status} {e.message}")
        except Exception as e:
            logger.error(f"OCI call failed: {e}")
            raise OCIAnalysisError(f"OCI call failed: {e}")
    
    def _clean_json_response(self, response: str) -> str:
        """Clean and prepare JSON response for parsing."""
        if not response:
            raise ValueError("Empty response")
        
        # Remove markdown code blocks
        clean_response = response.strip()
        if clean_response.startswith('```json'):
            clean_response = clean_response.replace('```json', '').replace('```', '').strip()
        elif clean_response.startswith('```'):
            clean_response = clean_response.replace('```', '').strip()
        
        # Handle cases where response starts with explanatory text
        # Look for JSON object start more aggressively
        json_patterns = ['{', '[']
        json_start = -1
        for pattern in json_patterns:
            start_pos = clean_response.find(pattern)
            if start_pos >= 0:
                if json_start == -1 or start_pos < json_start:
                    json_start = start_pos
        
        if json_start > 0:
            clean_response = clean_response[json_start:]
        
        # Remove any text after the last '}' or ']'
        json_end_patterns = ['}', ']']
        json_end = -1
        for pattern in json_end_patterns:
            end_pos = clean_response.rfind(pattern)
            if end_pos >= 0:
                if json_end == -1 or end_pos > json_end:
                    json_end = end_pos
        
        if json_end >= 0:
            clean_response = clean_response[:json_end + 1]
        
        # Fix common JSON issues
        clean_response = clean_response.replace("'", '"')  # Replace single quotes
        clean_response = clean_response.replace('\\n', '\n')  # Fix escaped newlines
        
        return clean_response
    
    async def analyze_project(self, project_data: Dict[str, Any]) -> ProjectAnalysis:
        """Perform complete project analysis."""
        logger.info("Starting comprehensive project analysis with OCI")
        
        try:
            # Analyze text content first
            language_info = await self.detect_language_multimodal(
                project_data.get("files", []),
                project_data.get("file_contents", {}),
                project_data.get("images", {})
            )
            
            # Analyze architecture
            architecture = await self.analyze_architecture_multimodal(
                project_data.get("file_contents", {}),
                project_data.get("images", {}),
                language_info
            )
            
            # Note: Vision analysis not yet implemented for OCI, focusing on text analysis
            visual_insights = []
            if project_data.get("images"):
                logger.info(f"Skipping analysis of {len(project_data['images'])} images - OCI vision analysis not yet implemented")
            
            # Generate comprehensive description
            description = await self.generate_integrated_description(
                project_data.get("repository", {}),
                language_info,
                architecture,
                visual_insights
            )
            
            # Generate installation steps
            installation_steps = await self.generate_installation_steps(
                language_info,
                architecture,
                project_data.get("file_contents", {})
            )
            
            # Generate usage examples
            usage_examples = await self.generate_usage_examples(
                project_data.get("file_contents", {}),
                language_info,
                architecture
            )
            
            # Generate project structure description
            project_structure = await self.generate_project_structure(
                project_data.get("files", []),
                language_info
            )
            
            return ProjectAnalysis(
                language=language_info,
                architecture=architecture,
                description=description,
                visual_insights=visual_insights,
                installation_steps=installation_steps,
                usage_examples=usage_examples,
                project_structure=project_structure
            )
            
        except Exception as e:
            logger.error(f"Project analysis failed: {e}")
            raise OCIAnalysisError(f"Analysis failed: {e}")
    
    async def detect_language_multimodal(
        self,
        files: List[Dict[str, Any]],
        file_contents: Dict[str, str],
        images: Dict[str, str]
    ) -> LanguageInfo:
        """Detect language and framework using multimodal analysis."""
        
        # Prepare file structure summary (limit for token efficiency)
        file_summary = []
        for file_info in files[:30]:  # Limit to first 30 files
            if file_info.get("type") == "file":
                file_summary.append(file_info.get("path", ""))
        
        # Prepare key file contents summary
        key_files_summary = {}
        priority_files = ["package.json", "requirements.txt", "go.mod", "Cargo.toml", "pom.xml", "pyproject.toml"]
        
        for file_path, content in file_contents.items():
            file_name = file_path.split("/")[-1].lower()
            if any(priority in file_name for priority in priority_files):
                # Limit content to save tokens
                key_files_summary[file_path] = content[:1000]  # First 1000 chars
        
        # Prepare images summary
        image_summary = ""
        if images:
            image_summary = f"Found {len(images)} images including: {', '.join(list(images.keys())[:3])}"
        
        prompt = f"""You are an expert software engineer. Analyze this project comprehensively and determine all technologies used.

PROJECT STRUCTURE:
Files: {json.dumps(file_summary[:20], indent=1)}

CONFIGURATION FILES CONTENT:
{json.dumps(key_files_summary, indent=1)}

ADDITIONAL CONTEXT:
Images found: {image_summary}

Instructions:
1. Examine file extensions, configuration files, and dependencies
2. Identify the primary programming language with high accuracy
3. Detect all frameworks, libraries, and tools used
4. Determine package managers, build systems, and testing frameworks
5. Provide confidence based on evidence strength

Return ONLY a JSON object:
{{
  "primary_language": "detected language name",
  "confidence": 0.0-1.0,
  "frameworks": ["framework1", "framework2"],
  "package_managers": ["manager1", "manager2"], 
  "build_tools": ["tool1", "tool2"],
  "testing_frameworks": ["test1", "test2"]
}}

No explanatory text - JSON only."""
        
        try:
            response = await self._call_oci_model(prompt, max_tokens=800)
            
            # Try to parse JSON response with better error handling
            try:
                clean_response = self._clean_json_response(response)
                result = json.loads(clean_response)
                return LanguageInfo(**result)
            except (json.JSONDecodeError, ValueError) as json_error:
                logger.warning(f"JSON parsing failed: {json_error}")
                logger.warning(f"Response was: {response[:200]}...")
                
                # Try to extract info from raw text response
                return self._extract_language_from_text(response, files)
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            logger.error(f"Response was: {response if 'response' in locals() else 'No response'}")
            # Fallback to basic detection
            return self._fallback_language_detection(files)
    
    async def analyze_architecture_multimodal(
        self,
        file_contents: Dict[str, str],
        images: Dict[str, str],
        language_info: LanguageInfo
    ) -> ArchitectureInfo:
        """Analyze project architecture using text information."""
        
        # Prepare architectural files (limit for token efficiency)
        arch_files = {}
        for file_path, content in file_contents.items():
            file_name = file_path.lower()
            if any(keyword in file_name for keyword in ["readme", "architecture", "design", "main", "app", "index"]):
                arch_files[file_path] = content[:1500]  # First 1500 chars
        
        prompt = f"""Analyze the architecture of this {language_info.primary_language} project.

LANGUAGE CONTEXT:
- Primary Language: {language_info.primary_language}
- Frameworks: {', '.join(language_info.frameworks)}
- Package Managers: {', '.join(language_info.package_managers)}

KEY FILES:
{json.dumps(arch_files, indent=1)}

IMAGES AVAILABLE: {len(images)} images found

Based on this analysis, provide a JSON response with:
- project_type: Type of project (e.g., "web_application", "cli_tool", "library", "microservice", "mobile_app")
- architectural_patterns: List of patterns used (e.g., ["MVC", "REST API", "microservices"])
- main_components: List of main components/modules
- data_flow: Brief description of data flow
- external_dependencies: List of major external dependencies
- deployment_info: Object with deployment-related information

Respond with ONLY valid JSON, no additional text."""
        
        try:
            response = await self._call_oci_model(prompt, max_tokens=1000)
            
            # Clean and parse JSON response
            clean_response = self._clean_json_response(response)
            result = json.loads(clean_response)
            return ArchitectureInfo(**result)
            
        except Exception as e:
            logger.error(f"Architecture analysis failed: {e}")
            logger.error(f"Raw response: {response if 'response' in locals() else 'No response'}")
            return ArchitectureInfo(
                project_type="unknown",
                architectural_patterns=[],
                main_components=[],
                data_flow="Unable to determine data flow",
                external_dependencies=[],
                deployment_info={}
            )
    
    async def generate_integrated_description(
        self,
        repository: Dict[str, Any],
        language_info: LanguageInfo,
        architecture: ArchitectureInfo,
        visual_insights: List[VisualInsight]
    ) -> str:
        """Generate integrated project description."""
        
        prompt = f"""Generate a professional, comprehensive description for this GitHub repository.

REPOSITORY INFO:
- Name: {repository.get('name', 'Unknown')}
- Description: {repository.get('description', 'No description')}
- Language: {language_info.primary_language}
- Stars: {repository.get('stargazers_count', 0)}
- Private: {repository.get('private', False)}

TECHNICAL ANALYSIS:
- Project Type: {architecture.project_type}
- Frameworks: {', '.join(language_info.frameworks)}
- Architectural Patterns: {', '.join(architecture.architectural_patterns)}
- Main Components: {', '.join(architecture.main_components)}

Generate a 2-3 paragraph description that:
1. Explains what the project does and its purpose
2. Highlights key technologies and architectural decisions
3. Uses professional, technical language suitable for developers

Do not include installation instructions or usage examples."""
        
        try:
            response = await self._call_oci_model(prompt, max_tokens=600)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Description generation failed: {e}")
            return f"A {language_info.primary_language} project implementing {architecture.project_type} functionality."
    
    async def generate_installation_steps(
        self,
        language_info: LanguageInfo,
        architecture: ArchitectureInfo,
        file_contents: Dict[str, str]
    ) -> List[str]:
        """Generate installation steps based on project analysis."""
        
        # Look for configuration files (limit for tokens)
        config_files = {}
        for file_path, content in file_contents.items():
            file_name = file_path.split("/")[-1].lower()
            if any(config in file_name for config in ["package.json", "requirements", "cargo.toml", "go.mod", "pom.xml"]):
                config_files[file_path] = content[:500]  # Limit to 500 chars
        
        prompt = f"""Generate installation instructions for this {language_info.primary_language} project.

PROJECT DETAILS:
- Language: {language_info.primary_language}
- Package Managers: {', '.join(language_info.package_managers)}
- Build Tools: {', '.join(language_info.build_tools)}
- Project Type: {architecture.project_type}

CONFIGURATION FILES:
{json.dumps(config_files, indent=1)}

Generate a numbered list of installation steps. Include:
1. Prerequisites (language version, system requirements)
2. Clone repository step
3. Dependency installation
4. Configuration setup (if needed)
5. Build steps (if applicable)
6. Verification step

Keep it concise and practical. Return as a JSON array of strings."""
        
        try:
            response = await self._call_oci_model(prompt, max_tokens=800)
            
            # Clean and parse JSON response
            clean_response = self._clean_json_response(response)
            result = json.loads(clean_response)
            return result if isinstance(result, list) else []
            
        except Exception as e:
            logger.error(f"Installation steps generation failed: {e}")
            return self._fallback_installation_steps(language_info)
    
    async def generate_usage_examples(
        self,
        file_contents: Dict[str, str],
        language_info: LanguageInfo,
        architecture: ArchitectureInfo
    ) -> List[str]:
        """Generate usage examples based on project analysis."""
        
        # Look for main files and examples (limit for tokens)
        main_files = {}
        for file_path, content in file_contents.items():
            file_name = file_path.lower()
            if any(keyword in file_name for keyword in ["main", "app", "index", "example", "demo", "cli"]):
                main_files[file_path] = content[:1000]  # First 1000 chars
        
        prompt = f"""Generate usage examples for this {language_info.primary_language} {architecture.project_type}.

MAIN FILES:
{json.dumps(main_files, indent=1)}

PROJECT CONTEXT:
- Language: {language_info.primary_language}
- Type: {architecture.project_type}
- Frameworks: {', '.join(language_info.frameworks)}

Generate 2-3 practical usage examples showing:
1. Basic usage
2. Common use case
3. Advanced example (if applicable)

Format as code blocks with brief explanations. Return as JSON array of strings."""
        
        try:
            response = await self._call_oci_model(prompt, max_tokens=900)
            
            # Clean and parse JSON response
            clean_response = self._clean_json_response(response)
            result = json.loads(clean_response)
            return result if isinstance(result, list) else []
            
        except Exception as e:
            logger.error(f"Usage examples generation failed: {e}")
            return [f"See the main {language_info.primary_language} files for usage examples."]
    
    async def generate_project_structure(
        self,
        files: List[Dict[str, Any]],
        language_info: LanguageInfo
    ) -> str:
        """Generate project structure description."""
        
        # Create a tree-like structure (limit for tokens)
        structure = []
        for file_info in files[:20]:  # Limit to first 20 files
            if file_info.get("type") == "file":
                path = file_info.get("path", "")
                structure.append(path)
        
        prompt = f"""Analyze this {language_info.primary_language} project structure and create a clean directory tree.

FILES:
{json.dumps(structure, indent=1)}

Create a markdown-formatted directory tree showing:
1. Main directories and their purpose
2. Key files and their roles
3. Configuration files
4. Documentation files

Use standard tree notation with ├── and └── characters.
Focus on the most important files and directories."""
        
        try:
            response = await self._call_oci_model(prompt, max_tokens=600)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Project structure generation failed: {e}")
            return "```\n" + "\n".join(structure[:15]) + "\n```"
    
    def _fallback_language_detection(self, files: List[Dict[str, Any]]) -> LanguageInfo:
        """Fallback language detection based on file extensions."""
        extensions = {}
        
        for file_info in files:
            if file_info.get("type") == "file":
                path = file_info.get("path", "")
                if "." in path:
                    ext = path.split(".")[-1].lower()
                    extensions[ext] = extensions.get(ext, 0) + 1
        
        # Language mapping
        lang_map = {
            "py": "Python", "js": "JavaScript", "ts": "TypeScript", "go": "Go",
            "rs": "Rust", "java": "Java", "cpp": "C++", "c": "C", "php": "PHP",
            "rb": "Ruby", "swift": "Swift", "kt": "Kotlin", "cs": "C#"
        }
        
        if extensions:
            most_common_ext = max(extensions.items(), key=lambda x: x[1])[0]
            primary_language = lang_map.get(most_common_ext, most_common_ext.upper())
        else:
            primary_language = "Unknown"
        
        return LanguageInfo(
            primary_language=primary_language,
            confidence=0.7,
            frameworks=[],
            package_managers=[],
            build_tools=[],
            testing_frameworks=[]
        )
    
    def _fallback_installation_steps(self, language_info: LanguageInfo) -> List[str]:
        """Fallback installation steps."""
        lang = language_info.primary_language.lower()
        
        if lang == "python":
            return [
                "Clone the repository: `git clone <repo-url>`",
                "Navigate to project directory: `cd <project-name>`",
                "Install dependencies: `pip install -r requirements.txt`",
                "Run the application"
            ]
        elif lang in ["javascript", "typescript"]:
            return [
                "Clone the repository: `git clone <repo-url>`",
                "Navigate to project directory: `cd <project-name>`",
                "Install dependencies: `npm install`",
                "Start the application: `npm start`"
            ]
        else:
            return [
                "Clone the repository: `git clone <repo-url>`",
                "Navigate to project directory: `cd <project-name>`",
                "Follow language-specific setup instructions",
                "Build and run the application"
            ]
    
    def _condense_project_data(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Condense project data to reduce token usage for OCI 4000 token limit"""
        condensed = {}
        
        # Repository basic info
        repo = project_data.get("repository", {})
        condensed["repository"] = {
            "name": repo.get("name", "Unknown"),
            "description": repo.get("description", "")[:150],  # Limit to 150 chars
            "language": repo.get("language", ""),
            "private": repo.get("private", False),
            "stars": repo.get("stargazers_count", 0)
        }
        
        # Files structure (limit to 15 most important files)
        files = project_data.get("files", [])
        condensed["files"] = files[:15]
        
        # File contents (more aggressive limiting for OCI token limits)
        file_contents = project_data.get("file_contents", {})
        condensed["file_contents"] = {}
        
        # Skip large lock files and node_modules related files
        skip_patterns = ['package-lock.json', 'yarn.lock', 'pnpm-lock.yaml', 'node_modules', '.lock']
        
        count = 0
        for path, content in file_contents.items():
            if count >= 5:  # Limit to 5 files max for OCI
                break
                
            # Skip large lock files
            if any(pattern in path.lower() for pattern in skip_patterns):
                continue
                
            # Aggressive content size limiting for OCI token limits
            if path.endswith(('.json', '.md', '.txt')):
                max_chars = 400  # Smaller limit for OCI
            else:
                max_chars = 250
                
            if len(content) > max_chars:
                condensed["file_contents"][path] = content[:max_chars] + "..."
            else:
                condensed["file_contents"][path] = content
                
            count += 1
        
        # Images info (just count and types, not content)
        images = project_data.get("images", {})
        if images:
            condensed["images_info"] = {
                "count": len(images),
                "types": [path.split('.')[-1].lower() for path in list(images.keys())[:3]]
            }
        
        return condensed
    
    def _extract_language_from_text(self, response: str, files: List[Dict[str, Any]]) -> LanguageInfo:
        """Extract language info from plain text response using AI analysis instead of hardcoded rules."""
        logger.info("Using AI to extract language info from text response")
        
        # Use a simpler AI-based approach to re-analyze the response
        try:
            # Create a prompt to extract structured information from the failed response
            extraction_prompt = f"""
Analyze this text response about a software project and extract the key information:

RESPONSE TEXT:
{response[:1000]}

Extract and return ONLY a JSON object with:
- primary_language: The main programming language mentioned
- confidence: A number between 0.0 and 1.0
- frameworks: Array of frameworks/libraries mentioned
- package_managers: Array of package managers mentioned
- build_tools: Array of build tools mentioned  
- testing_frameworks: Array of testing frameworks mentioned

Example format:
{{"primary_language": "TypeScript", "confidence": 0.9, "frameworks": ["Next.js", "React"], "package_managers": ["npm"], "build_tools": ["webpack"], "testing_frameworks": ["jest"]}}
"""
            
            # Use a smaller token limit for this extraction
            extracted_response = asyncio.create_task(self._call_oci_model(extraction_prompt, max_tokens=300))
            extracted_response = asyncio.get_event_loop().run_until_complete(extracted_response)
            
            # Try to parse the extracted JSON
            clean_response = self._clean_json_response(extracted_response)
            result = json.loads(clean_response)
            return LanguageInfo(**result)
            
        except Exception as e:
            logger.warning(f"AI extraction failed: {e}, falling back to file analysis")
            return self._fallback_language_detection(files)
    
    
    async def generate_readme_with_template(
        self,
        project_data: Dict[str, Any],
        template_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate README using specific template with OCI integration."""
        
        logger.info(f"Generating README with OCI and template: {template_type or 'auto-selected'}")
        
        # Auto-select template if not specified
        if not template_type:
            template_type = auto_select_template_type(project_data)
            logger.info(f"Auto-selected template: {template_type}")
        
        # Get template prompt
        template_prompt = get_template_by_type(template_type, project_data)
        
        return await self._generate_with_oci(template_prompt, project_data, template_type)
    
    async def _generate_with_oci(
        self,
        template_prompt: str,
        project_data: Dict[str, Any],
        template_type: str
    ) -> Dict[str, Any]:
        """Generate README using OCI Generative AI."""
        
        try:
            start_time = time.time()
            
            # Prepare condensed project data to fit within OCI 4000 token limit
            condensed_data = self._condense_project_data(project_data)
            
            # Format the prompt with condensed project data
            formatted_prompt = template_prompt.format(project_data=json.dumps(condensed_data, indent=1))
            
            # Log prompt size for debugging
            prompt_length = len(formatted_prompt)
            logger.info(f"Formatted prompt length: {prompt_length} characters")
            
            # Ensure we don't exceed token limits (rough estimate: 4 chars = 1 token)
            if prompt_length > 12000:  # ~3000 tokens for prompt, leaving 1000 for response
                logger.warning(f"Prompt too long ({prompt_length} chars), truncating...")
                formatted_prompt = formatted_prompt[:12000] + "\n\nGenerate README based on the above information."
            
            # Generate README with OCI
            logger.info(f"Generating README using OCI model: {self.model_id}")
            response = await self._call_oci_model(formatted_prompt, max_tokens=self.max_tokens)
            
            generation_time = time.time() - start_time
            
            # Validate response
            if not response:
                raise Exception("No response from OCI model")
            
            return {
                "success": True,
                "template_type": template_type,
                "model_used": self.model_id,
                "method": "oci_generative_ai",
                "documentation": response,
                "tokens_used": len(response.split()),  # Approximate
                "template_prompt_length": len(template_prompt),
                "generation_time": generation_time,
                "endpoint": self.endpoint
            }
            
        except Exception as e:
            logger.error(f"OCI generation failed: {e}")
            # Fallback to simulation mode
            return await self._generate_simulation(template_prompt, project_data, template_type)
    
    async def _generate_simulation(
        self,
        template_prompt: str,
        project_data: Dict[str, Any],
        template_type: str
    ) -> Dict[str, Any]:
        """Generate README using simulation mode when OCI models fail."""
        
        logger.warning("Using simulation mode for README generation")
        
        # Get basic project info
        repo_info = project_data.get('repository', {})
        project_name = repo_info.get('name', 'Project')
        description = repo_info.get('description', 'A software project')
        
        # Detect language from files
        files = project_data.get('files', [])
        language_info = self._fallback_language_detection(files)
        language = language_info.primary_language
        
        # Generate template-specific simulated README
        simulation_templates = {
            'minimal': f"""# {project_name}

{description}

## Installation

```bash
git clone <repository-url>
cd {project_name.lower()}
# Install dependencies based on {language}
```

## Usage

Basic usage instructions for {project_name}.

## License

See LICENSE file for details.""",

            'emoji_rich': f"""# 🚀 {project_name}

✨ {description}

## 🎯 Features

- 🔥 Built with {language}
- ⚡ Fast and efficient
- 🛠️ Easy to use

## 📦 Installation

```bash
git clone <repository-url>
cd {project_name.lower()}
# Install dependencies
```

## 💻 Usage

🚀 Getting started with {project_name} is easy!

## 🤝 Contributing

We welcome contributions! 🎉

## 📄 License

MIT License ⚖️""",

            'modern': f"""# {project_name}

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](link)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Language](https://img.shields.io/badge/language-{language}-red)](link)

> {description}

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Features

| Feature | Description |
|---------|-------------|
| Core Functionality | Primary features of {project_name} |
| {language} Support | Built with {language} |

## Installation

### Prerequisites

- {language} runtime environment

### Steps

```bash
git clone <repository-url>
cd {project_name.lower()}
# Setup instructions
```

## Contributing

Please read our contributing guidelines before submitting pull requests.""",

            'technical_deep': f"""# {project_name}

## Overview

{description}

## Architecture

### System Requirements

- Runtime: {language}
- Platform: Cross-platform compatible

### Technical Specifications

#### Core Components

- Main Application Layer
- Data Processing Module
- API Interface (if applicable)

#### Dependencies

Project dependencies will be automatically resolved during installation.

## Installation and Configuration

### Prerequisites

Ensure {language} development environment is properly configured.

### Installation Steps

1. Clone repository
2. Install dependencies
3. Configure environment variables
4. Build application
5. Run tests

### Configuration Options

Configuration parameters can be modified in the appropriate config files.

## Development

### Development Environment Setup

Standard {language} development tools are required.

### Build Process

Follow standard {language} build procedures.

## Deployment

### Production Deployment

Deploy following standard {language} application deployment practices.

## License

Licensed under standard terms."""
        }
        
        # Get template-specific simulation
        readme_content = simulation_templates.get(template_type, simulation_templates['modern'])
        
        return {
            "success": True,
            "template_type": template_type,
            "model_used": "simulation_mode",
            "method": "simulation",
            "documentation": readme_content,
            "tokens_used": len(readme_content.split()),
            "template_prompt_length": len(template_prompt),
            "warning": "Generated using simulation mode - OCI models unavailable"
        }