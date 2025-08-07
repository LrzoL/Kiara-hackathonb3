"""Groq AI analyzer for multimodal project analysis with LangChain integration and template system."""

import json
import logging
from typing import Any, Dict, List, Optional

from groq import Groq
from pydantic import BaseModel

try:
    from langchain_groq import ChatGroq
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ChatGroq = None

from .config import Settings, get_settings
from .readme_templates import get_template_by_type, auto_select_template_type, AVAILABLE_TEMPLATES

logger = logging.getLogger(__name__)


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


class GroqAnalysisError(Exception):
    """Groq analysis error."""
    pass


class GroqAnalyzer:
    """Groq AI analyzer for multimodal project analysis."""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        
        # Use LangChain ChatGroq if available, fallback to direct Groq client
        if LANGCHAIN_AVAILABLE:
            self.langchain_client = ChatGroq(
                groq_api_key=self.settings.groq.api_key,
                model_name=self.settings.groq.model_text,
                temperature=self.settings.groq.temperature,
                max_tokens=self.settings.groq.max_tokens
            )
            logger.info("Using LangChain ChatGroq client")
        else:
            self.langchain_client = None
            logger.warning("LangChain not available, using direct Groq client")
        
        # Keep direct Groq client for vision tasks
        self.client = Groq(api_key=self.settings.groq.api_key)
        self.text_model = self.settings.groq.model_text
        self.vision_model = self.settings.groq.model_vision
    
    async def _call_langchain(self, prompt: str) -> str:
        """Call LangChain ChatGroq client."""
        try:
            from langchain_core.messages import HumanMessage
            message = HumanMessage(content=prompt)
            # Use synchronous invoke instead of async ainvoke
            response = self.langchain_client.invoke([message])
            return response.content
        except Exception as e:
            logger.error(f"LangChain call failed: {e}")
            # Fallback to direct Groq call
            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.settings.groq.max_tokens,
                temperature=self.settings.groq.temperature
            )
            return response.choices[0].message.content
    
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
        
        # Remove any text before the first '{'
        json_start = clean_response.find('{')
        if json_start > 0:
            clean_response = clean_response[json_start:]
        
        # Remove any text after the last '}'
        json_end = clean_response.rfind('}')
        if json_end >= 0:
            clean_response = clean_response[:json_end + 1]
        
        return clean_response
    
    async def analyze_project(self, project_data: Dict[str, Any]) -> ProjectAnalysis:
        """Perform complete project analysis."""
        logger.info("Starting comprehensive project analysis")
        
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
            
            # Analyze images if available and vision analysis is enabled
            visual_insights = []
            if project_data.get("images") and self.settings.vision.analysis_enabled:
                visual_insights = await self.analyze_images(
                    project_data["images"],
                    language_info,
                    architecture
                )
            elif project_data.get("images") and not self.settings.vision.analysis_enabled:
                logger.info(f"Skipping analysis of {len(project_data['images'])} images - vision analysis disabled")
            
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
            raise GroqAnalysisError(f"Analysis failed: {e}")
    
    async def detect_language_multimodal(
        self,
        files: List[Dict[str, Any]],
        file_contents: Dict[str, str],
        images: Dict[str, str]
    ) -> LanguageInfo:
        """Detect language and framework using multimodal analysis."""
        
        # Prepare file structure summary
        file_summary = []
        for file_info in files[:50]:  # Limit to first 50 files
            if file_info.get("type") == "file":
                file_summary.append(file_info.get("path", ""))
        
        # Prepare key file contents summary
        key_files_summary = {}
        priority_files = ["package.json", "requirements.txt", "go.mod", "Cargo.toml", "pom.xml", "pyproject.toml"]
        
        for file_path, content in file_contents.items():
            file_name = file_path.split("/")[-1].lower()
            if any(priority in file_name for priority in priority_files):
                key_files_summary[file_path] = content[:2000]  # First 2000 chars
        
        # Prepare images summary for language detection
        image_summary = ""
        if images:
            image_summary = f"Found {len(images)} images including: {', '.join(list(images.keys())[:5])}"
        
        prompt = f"""Analyze this project and determine the primary programming language and technologies used.

PROJECT FILES:
{json.dumps(file_summary[:30], indent=2)}

KEY CONFIGURATION FILES:
{json.dumps(key_files_summary, indent=2)}

IMAGES FOUND:
{image_summary}

Based on this analysis, provide a JSON response with:
- primary_language: The main programming language (e.g., "Python", "JavaScript", "Go", "Rust", "Java")
- confidence: Confidence level from 0.0 to 1.0
- frameworks: List of detected frameworks/libraries
- package_managers: List of package managers (e.g., ["npm"], ["pip"], ["cargo"])
- build_tools: List of build tools (e.g., ["webpack"], ["maven"], ["make"])
- testing_frameworks: List of testing frameworks found

Respond with ONLY valid JSON, no additional text."""
        
        try:
            # Use LangChain if available, otherwise fallback to direct Groq
            if self.langchain_client:
                response = await self._call_langchain(prompt)
            else:
                response = self.client.chat.completions.create(
                    model=self.text_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=self.settings.groq.temperature
                )
                response = response.choices[0].message.content
            
            # Clean response and try to parse JSON
            clean_response = self._clean_json_response(response)
            result = json.loads(clean_response)
            return LanguageInfo(**result)
            
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
        """Analyze project architecture using text and visual information."""
        
        # Prepare architectural files
        arch_files = {}
        for file_path, content in file_contents.items():
            file_name = file_path.lower()
            if any(keyword in file_name for keyword in ["readme", "architecture", "design", "main", "app", "index"]):
                arch_files[file_path] = content[:3000]  # First 3000 chars
        
        prompt = f"""Analyze the architecture of this {language_info.primary_language} project.

LANGUAGE CONTEXT:
- Primary Language: {language_info.primary_language}
- Frameworks: {', '.join(language_info.frameworks)}
- Package Managers: {', '.join(language_info.package_managers)}

KEY FILES:
{json.dumps(arch_files, indent=2)}

IMAGES AVAILABLE: {len(images)} images found (will be analyzed separately)

Based on this analysis, provide a JSON response with:
- project_type: Type of project (e.g., "web_application", "cli_tool", "library", "microservice", "mobile_app")
- architectural_patterns: List of patterns used (e.g., ["MVC", "REST API", "microservices"])
- main_components: List of main components/modules
- data_flow: Brief description of data flow
- external_dependencies: List of major external dependencies
- deployment_info: Object with deployment-related information

Respond with ONLY valid JSON, no additional text."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=self.settings.groq.temperature
            )
            
            # Use LangChain if available
            if self.langchain_client:
                response = await self._call_langchain(prompt)
            else:
                response = response.choices[0].message.content
            
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
    
    async def analyze_images(
        self,
        images: Dict[str, str],
        language_info: LanguageInfo,
        architecture: ArchitectureInfo
    ) -> List[VisualInsight]:
        """Analyze images using vision model."""
        visual_insights = []
        
        for image_path, image_base64 in list(images.items())[:5]:  # Limit to 5 images
            try:
                insight = await self.analyze_single_image(
                    image_path,
                    image_base64,
                    language_info,
                    architecture
                )
                if insight:
                    visual_insights.append(insight)
                    
            except Exception as e:
                logger.warning(f"Failed to analyze image {image_path}: {e}")
        
        return visual_insights
    
    async def analyze_single_image(
        self,
        image_path: str,
        image_base64: str,
        language_info: LanguageInfo,
        architecture: ArchitectureInfo
    ) -> Optional[VisualInsight]:
        """Analyze a single image using vision model."""
        
        prompt = f"""Analyze this image from a {language_info.primary_language} project.

PROJECT CONTEXT:
- Language: {language_info.primary_language}
- Project Type: {architecture.project_type}
- Frameworks: {', '.join(language_info.frameworks)}

IMAGE PATH: {image_path}

Identify:
1. Type of diagram/image (architecture, flowchart, UI mockup, screenshot, etc.)
2. Technical components visible
3. Relationships and data flow shown
4. Technologies or patterns visible
5. How this relates to the codebase
6. Relevance score (0.0-1.0) for technical documentation

Provide a detailed technical description focusing on software architecture and engineering aspects."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                            }
                        ]
                    }
                ],
                max_tokens=800,
                temperature=self.settings.groq.temperature
            )
            
            description = response.choices[0].message.content
            
            # Extract technical relevance and components using text analysis
            relevance = await self._assess_technical_relevance(description)
            image_type = await self._classify_image_type(image_path, description)
            components = await self._extract_components(description)
            
            return VisualInsight(
                image_path=image_path,
                image_type=image_type,
                description=description,
                technical_relevance=relevance,
                extracted_components=components,
                architectural_info={}
            )
            
        except Exception as e:
            logger.error(f"Image analysis failed for {image_path}: {e}")
            return None
    
    async def generate_integrated_description(
        self,
        repository: Dict[str, Any],
        language_info: LanguageInfo,
        architecture: ArchitectureInfo,
        visual_insights: List[VisualInsight]
    ) -> str:
        """Generate integrated project description."""
        
        # Prepare visual insights summary
        visual_summary = ""
        if visual_insights:
            high_relevance_insights = [v for v in visual_insights if v.technical_relevance > 0.6]
            if high_relevance_insights:
                visual_summary = "Visual analysis reveals: " + "; ".join([
                    f"{v.image_type} showing {', '.join(v.extracted_components[:3])}"
                    for v in high_relevance_insights[:3]
                ])
        
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

VISUAL INSIGHTS:
{visual_summary}

Generate a 2-3 paragraph description that:
1. Explains what the project does and its purpose
2. Highlights key technologies and architectural decisions
3. Mentions any notable visual documentation or diagrams
4. Uses professional, technical language suitable for developers

Do not include installation instructions or usage examples."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=self.settings.groq.temperature
            )
            
            return response.choices[0].message.content.strip()
            
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
        
        # Look for configuration files
        config_files = {}
        for file_path, content in file_contents.items():
            file_name = file_path.split("/")[-1].lower()
            if any(config in file_name for config in ["package.json", "requirements", "cargo.toml", "go.mod", "pom.xml"]):
                config_files[file_path] = content[:1000]
        
        prompt = f"""Generate installation instructions for this {language_info.primary_language} project.

PROJECT DETAILS:
- Language: {language_info.primary_language}
- Package Managers: {', '.join(language_info.package_managers)}
- Build Tools: {', '.join(language_info.build_tools)}
- Project Type: {architecture.project_type}

CONFIGURATION FILES:
{json.dumps(config_files, indent=2)}

Generate a numbered list of installation steps. Include:
1. Prerequisites (language version, system requirements)
2. Clone repository step
3. Dependency installation
4. Configuration setup (if needed)
5. Build steps (if applicable)
6. Verification step

Keep it concise and practical. Return as a JSON array of strings."""
        
        try:
            # Use LangChain if available
            if self.langchain_client:
                response = await self._call_langchain(prompt)
            else:
                response = self.client.chat.completions.create(
                    model=self.text_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=self.settings.groq.temperature
                )
                response = response.choices[0].message.content
            
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
        
        # Look for main files and examples
        main_files = {}
        for file_path, content in file_contents.items():
            file_name = file_path.lower()
            if any(keyword in file_name for keyword in ["main", "app", "index", "example", "demo", "cli"]):
                main_files[file_path] = content[:2000]
        
        prompt = f"""Generate usage examples for this {language_info.primary_language} {architecture.project_type}.

MAIN FILES:
{json.dumps(main_files, indent=2)}

PROJECT CONTEXT:
- Language: {language_info.primary_language}
- Type: {architecture.project_type}
- Frameworks: {', '.join(language_info.frameworks)}

Generate 2-4 practical usage examples showing:
1. Basic usage
2. Common use case
3. Advanced example (if applicable)

Format as code blocks with brief explanations. Return as JSON array of strings."""
        
        try:
            # Use LangChain if available
            if self.langchain_client:
                response = await self._call_langchain(prompt)
            else:
                response = self.client.chat.completions.create(
                    model=self.text_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1200,
                    temperature=self.settings.groq.temperature
                )
                response = response.choices[0].message.content
            
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
        
        # Create a tree-like structure
        structure = []
        for file_info in files[:30]:  # Limit to first 30 files
            if file_info.get("type") == "file":
                path = file_info.get("path", "")
                structure.append(path)
        
        prompt = f"""Analyze this {language_info.primary_language} project structure and create a clean directory tree.

FILES:
{json.dumps(structure, indent=2)}

Create a markdown-formatted directory tree showing:
1. Main directories and their purpose
2. Key files and their roles
3. Configuration files
4. Documentation files

Use standard tree notation with ├── and └── characters.
Focus on the most important files and directories."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=self.settings.groq.temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Project structure generation failed: {e}")
            return "```\n" + "\n".join(structure[:20]) + "\n```"
    
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
    
    async def _assess_technical_relevance(self, description: str) -> float:
        """Assess technical relevance of image description."""
        technical_keywords = [
            "architecture", "diagram", "flow", "component", "service", "api", "database",
            "ui", "interface", "wireframe", "mockup", "schema", "model", "class",
            "deployment", "infrastructure", "network", "system"
        ]
        
        description_lower = description.lower()
        relevance = sum(1 for keyword in technical_keywords if keyword in description_lower)
        return min(relevance / 5.0, 1.0)  # Normalize to 0-1
    
    async def _classify_image_type(self, image_path: str, description: str) -> str:
        """Classify image type based on path and description."""
        path_lower = image_path.lower()
        desc_lower = description.lower()
        
        if any(keyword in path_lower for keyword in ["architecture", "arch", "diagram"]):
            return "architecture_diagram"
        elif any(keyword in desc_lower for keyword in ["flowchart", "flow"]):
            return "flowchart"
        elif any(keyword in desc_lower for keyword in ["ui", "interface", "mockup", "wireframe"]):
            return "ui_design"
        elif any(keyword in desc_lower for keyword in ["screenshot", "screen"]):
            return "screenshot"
        elif any(keyword in path_lower for keyword in ["logo", "icon"]):
            return "branding"
        else:
            return "technical_diagram"
    
    async def _extract_components(self, description: str) -> List[str]:
        """Extract components from image description."""
        # Simple keyword extraction - could be improved with NLP
        potential_components = []
        
        # Look for technical terms
        words = description.lower().split()
        technical_terms = [
            "api", "database", "server", "client", "service", "component", "module",
            "controller", "model", "view", "router", "middleware", "cache", "queue"
        ]
        
        for word in words:
            clean_word = word.strip(".,;:!?()")
            if clean_word in technical_terms:
                potential_components.append(clean_word)
        
        return list(set(potential_components))[:10]  # Return unique, limit to 10
    
    def _condense_project_data(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Condense project data to reduce token usage"""
        condensed = {}
        
        # Repository basic info
        repo = project_data.get("repository", {})
        condensed["repository"] = {
            "name": repo.get("name", "Unknown"),
            "description": repo.get("description", "")[:200],  # Limit to 200 chars
            "language": repo.get("language", ""),
            "private": repo.get("private", False),
            "stars": repo.get("stargazers_count", 0),
            "forks": repo.get("forks_count", 0)
        }
        
        # Files structure (limit to 20 most important files)
        files = project_data.get("files", [])
        condensed["files"] = files[:20]
        
        # File contents (skip large lock files and limit others)
        file_contents = project_data.get("file_contents", {})
        condensed["file_contents"] = {}
        
        # Skip large lock files and node_modules related files
        skip_patterns = ['package-lock.json', 'yarn.lock', 'pnpm-lock.yaml', 'node_modules', '.lock']
        
        count = 0
        for path, content in file_contents.items():
            if count >= 8:  # Limit to 8 files max
                break
                
            # Skip large lock files
            if any(pattern in path.lower() for pattern in skip_patterns):
                continue
                
            # Limit content size based on file type
            if path.endswith(('.json', '.md', '.txt')):
                max_chars = 800  # Allow more for important config files
            else:
                max_chars = 500
                
            if len(content) > max_chars:
                condensed["file_contents"][path] = content[:max_chars] + "..."
            else:
                condensed["file_contents"][path] = content
                
            count += 1
        
        # Language info (if available from analysis)
        if "language" in project_data:
            condensed["language"] = project_data["language"]
        
        # Architecture info (if available from analysis)
        if "architecture" in project_data:
            condensed["architecture"] = project_data["architecture"]
        
        # Images info (just count and types, not content)
        images = project_data.get("images", {})
        if images:
            condensed["images_info"] = {
                "count": len(images),
                "types": [path.split('.')[-1].lower() for path in list(images.keys())[:5]]
            }
        
        return condensed
    
    async def generate_readme_with_template(
        self,
        project_data: Dict[str, Any],
        template_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate README using specific template with LangChain Groq integration."""
        
        logger.info(f"Generating README with template: {template_type or 'auto-selected'}")
        
        # Auto-select template if not specified
        if not template_type:
            template_type = auto_select_template_type(project_data)
            logger.info(f"Auto-selected template: {template_type}")
        
        # Get template prompt
        template_prompt = get_template_by_type(template_type, project_data)
        
        # Use LangChain if available, fallback to direct Groq client
        if LANGCHAIN_AVAILABLE and ChatGroq:
            return await self._generate_with_langchain(template_prompt, project_data, template_type)
        else:
            return await self._generate_with_groq_client(template_prompt, project_data, template_type)
    
    async def _generate_with_langchain(
        self,
        template_prompt: str,
        project_data: Dict[str, Any],
        template_type: str
    ) -> Dict[str, Any]:
        """Generate README using LangChain ChatGroq."""
        
        try:
            # Initialize LangChain ChatGroq - prefer llama-3.1-8b-instant for all templates
            model_name = self.text_model  # Always use text model (llama-3.1-8b-instant) as preferred
            
            # Note: Using llama-3.1-8b-instant for all templates as requested for better performance
            
            chat_groq = ChatGroq(
                groq_api_key=self.settings.groq.api_key,
                model_name=model_name,
                temperature=self.settings.groq.temperature,
                max_tokens=self.settings.groq.max_tokens
            )
            
            # Prepare condensed project data to avoid token limit issues
            condensed_data = self._condense_project_data(project_data)
            # Format the prompt with condensed project data
            formatted_prompt = template_prompt.format(project_data=json.dumps(condensed_data, indent=2))
            
            # Generate README
            logger.info(f"Generating README using LangChain with {model_name}")
            logger.info(f"Formatted prompt length: {len(formatted_prompt)} characters")
            
            from langchain_core.messages import HumanMessage
            message = HumanMessage(content=formatted_prompt)
            
            logger.info("Invoking ChatGroq...")
            response = chat_groq.invoke([message])
            logger.info(f"ChatGroq response received: {type(response)}")
            
            # Validate response
            if not response:
                raise Exception("No response from LangChain ChatGroq")
            if not hasattr(response, 'content'):
                raise Exception(f"Response missing content attribute. Response type: {type(response)}, attributes: {dir(response)}")
            if response.content is None:
                raise Exception("Response content is None")
            
            return {
                "success": True,
                "template_type": template_type,
                "model_used": model_name,
                "method": "langchain_groq",
                "documentation": response.content,
                "tokens_used": len(response.content.split()),  # Approximate
                "template_prompt_length": len(template_prompt)
            }
            
        except Exception as e:
            logger.error(f"LangChain generation failed: {e}")
            # Fallback to simulation mode
            return await self._generate_simulation(template_prompt, project_data, template_type)
    
    async def _generate_with_groq_client(
        self,
        template_prompt: str,
        project_data: Dict[str, Any],
        template_type: str
    ) -> Dict[str, Any]:
        """Generate README using direct Groq client."""
        
        try:
            # Prepare condensed project data to avoid token limit issues
            condensed_data = self._condense_project_data(project_data)
            # Format the prompt with condensed project data
            formatted_prompt = template_prompt.format(project_data=json.dumps(condensed_data, indent=2))
            
            # Use preferred model - llama-3.1-8b-instant for all templates
            model_name = self.text_model  # Always use text model as preferred
            
            # Generate README
            logger.info(f"Generating README using direct Groq client with {model_name}")
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": formatted_prompt}],
                max_tokens=self.settings.groq.max_tokens,
                temperature=self.settings.groq.temperature
            )
            
            return {
                "success": True,
                "template_type": template_type,
                "model_used": model_name,
                "method": "groq_direct",
                "documentation": response.choices[0].message.content,
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else 0,
                "template_prompt_length": len(template_prompt)
            }
            
        except Exception as e:
            logger.error(f"Groq direct client generation failed: {e}")
            # Fallback to simulation mode
            return await self._generate_simulation(template_prompt, project_data, template_type)
    
    async def _generate_simulation(
        self,
        template_prompt: str,
        project_data: Dict[str, Any],
        template_type: str
    ) -> Dict[str, Any]:
        """Generate README using simulation mode when AI models fail."""
        
        logger.warning("Using simulation mode for README generation")
        
        # Get basic project info
        repo_info = project_data.get('repository', {})
        project_name = repo_info.get('name', 'Project')
        description = repo_info.get('description', 'A software project')
        language = project_data.get('language', {}).get('primary_language', 'Unknown')
        
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
        readme_content = simulation_templates.get(template_type, simulation_templates['emoji_rich'])
        
        return {
            "success": True,
            "template_type": template_type,
            "model_used": "simulation_mode",
            "method": "simulation",
            "documentation": readme_content,
            "tokens_used": len(readme_content.split()),
            "template_prompt_length": len(template_prompt),
            "warning": "Generated using simulation mode - AI models unavailable"
        }

# Utility functions for external use
async def test_all_templates(project_data: Dict[str, Any], analyzer: Optional[GroqAnalyzer] = None) -> Dict[str, Any]:
    """Test all 5 template types and return results."""
    
    if not analyzer:
        analyzer = GroqAnalyzer()
    
    results = {}
    
    for template_type in AVAILABLE_TEMPLATES:
        logger.info(f"Testing template: {template_type}")
        try:
            result = await analyzer.generate_readme_with_template(project_data, template_type)
            results[template_type] = {
                "success": result["success"],
                "method": result["method"],
                "model_used": result["model_used"],
                "content_length": len(result["documentation"]),
                "tokens_used": result.get("tokens_used", 0),
                "preview": result["documentation"][:200] + "..." if len(result["documentation"]) > 200 else result["documentation"]
            }
        except Exception as e:
            results[template_type] = {
                "success": False,
                "error": str(e),
                "content_length": 0,
                "tokens_used": 0
            }
    
    return results