"""Template engine for adaptive documentation generation."""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from .config import Settings, get_settings
from .groq_analyzer import ProjectAnalysis, VisualInsight

logger = logging.getLogger(__name__)


class TemplateSection(BaseModel):
    """Documentation template section."""
    
    name: str
    content: str
    optional: bool = False
    conditions: List[str] = []


class DocumentationTemplate(BaseModel):
    """Complete documentation template."""
    
    name: str
    language: str
    project_types: List[str]
    sections: List[TemplateSection]
    metadata: Dict[str, Any] = {}


class TemplateEngine:
    """Adaptive template engine for documentation generation."""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, DocumentationTemplate]:
        """Load predefined templates."""
        templates = {}
        
        # Base universal template
        templates["universal"] = DocumentationTemplate(
            name="Universal Template",
            language="*",
            project_types=["*"],
            sections=[
                TemplateSection(
                    name="header",
                    content=self._get_header_template()
                ),
                TemplateSection(
                    name="badges",
                    content=self._get_badges_template(),
                    optional=True
                ),
                TemplateSection(
                    name="description",
                    content=self._get_description_template()
                ),
                TemplateSection(
                    name="visual_overview",
                    content=self._get_visual_overview_template(),
                    optional=True,
                    conditions=["has_diagrams"]
                ),
                TemplateSection(
                    name="features",
                    content=self._get_features_template(),
                    optional=True
                ),
                TemplateSection(
                    name="technologies",
                    content=self._get_technologies_template()
                ),
                TemplateSection(
                    name="prerequisites",
                    content=self._get_prerequisites_template()
                ),
                TemplateSection(
                    name="installation",
                    content=self._get_installation_template()
                ),
                TemplateSection(
                    name="usage",
                    content=self._get_usage_template()
                ),
                TemplateSection(
                    name="project_structure",
                    content=self._get_project_structure_template(),
                    optional=True
                ),
                TemplateSection(
                    name="api_documentation",
                    content=self._get_api_documentation_template(),
                    optional=True,
                    conditions=["is_api_project"]
                ),
                TemplateSection(
                    name="configuration",
                    content=self._get_configuration_template(),
                    optional=True,
                    conditions=["has_config_files"]
                ),
                TemplateSection(
                    name="testing",
                    content=self._get_testing_template(),
                    optional=True,
                    conditions=["has_tests"]
                ),
                TemplateSection(
                    name="deployment",
                    content=self._get_deployment_template(),
                    optional=True,
                    conditions=["has_deployment_config"]
                ),
                TemplateSection(
                    name="contributing",
                    content=self._get_contributing_template(),
                    optional=True
                ),
                TemplateSection(
                    name="license",
                    content=self._get_license_template(),
                    optional=True
                ),
                TemplateSection(
                    name="acknowledgments",
                    content=self._get_acknowledgments_template(),
                    optional=True
                )
            ]
        )
        
        # Python-specific template
        templates["python"] = self._create_python_template(templates["universal"])
        
        # JavaScript/TypeScript template
        templates["javascript"] = self._create_javascript_template(templates["universal"])
        
        # Java template
        templates["java"] = self._create_java_template(templates["universal"])
        
        # Go template
        templates["go"] = self._create_go_template(templates["universal"])
        
        # Web application template
        templates["web_app"] = self._create_web_app_template(templates["universal"])
        
        # CLI tool template
        templates["cli_tool"] = self._create_cli_template(templates["universal"])
        
        # Library template
        templates["library"] = self._create_library_template(templates["universal"])
        
        return templates
    
    def select_template(self, analysis: ProjectAnalysis) -> DocumentationTemplate:
        """Select the most appropriate template."""
        language = analysis.language.primary_language.lower()
        project_type = analysis.architecture.project_type.lower()
        
        # Priority order for template selection
        template_priority = [
            f"{language}_{project_type}",
            project_type,
            language,
            "universal"
        ]
        
        for template_key in template_priority:
            if template_key in self.templates:
                logger.info(f"Selected template: {template_key}")
                return self.templates[template_key]
        
        logger.info("Using universal template as fallback")
        return self.templates["universal"]
    
    def generate_documentation(
        self,
        analysis: ProjectAnalysis,
        repository_data: Dict[str, Any]
    ) -> str:
        """Generate complete documentation."""
        
        template = self.select_template(analysis)
        
        # Prepare template context
        context = self._prepare_context(analysis, repository_data)
        
        # Generate sections
        sections = []
        for section in template.sections:
            if self._should_include_section(section, context):
                rendered_content = self._render_section(section, context)
                if rendered_content.strip():
                    sections.append(rendered_content)
        
        # Combine sections
        documentation = "\n\n".join(sections)
        
        # Post-process
        documentation = self._post_process_documentation(documentation, context)
        
        return documentation
    
    def _prepare_context(
        self,
        analysis: ProjectAnalysis,
        repository_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare template rendering context."""
        
        repo = repository_data.get("repository", {})
        
        # Basic repository information
        repo_full_name = repo.get('full_name', '')
        context = {
            "project_name": repo.get("name", "Unknown Project"),
            "project_description": analysis.description,
            "repository_url": f"https://github.com/{repo_full_name}",
            "repository_path": repo_full_name,
            "language": analysis.language.primary_language,
            "frameworks": analysis.language.frameworks,
            "package_managers": analysis.language.package_managers,
            "build_tools": analysis.language.build_tools,
            "testing_frameworks": analysis.language.testing_frameworks,
            "project_type": analysis.architecture.project_type,
            "architectural_patterns": analysis.architecture.architectural_patterns,
            "main_components": analysis.architecture.main_components,
            "installation_steps": analysis.installation_steps,
            "usage_examples": analysis.usage_examples,
            "project_structure": analysis.project_structure,
            "visual_insights": analysis.visual_insights,
            "private_repo": repo.get("private", False),
            "license": (repo.get("license") or {}).get("name", ""),
            "topics": repo.get("topics", []),
            "stars": repo.get("stargazers_count", 0),
            "forks": repo.get("forks_count", 0),
            "created_date": repo.get("created_at", ""),
            "updated_date": repo.get("updated_at", "")
        }
        
        # Add conditional flags
        context.update({
            "has_diagrams": len([v for v in analysis.visual_insights if v.technical_relevance > 0.5]) > 0,
            "has_tests": len(analysis.language.testing_frameworks) > 0,
            "has_deployment_config": any(
                keyword in analysis.language.frameworks
                for keyword in ["Docker", "Kubernetes", "AWS", "Azure", "GCP"]
            ),
            "has_config_files": any(
                keyword in str(analysis.language.frameworks).lower()
                for keyword in ["config", "env", "settings"]
            ),
            "is_api_project": any(
                keyword in analysis.architecture.project_type.lower()
                for keyword in ["api", "service", "web"]
            ),
            "is_web_app": "web" in analysis.architecture.project_type.lower(),
            "is_cli_tool": "cli" in analysis.architecture.project_type.lower(),
            "is_library": "library" in analysis.architecture.project_type.lower(),
            "has_multiple_languages": len(analysis.language.frameworks) > 2
        })
        
        return context
    
    def _should_include_section(self, section: TemplateSection, context: Dict[str, Any]) -> bool:
        """Check if section should be included."""
        if not section.optional:
            return True
        
        if not section.conditions:
            return True
        
        return all(context.get(condition, False) for condition in section.conditions)
    
    def _render_section(self, section: TemplateSection, context: Dict[str, Any]) -> str:
        """Render a template section."""
        try:
            return section.content.format(**context)
        except KeyError as e:
            logger.warning(f"Missing template variable {e} in section {section.name}")
            return section.content
        except Exception as e:
            logger.error(f"Error rendering section {section.name}: {e}")
            return f"<!-- Error rendering {section.name} -->"
    
    def _post_process_documentation(self, documentation: str, context: Dict[str, Any]) -> str:
        """Post-process generated documentation."""
        
        # Remove excessive empty lines
        lines = documentation.split('\n')
        processed_lines = []
        empty_line_count = 0
        
        for line in lines:
            if line.strip() == '':
                empty_line_count += 1
                if empty_line_count <= 2:  # Max 2 consecutive empty lines
                    processed_lines.append(line)
            else:
                empty_line_count = 0
                processed_lines.append(line)
        
        documentation = '\n'.join(processed_lines)
        
        # Add generation notice
        generation_notice = f"""
---

> 🤖 *This documentation was automatically generated using [GitHub Doc Agent](https://github.com/your-org/github-doc-agent) with multimodal AI analysis.*
> 
> **Generated on:** {context.get('updated_date', 'Unknown')}  
> **Analysis confidence:** {context.get('language_confidence', 'N/A')}  
> **Detected language:** {context.get('language', 'Unknown')}
"""
        
        documentation += generation_notice
        
        return documentation
    
    # Template content methods
    def _get_header_template(self) -> str:
        return """# {project_name}

> {project_description}

## 📖 Overview

{project_description}"""
    
    def _get_badges_template(self) -> str:
        return """![GitHub stars](https://img.shields.io/github/stars/{repository_path}?style=flat-square)
![GitHub forks](https://img.shields.io/github/forks/{repository_path}?style=flat-square)
![Language](https://img.shields.io/badge/language-{language}-blue?style=flat-square)
![GitHub last commit](https://img.shields.io/github/last-commit/{repository_path}?style=flat-square)
![GitHub repo size](https://img.shields.io/github/repo-size/{repository_path}?style=flat-square)"""
    
    def _get_description_template(self) -> str:
        return """{project_description}"""
    
    def _get_visual_overview_template(self) -> str:
        return """## 📊 Visual Overview

This project includes visual documentation and diagrams:

{visual_insights_formatted}"""
    
    def _get_features_template(self) -> str:
        return """## ✨ Features

{features_list}"""
    
    def _get_technologies_template(self) -> str:
        return """## 🚀 Technologies

- **Language:** {language}
{frameworks_list}
{package_managers_list}
{build_tools_list}"""
    
    def _get_prerequisites_template(self) -> str:
        return """## 📋 Prerequisites

Before running this project, make sure you have the following installed:

{prerequisites_list}"""
    
    def _get_installation_template(self) -> str:
        return """## 🔧 Installation

{installation_steps_formatted}"""
    
    def _get_usage_template(self) -> str:
        return """## 💻 Usage

{usage_examples_formatted}"""
    
    def _get_project_structure_template(self) -> str:
        return """## 📁 Project Structure

{project_structure}"""
    
    def _get_api_documentation_template(self) -> str:
        return """## 📚 API Documentation

This {project_type} provides the following API endpoints:

<!-- Add API documentation here -->"""
    
    def _get_configuration_template(self) -> str:
        return """## ⚙️ Configuration

Configure the application using the following methods:

<!-- Add configuration details here -->"""
    
    def _get_testing_template(self) -> str:
        return """## 🧪 Testing

Run tests using the following commands:

{testing_commands}"""
    
    def _get_deployment_template(self) -> str:
        return """## 🚀 Deployment

Deploy this application using:

<!-- Add deployment instructions here -->"""
    
    def _get_contributing_template(self) -> str:
        return """## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request"""
    
    def _get_license_template(self) -> str:
        return """## 📄 License

{license_text}"""
    
    def _get_acknowledgments_template(self) -> str:
        return """## 🙏 Acknowledgments

- Thanks to all contributors
- Built with {language} and {', '.join(frameworks[:2]) if frameworks else 'standard tools'}"""
    
    # Specialized templates
    def _create_python_template(self, base_template: DocumentationTemplate) -> DocumentationTemplate:
        """Create Python-specific template."""
        python_template = base_template.model_copy()
        python_template.language = "Python"
        python_template.name = "Python Template"
        
        # Customize installation section for Python
        for section in python_template.sections:
            if section.name == "installation":
                section.content = """## 🔧 Installation

### Using pip

```bash
# Clone the repository
git clone {repository_url}
cd {project_name}

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### Using Poetry (if available)

```bash
# Clone the repository
git clone {repository_url}
cd {project_name}

# Install dependencies
poetry install
poetry shell
```

{installation_steps_formatted}"""
        
        return python_template
    
    def _create_javascript_template(self, base_template: DocumentationTemplate) -> DocumentationTemplate:
        """Create JavaScript/TypeScript-specific template."""
        js_template = base_template.model_copy()
        js_template.language = "JavaScript"
        js_template.name = "JavaScript Template"
        
        # Customize installation section for JavaScript
        for section in js_template.sections:
            if section.name == "installation":
                section.content = """## 🔧 Installation

### Using npm

```bash
# Clone the repository
git clone {repository_url}
cd {project_name}

# Install dependencies
npm install

# Start development server
npm start
```

### Using yarn

```bash
# Clone the repository
git clone {repository_url}
cd {project_name}

# Install dependencies
yarn install

# Start development server
yarn start
```

{installation_steps_formatted}"""
        
        return js_template
    
    def _create_java_template(self, base_template: DocumentationTemplate) -> DocumentationTemplate:
        """Create Java-specific template."""
        java_template = base_template.model_copy()
        java_template.language = "Java"
        java_template.name = "Java Template"
        
        return java_template
    
    def _create_go_template(self, base_template: DocumentationTemplate) -> DocumentationTemplate:
        """Create Go-specific template."""
        go_template = base_template.model_copy()
        go_template.language = "Go"
        go_template.name = "Go Template"
        
        return go_template
    
    def _create_web_app_template(self, base_template: DocumentationTemplate) -> DocumentationTemplate:
        """Create web application template."""
        web_template = base_template.model_copy()
        web_template.project_types = ["web_application"]
        web_template.name = "Web Application Template"
        
        return web_template
    
    def _create_cli_template(self, base_template: DocumentationTemplate) -> DocumentationTemplate:
        """Create CLI tool template."""
        cli_template = base_template.model_copy()
        cli_template.project_types = ["cli_tool"]
        cli_template.name = "CLI Tool Template"
        
        return cli_template
    
    def _create_library_template(self, base_template: DocumentationTemplate) -> DocumentationTemplate:
        """Create library template."""
        lib_template = base_template.model_copy()
        lib_template.project_types = ["library"]
        lib_template.name = "Library Template"
        
        return lib_template
    
    def format_context_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Format context data for template rendering."""
        formatted_context = context.copy()
        
        # Format frameworks list
        if context.get("frameworks"):
            frameworks_list = "\n".join([f"- **{fw}**" for fw in context["frameworks"]])
            formatted_context["frameworks_list"] = f"- **Frameworks:** \n{frameworks_list}"
        else:
            formatted_context["frameworks_list"] = ""
        
        # Format package managers list
        if context.get("package_managers"):
            pm_list = ", ".join(context["package_managers"])
            formatted_context["package_managers_list"] = f"- **Package Managers:** {pm_list}"
        else:
            formatted_context["package_managers_list"] = ""
        
        # Format build tools list
        if context.get("build_tools"):
            bt_list = ", ".join(context["build_tools"])
            formatted_context["build_tools_list"] = f"- **Build Tools:** {bt_list}"
        else:
            formatted_context["build_tools_list"] = ""
        
        # Format installation steps
        if context.get("installation_steps"):
            steps_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(context["installation_steps"])])
            formatted_context["installation_steps_formatted"] = steps_text
        else:
            formatted_context["installation_steps_formatted"] = "1. Clone the repository\n2. Follow language-specific setup instructions"
        
        # Format usage examples
        if context.get("usage_examples"):
            usage_text = "\n\n".join(context["usage_examples"])
            formatted_context["usage_examples_formatted"] = usage_text
        else:
            formatted_context["usage_examples_formatted"] = "See the source code for usage examples."
        
        # Format visual insights
        if context.get("visual_insights"):
            visual_text = ""
            for insight in context["visual_insights"][:3]:  # Limit to top 3
                visual_text += f"- **{insight.image_path}**: {insight.description[:100]}...\n"
            formatted_context["visual_insights_formatted"] = visual_text
        else:
            formatted_context["visual_insights_formatted"] = ""
        
        # Format prerequisites
        lang = context.get("language", "").lower()
        prerequisites = []
        
        if lang == "python":
            prerequisites.extend(["Python 3.8+", "pip or poetry"])
        elif lang in ["javascript", "typescript"]:
            prerequisites.extend(["Node.js 16+", "npm or yarn"])
        elif lang == "java":
            prerequisites.extend(["Java 11+", "Maven or Gradle"])
        elif lang == "go":
            prerequisites.extend(["Go 1.19+"])
        elif lang == "rust":
            prerequisites.extend(["Rust 1.70+", "Cargo"])
        else:
            prerequisites.append(f"{context.get('language', 'Required runtime')} installed")
        
        if context.get("package_managers"):
            for pm in context["package_managers"]:
                if pm not in [p.lower() for p in prerequisites]:
                    prerequisites.append(pm)
        
        formatted_context["prerequisites_list"] = "\n".join([f"- {req}" for req in prerequisites])
        
        # Format testing commands
        if context.get("testing_frameworks"):
            test_commands = []
            for tf in context["testing_frameworks"]:
                if tf == "pytest":
                    test_commands.append("```bash\npytest\n```")
                elif tf == "jest":
                    test_commands.append("```bash\nnpm test\n```")
                elif tf == "go test":
                    test_commands.append("```bash\ngo test ./...\n```")
                else:
                    test_commands.append(f"```bash\n# Run {tf} tests\n```")
            formatted_context["testing_commands"] = "\n\n".join(test_commands)
        else:
            formatted_context["testing_commands"] = "```bash\n# Add test commands here\n```"
        
        # Format license text
        license_name = context.get("license", "")
        if license_name:
            formatted_context["license_text"] = f"This project is licensed under the {license_name} License."
        else:
            formatted_context["license_text"] = "See the LICENSE file for details."
        
        # Generate dynamic features list
        features = []
        features.append(f"🔧 **{context.get('language', 'Modern')}** implementation with best practices")
        
        if context.get("frameworks"):
            frameworks_text = ", ".join(context["frameworks"][:3])
            features.append(f"⚡ Built with **{frameworks_text}**")
        
        project_type = context.get("project_type", "").replace("_", " ").title()
        if project_type and project_type != "Unknown":
            features.append(f"🏗️ **{project_type}** architecture")
        
        if context.get("testing_frameworks"):
            features.append(f"🧪 Comprehensive testing with **{', '.join(context['testing_frameworks'])}**")
        
        if context.get("build_tools"):
            features.append(f"📦 Build tools: **{', '.join(context['build_tools'])}**")
        
        if context.get("package_managers"):
            features.append(f"📋 Package management with **{', '.join(context['package_managers'])}**")
        
        # Add generic features if none specific found
        if len(features) == 1:  # Only has the language feature
            features.extend([
                "📱 Responsive and modern design",
                "🔒 Secure implementation",
                "🚀 Optimized for performance"
            ])
        
        formatted_context["features_list"] = "\n".join([f"- {feature}" for feature in features])
        
        return formatted_context