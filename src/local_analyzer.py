"""Local project analyzer for Kiara - analyzes local directories using OCI AI."""

import os
import json
import logging
import mimetypes
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from .config import get_settings
from .oci_analyzer import OCIAnalyzer, LanguageInfo, ArchitectureInfo, ProjectAnalysis

logger = logging.getLogger(__name__)


class LocalProjectAnalyzer:
    """Analyzer for local projects using OCI AI."""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.oci_analyzer = OCIAnalyzer(self.settings)
        
        # File patterns to ignore
        self.ignore_patterns = {
            # Directories
            '__pycache__', '.git', '.vscode', 'node_modules', '.pytest_cache',
            '.mypy_cache', 'dist', 'build', '.next', '.nuxt', 'coverage',
            '.coverage', '.env', '.venv', 'venv', 'env', '.tox',
            
            # File patterns
            '*.pyc', '*.pyo', '*.pyd', '*.so', '*.dll', '*.dylib',
            '*.log', '*.tmp', '*.temp', '*.cache', '*.swp', '*.swo',
            '*.DS_Store', '*.lockb', 'package-lock.json', 'yarn.lock',
            'poetry.lock', 'Pipfile.lock', '*.db', '*.sqlite', '*.sqlite3'
        }
        
        # Important file patterns to prioritize
        self.important_patterns = {
            'README.md', 'readme.md', 'README.txt', 'readme.txt',
            'package.json', 'pyproject.toml', 'setup.py', 'requirements.txt',
            'Cargo.toml', 'go.mod', 'pom.xml', 'build.gradle',
            'tsconfig.json', 'webpack.config.js', 'next.config.js',
            'tailwind.config.js', 'postcss.config.js', '.env.example',
            'docker-compose.yml', 'Dockerfile', 'Makefile'
        }
        
        # Text file extensions we can read
        self.text_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.scss',
            '.less', '.sass', '.json', '.yaml', '.yml', '.toml', '.ini',
            '.cfg', '.conf', '.md', '.txt', '.rst', '.xml', '.sql',
            '.sh', '.bat', '.ps1', '.dockerfile', '.gitignore', '.env'
        }
    
    def should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored."""
        name = path.name
        
        # Check directory patterns
        if path.is_dir():
            if name in self.ignore_patterns or name.startswith('.'):
                return True
        
        # Check file patterns
        for pattern in self.ignore_patterns:
            if '*' in pattern:
                # Simple glob pattern matching
                if pattern.startswith('*.'):
                    ext = pattern[1:]  # Remove *
                    if name.endswith(ext):
                        return True
            else:
                if name == pattern:
                    return True
        
        return False
    
    def is_text_file(self, file_path: Path) -> bool:
        """Check if a file is likely a text file we can read."""
        if file_path.suffix.lower() in self.text_extensions:
            return True
        
        # Check MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type and mime_type.startswith('text/'):
            return True
        
        return False
    
    def scan_directory(self, project_path: Path, max_files: int = 50) -> Dict[str, Any]:
        """Scan a local directory and collect project information."""
        logger.info(f"Scanning local project: {project_path}")
        
        project_data = {
            "repository": {
                "name": project_path.name,
                "description": "",
                "path": str(project_path.absolute()),
                "type": "local_project"
            },
            "files": [],
            "file_contents": {},
            "images": {},
            "directory_structure": []
        }
        
        file_count = 0
        important_files_found = []
        
        try:
            # Walk through directory
            for root, dirs, files in os.walk(project_path):
                root_path = Path(root)
                
                # Filter directories to ignore
                dirs[:] = [d for d in dirs if not self.should_ignore(root_path / d)]
                
                # Process files in current directory
                for file_name in files:
                    file_path = root_path / file_name
                    
                    if self.should_ignore(file_path):
                        continue
                    
                    # Get relative path from project root
                    try:
                        rel_path = file_path.relative_to(project_path)
                    except ValueError:
                        continue
                    
                    # Add to file list
                    file_info = {
                        "type": "file",
                        "path": str(rel_path).replace('\\', '/'),
                        "name": file_name,
                        "size": file_path.stat().st_size if file_path.exists() else 0
                    }
                    
                    project_data["files"].append(file_info)
                    file_count += 1
                    
                    # Check if it's an important file
                    if file_name in self.important_patterns:
                        important_files_found.append(str(rel_path))
                    
                    # Read important text files (limited)
                    if (file_name in self.important_patterns or 
                        len(project_data["file_contents"]) < 10) and self.is_text_file(file_path):
                        
                        try:
                            content = self.read_file_safely(file_path)
                            if content:
                                project_data["file_contents"][str(rel_path).replace('\\', '/')] = content
                        except Exception as e:
                            logger.warning(f"Could not read {file_path}: {e}")
                    
                    # Stop if we have too many files
                    if file_count >= max_files:
                        break
                
                if file_count >= max_files:
                    break
            
            logger.info(f"Scanned {file_count} files, found {len(important_files_found)} important files")
            logger.info(f"Read {len(project_data['file_contents'])} files for content analysis")
            
            return project_data
            
        except Exception as e:
            logger.error(f"Error scanning directory {project_path}: {e}")
            raise
    
    def read_file_safely(self, file_path: Path, max_size: int = 10000) -> Optional[str]:
        """Safely read a text file with size limits."""
        try:
            if file_path.stat().st_size > max_size:
                # Read only first part of large files
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(max_size)
                    return content + "\n... [file truncated]"
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
        except Exception as e:
            logger.debug(f"Could not read {file_path}: {e}")
            return None
    
    async def analyze_local_project(self, project_path: str) -> ProjectAnalysis:
        """Analyze a local project using OCI AI."""
        logger.info(f"Starting local project analysis: {project_path}")
        
        project_path = Path(project_path).resolve()
        
        if not project_path.exists():
            raise FileNotFoundError(f"Project directory does not exist: {project_path}")
        
        if not project_path.is_dir():
            raise ValueError(f"Path is not a directory: {project_path}")
        
        # Scan the directory
        project_data = self.scan_directory(project_path)
        
        # Add project description from README if available
        readme_content = self.extract_readme_description(project_data["file_contents"])
        if readme_content:
            project_data["repository"]["description"] = readme_content
        
        # Use OCI analyzer for AI-powered analysis
        logger.info("Performing OCI AI analysis on local project")
        analysis = await self.oci_analyzer.analyze_project(project_data)
        
        return analysis
    
    def extract_readme_description(self, file_contents: Dict[str, str]) -> str:
        """Extract project description from README files."""
        readme_files = ['README.md', 'readme.md', 'README.txt', 'readme.txt']
        
        for file_path, content in file_contents.items():
            file_name = Path(file_path).name
            if file_name in readme_files:
                # Extract first paragraph or line as description
                lines = content.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('```'):
                        # Remove markdown formatting
                        description = line.replace('*', '').replace('_', '').replace('`', '')
                        if len(description) > 10:  # Meaningful description
                            return description[:200]  # Limit length
                break
        
        return ""
    
    async def generate_local_documentation(
        self, 
        project_path: str, 
        template_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate documentation for a local project."""
        logger.info(f"Generating documentation for local project: {project_path}")
        
        # Scan and analyze the project
        project_path_obj = Path(project_path).resolve()
        project_data = self.scan_directory(project_path_obj)
        
        # Add project description from README if available
        readme_content = self.extract_readme_description(project_data["file_contents"])
        if readme_content:
            project_data["repository"]["description"] = readme_content
        
        # Generate documentation using OCI
        result = await self.oci_analyzer.generate_readme_with_template(
            project_data,
            template_type=template_type
        )
        
        # Add local-specific metadata
        result["local_project"] = True
        result["project_path"] = str(project_path_obj)
        result["files_analyzed"] = len(project_data["files"])
        result["content_files_read"] = len(project_data["file_contents"])
        
        return result


# Utility functions for CLI integration
async def analyze_local_project_cli(project_path: str) -> ProjectAnalysis:
    """CLI helper for local project analysis."""
    analyzer = LocalProjectAnalyzer()
    return await analyzer.analyze_local_project(project_path)


async def generate_local_documentation_cli(
    project_path: str, 
    template_type: Optional[str] = None
) -> Dict[str, Any]:
    """CLI helper for local documentation generation."""
    analyzer = LocalProjectAnalyzer()
    return await analyzer.generate_local_documentation(project_path, template_type)