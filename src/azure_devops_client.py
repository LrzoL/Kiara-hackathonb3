"""Azure DevOps client for accessing repositories and generating documentation."""

import base64
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlparse

import httpx
from pydantic import BaseModel

from .config import Settings

logger = logging.getLogger(__name__)


class AzureDevOpsConfig(BaseModel):
    """Azure DevOps configuration."""
    
    organization: str
    project: str
    repository: str
    pat_token: str
    api_version: str = "7.1"


class AzureDevOpsFileInfo(BaseModel):
    """Information about an Azure DevOps repository file."""
    
    object_id: str
    path: str
    url: str
    content_type: str = "file"
    size: Optional[int] = None
    is_folder: bool = False
    commit_id: Optional[str] = None


class AzureDevOpsRepositoryInfo(BaseModel):
    """Azure DevOps repository information."""
    
    id: str
    name: str
    project: str
    organization: str
    url: str
    web_url: str
    clone_url: str
    default_branch: str
    size: Optional[int] = None


class AzureDevOpsClient:
    """Client for Azure DevOps REST API operations."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = "https://dev.azure.com"
        self.api_version = "7.1"
        
        # HTTP client with authentication
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "User-Agent": "GitHub-Doc-Agent/1.0",
                "Accept": "application/json",
            }
        )
    
    def parse_repository_url(self, url: str) -> Optional[AzureDevOpsConfig]:
        """Parse Azure DevOps repository URL and extract components."""
        
        # Azure DevOps URL formats:
        # https://dev.azure.com/{organization}/{project}/_git/{repository}
        # https://{organization}.visualstudio.com/{project}/_git/{repository}
        # https://{organization}.visualstudio.com/DefaultCollection/{project}/_git/{repository}
        
        parsed = urlparse(url.strip())
        
        if not parsed.netloc:
            return None
        
        # Handle dev.azure.com format
        if parsed.netloc == "dev.azure.com":
            path_parts = parsed.path.strip("/").split("/")
            if len(path_parts) >= 4 and path_parts[2] == "_git":
                organization = path_parts[0]
                project = path_parts[1]
                repository = path_parts[3]
                
                return AzureDevOpsConfig(
                    organization=organization,
                    project=project,
                    repository=repository,
                    pat_token=getattr(self.settings, 'azure_devops_pat', '')
                )
        
        # Handle visualstudio.com format
        elif parsed.netloc.endswith(".visualstudio.com"):
            organization = parsed.netloc.split(".")[0]
            path_parts = parsed.path.strip("/").split("/")
            
            # Handle with/without DefaultCollection
            if len(path_parts) >= 3 and path_parts[-2] == "_git":
                repository = path_parts[-1]
                if path_parts[0] == "DefaultCollection":
                    project = path_parts[1] if len(path_parts) > 3 else "DefaultCollection"
                else:
                    project = path_parts[0]
                
                return AzureDevOpsConfig(
                    organization=organization,
                    project=project,
                    repository=repository,
                    pat_token=getattr(self.settings, 'azure_devops_pat', '')
                )
        
        return None
    
    def _get_auth_header(self, pat_token: str) -> Dict[str, str]:
        """Get authentication header for Azure DevOps API."""
        
        # Azure DevOps uses Basic auth with PAT token
        # Username can be empty, PAT token goes in password field
        auth_string = f":{pat_token}"
        auth_bytes = auth_string.encode('ascii')
        auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
        
        return {"Authorization": f"Basic {auth_b64}"}
    
    async def get_repository_info(self, config: AzureDevOpsConfig) -> AzureDevOpsRepositoryInfo:
        """Get repository information from Azure DevOps."""
        
        url = (f"{self.base_url}/{config.organization}/{config.project}/"
               f"_apis/git/repositories/{config.repository}")
        
        params = {"api-version": config.api_version}
        headers = self._get_auth_header(config.pat_token)
        
        logger.info(f"Fetching repository info from: {url}")
        
        response = await self.client.get(url, params=params, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        return AzureDevOpsRepositoryInfo(
            id=data["id"],
            name=data["name"],
            project=data["project"]["name"],
            organization=config.organization,
            url=data["url"],
            web_url=data["webUrl"],
            clone_url=data["remoteUrl"],
            default_branch=data.get("defaultBranch", "main").replace("refs/heads/", ""),
            size=data.get("size")
        )
    
    async def list_repository_files(
        self, 
        config: AzureDevOpsConfig, 
        path: str = "",
        recursion_level: str = "Full"
    ) -> List[AzureDevOpsFileInfo]:
        """List all files in repository with recursion."""
        
        url = (f"{self.base_url}/{config.organization}/{config.project}/"
               f"_apis/git/repositories/{config.repository}/items")
        
        params = {
            "api-version": config.api_version,
            "recursionLevel": recursion_level,
        }
        
        if path:
            params["scopePath"] = path
        
        headers = self._get_auth_header(config.pat_token)
        
        logger.info(f"Listing repository files: {url}")
        
        response = await self.client.get(url, params=params, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        files = []
        
        for item in data.get("value", []):
            # Skip root folder
            if item.get("path") == "/":
                continue
                
            files.append(AzureDevOpsFileInfo(
                object_id=item["objectId"],
                path=item["path"],
                url=item["url"],
                content_type=item.get("gitObjectType", "blob"),
                size=item.get("size"),
                is_folder=item.get("isFolder", False),
                commit_id=item.get("commitId")
            ))
        
        # Filter out folders, keep only files
        files = [f for f in files if not f.is_folder]
        
        logger.info(f"Found {len(files)} files in repository")
        return files
    
    async def get_file_content(self, config: AzureDevOpsConfig, file_path: str) -> Optional[str]:
        """Get content of a specific file."""
        
        url = (f"{self.base_url}/{config.organization}/{config.project}/"
               f"_apis/git/repositories/{config.repository}/items")
        
        params = {
            "path": file_path,
            "api-version": config.api_version,
            "includeContent": "true"
        }
        
        headers = self._get_auth_header(config.pat_token)
        
        try:
            response = await self.client.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            # Azure DevOps returns file content directly as text for text files
            content = response.text
            
            # Try to parse as JSON to see if it's metadata
            try:
                json_data = json.loads(content)
                if "content" in json_data:
                    # Content is in JSON response
                    return json_data["content"]
                elif "value" in json_data and json_data["value"]:
                    # Multiple items returned, get first one
                    first_item = json_data["value"][0]
                    if "content" in first_item:
                        return first_item["content"]
            except json.JSONDecodeError:
                # Not JSON, return as-is (direct file content)
                return content
            
            return content
            
        except httpx.HTTPStatusError as e:
            logger.warning(f"Could not fetch file {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching file {file_path}: {e}")
            return None
    
    async def get_binary_file_content(self, config: AzureDevOpsConfig, file_path: str) -> Optional[bytes]:
        """Get binary content of a file (for images, etc.)."""
        
        url = (f"{self.base_url}/{config.organization}/{config.project}/"
               f"_apis/git/repositories/{config.repository}/items")
        
        params = {
            "path": file_path,
            "api-version": config.api_version,
            "download": "true"
        }
        
        headers = self._get_auth_header(config.pat_token)
        
        try:
            response = await self.client.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            return response.content
            
        except httpx.HTTPStatusError as e:
            logger.warning(f"Could not fetch binary file {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching binary file {file_path}: {e}")
            return None
    
    async def collect_repository_data(self, repository_url: str) -> Dict[str, Any]:
        """Collect complete repository data for documentation generation."""
        
        config = self.parse_repository_url(repository_url)
        if not config:
            raise ValueError(f"Invalid Azure DevOps repository URL: {repository_url}")
        
        if not config.pat_token:
            raise ValueError("Azure DevOps PAT token is required")
        
        logger.info(f"Collecting data for Azure DevOps repository: {repository_url}")
        
        # Get repository information
        repo_info = await self.get_repository_info(config)
        
        # List all files
        files = await self.list_repository_files(config)
        
        # Filter important files for content analysis
        important_files = self._identify_important_files(files)
        
        # Read file contents
        file_contents = {}
        for file_info in important_files:
            content = await self.get_file_content(config, file_info.path)
            if content:
                file_contents[file_info.path] = content
        
        # Process images
        image_files = self._identify_image_files(files)
        images = {}
        
        for file_info in image_files:
            binary_content = await self.get_binary_file_content(config, file_info.path)
            if binary_content:
                # Convert to base64
                images[file_info.path] = base64.b64encode(binary_content).decode('utf-8')
        
        # Convert to MCP-compatible format
        mcp_data = {
            'repository': {
                'name': repo_info.name,
                'full_name': f"{repo_info.organization}/{repo_info.project}/{repo_info.name}",
                'description': f"Azure DevOps repository: {repo_info.name}",
                'language': None,  # Will be detected
                'private': True,  # Assume private for Azure DevOps
                'html_url': repo_info.web_url,
                'clone_url': repo_info.clone_url,
                'homepage': None,
                'topics': [],
                'license': None,
                'size': repo_info.size,
                'created_at': None,
                'updated_at': None,
                'pushed_at': None,
                'stargazers_count': 0,
                'watchers_count': 0,
                'forks_count': 0,
                'open_issues_count': 0,
                'default_branch': repo_info.default_branch,
            },
            'files': [
                {
                    'name': file_info.path.split('/')[-1],
                    'path': file_info.path,
                    'size': file_info.size,
                    'type': 'file',
                    'download_url': None,
                    'git_url': file_info.url,
                    'html_url': None,
                    'sha': file_info.object_id,
                }
                for file_info in files
            ],
            'file_contents': file_contents,
            'images': images,
            'recent_commits': [],  # Could be implemented later
            'metadata': {
                'source': 'azure_devops',
                'organization': repo_info.organization,
                'project': repo_info.project,
                'total_files': len(files),
                'text_files_count': len(file_contents),
                'images_count': len(images),
            },
        }
        
        logger.info(f"Azure DevOps data collection complete: {len(files)} files, "
                   f"{len(file_contents)} text files, {len(images)} images")
        
        return mcp_data
    
    def _identify_important_files(self, files: List[AzureDevOpsFileInfo]) -> List[AzureDevOpsFileInfo]:
        """Identify important files to read for analysis."""
        
        important_extensions = {
            '.md', '.txt', '.rst', '.asciidoc', '.adoc',
            '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.scss', '.sass',
            '.java', '.c', '.cpp', '.h', '.hpp', '.cs', '.php', '.rb', '.go',
            '.rs', '.swift', '.kt', '.scala', '.clj', '.hs', '.ml', '.fs',
            '.yml', '.yaml', '.json', '.xml', '.toml', '.ini', '.cfg', '.conf',
            '.sql', '.r', '.m', '.pl', '.lua', '.vim', '.elisp',
            '.dockerfile', '.makefile', '.cmake', '.gradle',
        }
        
        important_filenames = {
            'readme', 'readme.md', 'readme.txt', 'readme.rst',
            'license', 'license.md', 'license.txt',
            'changelog', 'changelog.md', 'changelog.txt',
            'contributing', 'contributing.md',
            'package.json', 'requirements.txt', 'pyproject.toml',
            'pom.xml', 'build.gradle', 'cargo.toml',
            'go.mod', 'composer.json', 'gemfile',
            'dockerfile', 'docker-compose.yml',
            'makefile', 'makefile.am', 'cmake.txt',
            '.gitignore', '.dockerignore',
        }
        
        important_files = []
        
        for file_info in files:
            file_name = file_info.path.split('/')[-1].lower()
            file_ext = '.' + file_name.split('.')[-1] if '.' in file_name else ''
            
            # Check by filename or extension
            if (file_name in important_filenames or 
                file_ext in important_extensions or
                any(pattern in file_name for pattern in ['config', 'settings', 'env'])):
                
                # Skip very large files (> 1MB)
                if file_info.size and file_info.size > 1024 * 1024:
                    continue
                    
                important_files.append(file_info)
        
        # Limit to prevent too many API calls
        return important_files[:100]
    
    def _identify_image_files(self, files: List[AzureDevOpsFileInfo]) -> List[AzureDevOpsFileInfo]:
        """Identify image files for visual analysis."""
        
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg'}
        
        image_files = []
        
        for file_info in files:
            file_name = file_info.path.split('/')[-1].lower()
            file_ext = '.' + file_name.split('.')[-1] if '.' in file_name else ''
            
            if file_ext in image_extensions:
                # Skip very large images (> 5MB)
                if file_info.size and file_info.size > 5 * 1024 * 1024:
                    continue
                    
                image_files.append(file_info)
        
        # Limit to prevent too many API calls and memory usage
        return image_files[:20]
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()