"""MCP Client for GitHub MCP Server communication."""

import asyncio
import base64
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


class RepositoryInfo(BaseModel):
    """Repository information model."""
    
    owner: str
    name: str
    full_name: str
    private: bool
    description: Optional[str] = None
    language: Optional[str] = None
    default_branch: str = "main"
    topics: List[str] = []
    license: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    size: int = 0
    stargazers_count: int = 0
    forks_count: int = 0


class FileInfo(BaseModel):
    """File information model."""
    
    path: str
    name: str
    type: str  # "file" or "dir"
    size: Optional[int] = None
    sha: Optional[str] = None
    download_url: Optional[str] = None


class CommitInfo(BaseModel):
    """Commit information model."""
    
    sha: str
    message: str
    author: Dict[str, Any]
    date: str
    url: str


class MCPError(Exception):
    """MCP-related error."""
    pass


class MCPClient:
    """Client for communicating with GitHub MCP Server via Docker."""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._mcp_process: Optional[asyncio.subprocess.Process] = None
        self._connected = False
        self._container_id: Optional[str] = None
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    async def connect(self) -> None:
        """Connect to the GitHub API directly (fallback mode)."""
        if self._connected:
            return
        
        try:
            # For now, we'll use direct GitHub API calls instead of Docker MCP server
            # This avoids Docker-in-Docker issues when running inside a container
            logger.info("Using direct GitHub API access (Docker MCP unavailable)")
            self._connected = True
            logger.info("Successfully connected to GitHub API directly")
            
        except Exception as e:
            logger.error(f"Failed to connect to GitHub API: {e}")
            raise MCPError(f"Connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from the GitHub API."""
        if not self._connected:
            return
        
        # No cleanup needed for direct API access
        self._connected = False
        logger.info("Disconnected from GitHub API")
    
    async def _start_docker_mcp_server(self) -> None:
        """Start the MCP server in a Docker container."""
        try:
            # Start Docker container with GitHub MCP server in stdio mode
            cmd = [
                "docker", "run", "-d", "-i",
                "-e", f"GITHUB_PERSONAL_ACCESS_TOKEN={self.settings.github.token}",
                "ghcr.io/github/github-mcp-server",
                "stdio"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self._container_id = result.stdout.strip()
            
            logger.info(f"Started GitHub MCP Server in Docker container: {self._container_id}")
            
            # Give the server time to start
            await asyncio.sleep(3)
            
            # Verify container is running
            check_cmd = ["docker", "ps", "-q", "-f", f"id={self._container_id}"]
            check_result = subprocess.run(check_cmd, capture_output=True, text=True)
            
            if not check_result.stdout.strip():
                raise MCPError("MCP server container failed to start or exited")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start MCP server: {e.stderr}")
            raise MCPError(f"Docker MCP server startup failed: {e.stderr}")
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            raise MCPError(f"MCP server startup failed: {e}")
    
    def parse_github_url(self, url: str) -> Dict[str, str]:
        """Parse GitHub repository URL."""
        url = url.strip()
        
        # Handle different URL formats
        if url.startswith("https://github.com/"):
            url = url.replace("https://github.com/", "")
        elif url.startswith("git@github.com:"):
            url = url.replace("git@github.com:", "")
        
        # Remove .git suffix
        if url.endswith(".git"):
            url = url[:-4]
        
        parts = url.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid GitHub URL format: {url}")
        
        return {"owner": parts[0], "name": parts[1]}
    
    async def _execute_mcp_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute an MCP tool via Docker container."""
        if not self._connected or not self._container_id:
            raise MCPError("Not connected to MCP server")
        
        try:
            # Create MCP request payload
            request = {
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": kwargs
                }
            }
            
            # Execute via docker exec
            cmd = [
                "docker", "exec", "-i", self._container_id,
                "sh", "-c", f"echo '{json.dumps(request)}' | cat"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                raise MCPError(f"MCP tool execution failed: {result.stderr}")
            
            # For now, return a placeholder - we'll implement proper MCP protocol later
            return {"tool": tool_name, "args": kwargs, "status": "simulated"}
            
        except subprocess.TimeoutExpired:
            raise MCPError(f"MCP tool '{tool_name}' timed out")
        except Exception as e:
            raise MCPError(f"Failed to execute MCP tool '{tool_name}': {e}")
    
    async def get_repository(self, repo_url: str) -> RepositoryInfo:
        """Get repository information using GitHub MCP Server."""
        if not self._connected:
            await self.connect()
        
        repo_info = self.parse_github_url(repo_url)
        
        try:
            # Use GitHub API directly for now (fallback)
            # TODO: Replace with actual MCP tool call when MCP protocol is implemented
            import httpx
            
            async with httpx.AsyncClient() as client:
                url = f"https://api.github.com/repos/{repo_info['owner']}/{repo_info['name']}"
                headers = {
                    "Authorization": f"token {self.settings.github.token}",
                    "Accept": "application/vnd.github.v3+json",
                    "User-Agent": "github-doc-agent/0.1.0"
                }
                
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                return RepositoryInfo(
                    owner=data["owner"]["login"],
                    name=data["name"],
                    full_name=data["full_name"],
                    private=data["private"],
                    description=data.get("description"),
                    language=data.get("language"),
                    default_branch=data.get("default_branch", "main"),
                    topics=data.get("topics", []),
                    license=data.get("license"),
                    created_at=data.get("created_at"),
                    updated_at=data.get("updated_at"),
                    size=data.get("size", 0),
                    stargazers_count=data.get("stargazers_count", 0),
                    forks_count=data.get("forks_count", 0)
                )
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise MCPError(f"Repository not found: {repo_url}")
            elif e.response.status_code == 403:
                raise MCPError(f"Access denied to repository: {repo_url}")
            else:
                raise MCPError(f"HTTP error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise MCPError(f"Failed to get repository info: {e}")
    
    async def list_repository_contents(
        self, 
        repo_url: str, 
        path: str = "", 
        recursive: bool = False
    ) -> List[FileInfo]:
        """List repository contents using GitHub MCP Server."""
        if not self._connected:
            await self.connect()
        
        repo_info = self.parse_github_url(repo_url)
        
        try:
            # Use GitHub API directly for now (fallback)
            # TODO: Replace with actual MCP tool call: get_file_contents
            import httpx
            
            async with httpx.AsyncClient() as client:
                url = f"https://api.github.com/repos/{repo_info['owner']}/{repo_info['name']}/contents"
                
                if path:
                    url += f"/{path}"
                
                headers = {
                    "Authorization": f"token {self.settings.github.token}",
                    "Accept": "application/vnd.github.v3+json",
                    "User-Agent": "github-doc-agent/0.1.0"
                }
                
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                files = []
                items = data if isinstance(data, list) else [data]
                
                for item in items:
                    file_info = FileInfo(
                        path=item["path"],
                        name=item["name"],
                        type=item["type"],
                        size=item.get("size"),
                        sha=item.get("sha"),
                        download_url=item.get("download_url")
                    )
                    files.append(file_info)
                    
                    # Recursively list directories if requested
                    if recursive and item["type"] == "dir":
                        subfiles = await self.list_repository_contents(
                            repo_url, item["path"], recursive=True
                        )
                        files.extend(subfiles)
                
                return files
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise MCPError(f"Path not found: {path}")
            else:
                raise MCPError(f"HTTP error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise MCPError(f"Failed to list repository contents: {e}")
    
    async def get_file_contents(self, repo_url: str, file_path: str) -> str:
        """Get file contents as text using GitHub MCP Server."""
        if not self._connected:
            await self.connect()
        
        repo_info = self.parse_github_url(repo_url)
        
        try:
            # Use GitHub API directly for now (fallback)
            # TODO: Replace with actual MCP tool call: get_file_contents
            import httpx
            
            async with httpx.AsyncClient() as client:
                url = f"https://api.github.com/repos/{repo_info['owner']}/{repo_info['name']}/contents/{file_path}"
                headers = {
                    "Authorization": f"token {self.settings.github.token}",
                    "Accept": "application/vnd.github.v3+json",
                    "User-Agent": "github-doc-agent/0.1.0"
                }
                
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                if data["type"] != "file":
                    raise MCPError(f"Path is not a file: {file_path}")
                
                # Decode base64 content
                content = base64.b64decode(data["content"]).decode("utf-8")
                return content
                
        except UnicodeDecodeError:
            raise MCPError(f"File is not text-readable: {file_path}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise MCPError(f"File not found: {file_path}")
            else:
                raise MCPError(f"HTTP error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise MCPError(f"Failed to get file contents: {e}")
    
    async def get_binary_file_contents(self, repo_url: str, file_path: str) -> bytes:
        """Get binary file contents using GitHub MCP Server."""
        if not self._connected:
            await self.connect()
        
        repo_info = self.parse_github_url(repo_url)
        
        try:
            # Use GitHub API directly for now (fallback)
            # TODO: Replace with actual MCP tool call: get_file_contents
            import httpx
            
            async with httpx.AsyncClient() as client:
                url = f"https://api.github.com/repos/{repo_info['owner']}/{repo_info['name']}/contents/{file_path}"
                headers = {
                    "Authorization": f"token {self.settings.github.token}",
                    "Accept": "application/vnd.github.v3+json",
                    "User-Agent": "github-doc-agent/0.1.0"
                }
                
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                if data["type"] != "file":
                    raise MCPError(f"Path is not a file: {file_path}")
                
                # Decode base64 content to bytes
                content = base64.b64decode(data["content"])
                return content
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise MCPError(f"File not found: {file_path}")
            else:
                raise MCPError(f"HTTP error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise MCPError(f"Failed to get binary file contents: {e}")
    
    async def list_commits(self, repo_url: str, limit: int = 10) -> List[CommitInfo]:
        """List recent commits using GitHub MCP Server."""
        if not self._connected:
            await self.connect()
        
        repo_info = self.parse_github_url(repo_url)
        
        try:
            # Use GitHub API directly for now (fallback)
            # TODO: Replace with actual MCP tool call: list_commits
            import httpx
            
            async with httpx.AsyncClient() as client:
                url = f"https://api.github.com/repos/{repo_info['owner']}/{repo_info['name']}/commits"
                headers = {
                    "Authorization": f"token {self.settings.github.token}",
                    "Accept": "application/vnd.github.v3+json",
                    "User-Agent": "github-doc-agent/0.1.0"
                }
                params = {"per_page": limit}
                
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                commits = []
                for item in data:
                    commit_info = CommitInfo(
                        sha=item["sha"],
                        message=item["commit"]["message"],
                        author=item["commit"]["author"],
                        date=item["commit"]["author"]["date"],
                        url=item["html_url"]
                    )
                    commits.append(commit_info)
                
                return commits
                
        except httpx.HTTPStatusError as e:
            raise MCPError(f"HTTP error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise MCPError(f"Failed to list commits: {e}")
    
    async def get_latest_commit(self, repo_url: str) -> CommitInfo:
        """Get the latest commit for cache validation."""
        if not self._connected:
            await self.connect()
        
        repo_info = self.parse_github_url(repo_url)
        
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                url = f"https://api.github.com/repos/{repo_info['owner']}/{repo_info['name']}/commits"
                headers = {
                    "Authorization": f"token {self.settings.github.token}",
                    "Accept": "application/vnd.github.v3+json",
                    "User-Agent": "github-doc-agent/0.1.0"
                }
                params = {"per_page": 1}
                
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    raise MCPError("No commits found in repository")
                
                item = data[0]
                return CommitInfo(
                    sha=item["sha"],
                    message=item["commit"]["message"],
                    author=item["commit"]["author"],
                    date=item["commit"]["author"]["date"],
                    url=item["html_url"]
                )
                
        except httpx.HTTPStatusError as e:
            raise MCPError(f"HTTP error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise MCPError(f"Failed to get latest commit: {e}")
    
    def identify_image_files(self, files: List[FileInfo]) -> List[str]:
        """Identify image files from file list."""
        image_files = []
        
        for file_info in files:
            if file_info.type == "file" and self.settings.is_image_file(file_info.name):
                # Check file size if available
                if file_info.size and file_info.size > self.settings.vision.max_image_size:
                    logger.warning(f"Skipping large image file: {file_info.path} ({file_info.size} bytes)")
                    continue
                image_files.append(file_info.path)
        
        return image_files
    
    def identify_important_files(self, files: List[FileInfo]) -> List[str]:
        """Identify important files for analysis."""
        important_extensions = {
            ".md", ".txt", ".rst", ".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c", ".h",
            ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".xml", ".html", ".css",
            ".dockerfile", ".gitignore", ".gitattributes", ".editorconfig", ".ipynb", ".r", ".R",
            ".scala", ".kt", ".swift", ".php", ".rb", ".sh", ".bat", ".ps1"
        }
        
        important_filenames = {
            "readme", "license", "changelog", "contributing", "dockerfile", "makefile", 
            "package.json", "requirements.txt", "setup.py", "pyproject.toml", "cargo.toml",
            "go.mod", "pom.xml", "build.gradle", "composer.json", "gemfile"
        }
        
        important_files = []
        
        for file_info in files:
            if file_info.type != "file":
                continue
            
            # Check by extension
            file_ext = Path(file_info.name).suffix.lower()
            if file_ext in important_extensions:
                important_files.append(file_info.path)
                continue
            
            # Check by filename
            filename_lower = file_info.name.lower()
            if any(important in filename_lower for important in important_filenames):
                important_files.append(file_info.path)
        
        return important_files
    
    async def collect_project_data(self, repo_url: str) -> Dict[str, Any]:
        """Collect comprehensive project data."""
        if not self._connected:
            await self.connect()
        
        logger.info(f"Collecting project data for: {repo_url}")
        
        # Get repository information
        repository = await self.get_repository(repo_url)
        
        # Get file structure
        files = await self.list_repository_contents(repo_url, recursive=True)
        
        # Identify and download images
        image_files = self.identify_image_files(files)
        images = {}
        
        for image_path in image_files[:10]:  # Limit to first 10 images
            try:
                image_content = await self.get_binary_file_contents(repo_url, image_path)
                images[image_path] = base64.b64encode(image_content).decode("utf-8")
                logger.info(f"Downloaded image: {image_path}")
            except Exception as e:
                logger.warning(f"Failed to download image {image_path}: {e}")
        
        # Get important file contents
        important_files = self.identify_important_files(files)
        file_contents = {}
        
        for file_path in important_files[:20]:  # Limit to first 20 files
            try:
                content = await self.get_file_contents(repo_url, file_path)
                file_contents[file_path] = content
                logger.info(f"Read file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to read file {file_path}: {e}")
        
        # Get recent commits
        try:
            recent_commits = await self.list_commits(repo_url, limit=5)
        except Exception as e:
            logger.warning(f"Failed to get commits: {e}")
            recent_commits = []
        
        return {
            "repository": repository.model_dump(),
            "files": [f.model_dump() for f in files],
            "images": images,
            "file_contents": file_contents,
            "recent_commits": [c.model_dump() for c in recent_commits]
        }