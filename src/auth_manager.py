"""Authentication manager for GitHub access."""

import logging
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


class GitHubScope(BaseModel):
    """GitHub token scope information."""
    
    name: str
    description: str


class TokenInfo(BaseModel):
    """GitHub token information."""
    
    scopes: List[str]
    valid: bool
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[int] = None
    user: Optional[str] = None


class AuthError(Exception):
    """Authentication-related error."""
    pass


class AuthManager:
    """Manages GitHub authentication and permissions."""
    
    REQUIRED_SCOPES = {
        "repo": "Full control of private repositories",
        "read:user": "Read user profile data",
        "user:email": "Access user email addresses"
    }
    
    OPTIONAL_SCOPES = {
        "read:org": "Read organization data",
        "read:public_key": "Read public SSH keys",
        "read:repo_hook": "Read repository hooks"
    }
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._session: Optional[httpx.AsyncClient] = None
        self._token_info: Optional[TokenInfo] = None
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
    
    async def initialize(self) -> None:
        """Initialize the authentication manager."""
        self._session = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            headers={
                "Authorization": f"token {self.settings.github.token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "github-doc-agent/0.1.0"
            }
        )
        
        # Validate token on initialization
        await self.validate_token()
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._session:
            await self._session.aclose()
            self._session = None
    
    async def validate_token(self) -> TokenInfo:
        """Validate GitHub token and get information."""
        if not self.settings.github.token:
            raise AuthError("GitHub token is not configured")
        
        try:
            # Get token information
            response = await self._session.get(f"{self.settings.github.api_url}/user")
            response.raise_for_status()
            
            user_data = response.json()
            scopes = response.headers.get("X-OAuth-Scopes", "").split(", ")
            scopes = [scope.strip() for scope in scopes if scope.strip()]
            
            # Get rate limit information
            rate_limit_response = await self._session.get(f"{self.settings.github.api_url}/rate_limit")
            rate_limit_data = rate_limit_response.json()
            
            self._token_info = TokenInfo(
                scopes=scopes,
                valid=True,
                rate_limit_remaining=rate_limit_data["rate"]["remaining"],
                rate_limit_reset=rate_limit_data["rate"]["reset"],
                user=user_data.get("login")
            )
            
            logger.info(f"Token validated for user: {self._token_info.user}")
            logger.info(f"Available scopes: {', '.join(scopes)}")
            logger.info(f"Rate limit remaining: {self._token_info.rate_limit_remaining}")
            
            return self._token_info
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise AuthError("Invalid GitHub token")
            elif e.response.status_code == 403:
                raise AuthError("GitHub token lacks required permissions")
            else:
                raise AuthError(f"Token validation failed: {e.response.status_code}")
        except Exception as e:
            raise AuthError(f"Token validation error: {e}")
    
    def check_required_scopes(self) -> Dict[str, bool]:
        """Check if token has required scopes."""
        if not self._token_info:
            raise AuthError("Token not validated. Call validate_token() first.")
        
        scope_status = {}
        
        for scope, description in self.REQUIRED_SCOPES.items():
            has_scope = scope in self._token_info.scopes
            scope_status[scope] = has_scope
            
            if has_scope:
                logger.debug(f"✓ Required scope '{scope}': {description}")
            else:
                logger.debug(f"✗ Missing required scope '{scope}': {description}")
        
        return scope_status
    
    def check_optional_scopes(self) -> Dict[str, bool]:
        """Check if token has optional scopes."""
        if not self._token_info:
            raise AuthError("Token not validated. Call validate_token() first.")
        
        scope_status = {}
        
        for scope, description in self.OPTIONAL_SCOPES.items():
            has_scope = scope in self._token_info.scopes
            scope_status[scope] = has_scope
            
            if has_scope:
                logger.info(f"✓ Optional scope '{scope}': {description}")
            else:
                logger.info(f"○ Optional scope '{scope}' not available: {description}")
        
        return scope_status
    
    def can_access_private_repos(self) -> bool:
        """Check if token can access private repositories."""
        if not self._token_info:
            return False
        
        return "repo" in self._token_info.scopes
    
    def can_access_organization_repos(self) -> bool:
        """Check if token can access organization repositories."""
        if not self._token_info:
            return False
        
        return "read:org" in self._token_info.scopes or "repo" in self._token_info.scopes
    
    async def check_repository_access(self, owner: str, repo: str) -> Dict[str, bool]:
        """Check access permissions for a specific repository."""
        if not self._session:
            raise AuthError("Authentication manager not initialized")
        
        url = f"{self.settings.github.api_url}/repos/{owner}/{repo}"
        
        try:
            response = await self._session.get(url)
            
            if response.status_code == 200:
                repo_data = response.json()
                return {
                    "exists": True,
                    "accessible": True,
                    "private": repo_data.get("private", False),
                    "admin": False,  # Would need to check collaborator permissions
                    "push": False,   # Would need to check collaborator permissions
                    "pull": True
                }
            elif response.status_code == 404:
                # Could be private repo without access or non-existent
                return {
                    "exists": False,
                    "accessible": False,
                    "private": None,
                    "admin": False,
                    "push": False,
                    "pull": False
                }
            elif response.status_code == 403:
                return {
                    "exists": True,
                    "accessible": False,
                    "private": True,
                    "admin": False,
                    "push": False,
                    "pull": False
                }
            else:
                raise AuthError(f"Unexpected response: {response.status_code}")
                
        except httpx.HTTPStatusError as e:
            raise AuthError(f"Failed to check repository access: {e}")
        except Exception as e:
            raise AuthError(f"Repository access check error: {e}")
    
    async def get_user_repositories(self, include_private: bool = True) -> List[Dict[str, Any]]:
        """Get list of user repositories."""
        if not self._session:
            raise AuthError("Authentication manager not initialized")
        
        if include_private and not self.can_access_private_repos():
            logger.warning("Token cannot access private repositories")
            include_private = False
        
        url = f"{self.settings.github.api_url}/user/repos"
        params = {
            "visibility": "all" if include_private else "public",
            "sort": "updated",
            "per_page": 100
        }
        
        try:
            response = await self._session.get(url, params=params)
            response.raise_for_status()
            
            repos = response.json()
            logger.info(f"Found {len(repos)} repositories for user")
            
            return repos
            
        except httpx.HTTPStatusError as e:
            raise AuthError(f"Failed to get user repositories: {e}")
        except Exception as e:
            raise AuthError(f"Repository listing error: {e}")
    
    def get_token_info(self) -> Optional[TokenInfo]:
        """Get current token information."""
        return self._token_info
    
    async def check_rate_limit(self) -> Dict[str, int]:
        """Check current rate limit status."""
        if not self._session:
            raise AuthError("Authentication manager not initialized")
        
        try:
            response = await self._session.get(f"{self.settings.github.api_url}/rate_limit")
            response.raise_for_status()
            
            data = response.json()
            rate_info = data["rate"]
            
            logger.info(f"Rate limit: {rate_info['remaining']}/{rate_info['limit']} remaining")
            
            return {
                "limit": rate_info["limit"],
                "remaining": rate_info["remaining"],
                "reset": rate_info["reset"],
                "used": rate_info["used"]
            }
            
        except Exception as e:
            raise AuthError(f"Rate limit check error: {e}")
    
    def generate_token_setup_instructions(self) -> str:
        """Generate instructions for setting up a GitHub token."""
        missing_scopes = []
        if self._token_info:
            required_status = self.check_required_scopes()
            missing_scopes = [scope for scope, has_scope in required_status.items() if not has_scope]
        
        instructions = """
GitHub Token Setup Instructions:

1. Go to GitHub Settings > Developer settings > Personal access tokens
2. Click "Generate new token (classic)"
3. Give your token a descriptive name like "GitHub Doc Agent"
4. Select the following scopes:

Required scopes:
"""
        
        for scope, description in self.REQUIRED_SCOPES.items():
            status = "❌ MISSING" if scope in missing_scopes else "✅"
            instructions += f"  □ {scope} - {description} {status}\n"
        
        instructions += "\nOptional scopes (for enhanced functionality):\n"
        
        for scope, description in self.OPTIONAL_SCOPES.items():
            instructions += f"  □ {scope} - {description}\n"
        
        instructions += """
5. Click "Generate token"
6. Copy the token and add it to your .env file:
   GITHUB_TOKEN=your_token_here

For private repository access, the 'repo' scope is required.
"""
        
        return instructions