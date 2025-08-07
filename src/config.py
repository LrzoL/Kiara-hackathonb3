"""Configuration management for GitHub Documentation Agent."""

import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GitHubConfig(BaseSettings):
    """GitHub-related configuration."""
    
    token: Optional[str] = Field(default=None, description="GitHub personal access token")
    api_url: str = Field(default="https://api.github.com", description="GitHub API URL")
    
    model_config = SettingsConfigDict(
        env_prefix="GITHUB_",
        case_sensitive=False
    )


class OracleCloudConfig(BaseSettings):
    """Oracle Cloud Infrastructure AI configuration."""
    
    compartment_id: str = Field(..., description="OCI compartment OCID")
    config_profile: str = Field(default="DEFAULT", description="OCI config profile")
    model_id: str = Field(default="meta.llama-4-maverick-17b-128e-instruct", description="Oracle Cloud AI text model")
    model_vision: str = Field(default="Scout", description="Oracle Cloud AI vision model")
    endpoint: str = Field(..., description="OCI Generative AI endpoint URL")
    max_tokens: int = Field(default=4000, description="Maximum tokens per request")
    temperature: float = Field(default=0.2, description="Model temperature")
    enabled: bool = Field(default=True, description="Enable Oracle Cloud AI")
    
    model_config = SettingsConfigDict(
        env_prefix="OCI_",
        case_sensitive=False
    )


class XAIGrokConfig(BaseSettings):
    """xAI Grok 4 configuration with 131K output tokens support."""
    
    enabled: bool = Field(default=False, description="Enable xAI Grok 4")
    compartment_id: str = Field(default="", description="OCI compartment OCID for xAI Grok")
    config_profile: str = Field(default="DEFAULT", description="OCI config profile") 
    model_id: str = Field(
        default="ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceya3bsfz4ogiuv3yc7gcnlry7gi3zzx6tnikg6jltqszm2q",
        description="xAI Grok 4 model OCID"
    )
    endpoint: str = Field(
        default="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        description="OCI endpoint for xAI Grok"
    )
    max_tokens: int = Field(default=131000, description="Maximum output tokens (131K for Grok 4)")
    temperature: float = Field(default=0, description="Model temperature")
    frequency_penalty: float = Field(default=1, description="Frequency penalty")
    presence_penalty: float = Field(default=0, description="Presence penalty")
    top_p: float = Field(default=1, description="Top-p sampling")
    top_k: int = Field(default=0, description="Top-k sampling")
    
    model_config = SettingsConfigDict(
        env_prefix="XAI_GROK_",
        case_sensitive=False
    )


# Legacy Groq support (deprecated - now using Oracle Cloud)
class LegacyGroqConfig(BaseSettings):
    """Legacy Groq AI configuration (deprecated)."""
    
    api_key: Optional[str] = Field(default=None, description="Legacy Groq API key (deprecated)")
    model_text: str = Field(default="llama-3.1-70b-versatile", description="Legacy text analysis model")
    model_vision: str = Field(default="llama-3.2-90b-vision-preview", description="Legacy vision analysis model")
    max_tokens: int = Field(default=8000, description="Maximum tokens per request")
    temperature: float = Field(default=0.1, description="Model temperature")
    
    model_config = SettingsConfigDict(
        env_prefix="GROQ_",
        case_sensitive=False
    )


class MCPConfig(BaseSettings):
    """MCP Server configuration."""
    
    server_command: str = Field(default="npx", description="MCP server command")
    server_args: List[str] = Field(default=["@github/github-mcp-server"], description="MCP server arguments")
    
    model_config = SettingsConfigDict(
        env_prefix="MCP_",
        case_sensitive=False
    )


class ApplicationConfig(BaseSettings):
    """General application configuration."""
    
    log_level: str = Field(default="INFO", description="Logging level")
    cache_enabled: bool = Field(default=True, description="Enable caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    max_file_size: int = Field(default=10485760, description="Max file size in bytes (10MB)")
    rate_limit_requests: int = Field(default=30, description="Rate limit requests per period")
    rate_limit_period: int = Field(default=60, description="Rate limit period in seconds")
    
    model_config = SettingsConfigDict(
        env_prefix="APP_",
        case_sensitive=False
    )


class VisionConfig(BaseSettings):
    """Vision processing configuration."""
    
    supported_image_formats: List[str] = Field(
        default=["png", "jpg", "jpeg", "gif", "bmp", "webp", "svg"],
        description="Supported image formats"
    )
    max_image_size: int = Field(default=5242880, description="Max image size in bytes (5MB)")
    analysis_enabled: bool = Field(default=True, description="Enable vision analysis")
    
    model_config = SettingsConfigDict(
        env_prefix="VISION_",
        case_sensitive=False
    )


class RepositoryConfig(BaseSettings):
    """Repository access configuration."""
    
    private_repos_enabled: bool = Field(default=True, description="Enable private repository access")
    user_repos_only: bool = Field(default=False, description="Restrict to user repositories only")
    
    model_config = SettingsConfigDict(
        env_prefix="REPO_",
        case_sensitive=False
    )


class RedisConfig(BaseSettings):
    """Redis cache configuration."""
    
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    password: Optional[str] = Field(default=None, description="Redis password")
    db: int = Field(default=0, description="Redis database number")
    ttl: int = Field(default=7200, description="Default TTL in seconds (2 hours)")
    enabled: bool = Field(default=True, description="Enable Redis caching")
    
    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        case_sensitive=False
    )


class Settings(BaseSettings):
    """Main application settings."""
    
    # GitHub settings
    github_token: Optional[str] = Field(default=None, description="GitHub personal access token")
    github_api_url: str = Field(default="https://api.github.com", description="GitHub API URL")
    
    # Azure DevOps settings
    azure_devops_pat: Optional[str] = Field(default=None, description="Azure DevOps Personal Access Token")
    
    # Oracle Cloud Infrastructure AI settings (primary AI provider)
    oci_compartment_id: str = Field(..., description="OCI compartment OCID")
    oci_config_profile: str = Field(default="DEFAULT", description="OCI config profile")
    oci_model_id: str = Field(default="meta.llama-4-maverick-17b-128e-instruct", description="Oracle Cloud AI text model")
    oci_model_vision: str = Field(default="Scout", description="Oracle Cloud AI vision model")
    oci_endpoint: str = Field(..., description="OCI Generative AI endpoint URL")
    oci_max_tokens: int = Field(default=4000, description="Maximum tokens per request")
    oci_temperature: float = Field(default=0.2, description="Model temperature")
    oci_enabled: bool = Field(default=True, description="Enable Oracle Cloud AI")
    
    # xAI Grok 4 settings (high-output model)
    xai_grok_enabled: bool = Field(default=False, description="Enable xAI Grok 4")
    xai_grok_compartment_id: str = Field(default="", description="OCI compartment OCID for xAI Grok")
    xai_grok_config_profile: str = Field(default="DEFAULT", description="OCI config profile")
    xai_grok_model_id: str = Field(
        default="ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceya3bsfz4ogiuv3yc7gcnlry7gi3zzx6tnikg6jltqszm2q",
        description="xAI Grok 4 model OCID"
    )
    xai_grok_endpoint: str = Field(
        default="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        description="OCI endpoint for xAI Grok"
    )
    xai_grok_max_tokens: int = Field(default=131000, description="Maximum output tokens (131K for Grok 4)")
    xai_grok_temperature: float = Field(default=0, description="Model temperature")
    xai_grok_frequency_penalty: float = Field(default=1, description="Frequency penalty")
    xai_grok_presence_penalty: float = Field(default=0, description="Presence penalty")
    xai_grok_top_p: float = Field(default=1, description="Top-p sampling")
    xai_grok_top_k: int = Field(default=0, description="Top-k sampling")
    
    # Legacy support (deprecated)
    groq_api_key: Optional[str] = Field(default=None, description="Legacy API key (deprecated)")
    groq_model_text: str = Field(default="llama-4-maverick-17b-128e-instruct", description="Legacy text analysis model")
    groq_model_vision: str = Field(default="llama-3.2-90b-vision-preview", description="Legacy vision analysis model")
    groq_max_tokens: int = Field(default=8000, description="Maximum tokens per request")
    groq_temperature: float = Field(default=0.1, description="Model temperature")
    
    # MCP settings
    mcp_server_command: str = Field(default="npx", description="MCP server command")
    mcp_server_args: List[str] = Field(default=["@github/github-mcp-server"], description="MCP server arguments")
    
    # Application settings
    app_log_level: str = Field(default="INFO", description="Logging level")
    app_cache_enabled: bool = Field(default=True, description="Enable caching")
    app_cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    app_max_file_size: int = Field(default=10485760, description="Max file size in bytes (10MB)")
    app_rate_limit_requests: int = Field(default=30, description="Rate limit requests per period")
    app_rate_limit_period: int = Field(default=60, description="Rate limit period in seconds")
    
    # Vision settings
    vision_supported_image_formats: List[str] = Field(
        default=["png", "jpg", "jpeg", "gif", "bmp", "webp", "svg"],
        description="Supported image formats"
    )
    vision_max_image_size: int = Field(default=5242880, description="Max image size in bytes (5MB)")
    vision_analysis_enabled: bool = Field(default=True, description="Enable vision analysis")
    
    # Repository settings
    repo_private_repos_enabled: bool = Field(default=True, description="Enable private repository access")
    repo_user_repos_only: bool = Field(default=False, description="Restrict to user repositories only")
    
    # Redis settings
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    redis_db: int = Field(default=0, description="Redis database number")
    redis_ttl: int = Field(default=7200, description="Default TTL in seconds (2 hours)")
    redis_enabled: bool = Field(default=True, description="Enable Redis caching")
    
    @property
    def github(self) -> GitHubConfig:
        return GitHubConfig(token=self.github_token, api_url=self.github_api_url)
    
    @property
    def oracle_cloud(self) -> OracleCloudConfig:
        """Primary Oracle Cloud AI configuration."""
        return OracleCloudConfig(
            compartment_id=self.oci_compartment_id,
            config_profile=self.oci_config_profile,
            model_id=self.oci_model_id,
            model_vision=self.oci_model_vision,
            endpoint=self.oci_endpoint,
            max_tokens=self.oci_max_tokens,
            temperature=self.oci_temperature,
            enabled=self.oci_enabled
        )
    
    @property
    def oci(self) -> OracleCloudConfig:
        """Alias for oracle_cloud for backward compatibility."""
        return self.oracle_cloud
    
    @property
    def groq(self) -> LegacyGroqConfig:
        """Legacy configuration (deprecated)."""
        return LegacyGroqConfig(
            api_key=self.groq_api_key or "",
            model_text=self.groq_model_text,
            model_vision=self.groq_model_vision,
            max_tokens=self.groq_max_tokens,
            temperature=self.groq_temperature
        )
    
    @property
    def mcp(self) -> MCPConfig:
        return MCPConfig(
            server_command=self.mcp_server_command,
            server_args=self.mcp_server_args
        )
    
    @property
    def app(self) -> ApplicationConfig:
        return ApplicationConfig(
            log_level=self.app_log_level,
            cache_enabled=self.app_cache_enabled,
            cache_ttl=self.app_cache_ttl,
            max_file_size=self.app_max_file_size,
            rate_limit_requests=self.app_rate_limit_requests,
            rate_limit_period=self.app_rate_limit_period
        )
    
    @property
    def vision(self) -> VisionConfig:
        return VisionConfig(
            supported_image_formats=self.vision_supported_image_formats,
            max_image_size=self.vision_max_image_size,
            analysis_enabled=self.vision_analysis_enabled
        )
    
    @property
    def repository(self) -> RepositoryConfig:
        return RepositoryConfig(
            private_repos_enabled=self.repo_private_repos_enabled,
            user_repos_only=self.repo_user_repos_only
        )
    
    @property
    def redis(self) -> RedisConfig:
        return RedisConfig(
            host=self.redis_host,
            port=self.redis_port,
            password=self.redis_password,
            db=self.redis_db,
            ttl=self.redis_ttl,
            enabled=self.redis_enabled
        )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # Environment variables take precedence over .env files
        env_ignore_empty=True
    )
    
    @classmethod
    def load_from_env(cls, env_file: Optional[Path] = None) -> "Settings":
        """Load settings from environment file."""
        if env_file is None:
            # Try multiple locations for .env file
            locations = [
                Path(__file__).parent.parent / ".env",  # Kiara/.env (project root)
                Path.home() / ".env",  # User home directory
                Path(".env"),  # Current directory
                Path.home() / ".kiara" / ".env",  # User kiara config directory
                # Add common Kiara installation paths
                Path("C:/Users/athos/hackathonb3/Kiara/.env") if os.name == 'nt' else None,
            ]
            
            # Filter out None values and find first existing file
            for location in filter(None, locations):
                if location.exists():
                    env_file = location
                    break
            else:
                env_file = None
        
        if env_file and env_file.exists():
            return cls(_env_file=env_file)
        return cls()
    
    def validate_required_tokens(self) -> None:
        """Validate that required tokens are present."""
        if not self.github_token:
            raise ValueError("GITHUB_TOKEN is required")
        
        # Validate Oracle Cloud AI configuration
        if self.oci_enabled:
            if not self.oci_compartment_id:
                raise ValueError("OCI_COMPARTMENT_ID is required for Oracle Cloud AI")
            if not self.oci_model_id:
                raise ValueError("OCI_MODEL_ID is required for Oracle Cloud AI")
            if not self.oci_endpoint:
                raise ValueError("OCI_ENDPOINT is required for Oracle Cloud AI")
    
    def get_supported_image_extensions(self) -> List[str]:
        """Get supported image file extensions with dots."""
        return [f".{fmt}" for fmt in self.vision.supported_image_formats]
    
    def is_image_file(self, filename: str) -> bool:
        """Check if a file is a supported image format."""
        return any(filename.lower().endswith(ext) for ext in self.get_supported_image_extensions())


def get_settings() -> Settings:
    """Get application settings singleton."""
    if not hasattr(get_settings, "_settings"):
        get_settings._settings = Settings.load_from_env()
    return get_settings._settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    if hasattr(get_settings, "_settings"):
        delattr(get_settings, "_settings")
    return get_settings()


def get_local_settings() -> Settings:
    """Get settings for local-only operations (GitHub token not required)."""
    
    from dotenv import load_dotenv
    
    # Load .env file explicitly
    load_dotenv()
    
    # Check if Groq API key is available
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY is required for local project analysis")
    
    # Create settings with only required fields for local operations
    try:
        return Settings.load_from_env()
    except Exception as e:
        # If there's a validation error, try to create minimal settings
        if "groq_api_key" in str(e).lower():
            raise ValueError("GROQ_API_KEY is required for local project analysis")
        
        # For other validation errors (like missing GitHub token), continue with local-only mode
        # Create a basic settings instance with required Groq settings
        return Settings(
            groq_api_key=groq_key,
            # All other fields have defaults
        )