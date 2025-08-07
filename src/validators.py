"""Input validation utilities for GitHub Documentation Agent."""

import re
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, field_validator

from .config import Settings, get_settings


class ValidationError(Exception):
    """Validation error."""
    pass


class ValidationResult(BaseModel):
    """Validation result."""
    
    valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    normalized_value: Optional[Any] = None


class AzureDevOpsURLValidator:
    """Validates and normalizes Azure DevOps repository URLs."""
    
    AZURE_DOMAINS = ["dev.azure.com", "visualstudio.com"]
    
    # Azure DevOps URL patterns
    PATTERNS = [
        # https://dev.azure.com/{organization}/{project}/_git/{repository}
        r"^https?://dev\.azure\.com/([^/]+)/([^/]+)/_git/([^/]+?)/?(?:\?.*)?(?:#.*)?$",
        # https://{organization}.visualstudio.com/{project}/_git/{repository}
        r"^https?://([^\.]+)\.visualstudio\.com/([^/]+)/_git/([^/]+?)/?(?:\?.*)?(?:#.*)?$",
        # https://{organization}.visualstudio.com/DefaultCollection/{project}/_git/{repository}
        r"^https?://([^\.]+)\.visualstudio\.com/DefaultCollection/([^/]+)/_git/([^/]+?)/?(?:\?.*)?(?:#.*)?$",
    ]
    
    def validate(self, url: str) -> ValidationResult:
        """Validate Azure DevOps repository URL."""
        errors = []
        warnings = []
        
        if not url or not isinstance(url, str):
            return ValidationResult(
                valid=False,
                errors=["URL is required and must be a string"]
            )
        
        url = url.strip()
        
        if not url:
            return ValidationResult(
                valid=False,
                errors=["URL cannot be empty"]
            )
        
        # Try to match against patterns
        organization, project, repository = None, None, None
        
        for pattern in self.PATTERNS:
            match = re.match(pattern, url, re.IGNORECASE)
            if match:
                if len(match.groups()) == 3:
                    organization, project, repository = match.groups()
                break
        
        if not organization or not project or not repository:
            return ValidationResult(
                valid=False,
                errors=["Invalid Azure DevOps repository URL format"]
            )
        
        # Validate names
        org_errors = self._validate_azure_name(organization, "organization")
        proj_errors = self._validate_azure_name(project, "project")
        repo_errors = self._validate_azure_name(repository, "repository")
        
        errors.extend(org_errors)
        errors.extend(proj_errors)
        errors.extend(repo_errors)
        
        # Generate normalized URL (prefer dev.azure.com format)
        normalized_url = f"https://dev.azure.com/{organization}/{project}/_git/{repository}"
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            normalized_value=normalized_url if len(errors) == 0 else None
        )
    
    def _validate_azure_name(self, name: str, name_type: str) -> List[str]:
        """Validate Azure DevOps organization/project/repository name."""
        errors = []
        
        if not name:
            errors.append(f"Azure DevOps {name_type} name cannot be empty")
            return errors
        
        if len(name) > 64:
            errors.append(f"Azure DevOps {name_type} name is too long (max 64 characters)")
        
        # Basic character validation (Azure DevOps is more permissive than GitHub)
        if not re.match(r'^[a-zA-Z0-9\-_.]+$', name):
            errors.append(f"Azure DevOps {name_type} name contains invalid characters")
        
        return errors


class GitHubURLValidator:
    """Validates and normalizes GitHub repository URLs."""
    
    GITHUB_DOMAINS = ["github.com", "www.github.com"]
    
    # GitHub URL patterns
    PATTERNS = [
        r"^https?://(?:www\.)?github\.com/([^/]+)/([^/]+?)(?:\.git)?/?(?:\?.*)?(?:#.*)?$",
        r"^git@github\.com:([^/]+)/([^/]+?)(?:\.git)?/?$",
        r"^(?:https?://)?(?:www\.)?github\.com/([^/]+)/([^/]+?)(?:\.git)?/?(?:\?.*)?(?:#.*)?$",
        r"^([^/]+)/([^/]+)$"  # Simple owner/repo format
    ]
    
    def validate(self, url: str) -> ValidationResult:
        """Validate GitHub repository URL."""
        errors = []
        warnings = []
        
        if not url or not isinstance(url, str):
            return ValidationResult(
                valid=False,
                errors=["URL is required and must be a string"]
            )
        
        url = url.strip()
        
        if not url:
            return ValidationResult(
                valid=False,
                errors=["URL cannot be empty"]
            )
        
        # Try to match against patterns
        owner, repo = None, None
        
        for pattern in self.PATTERNS:
            match = re.match(pattern, url, re.IGNORECASE)
            if match:
                owner, repo = match.groups()
                break
        
        if not owner or not repo:
            return ValidationResult(
                valid=False,
                errors=["Invalid GitHub repository URL format"]
            )
        
        # Validate owner/repo names
        owner_errors = self._validate_github_name(owner, "owner")
        repo_errors = self._validate_github_name(repo, "repository")
        
        errors.extend(owner_errors)
        errors.extend(repo_errors)
        
        # Check for common issues
        if repo.endswith('.git'):
            repo = repo[:-4]
            warnings.append("Removed .git suffix from repository name")
        
        if '.' in owner and not owner.endswith('.github.io'):
            warnings.append("Owner name contains dots, ensure this is correct")
        
        # Generate normalized URL
        normalized_url = f"https://github.com/{owner}/{repo}"
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            normalized_value=normalized_url if len(errors) == 0 else None
        )
    
    def _validate_github_name(self, name: str, name_type: str) -> List[str]:
        """Validate GitHub owner/repository name."""
        errors = []
        
        if not name:
            errors.append(f"{name_type} name cannot be empty")
            return errors
        
        # Length constraints
        if len(name) > 39:
            errors.append(f"{name_type} name too long (max 39 characters)")
        
        if len(name) < 1:
            errors.append(f"{name_type} name too short (min 1 character)")
        
        # Character constraints
        if not re.match(r'^[a-zA-Z0-9._-]+$', name):
            errors.append(f"{name_type} name contains invalid characters (only alphanumeric, dots, hyphens, underscores allowed)")
        
        # Cannot start or end with special characters
        if name.startswith(('.', '-')) or name.endswith(('.', '-')):
            errors.append(f"{name_type} name cannot start or end with dots or hyphens")
        
        # Cannot have consecutive dots
        if '..' in name:
            errors.append(f"{name_type} name cannot contain consecutive dots")
        
        # Reserved names
        reserved_names = {
            'owner': ['www', 'api', 'support', 'help', 'blog', 'status', 'admin'],
            'repository': ['www', 'api', 'support', 'help', 'blog', 'status']
        }
        
        if name.lower() in reserved_names.get(name_type, []):
            errors.append(f"{name_type} name '{name}' is reserved")
        
        return errors
    
    def extract_owner_repo(self, url: str) -> Optional[Tuple[str, str]]:
        """Extract owner and repository name from URL."""
        result = self.validate(url)
        if result.valid and result.normalized_value:
            match = re.match(r'https://github\.com/([^/]+)/([^/]+)', result.normalized_value)
            if match:
                return match.groups()
        return None


class SettingsValidator:
    """Validates application settings and configuration."""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
    
    def validate_github_token(self, token: str) -> ValidationResult:
        """Validate GitHub token format."""
        errors = []
        warnings = []
        
        if not token:
            return ValidationResult(
                valid=False,
                errors=["GitHub token is required"]
            )
        
        # Check token format
        if not token.startswith(('ghp_', 'gho_', 'ghu_', 'ghs_', 'ghr_')):
            errors.append("Invalid GitHub token format (should start with ghp_, gho_, ghu_, ghs_, or ghr_)")
        
        # Check token length
        if len(token) < 40:
            warnings.append("GitHub token seems short, ensure it's complete")
        elif len(token) > 100:
            warnings.append("GitHub token seems long, ensure it's correct")
        
        # Check for common issues
        if ' ' in token:
            errors.append("GitHub token contains spaces")
        
        if token != token.strip():
            warnings.append("GitHub token has leading/trailing whitespace")
            token = token.strip()
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            normalized_value=token.strip() if len(errors) == 0 else None
        )
    
    def validate_groq_api_key(self, api_key: str) -> ValidationResult:
        """Validate Groq API key format."""
        errors = []
        warnings = []
        
        if not api_key:
            return ValidationResult(
                valid=False,
                errors=["Groq API key is required"]
            )
        
        # Check key format
        if not api_key.startswith('gsk_'):
            errors.append("Invalid Groq API key format (should start with gsk_)")
        
        # Check key length
        if len(api_key) < 50:
            warnings.append("Groq API key seems short, ensure it's complete")
        
        # Check for common issues
        if ' ' in api_key:
            errors.append("Groq API key contains spaces")
        
        if api_key != api_key.strip():
            warnings.append("Groq API key has leading/trailing whitespace")
            api_key = api_key.strip()
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            normalized_value=api_key.strip() if len(errors) == 0 else None
        )
    
    def validate_model_name(self, model_name: str, model_type: str = "text") -> ValidationResult:
        """Validate Groq model name."""
        errors = []
        warnings = []
        
        if not model_name:
            return ValidationResult(
                valid=False,
                errors=[f"{model_type} model name is required"]
            )
        
        # Known valid models
        valid_text_models = [
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma-7b-it"
        ]
        
        valid_vision_models = [
            "llama-3.2-90b-vision-preview",
            "llama-3.2-11b-vision-preview"
        ]
        
        if model_type == "text" and model_name not in valid_text_models:
            warnings.append(f"Unknown text model '{model_name}', ensure it's supported by Groq")
        elif model_type == "vision" and model_name not in valid_vision_models:
            warnings.append(f"Unknown vision model '{model_name}', ensure it's supported by Groq")
        
        return ValidationResult(
            valid=True,
            errors=errors,
            warnings=warnings,
            normalized_value=model_name
        )
    
    def validate_complete_settings(self) -> ValidationResult:
        """Validate complete application settings."""
        errors = []
        warnings = []
        
        try:
            # Validate required tokens
            self.settings.validate_required_tokens()
        except ValueError as e:
            errors.append(str(e))
        
        # Validate GitHub token
        github_result = self.validate_github_token(self.settings.github.token)
        errors.extend(github_result.errors)
        warnings.extend(github_result.warnings)
        
        # Validate Groq API key
        groq_result = self.validate_groq_api_key(self.settings.groq.api_key)
        errors.extend(groq_result.errors)
        warnings.extend(groq_result.warnings)
        
        # Validate model names
        text_model_result = self.validate_model_name(self.settings.groq.model_text, "text")
        warnings.extend(text_model_result.warnings)
        
        vision_model_result = self.validate_model_name(self.settings.groq.model_vision, "vision")
        warnings.extend(vision_model_result.warnings)
        
        # Validate numeric settings
        if self.settings.groq.max_tokens <= 0:
            errors.append("Max tokens must be positive")
        elif self.settings.groq.max_tokens > 32768:
            warnings.append("Max tokens is very high, may cause issues")
        
        if not 0.0 <= self.settings.groq.temperature <= 2.0:
            errors.append("Temperature must be between 0.0 and 2.0")
        
        if self.settings.vision.max_image_size <= 0:
            errors.append("Max image size must be positive")
        elif self.settings.vision.max_image_size > 50 * 1024 * 1024:  # 50MB
            warnings.append("Max image size is very large")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


class InputSanitizer:
    """Sanitizes and cleans user inputs."""
    
    @staticmethod
    def sanitize_url(url: str) -> str:
        """Sanitize URL input."""
        if not url:
            return ""
        
        # Remove dangerous characters
        url = re.sub(r'[<>"\']', '', url)
        
        # Normalize whitespace
        url = ' '.join(url.split())
        
        # URL decode if needed
        try:
            url = urllib.parse.unquote(url)
        except Exception:
            pass  # Keep original if decoding fails
        
        return url.strip()
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename input."""
        if not filename:
            return "README.md"
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        
        # Remove control characters
        filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)
        
        # Normalize whitespace
        filename = ' '.join(filename.split())
        
        # Ensure it ends with .md
        if not filename.lower().endswith('.md'):
            filename += '.md'
        
        # Ensure it's not empty
        if not filename or filename == '.md':
            filename = 'README.md'
        
        return filename
    
    @staticmethod
    def sanitize_path(path: str) -> str:
        """Sanitize file path input."""
        if not path:
            return "."
        
        # Remove dangerous characters
        path = re.sub(r'[<>"|?*]', '', path)
        
        # Remove control characters
        path = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', path)
        
        # Normalize path separators
        path = path.replace('\\', '/')
        
        # Remove relative path components for security
        path = re.sub(r'\.\./', '', path)
        path = re.sub(r'/\.\.', '', path)
        
        # Normalize whitespace
        path = ' '.join(path.split())
        
        return path.strip() or "."


class UniversalRepositoryValidator:
    """Universal validator that can handle multiple repository types."""
    
    def __init__(self):
        self.github_validator = GitHubURLValidator()
        self.azure_devops_validator = AzureDevOpsURLValidator()
    
    def detect_repository_type(self, url: str) -> Optional[str]:
        """Detect the type of repository URL."""
        url = url.strip().lower()
        
        if 'github.com' in url:
            return 'github'
        elif 'dev.azure.com' in url or '.visualstudio.com' in url:
            return 'azure_devops'
        
        return None
    
    def validate(self, repo_input: str) -> Tuple[ValidationResult, Optional[str]]:
        """Validate repository input and return validation result + detected type."""
        if not repo_input:
            return ValidationResult(
                valid=False,
                errors=["Repository input is required"]
            ), None
        
        repo_type = self.detect_repository_type(repo_input)
        
        if repo_type == 'github':
            result = self.github_validator.validate(repo_input)
            return result, 'github'
        elif repo_type == 'azure_devops':
            result = self.azure_devops_validator.validate(repo_input)
            return result, 'azure_devops'
        else:
            # Try GitHub first (for backwards compatibility)
            github_result = self.github_validator.validate(repo_input)
            if github_result.valid:
                return github_result, 'github'
            
            # Try Azure DevOps
            azure_result = self.azure_devops_validator.validate(repo_input)
            if azure_result.valid:
                return azure_result, 'azure_devops'
            
            # Neither worked
            return ValidationResult(
                valid=False,
                errors=[
                    "Invalid repository URL format. Supported formats:",
                    "- GitHub: https://github.com/owner/repo",
                    "- Azure DevOps: https://dev.azure.com/org/project/_git/repo"
                ]
            ), None


class ValidatorCollection:
    """Collection of all validators for easy access."""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.github_url = GitHubURLValidator()
        self.azure_devops_url = AzureDevOpsURLValidator()
        self.universal_repo = UniversalRepositoryValidator()
        self.settings = SettingsValidator(settings)
        self.sanitizer = InputSanitizer()
    
    def validate_repository_input(self, repo_input: str) -> ValidationResult:
        """Validate repository input (URL or owner/repo) - GitHub only for backwards compatibility."""
        if not repo_input:
            return ValidationResult(
                valid=False,
                errors=["Repository input is required"]
            )
        
        # Sanitize input
        sanitized_input = self.sanitizer.sanitize_url(repo_input)
        
        # Validate as GitHub URL
        return self.github_url.validate(sanitized_input)
    
    def validate_universal_repository_input(self, repo_input: str) -> Tuple[ValidationResult, Optional[str]]:
        """Validate repository input supporting multiple platforms."""
        if not repo_input:
            return ValidationResult(
                valid=False,
                errors=["Repository input is required"]
            ), None
        
        # Sanitize input
        sanitized_input = self.sanitizer.sanitize_url(repo_input)
        
        # Use universal validator
        return self.universal_repo.validate(sanitized_input)
    
    def validate_output_settings(
        self,
        output_path: Optional[str] = None,
        filename: Optional[str] = None
    ) -> ValidationResult:
        """Validate output settings."""
        errors = []
        warnings = []
        normalized = {}
        
        # Validate output path
        if output_path:
            sanitized_path = self.sanitizer.sanitize_path(output_path)
            if not sanitized_path or sanitized_path == ".":
                warnings.append("Using current directory as output path")
            normalized["output_path"] = sanitized_path
        
        # Validate filename
        if filename:
            sanitized_filename = self.sanitizer.sanitize_filename(filename)
            if sanitized_filename != filename:
                warnings.append(f"Filename sanitized to: {sanitized_filename}")
            normalized["filename"] = sanitized_filename
        
        return ValidationResult(
            valid=True,
            errors=errors,
            warnings=warnings,
            normalized_value=normalized
        )
    
    def validate_all_inputs(
        self,
        repo_url: str,
        output_path: Optional[str] = None,
        filename: Optional[str] = None
    ) -> Dict[str, ValidationResult]:
        """Validate all inputs at once."""
        results = {}
        
        results["repository"] = self.validate_repository_input(repo_url)
        results["settings"] = self.settings.validate_complete_settings()
        results["output"] = self.validate_output_settings(output_path, filename)
        
        return results
    
    def get_validation_summary(self, results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Get summary of validation results."""
        all_valid = all(result.valid for result in results.values())
        total_errors = sum(len(result.errors) for result in results.values())
        total_warnings = sum(len(result.warnings) for result in results.values())
        
        return {
            "all_valid": all_valid,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "details": {name: {"valid": result.valid, "errors": result.errors, "warnings": result.warnings} 
                       for name, result in results.items()}
        }