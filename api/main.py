"""
FastAPI application for GitHub Documentation Agent.
Provides REST API endpoints for automatic README generation.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, validator

# Import our documentation generator and GitHub processor
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.doc_generator import DocumentationGenerator
from src.validators import ValidationError
from github_processor import GitHubProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
doc_generator: Optional[DocumentationGenerator] = None
github_processor: Optional[GitHubProcessor] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources."""
    global doc_generator, github_processor
    
    try:
        # Initialize documentation generator and GitHub processor
        logger.info("Initializing GitHub Documentation Agent...")
        settings = get_settings()
        doc_generator = DocumentationGenerator(settings)
        github_processor = GitHubProcessor()
        logger.info("Documentation generator and GitHub processor initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    finally:
        # Cleanup resources
        if doc_generator:
            logger.info("Cleaning up resources...")

# Create FastAPI application
app = FastAPI(
    title="GitHub Documentation Agent API",
    description="Automatic README generation using GitHub MCP Server and Groq AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class GenerateDocsRequest(BaseModel):
    """Request model for documentation generation."""
    
    repository_url: HttpUrl
    include_vision: Optional[bool] = True
    output_format: Optional[str] = "markdown"
    force_regenerate: Optional[bool] = False
    template_type: Optional[str] = None  # Allow user to specify template: minimal, emoji_rich, modern, technical_deep
    
    @validator('repository_url')
    def validate_github_url(cls, v):
        """Validate that URL is a GitHub repository."""
        url_str = str(v)
        if not url_str.startswith(('https://github.com/', 'http://github.com/')):
            raise ValueError('URL must be a GitHub repository')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "repository_url": "https://github.com/username/repository",
                "include_vision": True,
                "output_format": "markdown",
                "force_regenerate": False,
                "template_type": "modern"
            }
        }

class GenerateDocsResponse(BaseModel):
    """Response model for documentation generation."""
    
    success: bool
    repository_url: str
    generation_time: float
    documentation: Optional[str] = None
    analysis: Optional[dict] = None
    errors: list[str] = []
    warnings: list[str] = []
    cache_hit: bool = False
    cache_status: str = "not_checked"
    commit_status: str = "not_checked"
    commit_sha: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "repository_url": "https://github.com/username/repository",
                "generation_time": 25.3,
                "documentation": "# Project Name\n\n> Description...",
                "analysis": {
                    "language": "Python",
                    "frameworks": ["FastAPI", "Pydantic"],
                    "project_type": "web_application"
                },
                "errors": [],
                "warnings": [],
                "cache_hit": False,
                "cache_status": "cache_stale"
            }
        }

class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    version: str
    components: dict
    timestamp: str

# API Endpoints

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "GitHub Documentation Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "generate": "POST /generate-docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint - fast response."""
    global doc_generator, github_processor
    
    # Quick status without heavy operations
    components = {
        "documentation_generator": "healthy" if doc_generator else "unavailable",
        "github_processor": "healthy" if github_processor else "unavailable",
        "groq_api": "configured",
        "github_mcp": "configured"
    }
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        components=components,
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    )

@app.post("/generate-docs", response_model=GenerateDocsResponse)
async def generate_documentation(request: GenerateDocsRequest):
    """
    Generate README.md documentation for a GitHub repository with intelligent commit tracking.
    
    This endpoint uses the complete GitHub Documentation Agent pipeline:
    - Automatic commit tracking and comparison
    - GitHub MCP Server for repository access
    - Groq multimodal AI for intelligent analysis  
    - Universal language detection (20+ languages)
    - Adaptive template engine
    - Professional documentation generation
    - Only regenerates if new commits are detected
    """
    global doc_generator, github_processor
    
    if not doc_generator or not github_processor:
        raise HTTPException(
            status_code=503,
            detail="Documentation generator or GitHub processor not available"
        )
    
    start_time = time.time()
    repo_url = str(request.repository_url)
    
    logger.info(f"Starting smart documentation generation for: {repo_url}")
    
    try:
        # Use GitHub processor for intelligent commit tracking
        if not request.force_regenerate:
            success = await github_processor.process_github_url(repo_url)
            generation_time = time.time() - start_time
            
            if success:
                # Get repository stats for response
                stats = github_processor.tracker.get_repository_stats(repo_url)
                
                # Check if README was generated or skipped
                if stats.get('latest_generation'):
                    latest_gen = stats['latest_generation']
                    commit_sha = latest_gen.get('sha', 'unknown')
                    
                    # Read generated README.md if it exists
                    readme_path = Path("README.md")
                    documentation = None
                    if readme_path.exists():
                        documentation = readme_path.read_text(encoding='utf-8')
                    
                    return GenerateDocsResponse(
                        success=True,
                        repository_url=repo_url,
                        generation_time=generation_time,
                        documentation=documentation,
                        analysis={
                            "repository_stats": stats,
                            "method": "smart_commit_tracking"
                        },
                        errors=[],
                        warnings=[],
                        commit_status="processed_or_skipped",
                        commit_sha=commit_sha
                    )
                else:
                    return GenerateDocsResponse(
                        success=False,
                        repository_url=repo_url,
                        generation_time=generation_time,
                        errors=["Failed to process repository"],
                        commit_status="processing_failed"
                    )
            else:
                return GenerateDocsResponse(
                    success=False,
                    repository_url=repo_url,
                    generation_time=time.time() - start_time,
                    errors=["GitHub processor failed"],
                    commit_status="processing_failed"
                )
        else:
            # Force regeneration - use original doc generator
            logger.info("Force regeneration requested - using original generator")
            result = await doc_generator.generate(repo_url, force_regenerate=True, template_type=request.template_type)
        
        generation_time = time.time() - start_time
        
        if result.success:
            logger.info(f"Documentation generated successfully in {generation_time:.2f}s")
            
            # Prepare analysis summary
            analysis_summary = {
                "language": result.analysis.language.primary_language,
                "language_confidence": result.analysis.language.confidence,
                "frameworks": result.analysis.language.frameworks,
                "package_managers": result.analysis.language.package_managers,
                "build_tools": result.analysis.language.build_tools,
                "testing_frameworks": result.analysis.language.testing_frameworks,
                "project_type": result.analysis.architecture.project_type,
                "architectural_patterns": result.analysis.architecture.architectural_patterns,
                "main_components": result.analysis.architecture.main_components,
                "installation_steps_count": len(result.analysis.installation_steps),
                "usage_examples_count": len(result.analysis.usage_examples),
                "visual_insights_count": len(result.analysis.visual_insights)
            }
            
            return GenerateDocsResponse(
                success=True,
                repository_url=repo_url,
                generation_time=generation_time,
                documentation=result.documentation,
                analysis=analysis_summary,
                errors=result.errors,
                warnings=result.warnings,
                cache_hit=result.cache_hit,
                cache_status=result.cache_status
            )
        
        else:
            logger.error(f"Documentation generation failed for {repo_url}")
            return GenerateDocsResponse(
                success=False,
                repository_url=repo_url,
                generation_time=generation_time,
                errors=result.errors,
                warnings=result.warnings,
                cache_hit=result.cache_hit,
                cache_status=result.cache_status
            )
            
    except ValidationError as e:
        logger.error(f"Validation error for {repo_url}: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid repository URL or access denied: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Unexpected error generating docs for {repo_url}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/generate-docs/{owner}/{repo}")
async def generate_docs_from_path(owner: str, repo: str):
    """
    Generate documentation from repository owner and name.
    Convenience endpoint that constructs the GitHub URL.
    """
    repository_url = f"https://github.com/{owner}/{repo}"
    
    request = GenerateDocsRequest(
        repository_url=repository_url,
        include_vision=True,
        output_format="markdown"
    )
    
    return await generate_documentation(request)

@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    global doc_generator
    
    if not doc_generator:
        raise HTTPException(
            status_code=503,
            detail="Documentation generator not available"
        )
    
    try:
        stats = await doc_generator.get_cache_stats()
        return {"cache_stats": stats}
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get cache statistics: {str(e)}"
        )

@app.delete("/cache/{owner}/{repo}")
async def invalidate_repository_cache(owner: str, repo: str):
    """Invalidate cache for a specific repository."""
    global doc_generator
    
    if not doc_generator:
        raise HTTPException(
            status_code=503,
            detail="Documentation generator not available"
        )
    
    repository_url = f"https://github.com/{owner}/{repo}"
    
    try:
        success = await doc_generator.invalidate_cache(repository_url)
        return {
            "success": success,
            "repository_url": repository_url,
            "message": "Cache invalidated successfully" if success else "Cache invalidation failed"
        }
    except Exception as e:
        logger.error(f"Error invalidating cache for {repository_url}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to invalidate cache: {str(e)}"
        )

@app.delete("/cache")
async def clear_all_cache():
    """Clear all cached data."""
    global doc_generator
    
    if not doc_generator:
        raise HTTPException(
            status_code=503,
            detail="Documentation generator not available"
        )
    
    try:
        success = await doc_generator.clear_cache()
        return {
            "success": success,
            "message": "All cache cleared successfully" if success else "Cache clear failed"
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )

@app.get("/status")
async def api_status():
    """Get API status and configuration."""
    global doc_generator
    
    try:
        settings = get_settings()
        
        return {
            "api_version": "1.0.0",
            "status": "running",
            "documentation_generator": "available" if doc_generator else "unavailable",
            "supported_languages": [
                "Python", "JavaScript", "TypeScript", "Java", "Go", "Rust",
                "C++", "C#", "PHP", "Ruby", "Swift", "Kotlin", "Dart"
            ],
            "features": [
                "GitHub MCP Server integration",
                "Groq multimodal AI analysis", 
                "Universal language detection",
                "Adaptive template engine",
                "Vision processing (images/diagrams)",
                "Professional README generation"
            ],
            "configuration": {
                "groq_model_text": settings.groq.model_text,
                "groq_model_vision": settings.groq.model_vision,
                "vision_enabled": settings.vision.analysis_enabled,
                "max_tokens": settings.groq.max_tokens,
                "rate_limit": f"{settings.app.rate_limit_requests}/{settings.app.rate_limit_period}s"
            }
        }
    except Exception as e:
        return {"error": f"Failed to get status: {e}"}

# Error handlers
@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    """Handle validation errors."""
    return JSONResponse(
        status_code=400,
        content={"detail": f"Validation error: {str(exc)}"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )