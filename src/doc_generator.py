"""Documentation generator that orchestrates the complete documentation generation process."""

import asyncio
import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from .auth_manager import AuthManager
from .cache_manager import CacheManager, create_cache_manager
from .config import Settings, get_settings
from .groq_analyzer import GroqAnalyzer
from .oci_analyzer import OCIAnalyzer
from .xai_analyzer import XAIGrokAnalyzer
from .language_detector import LanguageDetector
from .mcp_client import MCPClient
from .template_engine import TemplateEngine
from .vision_processor import VisionProcessor
from .simple_commit_tracker import SimpleCommitTracker
from .models import ProjectAnalysis

logger = logging.getLogger(__name__)


class GenerationResult(BaseModel):
    """Documentation generation result."""
    
    success: bool
    documentation: str = ""
    analysis: Optional[ProjectAnalysis] = None
    metadata: Dict[str, Any] = {}
    errors: List[str] = []
    warnings: List[str] = []
    generation_time: float = 0.0
    cache_hit: bool = False
    cache_status: str = "not_checked"
    commit_detected: bool = False
    commit_info: Optional[Dict[str, Any]] = None
    update_message: str = ""


class DocumentationGenerator:
    """Main documentation generator that orchestrates all components."""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        
        # Initialize components
        self.auth_manager = AuthManager(self.settings)
        self.mcp_client = MCPClient(self.settings)
        
        # Initialize AI analyzer (xAI Grok 4, OCI, or Groq based on settings)
        if self.settings.xai_grok_enabled:
            logger.info("Using xAI Grok 4 analyzer (131K tokens)")
            self.ai_analyzer = XAIGrokAnalyzer(self.settings)
        elif self.settings.oci_enabled:
            logger.info("Using OCI AI analyzer")
            self.ai_analyzer = OCIAnalyzer(self.settings)
        else:
            logger.info("Using Groq AI analyzer (fallback)")
            self.ai_analyzer = GroqAnalyzer(self.settings)
        
        self.vision_processor = VisionProcessor(self.settings)
        self.language_detector = LanguageDetector(self.settings)
        self.template_engine = TemplateEngine(self.settings)
        
        # Cache manager will be initialized when needed
        self.cache_manager: Optional[CacheManager] = None
        
        # Simple commit tracker
        self.commit_tracker = SimpleCommitTracker()
        
        self.errors = []
        self.warnings = []
    
    async def generate(self, repo_url: str, force_regenerate: bool = False, template_type: str = None) -> GenerationResult:
        """Generate documentation for a repository with cache validation."""
        import time
        start_time = time.time()
        
        logger.info(f"Starting documentation generation for: {repo_url}")
        
        try:
            # Step 1: Check for new commits
            has_new_commit, commit_message = await self.commit_tracker.check_for_new_commit(repo_url)
            
            # Initialize cache manager
            if not self.cache_manager:
                self.cache_manager = await create_cache_manager(self.settings)
            
            cache_status = "cache_unavailable"
            cache_hit = False
            
            # Step 2: Check cache validity if not forced regeneration AND no new commits
            if not force_regenerate and not has_new_commit and self.cache_manager._connected:
                cache_status, cache_hit, cached_documentation = await self._check_cache_validity(repo_url)
                
                if cache_hit and cached_documentation:
                    generation_time = time.time() - start_time
                    logger.info(f"Using cached documentation for {repo_url} (retrieved in {generation_time:.3f}s)")
                    
                    # Get cached repository info for metadata
                    cached_repo_info = await self.cache_manager.get_repository_info(repo_url)
                    
                    return GenerationResult(
                        success=True,
                        documentation=cached_documentation,
                        analysis=None,  # Analysis not cached separately yet
                        metadata=self._generate_cache_metadata(cached_repo_info, generation_time),
                        errors=self.errors,
                        warnings=self.warnings,
                        generation_time=generation_time,
                        cache_hit=True,
                        cache_status=cache_status,
                        commit_detected=has_new_commit,
                        commit_info=None,
                        update_message=commit_message
                    )
            
            # Step 3: Validate authentication
            await self._validate_authentication()
            
            # Step 4: Collect project data (includes latest commit info)
            project_data = await self._collect_project_data(repo_url)
            
            # Step 5: Process images
            processed_images = await self._process_images(project_data.get("images", {}))
            
            # Step 6: Analyze project with AI
            analysis = await self._analyze_project(project_data, processed_images)
            
            # Step 7: Generate documentation
            documentation = await self._generate_documentation(analysis, project_data, template_type)
            
            # Step 8: Validate and finalize
            final_documentation = await self._finalize_documentation(documentation, analysis)
            
            # Step 9: Update cache with new documentation
            await self._update_cache(repo_url, project_data, final_documentation, analysis)
            
            generation_time = time.time() - start_time
            
            logger.info(f"Documentation generation completed in {generation_time:.2f} seconds")
            
            # Mark README as updated (get current commit and store it)
            if has_new_commit and repo_url.startswith('https://github.com/'):
                current_commit = await self.commit_tracker.get_current_commit(repo_url)
                if current_commit:
                    self.commit_tracker.mark_readme_updated(repo_url, current_commit)
            
            return GenerationResult(
                success=True,
                documentation=final_documentation,
                analysis=analysis,
                metadata=self._generate_metadata(project_data, analysis, generation_time),
                errors=self.errors,
                warnings=self.warnings,
                generation_time=generation_time,
                cache_hit=cache_hit,
                cache_status=cache_status,
                commit_detected=has_new_commit,
                commit_info=None,
                update_message=commit_message
            )
            
        except Exception as e:
            logger.error(f"Documentation generation failed: {e}")
            self.errors.append(str(e))
            
            generation_time = time.time() - start_time
            
            return GenerationResult(
                success=False,
                documentation="",
                analysis=None,
                metadata={"error": str(e)},
                errors=self.errors,
                warnings=self.warnings,
                generation_time=generation_time,
                cache_hit=False,
                cache_status=cache_status if 'cache_status' in locals() else "error"
            )
    
    async def _validate_authentication(self) -> None:
        """Validate GitHub authentication."""
        try:
            async with self.auth_manager as auth:
                token_info = await auth.validate_token()
                
                # Check required scopes (silently for better UX)
                required_scopes = auth.check_required_scopes()
                missing_scopes = [scope for scope, has_scope in required_scopes.items() if not has_scope]
                
                if missing_scopes:
                    # Log at debug level instead of warning to avoid cluttering output
                    logger.debug(f"Missing GitHub scopes: {', '.join(missing_scopes)}")
                
                # Check rate limits
                rate_limit = await auth.check_rate_limit()
                if rate_limit["remaining"] < 100:
                    warning = f"Low GitHub API rate limit: {rate_limit['remaining']} requests remaining"
                    self.warnings.append(warning)
                    logger.warning(warning)
                
                logger.info(f"Authentication validated for user: {token_info.user}")
                
        except Exception as e:
            error = f"Authentication validation failed: {e}"
            self.errors.append(error)
            raise Exception(error)
    
    async def _collect_project_data(self, repo_url: str) -> Dict[str, Any]:
        """Collect comprehensive project data via MCP."""
        try:
            async with self.mcp_client as mcp:
                project_data = await mcp.collect_project_data(repo_url)
                
                logger.info(f"Collected data for {len(project_data.get('files', []))} files")
                logger.info(f"Found {len(project_data.get('images', {}))} images")
                logger.info(f"Read {len(project_data.get('file_contents', {}))} important files")
                
                return project_data
                
        except Exception as e:
            error = f"Failed to collect project data: {e}"
            self.errors.append(error)
            raise Exception(error)
    
    async def _process_images(self, images_data: Dict[str, str]) -> Dict[str, Any]:
        """Process and analyze images."""
        if not images_data:
            logger.info("No images found for processing")
            return {}
        
        if not self.settings.vision.analysis_enabled:
            logger.info("Vision analysis disabled")
            return {}
        
        try:
            # Convert base64 strings to bytes
            images_bytes = {}
            for image_path, base64_data in images_data.items():
                try:
                    image_bytes = base64.b64decode(base64_data)
                    images_bytes[image_path] = image_bytes
                except Exception as e:
                    warning = f"Failed to decode image {image_path}: {e}"
                    self.warnings.append(warning)
                    logger.warning(warning)
            
            # Process images
            processed_images = self.vision_processor.batch_process_images(images_bytes)
            
            # Filter by relevance
            relevant_images = self.vision_processor.filter_by_relevance(
                processed_images,
                min_score=0.3,
                max_count=10
            )
            
            # Generate summary
            image_summary = self.vision_processor.generate_image_summary(relevant_images)
            
            logger.info(f"Processed {len(processed_images)} images, {len(relevant_images)} deemed relevant")
            
            return {
                "processed": processed_images,
                "relevant": relevant_images,
                "summary": image_summary
            }
            
        except Exception as e:
            warning = f"Image processing failed: {e}"
            self.warnings.append(warning)
            logger.warning(warning)
            return {}
    
    async def _analyze_project(
        self,
        project_data: Dict[str, Any],
        processed_images: Dict[str, Any]
    ) -> ProjectAnalysis:
        """Analyze project using AI and language detection."""
        try:
            # Basic language detection first
            files = project_data.get("files", [])
            file_contents = project_data.get("file_contents", {})
            
            basic_detection = self.language_detector.detect_language(files, file_contents)
            
            logger.info(f"Detected primary language: {basic_detection.language} "
                       f"(confidence: {basic_detection.confidence:.2f})")
            
            # Prepare data for Groq analysis
            enhanced_project_data = project_data.copy()
            
            # Add processed images data
            if processed_images.get("relevant"):
                enhanced_project_data["processed_images"] = {
                    path: img.base64_data 
                    for path, img in processed_images["relevant"].items()
                }
                enhanced_project_data["image_insights"] = [
                    {
                        "path": path,
                        "type": img.metadata.format,
                        "technical_score": img.technical_score,
                        "elements": img.detected_elements
                    }
                    for path, img in processed_images["relevant"].items()
                ]
            
            # Perform comprehensive AI analysis
            analysis = await self.ai_analyzer.analyze_project(enhanced_project_data)
            
            logger.info(f"AI analysis completed for {analysis.language.primary_language} project")
            logger.info(f"Detected frameworks: {', '.join(analysis.language.frameworks)}")
            logger.info(f"Project type: {analysis.architecture.project_type}")
            
            return analysis
            
        except Exception as e:
            error = f"Project analysis failed: {e}"
            self.errors.append(error)
            raise Exception(error)
    
    async def _generate_documentation(
        self,
        analysis: ProjectAnalysis,
        project_data: Dict[str, Any],
        template_type: str = None
    ) -> str:
        """Generate documentation using proper templates from readme_templates.py."""
        try:
            # Use Groq Analyzer with proper template system instead of template engine
            # Use appropriate method based on analyzer type
            if isinstance(self.ai_analyzer, XAIGrokAnalyzer):
                # Use comprehensive generation for xAI Grok 4 with 131K tokens
                documentation_result = await self.ai_analyzer.generate_comprehensive_readme(
                    project_data, 
                    template_type=template_type
                )
            else:
                # Use standard template generation for OCI/Groq
                documentation_result = await self.ai_analyzer.generate_readme_with_template(
                    project_data, 
                    template_type=template_type
                )
            
            if documentation_result["success"]:
                logger.info(f"Documentation generated using {documentation_result['template_type']} template via {documentation_result['method']}")
                logger.info(f"Model used: {documentation_result['model_used']}")
                return documentation_result["documentation"]
            else:
                logger.error("Template-based generation failed, falling back to basic template engine")
                # Fallback to template engine
                documentation = self.template_engine.generate_documentation(analysis, project_data)
                return documentation
            
        except Exception as e:
            logger.error(f"Template generation failed: {e}")
            # Final fallback to template engine
            try:
                documentation = self.template_engine.generate_documentation(analysis, project_data)
                logger.warning("Used fallback template engine")
                return documentation
            except Exception as fallback_error:
                error = f"Both template generation and fallback failed: {e}, {fallback_error}"
                self.errors.append(error)
                raise Exception(error)
    
    async def _finalize_documentation(
        self,
        documentation: str,
        analysis: ProjectAnalysis
    ) -> str:
        """Finalize and validate documentation."""
        try:
            # Basic validation
            if len(documentation.strip()) < 100:
                warning = "Generated documentation seems very short"
                self.warnings.append(warning)
                logger.warning(warning)
            
            # Check for placeholder content
            placeholders = [
                "<!-- Add", "TODO:", "FIXME:", "{", "}"
            ]
            
            for placeholder in placeholders:
                if placeholder in documentation:
                    warning = f"Documentation contains placeholder content: {placeholder}"
                    self.warnings.append(warning)
                    logger.warning(warning)
            
            # Ensure proper markdown structure
            if not documentation.startswith("#"):
                documentation = f"# {analysis.language.primary_language} Project\n\n" + documentation
            
            # Add table of contents if documentation is long
            if len(documentation) > 2000:
                toc = self._generate_table_of_contents(documentation)
                if toc:
                    # Insert TOC after the first heading
                    lines = documentation.split('\n')
                    for i, line in enumerate(lines):
                        if line.startswith('# '):
                            lines.insert(i + 2, toc)
                            break
                    documentation = '\n'.join(lines)
            
            logger.info("Documentation finalized")
            
            return documentation
            
        except Exception as e:
            warning = f"Documentation finalization failed: {e}"
            self.warnings.append(warning)
            logger.warning(warning)
            return documentation  # Return as-is if finalization fails
    
    def _generate_table_of_contents(self, documentation: str) -> str:
        """Generate table of contents from headers."""
        try:
            lines = documentation.split('\n')
            toc_lines = ["## 📚 Table of Contents\n"]
            
            for line in lines:
                if line.startswith('## '):
                    title = line[3:].strip()
                    # Clean emojis and special characters for anchor
                    anchor = title.lower()
                    anchor = ''.join(c if c.isalnum() or c in ' -_' else '' for c in anchor)
                    anchor = anchor.replace(' ', '-')
                    toc_lines.append(f"- [{title}](#{anchor})")
                elif line.startswith('### '):
                    title = line[4:].strip()
                    anchor = title.lower()
                    anchor = ''.join(c if c.isalnum() or c in ' -_' else '' for c in anchor)
                    anchor = anchor.replace(' ', '-')
                    toc_lines.append(f"  - [{title}](#{anchor})")
            
            if len(toc_lines) > 2:  # Only add TOC if there are headers
                return '\n'.join(toc_lines) + '\n'
            
            return ""
            
        except Exception as e:
            logger.warning(f"TOC generation failed: {e}")
            return ""
    
    def _generate_metadata(
        self,
        project_data: Dict[str, Any],
        analysis: Optional[ProjectAnalysis],
        generation_time: float
    ) -> Dict[str, Any]:
        """Generate metadata about the documentation generation process."""
        
        repo = project_data.get("repository", {})
        
        metadata = {
            "generation_time": generation_time,
            "timestamp": str(asyncio.get_event_loop().time()),
            "repository": {
                "name": repo.get("name", "Unknown"),
                "full_name": repo.get("full_name", ""),
                "private": repo.get("private", False),
                "language": repo.get("language", ""),
                "size": repo.get("size", 0),
                "stars": repo.get("stargazers_count", 0),
                "forks": repo.get("forks_count", 0)
            },
            "analysis": {
                "files_analyzed": len(project_data.get("files", [])),
                "images_found": len(project_data.get("images", {})),
                "important_files_read": len(project_data.get("file_contents", {}))
            },
            "settings": {
                "vision_enabled": self.settings.vision.analysis_enabled,
                "groq_model_text": self.settings.groq.model_text,
                "groq_model_vision": self.settings.groq.model_vision
            }
        }
        
        if analysis:
            metadata["analysis"].update({
                "detected_language": analysis.language.primary_language,
                "language_confidence": analysis.language.confidence,
                "frameworks": analysis.language.frameworks,
                "project_type": analysis.architecture.project_type,
                "visual_insights_count": len(analysis.visual_insights)
            })
        
        return metadata
    
    async def generate_batch(self, repo_urls: List[str]) -> List[GenerationResult]:
        """Generate documentation for multiple repositories."""
        results = []
        
        logger.info(f"Starting batch generation for {len(repo_urls)} repositories")
        
        for i, repo_url in enumerate(repo_urls, 1):
            logger.info(f"Processing repository {i}/{len(repo_urls)}: {repo_url}")
            
            try:
                result = await self.generate(repo_url)
                results.append(result)
                
                if result.success:
                    logger.info(f"✓ Successfully generated documentation for {repo_url}")
                else:
                    logger.error(f"✗ Failed to generate documentation for {repo_url}")
                
            except Exception as e:
                logger.error(f"✗ Error processing {repo_url}: {e}")
                results.append(GenerationResult(
                    success=False,
                    documentation="",
                    analysis=None,
                    metadata={"error": str(e)},
                    errors=[str(e)],
                    warnings=[],
                    generation_time=0.0
                ))
            
            # Add delay between requests to respect rate limits
            if i < len(repo_urls):
                await asyncio.sleep(1)
        
        successful = len([r for r in results if r.success])
        logger.info(f"Batch generation completed: {successful}/{len(repo_urls)} successful")
        
        return results
    
    async def save_documentation(
        self,
        documentation: str,
        output_path: Optional[Path] = None,
        filename: str = "README.md"
    ) -> Path:
        """Save generated documentation to file."""
        
        if output_path is None:
            output_path = Path.cwd()
        
        file_path = output_path / filename
        
        try:
            # Ensure directory exists
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Write documentation
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(documentation)
            
            logger.info(f"Documentation saved to: {file_path}")
            return file_path
            
        except Exception as e:
            error = f"Failed to save documentation: {e}"
            logger.error(error)
            raise Exception(error)
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about the generation process."""
        return {
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "error_messages": self.errors,
            "warning_messages": self.warnings
        }
    
    async def _check_cache_validity(self, repo_url: str) -> tuple[str, bool, Optional[str]]:
        """
        Check if cached documentation is still valid.
        
        Returns:
            tuple[cache_status, cache_hit, cached_documentation]
        """
        try:
            # Get latest commit information
            async with self.mcp_client as mcp:
                latest_commit = await mcp.get_latest_commit(repo_url)
                
            # Check if we should regenerate documentation
            should_regenerate = await self.cache_manager.should_regenerate_documentation(
                repo_url, 
                latest_commit.sha,
                latest_commit.date
            )
            
            if not should_regenerate:
                # Cache is valid, retrieve cached documentation
                cached_documentation = await self.cache_manager.get_cached_documentation(repo_url)
                if cached_documentation:
                    return "cache_valid", True, cached_documentation
                else:
                    # Cache entry exists but documentation is missing
                    return "cache_partial", False, None
            else:
                # Repository has been updated or cache has expired
                return "cache_stale", False, None
                
        except Exception as e:
            logger.warning(f"Cache validation failed: {e}")
            return "cache_error", False, None
    
    async def _update_cache(
        self, 
        repo_url: str, 
        project_data: Dict[str, Any], 
        documentation: str,
        analysis: ProjectAnalysis
    ) -> None:
        """Update cache with new documentation and repository information."""
        if not self.cache_manager or not self.cache_manager._connected:
            logger.warning("Cache manager not available, skipping cache update")
            return
        
        try:
            # Get the latest commit information
            recent_commits = project_data.get("recent_commits", [])
            if not recent_commits:
                logger.warning("No commit information available for cache update")
                return
            
            latest_commit = recent_commits[0]
            
            # Generate documentation hash
            doc_hash = self.cache_manager.generate_documentation_hash(documentation)
            
            # Prepare analysis summary for cache
            analysis_summary = {
                "language": analysis.language.primary_language,
                "language_confidence": analysis.language.confidence,
                "frameworks": analysis.language.frameworks,
                "project_type": analysis.architecture.project_type,
                "visual_insights_count": len(analysis.visual_insights)
            }
            
            # Store repository information
            await self.cache_manager.set_repository_info(
                repo_url,
                latest_commit["sha"],
                latest_commit["date"],
                doc_hash,
                analysis_summary
            )
            
            # Store documentation
            await self.cache_manager.set_cached_documentation(repo_url, documentation)
            
            logger.info(f"Updated cache for repository: {repo_url}")
            
        except Exception as e:
            logger.warning(f"Failed to update cache: {e}")
    
    def _generate_cache_metadata(
        self, 
        cached_repo_info, 
        generation_time: float
    ) -> Dict[str, Any]:
        """Generate metadata for cached documentation."""
        
        base_metadata = {
            "generation_time": generation_time,
            "timestamp": str(asyncio.get_event_loop().time()),
            "cache": {
                "hit": True,
                "last_generated": cached_repo_info.last_documentation_generated if cached_repo_info else "unknown",
                "commit_sha": cached_repo_info.last_commit_sha if cached_repo_info else "unknown"
            }
        }
        
        if cached_repo_info:
            base_metadata["analysis"] = cached_repo_info.analysis_summary
            
        return base_metadata
    
    async def invalidate_cache(self, repo_url: str) -> bool:
        """Invalidate cache for a specific repository."""
        if not self.cache_manager:
            self.cache_manager = await create_cache_manager(self.settings)
        
        return await self.cache_manager.invalidate_repository(repo_url)
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cache_manager:
            self.cache_manager = await create_cache_manager(self.settings)
        
        return await self.cache_manager.get_cache_stats()
    
    async def clear_cache(self) -> bool:
        """Clear all cached data."""
        if not self.cache_manager:
            self.cache_manager = await create_cache_manager(self.settings)
        
        return await self.cache_manager.clear_cache()