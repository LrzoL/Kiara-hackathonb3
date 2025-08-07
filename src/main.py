"""Main orchestrator and CLI interface for GitHub Documentation Agent."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .azure_devops_client import AzureDevOpsClient
from .config import get_settings, get_local_settings, reload_settings
from .doc_generator import DocumentationGenerator
from .local_analyzer import LocalAnalyzer
from .validators import ValidatorCollection

# Import GitHub processor for intelligent commit tracking
sys.path.insert(0, str(Path(__file__).parent.parent))
from github_processor import GitHubProcessor

# Setup rich console with Windows compatibility
console = Console(force_terminal=True, legacy_windows=False)

# Setup logging
def setup_logging(level: str = "INFO") -> None:
    """Setup logging with rich formatting."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)]
    )


@click.group()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Set logging level"
)
@click.option(
    "--config-file",
    type=click.Path(exists=True),
    help="Path to custom configuration file"
)
def cli(log_level: str, config_file: Optional[str]) -> None:
    """GitHub Documentation Agent - Generate automatic documentation for repositories."""
    setup_logging(log_level)
    
    if config_file:
        # TODO: Implement custom config file loading
        console.print(f"[yellow]Custom config file support not yet implemented: {config_file}[/yellow]")


@cli.command()
@click.argument("repository_url_or_path")
@click.option(
    "--output-path",
    "-o",
    type=click.Path(),
    help="Output directory for generated documentation"
)
@click.option(
    "--filename",
    "-f",
    default="README.md",
    help="Output filename (default: README.md)"
)
@click.option(
    "--no-vision",
    is_flag=True,
    help="Disable vision analysis of images"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate inputs without generating documentation"
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output"
)
@click.option(
    "--force",
    is_flag=True,
    help="Force regeneration, ignore cache"
)
@click.option(
    "--local",
    "-l",
    is_flag=True,
    help="Treat input as local project directory path instead of GitHub URL"
)
@click.option(
    "--smart-tracking",
    "-s",
    is_flag=True,
    help="Use intelligent commit tracking (only generate if new commits detected)"
)
def generate(
    repository_url_or_path: str,
    output_path: Optional[str],
    filename: str,
    no_vision: bool,
    dry_run: bool,
    verbose: bool,
    force: bool,
    local: bool,
    smart_tracking: bool
) -> None:
    """Generate documentation for a GitHub repository or local project directory."""
    
    if verbose:
        setup_logging("DEBUG")
    
    # Detect project type
    is_local_project = local or _is_local_path(repository_url_or_path)
    repository_type = None
    
    if not is_local_project:
        # Detect repository type for remote URLs
        validator = ValidatorCollection()
        validation_result, repo_type = validator.universal_repo.validate(repository_url_or_path)
        repository_type = repo_type
    
    # Display appropriate header
    if is_local_project:
        console.print(Panel.fit(
            "[bold blue]GitHub Documentation Agent - Local Mode[/bold blue]\n"
            "Generating documentation for local project directory",
            title="Local Doc Agent"
        ))
    elif repository_type == 'azure_devops':
        console.print(Panel.fit(
            "[bold blue]GitHub Documentation Agent - Azure DevOps Mode[/bold blue]\n"
            "Generating documentation for Azure DevOps repository",
            title="Azure DevOps Doc Agent"
        ))
    else:
        console.print(Panel.fit(
            "[bold blue]GitHub Documentation Agent[/bold blue]\n"
            "Generating automatic documentation with AI analysis",
            title="Doc Agent"
        ))
    
    # Load and validate settings
    try:
        if is_local_project:
            # For local projects, only require Groq API key
            settings = get_local_settings()
        elif repository_type == 'azure_devops':
            # For Azure DevOps, require Groq + Azure DevOps PAT
            settings = get_settings()
            if not settings.azure_devops_pat:
                raise ValueError("AZURE_DEVOPS_PAT is required for Azure DevOps repositories")
        else:
            # For GitHub projects, require all tokens
            settings = get_settings()
        
        if no_vision:
            settings.vision.analysis_enabled = False
    except Exception as e:
        console.print(f"[red]Error loading settings: {e}[/red]")
        if is_local_project:
            console.print("[yellow]For local projects, only GROQ_API_KEY is required[/yellow]")
        elif repository_type == 'azure_devops':
            console.print("[yellow]For Azure DevOps repositories, GROQ_API_KEY and AZURE_DEVOPS_PAT are required[/yellow]")
        else:
            console.print("[yellow]Please check your .env file and configuration[/yellow]")
        sys.exit(1)
    
    # Handle validation based on project type
    if is_local_project:
        # Validate local path
        if not _validate_local_path(repository_url_or_path):
            console.print(f"[red]Invalid local project path: {repository_url_or_path}[/red]")
            sys.exit(1)
        
        if dry_run:
            console.print("[green]+ Dry run completed successfully. Local path is valid.[/green]")
            return
        
        final_output_path = output_path or str(Path.cwd())
        final_filename = filename
        normalized_repo = repository_url_or_path
        
    elif repository_type in ['github', 'azure_devops']:
        # Validate remote repository inputs
        validator = ValidatorCollection(settings)
        
        if repository_type == 'azure_devops':
            # Use universal validator for Azure DevOps
            repo_validation, _ = validator.validate_universal_repository_input(repository_url_or_path)
            validation_results = {
                "repository": repo_validation,
                "output": validator.validate_output_settings(output_path, filename)
            }
        else:
            # Use existing validation for GitHub
            validation_results = validator.validate_all_inputs(
                repository_url_or_path,
                output_path,
                filename
            )
        
        # Display validation results
        _display_validation_results(validation_results)
        
        # Check if validation passed
        validation_summary = validator.get_validation_summary(validation_results)
        if not validation_summary["all_valid"]:
            console.print("[red]Validation failed. Please fix the errors above.[/red]")
            sys.exit(1)
        
        if validation_summary["total_warnings"] > 0:
            console.print(f"[yellow]Continuing with {validation_summary['total_warnings']} warnings...[/yellow]")
        
        if dry_run:
            console.print("[green]+ Dry run completed successfully. All inputs are valid.[/green]")
            return
        
        # Get normalized values
        normalized_repo = validation_results["repository"].normalized_value
        normalized_output = validation_results["output"].normalized_value or {}
        final_output_path = normalized_output.get("output_path", output_path)
        final_filename = normalized_output.get("filename", filename)
    
    else:
        console.print("[red]Unsupported repository type or invalid URL format[/red]")
        sys.exit(1)
    
    # Run documentation generation
    if is_local_project:
        asyncio.run(_generate_local_documentation(
            repository_url_or_path,
            final_output_path,
            final_filename,
            settings,
            force
        ))
    elif repository_type == 'azure_devops':
        asyncio.run(_generate_azure_devops_documentation(
            normalized_repo,
            final_output_path,
            final_filename,
            settings,
            force
        ))
    else:  # GitHub
        if smart_tracking and not force:
            # Use intelligent commit tracking
            asyncio.run(_generate_smart_documentation(
                normalized_repo,
                final_output_path,
                final_filename,
                settings
            ))
        else:
            # Use traditional generation
            asyncio.run(_generate_documentation(
                normalized_repo,
                final_output_path,
                final_filename,
                settings,
                force
            ))


@cli.command()
@click.argument("project_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option(
    "--output-path",
    "-o",
    type=click.Path(),
    help="Output directory for generated documentation"
)
@click.option(
    "--filename",
    "-f",
    default="README.md",
    help="Output filename (default: README.md)"
)
@click.option(
    "--no-vision",
    is_flag=True,
    help="Disable vision analysis of images"
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output"
)
def local(
    project_path: str,
    output_path: Optional[str],
    filename: str,
    no_vision: bool,
    verbose: bool
) -> None:
    """Generate documentation for a local project directory."""
    
    if verbose:
        setup_logging("DEBUG")
    
    console.print(Panel.fit(
        "[bold blue]GitHub Documentation Agent - Local Mode[/bold blue]\n"
        "Generating documentation for local project directory",
        title="Local Doc Agent"
    ))
    
    # Load settings
    try:
        settings = get_local_settings()
        if no_vision:
            settings.vision.analysis_enabled = False
    except Exception as e:
        console.print(f"[red]Error loading settings: {e}[/red]")
        console.print("[yellow]For local projects, only GROQ_API_KEY is required[/yellow]")
        console.print("[yellow]Please add GROQ_API_KEY to your .env file[/yellow]")
        sys.exit(1)
    
    final_output_path = output_path or str(Path.cwd())
    
    # Run local documentation generation
    asyncio.run(_generate_local_documentation(
        project_path,
        final_output_path,
        filename,
        settings,
        False  # force_regenerate
    ))


@cli.command()
@click.argument("repositories_file", type=click.File('r'))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    help="Output directory for all generated documentation"
)
@click.option(
    "--no-vision",
    is_flag=True,
    help="Disable vision analysis of images"
)
@click.option(
    "--concurrent",
    "-c",
    default=1,
    type=int,
    help="Number of concurrent generations (default: 1)"
)
def batch(
    repositories_file,
    output_dir: Optional[str],
    no_vision: bool,
    concurrent: int
) -> None:
    """Generate documentation for multiple repositories from a file."""
    
    console.print(Panel.fit(
        "[bold blue]GitHub Documentation Agent - Batch Mode[/bold blue]\n"
        "Processing multiple repositories",
        title="Batch Processing"
    ))
    
    # Read repository URLs
    try:
        repo_urls = [line.strip() for line in repositories_file.readlines() if line.strip()]
        if not repo_urls:
            console.print("[red]No repository URLs found in file[/red]")
            sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error reading repositories file: {e}[/red]")
        sys.exit(1)
    
    console.print(f"Found {len(repo_urls)} repositories to process")
    
    # Load settings
    try:
        settings = get_settings()
        if no_vision:
            settings.vision.analysis_enabled = False
    except Exception as e:
        console.print(f"[red]Error loading settings: {e}[/red]")
        sys.exit(1)
    
    # Run batch generation
    asyncio.run(_batch_generate_documentation(
        repo_urls,
        output_dir,
        concurrent,
        settings
    ))


@cli.command()
def config() -> None:
    """Display current configuration and check setup."""
    
    console.print(Panel.fit(
        "[bold blue]Configuration Check[/bold blue]",
        title="Settings"
    ))
    
    try:
        settings = get_settings()
        validator = ValidatorCollection(settings)
        
        # Validate settings
        settings_result = validator.settings.validate_complete_settings()
        
        # Display configuration table
        table = Table(title="Current Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Status", style="yellow")
        
        # GitHub settings
        github_token = settings.github.token[:10] + "..." if settings.github.token else "Not set"
        table.add_row("GitHub Token", github_token, "OK" if settings.github.token else "MISSING")
        table.add_row("GitHub API URL", settings.github.api_url, "OK")
        
        # Groq settings
        groq_key = settings.groq.api_key[:10] + "..." if settings.groq.api_key else "Not set"
        table.add_row("Groq API Key", groq_key, "OK" if settings.groq.api_key else "MISSING")
        table.add_row("Text Model", settings.groq.model_text, "OK")
        table.add_row("Vision Model", settings.groq.model_vision, "OK")
        
        # Application settings
        table.add_row("Log Level", settings.app.log_level, "OK")
        table.add_row("Cache Enabled", str(settings.app.cache_enabled), "OK")
        table.add_row("Vision Analysis", str(settings.vision.analysis_enabled), "OK")
        
        console.print(table)
        
        # Display validation results
        if settings_result.errors:
            console.print("\n[red]Configuration Errors:[/red]")
            for error in settings_result.errors:
                console.print(f"  X {error}")
        
        if settings_result.warnings:
            console.print("\n[yellow]Configuration Warnings:[/yellow]")
            for warning in settings_result.warnings:
                console.print(f"  ! {warning}")
        
        if settings_result.valid:
            console.print("\n[green]Configuration is valid and ready to use![/green]")
        else:
            console.print("\n[red]Configuration has errors that need to be fixed.[/red]")
            console.print("\nPlease check your .env file and ensure all required tokens are set.")
    
    except Exception as e:
        console.print(f"[red]Error checking configuration: {e}[/red]")
        sys.exit(1)


@cli.command()
def setup() -> None:
    """Interactive setup wizard for configuration."""
    
    console.print(Panel.fit(
        "[bold blue]Setup Wizard[/bold blue]\n"
        "Configure GitHub Documentation Agent",
        title="Setup"
    ))
    
    # Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        if not click.confirm("A .env file already exists. Do you want to update it?"):
            return
    
    console.print("\n[yellow]Please provide the following configuration:[/yellow]\n")
    
    # Get GitHub token
    github_token = click.prompt(
        "GitHub Personal Access Token",
        type=str,
        hide_input=True,
        confirmation_prompt=False
    )
    
    # Validate GitHub token
    validator = ValidatorCollection()
    github_result = validator.settings.validate_github_token(github_token)
    
    if not github_result.valid:
        console.print(f"[red]Invalid GitHub token: {', '.join(github_result.errors)}[/red]")
        return
    
    if github_result.warnings:
        for warning in github_result.warnings:
            console.print(f"[yellow]Warning: {warning}[/yellow]")
    
    # Get Groq API key
    groq_api_key = click.prompt(
        "Groq API Key",
        type=str,
        hide_input=True,
        confirmation_prompt=False
    )
    
    # Validate Groq API key
    groq_result = validator.settings.validate_groq_api_key(groq_api_key)
    
    if not groq_result.valid:
        console.print(f"[red]Invalid Groq API key: {', '.join(groq_result.errors)}[/red]")
        return
    
    if groq_result.warnings:
        for warning in groq_result.warnings:
            console.print(f"[yellow]Warning: {warning}[/yellow]")
    
    # Optional settings
    enable_vision = click.confirm("Enable vision analysis for images?", default=True)
    log_level = click.prompt(
        "Log level",
        type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
        default="INFO"
    )
    
    # Create .env content
    env_content = f"""# GitHub Configuration
GITHUB_TOKEN={github_token}
GITHUB_API_URL=https://api.github.com

# Groq Configuration
GROQ_API_KEY={groq_api_key}
GROQ_MODEL_TEXT=llama-3.1-70b-versatile
GROQ_MODEL_VISION=llama-3.2-90b-vision-preview
GROQ_MAX_TOKENS=8000
GROQ_TEMPERATURE=0.1

# MCP Configuration
MCP_SERVER_COMMAND=npx
MCP_SERVER_ARGS=@github/github-mcp-server

# Application Settings
APP_LOG_LEVEL={log_level}
APP_CACHE_ENABLED=true
APP_CACHE_TTL=3600
APP_MAX_FILE_SIZE=10485760
APP_RATE_LIMIT_REQUESTS=30
APP_RATE_LIMIT_PERIOD=60

# Vision Processing
VISION_SUPPORTED_IMAGE_FORMATS=png,jpg,jpeg,gif,bmp,webp,svg
VISION_MAX_IMAGE_SIZE=5242880
VISION_ANALYSIS_ENABLED={str(enable_vision).lower()}

# Repository Access
REPO_PRIVATE_REPOS_ENABLED=true
REPO_USER_REPOS_ONLY=false

# Redis Cache Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
REDIS_TTL=7200
REDIS_ENABLED=true
"""
    
    # Write .env file
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        console.print(f"\n[green]+ Configuration saved to {env_file}[/green]")
        console.print("\n[blue]Setup completed! You can now use 'github-doc-agent generate <repo-url>' to generate documentation.[/blue]")
        
    except Exception as e:
        console.print(f"[red]Error saving configuration: {e}[/red]")


@cli.group()
def cache() -> None:
    """Cache management commands."""
    pass


@cache.command("stats")
def cache_stats() -> None:
    """Show cache statistics."""
    console.print(Panel.fit(
        "[bold blue]Cache Statistics[/bold blue]",
        title="Cache Info"
    ))
    
    asyncio.run(_show_cache_stats())


@cache.command("clear")
@click.confirmation_option(prompt="Are you sure you want to clear all cache?")
def cache_clear() -> None:
    """Clear all cached data."""
    console.print(Panel.fit(
        "[bold blue]Clearing Cache[/bold blue]",
        title="Cache Management"
    ))
    
    asyncio.run(_clear_cache())


@cache.command("invalidate")
@click.argument("repository_url")
def cache_invalidate(repository_url: str) -> None:
    """Invalidate cache for a specific repository."""
    console.print(Panel.fit(
        f"[bold blue]Invalidating Cache[/bold blue]\n{repository_url}",
        title="Cache Management"
    ))
    
    asyncio.run(_invalidate_cache(repository_url))


async def _show_cache_stats() -> None:
    """Show cache statistics."""
    try:
        settings = get_settings()
        generator = DocumentationGenerator(settings)
        stats = await generator.get_cache_stats()
        
        if stats.get("status") == "disconnected":
            console.print("[yellow]Cache is not connected[/yellow]")
            return
        
        table = Table(title="Cache Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Status", stats.get("status", "unknown"))
        table.add_row("Redis Version", stats.get("redis_version", "unknown"))
        table.add_row("Used Memory", stats.get("used_memory", "unknown"))
        table.add_row("Connected Clients", str(stats.get("connected_clients", "unknown")))
        table.add_row("Total Keys", str(stats.get("total_keys", "unknown")))
        
        config = stats.get("config", {})
        table.add_row("Host", config.get("host", "unknown"))
        table.add_row("Port", str(config.get("port", "unknown")))
        table.add_row("TTL", f"{config.get('ttl', 'unknown')}s")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error getting cache stats: {e}[/red]")


async def _clear_cache() -> None:
    """Clear all cache."""
    try:
        settings = get_settings()
        generator = DocumentationGenerator(settings)
        success = await generator.clear_cache()
        
        if success:
            console.print("[green]✓ All cache cleared successfully[/green]")
        else:
            console.print("[yellow]! Cache clear may have failed[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Error clearing cache: {e}[/red]")


async def _invalidate_cache(repo_url: str) -> None:
    """Invalidate cache for a repository."""
    try:
        settings = get_settings()
        generator = DocumentationGenerator(settings)
        success = await generator.invalidate_cache(repo_url)
        
        if success:
            console.print(f"[green]✓ Cache invalidated for {repo_url}[/green]")
        else:
            console.print(f"[yellow]! Cache invalidation may have failed for {repo_url}[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Error invalidating cache: {e}[/red]")


async def _generate_documentation(
    repo_url: str,
    output_path: Optional[str],
    filename: str,
    settings,
    force_regenerate: bool = False
) -> None:
    """Generate documentation for a single repository."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Initializing...", total=None)
        
        try:
            generator = DocumentationGenerator(settings)
            
            progress.update(task, description="Generating documentation...")
            result = await generator.generate(repo_url, force_regenerate=force_regenerate)
            
            if result.success:
                progress.update(task, description="Saving documentation...")
                
                # Save documentation
                output_dir = Path(output_path) if output_path else Path.cwd()
                file_path = await generator.save_documentation(
                    result.documentation,
                    output_dir,
                    filename
                )
                
                progress.update(task, description="Completed!")
                
                # Display success information
                _display_generation_success(result, file_path)
                
            else:
                progress.update(task, description="Failed!")
                _display_generation_failure(result)
                
        except Exception as e:
            progress.update(task, description="Error!")
            console.print(f"[red]Unexpected error: {e}[/red]")
            console.print("\n[yellow]Please check your configuration and try again.[/yellow]")


async def _batch_generate_documentation(
    repo_urls: list,
    output_dir: Optional[str],
    concurrent: int,
    settings
) -> None:
    """Generate documentation for multiple repositories."""
    
    output_path = Path(output_dir) if output_dir else Path.cwd() / "generated_docs"
    output_path.mkdir(parents=True, exist_ok=True)
    
    generator = DocumentationGenerator(settings)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        main_task = progress.add_task(f"Processing {len(repo_urls)} repositories...", total=len(repo_urls))
        
        # Process repositories in batches
        if concurrent > 1:
            console.print(f"[yellow]Note: Concurrent processing ({concurrent}) may hit rate limits[/yellow]")
        
        results = []
        for i in range(0, len(repo_urls), concurrent):
            batch = repo_urls[i:i + concurrent]
            batch_results = await generator.generate_batch(batch)
            results.extend(batch_results)
            
            # Save successful results
            for j, result in enumerate(batch_results):
                repo_url = batch[j]
                if result.success:
                    # Extract repo name for filename
                    repo_name = repo_url.split("/")[-1].replace(".git", "")
                    filename = f"{repo_name}_README.md"
                    
                    try:
                        await generator.save_documentation(
                            result.documentation,
                            output_path,
                            filename
                        )
                    except Exception as e:
                        console.print(f"[yellow]Warning: Failed to save {filename}: {e}[/yellow]")
                
                progress.advance(main_task)
        
        # Display batch results
        _display_batch_results(results, repo_urls, output_path)


def _display_validation_results(validation_results: dict) -> None:
    """Display validation results in a formatted way."""
    
    for category, result in validation_results.items():
        if result.errors:
            console.print(f"\n[red]{category.title()} Errors:[/red]")
            for error in result.errors:
                console.print(f"  X {error}")
        
        if result.warnings:
            console.print(f"\n[yellow]{category.title()} Warnings:[/yellow]")
            for warning in result.warnings:
                console.print(f"  ! {warning}")


def _display_generation_success(result, file_path: Path) -> None:
    """Display successful generation results."""
    
    if result.cache_hit:
        console.print("\n[green]+ Documentation retrieved from cache![/green]")
    else:
        console.print("\n[green]+ Documentation generated successfully![/green]")
    
    # Create results table
    table = Table(title="Generation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Output File", str(file_path))
    table.add_row("Generation Time", f"{result.generation_time:.2f}s")
    table.add_row("Cache Hit", "✓" if result.cache_hit else "✗")
    table.add_row("Cache Status", result.cache_status)
    
    if result.analysis:
        table.add_row("Detected Language", result.analysis.language.primary_language)
        table.add_row("Project Type", result.analysis.architecture.project_type)
        table.add_row("Frameworks", ", ".join(result.analysis.language.frameworks[:3]))
        table.add_row("Visual Insights", str(len(result.analysis.visual_insights)))
    
    console.print(table)
    
    if result.warnings:
        console.print(f"\n[yellow]Warnings ({len(result.warnings)}):[/yellow]")
        for warning in result.warnings[:5]:  # Show first 5 warnings
            console.print(f"  ! {warning}")
    
    console.print(f"\n[blue]Documentation saved to: {file_path}[/blue]")


def _display_generation_failure(result) -> None:
    """Display failed generation results."""
    
    console.print("\n[red]X Documentation generation failed![/red]")
    
    if result.errors:
        console.print(f"\n[red]Errors ({len(result.errors)}):[/red]")
        for error in result.errors:
            console.print(f"  X {error}")
    
    if result.warnings:
        console.print(f"\n[yellow]Warnings ({len(result.warnings)}):[/yellow]")
        for warning in result.warnings:
            console.print(f"  ! {warning}")


def _display_batch_results(results: list, repo_urls: list, output_path: Path) -> None:
    """Display batch generation results."""
    
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    console.print(f"\n[blue]Batch Processing Complete![/blue]")
    console.print(f"+ Successful: {len(successful)}")
    console.print(f"X Failed: {len(failed)}")
    console.print(f"Output directory: {output_path}")
    
    if failed:
        console.print(f"\n[red]Failed repositories:[/red]")
        for i, result in enumerate(failed):
            repo_url = repo_urls[results.index(result)]
            console.print(f"  X {repo_url}")
            if result.errors:
                console.print(f"    Error: {result.errors[0]}")


def _is_local_path(path_str: str) -> bool:
    """Check if the input string is a local file path."""
    
    # Check if it's clearly a URL
    if path_str.startswith(('http://', 'https://', 'git@')):
        return False
    
    # Check if it's a file system path
    if '/' in path_str or '\\' in path_str:
        return True
    
    # Check if it's a relative path that exists
    if Path(path_str).exists():
        return True
    
    return False


def _validate_local_path(path_str: str) -> bool:
    """Validate that the local path exists and is a directory."""
    
    path = Path(path_str)
    
    if not path.exists():
        console.print(f"[red]Path does not exist: {path_str}[/red]")
        return False
    
    if not path.is_dir():
        console.print(f"[red]Path is not a directory: {path_str}[/red]")
        return False
    
    return True


async def _generate_azure_devops_documentation(
    repo_url: str,
    output_path: Optional[str],
    filename: str,
    settings,
    force_regenerate: bool = False
) -> None:
    """Generate documentation for an Azure DevOps repository."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Initializing Azure DevOps client...", total=None)
        
        try:
            # Initialize Azure DevOps client
            azure_client = AzureDevOpsClient(settings)
            
            progress.update(task, description="Collecting repository data...")
            
            # Collect repository data
            project_data = await azure_client.collect_repository_data(repo_url)
            
            progress.update(task, description="Initializing AI analyzer...")
            
            # Use existing components to generate documentation
            from .groq_analyzer import GroqAnalyzer
            from .template_engine import TemplateEngine
            
            groq_analyzer = GroqAnalyzer(settings)
            template_engine = TemplateEngine(settings)
            
            progress.update(task, description="Analyzing project with AI...")
            
            # Analyze the project using existing analyzers
            analysis = await groq_analyzer.analyze_project(project_data)
            
            progress.update(task, description="Generating documentation...")
            
            # Generate documentation using template engine
            documentation = template_engine.generate_documentation(analysis, project_data)
            
            progress.update(task, description="Saving documentation...")
            
            # Save documentation
            output_dir = Path(output_path) if output_path else Path.cwd()
            output_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = output_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(documentation)
            
            progress.update(task, description="Completed!")
            
            # Display success information
            _display_azure_devops_generation_success(project_data, analysis, file_path)
            
            # Close the Azure DevOps client
            await azure_client.close()
            
        except Exception as e:
            progress.update(task, description="Error!")
            console.print(f"[red]Error generating documentation for Azure DevOps repository: {e}[/red]")
            console.print("\n[yellow]Please check your Azure DevOps PAT token and repository URL.[/yellow]")


async def _generate_local_documentation(
    project_path: str,
    output_path: Optional[str],
    filename: str,
    settings,
    force_regenerate: bool = False
) -> None:
    """Generate documentation for a local project directory."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Analyzing local project...", total=None)
        
        try:
            # Initialize local analyzer
            local_analyzer = LocalAnalyzer(settings)
            
            progress.update(task, description="Scanning project directory...")
            
            # Analyze the local project
            project_data = await local_analyzer.analyze_local_project(project_path)
            
            progress.update(task, description="Converting to MCP format...")
            
            # Convert to MCP-compatible format
            mcp_data = local_analyzer.convert_to_mcp_format(project_data)
            
            progress.update(task, description="Initializing AI analyzer...")
            
            # Use existing components to generate documentation
            from .groq_analyzer import GroqAnalyzer
            from .template_engine import TemplateEngine
            
            groq_analyzer = GroqAnalyzer(settings)
            template_engine = TemplateEngine(settings)
            
            progress.update(task, description="Analyzing project with AI...")
            
            # Analyze the project using existing analyzers
            analysis = await groq_analyzer.analyze_project(mcp_data)
            
            progress.update(task, description="Generating documentation...")
            
            # Generate documentation using template engine
            documentation = template_engine.generate_documentation(analysis, mcp_data)
            
            progress.update(task, description="Saving documentation...")
            
            # Save documentation
            output_dir = Path(output_path) if output_path else Path.cwd()
            output_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = output_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(documentation)
            
            progress.update(task, description="Completed!")
            
            # Display success information
            _display_local_generation_success(project_data, analysis, file_path)
            
        except Exception as e:
            progress.update(task, description="Error!")
            console.print(f"[red]Error generating documentation for local project: {e}[/red]")
            console.print("\n[yellow]Please check that the project directory is accessible and try again.[/yellow]")


def _display_azure_devops_generation_success(project_data, analysis, file_path: Path) -> None:
    """Display successful Azure DevOps generation results."""
    
    console.print("\n[green]+ Documentation generated successfully for Azure DevOps repository![/green]")
    
    # Create results table
    table = Table(title="Azure DevOps Generation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Output File", str(file_path))
    table.add_row("Repository", project_data['repository']['full_name'])
    table.add_row("Organization", project_data['metadata'].get('organization', 'Unknown'))
    table.add_row("Project", project_data['metadata'].get('project', 'Unknown'))
    table.add_row("Total Files", str(project_data['metadata'].get('total_files', 0)))
    table.add_row("Text Files", str(project_data['metadata'].get('text_files_count', 0)))
    table.add_row("Images", str(project_data['metadata'].get('images_count', 0)))
    table.add_row("Repository Size", f"{project_data['repository'].get('size', 0):,} bytes")
    
    if analysis:
        table.add_row("Detected Language", analysis.language.primary_language)
        table.add_row("Project Type", analysis.architecture.project_type)
        table.add_row("Frameworks", ", ".join(analysis.language.frameworks[:3]))
    
    console.print(table)
    
    console.print(f"\n[blue]Documentation saved to: {file_path}[/blue]")


def _display_local_generation_success(project_data, analysis, file_path: Path) -> None:
    """Display successful local generation results."""
    
    console.print("\n[green]+ Documentation generated successfully for local project![/green]")
    
    # Create results table
    table = Table(title="Local Generation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Output File", str(file_path))
    table.add_row("Project Path", project_data.project_path)
    table.add_row("Project Name", project_data.project_name)
    table.add_row("Total Files", str(project_data.metadata.get('total_files', 0)))
    table.add_row("Text Files", str(project_data.metadata.get('text_files_count', 0)))
    table.add_row("Images", str(project_data.metadata.get('images_count', 0)))
    table.add_row("Project Size", f"{project_data.metadata.get('total_size', 0):,} bytes")
    
    if analysis:
        table.add_row("Detected Language", analysis.language.primary_language)
        table.add_row("Project Type", analysis.architecture.project_type)
        table.add_row("Frameworks", ", ".join(analysis.language.frameworks[:3]))
    
    console.print(table)
    
    # Show project characteristics
    characteristics = project_data.metadata.get('project_characteristics', [])
    if characteristics:
        console.print(f"\n[blue]Project Characteristics:[/blue] {', '.join(characteristics)}")
    
    console.print(f"\n[blue]Documentation saved to: {file_path}[/blue]")


async def _generate_smart_documentation(
    repository_url: str,
    output_path: Optional[str],
    filename: str,
    settings
) -> None:
    """Generate documentation using intelligent commit tracking."""
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            
            task = progress.add_task("Initializing smart documentation generator...", total=None)
            
            # Initialize GitHub processor
            github_processor = GitHubProcessor()
            
            progress.update(task, description="Checking repository commits...")
            
            # Process repository with intelligent commit tracking
            success = await github_processor.process_github_url(repository_url)
            
            if success:
                progress.update(task, description="Documentation completed!")
                
                # Get repository stats
                stats = github_processor.tracker.get_repository_stats(repository_url)
                
                # Display success information
                _display_smart_generation_success(stats, filename)
                
            else:
                progress.update(task, description="Generation failed!")
                console.print("[red]Failed to generate documentation with smart tracking[/red]")
                console.print("[yellow]Try using --force to bypass commit tracking[/yellow]")
                
    except Exception as e:
        console.print(f"[red]Error in smart documentation generation: {e}[/red]")


def _display_smart_generation_success(stats: dict, filename: str) -> None:
    """Display smart generation results."""
    
    if 'error' in stats:
        console.print(f"[red]Error: {stats['error']}[/red]")
        return
    
    repo = stats['repository']
    
    console.print("\n[green]+ Smart documentation generation completed![/green]")
    
    # Create results table
    table = Table(title="Smart Generation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Repository", f"{repo['owner']}/{repo['name']}")
    table.add_row("Language", repo['language'] or 'Unknown')
    table.add_row("Stars", str(repo['stars']))
    table.add_row("Private", "Yes" if repo['is_private'] else "No")
    table.add_row("First Processed", repo['first_processed_at'])
    table.add_row("Last Checked", repo['last_checked_at'])
    table.add_row("Total Commits Tracked", str(stats['total_commits']))
    table.add_row("Successful Generations", str(stats['successful_generations']))
    
    if stats['latest_generation']:
        gen = stats['latest_generation']
        table.add_row("Latest Generation", gen['generated_at'])
        table.add_row("Generated for Commit", gen['sha'][:8])
        table.add_row("Commit Message", gen['message'][:50] + "..." if len(gen['message']) > 50 else gen['message'])
    
    console.print(table)
    
    # Check if README was generated or skipped
    readme_path = Path(filename)
    if readme_path.exists():
        console.print(f"\n[blue]Documentation saved to: {readme_path.absolute()}[/blue]")
    else:
        console.print(f"\n[yellow]README already up to date - no generation needed[/yellow]")


def main() -> None:
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()