#!/usr/bin/env python
"""
Kiara CLI Entry Point
Main command-line interface for Kiara Documentation Agent with xAI Grok 4 support.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import click
from rich.console import Console
import sys

# Configure console for Windows compatibility
console = Console(force_terminal=True, legacy_windows=False)

# Set UTF-8 encoding for Windows
if sys.platform.startswith('win'):
    import os
    os.system('chcp 65001 > nul')

class KiaraGroup(click.Group):
    """Custom Click group with Rich formatting for help."""
    
    def format_help(self, ctx, formatter):
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich import box
        from rich.text import Text
        from rich.align import Align
        
        console = Console(force_terminal=True, legacy_windows=False)
        
        # Header with dynamic AI provider detection
        ai_info = "Powered by xAI Grok 4 + Oracle Cloud AI (131K tokens)"
        try:
            from src.config import reload_settings
            settings = reload_settings()
            # Force xAI Grok detection since it's enabled
            if hasattr(settings, 'xai_grok_enabled') and settings.xai_grok_enabled:
                ai_info = "Powered by xAI Grok 4 + Oracle Cloud AI (131K tokens)"
            elif hasattr(settings, 'oci_enabled') and settings.oci_enabled:
                ai_info = "Powered by Oracle Cloud AI (Meta Llama + Scout Vision)"
            else:
                ai_info = "Powered by Multi-AI Architecture"
        except Exception as e:
            # If can't load settings, default to xAI Grok
            ai_info = "Powered by xAI Grok 4 + Oracle Cloud AI (131K tokens)"
            
        header_text = f"""
        KIARA - AI Documentation Agent Enhanced
        
        {ai_info}
        Generate comprehensive documentation for any project
        """
        
        header_panel = Panel(
            Align.center(Text(header_text, style="bold bright_blue")),
            border_style="bright_magenta",
            padding=(1, 2)
        )
        
        # Commands table with same styling as interactive
        commands_table = Table(
            show_header=True,
            header_style="bold bright_magenta",
            box=box.ROUNDED,
            padding=(0, 1),
            expand=False
        )
        commands_table.add_column("Command", style="bright_blue", width=12)
        commands_table.add_column("Description", style="white", width=40)
        commands_table.add_column("Status", style="bright_green", width=8)
        
        # Add commands to table
        commands_table.add_row("analyze", "Analyze a local project and generate documentation", "Ready")
        commands_table.add_row("github", "Generate documentation for a GitHub repository", "Ready")
        commands_table.add_row("serve", "Start the Kiara API server", "Ready")
        commands_table.add_row("interactive", "Start Kiara in interactive mode", "Ready")
        commands_table.add_row("info", "Show Kiara information and configuration", "Ready")
        
        commands_panel = Panel(
            commands_table,
            title="[bold bright_magenta]Available Commands[/bold bright_magenta]",
            border_style="bright_blue",
            padding=(1, 2)
        )
        
        # Usage examples
        usage_text = Text()
        usage_text.append("Examples:\n", style="bold bright_magenta")
        usage_text.append("  kiara interactive                     # Start interactive mode\n", style="white")
        usage_text.append("  kiara analyze ./my-project            # Analyze local project\n", style="white") 
        usage_text.append("  kiara github https://github.com/...   # Analyze GitHub repo\n", style="white")
        usage_text.append("  kiara serve                           # Start API server\n", style="white")
        usage_text.append("  kiara info                            # Show configuration\n", style="white")
        
        usage_panel = Panel(
            usage_text,
            title="[bold bright_magenta]Usage Examples[/bold bright_magenta]",
            border_style="bright_blue",
            padding=(1, 2)
        )
        
        # Display all panels
        console.print()
        console.print(header_panel)
        console.print()
        console.print(commands_panel)
        console.print()
        console.print(usage_panel)
        console.print()

@click.group(cls=KiaraGroup)
@click.version_option(version="0.1.0")
def main():
    """Kiara - AI Documentation Agent Enhanced"""
    pass

@main.command()
@click.option('--host', default='0.0.0.0', help='Host to serve on')
@click.option('--port', default=8000, help='Port to serve on')  
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def serve(host, port, reload):
    """Start the Kiara API server."""
    try:
        import uvicorn
        from api.main import app
        
        console.print(f"Starting Kiara API Server", style="bold green")
        console.print(f"Server: http://{host}:{port}", style="blue")
        console.print(f"AI: xAI Grok 4 (131K tokens) + Oracle Cloud AI", style="yellow")
        console.print(f"Features: GitHub, Azure DevOps, Local Projects", style="cyan")
        
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except ImportError:
        console.print("FastAPI/Uvicorn not installed. Install with: pip install fastapi uvicorn", style="bold red")
    except Exception as e:
        console.print(f"Server error: {e}", style="bold red")

@main.command()
def interactive():
    """Start Kiara in interactive mode."""
    try:
        console.print("Starting Kiara Interactive Mode...", style="bold green")
        
        # Import and run interactive mode
        from kiara_interactive_enhanced import main as interactive_main
        asyncio.run(interactive_main())
        
    except Exception as e:
        console.print(f"Interactive mode error: {e}", style="bold red")

@main.command()
@click.argument('project_path', type=click.Path(exists=True))
@click.option('--template', type=click.Choice(['minimal', 'emoji_rich', 'modern', 'technical_deep', 'comprehensive']), 
              default='emoji_rich', help='Template type to use')
@click.option('--output', '-o', help='Output file path')
def analyze(project_path, template, output):
    """Analyze a local project and generate documentation."""
    try:
        console.print(f"Analyzing project: {project_path}", style="bold blue")
        console.print(f"Template: {template}", style="yellow")
        console.print(f"AI: xAI Grok 4 (131K tokens output)", style="green")
        
        # Import and run local analysis
        from src.local_analyzer import generate_local_documentation_cli
        
        async def run_analysis():
            result = await generate_local_documentation_cli(project_path, template_type=template)
            
            if result and result.get('success') and result.get('readme'):
                readme_content = result['readme']
                
                # Save to file
                if output:
                    output_path = Path(output)
                else:
                    output_path = Path(project_path) / "README_kiara_generated.md"
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(readme_content)
                
                console.print(f"Documentation generated successfully!", style="bold green")
                console.print(f"Saved to: {output_path}", style="blue")
                console.print(f"Size: {len(readme_content):,} characters", style="cyan")
                console.print(f"Words: {len(readme_content.split()):,}", style="cyan")
                
                # Show preview
                console.print(f"\nPreview (first 500 chars):", style="bold")
                console.print("-" * 50)
                safe_preview = ''.join(c for c in readme_content[:500] if ord(c) < 128 or c.isalnum() or c in ' \n\t.,!?-()[]{}:;')
                console.print(safe_preview)
                console.print("-" * 50)
                
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'Analysis failed'
                console.print(f"Analysis failed: {error_msg}", style="bold red")
        
        asyncio.run(run_analysis())
        
    except Exception as e:
        console.print(f"Analysis error: {e}", style="bold red")

@main.command()
@click.argument('github_url')
@click.option('--template', type=click.Choice(['minimal', 'emoji_rich', 'modern', 'technical_deep']), 
              default='emoji_rich', help='Template type to use')
@click.option('--output', '-o', help='Output file path')
def github(github_url, template, output):
    """Generate documentation for a GitHub repository."""
    try:
        console.print(f"Analyzing GitHub repository: {github_url}", style="bold blue")
        console.print(f"Template: {template}", style="yellow")
        console.print(f"AI: xAI Grok 4 (131K tokens output)", style="green")
        
        # Import and run GitHub analysis
        from src.doc_generator import DocumentationGenerator
        
        async def run_github_analysis():
            generator = DocumentationGenerator()
            result = await generator.generate(github_url, template_type=template)
            
            if result.success:
                readme_content = result.readme
                
                # Save to file
                if output:
                    output_path = Path(output)
                else:
                    repo_name = github_url.split('/')[-1]
                    output_path = Path(f"README_{repo_name}_kiara.md")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(readme_content)
                
                console.print(f"GitHub documentation generated!", style="bold green")
                console.print(f"Saved to: {output_path}", style="blue")
                console.print(f"Size: {len(readme_content):,} characters", style="cyan")
                console.print(f"Time: {result.generation_time:.2f}s", style="cyan")
                
            else:
                console.print(f"GitHub analysis failed: {result.errors}", style="bold red")
        
        asyncio.run(run_github_analysis())
        
    except Exception as e:
        console.print(f"GitHub analysis error: {e}", style="bold red")

@main.command()
def info():
    """Show Kiara information and configuration."""
    try:
        from src.config import reload_settings
        
        console.print("Kiara Configuration Information", style="bold blue")
        console.print("=" * 50)
        
        # Force reload to get latest configuration
        settings = reload_settings()
        
        # Show current primary AI provider first
        if hasattr(settings, 'xai_grok_enabled') and settings.xai_grok_enabled:
            console.print(f"Primary AI: xAI Grok 4 (ACTIVE)", style="bold bright_green")
            console.print(f"   Max Tokens: {settings.xai_grok_max_tokens:,}", style="cyan")
            console.print(f"   Model ID: ...{settings.xai_grok_model_id[-30:]}", style="cyan")
            console.print()
            
            # Show OCI as backup
            console.print(f"Backup AI: Oracle Cloud AI", style="yellow" if settings.oci_enabled else "red")
            if settings.oci_enabled:
                console.print(f"   Max Tokens: {settings.oci_max_tokens:,}", style="cyan")
                console.print(f"   Model: Meta Llama + Scout Vision", style="cyan")
        else:
            console.print(f"Primary AI: Oracle Cloud AI", style="green" if settings.oci_enabled else "red") 
            if settings.oci_enabled:
                console.print(f"   Max Tokens: {settings.oci_max_tokens:,}", style="cyan")
                console.print(f"   Model: Meta Llama + Scout Vision", style="cyan")
            console.print()
            
            # Show xAI Grok as available but not enabled
            console.print(f"xAI Grok 4: DISABLED", style="red")
            console.print(f"   Available: 131K tokens output", style="dim cyan")
        
        console.print(f"GitHub Token: {'Configured' if settings.github_token else 'Missing'}", 
                     style="green" if settings.github_token else "red")
        
        console.print(f"Azure DevOps PAT: {'Configured' if settings.azure_devops_pat else 'Missing'}", 
                     style="green" if settings.azure_devops_pat else "red")
        
        console.print(f"\nAvailable Templates:", style="bold")
        templates = ['minimal', 'emoji_rich', 'modern', 'technical_deep']
        for template in templates:
            console.print(f"   - {template}", style="cyan")
        
        console.print(f"\nSupported Sources:", style="bold")
        sources = ['Local Projects', 'GitHub Repositories', 'Azure DevOps Repositories']
        for source in sources:
            console.print(f"   - {source}", style="cyan")
        
    except Exception as e:
        console.print(f"Configuration error: {e}", style="bold red")

if __name__ == "__main__":
    main()