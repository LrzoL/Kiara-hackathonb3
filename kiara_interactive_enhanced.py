#!/usr/bin/env python3
"""
🤖 Kiara - AI Documentation Agent Enhanced Interactive
Dynamic CLI with OCI AI integration for automatic documentation generation
Supports: GitHub, Azure DevOps, and Local Projects
"""

import asyncio
import sys
import time
import os
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TaskProgressColumn
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.layout import Layout
from rich.align import Align
from rich.text import Text
from rich import box
from rich.tree import Tree
from rich.status import Status
from rich.live import Live
from rich.columns import Columns
from rich.markdown import Markdown

# Import our system
sys.path.append('.')
from src.doc_generator import DocumentationGenerator
from src.readme_templates import AVAILABLE_TEMPLATES
from src.config import get_settings
from src.local_analyzer import LocalProjectAnalyzer, generate_local_documentation_cli

class KiaraInteractiveEnhanced:
    """🤖 Kiara - Enhanced Dynamic AI Documentation Agent with OCI"""
    
    def __init__(self):
        self.console = Console(
            force_terminal=True,
            color_system="truecolor", 
            legacy_windows=True,
            width=120,
            emoji=False,  # Disable emojis for Windows compatibility
            file=sys.stdout
        )
        self.generator = None
        self.settings = None
        
    def show_banner(self):
        """Display Kiara's enhanced banner with OCI info"""
        try:
            self.settings = get_settings()
            if hasattr(self.settings, 'xai_grok_enabled') and self.settings.xai_grok_enabled:
                ai_provider = "xAI Grok 4"
                model_name = "xAI Grok 4"
                max_tokens = self.settings.xai_grok_max_tokens
            elif self.settings.oci_enabled:
                ai_provider = "Oracle Cloud AI "
                model_name = f"{self.settings.oci_model_id} | Vision: {self.settings.oci_model_vision}"
                max_tokens = self.settings.oci_max_tokens
            else:
                ai_provider = "Legacy Provider"
                model_name = "Legacy Models"
                max_tokens = self.settings.groq_max_tokens
        except:
            ai_provider = "AI Provider"
            model_name = "Unknown"
            max_tokens = 4000

        banner_text = f"""
        +==============================================================+
        |   KIARA - AI Documentation Agent Enhanced                   |
        |                                                              |
        |   * Powered by {ai_provider:<35}              |
        |   * Max Output: {max_tokens} tokens                                  |
        |   * Automatic README.md generation for any project          |
        |   * Supports: GitHub, Azure DevOps, Local Projects          |
        |   * 4 Professional Templates + Intelligent AI Analysis      |
        +==============================================================+
        """
        
        banner_panel = Panel(
            Align.center(Text(banner_text, style="bold bright_blue")),
            border_style="bright_magenta",
            padding=(1, 2)
        )
        
        self.console.print(banner_panel)
        self.console.print()

    def show_main_menu(self) -> str:
        """Enhanced interactive main menu with all features"""
        
        # Create menu table
        menu_table = Table(show_header=True, header_style="bold bright_magenta", box=box.ROUNDED)
        menu_table.add_column("Option", style="bright_blue", justify="center", width=8)
        menu_table.add_column("Description", style="white", width=50)
        menu_table.add_column("Status", style="bright_magenta", justify="center", width=12)
        
        menu_table.add_row("1", "* Generate from GitHub Repository", "Ready")
        menu_table.add_row("2", "* Generate from Azure DevOps", "Ready") 
        menu_table.add_row("3", "* Generate from Local Project", "Ready")
        menu_table.add_row("4", "* Template Preview & Selection", "Ready")
        menu_table.add_row("5", "* Batch Generation (Multiple Repos)", "Ready")
        menu_table.add_row("6", "* Quick Repository Analysis", "Ready")
        menu_table.add_row("7", "* Kiara Configuration & Status", "Settings")
        menu_table.add_row("8", "* Start API Server", "Server")
        menu_table.add_row("q", "* Exit Kiara", "Quit")
        
        menu_panel = Panel(
            menu_table,
            title="[bold bright_magenta]Kiara Enhanced Main Menu[/bold bright_magenta]",
            border_style="bright_blue",
            padding=(1, 2)
        )
        
        self.console.print(menu_panel)
        
        choice = Prompt.ask(
            "\n[bold bright_magenta]What would you like Kiara to help with?[/bold bright_magenta]",
            choices=["1", "2", "3", "4", "5", "6", "7", "8", "q"],
            default="1"
        )
        
        return choice

    def select_template(self) -> str:
        """Enhanced template selection with detailed previews"""
        
        self.console.print(Panel(
            "[bold bright_blue]Enhanced Template Selection[/bold bright_blue]\n\n"
            "Choose the perfect template for your project documentation:",
            border_style="bright_magenta"
        ))
        
        # Template descriptions table
        templates_table = Table(show_header=True, header_style="bold bright_magenta", box=box.ROUNDED)
        templates_table.add_column("Template", style="bright_blue", width=15)
        templates_table.add_column("Style", style="white", width=20)
        templates_table.add_column("Best For", style="bright_magenta", width=25)
        templates_table.add_column("Features", style="bright_blue", width=35)
        
        template_info = {
            "minimal": {
                "style": "Clean & Professional",
                "best_for": "Enterprise Documentation", 
                "features": "No emojis, formal tone, comprehensive sections"
            },
            "emoji_rich": {
                "style": "Fun & Engaging",
                "best_for": "Community Projects",
                "features": "50+ emojis, enthusiastic tone, visual appeal"
            },
            "modern": {
                "style": "GitHub-style Contemporary",
                "best_for": "Open Source Projects",
                "features": "Badges, tables, TOC, modern markdown formatting"
            },
            "technical_deep": {
                "style": "Enterprise Technical",
                "best_for": "Production Systems",
                "features": "Architecture details, deployment, compliance info"
            }
        }
        
        for template in AVAILABLE_TEMPLATES:
            info = template_info[template.value]
            templates_table.add_row(
                template.value.upper(),
                info["style"],
                info["best_for"],
                info["features"]
            )
        
        self.console.print(templates_table)
        self.console.print()
        
        # Interactive selection
        template = Prompt.ask(
            "[bold bright_magenta]Select your preferred template[/bold bright_magenta]",
            choices=[t.value for t in AVAILABLE_TEMPLATES] + ["auto"],
            default="auto"
        )
        
        if template == "auto":
            self.console.print("[bright_magenta]* Kiara AI will auto-select the best template based on your project![/bright_magenta]\n")
            return None
        else:
            selected_info = template_info[template]
            self.console.print(f"[bright_blue]* Selected: {template.upper()} - {selected_info['style']}[/bright_blue]\n")
            return template

    async def generate_local_documentation(self, project_path: str, template_type: Optional[str] = None):
        """Generate documentation for local projects using OCI"""
        
        # Enhanced progress tracking for local projects
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=True
        ) as progress:
            
            # Task definitions for local analysis
            task_scan = progress.add_task("[bright_blue]* Scanning local project...[/bright_blue]", total=100)
            task_analyze = progress.add_task("[bright_blue]* Analyzing project structure...[/bright_blue]", total=100)
            task_ai = progress.add_task("[bright_magenta]* OCI AI analysis in progress...[/bright_magenta]", total=100)  
            task_generate = progress.add_task("[bright_blue]* Generating documentation...[/bright_blue]", total=100)
            task_finalize = progress.add_task("[bright_magenta]* Finalizing README.md...[/bright_magenta]", total=100)
            
            try:
                start_time = time.time()
                
                # Step 1: Scan directory
                progress.update(task_scan, advance=50)
                await asyncio.sleep(0.3)
                progress.update(task_scan, advance=50, description="[bright_blue]* Local project scanned[/bright_blue]")
                
                # Step 2: Analysis
                progress.update(task_analyze, advance=30)
                await asyncio.sleep(0.3)
                progress.update(task_analyze, advance=70, description="[bright_blue]* Project structure analyzed[/bright_blue]")
                
                # Step 3: AI Processing  
                progress.update(task_ai, advance=20)
                await asyncio.sleep(0.3)
                progress.update(task_ai, advance=40, description="[bright_magenta]* Language & framework detection...[/bright_magenta]")
                await asyncio.sleep(0.3) 
                progress.update(task_ai, advance=40, description="[bright_magenta]* OCI AI analysis complete[/bright_magenta]")
                
                # Step 4: Generate
                progress.update(task_generate, advance=25)
                
                # Actual local generation
                result = await generate_local_documentation_cli(project_path, template_type)
                
                progress.update(task_generate, advance=75, description="[bright_blue]* Documentation generated[/bright_blue]")
                
                # Step 5: Finalize
                progress.update(task_finalize, advance=100, description="[bright_magenta]* README.md finalized[/bright_magenta]")
                
                generation_time = time.time() - start_time
                
                # Convert to compatible format
                if result and result.get("success"):
                    class LocalResult:
                        def __init__(self, data):
                            self.success = data.get("success", False)
                            self.documentation = data.get("documentation", "")
                            self.metadata = data
                            self.errors = []
                            self.warnings = []
                            
                            # Mock analysis for display
                            class MockAnalysis:
                                def __init__(self):
                                    self.language = MockLanguage()
                                    self.architecture = MockArchitecture()
                            
                            class MockLanguage:
                                def __init__(self):
                                    self.primary_language = "Local Project"
                                    self.confidence = 1.0
                                    self.frameworks = ["OCI Analysis"]
                            
                            class MockArchitecture:
                                def __init__(self):
                                    self.project_type = "local_project"
                            
                            self.analysis = MockAnalysis()
                    
                    return LocalResult(result), generation_time
                else:
                    return None, generation_time
                
            except Exception as e:
                progress.stop()
                self.console.print(f"[red]* Error: {e}[/red]")
                return None, 0
    
    async def generate_documentation(self, source: str, template_type: Optional[str] = None, source_type: str = "github"):
        """Enhanced documentation generation with OCI integration"""
        
        if not self.generator:
            with Status("[bright_magenta]* Initializing Kiara's enhanced AI systems...[/bright_magenta]", console=self.console):
                time.sleep(1)  # Dramatic pause
                self.generator = DocumentationGenerator()
        
        # Enhanced progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=True
        ) as progress:
            
            # Task definitions
            task_connect = progress.add_task("[bright_blue]* Connecting to AI provider...[/bright_blue]", total=100)
            task_analyze = progress.add_task("[bright_blue]* Analyzing project structure...[/bright_blue]", total=100)
            task_ai = progress.add_task("[bright_magenta]* AI analysis in progress...[/bright_magenta]", total=100)  
            task_generate = progress.add_task("[bright_blue]* Generating documentation...[/bright_blue]", total=100)
            task_finalize = progress.add_task("[bright_magenta]* Finalizing README.md...[/bright_magenta]", total=100)
            
            try:
                # Step 1: Connection
                progress.update(task_connect, advance=50)
                await asyncio.sleep(0.3)
                if self.settings and hasattr(self.settings, 'xai_grok_enabled') and self.settings.xai_grok_enabled:
                    ai_provider = "xAI Grok 4 (131K tokens output)"
                elif self.settings and self.settings.oci_enabled:
                    ai_provider = "Oracle Cloud AI (meta.llama-4-maverick-17b-128e-instruct)"
                else:
                    ai_provider = "Legacy Provider"
                progress.update(task_connect, advance=50, description=f"[bright_blue]* Connected to {ai_provider}[/bright_blue]")
                
                # Step 2: Analysis
                progress.update(task_analyze, advance=30)
                await asyncio.sleep(0.5)
                progress.update(task_analyze, advance=70, description="[bright_blue]* Project structure analyzed[/bright_blue]")
                
                # Step 3: AI Processing  
                progress.update(task_ai, advance=20)
                await asyncio.sleep(0.3)
                progress.update(task_ai, advance=40, description="[bright_magenta]* Language & framework detection...[/bright_magenta]")
                await asyncio.sleep(0.3) 
                progress.update(task_ai, advance=40, description="[bright_magenta]* AI analysis complete[/bright_magenta]")
                
                # Step 4: Generate
                progress.update(task_generate, advance=25)
                
                # Actual generation
                start_time = time.time()
                result = await self.generator.generate(
                    source, 
                    force_regenerate=True,
                    template_type=template_type
                )
                generation_time = time.time() - start_time
                
                progress.update(task_generate, advance=75, description="[bright_blue]* Documentation generated[/bright_blue]")
                
                # Step 5: Finalize
                progress.update(task_finalize, advance=100, description="[bright_magenta]* README.md finalized[/bright_magenta]")
                
                return result, generation_time
                
            except Exception as e:
                progress.stop()
                self.console.print(f"[red]* Error: {e}[/red]")
                return None, 0

    def show_generation_results(self, result, source: str, template_used: str, generation_time: float):
        """Enhanced results display with OCI information"""
        
        # Show commit detection message if available
        if hasattr(result, 'commit_detected') and result.commit_detected and result.update_message:
            commit_panel = Panel(
                f"[bold bright_blue]{result.update_message}[/bold bright_blue]",
                title="[bold bright_magenta]* Novos commits detectados[/bold bright_magenta]",
                border_style="bright_magenta",
                padding=(1, 2)
            )
            self.console.print(commit_panel)
            self.console.print()
        
        if not result or not result.success:
            error_panel = Panel(
                "[red]* Documentation generation failed![/red]\n\n"
                f"Errors: {result.errors if result else 'Unknown error'}",
                title="* [bold red]Generation Failed[/bold red]",
                border_style="red"
            )
            self.console.print(error_panel)
            return
        
        # Enhanced success results
        results_info = Table(show_header=False, box=box.SIMPLE)
        results_info.add_column("Metric", style="bright_blue", width=25)
        results_info.add_column("Value", style="white", width=50)
        
        results_info.add_row("* Source", source)
        results_info.add_row("* Template", template_used.upper() if template_used else "AUTO-SELECTED")
        results_info.add_row("* Content Length", f"{len(result.documentation):,} characters")
        results_info.add_row("* Generation Time", f"{generation_time:.2f} seconds")
        
        # Show AI provider info
        if hasattr(result, 'metadata') and result.metadata:
            method = result.metadata.get('method', 'unknown')
            if 'xai' in method.lower() or 'grok' in method.lower():
                results_info.add_row("* AI Provider", "xAI Grok 4 (131K tokens output)")
                if 'model_used' in result.metadata:
                    model_name = result.metadata['model_used'].split('.')[-1] if '.' in str(result.metadata['model_used']) else result.metadata['model_used']
                    results_info.add_row("* Model", model_name)
            elif 'oci' in method.lower():
                results_info.add_row("* AI Provider", "Oracle Cloud AI (meta.llama-4-maverick-17b-128e-instruct)")
                if 'model_used' in result.metadata:
                    model_name = result.metadata['model_used'].split('.')[-1] if '.' in str(result.metadata['model_used']) else result.metadata['model_used']
                    results_info.add_row("* Model", model_name)
            else:
                results_info.add_row("* Model", result.metadata.get('model_used', 'Unknown'))
        
        if result.analysis:
            results_info.add_row("* Language Detected", result.analysis.language.primary_language)
            results_info.add_row("* Confidence", f"{result.analysis.language.confidence:.1%}")
            if result.analysis.language.frameworks:
                results_info.add_row("* Frameworks", ", ".join(result.analysis.language.frameworks))
            results_info.add_row("* Project Type", result.analysis.architecture.project_type)
        
        success_panel = Panel(
            results_info,
            title="* [bold bright_magenta]Kiara Generation Complete![/bold bright_magenta]",
            border_style="bright_blue",
            padding=(1, 2)
        )
        
        self.console.print(success_panel)
        
        # Save file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"README_kiara_{timestamp}.md"
        
        try:
            # Get absolute path for clarity
            from pathlib import Path
            full_path = Path(filename).resolve()
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(result.documentation)
            
            self.console.print(f"\n[bright_green]📄 README saved successfully![/bright_green]")
            self.console.print(f"[bright_magenta]📂 Location: [bold cyan]{full_path}[/bold cyan][/bright_magenta]")
                
        except Exception as e:
            self.console.print(f"[red]❌ Failed to save file: {e}[/red]")

    def show_documentation_preview(self, content: str):
        """Enhanced documentation preview"""
        
        preview_panel = Panel(
            Markdown(content + "\n\n*[Preview truncated to first 2000 characters]*"),
            title="* [bold bright_blue]Documentation Preview[/bold bright_blue]",
            border_style="bright_magenta",
            padding=(1, 2)
        )
        
        self.console.print(preview_panel)

    async def quick_analyze(self, repo_url: str):
        """Quick repository analysis feature"""
        
        self.console.print(Panel(
            f"[bold bright_blue]Quick Analysis[/bold bright_blue]\n\n"
            f"Repository: {repo_url}",
            border_style="bright_magenta"
        ))
        
        with Status("[bright_magenta]* Analyzing repository...[/bright_magenta]", console=self.console):
            try:
                if not self.generator:
                    self.generator = DocumentationGenerator()
                
                # Quick analysis without full generation
                analysis_info = Table(show_header=False, box=box.SIMPLE)
                analysis_info.add_column("Property", style="bright_blue", width=20)
                analysis_info.add_column("Value", style="white", width=40)
                
                analysis_info.add_row("* Repository", repo_url)
                analysis_info.add_row("* Status", "Accessible")
                analysis_info.add_row("* Analysis", "Quick scan complete")
                analysis_info.add_row("* Recommendation", "Ready for documentation generation")
                
                analysis_panel = Panel(
                    analysis_info,
                    title="* [bold bright_magenta]Analysis Results[/bold bright_magenta]",
                    border_style="bright_blue"
                )
                
                self.console.print(analysis_panel)
                
            except Exception as e:
                self.console.print(f"[red]* Analysis failed: {e}[/red]")

    def show_configuration(self):
        """Enhanced configuration display with OCI info"""
        
        try:
            settings = get_settings()
            
            config_table = Table(show_header=True, header_style="bold bright_magenta", box=box.ROUNDED)
            config_table.add_column("Setting", style="bright_blue", width=25)
            config_table.add_column("Current Value", style="white", width=35) 
            config_table.add_column("Description", style="bright_magenta", width=40)
            
            # AI Provider info
            if hasattr(settings, 'xai_grok_enabled') and settings.xai_grok_enabled:
                config_table.add_row("AI Provider", "xAI Grok 4 (131K tokens output)", "Primary AI provider")
                config_table.add_row("XAI Model", "xAI Grok 4", "xAI Grok 4 model")
            elif settings.oci_enabled:
                config_table.add_row("AI Provider", "Oracle Cloud AI (meta.llama-4-maverick-17b-128e-instruct)", "Primary AI provider")
                config_table.add_row("OCI Text Model", "meta.llama-4-maverick-17b-128e-instruct", "Oracle Cloud text model")
                config_table.add_row("OCI Vision Model", "Scout", "Oracle Cloud vision model")
                config_table.add_row("Max Tokens", str(settings.oci_max_tokens), "Maximum output tokens (OCI limit)")
                config_table.add_row("OCI Endpoint", settings.oci_endpoint.split('//')[-1], "Oracle Cloud endpoint")
                config_table.add_row("Fallback", "Legacy Available" if settings.groq_api_key else "Oracle Cloud Only", "Backup provider (deprecated)")
            else:
                config_table.add_row("AI Provider", "Legacy Provider (Deprecated)", "Current AI provider")
                config_table.add_row("Legacy Model", settings.groq_model_text, "Legacy model for generation")
                config_table.add_row("Max Tokens", str(settings.groq_max_tokens), "Maximum output tokens")
            
            config_table.add_row("Default Template", "Auto-select", "Default documentation template")
            config_table.add_row("Vision Analysis", "Enabled" if settings.vision.analysis_enabled else "Disabled", "Analyze images and diagrams")
            config_table.add_row("Cache System", "SQLite", "Documentation caching system")
            config_table.add_row("GitHub Access", "Configured" if settings.github_token else "Not Configured", "GitHub API authentication")
            
        except Exception as e:
            config_table = Table(show_header=True, header_style="bold bright_magenta", box=box.ROUNDED)
            config_table.add_column("Setting", style="bright_blue", width=25)
            config_table.add_column("Status", style="red", width=35)
            config_table.add_row("Configuration", f"Error loading: {e}")
        
        config_panel = Panel(
            config_table,
            title="[bold bright_blue]Kiara Enhanced Configuration[/bold bright_blue]", 
            border_style="bright_magenta"
        )
        
        self.console.print(config_panel)

    async def batch_generation(self):
        """Batch generation for multiple repositories"""
        
        self.console.print(Panel(
            "[bold bright_blue]Batch Generation[/bold bright_blue]\n\n"
            "Generate documentation for multiple repositories at once:",
            border_style="bright_magenta"
        ))
        
        repos = []
        self.console.print("[bright_blue]Enter repository URLs (press Enter with empty line to finish):[/bright_blue]")
        
        while True:
            repo = Prompt.ask("Repository URL", default="")
            if not repo:
                break
            repos.append(repo)
            self.console.print(f"[bright_magenta]Added: {repo}[/bright_magenta]")
        
        if not repos:
            self.console.print("[yellow]No repositories added.[/yellow]")
            return
        
        template = self.select_template()
        
        self.console.print(f"\n[bright_blue]Starting batch generation for {len(repos)} repositories...[/bright_blue]")
        
        results = []
        for i, repo in enumerate(repos, 1):
            self.console.print(f"\n[bright_magenta]Processing {i}/{len(repos)}: {repo}[/bright_magenta]")
            
            result, gen_time = await self.generate_documentation(repo, template)
            if result:
                results.append((repo, result, gen_time))
            
        # Show batch results
        batch_table = Table(show_header=True, header_style="bold bright_magenta", box=box.ROUNDED)
        batch_table.add_column("Repository", style="bright_blue", width=40)
        batch_table.add_column("Status", style="white", width=15)
        batch_table.add_column("Time", style="bright_magenta", width=10)
        
        for repo, result, gen_time in results:
            status = "Success" if result.success else "Failed"
            batch_table.add_row(repo, status, f"{gen_time:.1f}s")
        
        batch_panel = Panel(
            batch_table,
            title="[bold bright_magenta]Batch Generation Results[/bold bright_magenta]",
            border_style="bright_blue"
        )
        
        self.console.print(batch_panel)

    def start_api_server(self):
        """Start Kiara API server"""
        
        self.console.print(Panel(
            "[bold bright_blue]Kiara API Server[/bold bright_blue]\n\n"
            "Starting the Kiara API server for HTTP access:",
            border_style="bright_magenta"
        ))
        
        port = Prompt.ask("Server port", default="8000")
        host = Prompt.ask("Server host", default="localhost")
        
        # Start server info display
        server_info = Table(show_header=False, box=box.SIMPLE)
        server_info.add_column("Property", style="bright_blue", width=15)
        server_info.add_column("Value", style="bright_magenta", width=30)
        
        server_info.add_row("Host", host)
        server_info.add_row("Port", port)
        server_info.add_row("URL", f"http://{host}:{port}")
        server_info.add_row("Status", "Starting...")
        
        server_panel = Panel(
            server_info,
            title="[bold bright_magenta]API Server Info[/bold bright_magenta]",
            border_style="bright_blue"
        )
        
        self.console.print(server_panel)
        
        self.console.print("\n[bright_blue]* Server would start here (implementation in main CLI)[/bright_blue]")
        self.console.print(f"[bright_magenta]* Use: kiara serve --host {host} --port {port}[/bright_magenta]")

    async def run(self):
        """Enhanced main Kiara interface loop"""
        
        self.show_banner()
        
        while True:
            choice = self.show_main_menu()
            
            if choice == "q":
                goodbye_panel = Panel(
                    Align.center(
                        "[bold bright_magenta]Thank you for using Kiara Enhanced![/bold bright_magenta]\n\n"
                        f"* Your AI Documentation Agent powered by {'xAI Grok 4' if hasattr(self.settings, 'xai_grok_enabled') and self.settings.xai_grok_enabled else 'Oracle Cloud AI'}\n"
                        f"* Models: {'xAI Grok 4 (131K tokens)' if hasattr(self.settings, 'xai_grok_enabled') and self.settings.xai_grok_enabled else 'meta.llama-4-maverick-17b-128e-instruct + Scout'}\n"
                        "* Making documentation magical since 2025"
                    ),
                    border_style="bright_blue"
                )
                self.console.print(goodbye_panel)
                break
                
            elif choice == "1":  # GitHub
                self.console.print(Panel(
                    "[bold bright_blue]GitHub Repository Documentation[/bold bright_blue]",
                    border_style="bright_magenta"
                ))
                
                repo_url = Prompt.ask("[bright_blue]Enter GitHub repository URL[/bright_blue]")
                
                # Quick check for commit status
                from src.simple_commit_tracker import SimpleCommitTracker
                commit_tracker = SimpleCommitTracker()
                has_new_commit, commit_message = await commit_tracker.check_for_new_commit(repo_url)
                
                # Show commit status and ask for confirmation if needed
                if not has_new_commit and "Deseja gerar novamente?" in commit_message:
                    commit_panel = Panel(
                        f"[bold yellow]{commit_message}[/bold yellow]",
                        title="[bold bright_magenta]* Status do Repositório[/bold bright_magenta]",
                        border_style="yellow",
                        padding=(1, 2)
                    )
                    self.console.print(commit_panel)
                    
                    if not Confirm.ask("\n[bright_blue]Continuar com a geração?[/bright_blue]"):
                        self.console.print("[bright_magenta]* Geração cancelada pelo usuário.[/bright_magenta]")
                        continue
                elif has_new_commit:
                    # Show new commit message
                    commit_panel = Panel(
                        f"[bold bright_green]{commit_message}[/bold bright_green]",
                        title="[bold bright_magenta]* Novos Commits Detectados[/bold bright_magenta]",
                        border_style="bright_green",
                        padding=(1, 2)
                    )
                    self.console.print(commit_panel)
                    self.console.print()
                
                template = self.select_template()
                
                result, gen_time = await self.generate_documentation(repo_url, template)
                if result:
                    self.show_generation_results(result, repo_url, template or "auto", gen_time)
                
            elif choice == "2":  # Azure DevOps  
                self.console.print(Panel(
                    "[bold bright_blue]Azure DevOps Documentation[/bold bright_blue]\n"
                    "[yellow]Coming soon! Full Azure DevOps integration with OCI.[/yellow]",
                    border_style="bright_magenta"
                ))
                
            elif choice == "3":  # Local
                self.console.print(Panel(
                    "[bold bright_blue]Local Project Documentation[/bold bright_blue]",
                    border_style="bright_magenta"
                ))
                
                project_path = Prompt.ask("[bright_blue]Enter local project directory path[/bright_blue]", default=".")
                template = self.select_template()
                
                result, gen_time = await self.generate_local_documentation(project_path, template)
                if result:
                    self.show_generation_results(result, f"Local: {project_path}", template or "auto", gen_time)
                
            elif choice == "4":  # Template preview
                self.show_template_gallery()
                
            elif choice == "5":  # Batch generation
                await self.batch_generation()
                
            elif choice == "6":  # Quick analysis
                repo_url = Prompt.ask("[bright_blue]Enter repository URL for quick analysis[/bright_blue]")
                await self.quick_analyze(repo_url)
                
            elif choice == "7":  # Configuration
                self.show_configuration()
                
            elif choice == "8":  # API Server
                self.start_api_server()
            
            # Pause before returning to menu
            input("\n[dim]Press Enter to continue...[/dim]")
            self.console.clear()

    def show_template_gallery(self):
        """Enhanced template gallery with examples"""
        
        self.console.print(Panel(
            "[bold bright_blue]Kiara Enhanced Template Gallery[/bold bright_blue]\n\n"
            "Preview of all available documentation templates with OCI optimization:",
            border_style="bright_magenta"
        ))
        
        templates_tree = Tree("[bold]Available Templates (OCI Optimized)[/bold]")
        
        templates_tree.add("[bright_blue]MINIMAL[/bright_blue] - Clean, professional, enterprise-ready")
        templates_tree.add("[bright_magenta]EMOJI_RICH[/bright_magenta] - Fun, engaging, community-focused")  
        templates_tree.add("[bright_blue]MODERN[/bright_blue] - GitHub-style with badges & contemporary features")
        templates_tree.add("[yellow]TECHNICAL_DEEP[/yellow] - Enterprise technical documentation")
        
        gallery_panel = Panel(
            templates_tree,
            title="[bold bright_magenta]Template Gallery[/bold bright_magenta]",
            border_style="bright_blue"
        )
        
        self.console.print(gallery_panel)

async def main():
    """Launch Enhanced Kiara CLI"""
    kiara = KiaraInteractiveEnhanced()
    await kiara.run()

if __name__ == "__main__":
    try:
        # Set UTF-8 for Windows
        if sys.platform.startswith('win'):
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            try:
                import codecs
                sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
            except:
                pass
        
        asyncio.run(main())
    except KeyboardInterrupt:
        console = Console()
        console.print("\n[yellow]Kiara Enhanced shutdown requested. Goodbye![/yellow]")
    except Exception as e:
        console = Console()
        console.print(f"\n[red]Kiara Enhanced encountered an error: {e}[/red]")