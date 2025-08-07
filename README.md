# 🤖 Kiara - AI Documentation Agent Enhanced

**AI-powered automatic documentation generator with multi-provider support**

Kiara is an intelligent AI documentation agent that automatically generates professional technical documentation for any project. Leveraging multiple AI providers including xAI Grok 4, Oracle Cloud AI, and Groq, Kiara analyzes GitHub repositories, Azure DevOps projects, and local codebases to create comprehensive, adaptive README files.

## ✨ Key Features

- 🧠 **Multi-AI Architecture**: Primary support for xAI Grok 4 (131K tokens), Oracle Cloud AI (Meta Llama + Scout Vision), and Groq
- 🔍 **Intelligent Analysis**: Automatic detection of languages, frameworks, architectures, and project patterns
- 📝 **Professional Templates**: 4+ adaptive templates (Minimal, Emoji Rich, Modern, Technical Deep, Comprehensive)
- 🚀 **Universal Support**: GitHub, Azure DevOps, and local project analysis
- 🔄 **Smart Commit Detection**: Intelligent system that tracks repository updates and suggests regeneration
- 💾 **Intelligent Caching**: SQLite3-based caching with Redis support for optimization
- 🎯 **Multiple Interfaces**: Simple CLI, enhanced interactive mode, and REST API server
- 👁️ **Vision Analysis**: Advanced image processing and screenshot understanding
- 📊 **MCP Integration**: Model Context Protocol support for enhanced GitHub analysis

## 🛠️ Architecture

### AI Providers
- **xAI Grok 4**: Primary provider with 131K token output capacity
- **Oracle Cloud AI**: Meta Llama models with Scout vision capabilities
- **Groq**: Fast inference with Llama 3.1 and vision models
- **Fallback System**: Automatic provider switching for reliability

### Core Components
- **Documentation Generator**: Main orchestrator for generating documentation
- **Template Engine**: Adaptive template system with professional styling
- **Language Detector**: Advanced programming language and framework detection
- **Vision Processor**: Image analysis and screenshot processing
- **Commit Tracker**: Intelligent repository change detection
- **MCP Client**: GitHub repository analysis via Model Context Protocol
- **Cache Manager**: Multi-tier caching with SQLite and Redis support

## 🚀 Installation

### Prerequisites

- **Python 3.11+** (required)
- **Node.js 18+** (for GitHub MCP Server)
- **API Keys**: At least one AI provider (xAI Grok, Oracle Cloud, or Groq)
- **GitHub Token** (for private repositories)

### Quick Installation

```bash
# Clone the project
git clone <repository-url>
cd Kiara

# Install dependencies (using uv - recommended)
pip install uv
uv sync

# Or using pip traditional
pip install -r requirements.txt

# Install GitHub MCP Server
npm install -g @github/github-mcp-server
```

### Configuration

1. **Set up environment file:**
```bash
cp .env.example .env
nano .env  # or your preferred editor
```

2. **Essential environment variables:**
```bash
# Primary AI Provider - xAI Grok (Recommended)
XAI_GROK_API_KEY=xai-your_api_key_here
XAI_GROK_MODEL_ID=grok-4-turbo-128k
XAI_GROK_MAX_TOKENS=131072
XAI_GROK_ENABLED=true

# Backup AI Provider - Oracle Cloud
OCI_COMPARTMENT_ID=ocid1.tenancy.oc1..your_compartment_id_here
OCI_MODEL_ID=meta.llama-4-maverick-17b-128e-instruct
OCI_MODEL_VISION=Scout
OCI_ENABLED=true

# GitHub Integration
GITHUB_TOKEN=ghp_your_github_token_here

# Optional: Azure DevOps
AZURE_DEVOPS_PAT=your_azure_devops_token_here
```

3. **Configure Oracle Cloud CLI (if using OCI):**
```bash
oci setup config
# Follow instructions to set up OCI credentials
```

## 🎯 Usage

### Option 1: Entry Point CLI
```bash
python kiara_entry.py
```

**Available Commands:**
- `analyze` - Analyze local project and generate documentation
- `github` - Generate documentation for GitHub repository  
- `serve` - Start API server
- `interactive` - Launch enhanced interactive mode
- `info` - Show configuration and status

### Option 2: Simple CLI
```bash
python kiara_cli.py
```

**Features:**
- GitHub repository documentation
- Local project analysis
- Template selection
- Automatic commit detection
- Clean, objective interface

### Option 3: Enhanced Interactive Mode
```bash
python kiara_interactive_enhanced.py
```

**Additional Features:**
- Batch processing (multiple repositories)
- Quick repository analysis
- Integrated API server
- Advanced configurations
- Rich interface with extensive options

### Option 4: API Server Mode
```bash
python kiara_entry.py serve --host 0.0.0.0 --port 8000
```

**Available Endpoints:**
- `POST /generate` - Generate documentation
- `GET /health` - API status
- `GET /templates` - List available templates
- `GET /config` - Show configuration status

## 🔄 Smart Commit Detection System

Kiara features an intelligent system that monitors GitHub repository updates:

### How It Works

1. **First Analysis**: System silently registers current commit
2. **Same Commit**: Asks if you want to regenerate documentation
3. **New Commits**: Shows: *"I see you've updated your project, let's update the README too!"*
4. **Smart Caching**: Avoids unnecessary regenerations while staying current

### Database Schema

- **SQLite3**: `cache/github_repos.db`
- **Tables**: 
  - `repositories` (url, owner, name, last_commit_sha, last_checked_at)
  - `generations` (commit history, success rates, metadata)
- **Scope**: GitHub repositories with intelligent change detection

## 📋 Available Templates

| Template | Style | Best For | Characteristics |
|----------|--------|----------|----------------|
| **MINIMAL** | Clean & Professional | Enterprise Documentation | No emojis, formal tone, comprehensive |
| **EMOJI_RICH** | Fun & Engaging | Community Projects | 50+ emojis, enthusiastic tone, visual appeal |
| **MODERN** | GitHub-style Contemporary | Open Source Projects | Badges, tables, TOC, modern markdown |
| **TECHNICAL_DEEP** | Enterprise Technical | Production Systems | Architecture, deployment, compliance |
| **COMPREHENSIVE** | All-inclusive | Complex Projects | Combines best features of all templates |

## ⚙️ Advanced Configuration

### xAI Grok 4 (Primary AI)

```bash
XAI_GROK_API_KEY=xai-your_api_key_here
XAI_GROK_MODEL_ID=grok-4-turbo-128k        # Latest Grok model
XAI_GROK_MAX_TOKENS=131072                  # 131K token output
XAI_GROK_TEMPERATURE=0.1                    # Creativity (0.0-1.0)
XAI_GROK_ENABLED=true
```

### Oracle Cloud AI (Backup)

```bash
OCI_COMPARTMENT_ID=ocid1.tenancy.oc1..your_id_here
OCI_MODEL_ID=meta.llama-4-maverick-17b-128e-instruct    # Text model
OCI_MODEL_VISION=Scout                                   # Vision model
OCI_MAX_TOKENS=4000                                     # Token limit
OCI_TEMPERATURE=0.2                                     # Creativity
OCI_ENABLED=true
```

### Groq (Fallback)

```bash
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL_TEXT=llama-3.1-70b-versatile
GROQ_MODEL_VISION=llama-3.2-90b-vision-preview
GROQ_MAX_TOKENS=8000
```

### GitHub Integration

```bash
# Required scopes for GitHub token:
# - repo (full repository access)
# - read:user (user information)
# - user:email (user emails)
GITHUB_TOKEN=ghp_your_token_with_repo_scope
GITHUB_API_URL=https://api.github.com
```

### Vision Analysis

```bash
VISION_ANALYSIS_ENABLED=true
VISION_MAX_IMAGE_SIZE=5242880           # 5MB
VISION_SUPPORTED_FORMATS=png,jpg,jpeg,gif,bmp,webp,svg
```

### Caching Configuration

```bash
# Redis (optional, improves performance)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
REDIS_TTL=7200
REDIS_ENABLED=true

# SQLite (always enabled)
APP_CACHE_ENABLED=true
APP_CACHE_TTL=3600
```

## 🔍 Usage Examples

### Analyze GitHub Repository

```bash
python kiara_entry.py github https://github.com/user/repo --template=modern
```

### Analyze Local Project

```bash
python kiara_entry.py analyze ./my-project --template=emoji_rich --output=README_generated.md
```

### Start Interactive Mode

```bash
python kiara_entry.py interactive
```

### Batch Processing

```bash
python kiara_interactive_enhanced.py
# Choose option 5 (Batch Generation)
# Add multiple repository URLs
# Select common template
# Kiara processes all automatically
```

### API Usage

```bash
# Start server
python kiara_entry.py serve

# Generate documentation via API
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://github.com/user/repo", "template": "modern"}'
```

## 📊 Generated Files

- **README_kiara_YYYYMMDD_HHMMSS.md**: Generated documentation with timestamp
- **cache/github_repos.db**: SQLite3 database with repository history
- **Logs**: Detailed process information and error tracking
- **API responses**: JSON format for programmatic access

## 🧪 Testing Installation

```bash
# Basic configuration test
python -c "from src.config import get_settings; print('✅ Configuration OK')"

# Test xAI Grok (if configured)
python -c "from src.xai_analyzer import XAIAnalyzer; print('✅ xAI Grok OK')"

# Test Oracle Cloud AI (if configured)
python -c "from src.oci_analyzer import OCIAnalyzer; print('✅ OCI OK')"

# Test GitHub integration
python -c "from src.auth_manager import AuthManager; print('✅ GitHub OK')"

# Test commit tracking
python -c "from src.simple_commit_tracker import SimpleCommitTracker; print('✅ Commit Tracker OK')"

# Run all tests
python kiara_entry.py info
```

## 🔧 Troubleshooting

### xAI Grok API Issues
```
Error: xAI API authentication failed
```
**Solution**: Verify your xAI API key is correct and has sufficient quota

### GitHub Token Issues
```
Error: Authentication validation failed
```
**Solution**: Ensure token has correct scopes (repo, read:user, user:email)

### Oracle Cloud Issues
```
Error: OCI configuration not found
```
**Solution**: Run `oci setup config` and configure credentials

### Dependency Issues
```
ModuleNotFoundError: No module named 'xyz'
```
**Solution**: Run `uv sync` or `pip install -r requirements.txt`

### Windows Unicode Issues
```
UnicodeDecodeError or broken emojis
```
**Solution**: Kiara auto-detects Windows and handles emoji conversion

### MCP Server Issues
```
Error: GitHub MCP Server not found
```
**Solution**: Install with `npm install -g @github/github-mcp-server`

## 📈 Performance

### Generation Times
- **Small repositories**: 15-30 seconds
- **Medium repositories**: 30-60 seconds
- **Large repositories**: 60-120 seconds
- **Cache hit**: 2-5 seconds
- **Batch processing**: Automatic with rate limiting

### AI Provider Comparison
- **xAI Grok 4**: Highest quality, 131K tokens, best for complex projects
- **Oracle Cloud**: Balanced performance, good vision capabilities
- **Groq**: Fastest inference, good for quick iterations

### Optimization Tips
- Enable Redis caching for better performance
- Use appropriate templates for your project size
- Enable smart commit detection to avoid unnecessary regenerations

## 🚦 Project Status

### ✅ Stable Features
- GitHub repository documentation generation
- Local project analysis
- Smart commit detection system
- Adaptive templates
- Multi-AI provider support
- Vision analysis
- Caching systems
- API server

### 🚧 In Development
- Complete Azure DevOps integration
- Custom plugin system
- Advanced analytics dashboard
- Team collaboration features

### 🎯 Planned Features
- GitLab support
- Bitbucket integration
- Custom template editor
- Webhook integrations
- Documentation versioning

## 📚 API Reference

### REST Endpoints

#### POST /generate
Generate documentation for a repository

**Request:**
```json
{
  "url": "https://github.com/user/repo",
  "template": "modern",
  "force_regenerate": false,
  "enable_vision": true
}
```

**Response:**
```json
{
  "success": true,
  "documentation": "# Generated README content...",
  "generation_time": 45.2,
  "cache_hit": false,
  "analysis": {
    "language": "Python",
    "framework": ["FastAPI", "Pydantic"],
    "project_type": "Web API"
  }
}
```

#### GET /health
Check API health status

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "ai_providers": {
    "xai_grok": "enabled",
    "oracle_cloud": "enabled",
    "groq": "available"
  }
}
```

## 🏗️ Project Structure

```
Kiara/
├── kiara_entry.py                  # Main entry point CLI
├── kiara_cli.py                    # Simple CLI interface
├── kiara_interactive_enhanced.py   # Enhanced interactive mode
├── run_kiara.py                    # Legacy launcher
├── pyproject.toml                  # Python project configuration
├── requirements.txt                # Dependencies
├── .env.example                    # Configuration template
├── README.md                       # This file
├── SETUP.md                        # Quick setup guide
├── src/
│   ├── __init__.py                 # Package initialization
│   ├── config.py                   # Centralized configuration
│   ├── main.py                     # Main orchestrator
│   ├── xai_analyzer.py            # xAI Grok integration
│   ├── oci_analyzer.py            # Oracle Cloud AI integration
│   ├── groq_analyzer.py           # Groq AI integration
│   ├── doc_generator.py           # Documentation generator
│   ├── template_engine.py         # Template system
│   ├── language_detector.py       # Language detection
│   ├── vision_processor.py        # Image analysis
│   ├── simple_commit_tracker.py   # Commit detection
│   ├── smart_commit_detector.py   # Advanced commit tracking
│   ├── auth_manager.py            # Authentication management
│   ├── mcp_client.py              # MCP protocol client
│   ├── cache_manager.py           # Caching system
│   ├── local_analyzer.py          # Local project analysis
│   ├── azure_devops_client.py     # Azure DevOps integration
│   ├── readme_templates.py        # Template definitions
│   ├── models.py                  # Data models
│   └── validators.py              # Input validation
├── api/
│   ├── __init__.py                 # API package
│   ├── main.py                     # FastAPI application
│   └── test_api.py                 # API tests
├── cache/
│   └── github_repos.db            # SQLite cache database
└── tests/
    └── test_*.py                   # Unit tests
```

## 🤝 Contributing

Kiara is designed to be a comprehensive and professional documentation agent. Contributions and improvements are welcome!

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd Kiara

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
black src/
isort src/
ruff check src/
mypy src/
```

### Contribution Guidelines

1. **Code Quality**: Follow PEP 8 and use provided linting tools
2. **Testing**: Add tests for new features
3. **Documentation**: Update README and docstrings
4. **AI Providers**: Maintain compatibility across all providers
5. **Backward Compatibility**: Ensure existing features continue working

## 📄 License

This project is open source and available under the MIT License. See the LICENSE file for more details.

## 🏷️ Version Information

**Kiara v0.1.0** - AI Documentation Agent Enhanced

### Core Technologies
- **Primary AI**: xAI Grok 4 (131K tokens output)
- **Backup AI**: Oracle Cloud AI (Meta Llama + Scout Vision)  
- **Fallback AI**: Groq (Llama 3.1 + Vision)
- **Framework**: FastAPI + Click + Rich
- **Database**: SQLite3 + Redis
- **Protocol**: Model Context Protocol (MCP)

### Key Capabilities
- Multi-provider AI architecture with automatic failover
- Intelligent commit detection and change tracking
- Professional adaptive templates
- Comprehensive GitHub + Local project support
- Advanced vision analysis and image processing
- Enterprise-ready caching and performance optimization

---

**🤖 Built with artificial intelligence to create intelligent documentation**

*"Making documentation magical since 2025"* ✨