# 🤖 Kiara - AI Documentation Agent

**Agente inteligente de documentação automática powered by Oracle Cloud AI**

Kiara é um agente de IA especializado em gerar documentação técnica profissional para qualquer projeto automaticamente. Utilizando Oracle Cloud Infrastructure (OCI) AI com modelos avançados, Kiara analisa repositórios GitHub, projetos locais e gera READMEs completos e adaptativos.

## ✨ Principais Funcionalidades

- 🔍 **Análise Inteligente**: Detecção automática de linguagens, frameworks e arquiteturas
- 📝 **Documentação Profissional**: 4 templates adaptativos (Minimal, Emoji Rich, Modern, Technical Deep)
- 🚀 **Oracle Cloud AI**: Powered by `meta.llama-4-maverick-17b-128e-instruct` + `Scout` (visão)
- 📊 **Suporte Universal**: GitHub, Azure DevOps, e projetos locais
- 🔄 **Detecção de Commits**: Sistema inteligente que detecta atualizações nos repositórios
- 💾 **Cache Inteligente**: SQLite3 para otimização e histórico
- 🎯 **Interface Dupla**: CLI simples e interface interativa aprimorada

## 🚀 Instalação

### Pré-requisitos

- **Python 3.11+** (obrigatório)
- **Node.js 18+** (para GitHub MCP Server)
- **Token GitHub** (para repositórios privados)
- **Oracle Cloud AI** configurado

### Instalação Rápida

```bash
# Clone ou descompacte o projeto Kiara
cd Kiara

# Instale dependências (usando uv - recomendado)
pip install uv
uv sync

# Ou usando pip tradicional
pip install -r requirements.txt

# Instalar GitHub MCP Server
npm install -g @github/github-mcp-server
```

### Configuração

1. **Configure o arquivo de ambiente:**
```bash
cp .env.example .env
nano .env  # ou seu editor preferido
```

2. **Variáveis essenciais no .env:**
```bash
# GitHub (obrigatório)
GITHUB_TOKEN=ghp_your_github_token_here

# Oracle Cloud AI (obrigatório)
OCI_COMPARTMENT_ID=ocid1.tenancy.oc1..your_compartment_id_here
OCI_MODEL_ID=meta.llama-4-maverick-17b-128e-instruct
OCI_MODEL_VISION=Scout
OCI_ENABLED=true
```

3. **Configure Oracle Cloud CLI:**
```bash
oci setup config
# Siga as instruções para configurar credenciais OCI
```

## 🎯 Como Usar

### Opção 1: CLI Simples (Recomendado para início)

```bash
python kiara_cli.py
```

**Funcionalidades:**
- Geração de documentação para GitHub
- Análise de projetos locais  
- Seleção de templates
- Detecção automática de commits
- Interface limpa e objetiva

### Opção 2: Interface Interativa Aprimorada

```bash
python kiara_interactive_enhanced.py
```

**Funcionalidades extras:**
- Geração em lote (múltiplos repositórios)
- Análise rápida de repositórios
- Servidor API integrado
- Configurações avançadas
- Interface rica com mais opções

### Opção 3: Modo API/Servidor

```bash
python -m src.main serve --host 0.0.0.0 --port 8000
```

**Endpoints disponíveis:**
- `POST /generate` - Gerar documentação
- `GET /health` - Status da API
- `GET /templates` - Listar templates

## 🛠️ Sistema de Detecção de Commits

Kiara possui um sistema inteligente que monitora atualizações em repositórios GitHub:

### 🔄 **Como Funciona**

1. **Primeira análise**: Sistema registra o commit atual silenciosamente
2. **Commits iguais**: Pergunta se deseja gerar novamente  
3. **Novos commits**: Mostra: *"Vejo que você atualizou seu projeto, vamos atualizar o readme também?"*

### 📊 **Banco de Dados**

- **SQLite3**: `cache/github_repos.db`
- **Tabela**: `repositories` (url, owner, name, last_commit_sha, last_checked_at)
- **Escopo**: Apenas repositórios GitHub

## 📋 Templates Disponíveis

| Template | Estilo | Melhor Para | Características |
|----------|--------|-------------|----------------|
| **MINIMAL** | Clean & Professional | Documentação Empresarial | Sem emojis, tom formal, abrangente |
| **EMOJI_RICH** | Fun & Engaging | Projetos da Comunidade | 50+ emojis, tom entusiasmado, apelo visual |
| **MODERN** | GitHub-style Contemporary | Projetos Open Source | Badges, tabelas, TOC, markdown moderno |
| **TECHNICAL_DEEP** | Enterprise Technical | Sistemas de Produção | Arquitetura, deployment, compliance |

## ⚙️ Configuração Avançada

### Oracle Cloud AI

```bash
# Modelos disponíveis
OCI_MODEL_ID=meta.llama-4-maverick-17b-128e-instruct    # Texto
OCI_MODEL_VISION=Scout                                   # Visão
OCI_MAX_TOKENS=4000                                     # Limite de tokens
OCI_TEMPERATURE=0.2                                     # Criatividade (0.0-1.0)
```

### GitHub Integration

```bash
# Scopes necessários no token GitHub
# - repo (acesso total a repositórios)
# - read:user (informações do usuário)  
# - user:email (emails do usuário)
GITHUB_TOKEN=ghp_your_token_with_repo_scope
```

### Análise de Visão

```bash
# Processamento de imagens
VISION_ANALYSIS_ENABLED=true
VISION_MAX_IMAGE_SIZE=5242880           # 5MB
VISION_SUPPORTED_FORMATS=png,jpg,jpeg,gif,bmp,webp,svg
```

## 🔍 Exemplos de Uso

### Repositório GitHub

```bash
python kiara_cli.py

# Escolha opção 1 (GitHub)
# Digite: https://github.com/usuario/projeto
# Selecione template ou deixe no automático
# Aguarde a mágica acontecer ✨
```

### Projeto Local

```bash
python kiara_cli.py

# Escolha opção 3 (Local Project)
# Digite o caminho: /path/to/your/project (ou . para atual)
# Selecione template
# README será gerado automaticamente
```

### Geração em Lote

```bash
python kiara_interactive_enhanced.py

# Escolha opção 5 (Batch Generation)
# Adicione múltiplas URLs de repositórios
# Selecione template comum
# Kiara processará todos automaticamente
```

## 📊 Arquivos Gerados

- **README_kiara_YYYYMMDD_HHMMSS.md**: Documentação gerada com timestamp
- **cache/github_repos.db**: Banco de dados SQLite3 com histórico
- **Logs**: Informações detalhadas do processo de geração

## 🧪 Testando a Instalação

```bash
# Teste básico
python -c "from src.config import get_settings; print('✅ Configuração OK')"

# Teste Oracle Cloud AI
python -c "from src.oci_analyzer import OCIAnalyzer; print('✅ OCI OK')"

# Teste GitHub
python -c "from src.auth_manager import AuthManager; print('✅ GitHub OK')"

# Teste detecção de commits
python -c "from src.simple_commit_tracker import SimpleCommitTracker; print('✅ Commit Tracker OK')"
```

## 🔧 Solução de Problemas

### Erro de Token GitHub
```
Error: Authentication validation failed
```
**Solução**: Verifique se o token no `.env` tem os scopes corretos (repo, read:user, user:email)

### Erro Oracle Cloud
```
Error: OCI configuration not found
```
**Solução**: Execute `oci setup config` e configure suas credenciais

### Erro de Dependências
```
ModuleNotFoundError: No module named 'xyz'
```
**Solução**: Execute `uv sync` ou `pip install -r requirements.txt`

### Problemas no Windows
```
UnicodeDecodeError ou emojis quebrados
```
**Solução**: Kiara já está otimizada para Windows - emojis são convertidos automaticamente

## 📈 Performance

- **Repositórios pequenos**: ~15-30 segundos
- **Repositórios médios**: ~30-60 segundos  
- **Repositórios grandes**: ~60-120 segundos
- **Cache hit**: ~2-5 segundos
- **Batch processing**: Automático com rate limiting

## 🚦 Status do Projeto

- ✅ **Estável**: Geração de documentação GitHub
- ✅ **Estável**: Projetos locais  
- ✅ **Estável**: Sistema de detecção de commits
- ✅ **Estável**: Templates adaptativos
- ✅ **Estável**: Oracle Cloud AI integration
- 🚧 **Em desenvolvimento**: Azure DevOps completo
- 🚧 **Planejado**: Plugins personalizados

## 🤝 Contribuição

Kiara foi desenvolvida para ser um agente de documentação completo e profissional. Sugestões e melhorias são bem-vindas!

### Estrutura do Projeto

```
Kiara/
├── kiara_cli.py                    # CLI principal
├── kiara_interactive_enhanced.py   # Interface interativa
├── pyproject.toml                  # Configuração Python
├── .env.example                    # Template de configuração
├── src/
│   ├── config.py                   # Configurações centralizadas
│   ├── oci_analyzer.py            # Oracle Cloud AI integration
│   ├── doc_generator.py           # Orquestrador principal
│   ├── simple_commit_tracker.py   # Detecção de commits
│   ├── auth_manager.py            # Autenticação GitHub/Azure
│   ├── mcp_client.py              # GitHub MCP Server client
│   ├── readme_templates.py        # Templates de documentação
│   └── ...                        # Outros módulos
├── cache/
│   └── github_repos.db            # Cache SQLite3
└── api/
    └── main.py                     # FastAPI server
```

## 📄 Licença

Este projeto é open source e está disponível sob licença MIT.

## 🏷️ Versão

**Kiara v1.0** - AI Documentation Agent Enhanced
- Powered by Oracle Cloud AI (meta.llama-4-maverick-17b-128e-instruct + Scout)
- Detecção inteligente de commits
- Templates adaptativos profissionais
- Suporte completo GitHub + Local Projects

---

**🤖 Desenvolvido com inteligência artificial para gerar documentação inteligente**

*"Making documentation magical since 2025"*