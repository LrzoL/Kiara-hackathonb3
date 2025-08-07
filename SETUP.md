# 🚀 Kiara - Setup Rápido

## ⚡ Instalação em 3 Passos

### 1️⃣ **Instalar Dependências**
```bash
# Opção A: Usar uv (recomendado - mais rápido)
pip install uv
uv sync

# Opção B: Usar pip tradicional
pip install -r requirements.txt

# Instalar GitHub MCP Server
npm install -g @github/github-mcp-server
```

### 2️⃣ **Configurar Variáveis de Ambiente**
```bash
# Copie o template
cp .env.example .env

# Edite o arquivo .env com suas credenciais
# Mínimo necessário:
GITHUB_TOKEN=ghp_your_github_token_here
OCI_COMPARTMENT_ID=ocid1.tenancy.oc1..your_compartment_id_here
```

### 3️⃣ **Testar e Executar**
```bash
# Teste a instalação
python run_kiara.py --test

# Execute Kiara (interface interativa)
python run_kiara.py

# Ou use CLI simples
python run_kiara.py --simple
```

## 🎯 Uso Rápido

### **Gerar README para repositório GitHub:**
1. Execute: `python run_kiara.py`
2. Escolha opção `1` (GitHub Repository)
3. Digite a URL: `https://github.com/usuario/repo`
4. Selecione template ou deixe automático
5. ✨ README.md será gerado automaticamente!

### **Detectar commits novos:**
- Sistema detecta automaticamente se há novos commits
- Para novos commits: mostra *"Vejo que você atualizou seu projeto, vamos atualizar o readme também?"*
- Para mesmo commit: pergunta se deseja gerar novamente

## ⚙️ Configuração Oracle Cloud

1. **Configure OCI CLI:**
```bash
oci setup config
```

2. **No arquivo .env:**
```bash
OCI_COMPARTMENT_ID=ocid1.tenancy.oc1..sua_id_aqui
OCI_MODEL_ID=meta.llama-4-maverick-17b-128e-instruct
OCI_MODEL_VISION=Scout
OCI_ENABLED=true
```

## 🔑 Token GitHub

1. Vá em: https://github.com/settings/tokens
2. Clique "Generate new token (classic)"
3. Selecione scopes: `repo`, `read:user`, `user:email`
4. Copie o token para `.env`

## ✅ Verificação de Funcionamento

```bash
# Teste completo
python run_kiara.py --test

# Deve mostrar:
# ✅ Oracle Cloud AI: Enabled
# ✅ GitHub Token: Configured
# ✅ Commit tracker initialized!
```

## 🐛 Problemas Comuns

### **Erro: "OCI configuration not found"**
```bash
# Solução:
oci setup config
# Configure suas credenciais OCI
```

### **Erro: "GitHub token invalid"**
```bash
# Solução:
# 1. Verifique se o token no .env está correto
# 2. Verifique se tem os scopes necessários (repo, read:user, user:email)
```

### **Erro: "Module not found"**
```bash
# Solução:
pip install -r requirements.txt
# ou
uv sync
```

## 📁 Estrutura Final

```
Kiara/
├── run_kiara.py           # Launcher principal ⭐
├── kiara_cli.py          # CLI simples
├── kiara_interactive_enhanced.py  # Interface completa
├── .env                  # Suas configurações
├── README.md             # Documentação completa
├── SETUP.md              # Este arquivo
└── src/                  # Código fonte
```

---

**🎉 Pronto! Kiara está configurado e funcionando!**

Execute `python run_kiara.py` e comece a gerar documentação automaticamente!