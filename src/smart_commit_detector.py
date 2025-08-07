"""Smart commit detector integrated with Kiara Oracle Cloud AI system."""

import asyncio
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import httpx

from .config import get_settings

logger = logging.getLogger(__name__)


class SmartCommitDetector:
    """Smart commit detector with Oracle Cloud AI integration."""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.db_path = Path("cache/github_repos.db")
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for commit tracking."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS repositories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE NOT NULL,
                    name TEXT,
                    owner TEXT,
                    last_commit_sha TEXT,
                    last_processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_readme_generated_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS commit_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repo_id INTEGER,
                    commit_sha TEXT NOT NULL,
                    commit_message TEXT,
                    commit_author TEXT,
                    commit_date TIMESTAMP,
                    readme_generated BOOLEAN DEFAULT FALSE,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (repo_id) REFERENCES repositories (id),
                    UNIQUE(repo_id, commit_sha)
                )
            """)
            
            conn.commit()
    
    def _parse_github_url(self, repo_url: str) -> Tuple[str, str]:
        """Parse GitHub URL to extract owner and repo name."""
        # Remove protocol and domain
        path = repo_url.replace('https://github.com/', '').replace('http://github.com/', '')
        # Remove .git suffix if present
        if path.endswith('.git'):
            path = path[:-4]
        
        parts = path.split('/')
        if len(parts) >= 2:
            return parts[0], parts[1]
        else:
            raise ValueError(f"Invalid GitHub URL format: {repo_url}")
    
    async def get_latest_commit_from_github(self, repo_url: str) -> Optional[Dict[str, Any]]:
        """Get latest commit information from GitHub API."""
        try:
            owner, repo = self._parse_github_url(repo_url)
            
            headers = {
                "Authorization": f"token {self.settings.github_token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "kiara-doc-agent/1.0.0"
            }
            
            async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
                response = await client.get(f"https://api.github.com/repos/{owner}/{repo}/commits/HEAD")
                response.raise_for_status()
                
                commit_data = response.json()
                return {
                    'sha': commit_data['sha'],
                    'message': commit_data['commit']['message'],
                    'author': commit_data['commit']['author']['name'],
                    'date': commit_data['commit']['author']['date'],
                    'url': commit_data['html_url']
                }
        
        except Exception as e:
            logger.error(f"Failed to get latest commit for {repo_url}: {e}")
            return None
    
    def get_last_processed_commit(self, repo_url: str) -> Optional[str]:
        """Get the last commit SHA that was processed for README generation."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT last_commit_sha 
                FROM repositories 
                WHERE url = ? AND last_readme_generated_at IS NOT NULL
            """, (repo_url,))
            
            row = cursor.fetchone()
            return row[0] if row else None
    
    def store_repository_commit(self, repo_url: str, commit_info: Dict[str, Any], readme_generated: bool = False):
        """Store repository and commit information."""
        owner, repo_name = self._parse_github_url(repo_url)
        
        with sqlite3.connect(self.db_path) as conn:
            # Insert or update repository
            conn.execute("""
                INSERT OR REPLACE INTO repositories (url, name, owner, last_commit_sha, last_processed_at, last_readme_generated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
            """, (repo_url, repo_name, owner, commit_info['sha'], 
                  datetime.utcnow().isoformat() if readme_generated else None))
            
            # Get repository ID
            cursor = conn.execute("SELECT id FROM repositories WHERE url = ?", (repo_url,))
            repo_id = cursor.fetchone()[0]
            
            # Store commit history
            commit_date = datetime.fromisoformat(commit_info['date'].replace('Z', '+00:00'))
            
            conn.execute("""
                INSERT OR REPLACE INTO commit_history 
                (repo_id, commit_sha, commit_message, commit_author, commit_date, readme_generated)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (repo_id, commit_info['sha'], commit_info['message'], 
                  commit_info['author'], commit_date, readme_generated))
            
            conn.commit()
    
    async def check_for_updates(self, repo_url: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Check if repository has updates that require README regeneration.
        
        Returns:
            - bool: True if README should be updated
            - Dict: Latest commit info (if available)
            - str: Update message for user
        """
        logger.info(f"Checking for updates in: {repo_url}")
        
        # Get latest commit from GitHub
        latest_commit = await self.get_latest_commit_from_github(repo_url)
        if not latest_commit:
            return False, None, "Não foi possível acessar o repositório."
        
        # Get last processed commit from database
        last_processed = self.get_last_processed_commit(repo_url)
        
        # Store current commit info (not marked as README generated yet)
        self.store_repository_commit(repo_url, latest_commit, readme_generated=False)
        
        current_sha = latest_commit['sha']
        
        if not last_processed:
            # First time processing this repository
            message = f"🎯 Vejo que é a primeira vez analisando este projeto! Vamos gerar um README.md completo usando Oracle Cloud AI?"
            return True, latest_commit, message
        
        if last_processed == current_sha:
            # No new commits
            message = f"✅ Seu projeto está atualizado! O README já foi gerado para o último commit ({current_sha[:8]})."
            return False, latest_commit, message
        
        # New commits detected
        commit_message_short = latest_commit['message'].split('\n')[0][:50]
        if len(latest_commit['message']) > 50:
            commit_message_short += "..."
        
        message = f"""🔄 Vejo que você atualizou seu projeto! 

📍 **Novo commit detectado:**
   • Commit: {current_sha[:8]}
   • Mensagem: "{commit_message_short}"
   • Autor: {latest_commit['author']}

🤖 Vamos atualizar o README.md com as últimas mudanças usando Oracle Cloud AI (meta.llama-4-maverick-17b-128e-instruct)?"""
        
        return True, latest_commit, message
    
    def mark_readme_generated(self, repo_url: str, commit_sha: str):
        """Mark that README was generated for a specific commit."""
        with sqlite3.connect(self.db_path) as conn:
            # Update repository last generated timestamp
            conn.execute("""
                UPDATE repositories 
                SET last_readme_generated_at = CURRENT_TIMESTAMP,
                    last_commit_sha = ?
                WHERE url = ?
            """, (commit_sha, repo_url))
            
            # Update commit history
            cursor = conn.execute("SELECT id FROM repositories WHERE url = ?", (repo_url,))
            repo_id = cursor.fetchone()[0]
            
            conn.execute("""
                UPDATE commit_history 
                SET readme_generated = TRUE, processed_at = CURRENT_TIMESTAMP
                WHERE repo_id = ? AND commit_sha = ?
            """, (repo_id, commit_sha))
            
            conn.commit()
            logger.info(f"Marked README as generated for commit {commit_sha[:8]}")
    
    def get_repository_stats(self, repo_url: str) -> Dict[str, Any]:
        """Get repository statistics from database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get repository info
            cursor = conn.execute("""
                SELECT r.*, 
                       COUNT(ch.id) as total_commits,
                       COUNT(CASE WHEN ch.readme_generated = TRUE THEN 1 END) as readme_updates
                FROM repositories r
                LEFT JOIN commit_history ch ON r.id = ch.repo_id
                WHERE r.url = ?
                GROUP BY r.id
            """, (repo_url,))
            
            repo_stats = cursor.fetchone()
            
            if repo_stats:
                return {
                    'repository_name': repo_stats['name'],
                    'owner': repo_stats['owner'], 
                    'last_commit': repo_stats['last_commit_sha'],
                    'last_processed': repo_stats['last_processed_at'],
                    'last_readme_generated': repo_stats['last_readme_generated_at'],
                    'total_commits_tracked': repo_stats['total_commits'],
                    'readme_updates': repo_stats['readme_updates']
                }
            else:
                return {
                    'repository_name': 'Unknown',
                    'owner': 'Unknown',
                    'total_commits_tracked': 0,
                    'readme_updates': 0
                }


async def create_smart_detector(settings=None) -> SmartCommitDetector:
    """Factory function to create smart commit detector."""
    return SmartCommitDetector(settings)