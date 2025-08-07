"""Simple commit tracker using SQLite3 - just repository and commit comparison."""

import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import httpx

from .config import get_settings

logger = logging.getLogger(__name__)


class SimpleCommitTracker:
    """Simple SQLite3 commit tracker for repositories."""
    
    def __init__(self):
        self.settings = get_settings()
        self.db_path = Path("cache/github_repos.db")
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize simple SQLite database."""
        # Database already exists with a different structure
        # We'll use the existing structure and just update last_checked_at
        pass
    
    async def get_current_commit(self, repo_url: str) -> Optional[str]:
        """Get current commit SHA from GitHub."""
        try:
            # Parse URL to get owner/repo
            parts = repo_url.replace('https://github.com/', '').split('/')
            if len(parts) < 2:
                return None
            
            owner, repo = parts[0], parts[1]
            
            # Call GitHub API
            headers = {
                "Authorization": f"token {self.settings.github_token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "kiara-doc-agent/1.0.0"
            }
            
            async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
                response = await client.get(f"https://api.github.com/repos/{owner}/{repo}/commits/HEAD")
                response.raise_for_status()
                
                commit_data = response.json()
                return commit_data['sha']
                
        except Exception as e:
            logger.error(f"Failed to get current commit for {repo_url}: {e}")
            return None
    
    def get_stored_commit(self, repo_url: str) -> Optional[str]:
        """Get stored commit SHA from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT last_commit_sha FROM repositories WHERE url = ?", 
                (repo_url,)
            )
            row = cursor.fetchone()
            return row[0] if row else None
    
    def store_commit(self, repo_url: str, commit_sha: str):
        """Store or update commit SHA in database."""
        # Parse URL to get owner and repo name
        parts = repo_url.replace('https://github.com/', '').split('/')
        owner = parts[0] if len(parts) >= 1 else 'unknown'
        repo_name = parts[1] if len(parts) >= 2 else 'unknown'
        
        with sqlite3.connect(self.db_path) as conn:
            # Try to update existing record first
            cursor = conn.execute(
                "UPDATE repositories SET last_commit_sha = ?, last_checked_at = CURRENT_TIMESTAMP WHERE url = ?",
                (commit_sha, repo_url)
            )
            
            # If no rows were updated, insert new record
            if cursor.rowcount == 0:
                conn.execute("""
                    INSERT INTO repositories (url, owner, name, last_commit_sha, last_checked_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (repo_url, owner, repo_name, commit_sha))
            
            conn.commit()
    
    async def check_for_new_commit(self, repo_url: str) -> Tuple[bool, str]:
        """
        Check if repository has new commits.
        
        Returns:
            - bool: True if new commit detected
            - str: Message to show user
        """
        # Get current commit from GitHub
        current_commit = await self.get_current_commit(repo_url)
        if not current_commit:
            return False, "Não foi possível acessar o repositório."
        
        # Get stored commit from database
        stored_commit = self.get_stored_commit(repo_url)
        
        if not stored_commit:
            # First time - store commit and don't show update message
            self.store_commit(repo_url, current_commit)
            return False, "Primeira análise do repositório."
        
        if stored_commit == current_commit:
            # Same commit - ask for confirmation
            return False, "O README já foi gerado para este commit. Deseja gerar novamente?"
        
        # New commit detected!
        return True, "Vejo que você atualizou seu projeto, vamos atualizar o readme também?"
    
    def mark_readme_updated(self, repo_url: str, commit_sha: str):
        """Mark that README was updated for this commit."""
        self.store_commit(repo_url, commit_sha)