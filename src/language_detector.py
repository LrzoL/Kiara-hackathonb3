"""Universal language detection for repositories."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


class LanguagePattern(BaseModel):
    """Language detection pattern."""
    
    extensions: List[str] = []
    filenames: List[str] = []
    content_patterns: List[str] = []
    package_files: List[str] = []
    config_files: List[str] = []


class FrameworkPattern(BaseModel):
    """Framework detection pattern."""
    
    name: str
    indicators: List[str] = []
    dependencies: List[str] = []
    file_patterns: List[str] = []
    content_patterns: List[str] = []


class DetectionResult(BaseModel):
    """Language detection result."""
    
    language: str
    confidence: float
    frameworks: List[str] = []
    package_managers: List[str] = []
    build_tools: List[str] = []
    testing_frameworks: List[str] = []
    project_type: str = "unknown"
    evidence: Dict[str, Any] = {}


class LanguageDetector:
    """Universal language and framework detector."""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._language_patterns = self._load_language_patterns()
        self._framework_patterns = self._load_framework_patterns()
    
    def _load_language_patterns(self) -> Dict[str, LanguagePattern]:
        """Load language detection patterns."""
        return {
            "Python": LanguagePattern(
                extensions=[".py", ".pyw", ".pyx", ".pyi"],
                filenames=["__init__.py", "__main__.py"],
                content_patterns=["#!/usr/bin/env python", "# -*- coding:", "import ", "from ", "def ", "class "],
                package_files=["requirements.txt", "setup.py", "pyproject.toml", "Pipfile", "poetry.lock"],
                config_files=["tox.ini", "pytest.ini", ".pylintrc", "mypy.ini"]
            ),
            "JavaScript": LanguagePattern(
                extensions=[".js", ".mjs", ".cjs"],
                filenames=["index.js", "main.js", "app.js"],
                content_patterns=["function ", "const ", "let ", "var ", "require(", "import ", "export "],
                package_files=["package.json", "package-lock.json", "yarn.lock"],
                config_files=[".eslintrc", ".babelrc", "webpack.config.js", "jest.config.js"]
            ),
            "TypeScript": LanguagePattern(
                extensions=[".ts", ".tsx", ".d.ts"],
                filenames=["index.ts", "main.ts", "app.ts"],
                content_patterns=["interface ", "type ", "enum ", ": string", ": number", "implements "],
                package_files=["package.json", "tsconfig.json"],
                config_files=["tsconfig.json", "tslint.json", ".eslintrc"]
            ),
            "Java": LanguagePattern(
                extensions=[".java", ".class", ".jar"],
                filenames=["Main.java", "Application.java"],
                content_patterns=["public class ", "import java.", "package ", "public static void main"],
                package_files=["pom.xml", "build.gradle", "gradle.properties"],
                config_files=["application.properties", "application.yml"]
            ),
            "Go": LanguagePattern(
                extensions=[".go"],
                filenames=["main.go"],
                content_patterns=["package main", "func main()", "import (", "func ", "type ", "var "],
                package_files=["go.mod", "go.sum"],
                config_files=["go.work", ".golangci.yml"]
            ),
            "Rust": LanguagePattern(
                extensions=[".rs"],
                filenames=["main.rs", "lib.rs", "mod.rs"],
                content_patterns=["fn main()", "use ", "mod ", "pub fn", "impl ", "struct "],
                package_files=["Cargo.toml", "Cargo.lock"],
                config_files=["rust-toolchain", ".rustfmt.toml"]
            ),
            "C++": LanguagePattern(
                extensions=[".cpp", ".cxx", ".cc", ".c++", ".hpp", ".hxx", ".h++"],
                filenames=["main.cpp", "main.cxx"],
                content_patterns=["#include <", "using namespace", "class ", "template<", "std::"],
                package_files=["CMakeLists.txt", "Makefile"],
                config_files=["CMakeCache.txt", "conanfile.txt"]
            ),
            "C": LanguagePattern(
                extensions=[".c", ".h"],
                filenames=["main.c"],
                content_patterns=["#include <", "int main(", "printf(", "malloc(", "free("],
                package_files=["Makefile", "CMakeLists.txt"],
                config_files=["configure.ac", "Makefile.in"]
            ),
            "C#": LanguagePattern(
                extensions=[".cs", ".csx"],
                filenames=["Program.cs", "Startup.cs"],
                content_patterns=["using System", "namespace ", "public class", "static void Main"],
                package_files=["*.csproj", "*.sln", "packages.config"],
                config_files=["appsettings.json", "web.config"]
            ),
            "PHP": LanguagePattern(
                extensions=[".php", ".phtml", ".php3", ".php4", ".php5"],
                filenames=["index.php", "composer.json"],
                content_patterns=["<?php", "function ", "class ", "$", "->"],
                package_files=["composer.json", "composer.lock"],
                config_files=[".htaccess", "php.ini"]
            ),
            "Ruby": LanguagePattern(
                extensions=[".rb", ".rbw"],
                filenames=["Rakefile", "Gemfile"],
                content_patterns=["def ", "class ", "module ", "require ", "end"],
                package_files=["Gemfile", "Gemfile.lock", "*.gemspec"],
                config_files=[".rubocop.yml", "config.ru"]
            ),
            "Swift": LanguagePattern(
                extensions=[".swift"],
                filenames=["main.swift", "AppDelegate.swift"],
                content_patterns=["import ", "func ", "class ", "struct ", "enum ", "var ", "let "],
                package_files=["Package.swift", "*.xcodeproj"],
                config_files=["Info.plist", "*.entitlements"]
            ),
            "Kotlin": LanguagePattern(
                extensions=[".kt", ".kts"],
                filenames=["MainActivity.kt"],
                content_patterns=["fun main", "class ", "data class", "import ", "package "],
                package_files=["build.gradle.kts", "pom.xml"],
                config_files=["gradle.properties", "local.properties"]
            ),
            "Dart": LanguagePattern(
                extensions=[".dart"],
                filenames=["main.dart", "pubspec.yaml"],
                content_patterns=["void main()", "import 'dart:", "class ", "Widget "],
                package_files=["pubspec.yaml", "pubspec.lock"],
                config_files=["analysis_options.yaml"]
            ),
            "Shell": LanguagePattern(
                extensions=[".sh", ".bash", ".zsh", ".fish"],
                filenames=["install.sh", "build.sh", "deploy.sh"],
                content_patterns=["#!/bin/bash", "#!/bin/sh", "echo ", "if [", "function "],
                package_files=[],
                config_files=[".bashrc", ".zshrc"]
            ),
            "PowerShell": LanguagePattern(
                extensions=[".ps1", ".psm1", ".psd1"],
                filenames=[],
                content_patterns=["Write-Host", "Get-", "Set-", "$", "function "],
                package_files=[],
                config_files=[]
            ),
            "HTML": LanguagePattern(
                extensions=[".html", ".htm", ".xhtml"],
                filenames=["index.html", "index.htm"],
                content_patterns=["<!DOCTYPE", "<html", "<head", "<body", "<div"],
                package_files=["package.json"],
                config_files=[]
            ),
            "CSS": LanguagePattern(
                extensions=[".css", ".scss", ".sass", ".less"],
                filenames=["style.css", "main.css"],
                content_patterns=["{", "}", ":", ";", "@media", ".class"],
                package_files=["package.json"],
                config_files=[]
            ),
            "SQL": LanguagePattern(
                extensions=[".sql", ".mysql", ".pgsql", ".sqlite"],
                filenames=["schema.sql", "migration.sql"],
                content_patterns=["SELECT", "CREATE TABLE", "INSERT INTO", "UPDATE", "DELETE"],
                package_files=[],
                config_files=[]
            ),
            "YAML": LanguagePattern(
                extensions=[".yml", ".yaml"],
                filenames=["docker-compose.yml", ".github/workflows/*.yml"],
                content_patterns=["---", "  ", "- ", ": "],
                package_files=[],
                config_files=[]
            ),
            "JSON": LanguagePattern(
                extensions=[".json"],
                filenames=["package.json", "composer.json", "tsconfig.json"],
                content_patterns=["{", "}", "[", "]", ":", ","],
                package_files=[],
                config_files=[]
            )
        }
    
    def _load_framework_patterns(self) -> List[FrameworkPattern]:
        """Load framework detection patterns."""
        return [
            # Python Frameworks
            FrameworkPattern(
                name="Django",
                indicators=["manage.py", "settings.py", "urls.py", "wsgi.py"],
                dependencies=["django"],
                file_patterns=["*/migrations/*.py", "*/templates/*.html"],
                content_patterns=["from django.db import models", "Django"]
            ),
            FrameworkPattern(
                name="Flask",
                indicators=["app.py", "application.py"],
                dependencies=["flask"],
                content_patterns=["from flask import", "Flask(__name__)"]
            ),
            FrameworkPattern(
                name="FastAPI",
                dependencies=["fastapi"],
                content_patterns=["from fastapi import", "FastAPI()"]
            ),
            FrameworkPattern(
                name="Pytest",
                indicators=["pytest.ini", "conftest.py"],
                dependencies=["pytest"],
                content_patterns=["import pytest", "def test_"]
            ),
            
            # JavaScript/TypeScript Frameworks
            FrameworkPattern(
                name="React",
                dependencies=["react", "react-dom"],
                file_patterns=["*.jsx", "*.tsx"],
                content_patterns=["import React", "useState", "useEffect", "JSX.Element"]
            ),
            FrameworkPattern(
                name="Vue.js",
                dependencies=["vue"],
                file_patterns=["*.vue"],
                content_patterns=["<template>", "<script>", "<style>", "Vue.component"]
            ),
            FrameworkPattern(
                name="Angular",
                dependencies=["@angular/core"],
                indicators=["angular.json"],
                content_patterns=["@Component", "@Injectable", "@NgModule"]
            ),
            FrameworkPattern(
                name="Express.js",
                dependencies=["express"],
                content_patterns=["require('express')", "app.listen", "app.get"]
            ),
            FrameworkPattern(
                name="Next.js",
                dependencies=["next"],
                indicators=["next.config.js"],
                content_patterns=["import { NextPage }", "getServerSideProps"]
            ),
            FrameworkPattern(
                name="Svelte",
                dependencies=["svelte"],
                file_patterns=["*.svelte"],
                content_patterns=["<script>", "{#if", "{#each"]
            ),
            
            # Build Tools
            FrameworkPattern(
                name="Webpack",
                indicators=["webpack.config.js"],
                dependencies=["webpack"],
                content_patterns=["module.exports", "entry:", "output:"]
            ),
            FrameworkPattern(
                name="Vite",
                indicators=["vite.config.js", "vite.config.ts"],
                dependencies=["vite"],
                content_patterns=["import { defineConfig }"]
            ),
            
            # Java Frameworks
            FrameworkPattern(
                name="Spring Boot",
                dependencies=["spring-boot-starter"],
                indicators=["application.properties", "application.yml"],
                content_patterns=["@SpringBootApplication", "@RestController", "@Service"]
            ),
            FrameworkPattern(
                name="Maven",
                indicators=["pom.xml"],
                content_patterns=["<project>", "<groupId>", "<artifactId>"]
            ),
            FrameworkPattern(
                name="Gradle",
                indicators=["build.gradle", "gradle.properties"],
                content_patterns=["apply plugin:", "dependencies {"]
            ),
            
            # Go Frameworks
            FrameworkPattern(
                name="Gin",
                dependencies=["github.com/gin-gonic/gin"],
                content_patterns=["gin.Default()", "gin.Engine"]
            ),
            FrameworkPattern(
                name="Echo",
                dependencies=["github.com/labstack/echo"],
                content_patterns=["echo.New()", "echo.Echo"]
            ),
            
            # Testing Frameworks
            FrameworkPattern(
                name="Jest",
                dependencies=["jest"],
                indicators=["jest.config.js"],
                content_patterns=["describe(", "it(", "test(", "expect("]
            ),
            FrameworkPattern(
                name="Mocha",
                dependencies=["mocha"],
                content_patterns=["describe(", "it(", "before(", "after("]
            ),
            
            # DevOps/Infrastructure
            FrameworkPattern(
                name="Docker",
                indicators=["Dockerfile", "docker-compose.yml"],
                content_patterns=["FROM ", "RUN ", "COPY ", "EXPOSE "]
            ),
            FrameworkPattern(
                name="Kubernetes",
                indicators=["*.yaml", "*.yml"],
                content_patterns=["apiVersion:", "kind:", "metadata:", "spec:"]
            )
        ]
    
    def detect_language(
        self,
        files: List[Dict[str, Any]],
        file_contents: Optional[Dict[str, str]] = None
    ) -> DetectionResult:
        """Detect primary language and frameworks."""
        
        if not files:
            return DetectionResult(language="Unknown", confidence=0.0)
        
        # Count file extensions
        extension_counts = {}
        filenames = set()
        
        for file_info in files:
            if file_info.get("type") == "file":
                path = file_info.get("path", "")
                filename = Path(path).name.lower()
                filenames.add(filename)
                
                # Count extensions
                if "." in filename:
                    ext = "." + filename.split(".")[-1]
                    extension_counts[ext] = extension_counts.get(ext, 0) + 1
        
        # Score languages based on extensions
        language_scores = {}
        for lang, pattern in self._language_patterns.items():
            score = 0
            
            # Extension matching
            for ext in pattern.extensions:
                if ext in extension_counts:
                    score += extension_counts[ext] * 10
            
            # Filename matching
            for filename in pattern.filenames:
                if filename.lower() in filenames:
                    score += 50
            
            # Package file matching
            for package_file in pattern.package_files:
                if "*" in package_file:
                    # Handle wildcards
                    pattern_base = package_file.replace("*", "")
                    if any(pattern_base in fn for fn in filenames):
                        score += 30
                elif package_file.lower() in filenames:
                    score += 30
            
            # Config file matching
            for config_file in pattern.config_files:
                if config_file.lower() in filenames:
                    score += 10
            
            language_scores[lang] = score
        
        # Content analysis if available
        if file_contents:
            content_scores = self._analyze_content(file_contents)
            for lang, content_score in content_scores.items():
                language_scores[lang] = language_scores.get(lang, 0) + content_score
        
        # Determine primary language
        if not language_scores or max(language_scores.values()) == 0:
            return DetectionResult(language="Unknown", confidence=0.0)
        
        primary_language = max(language_scores.items(), key=lambda x: x[1])
        total_score = sum(language_scores.values())
        confidence = primary_language[1] / total_score if total_score > 0 else 0.0
        
        # Detect frameworks
        frameworks = self._detect_frameworks(files, filenames, file_contents)
        
        # Detect package managers and build tools
        package_managers = self._detect_package_managers(filenames)
        build_tools = self._detect_build_tools(filenames, frameworks)
        testing_frameworks = self._detect_testing_frameworks(frameworks, file_contents)
        
        # Determine project type
        project_type = self._determine_project_type(primary_language[0], frameworks, files)
        
        return DetectionResult(
            language=primary_language[0],
            confidence=min(confidence, 1.0),
            frameworks=frameworks,
            package_managers=package_managers,
            build_tools=build_tools,
            testing_frameworks=testing_frameworks,
            project_type=project_type,
            evidence={
                "extension_counts": extension_counts,
                "language_scores": language_scores,
                "detected_filenames": list(filenames)[:20]  # Limit for brevity
            }
        )
    
    def _analyze_content(self, file_contents: Dict[str, str]) -> Dict[str, float]:
        """Analyze file contents for language patterns."""
        content_scores = {}
        
        for file_path, content in file_contents.items():
            # Limit content analysis to avoid performance issues
            content_sample = content[:5000]  # First 5000 characters
            
            for lang, pattern in self._language_patterns.items():
                score = 0
                for content_pattern in pattern.content_patterns:
                    if content_pattern in content_sample:
                        score += 5
                
                content_scores[lang] = content_scores.get(lang, 0) + score
        
        return content_scores
    
    def _detect_frameworks(
        self,
        files: List[Dict[str, Any]],
        filenames: Set[str],
        file_contents: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """Detect frameworks used in the project."""
        detected_frameworks = []
        
        for framework in self._framework_patterns:
            score = 0
            
            # Check indicators
            for indicator in framework.indicators:
                if indicator.lower() in filenames:
                    score += 20
            
            # Check file patterns
            for file_info in files:
                if file_info.get("type") == "file":
                    path = file_info.get("path", "")
                    for pattern in framework.file_patterns:
                        if self._matches_pattern(path, pattern):
                            score += 10
            
            # Check dependencies in package files
            if file_contents:
                for file_path, content in file_contents.items():
                    if any(pkg in file_path.lower() for pkg in ["package.json", "requirements.txt", "go.mod", "cargo.toml", "pom.xml"]):
                        for dep in framework.dependencies:
                            if dep in content:
                                score += 30
                    
                    # Check content patterns
                    for pattern in framework.content_patterns:
                        if pattern in content:
                            score += 5
            
            if score >= 20:  # Threshold for framework detection
                detected_frameworks.append(framework.name)
        
        return detected_frameworks
    
    def _detect_package_managers(self, filenames: Set[str]) -> List[str]:
        """Detect package managers."""
        managers = []
        
        package_manager_indicators = {
            "npm": ["package.json", "package-lock.json"],
            "yarn": ["yarn.lock"],
            "pip": ["requirements.txt", "setup.py"],
            "poetry": ["pyproject.toml", "poetry.lock"],
            "pipenv": ["pipfile", "pipfile.lock"],
            "cargo": ["cargo.toml", "cargo.lock"],
            "go modules": ["go.mod", "go.sum"],
            "maven": ["pom.xml"],
            "gradle": ["build.gradle", "gradle.properties"],
            "composer": ["composer.json", "composer.lock"],
            "bundle": ["gemfile", "gemfile.lock"],
            "nuget": ["packages.config", "*.csproj"],
            "pub": ["pubspec.yaml", "pubspec.lock"]
        }
        
        for manager, indicators in package_manager_indicators.items():
            if any(indicator in filenames for indicator in indicators):
                managers.append(manager)
        
        return managers
    
    def _detect_build_tools(self, filenames: Set[str], frameworks: List[str]) -> List[str]:
        """Detect build tools."""
        tools = []
        
        build_tool_indicators = {
            "webpack": ["webpack.config.js"],
            "vite": ["vite.config.js", "vite.config.ts"],
            "rollup": ["rollup.config.js"],
            "parcel": [".parcelrc"],
            "make": ["makefile"],
            "cmake": ["cmakelists.txt"],
            "ninja": ["build.ninja"],
            "bazel": ["build", "workspace"],
            "meson": ["meson.build"],
            "ant": ["build.xml"],
            "sbt": ["build.sbt"],
            "leiningen": ["project.clj"]
        }
        
        for tool, indicators in build_tool_indicators.items():
            if any(indicator in filenames for indicator in indicators):
                tools.append(tool)
        
        # Add framework-specific build tools
        if "Maven" in frameworks:
            tools.append("maven")
        if "Gradle" in frameworks:
            tools.append("gradle")
        
        return tools
    
    def _detect_testing_frameworks(
        self,
        frameworks: List[str],
        file_contents: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """Detect testing frameworks."""
        testing_frameworks = []
        
        # From already detected frameworks
        framework_testing_map = {
            "Jest": "jest",
            "Mocha": "mocha",
            "Pytest": "pytest"
        }
        
        for framework in frameworks:
            if framework in framework_testing_map:
                testing_frameworks.append(framework_testing_map[framework])
        
        # Additional detection from content
        if file_contents:
            testing_patterns = {
                "unittest": ["import unittest", "TestCase"],
                "pytest": ["import pytest", "def test_"],
                "jest": ["describe(", "it(", "test(", "expect("],
                "mocha": ["describe(", "it(", "before(", "after("],
                "jasmine": ["describe(", "it(", "beforeEach(", "afterEach("],
                "rspec": ["describe ", "it ", "expect("],
                "junit": ["@Test", "import org.junit"],
                "go test": ["func Test", "testing.T"],
                "rust test": ["#[test]", "#[cfg(test)]"]
            }
            
            for file_path, content in file_contents.items():
                for framework, patterns in testing_patterns.items():
                    if any(pattern in content for pattern in patterns):
                        if framework not in testing_frameworks:
                            testing_frameworks.append(framework)
        
        return testing_frameworks
    
    def _determine_project_type(
        self,
        language: str,
        frameworks: List[str],
        files: List[Dict[str, Any]]
    ) -> str:
        """Determine project type based on analysis."""
        
        # Web application indicators
        web_indicators = ["React", "Vue.js", "Angular", "Django", "Flask", "Express.js", "Spring Boot"]
        if any(framework in frameworks for framework in web_indicators):
            return "web_application"
        
        # Mobile application indicators
        mobile_indicators = ["React Native", "Flutter", "Swift", "Kotlin"]
        if any(framework in frameworks for framework in mobile_indicators):
            return "mobile_application"
        
        # CLI tool indicators
        cli_indicators = ["click", "argparse", "commander", "cobra"]
        if any(indicator in str(frameworks).lower() for indicator in cli_indicators):
            return "cli_tool"
        
        # Check for main entry points
        entry_points = ["main.py", "main.js", "main.go", "main.rs", "main.cpp", "main.c"]
        has_main = any(
            file_info.get("path", "").split("/")[-1] in entry_points
            for file_info in files
            if file_info.get("type") == "file"
        )
        
        if has_main:
            return "application"
        
        # Library indicators
        library_indicators = ["setup.py", "pyproject.toml", "package.json", "cargo.toml", "pom.xml"]
        filenames = {file_info.get("path", "").split("/")[-1].lower() for file_info in files}
        
        if any(indicator in filenames for indicator in library_indicators):
            return "library"
        
        # Microservice indicators
        if "Docker" in frameworks or "Kubernetes" in frameworks:
            return "microservice"
        
        # Data science indicators
        data_science_indicators = ["jupyter", "pandas", "numpy", "matplotlib", "tensorflow", "pytorch"]
        if any(indicator in str(frameworks).lower() for indicator in data_science_indicators):
            return "data_science"
        
        # Default based on language
        language_defaults = {
            "Python": "script",
            "JavaScript": "web_application",
            "TypeScript": "web_application",
            "Java": "application",
            "Go": "application",
            "Rust": "application",
            "C++": "application",
            "C": "application",
            "Shell": "script"
        }
        
        return language_defaults.get(language, "unknown")
    
    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches a pattern with wildcards."""
        if "*" not in pattern:
            return pattern in path
        
        # Simple wildcard matching
        parts = pattern.split("*")
        pos = 0
        for part in parts:
            if not part:
                continue
            new_pos = path.find(part, pos)
            if new_pos == -1:
                return False
            pos = new_pos + len(part)
        
        return True
    
    def get_language_info(self, language: str) -> Optional[LanguagePattern]:
        """Get language pattern information."""
        return self._language_patterns.get(language)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return list(self._language_patterns.keys())
    
    def analyze_language_distribution(self, files: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Analyze language distribution in the project."""
        extension_counts = {}
        total_files = 0
        
        for file_info in files:
            if file_info.get("type") == "file":
                total_files += 1
                path = file_info.get("path", "")
                filename = Path(path).name
                
                if "." in filename:
                    ext = "." + filename.split(".")[-1].lower()
                    extension_counts[ext] = extension_counts.get(ext, 0) + 1
        
        # Map extensions to languages
        language_distribution = {}
        for lang, pattern in self._language_patterns.items():
            file_count = sum(extension_counts.get(ext, 0) for ext in pattern.extensions)
            if file_count > 0:
                percentage = (file_count / total_files) * 100 if total_files > 0 else 0
                language_distribution[lang] = {
                    "file_count": file_count,
                    "percentage": round(percentage, 2),
                    "extensions": [ext for ext in pattern.extensions if ext in extension_counts]
                }
        
        return language_distribution