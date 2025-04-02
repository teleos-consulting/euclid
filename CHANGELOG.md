# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Docker support for containerized deployment
- Comprehensive test suite with Docker integration
- Extended documentation (CONTRIBUTING.md, TEST_PLAN.md)
- Development setup with `pip install -e ".[dev]"`
- MCP-compatible API server for programmatic access
- Simple command-line experience (just run `euclid`)
- Advanced web browsing capability:
  - Content fetching with caching and summarization
  - Image extraction and analysis
  - HTML-to-markdown conversion with structure preservation
  - Support for various content types (HTML, JSON, text, images)
  - Web search with SerpAPI integration (optional)
  - Cache management for fast repeated access
- API endpoints for web browsing in the MCP server

### Changed
- Improved installation guide with Docker instructions
- Enhanced test script with better reporting
- Updated README with Docker and testing information

## [0.1.0] - 2025-04-01

### Added
- Initial implementation of Euclid
- Interactive CLI interface with rich markdown rendering
- Terminal User Interface (TUI) with split-screen view
- Function calling framework
- BatchTool for parallel execution
- File operation tools (View, LS, GlobTool, GrepTool, Edit, Replace)
- Model management (list, pull, remove, details, benchmark)
- RAG functionality with sentence-transformers
- Conversation history management
- Custom system prompts
- Autonomous agent functionality