#!/usr/bin/env python
"""
Example script demonstrating web browsing functionality in Euclid.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path so we can import euclid
sys.path.insert(0, str(Path(__file__).parent.parent))

from euclid.tools.web import web_tool, search_tool
from euclid.formatting import console, EnhancedMarkdown

def main():
    parser = argparse.ArgumentParser(description="Test web browsing functionality")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Fetch command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch and analyze a web page")
    fetch_parser.add_argument("url", help="URL to fetch")
    fetch_parser.add_argument("--prompt", "-p", help="Prompt for content analysis", default=None)
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search the web")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--num", "-n", type=int, help="Number of results", default=5)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "fetch":
        result = web_tool(args.url, args.prompt)
        console.print(EnhancedMarkdown(result))
    
    elif args.command == "search":
        result = search_tool(args.query, args.num)
        console.print(EnhancedMarkdown(result))

if __name__ == "__main__":
    main()