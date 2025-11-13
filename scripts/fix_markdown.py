#!/usr/bin/env python3
"""Utility script to fix markdown files with JSON wrapping or escaped newlines"""
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.crew import extract_markdown_from_response


def fix_markdown_file(filepath: str) -> None:
    """Fix a markdown file that has JSON wrapping or escaped characters"""
    file_path = Path(filepath)

    if not file_path.exists():
        print(f"Error: File not found: {filepath}")
        return

    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract and clean markdown
    cleaned_markdown = extract_markdown_from_response(content)

    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_markdown)

    print(f"✅ Fixed: {filepath}")
    print(f"   File now contains proper markdown format")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/fix_markdown.py <markdown_file>")
        print("Example: python scripts/fix_markdown.py documentation_project.md")
        sys.exit(1)

    fix_markdown_file(sys.argv[1])
