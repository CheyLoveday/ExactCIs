#!/usr/bin/env python3
"""
Link Checker for ExactCIs Documentation

This script checks for broken links in the converted Markdown files.
It identifies:
1. Internal links that point to non-existent files
2. Internal links that point to non-existent anchors
3. External links that return error status codes

Usage:
    python check_links.py [--markdown-dir MARKDOWN_DIR] [--check-external]

Options:
    --markdown-dir MARKDOWN_DIR  Directory containing Markdown files (default: markdown_output)
    --check-external             Also check external links (slower)
"""

import os
import re
import sys
import argparse
import logging
import requests
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('link_check_results.log')
    ]
)
logger = logging.getLogger(__name__)

# Regular expressions for finding links in Markdown
MARKDOWN_LINK_PATTERN = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
HTML_LINK_PATTERN = re.compile(r'<a\s+(?:[^>]*?\s+)?href="([^"]*)"')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Check for broken links in Markdown files')
    parser.add_argument('--markdown-dir', default='markdown_output',
                        help='Directory containing Markdown files (default: markdown_output)')
    parser.add_argument('--check-external', action='store_true',
                        help='Also check external links (slower)')
    return parser.parse_args()

def find_markdown_files(directory):
    """Find all Markdown files in the given directory and its subdirectories."""
    markdown_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                markdown_files.append(os.path.join(root, file))
    return markdown_files

def extract_links_from_file(file_path):
    """Extract all links from a Markdown file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    links = []
    
    # Find Markdown links
    for match in MARKDOWN_LINK_PATTERN.finditer(content):
        link_text, link_url = match.groups()
        links.append((link_url, link_text))
    
    # Find HTML links
    for match in HTML_LINK_PATTERN.finditer(content):
        link_url = match.group(1)
        links.append((link_url, None))
    
    return links

def is_external_link(link):
    """Check if a link is external (starts with http:// or https://)."""
    return link.startswith(('http://', 'https://'))

def check_external_link(link):
    """Check if an external link is valid by making a HEAD request."""
    try:
        response = requests.head(link, timeout=10, allow_redirects=True)
        if response.status_code >= 400:
            # Try GET request if HEAD fails (some servers don't support HEAD)
            response = requests.get(link, timeout=10, stream=True)
            response.close()  # Close the connection immediately
        
        if response.status_code >= 400:
            return False, f"HTTP status code: {response.status_code}"
        return True, None
    except requests.RequestException as e:
        return False, str(e)

def normalize_path(base_path, link_path):
    """Normalize a relative path based on the base path."""
    if link_path.startswith('/'):
        # Absolute path within the site
        return link_path.lstrip('/')
    
    # Relative path
    base_dir = os.path.dirname(base_path)
    return os.path.normpath(os.path.join(base_dir, link_path))

def check_internal_link(markdown_dir, source_file, link):
    """Check if an internal link points to an existing file and anchor."""
    # Split the link into file path and anchor
    if '#' in link:
        file_path, anchor = link.split('#', 1)
    else:
        file_path, anchor = link, None
    
    # Handle empty file path (link to an anchor in the same file)
    if not file_path:
        return True, None
    
    # Normalize the path
    relative_source = os.path.relpath(source_file, markdown_dir)
    normalized_path = normalize_path(relative_source, file_path)
    
    # Check if the file exists
    target_file = os.path.join(markdown_dir, normalized_path)
    if not os.path.isfile(target_file):
        # Check if adding .md extension helps
        if not target_file.endswith('.md'):
            md_target = f"{target_file}.md"
            if os.path.isfile(md_target):
                target_file = md_target
            else:
                return False, f"File not found: {normalized_path}"
        else:
            return False, f"File not found: {normalized_path}"
    
    # If there's an anchor, check if it exists in the target file
    if anchor:
        with open(target_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for HTML anchors
        if f'id="{anchor}"' not in content and f'name="{anchor}"' not in content:
            # Look for Markdown headings that would generate the anchor
            heading_pattern = re.compile(r'^#+\s+(.+?)$', re.MULTILINE)
            for match in heading_pattern.finditer(content):
                heading_text = match.group(1).strip()
                # Convert heading to anchor format (lowercase, spaces to hyphens)
                heading_anchor = re.sub(r'[^\w\- ]', '', heading_text).lower().replace(' ', '-')
                if heading_anchor == anchor:
                    return True, None
            
            return False, f"Anchor not found: #{anchor} in {normalized_path}"
    
    return True, None

def check_links(markdown_dir, check_external=False):
    """Check all links in all Markdown files."""
    markdown_files = find_markdown_files(markdown_dir)
    logger.info(f"Found {len(markdown_files)} Markdown files to check")
    
    broken_links = []
    total_links = 0
    
    for file_path in markdown_files:
        relative_path = os.path.relpath(file_path, markdown_dir)
        logger.info(f"Checking links in {relative_path}")
        
        links = extract_links_from_file(file_path)
        total_links += len(links)
        
        for link_url, link_text in links:
            # Skip empty links
            if not link_url or link_url.startswith('#'):
                continue
            
            # Check if the link is external
            if is_external_link(link_url):
                if check_external:
                    valid, error = check_external_link(link_url)
                    if not valid:
                        broken_links.append((relative_path, link_url, link_text, error))
                        logger.warning(f"Broken external link in {relative_path}: {link_url} - {error}")
            else:
                # Internal link
                valid, error = check_internal_link(markdown_dir, file_path, link_url)
                if not valid:
                    broken_links.append((relative_path, link_url, link_text, error))
                    logger.warning(f"Broken internal link in {relative_path}: {link_url} - {error}")
    
    return broken_links, total_links

def generate_report(broken_links, total_links, output_file='link_check_report.md'):
    """Generate a Markdown report of broken links."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Link Check Report\n\n")
        f.write(f"Total links checked: {total_links}\n")
        f.write(f"Broken links found: {len(broken_links)}\n\n")
        
        if broken_links:
            f.write("## Broken Links\n\n")
            f.write("| File | Link | Link Text | Error |\n")
            f.write("|------|------|-----------|-------|\n")
            
            for file_path, link_url, link_text, error in broken_links:
                # Escape pipe characters in Markdown table
                safe_link_text = str(link_text).replace('|', '\\|') if link_text else 'N/A'
                safe_link_url = link_url.replace('|', '\\|')
                safe_error = error.replace('|', '\\|')
                
                f.write(f"| {file_path} | {safe_link_url} | {safe_link_text} | {safe_error} |\n")
        else:
            f.write("No broken links found. All links are valid!\n")

def main():
    """Main function."""
    args = parse_arguments()
    
    markdown_dir = args.markdown_dir
    if not os.path.isdir(markdown_dir):
        logger.error(f"Directory not found: {markdown_dir}")
        sys.exit(1)
    
    logger.info(f"Checking links in {markdown_dir}")
    if args.check_external:
        logger.info("External link checking is enabled")
    
    broken_links, total_links = check_links(markdown_dir, args.check_external)
    
    # Generate report
    report_file = 'link_check_report.md'
    generate_report(broken_links, total_links, report_file)
    logger.info(f"Report generated: {report_file}")
    
    # Return exit code based on whether broken links were found
    if broken_links:
        logger.error(f"Found {len(broken_links)} broken links out of {total_links} total links")
        return 1
    else:
        logger.info(f"All {total_links} links are valid!")
        return 0

if __name__ == '__main__':
    sys.exit(main())