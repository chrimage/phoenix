# UV Guide: Creating Portable Python Scripts with Built-in Dependencies

This guide explains how to use [uv](https://github.com/astral-sh/uv), a high-performance Python package manager and resolver, to create portable Python scripts with built-in dependencies. This approach eliminates the need for manual environment management and ensures scripts work consistently across different systems.

## Table of Contents

- [Introduction to uv](#introduction-to-uv)
- [Setup and Installation](#setup-and-installation)
- [Creating Portable Scripts](#creating-portable-scripts)
  - [Declaring Dependencies](#declaring-dependencies)
  - [Locking Dependencies](#locking-dependencies)
  - [Using Alternative Package Indexes](#using-alternative-package-indexes)
  - [Improving Reproducibility](#improving-reproducibility)
- [Python Version Management](#python-version-management)
- [Advanced Features](#advanced-features)
- [Best Practices](#best-practices)
- [Example Use Cases](#example-use-cases)

## Introduction to uv

uv is a high-performance Python package manager and resolver designed to work alongside pip and other Python packaging tools. It offers several advantages for creating portable scripts:

- **Speed**: uv is significantly faster than pip for installing packages
- **Built-in dependency management**: No need to manually create and manage virtual environments
- **Self-contained scripts**: Dependencies can be declared within the script itself
- **Cross-version compatibility**: Easy specification of required Python versions
- **Reproducible environments**: Lock dependencies for consistent execution

## Setup and Installation

To install uv, use one of the following methods:

```bash
# Using pip
pip install uv

# Using brew (macOS)
brew install uv

# Install script
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Creating Portable Scripts

### Declaring Dependencies

UV supports the PEP 723 inline script metadata format, which allows dependencies to be declared directly in the script file.

#### Creating a New Script with Dependencies

Use `uv init --script` to create a new script with inline metadata:

```bash
uv init --script example.py --python 3.12
```

Add dependencies using the `uv add --script` command:

```bash
uv add --script example.py 'requests<3' 'rich'
```

This will add a `script` section at the top of the script declaring the dependencies using TOML:

```python
# /// script
# dependencies = [
#   "requests<3",
#   "rich",
# ]
# ///

import requests
from rich.pretty import pprint

resp = requests.get("https://peps.python.org/api/peps.json")
data = resp.json()
pprint([(k, v["title"]) for k, v in data.items()][:10])
```

#### Running a Script with Dependencies

Simply run the script with uv:

```bash
uv run example.py
```

uv will automatically create an environment with the necessary dependencies to run the script.

#### Ad-hoc Dependencies

For quick experiments without modifying the script, use the `--with` option:

```bash
uv run --with rich example.py
```

Constraints can be added to specify versions:

```bash
uv run --with 'rich>12,<13' example.py
```

Multiple dependencies can be requested by repeating the `--with` option.

### Locking Dependencies

For reproducible environments, lock dependencies using the `uv lock` command:

```bash
uv lock --script example.py
```

This creates a `.lock` file adjacent to the script (e.g., `example.py.lock`). Subsequent operations like `uv run`, `uv add`, `uv export`, and `uv tree` will reuse the locked dependencies, updating the lockfile if necessary.

### Using Alternative Package Indexes

To use an alternative package index to resolve dependencies, use the `--index` option:

```bash
uv add --index "https://example.com/simple" --script example.py 'requests<3' 'rich'
```

This will include the package index information in the inline metadata:

```python
# [[tool.uv.index]]
# url = "https://example.com/simple"
```

If authentication is required to access the package index, refer to the [uv package index documentation](https://docs.astral.sh/uv/configuration/indexes/).

### Improving Reproducibility

To improve reproducibility, use the `exclude-newer` field in the `tool.uv` section of the inline script metadata. This limits uv to only consider distributions released before a specific date:

```python
# /// script
# dependencies = [
#   "requests",
# ]
# [tool.uv]
# exclude-newer = "2023-10-16T00:00:00Z"
# ///

import requests
print(requests.__version__)
```

The date must be specified as an [RFC 3339](https://www.rfc-editor.org/rfc/rfc3339.html) timestamp (e.g., `2023-10-16T00:00:00Z`).

## Python Version Management

### Specifying Python Versions

uv allows you to specify Python version requirements directly in the script:

```python
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

# Use some syntax added in Python 3.12
type Point = tuple[float, float]
print(Point)
```

uv will automatically search for and use the required Python version, downloading it if necessary.

### Using Different Python Versions

You can also specify a Python version at runtime:

```bash
# Use the default Python version
uv run example.py

# Use a specific Python version
uv run --python 3.10 example.py
```

uv supports various version specification formats:

- `<version>` (e.g., `3`, `3.12`, `3.12.3`)
- `<version-specifier>` (e.g., `>=3.12,<3.13`)
- `<implementation>` (e.g., `cpython` or `cp`)
- `<implementation>@<version>` (e.g., `cpython@3.12`)
- `<implementation><version>` (e.g., `cpython3.12` or `cp312`)
- `<implementation><version-specifier>` (e.g., `cpython>=3.12,<3.13`)
- `<implementation>-<version>-<os>-<arch>-<libc>` (e.g., `cpython-3.12.3-macos-aarch64-none`)

You can also specify a system Python interpreter:

- `<executable-path>` (e.g., `/opt/homebrew/bin/python3`)
- `<executable-name>` (e.g., `mypython3`)
- `<install-dir>` (e.g., `/some/environment/`)

## Advanced Features

### GUI Scripts

On Windows, uv will run `.pyw` files using `pythonw`, allowing GUI applications without a console window:

```python
# example.pyw
from tkinter import Tk, ttk

root = Tk()
root.title("uv")
frm = ttk.Frame(root, padding=10)
frm.grid()
ttk.Label(frm, text="Hello World").grid(column=0, row=0)
root.mainloop()
```

Run with:

```bash
uv run example.pyw
```

This works with dependencies as well:

```bash
uv run --with PyQt5 example_pyqt.pyw
```

### Reading Scripts from Standard Input

You can pass scripts directly through standard input:

```bash
echo 'print("hello world!")' | uv run -
```

Or using a here-document:

```bash
uv run - <<EOF
print("hello world!")
EOF
```

## Best Practices

1. **Declare dependencies explicitly**: Use inline script metadata to make scripts self-contained and portable.

2. **Lock dependencies**: Use `uv lock --script` to create a lockfile for reproducible environments.

3. **Specify Python version requirements**: Include `requires-python` in the metadata to ensure compatibility.

4. **Add version constraints**: Use version specifiers for dependencies to prevent breaking changes.

5. **Consider reproducibility**: Use the `exclude-newer` field to limit packages to those available before a certain date.

6. **Use the appropriate Python implementation**: Specify the implementation (CPython, PyPy) if your script depends on implementation-specific features.

7. **Test on different Python versions**: Use the `--python` flag to verify that your script works across different Python versions.

8. **Keep scripts focused**: Create smaller, single-purpose scripts rather than large monolithic scripts.

## Example Use Cases

### Data Processing Script

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pandas>=2.0.0,<3.0.0",
#   "matplotlib>=3.7.0",
#   "numpy>=1.24.0",
# ]
# [tool.uv]
# exclude-newer = "2023-12-01T00:00:00Z"
# ///

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('data.csv')

# Process data
df['value'] = df['value'].apply(lambda x: np.log(x) if x > 0 else 0)

# Create visualization
plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['value'])
plt.title('Log Values Over Time')
plt.savefig('output.png')
print("Analysis complete, results saved to output.png")
```

### Web Scraping Utility

```python
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "requests>=2.28.0,<3.0.0",
#   "beautifulsoup4>=4.11.0",
#   "rich>=13.0.0",
# ]
# ///

import sys
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from rich.console import Console
from rich.table import Table

console = Console()

def scrape_webpage(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    title = soup.title.text.strip() if soup.title else "No title found"
    
    # Extract all links
    links = soup.find_all('a', href=True)
    valid_links = []
    
    base_domain = urlparse(url).netloc
    
    for link in links:
        href = link['href']
        if href.startswith('http'):
            domain = urlparse(href).netloc
            valid_links.append({
                'url': href, 
                'text': link.text.strip() or href,
                'external': domain != base_domain
            })
    
    # Display results
    console.print(f"[bold green]Title:[/bold green] {title}")
    console.print(f"[bold green]Found[/bold green] {len(valid_links)} links")
    
    table = Table(title="Links")
    table.add_column("Text", style="cyan")
    table.add_column("URL", style="green")
    table.add_column("Type", style="magenta")
    
    for link in valid_links[:20]:  # Limit to first 20 links
        table.add_row(
            link['text'][:50] + ('...' if len(link['text']) > 50 else ''),
            link['url'][:70] + ('...' if len(link['url']) > 70 else ''),
            "External" if link['external'] else "Internal"
        )
    
    console.print(table)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        console.print("[bold red]Usage: uv run scraper.py <url>[/bold red]")
        sys.exit(1)
    
    scrape_webpage(sys.argv[1])
```

### CLI Tool with Click

```python
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "click>=8.1.0",
#   "tqdm>=4.64.0",
#   "pyyaml>=6.0",
# ]
# ///

import os
import yaml
import click
from tqdm import tqdm
import time

@click.group()
def cli():
    """File Management Utility Tool"""
    pass

@cli.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--output', '-o', default='report.yaml', help='Output file name')
def analyze(directory, output):
    """Analyze a directory and generate a report."""
    click.echo(f"Analyzing directory: {directory}")
    
    results = {
        'directory': directory,
        'file_count': 0,
        'extension_stats': {},
        'total_size': 0,
    }
    
    for root, _, files in os.walk(directory):
        for file in tqdm(files, desc="Processing files"):
            file_path = os.path.join(root, file)
            size = os.path.getsize(file_path)
            
            results['file_count'] += 1
            results['total_size'] += size
            
            _, ext = os.path.splitext(file)
            ext = ext.lower() if ext else "(no extension)"
            
            if ext not in results['extension_stats']:
                results['extension_stats'][ext] = {
                    'count': 0,
                    'total_size': 0,
                }
            
            results['extension_stats'][ext]['count'] += 1
            results['extension_stats'][ext]['total_size'] += size
    
    # Convert sizes to MB for readability
    results['total_size'] = round(results['total_size'] / (1024 * 1024), 2)
    for ext in results['extension_stats']:
        results['extension_stats'][ext]['total_size'] = round(
            results['extension_stats'][ext]['total_size'] / (1024 * 1024), 2
        )
    
    # Write report to file
    with open(output, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    click.echo(f"Analysis complete! Report saved to {output}")
    click.echo(f"Total files: {results['file_count']}")
    click.echo(f"Total size: {results['total_size']} MB")

@cli.command()
@click.argument('source', type=click.Path(exists=True))
@click.argument('destination', type=click.Path())
@click.option('--compress/--no-compress', default=False, help='Compress the output')
def backup(source, destination, compress):
    """Backup a directory to another location."""
    click.echo(f"Backing up {source} to {destination}")
    
    # Simulate a backup operation with progress bar
    with tqdm(total=100, desc="Backing up") as pbar:
        for i in range(100):
            time.sleep(0.05)  # Simulate work
            pbar.update(1)
    
    if compress:
        click.echo("Compressing backup...")
        time.sleep(2)  # Simulate compression
    
    click.echo(f"✅ Backup complete: {destination}")

if __name__ == '__main__':
    cli()
```

Run the CLI tool with:

```bash
uv run tool.py analyze ~/Documents --output docs_report.yaml
uv run tool.py backup ~/Pictures ~/Backups --compress
```

These examples show how to create self-contained, portable Python scripts that handle their own dependencies and version requirements, making them easy to distribute and run consistently across different environments.
