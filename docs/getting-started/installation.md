# Installation

This guide will help you install and set up the EVE Pipeline on your system.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10 or higher**
- **uv** (recommended) or **pip** for package management

### Install uv (Recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation Methods

### Method 1: Using uv (Recommended)

1. **Clone the repository**

    ```bash
    git clone https://github.com/eve-esa/eve-pipeline.git
    cd eve-pipeline
    ```

2. **Install dependencies**

    ```bash
    uv sync
    ```

### Method 2: Using pip

1. **Clone the repository**

    ```bash
    git clone https://github.com/eve-esa/eve-pipeline.git
    cd eve-pipeline
    ```

2. **Create a virtual environment**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

## Optional Dependencies


## Server Setup

Some pipeline stages require external servers:

### PII Server

For PII (Personally Identifiable Information) removal:

```bash
cd server
python3 pii_server.py
```

### OCR Server

```bash
cd server
python3 nougat_server.py
```


## Next Steps

After installation, proceed to the [Quick Start](quick-start.md) guide to learn how to configure and run your first pipeline.