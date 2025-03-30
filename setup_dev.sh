#!/bin/bash
# Development environment setup script for Phoenix

# Ensure Python 3.11 is installed (for Fedora)
if ! command -v python3.11 &> /dev/null; then
    echo "Installing Python 3.11..."
    sudo dnf install -y python3.11 python3.11-devel
fi

# Create and activate virtual environment
echo "Creating virtual environment with Python 3.11..."
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install spaCy language model
echo "Installing spaCy language model..."
python -m spacy download en_core_web_sm

# Create .env file template if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env template..."
    echo "GOOGLE_API_KEY=your_google_api_key_here" > .env
    echo "Please update .env with your actual Google API key."
fi

echo "Development setup complete! Activate the virtual environment with:"
echo "  source venv/bin/activate"