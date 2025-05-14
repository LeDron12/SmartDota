#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${GREEN}[+]${NC} $1"
}

print_error() {
    echo -e "${RED}[!]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if conda is already installed
if command -v conda &> /dev/null; then
    print_status "Conda is already installed"
else
    print_status "Installing Miniconda..."
    
    # Detect OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
        if [[ $(uname -m) == "x86_64" ]]; then
            CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
        fi
    else
        # Linux
        CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    fi

    # Download and install Miniconda
    curl -L $CONDA_URL -o miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh

    # Add conda to PATH
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    if [[ -f ~/.zshrc ]]; then
        echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.zshrc
    fi
    
    # Source the updated profile
    source ~/.bashrc
    if [[ -f ~/.zshrc ]]; then
        source ~/.zshrc
    fi
    
    print_status "Miniconda installed successfully"
fi

# Create and activate conda environment
ENV_NAME="smartdota"
print_status "Creating conda environment '$ENV_NAME' with Python 3.10..."

# Remove environment if it exists
conda env remove -n $ENV_NAME -y 2>/dev/null

# Create new environment
conda create -n $ENV_NAME python=3.10 -y

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install pip dependencies
print_status "Installing Python dependencies..."
pip install -r requirements.txt

# Verify installation
print_status "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas version: {pandas.__version__}')"

print_status "Setup complete! To activate the environment, run:"
echo "conda activate $ENV_NAME"

# Print next steps
echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Activate the environment: conda activate $ENV_NAME"
echo "2. Verify the installation: python -c 'import torch; print(torch.cuda.is_available())'"
echo "3. Start developing!" 