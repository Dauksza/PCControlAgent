"""
Quick setup script for Mistral Agent Platform
"""
import os
import subprocess
import sys
from pathlib import Path

def setup_environment():
    """
    Automated setup for the platform
    """
    print("🚀 Mistral AI Agent Platform Setup\n")
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ required")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    
    print("✅ Python version OK")
    print(f"   Version: {sys.version}")
    
    # Create virtual environment
    print("\n📦 Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("   ✅ Virtual environment created")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to create virtual environment: {e}")
        sys.exit(1)
    
    # Determine pip path
    if os.name == "nt":  # Windows
        pip_path = "venv\\Scripts\\pip"
        python_path = "venv\\Scripts\\python"
        activate_cmd = "venv\\Scripts\\activate"
    else:  # Unix-like
        pip_path = "venv/bin/pip"
        python_path = "venv/bin/python"
        activate_cmd = "source venv/bin/activate"
    
    # Upgrade pip
    print("\n📦 Upgrading pip...")
    try:
        subprocess.run([python_path, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        print("   ✅ Pip upgraded")
    except subprocess.CalledProcessError as e:
        print(f"   ⚠️  Warning: Failed to upgrade pip: {e}")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    print("   This may take a few minutes...")
    try:
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        print("   ✅ Dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to install dependencies: {e}")
        sys.exit(1)
    
    # Create .env file
    if not os.path.exists(".env"):
        print("\n🔧 Creating .env file...")
        with open(".env", "w") as f:
            f.write("# Mistral API Configuration\n")
            f.write("MISTRAL_API_KEY=your_api_key_here\n")
            f.write("\n# Model Configuration\n")
            f.write("DEFAULT_MODEL=mistral-large-2407\n")
            f.write("MAX_ITERATIONS=50\n")
            f.write("\n# Feature Flags\n")
            f.write("ENABLE_STREAMING=True\n")
            f.write("ENABLE_VISION=True\n")
            f.write("\n# Logging\n")
            f.write("LOG_LEVEL=INFO\n")
        print("   ✅ .env file created")
        print("   ⚠️  Please edit .env and add your Mistral API key")
    else:
        print("\n✅ .env file already exists")
    
    # Create directory structure
    print("\n📁 Creating directory structure...")
    dirs = [
        "logs",
        "data",
        "cache",
        "uploads",
        "exports"
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(exist_ok=True)
    print("   ✅ Directories created")
    
    # Create .gitignore
    if not os.path.exists(".gitignore"):
        print("\n🔧 Creating .gitignore...")
        with open(".gitignore", "w") as f:
            f.write("# Python\n")
            f.write("__pycache__/\n")
            f.write("*.py[cod]\n")
            f.write("*$py.class\n")
            f.write("*.so\n")
            f.write(".Python\n")
            f.write("venv/\n")
            f.write("env/\n")
            f.write(".venv/\n")
            f.write("\n# Environment\n")
            f.write(".env\n")
            f.write(".env.local\n")
            f.write("\n# Logs and data\n")
            f.write("logs/\n")
            f.write("cache/\n")
            f.write("data/\n")
            f.write("uploads/\n")
            f.write("exports/\n")
            f.write("*.log\n")
            f.write("\n# IDE\n")
            f.write(".vscode/\n")
            f.write(".idea/\n")
            f.write("*.swp\n")
            f.write("*.swo\n")
            f.write("\n# Testing\n")
            f.write(".pytest_cache/\n")
            f.write(".coverage\n")
            f.write("htmlcov/\n")
        print("   ✅ .gitignore created")
    
    print("\n" + "="*50)
    print("✅ Setup complete!")
    print("="*50)
    print("\nNext steps:")
    print("1. Edit .env file with your Mistral API key:")
    print("   - Get your API key from: https://console.mistral.ai/")
    print(f"\n2. Activate virtual environment:")
    print(f"   {activate_cmd}")
    print("\n3. Run the application:")
    print("   python app.py")
    print("\n4. Visit the API documentation:")
    print("   http://localhost:8000/docs")
    print("\n5. (Optional) Run tests:")
    print("   pytest")
    print("="*50)

if __name__ == "__main__":
    try:
        setup_environment()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup failed: {e}")
        sys.exit(1)
