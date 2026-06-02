#!/usr/bin/env python3
"""
Cosmos3 Inference Studio - Launcher Script
Run this script to start the application.
"""

import sys
from pathlib import Path

def check_dependencies():
    """Check dependencies. Alert but don't crash on PyTorch/diffusers absence for remote execution."""
    critical = ["PyQt6", "requests", "PIL", "numpy", "imageio"]
    ml_libs = ["torch", "diffusers", "transformers", "accelerate"]
    
    missing_critical = []
    for package in critical:
        try:
            # Handle PIL import name vs package name
            import_name = "PIL" if package == "pillow" else package
            __import__(import_name)
        except ImportError:
            missing_critical.append(package)
            
    if missing_critical:
        print("Error: Missing critical packages required to run the interface:")
        print(f"  {', '.join(missing_critical)}")
        print("\nPlease run: pip install -r requirements.txt")
        sys.exit(1)
        
    missing_ml = []
    for package in ml_libs:
        try:
            __import__(package)
        except ImportError:
            missing_ml.append(package)
            
    if missing_ml:
        print("Notice: PyTorch or deep learning packages are not installed locally:")
        print(f"  {', '.join(missing_ml)}")
        print("You can run in 'Remote GPU' mode connecting to another computer on the network,")
        print("or run in Local Mock/Demo mode. Local GPU generation will be unavailable.\n")

def main():
    """Main launcher function."""
    print("Starting Cosmos3 Inference Studio...")
    
    # Check dependencies
    check_dependencies()
    
    # Add project to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Import and run main application
    from app.main import main as run_app
    run_app()

if __name__ == "__main__":
    main()
