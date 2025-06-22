#!/usr/bin/env python3
"""
Simple test runner for EatVul API endpoints
"""

import os
import sys
import subprocess

def run_tests():
    """Run the test suite"""
    print("🧪 Running EatVul API Tests...")
    print("=" * 50)
    
    # Change to backend directory
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(backend_dir)
    
    # Install test requirements if needed
    try:
        import pytest
        print("✅ pytest already installed")
    except ImportError:
        print("📦 Installing test dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-test.txt"], check=True)
    
    # Run pytest
    print("\n🚀 Running tests...")
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "test_api.py", 
        "-v", 
        "--tb=short",
        "--color=yes"
    ])
    
    if result.returncode == 0:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        
    return result.returncode

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code) 