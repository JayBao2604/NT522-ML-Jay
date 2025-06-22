#!/usr/bin/env python3
"""
Simple test to verify backend functionality
"""

import sys
import os
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="Simple Test API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Simple test server is running"}

@app.get("/test-attack-pool")
async def test_attack_pool():
    """Test if we can load and return attack pool data"""
    try:
        # Try to load attack pool
        attack_pool_path = "attack_pool.csv"
        if not os.path.exists(attack_pool_path):
            return {
                "success": False,
                "error": f"Attack pool file not found: {attack_pool_path}",
                "current_dir": os.getcwd()
            }
        
        attack_pool_data = pd.read_csv(attack_pool_path)
        
        # Check format
        if "adversarial_code" in attack_pool_data.columns:
            snippets = attack_pool_data["adversarial_code"].dropna().tolist()
        elif len(attack_pool_data.columns) >= 2:
            snippets = attack_pool_data.iloc[:, 1].dropna().tolist()
        else:
            snippets = attack_pool_data.iloc[:, 0].dropna().tolist()
        
        # Clean snippets
        snippets = [str(snippet) for snippet in snippets if pd.notna(snippet)]
        
        return {
            "success": True,
            "total_rows": len(attack_pool_data),
            "columns": list(attack_pool_data.columns),
            "valid_snippets": len(snippets),
            "sample_snippet": snippets[0][:100] if snippets else "None",
            "current_dir": os.getcwd()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "current_dir": os.getcwd()
        }

if __name__ == "__main__":
    print("Starting simple test server on http://localhost:3002")
    print("Test endpoints:")
    print("- GET http://localhost:3002/health")
    print("- GET http://localhost:3002/test-attack-pool")
    
    uvicorn.run(app, host="0.0.0.0", port=3002) 