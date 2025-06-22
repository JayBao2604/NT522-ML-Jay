from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager
import os
import sys
import json
import asyncio
import time
import pandas as pd
import numpy as np
from datetime import datetime
import google.generativeai as genai
import re
import tempfile
import shutil
import csv
from io import StringIO

# Add the current directory to Python path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom modules
from train_codebert_model import CodeBERTTrainer
from adversarial_learning import AdversarialLearning
from fga_selection import *
from c_function_parser import extract_c_function, CFunctionParser

# Global variables
codebert_trainer = None
model_loaded = False
adversarial_learner = None
attack_pool_data = None
best_attack_snippet_result = None

# Configure Gemini API (you'll need to set your API key)
GEMINI_API_KEY = "AIzaSyB2HPcy0LPZKiN2TihoICDdOU_23mhqfa8"
genai.configure(api_key=GEMINI_API_KEY)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the CodeBERT model on startup"""
    global codebert_trainer, model_loaded, attack_pool_data
    
    print("=== EatVul Backend Startup ===")
    print("Loading CodeBERT model...")
    
    # Initialize model loading variables
    model_loaded = False
    codebert_trainer = None
    
    try:
        # Initialize trainer first
        print("🔄 Initializing CodeBERTTrainer...")
        codebert_trainer = CodeBERTTrainer()
        print("✅ CodeBERTTrainer initialized")
        
        # Load model from the specified directory
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "codebert-model")
        print(f"📁 Model path: {model_path}")
        print(f"📁 Path exists: {os.path.exists(model_path)}")
        
        if not os.path.exists(model_path):
            print(f"❌ Model path not found: {model_path}")
            print(f"📁 Current working directory: {os.getcwd()}")
            print(f"📁 Backend file location: {__file__}")
            print(f"📁 Parent directory: {os.path.dirname(os.path.dirname(__file__))}")
            raise FileNotFoundError(f"Model directory not found at {model_path}")
        
        print(f"📂 Model directory contents:")
        required_files = ["best_model.pt", "model_config.json", "special_tokens_map.json"]
        missing_files = []
        
        for item in os.listdir(model_path):
            item_path = os.path.join(model_path, item)
            if os.path.isfile(item_path):
                size_mb = os.path.getsize(item_path) / (1024 * 1024)
                print(f"   {item} ({size_mb:.1f} MB)")
            else:
                print(f"   {item} (directory)")
        
        # Check for required files
        for required_file in required_files:
            if not os.path.exists(os.path.join(model_path, required_file)):
                missing_files.append(required_file)
        
        if missing_files:
            raise FileNotFoundError(f"Missing required model files: {missing_files}")
        
        print("🔄 Loading CodeBERT model...")
        codebert_trainer.load_model(model_path)
        print("✅ CodeBERT model loaded successfully!")
        
        # Test the model with a simple prediction to ensure it works
        test_code = "void test() { char buf[10]; strcpy(buf, input); }"
        print("🧪 Testing model prediction...")
        
        test_result = codebert_trainer.predict(test_code)
        print(f"✅ Model test successful!")
        print(f"   Test prediction: {test_result['prediction']} ({'Vulnerable' if test_result['prediction'] == 1 else 'Safe'})")
        print(f"   Test confidence: {test_result['confidence']:.4f}")
        
        # Only set model_loaded to True after successful test
        model_loaded = True
        print("✅ Model successfully loaded and tested!")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to load CodeBERT model!")
        print(f"   Error: {str(e)}")
        print(f"   Type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        
        # Ensure model is marked as not loaded
        model_loaded = False
        codebert_trainer = None
        
        # Don't continue startup if model loading fails
        print("❌ Backend startup aborted due to model loading failure")
        print("   Please fix the model loading issue before starting the backend")
        raise e  # Re-raise the exception to prevent startup
    
    # Load attack pool (only after successful model loading)
    try:
        attack_pool_path = os.path.join(os.path.dirname(__file__), "attack_pool.csv")
        print(f"\n📊 Loading attack pool from: {attack_pool_path}")
        
        if os.path.exists(attack_pool_path):
            attack_pool_data = pd.read_csv(attack_pool_path)
            print(f"✅ Attack pool loaded with {len(attack_pool_data)} snippets")
            print(f"📋 Attack pool columns: {list(attack_pool_data.columns)}")
            
            # Verify the expected columns exist
            if 'adversarial_code' not in attack_pool_data.columns:
                print("⚠️ Warning: 'adversarial_code' column not found in attack pool")
                if 'original_code' in attack_pool_data.columns and len(attack_pool_data.columns) >= 2:
                    print("📝 Using the structure: original_code, adversarial_code, label")
                else:
                    print("🔧 Creating default attack pool due to unexpected format")
                    attack_pool_data = None
        else:
            print(f"❌ Attack pool file not found: {attack_pool_path}")
            attack_pool_data = None
            
        # Create a default attack pool if loading failed or file doesn't exist
        if attack_pool_data is None:
            # Create a default attack pool if none exists
            default_attack_pool = [
                "// This is a harmless comment that might confuse the model",
                "char buffer[100]; // Large enough buffer",
                "if(strlen(input) < sizeof(buffer)) strcpy(buffer, input);",
                "system(\"echo Hello\"); // Fixed command, not injection",
                "int *ptr = malloc(10); if(ptr != NULL) { free(ptr); ptr = NULL; }",
                "gets(buffer); // Known buffer overflow",
                "strcpy(dst, src); // No bounds checking",
                "system(user_input); // Command injection",
                "sprintf(query, \"SELECT * FROM users WHERE name='%s'\", user_input); // SQL injection"
            ]
            attack_pool_data = pd.DataFrame({"adversarial_code": default_attack_pool})
            default_pool_path = os.path.join(os.path.dirname(__file__), "attack_pool_default.csv")
            attack_pool_data.to_csv(default_pool_path, index=False)
            print(f"🔧 Created default attack pool with {len(attack_pool_data)} snippets at {default_pool_path}")
            
    except Exception as e:
        print(f"⚠️ Warning: Failed to load attack pool: {e}")
        # Don't fail startup for attack pool issues
        attack_pool_data = None
    
    print(f"\n🎉 Backend startup complete!")
    print(f"   Model loaded: {model_loaded}")
    print(f"   Attack pool loaded: {attack_pool_data is not None}")
    print("="*50)
    
    yield
    
    # Cleanup code (if needed)
    print("🔄 Shutting down backend...")

app = FastAPI(title="EatVul Security Analysis API", version="1.0.0", lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for tracking FGA progress
fga_progress = {
    "current_generation": 0,
    "max_generations": 0,
    "best_fitness": 0.0,
    "attack_success_rate": 0.0,
    "time_elapsed": 0.0,
    "estimated_time_remaining": 0.0,
    "status": "idle",
    "start_time": 0,
    "current_phase": "idle",
    "sub_progress": 0,
    "sub_total": 0,
    "phase_description": ""
}

# Pydantic models
class CodeAnalysisRequest(BaseModel):
    code: str
    language: str

class VulnerabilityResponse(BaseModel):
    is_vulnerable: bool
    confidence: float
    probabilities: List[float]
    explanation: str
    vulnerable_lines: List[Dict[str, Any]]
    extraction_info: Optional[Dict[str, Any]] = None  # Add extraction info

class AdversarialAttackRequest(BaseModel):
    original_code: str
    language: str
    attack_snippet: str
    original_prediction: Optional[Dict[str, Any]] = None  # Add original prediction from vulnerability analysis

class AttackPoolResponse(BaseModel):
    attack_snippets: List[str]
    best_snippet: str
    total_snippets: int

class FGAProgressResponse(BaseModel):
    current_generation: int
    max_generations: int
    best_fitness: float
    attack_success_rate: float
    time_elapsed: float
    estimated_time_remaining: float
    status: str
    current_phase: str
    sub_progress: int
    sub_total: int
    phase_description: str

class FGAParameters(BaseModel):
    pop_size: int = 20
    clusters: int = 3
    max_generations: int = 10  # Reduced from 50 for faster execution
    decay_rate: float = 1.5
    alpha: float = 2.0
    penalty: float = 0.001  # Reduced penalty for more reasonable fitness scores
    verbose: int = 1
    sample_size: int = 1  # Number of samples to use from attack pool (default: 1)

class FGAStartRequest(BaseModel):
    parameters: Optional[FGAParameters] = None
    extracted_function: Optional[str] = None  # Add extracted function

class FunctionExtractionRequest(BaseModel):
    code: str
    language: str

class FunctionExtractionResponse(BaseModel):
    functions: List[Dict[str, Any]]
    best_function: Dict[str, Any]
    summary: Dict[str, Any]
    extraction_successful: bool

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/diagnostic")
async def diagnostic():
    """Diagnostic endpoint to check system status"""
    global attack_pool_data, model_loaded, codebert_trainer
    
    # Check attack pool
    attack_pool_status = "loaded" if attack_pool_data is not None else "not_loaded"
    attack_pool_size = len(attack_pool_data) if attack_pool_data is not None else 0
    attack_pool_columns = list(attack_pool_data.columns) if attack_pool_data is not None else []
    
    # Check model
    model_status = "loaded" if model_loaded and codebert_trainer is not None else "not_loaded"
    
    # Check file paths
    attack_pool_path = os.path.join(os.path.dirname(__file__), "attack_pool.csv")
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "codebert-model")
    
    return {
        "status": "healthy",
        "attack_pool": {
            "status": attack_pool_status,
            "size": attack_pool_size,
            "columns": attack_pool_columns,
            "path": attack_pool_path,
            "path_exists": os.path.exists(attack_pool_path)
        },
        "model": {
            "status": model_status,
            "path": model_path,
            "path_exists": os.path.exists(model_path)
        },
        "files_in_backend": os.listdir(os.path.dirname(__file__)),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/analyze-vulnerability", response_model=VulnerabilityResponse)
async def analyze_vulnerability(request: CodeAnalysisRequest):
    """Analyze code for vulnerabilities using CodeBERT with C function extraction"""
    global codebert_trainer, model_loaded
    
    print(f"\n🔍 Analyzing vulnerability for {len(request.code)} characters of {request.language} code")
    print(f"📊 Model status: loaded={model_loaded}, trainer_exists={codebert_trainer is not None}")
    
    # Ensure model is loaded and available
    if not model_loaded or codebert_trainer is None:
        error_msg = "CodeBERT model is not loaded. Please check backend logs and restart the backend."
        print(f"❌ ERROR: {error_msg}")
        print(f"   Model loaded flag: {model_loaded}")
        print(f"   Trainer object exists: {codebert_trainer is not None}")
        raise HTTPException(
            status_code=503, 
            detail=f"Service Unavailable: {error_msg}"
        )
    
    try:
        # Determine if we need to extract C functions
        extraction_info = None
        code_to_analyze = request.code
        
        if request.language.lower() in ['c', 'cpp', 'c++']:
            print("🔧 Extracting C function for analysis...")
            
            try:
                # Extract the best function for analysis
                extracted_function, extraction_info = extract_c_function(request.code)
                code_to_analyze = extracted_function
                
                print(f"✅ Function extraction successful:")
                print(f"   Selected function: {extraction_info['selected_function']['name']}")
                print(f"   Extraction method: {extraction_info['selected_function'].get('extraction_method', 'unknown')}")
                print(f"   Total functions found: {extraction_info['total_extracted']}")
                print(f"   Function length: {len(code_to_analyze)} characters")
                
                # Log the selected function code (first 200 chars)
                preview = code_to_analyze[:200] + "..." if len(code_to_analyze) > 200 else code_to_analyze
                print(f"   Function preview: {preview}")
                
            except Exception as e:
                print(f"⚠️ Warning: Function extraction failed: {e}")
                print("   Falling back to full code analysis")
                extraction_info = {
                    'error': str(e),
                    'fallback': True,
                    'selected_function': {
                        'name': 'full_code_fallback',
                        'extraction_method': 'fallback_due_to_error'
                    }
                }
                # Use original code if extraction fails
                code_to_analyze = request.code
        else:
            print(f"🔍 Non-C language ({request.language}), using full code analysis")
            extraction_info = {
                'language': request.language,
                'extraction_method': 'full_code_non_c'
            }
        
        print("🤖 Using CodeBERT model for prediction...")
        print(f"📏 Code length for analysis: {len(code_to_analyze)} characters")
        
        # Get prediction from CodeBERT
        prediction_result = codebert_trainer.predict(code_to_analyze)
        
        is_vulnerable = bool(prediction_result["prediction"])
        confidence = float(prediction_result["confidence"])
        probabilities = [float(p) for p in prediction_result["probabilities"]]
        
        print(f"✅ CodeBERT prediction: vulnerable={is_vulnerable}, confidence={confidence:.4f}")
        print(f"📊 Probabilities: {probabilities}")
        
        # Get explanation from Gemini (using the analyzed code, not original)
        explanation, vulnerable_lines = await get_gemini_explanation(
            code_to_analyze, 
            is_vulnerable, 
            request.language
        )
        
        # Adjust vulnerable line numbers if we extracted a function
        if extraction_info and 'selected_function' in extraction_info:
            selected_func = extraction_info['selected_function']
            # If we have start_line information, adjust the line numbers
            if 'start_line' in selected_func and vulnerable_lines:
                function_start_line = selected_func['start_line']
                for line_info in vulnerable_lines:
                    if 'line_number' in line_info:
                        # Map from function line number to original code line number
                        original_line_number = line_info['line_number'] + function_start_line - 1
                        line_info['line_number'] = original_line_number
                        line_info['extracted_function_line'] = line_info['line_number'] - function_start_line + 1
                        
                print(f"🔄 Adjusted {len(vulnerable_lines)} vulnerable line numbers for function extraction")
                for line_info in vulnerable_lines:
                    print(f"   Original line {line_info.get('extracted_function_line', '?')} -> Full code line {line_info['line_number']}")
        
        # Validate response format before returning
        if not isinstance(vulnerable_lines, list):
            print(f"Warning: vulnerable_lines is not a list, got: {type(vulnerable_lines)}")
            vulnerable_lines = []
        
        # Ensure all items in vulnerable_lines are dictionaries
        validated_lines = []
        for item in vulnerable_lines:
            if isinstance(item, dict):
                validated_lines.append(item)
            else:
                print(f"Warning: Invalid vulnerable line item: {item}")
                # Convert to proper format
                validated_lines.append({
                    "line_number": 1,
                    "code": str(item) if item else "Unknown",
                    "vulnerability_type": "Validation Error",
                    "reason": "Item was not in expected format",
                    "fix_suggestion": "Please check the analysis manually"
                })
        
        return VulnerabilityResponse(
            is_vulnerable=is_vulnerable,
            confidence=confidence,
            probabilities=probabilities,
            explanation=explanation,
            vulnerable_lines=validated_lines,
            extraction_info=extraction_info
        )
        
    except Exception as e:
        print(f"❌ Error during CodeBERT prediction: {str(e)}")
        print(f"📋 Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

async def get_gemini_explanation(code: str, is_vulnerable: bool, language: str) -> tuple:
    """Get explanation and vulnerable lines from Gemini API"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
        
        if is_vulnerable:
            prompt = f"""
            Analyze this {language} code for security vulnerabilities. Provide a concise analysis.
            
            Code:
            ```{language}
            {code}
            ```
            
            Respond with ONLY valid JSON in this format:
            {{
                "summary": "Brief 1-2 sentence summary of vulnerabilities found",
                "vulnerability_types": ["type1", "type2"],
                "severity": "high|medium|low",
                "vulnerable_lines": [
                    {{
                        "line_number": 1,
                        "code": "vulnerable code snippet",
                        "vulnerability_type": "buffer overflow",
                        "reason": "Brief explanation why this is vulnerable",
                        "fix_suggestion": "How to fix this issue"
                    }}
                ],
                "recommendations": ["Brief recommendation 1", "Brief recommendation 2"]
            }}
            """
        else:
            prompt = f"""
            Analyze this {language} code. It appears secure. Provide a brief analysis.
            
            Code:
            ```{language}
            {code}
            ```
            
            Respond with ONLY valid JSON in this format:
            {{
                "summary": "Brief explanation of what the code does and why it's secure",
                "security_practices": ["Good practice 1", "Good practice 2"],
                "severity": "none",
                "vulnerable_lines": [],
                "recommendations": ["Suggestion for improvement 1"]
            }}
            """
        
        response = model.generate_content(prompt)
        
        # Parse and format the response
        try:
            response_text = response.text.strip()
            
            # Extract JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                parsed_response = json.loads(json_text)
                
                # Format the explanation nicely
                explanation = format_analysis_explanation(parsed_response, is_vulnerable)
                
                # Process vulnerable lines
                vulnerable_lines = process_vulnerable_lines(parsed_response.get("vulnerable_lines", []))
                
                return explanation, vulnerable_lines
            else:
                # Fallback parsing for non-JSON responses
                return parse_non_json_response(response_text, is_vulnerable)
                
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return parse_non_json_response(response.text, is_vulnerable)
            
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        return create_fallback_response(is_vulnerable)

def format_analysis_explanation(parsed_response: dict, is_vulnerable: bool) -> str:
    """Format the parsed Gemini response into a readable explanation"""
    explanation_parts = []
    
    # Add summary
    summary = parsed_response.get("summary", "")
    if summary:
        explanation_parts.append(f"🔍 **Analysis Summary:**\n{summary}")
    
    if is_vulnerable:
        # Add vulnerability info
        vuln_types = parsed_response.get("vulnerability_types", [])
        if vuln_types:
            types_str = ", ".join(vuln_types)
            explanation_parts.append(f"⚠️ **Vulnerability Types:** {types_str}")
        
        severity = parsed_response.get("severity", "unknown")
        severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(severity, "⚪")
        explanation_parts.append(f"{severity_emoji} **Severity:** {severity.title()}")
        
        # Add recommendations
        recommendations = parsed_response.get("recommendations", [])
        if recommendations:
            rec_text = "\n".join([f"• {rec}" for rec in recommendations[:3]])  # Limit to 3 recommendations
            explanation_parts.append(f"💡 **Recommendations:**\n{rec_text}")
    else:
        # Add security practices for secure code
        practices = parsed_response.get("security_practices", [])
        if practices:
            practices_text = "\n".join([f"• {practice}" for practice in practices[:3]])
            explanation_parts.append(f"✅ **Good Security Practices:**\n{practices_text}")
        
        recommendations = parsed_response.get("recommendations", [])
        if recommendations:
            rec_text = "\n".join([f"• {rec}" for rec in recommendations[:2]])
            explanation_parts.append(f"💡 **Suggestions:**\n{rec_text}")
    
    return "\n\n".join(explanation_parts)

def process_vulnerable_lines(vulnerable_lines_raw: list) -> list:
    """Process and validate vulnerable lines from Gemini response"""
    vulnerable_lines = []
    
    if not isinstance(vulnerable_lines_raw, list):
        return vulnerable_lines
    
    for item in vulnerable_lines_raw:
        if isinstance(item, dict):
            # Clean and validate the item
            cleaned_item = {
                "line_number": int(item.get("line_number", 1)),
                "code": str(item.get("code", "Unknown code"))[:100],  # Limit code length
                "vulnerability_type": str(item.get("vulnerability_type", "Unknown"))[:50],
                "reason": str(item.get("reason", "No reason provided"))[:200],  # Limit reason length
                "fix_suggestion": str(item.get("fix_suggestion", "No suggestion"))[:200]
            }
            vulnerable_lines.append(cleaned_item)
        elif isinstance(item, str):
            # Convert string to proper format
            vulnerable_lines.append({
                "line_number": 1,
                "code": str(item)[:100],
                "vulnerability_type": "Potential Issue",
                "reason": "Flagged by analysis",
                "fix_suggestion": "Review this code for security issues"
            })
    
    return vulnerable_lines[:5]  # Limit to 5 vulnerable lines max

def parse_non_json_response(response_text: str, is_vulnerable: bool) -> tuple:
    """Parse non-JSON response from Gemini and extract useful information"""
    
    # Try to extract key information from the text
    lines = response_text.split('\n')
    summary_lines = []
    
    # Look for key phrases and extract meaningful sentences
    for line in lines:
        line = line.strip()
        if line and len(line) > 20:  # Skip very short lines
            # Look for sentences that contain key security terms
            if any(term in line.lower() for term in ['vulnerability', 'security', 'buffer', 'overflow', 'injection', 'unsafe']):
                summary_lines.append(line)
                if len(summary_lines) >= 3:  # Limit to 3 key sentences
                    break
    
    if summary_lines:
        explanation = "🔍 **Key Findings:**\n" + "\n".join([f"• {line}" for line in summary_lines])
    else:
        # Fallback to first few sentences
        first_sentences = response_text[:300] + "..." if len(response_text) > 300 else response_text
        explanation = f"📝 **Analysis:**\n{first_sentences}"
    
    # Create basic vulnerable lines if this is a vulnerable analysis
    vulnerable_lines = []
    if is_vulnerable:
        vulnerable_lines = [{
            "line_number": 1,
            "code": "Detected by analysis",
            "vulnerability_type": "Security Issue",
            "reason": "Code flagged as potentially vulnerable",
            "fix_suggestion": "Review code for security best practices"
        }]
    
    return explanation, vulnerable_lines

def create_fallback_response(is_vulnerable: bool, response_text: str = None) -> tuple:
    """Create a fallback response when Gemini parsing fails"""
    if is_vulnerable:
        explanation = "⚠️ **Vulnerability Detected**\n\nAutomated analysis flagged potential security issues in this code. Manual review recommended."
        vulnerable_lines = [{
            "line_number": 1,
            "code": "Analysis incomplete",
            "vulnerability_type": "Potential Security Issue",
            "reason": "Automated detection flagged this code",
            "fix_suggestion": "Please review the code manually for security issues"
        }]
    else:
        explanation = "✅ **Code Appears Secure**\n\nNo obvious security vulnerabilities detected in the analysis."
        vulnerable_lines = []
    
    return explanation, vulnerable_lines

@app.get("/attack-pool", response_model=AttackPoolResponse)
async def get_attack_pool():
    """Get available attack snippets from the attack pool"""
    global attack_pool_data
    
    if attack_pool_data is None:
        raise HTTPException(status_code=503, detail="Attack pool not loaded")
    
    try:
        # Handle both old and new CSV formats
        if "adversarial_code" in attack_pool_data.columns:
            # New format with adversarial_code column
            snippets = attack_pool_data["adversarial_code"].tolist()
        elif len(attack_pool_data.columns) >= 2:
            # Assume second column contains adversarial code
            snippets = attack_pool_data.iloc[:, 1].tolist()  # Second column
        else:
            # Fallback: use first column
            snippets = attack_pool_data.iloc[:, 0].tolist()
        
        # Clean up snippets - remove any NaN values
        snippets = [str(snippet) for snippet in snippets if pd.notna(snippet)]
        
        # For now, return the first snippet as "best" - this will be updated by FGA
        best_snippet = snippets[0] if snippets else ""
        
        print(f"Returning {len(snippets)} attack snippets")
        print(f"Sample snippet: {snippets[0][:100] if snippets else 'None'}...")
        
        return AttackPoolResponse(
            attack_snippets=snippets,
            best_snippet=best_snippet,
            total_snippets=len(snippets)
        )
        
    except Exception as e:
        print(f"Error in get_attack_pool: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get attack pool: {str(e)}")

@app.get("/fga-parameters")
async def get_fga_parameters():
    """Get default FGA parameters"""
    default_params = FGAParameters()
    return {
        "default_parameters": default_params.dict(),
        "parameter_descriptions": {
            "pop_size": "Population size for genetic algorithm (10-100)",
            "clusters": "Number of fuzzy clusters (2-10)",
            "max_generations": "Maximum number of generations (10-100)",
            "decay_rate": "Decay rate for fuzzy clustering (0.5-3.0)",
            "alpha": "Fuzziness factor (1.0-5.0)",
            "penalty": "Penalty factor for code snippet length (0.001-0.1)",
            "verbose": "Verbosity level (0-2)",
            "sample_size": "Number of samples to use from attack pool (default: 1)"
        },
        "recommended_ranges": {
            "pop_size": {"min": 10, "max": 100, "default": 20},
            "clusters": {"min": 2, "max": 10, "default": 3},
            "max_generations": {"min": 10, "max": 100, "default": 10},
            "decay_rate": {"min": 0.5, "max": 3.0, "default": 1.5},
            "alpha": {"min": 1.0, "max": 5.0, "default": 2.0},
            "penalty": {"min": 0.001, "max": 0.1, "default": 0.001},
            "verbose": {"min": 0, "max": 2, "default": 1},
            "sample_size": {"min": 1, "max": 100, "default": 1}
        }
    }

@app.post("/start-fga-selection")
async def start_fga_selection(request: FGAStartRequest, background_tasks: BackgroundTasks):
    """Start simplified FGA selection process to find the best attack snippet"""
    global fga_progress, attack_pool_data
    
    if attack_pool_data is None:
        raise HTTPException(status_code=503, detail="Attack pool not loaded")
    
    # Use provided parameters or defaults
    params = request.parameters if request.parameters else FGAParameters()
    
    # Reset progress with custom parameters
    fga_progress.update({
        "current_generation": 0,
        "max_generations": len(attack_pool_data),  # Use attack pool size
        "best_fitness": 0.0,
        "attack_success_rate": 0.0,
        "time_elapsed": 0.0,
        "estimated_time_remaining": 0.0,
        "status": "running",
        "start_time": time.time(),
        "parameters": params.dict(),
        "current_phase": "initializing",
        "phase_description": "Starting FGA selection",
        "sub_progress": 0,
        "sub_total": 100
    })
    
    # Start simplified FGA in background
    background_tasks.add_task(run_simplified_fga_selection, params, request.extracted_function)
    
    return {
        "message": "Simplified FGA selection started", 
        "status": "running",
        "parameters": params.dict(),
        "mode": "simplified"
    }

async def run_simplified_fga_selection(params: FGAParameters, extracted_function: str = None):
    """Run proper FGA selection using fuzzy clustering and genetic algorithm"""
    global fga_progress, attack_pool_data, codebert_trainer, best_attack_snippet_result
    
    import time
    start_time = time.time()
    
    try:
        print("Starting FGA selection with fuzzy clustering and genetic algorithm...")
        
        # Get attack pool snippets
        if attack_pool_data is None:
            raise Exception("Attack pool not loaded")
            
        # Extract snippets from attack pool
        if "adversarial_code" in attack_pool_data.columns:
            attack_snippets_series = attack_pool_data["adversarial_code"].dropna()
            attack_snippets = attack_snippets_series.tolist()
        elif len(attack_pool_data.columns) >= 2:
            attack_snippets_series = attack_pool_data.iloc[:, 1].dropna()
            attack_snippets = attack_snippets_series.tolist()
        else:
            attack_snippets_series = attack_pool_data.iloc[:, 0].dropna()
            attack_snippets = attack_snippets_series.tolist()
        
        attack_snippets = [str(snippet) for snippet in attack_snippets if snippet and str(snippet).strip()]
        
        if not attack_snippets:
            raise Exception("No valid attack snippets found in attack pool")
        
        print(f"📊 FGA Configuration:")
        print(f"   🎯 Population size: {params.pop_size}")
        print(f"   🧬 Generations: {params.max_generations}")
        print(f"   📦 Attack pool size: {len(attack_snippets)}")
        print(f"   ⚙️ Penalty factor: {params.penalty}")
        
        # Use provided extracted function or create a simple test function
        if not extracted_function:
            extracted_function = """void test_function() {
    char buffer[100];
    // Test function for vulnerability analysis
}"""
            print(f"🔧 Using default test function")
        else:
            print(f"🔧 Using extracted function ({len(extracted_function)} chars)")
            print(f"   📝 Function preview: {extracted_function[:100]}...")
        
        print(f"Initializing FGA with {len(attack_snippets)} attack snippets...")
        
        # Initialize FGA components from fga_selection.py
        from fga_selection import centriod_init, calcaulate_weight, select, update_global_pop
        
        # Initialize population with attack snippets and fitness scores
        population = {}
        print("Calculating initial fitness scores for population...")
        
        population_size = min(params.pop_size, len(attack_snippets))
        
        # Update progress for initialization phase
        fga_progress.update({
            "current_phase": "initialization",
            "phase_description": "Initializing population with attack snippets",
            "sub_progress": 0,
            "sub_total": population_size,
            "time_elapsed": time.time() - start_time
        })
        
        # Calculate fitness for initial population
        for i, snippet in enumerate(attack_snippets[:params.pop_size]):
            try:
                fga_progress.update({
                    "current_generation": 0,
                    "current_phase": "initialization",
                    "phase_description": f"Initializing population {i+1}/{population_size}",
                    "sub_progress": i + 1,
                    "sub_total": population_size,
                    "time_elapsed": time.time() - start_time
                })
                
                # Inject snippet into function and get fitness
                injected_code = inject_attack_snippet(extracted_function, snippet)
                
                if codebert_trainer and model_loaded:
                    # Get original prediction for extracted function
                    original_prediction = codebert_trainer.predict(extracted_function)
                    # Get prediction with injected attack
                    attack_prediction = codebert_trainer.predict(injected_code)
                    
                    if isinstance(original_prediction, dict) and isinstance(attack_prediction, dict):
                        orig_vuln = original_prediction.get('prediction', 0) == 1
                        attack_vuln = attack_prediction.get('prediction', 0) == 1
                        
                        orig_probs = original_prediction.get('probabilities', [0.5, 0.5])
                        attack_probs = attack_prediction.get('probabilities', [0.5, 0.5])
                        
                        # Attack success: originally vulnerable → becomes safe
                        attack_success = orig_vuln and not attack_vuln
                        
                        vulnerability_score = attack_probs[1]  # Probability of vulnerable
                        
                        # Calculate fitness: higher if attack makes it seem safe when originally vulnerable
                        if attack_success:
                            # Successful attack - high fitness based on confidence of making it safe
                            base_fitness = 1.0 - vulnerability_score  # Higher fitness when attack makes it look safer
                            # Apply length penalty as a percentage, not absolute
                            length_penalty = params.penalty * (len(snippet) / 100.0)  # Normalize penalty
                            fitness = base_fitness - length_penalty
                        else:
                            # Failed attack - very low fitness
                            fitness = 0.1  # Small positive value to avoid complete elimination
                        
                        population[snippet] = max(0.0, fitness)  # Ensure non-negative fitness
                        
                        print(f"🔍 Population {i+1}/{population_size} - Snippet: {snippet[:50]}...")
                        print(f"   📊 Original: vuln={orig_vuln}, probs={[f'{p:.3f}' for p in orig_probs]}, conf={original_prediction.get('confidence', 0):.3f}")
                        print(f"   📊 Attack:   vuln={attack_vuln}, probs={[f'{p:.3f}' for p in attack_probs]}, conf={attack_prediction.get('confidence', 0):.3f}")
                        print(f"   ✅ Attack Success: {attack_success}, Base Fitness: {base_fitness:.4f}, Final Fitness: {fitness:.4f}")
                        
                    else:
                        print(f"⚠️ Population {i+1}: Invalid prediction format")
                        population[snippet] = 0.0
                else:
                    print(f"⚠️ Population {i+1}: Model not available")
                    population[snippet] = 0.0
                        
            except Exception as e:
                print(f"Error initializing population member {i+1}: {str(e)}")
                population[snippet] = 0.0
        
        # Initialize fuzzy clustering centroids
        try:
            centroids = centriod_init(params.clusters, 0.1)  # min_distance = 0.1
            print(f"Initialized {params.clusters} centroids: {centroids}")
        except ValueError as e:
            print(f"Warning: {e}. Using default centroids.")
            centroids = np.linspace(0.1, 0.9, params.clusters)
        
        best_fitness = 0.0
        best_snippet = ""
        best_attack_rate = 0.0
        
        # Main FGA evolution loop
        for generation in range(params.max_generations):
            generation_start_time = time.time()
            
            # Calculate estimated time remaining
            if generation > 0:
                avg_time_per_generation = (time.time() - start_time) / generation
                estimated_remaining = avg_time_per_generation * (params.max_generations - generation)
            else:
                estimated_remaining = 0.0
            
            fga_progress.update({
                "current_generation": generation + 1,
                "max_generations": params.max_generations,
                "current_phase": "evolution",
                "phase_description": f"Generation {generation+1}/{params.max_generations}",
                "sub_progress": 0,
                "sub_total": 100,
                "time_elapsed": time.time() - start_time,
                "estimated_time_remaining": estimated_remaining
            })
            
            print(f"\n=== Generation {generation+1}/{params.max_generations} ===")
            
            # Get current population fitness values
            fitness_values = list(population.values())
            if not fitness_values:
                    break
        
            current_best_fitness = max(fitness_values)
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                # Find the snippet with best fitness and calculate actual attack success rate
                total_attacks = 0
                successful_attacks = 0
                
                for snippet, fitness in population.items():
                    if fitness == best_fitness:
                        best_snippet = snippet
                        
                        # Test this snippet to see if it's actually a successful attack
                        try:
                            injected_code = inject_attack_snippet(extracted_function, snippet)
                            original_pred = codebert_trainer.predict(extracted_function)
                            attack_pred = codebert_trainer.predict(injected_code)
                            
                            if isinstance(original_pred, dict) and isinstance(attack_pred, dict):
                                orig_vuln = original_pred.get('prediction', 0) == 1
                                attack_vuln = attack_pred.get('prediction', 0) == 1
                                attack_success = orig_vuln and not attack_vuln
                                
                                orig_probs = original_pred.get('probabilities', [0.5, 0.5])
                                attack_probs = attack_pred.get('probabilities', [0.5, 0.5])
                                
                                print(f"🏆 Best fitness snippet test:")
                                print(f"   📊 Original: vuln={orig_vuln}, probs={[f'{p:.3f}' for p in orig_probs]}")
                                print(f"   📊 Attack:   vuln={attack_vuln}, probs={[f'{p:.3f}' for p in attack_probs]}")
                                print(f"   ✅ Attack Success: {attack_success}")
                                
                                if attack_success:
                                    successful_attacks += 1
                                total_attacks += 1
                        except:
                            pass
                        break
                
                # Calculate attack success rate for the population
                if total_attacks > 0:
                    best_attack_rate = successful_attacks / total_attacks
                else:
                    best_attack_rate = 0.0
                        
                fga_progress["best_fitness"] = best_fitness
                fga_progress["attack_success_rate"] = best_attack_rate
            
            print(f"Generation {generation+1} best fitness: {current_best_fitness:.4f}")
            print(f"Overall best fitness: {best_fitness:.4f}")
            print(f"Population size: {len(population)}, Average fitness: {sum(population.values())/len(population):.4f}")
            print(f"Attack success rate: {best_attack_rate:.1%} ({best_attack_rate:.3f})")
            
            # Update sub-progress for parent selection
            fga_progress.update({
                "phase_description": f"Gen {generation+1}: Selecting parents",
                "sub_progress": 25,
                "sub_total": 100
            })
            
            # Select parents using fuzzy selection
            parents = []
            for _ in range(params.pop_size // 2):  # Select half the population as parents
                try:
                    # Use random centroid for selection
                    centroid = np.random.choice(centroids)
                    parent = select(population, centroid, centroids, params.decay_rate)
                    parents.append(parent)
                except Exception as e:
                    # Fallback to random selection if fuzzy selection fails
                    parent = np.random.choice(list(population.keys()))
                    parents.append(parent)
            
            # Update sub-progress for crossover
            fga_progress.update({
                "phase_description": f"Gen {generation+1}: Creating offspring",
                "sub_progress": 50,
                "sub_total": 100
            })
            
            # Generate offspring through crossover and mutation
            offspring = []
            for i in range(0, len(parents), 2):
                try:
                    if i + 1 < len(parents):
                        # Simple crossover: combine parts of two parents
                        parent1, parent2 = parents[i], parents[i+1]
                        
                        # Create offspring by combining parent snippets
                        if len(parent1) > 10 and len(parent2) > 10:
                            crossover_point = min(len(parent1), len(parent2)) // 2
                            child1 = parent1[:crossover_point] + parent2[crossover_point:]
                            child2 = parent2[:crossover_point] + parent1[crossover_point:]
                        else:
                            child1 = parent1 + " " + parent2[:10]  # Simple concatenation
                            child2 = parent2 + " " + parent1[:10]
                        
                        offspring.extend([child1, child2])
                    else:
                        # Single parent mutation
                        mutated = parents[i] + " // mutated"
                        offspring.append(mutated)
                        
                except Exception as e:
                    print(f"Error in crossover: {e}")
                    # Fallback: just add parent as offspring
                    offspring.append(parents[i] if i < len(parents) else list(population.keys())[0])
            
            # Update sub-progress for fitness evaluation
            fga_progress.update({
                "phase_description": f"Gen {generation+1}: Evaluating offspring",
                "sub_progress": 75,
                "sub_total": 100
            })
            
            # Calculate fitness for offspring
            offspring_fitness = dict()  # Use dict() instead of {} to avoid linter issue
            offspring_len = len(offspring)
            pop_half = params.pop_size // 2
            offspring_count = min(offspring_len, pop_half)
            for j, child in enumerate(offspring[:offspring_count]):
                try:
                    injected_code = inject_attack_snippet(extracted_function, child)
                    
                    if codebert_trainer and model_loaded:
                        # Get original prediction for extracted function
                        original_prediction = codebert_trainer.predict(extracted_function)
                        # Get prediction with injected attack
                        attack_prediction = codebert_trainer.predict(injected_code)
                        
                        if isinstance(original_prediction, dict) and isinstance(attack_prediction, dict):
                            orig_vuln = original_prediction.get('prediction', 0) == 1
                            attack_vuln = attack_prediction.get('prediction', 0) == 1
                            
                            orig_probs = original_prediction.get('probabilities', [0.5, 0.5])
                            attack_probs = attack_prediction.get('probabilities', [0.5, 0.5])
                            
                            # Attack success: originally vulnerable → becomes safe
                            attack_success = orig_vuln and not attack_vuln
                            
                            vulnerability_score = attack_probs[1]  # Probability of vulnerable
                            
                            # Calculate fitness: higher if attack makes it seem safe when originally vulnerable
                            if attack_success:
                                # Successful attack - high fitness based on confidence of making it safe
                                base_fitness = 1.0 - vulnerability_score  # Higher fitness when attack makes it look safer
                                # Apply length penalty as a percentage, not absolute
                                length_penalty = params.penalty * (len(child) / 100.0)  # Normalize penalty
                                fitness = base_fitness - length_penalty
                            else:
                                # Failed attack - very low fitness
                                fitness = 0.1  # Small positive value to avoid complete elimination
                            
                            offspring_fitness[child] = max(0.0, fitness)
                            
                            print(f"🧬 Gen {generation+1} Offspring {j+1}/{offspring_count} - Child: {child[:30]}...")
                            print(f"   📊 Original: vuln={orig_vuln}, probs={[f'{p:.3f}' for p in orig_probs]}")
                            print(f"   📊 Attack:   vuln={attack_vuln}, probs={[f'{p:.3f}' for p in attack_probs]}")
                            print(f"   ✅ Success: {attack_success}, Fitness: {fitness:.4f}")
                        else:
                            offspring_fitness[child] = 0.0
                            print(f"⚠️ Gen {generation+1} Offspring {j+1}: Invalid prediction format")
                    else:
                        offspring_fitness[child] = 0.0
                        
                except Exception as e:
                    print(f"Error evaluating offspring {j+1}: {e}")
                    offspring_fitness[child] = 0.0
            
            # Update sub-progress for population update
            fga_progress.update({
                "phase_description": f"Gen {generation+1}: Updating population",
                "sub_progress": 100,
                "sub_total": 100
            })
            
            # Update population using survival of the fittest
            try:
                population = update_global_pop(list(offspring_fitness.keys()), population, list(offspring_fitness.values()))
            except Exception as e:
                    print(f"Error updating population: {e}")
                    # Fallback: just add best offspring to population
                    if offspring_fitness:
                        best_item = max(offspring_fitness.items(), key=lambda x: x[1])
                        best_offspring = best_item
                        population[best_offspring[0]] = best_offspring[1]
        
        # Store results
        best_attack_snippet_result = {
            "best_adversarial_code": best_snippet,
            "best_fitness": best_fitness,
            "attack_success_rate": best_attack_rate,
            "parameters": params.dict(),
            "total_generations": params.max_generations,
            "final_population_size": len(population),
            "total_time": time.time() - start_time
        }
        
        # Update final progress
        fga_progress.update({
            "status": "completed",
            "current_generation": params.max_generations,
            "current_phase": "completed",
            "phase_description": "FGA selection completed successfully",
            "sub_progress": 100,
            "sub_total": 100,
            "time_elapsed": time.time() - start_time,
            "estimated_time_remaining": 0.0
        })
        
        print(f"\nFGA completed! Best snippet fitness: {best_fitness:.4f}, Attack rate: {best_attack_rate:.1f}")
        print(f"Total time: {time.time() - start_time:.2f}s")
        print(f"Best snippet: {best_snippet[:100]}..." if best_snippet else "No effective snippet found")
        
        # Final test of the best snippet to confirm results
        if best_snippet and codebert_trainer and model_loaded:
            try:
                print(f"\n🔬 Final verification of best snippet:")
                injected_code = inject_attack_snippet(extracted_function, best_snippet)
                final_orig = codebert_trainer.predict(extracted_function)
                final_attack = codebert_trainer.predict(injected_code)
                
                if isinstance(final_orig, dict) and isinstance(final_attack, dict):
                    final_orig_vuln = final_orig.get('prediction', 0) == 1
                    final_attack_vuln = final_attack.get('prediction', 0) == 1
                    final_success = final_orig_vuln and not final_attack_vuln
                    
                    final_orig_probs = final_orig.get('probabilities', [0.5, 0.5])
                    final_attack_probs = final_attack.get('probabilities', [0.5, 0.5])
                    
                    print(f"   📊 Final Original: vuln={final_orig_vuln}, probs={[f'{p:.3f}' for p in final_orig_probs]}")
                    print(f"   📊 Final Attack:   vuln={final_attack_vuln}, probs={[f'{p:.3f}' for p in final_attack_probs]}")
                    print(f"   🎯 Final Attack Success: {final_success}")
                    print(f"   📈 Confidence Change: {final_orig_probs[1]:.3f} → {final_attack_probs[1]:.3f}")
            except Exception as e:
                print(f"   ⚠️ Final verification failed: {e}")
            
    except Exception as e:
        print(f"Error in FGA selection: {str(e)}")
        import traceback
        traceback.print_exc()
        fga_progress.update({
            "status": "error",
            "phase_description": f"Error: {str(e)}",
            "time_elapsed": time.time() - start_time,
            "estimated_time_remaining": 0.0
        })

@app.get("/fga-progress", response_model=FGAProgressResponse)
async def get_fga_progress():
    """Get the current progress of FGA selection"""
    global fga_progress
    
    return FGAProgressResponse(**fga_progress)

@app.get("/best-attack-snippet")
async def get_best_attack_snippet():
    """Get the best attack snippet found by FGA"""
    global attack_pool_data, fga_progress, best_attack_snippet_result
    
    if attack_pool_data is None:
        raise HTTPException(status_code=503, detail="Attack pool not loaded")
    
    try:
        # Return actual FGA results if available
        if best_attack_snippet_result and fga_progress["status"] == "completed":
            best_snippet = best_attack_snippet_result.get("best_adversarial_code", "")
            if best_snippet:
                return {
                    "best_snippet": best_snippet,
                    "fitness_score": best_attack_snippet_result.get("best_fitness", 0.0),
                    "attack_success_rate": best_attack_snippet_result.get("attack_success_rate", 0.0),
                    "status": fga_progress["status"],
                    "parameters": best_attack_snippet_result.get("parameters", {}),
                    "source": "fga_algorithm"
                }
        
        # Fallback: Return a snippet from the attack pool if FGA hasn't completed yet
        if fga_progress["status"] == "running":
            # Return a default snippet from the attack pool while FGA is running
            if "adversarial_code" in attack_pool_data.columns:
                snippets_series = attack_pool_data["adversarial_code"].dropna()
                snippets = snippets_series.tolist()
            elif len(attack_pool_data.columns) >= 2:
                snippets_series = attack_pool_data.iloc[:, 1].dropna()
                snippets = snippets_series.tolist()
            else:
                snippets_series = attack_pool_data.iloc[:, 0].dropna()
                snippets = snippets_series.tolist()
            
            best_snippet = snippets[0] if snippets else "// Default attack snippet"
            
            return {
                "best_snippet": best_snippet,
                "fitness_score": fga_progress["best_fitness"],
                "attack_success_rate": fga_progress["attack_success_rate"],
                "status": fga_progress["status"],
                "source": "attack_pool_default"
            }
        
        # If FGA failed or no results available, return a sophisticated fallback
        if fga_progress["status"] == "error" or not best_attack_snippet_result:
            # Return a sophisticated attack snippet as fallback
            best_snippet = """// Advanced buffer overflow attack
char *overflow_ptr = malloc(8); 
if(overflow_ptr) { 
    strcpy(overflow_ptr, "this_string_is_longer_than_8_bytes"); 
    free(overflow_ptr); 
}"""
            
            return {
                "best_snippet": best_snippet,
                "fitness_score": fga_progress.get("best_fitness", 0.0),
                "attack_success_rate": fga_progress.get("attack_success_rate", 0.0),
                "status": fga_progress["status"],
                "source": "fallback"
            }
        
        # Default case - FGA not started yet
        if "adversarial_code" in attack_pool_data.columns:
            snippets_series = attack_pool_data["adversarial_code"].dropna()
            snippets = snippets_series.tolist()
        elif len(attack_pool_data.columns) >= 2:
            snippets_series = attack_pool_data.iloc[:, 1].dropna()
            snippets = snippets_series.tolist()
        else:
            snippets_series = attack_pool_data.iloc[:, 0].dropna()
            snippets = snippets_series.tolist()
        
        best_snippet = snippets[0] if snippets else "// Default attack snippet"
        
        return {
            "best_snippet": best_snippet,
            "fitness_score": 0.0,
            "attack_success_rate": 0.0,
            "status": "idle",
            "source": "attack_pool_initial"
        }
        
    except Exception as e:
        print(f"Error in get_best_attack_snippet: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get best attack snippet: {str(e)}")

@app.post("/adversarial-attack")
async def perform_adversarial_attack(request: AdversarialAttackRequest):
    """Perform adversarial attack by injecting attack snippet into code"""
    global codebert_trainer, model_loaded
    
    if not model_loaded or codebert_trainer is None:
        raise HTTPException(status_code=503, detail="CodeBERT model not loaded")
    
    try:
        # Use provided original prediction or get it by predicting
        if request.original_prediction:
            original_prediction = request.original_prediction
            print("Using provided original prediction from vulnerability analysis")
        else:
            original_prediction = codebert_trainer.predict(request.original_code)
            print("Re-predicting original code (no stored prediction provided)")
        
        # Create adversarial code by injecting attack snippet
        adversarial_code = inject_attack_snippet(request.original_code, request.attack_snippet)
        
        # Get adversarial prediction
        adversarial_prediction = codebert_trainer.predict(adversarial_code)
        
        # Attack success: originally vulnerable (1) → becomes safe (0)
        original_vulnerable = original_prediction.get("prediction", 0) == 1
        adversarial_vulnerable = adversarial_prediction.get("prediction", 0) == 1
        
        attack_success = original_vulnerable and not adversarial_vulnerable
        
        print(f"Attack analysis: Original={original_vulnerable}, Adversarial={adversarial_vulnerable}, Success={attack_success}")
        
        return {
            "original_prediction": {
                "is_vulnerable": bool(original_prediction.get("prediction", 0)),
                "confidence": float(original_prediction.get("confidence", 0.5)),
                "probabilities": [float(p) for p in original_prediction.get("probabilities", [0.5, 0.5])]
            },
            "adversarial_prediction": {
                "is_vulnerable": bool(adversarial_prediction["prediction"]),
                "confidence": float(adversarial_prediction["confidence"]),
                "probabilities": [float(p) for p in adversarial_prediction["probabilities"]]
            },
            "adversarial_code": adversarial_code,
            "attack_success": attack_success,
            "injection_info": get_injection_info(request.original_code, request.attack_snippet)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Adversarial attack failed: {str(e)}")

def inject_attack_snippet(original_code: str, attack_snippet: str) -> str:
    """Inject attack snippet into original code"""
    lines = original_code.split('\n')
    
    # Find a good insertion point (after function declaration, before main logic)
    insertion_point = 1  # Default to line 2
    
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        # Look for function declarations
        if any(keyword in stripped for keyword in ['void ', 'int ', 'char ', 'function ']) and '{' in stripped:
            insertion_point = i + 1
            break
        # Or variable declarations
        elif '=' in stripped:
            has_type_keyword = any(keyword in stripped for keyword in ['char ', 'int ', 'float ', 'double '])
            if has_type_keyword:
                insertion_point = i + 1
                break
    
    # Insert attack snippet
    lines.insert(insertion_point, f"    {attack_snippet}  // INJECTED_ATTACK_CODE")
    
    return '\n'.join(lines)

def get_injection_info(original_code: str, attack_snippet: str) -> dict:
    """Get information about where the attack snippet was injected"""
    lines = original_code.split('\n')
    
    # Find insertion point (same logic as inject_attack_snippet)
    insertion_point = 1
    
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if any(keyword in stripped for keyword in ['void ', 'int ', 'char ', 'function ']) and '{' in stripped:
            insertion_point = i + 1
            break
        elif '=' in stripped:
            has_type_keyword = any(keyword in stripped for keyword in ['char ', 'int ', 'float ', 'double '])
            if has_type_keyword:
                insertion_point = i + 1
                break
    
    return {
        "injection_line": insertion_point + 1,  # 1-based line number
        "injection_position": "after_function_declaration",
        "attack_snippet": attack_snippet,
        "marker": "// INJECTED_ATTACK_CODE"
    }

@app.post("/extract-functions", response_model=FunctionExtractionResponse)
async def extract_functions(request: FunctionExtractionRequest):
    """Extract and analyze C functions from code"""
    
    if request.language.lower() not in ['c', 'cpp', 'c++']:
        raise HTTPException(
            status_code=400, 
            detail=f"Function extraction only supported for C/C++ code, got: {request.language}"
        )
    
    try:
        print(f"🔧 Extracting functions from {len(request.code)} characters of {request.language} code")
        
        parser = CFunctionParser()
        
        # Extract all functions
        all_functions = parser.extract_functions(request.code)
        
        # Find vulnerable functions
        vulnerable_functions = parser.find_vulnerable_functions(request.code)
        
        # Get the best function for analysis
        best_function = parser.get_best_function_for_analysis(request.code)
        
        # Get summary
        summary = parser.get_function_summary(all_functions)
        
        print(f"✅ Function extraction completed:")
        print(f"   Total functions found: {len(all_functions)}")
        print(f"   Best function: {best_function['name']}")
        print(f"   Vulnerability score: {best_function.get('vulnerability_score', 0)}")
        
        return FunctionExtractionResponse(
            functions=all_functions,
            best_function=best_function,
            summary=summary,
            extraction_successful=True
        )
        
    except Exception as e:
        print(f"❌ Function extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Function extraction failed: {str(e)}")

@app.post("/upload-code")
async def upload_code_file(file: UploadFile = File(...)):
    """Upload a code file for analysis"""
    # Check file size (limit to 1MB)
    content = await file.read()
    if len(content) > 1024 * 1024:  # 1MB limit
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 1MB.")
    
    # Check file extension
    allowed_extensions = ['.c', '.cpp', '.h', '.hpp', '.java', '.py', '.js', '.ts', '.php', '.cs', '.go', '.rs', '.rb', '.swift', '.kt', '.scala', '.txt']
    file_extension = os.path.splitext(file.filename or "")[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed extensions: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Decode the file content
        code_content = content.decode('utf-8')
        
        # Determine language from file extension
        language_map = {
            '.c': 'c', '.h': 'c',
            '.cpp': 'cpp', '.hpp': 'cpp',
            '.java': 'java',
            '.py': 'python',
            '.js': 'javascript', '.ts': 'typescript',
            '.php': 'php',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.txt': 'text'
        }
        
        detected_language = language_map.get(file_extension, 'text')
        
        return {
            "filename": file.filename,
            "code": code_content,
            "language": detected_language,
            "size": len(content),
            "lines": len(code_content.split('\n')),
            "message": "Code file uploaded successfully"
        }
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be a valid text file with UTF-8 encoding")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@app.post("/upload-attack-pool")
async def upload_attack_pool(file: UploadFile = File(...)):
    """Upload a custom attack pool CSV file"""
    global attack_pool_data
    
    # Check if it's a CSV file
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV file with .csv extension")
    
    # Check file size (limit to 10MB)
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB.")
    
    try:
        # Decode and parse CSV
        content_str = content.decode('utf-8')
        csv_file = StringIO(content_str)
        
        # Read CSV with pandas
        uploaded_df = pd.read_csv(csv_file)
        
        # Validate CSV structure
        required_columns = ['original_code', 'adversarial_code', 'label']
        missing_columns = [col for col in required_columns if col not in uploaded_df.columns]
        
        if missing_columns:
            available_columns = list(uploaded_df.columns)
            raise HTTPException(
                status_code=400, 
                detail=f"CSV must contain columns: {', '.join(required_columns)}. "
                       f"Missing columns: {', '.join(missing_columns)}. "
                       f"Available columns: {', '.join(available_columns)}"
            )
        
        # Validate data
        if len(uploaded_df) == 0:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Check for missing values in critical columns
        if uploaded_df['adversarial_code'].isna().sum() > len(uploaded_df) * 0.5:
            raise HTTPException(
                status_code=400, 
                detail="Too many missing values in 'adversarial_code' column. More than 50% of entries are empty."
            )
        
        # Remove rows with missing adversarial_code
        original_count = len(uploaded_df)
        uploaded_df = uploaded_df.dropna(subset=['adversarial_code'])
        cleaned_count = len(uploaded_df)
        
        if cleaned_count == 0:
            raise HTTPException(
                status_code=400, 
                detail="No valid adversarial code entries found after cleaning"
            )
        
        # Update global attack pool
        attack_pool_data = uploaded_df
        
        # Save the uploaded attack pool
        upload_path = os.path.join(os.path.dirname(__file__), "attack_pool_uploaded.csv")
        attack_pool_data.to_csv(upload_path, index=False)
        
        print(f"New attack pool uploaded with {len(attack_pool_data)} snippets")
        print(f"Attack pool columns: {list(attack_pool_data.columns)}")
        
        return {
            "filename": file.filename,
            "message": "Attack pool uploaded successfully",
            "total_entries": original_count,
            "valid_entries": cleaned_count,
            "removed_entries": original_count - cleaned_count,
            "columns": list(uploaded_df.columns),
            "sample_adversarial_code": uploaded_df['adversarial_code'].iloc[0] if len(uploaded_df) > 0 else None
        }
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be a valid CSV file with UTF-8 encoding")
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty or invalid")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV file: {str(e)}")

@app.get("/attack-pool-format")
async def get_attack_pool_format():
    """Get information about the required attack pool CSV format"""
    return {
        "required_columns": [
            {
                "name": "original_code",
                "description": "The original vulnerable code snippet",
                "type": "string",
                "example": "char buffer[10]; strcpy(buffer, user_input);"
            },
            {
                "name": "adversarial_code", 
                "description": "The adversarial code snippet designed to evade detection",
                "type": "string",
                "example": "char *ptr = malloc(10); if(ptr) strcpy(ptr, data);"
            },
            {
                "name": "label",
                "description": "Binary label indicating vulnerability (0 = benign, 1 = vulnerable)",
                "type": "integer",
                "example": 1
            }
        ],
        "csv_example": "original_code,adversarial_code,label\n\"gets(buffer);\",\"char buf[100]; if(strlen(input)<100) strcpy(buf,input);\",0\n\"system(cmd);\",\"if(strcmp(cmd,\\\"safe\\\")==0) system(cmd);\",0",
        "requirements": [
            "File must have .csv extension",
            "File must be UTF-8 encoded",
            "Maximum file size: 10MB",
            "Must contain all three required columns",
            "adversarial_code column cannot be more than 50% empty",
            "At least one valid entry required"
        ],
        "tips": [
            "Use double quotes for text containing commas or newlines",
            "Escape double quotes within text by doubling them (\"\")",
            "Keep code snippets concise but meaningful",
            "Ensure adversarial examples are realistic and well-formed"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    # Run with no timeout limits for vulnerability detection
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=3002, 
        reload=True,
        timeout_keep_alive=0,  # Disable keep-alive timeout
        timeout_graceful_shutdown=None,  # No graceful shutdown timeout
        limit_concurrency=None,  # No concurrency limit
        limit_max_requests=None  # No max requests limit
    ) 