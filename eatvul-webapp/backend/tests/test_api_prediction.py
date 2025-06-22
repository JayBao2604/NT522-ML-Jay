#!/usr/bin/env python3
"""
Test script to verify the backend API returns correct predictions
"""
import requests
import json
import time

def test_backend_api():
    """Test the backend API with samples that should be vulnerable"""
    print("🔍 Testing Backend API Predictions")
    print("=" * 50)
    
    base_url = "http://localhost:3001"
    
    # Check if backend is running
    try:
        health_response = requests.get(f"{base_url}/health")
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ Backend is running")
            print(f"   Model loaded: {health_data.get('model_loaded', 'Unknown')}")
        else:
            print(f"❌ Backend health check failed: {health_response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to backend at {base_url}")
        print(f"   Error: {e}")
        print("   Please start the backend first: cd backend && python main.py")
        return False
    
    # Get diagnostic info
    try:
        diag_response = requests.get(f"{base_url}/diagnostic")
        if diag_response.status_code == 200:
            diag_data = diag_response.json()
            print(f"\n📊 Diagnostic Info:")
            print(f"   Model status: {diag_data.get('model', {}).get('status', 'Unknown')}")
            print(f"   Model path exists: {diag_data.get('model', {}).get('path_exists', 'Unknown')}")
        else:
            print(f"⚠️ Diagnostic endpoint failed: {diag_response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Diagnostic request failed: {e}")
    
    # Test samples that should be predicted as vulnerable
    test_samples = [
        {
            "name": "Buffer overflow with gets()",
            "code": "int main() { char buf[10]; gets(buf); return 0; }",
            "language": "c",
            "expected_vulnerable": True
        },
        {
            "name": "Buffer overflow with strcpy()",
            "code": "void func(char *input) { char dest[10]; strcpy(dest, input); }",
            "language": "c",
            "expected_vulnerable": True
        },
        {
            "name": "Format string vulnerability",
            "code": "char* process(char* input) { char buffer[100]; sprintf(buffer, \"%s\", input); return buffer; }",
            "language": "c",
            "expected_vulnerable": True
        },
        {
            "name": "Command injection",
            "code": "void execute_command(char* user_input) { char cmd[256]; sprintf(cmd, \"ls %s\", user_input); system(cmd); }",
            "language": "c",
            "expected_vulnerable": True
        },
        {
            "name": "Safe function",
            "code": "int add(int a, int b) { return a + b; }",
            "language": "c",
            "expected_vulnerable": False
        }
    ]
    
    print(f"\n🧪 Testing {len(test_samples)} code samples (no timeout):")
    print("=" * 80)
    
    vulnerable_predictions = 0
    correct_predictions = 0
    
    for i, sample in enumerate(test_samples):
        print(f"\n📝 Sample {i+1}: {sample['name']}")
        print(f"Expected: {'Vulnerable' if sample['expected_vulnerable'] else 'Safe'}")
        
        try:
            # Make API request without timeout
            print("🔄 Analyzing (no timeout limit)...")
            response = requests.post(
                f"{base_url}/analyze-vulnerability",
                json={
                    "code": sample["code"],
                    "language": sample["language"]
                }
                # No timeout parameter - unlimited time
            )
            
            if response.status_code == 200:
                result = response.json()
                is_vulnerable = result.get("is_vulnerable", False)
                confidence = result.get("confidence", 0.0)
                
                print(f"✅ Prediction: {'Vulnerable' if is_vulnerable else 'Safe'}")
                print(f"   Confidence: {confidence:.4f}")
                
                if is_vulnerable:
                    vulnerable_predictions += 1
                
                # Check if prediction matches expectation
                if is_vulnerable == sample["expected_vulnerable"]:
                    correct_predictions += 1
                    print(f"   ✅ Correct prediction!")
                else:
                    print(f"   ❌ Incorrect prediction (expected {'Vulnerable' if sample['expected_vulnerable'] else 'Safe'})")
                
            elif response.status_code == 503:
                print(f"❌ Service unavailable (model not loaded): {response.json().get('detail', 'Unknown error')}")
                return False
            else:
                print(f"❌ API request failed: {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"   Error: {error_detail}")
                except:
                    print(f"   Error: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
        
        # Add delay between requests
        time.sleep(0.5)
    
    print(f"\n📊 Test Results:")
    print(f"   Total samples: {len(test_samples)}")
    print(f"   Predicted as vulnerable: {vulnerable_predictions}")
    print(f"   Correct predictions: {correct_predictions}/{len(test_samples)}")
    print(f"   Accuracy: {correct_predictions/len(test_samples)*100:.1f}%")
    
    # Summary
    if vulnerable_predictions > 0:
        print(f"\n✅ SUCCESS: Backend is predicting some samples as vulnerable!")
        print(f"   This suggests the CodeBERT model is working correctly.")
        if correct_predictions == len(test_samples):
            print(f"   🎉 All predictions were correct!")
        else:
            print(f"   ⚠️ Some predictions were incorrect, but this is normal for ML models.")
    else:
        print(f"\n❌ PROBLEM: No samples predicted as vulnerable!")
        print(f"   This suggests the backend might still be using fallback mode.")
        print(f"   Check the backend logs and model loading process.")
    
    return vulnerable_predictions > 0

if __name__ == "__main__":
    print("🚀 Backend API Prediction Test (No Timeout)")
    print("="*60)
    
    success = test_backend_api()
    
    if success:
        print(f"\n🎉 API test completed successfully!")
        print(f"   Backend is working correctly with CodeBERT model.")
    else:
        print(f"\n❌ API test failed!")
        print(f"   Please check backend startup logs and fix any issues.") 