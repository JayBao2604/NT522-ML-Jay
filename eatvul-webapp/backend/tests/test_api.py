import pytest
import asyncio
import json
import pandas as pd
import os
import tempfile
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import sys

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import app

class TestEatVulAPI:
    """Test suite for EatVul Security Analysis API"""
    
    @pytest.fixture
    def client(self):
        """Create a test client"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_attack_pool(self):
        """Create a sample attack pool for testing"""
        return pd.DataFrame({
            'original_code': [
                'void test() { char buf[10]; }',
                'int main() { char *ptr = malloc(100); }'
            ],
            'adversarial_code': [
                '// Test comment\nchar safe_buf[100];',
                'char *safe_ptr = calloc(100, 1); if(safe_ptr) free(safe_ptr);'
            ],
            'label': [0, 0]
        })
    
    @pytest.fixture
    def mock_codebert_trainer(self):
        """Mock CodeBERT trainer for testing"""
        mock_trainer = MagicMock()
        mock_trainer.predict.return_value = {
            'prediction': 1,
            'confidence': 0.85,
            'probabilities': [0.15, 0.85]
        }
        return mock_trainer

    def test_health_check(self, client):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data
        assert data["status"] == "healthy"

    @patch('main.codebert_trainer')
    @patch('main.model_loaded', True)
    def test_analyze_vulnerability_success(self, mock_trainer, client):
        """Test vulnerability analysis with successful prediction"""
        mock_trainer.predict.return_value = {
            'prediction': 1,
            'confidence': 0.85,
            'probabilities': [0.15, 0.85]
        }
        
        with patch('main.get_gemini_explanation') as mock_gemini:
            mock_gemini.return_value = asyncio.run(self._mock_gemini_response(True))
            
            response = client.post("/analyze-vulnerability", json={
                "code": "void test() { char buf[10]; gets(buf); }",
                "language": "c"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["is_vulnerable"] == True
            assert data["confidence"] == 0.85
            assert len(data["probabilities"]) == 2
            assert "explanation" in data
            assert isinstance(data["vulnerable_lines"], list)

    @patch('main.codebert_trainer')
    @patch('main.model_loaded', True)
    def test_analyze_vulnerability_secure(self, mock_trainer, client):
        """Test vulnerability analysis with secure code"""
        mock_trainer.predict.return_value = {
            'prediction': 0,
            'confidence': 0.92,
            'probabilities': [0.92, 0.08]
        }
        
        with patch('main.get_gemini_explanation') as mock_gemini:
            mock_gemini.return_value = asyncio.run(self._mock_gemini_response(False))
            
            response = client.post("/analyze-vulnerability", json={
                "code": "void safe_func() { char buf[100]; strncpy(buf, input, 99); buf[99] = '\\0'; }",
                "language": "c"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["is_vulnerable"] == False
            assert data["confidence"] == 0.92

    def test_analyze_vulnerability_model_not_loaded(self, client):
        """Test vulnerability analysis when model is not loaded"""
        with patch('main.model_loaded', False):
            response = client.post("/analyze-vulnerability", json={
                "code": "void test() { }",
                "language": "c"
            })
            
            assert response.status_code == 503
            assert "CodeBERT model not loaded" in response.json()["detail"]

    @patch('main.attack_pool_data')
    def test_get_attack_pool_success(self, mock_pool, client, sample_attack_pool):
        """Test getting attack pool successfully"""
        mock_pool.return_value = sample_attack_pool
        mock_pool.columns = sample_attack_pool.columns
        mock_pool.__getitem__ = sample_attack_pool.__getitem__
        mock_pool.iloc = sample_attack_pool.iloc
        
        with patch('main.attack_pool_data', sample_attack_pool):
            response = client.get("/attack-pool")
            
            assert response.status_code == 200
            data = response.json()
            assert "attack_snippets" in data
            assert "best_snippet" in data
            assert "total_snippets" in data
            assert data["total_snippets"] == 2
            assert len(data["attack_snippets"]) == 2

    def test_get_attack_pool_not_loaded(self, client):
        """Test getting attack pool when not loaded"""
        with patch('main.attack_pool_data', None):
            response = client.get("/attack-pool")
            
            assert response.status_code == 503
            assert "Attack pool not loaded" in response.json()["detail"]

    @patch('main.attack_pool_data')
    def test_start_fga_selection_success(self, mock_pool, client, sample_attack_pool):
        """Test starting FGA selection successfully"""
        with patch('main.attack_pool_data', sample_attack_pool):
            response = client.post("/start-fga-selection")
            
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "FGA selection started"
            assert data["status"] == "running"

    def test_start_fga_selection_no_pool(self, client):
        """Test starting FGA selection without attack pool"""
        with patch('main.attack_pool_data', None):
            response = client.post("/start-fga-selection")
            
            assert response.status_code == 503
            assert "Attack pool not loaded" in response.json()["detail"]

    def test_get_fga_progress(self, client):
        """Test getting FGA progress"""
        # Set up mock progress
        mock_progress = {
            "current_generation": 5,
            "max_generations": 10,
            "best_fitness": 0.75,
            "attack_success_rate": 0.65,
            "time_elapsed": 30.0,
            "estimated_time_remaining": 30.0,
            "status": "running"
        }
        
        with patch('main.fga_progress', mock_progress):
            response = client.get("/fga-progress")
            
            assert response.status_code == 200
            data = response.json()
            assert data["current_generation"] == 5
            assert data["max_generations"] == 10
            assert data["best_fitness"] == 0.75
            assert data["status"] == "running"

    @patch('main.attack_pool_data')
    def test_get_best_attack_snippet_completed(self, mock_pool, client, sample_attack_pool):
        """Test getting best attack snippet when FGA is completed"""
        mock_progress = {
            "status": "completed",
            "best_fitness": 0.95,
            "attack_success_rate": 0.85
        }
        
        with patch('main.attack_pool_data', sample_attack_pool), \
             patch('main.fga_progress', mock_progress):
            
            response = client.get("/best-attack-snippet")
            
            assert response.status_code == 200
            data = response.json()
            assert "best_snippet" in data
            assert "fitness_score" in data
            assert "attack_success_rate" in data
            assert data["status"] == "completed"

    @patch('main.attack_pool_data')
    def test_get_best_attack_snippet_running(self, mock_pool, client, sample_attack_pool):
        """Test getting best attack snippet while FGA is running"""
        mock_progress = {
            "status": "running",
            "best_fitness": 0.45,
            "attack_success_rate": 0.35
        }
        
        with patch('main.attack_pool_data', sample_attack_pool), \
             patch('main.fga_progress', mock_progress):
            
            response = client.get("/best-attack-snippet")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "running"
            assert data["best_snippet"] in sample_attack_pool["adversarial_code"].values

    @patch('main.codebert_trainer')
    @patch('main.model_loaded', True)
    def test_adversarial_attack_success(self, mock_trainer, client):
        """Test successful adversarial attack"""
        # Mock predictions for original and adversarial code
        mock_trainer.predict.side_effect = [
            {'prediction': 1, 'confidence': 0.85, 'probabilities': [0.15, 0.85]},  # Original
            {'prediction': 0, 'confidence': 0.75, 'probabilities': [0.75, 0.25]}   # Adversarial
        ]
        
        response = client.post("/adversarial-attack", json={
            "original_code": "void test() { char buf[10]; gets(buf); }",
            "language": "c",
            "attack_snippet": "// Safe comment\nchar safe_buf[100];"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "original_prediction" in data
        assert "adversarial_prediction" in data
        assert "adversarial_code" in data
        assert "attack_success" in data
        assert "injection_info" in data
        
        # Check if attack was successful (vulnerable -> secure)
        assert data["original_prediction"]["is_vulnerable"] == True
        assert data["adversarial_prediction"]["is_vulnerable"] == False
        assert data["attack_success"] == True

    @patch('main.codebert_trainer')
    @patch('main.model_loaded', True)
    def test_adversarial_attack_failed(self, mock_trainer, client):
        """Test failed adversarial attack"""
        # Mock predictions - both return vulnerable
        mock_trainer.predict.side_effect = [
            {'prediction': 1, 'confidence': 0.85, 'probabilities': [0.15, 0.85]},  # Original
            {'prediction': 1, 'confidence': 0.80, 'probabilities': [0.20, 0.80]}   # Adversarial
        ]
        
        response = client.post("/adversarial-attack", json={
            "original_code": "void test() { char buf[10]; gets(buf); }",
            "language": "c",
            "attack_snippet": "// Ineffective comment"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["attack_success"] == False

    def test_adversarial_attack_model_not_loaded(self, client):
        """Test adversarial attack when model is not loaded"""
        with patch('main.model_loaded', False):
            response = client.post("/adversarial-attack", json={
                "original_code": "void test() { }",
                "language": "c",
                "attack_snippet": "// comment"
            })
            
            assert response.status_code == 503
            assert "CodeBERT model not loaded" in response.json()["detail"]

    def test_injection_logic(self, client):
        """Test the code injection logic"""
        from main import inject_attack_snippet, get_injection_info
        
        original_code = """void test_function() {
    char buffer[100];
    printf("Hello World");
}"""
        
        attack_snippet = "// Security check passed"
        
        # Test injection
        injected = inject_attack_snippet(original_code, attack_snippet)
        assert "// Security check passed" in injected
        assert "INJECTED_ATTACK_CODE" in injected
        
        # Test injection info
        info = get_injection_info(original_code, attack_snippet)
        assert "injection_line" in info
        assert "attack_snippet" in info
        assert "marker" in info
        assert info["attack_snippet"] == attack_snippet

    async def _mock_gemini_response(self, is_vulnerable):
        """Mock Gemini API response"""
        if is_vulnerable:
            return (
                "Buffer overflow vulnerability detected in gets() function call.",
                [
                    {
                        "line_number": 1,
                        "code": "gets(buf);",
                        "vulnerability_type": "buffer overflow",
                        "reason": "gets() function is unsafe and can cause buffer overflow",
                        "fix_suggestion": "Use fgets() instead of gets()"
                    }
                ]
            )
        else:
            return ("Code appears to be secure with proper bounds checking.", [])

    def test_csv_format_handling(self, client):
        """Test handling of different CSV formats"""
        # Test with adversarial_code column
        csv_content_1 = "adversarial_code\n// Comment 1\n// Comment 2"
        
        # Test with original_code, adversarial_code, label format
        csv_content_2 = "original_code,adversarial_code,label\nvoid test() {},// Comment,0"
        
        # Test with unknown format
        csv_content_3 = "unknown_column\nSome data"
        
        # These would need to be tested with actual file operations
        # For now, we can test the logic through the API endpoints
        pass

    @patch('main.attack_pool_data')
    @patch('main.os.path.exists')
    def test_attack_pool_loading_edge_cases(self, mock_exists, mock_pool, client):
        """Test attack pool loading with various edge cases"""
        # Test when file doesn't exist
        mock_exists.return_value = False
        
        # Test when CSV has wrong format
        wrong_format_df = pd.DataFrame({
            'wrong_column': ['data1', 'data2']
        })
        
        # These would be tested during app startup
        pass

# Integration tests
class TestIntegration:
    """Integration tests for full workflow"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @patch('main.codebert_trainer')
    @patch('main.model_loaded', True)
    @patch('main.attack_pool_data')
    def test_full_vulnerability_analysis_workflow(self, mock_pool, mock_trainer, client, sample_attack_pool):
        """Test the complete workflow from analysis to attack"""
        # Setup mocks
        mock_trainer.predict.side_effect = [
            {'prediction': 1, 'confidence': 0.85, 'probabilities': [0.15, 0.85]},  # Analysis
            {'prediction': 1, 'confidence': 0.85, 'probabilities': [0.15, 0.85]},  # Original attack
            {'prediction': 0, 'confidence': 0.75, 'probabilities': [0.75, 0.25]}   # Adversarial attack
        ]
        
        with patch('main.attack_pool_data', sample_attack_pool):
            # 1. Analyze vulnerability
            response = client.post("/analyze-vulnerability", json={
                "code": "void test() { char buf[10]; gets(buf); }",
                "language": "c"
            })
            assert response.status_code == 200
            
            # 2. Get attack pool
            response = client.get("/attack-pool")
            assert response.status_code == 200
            attack_data = response.json()
            
            # 3. Start FGA selection
            response = client.post("/start-fga-selection")
            assert response.status_code == 200
            
            # 4. Get best attack snippet
            response = client.get("/best-attack-snippet")
            assert response.status_code == 200
            snippet_data = response.json()
            
            # 5. Perform adversarial attack
            response = client.post("/adversarial-attack", json={
                "original_code": "void test() { char buf[10]; gets(buf); }",
                "language": "c",
                "attack_snippet": snippet_data["best_snippet"]
            })
            assert response.status_code == 200
            attack_result = response.json()
            assert attack_result["attack_success"] == True

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"]) 