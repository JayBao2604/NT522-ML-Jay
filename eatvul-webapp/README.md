# EatVul Security Analyzer

A professional web application for vulnerability detection and adversarial attacks using CodeBERT, Fuzzy Genetic Algorithm (FGA) selection, and Gemini AI explanations.

## Features

- **🔍 Code Vulnerability Detection**: Uses CodeBERT model to detect security vulnerabilities in C, C++, JavaScript, and Java code
- **🤖 AI-Powered Explanations**: Gemini AI provides detailed explanations of vulnerabilities and suggests fixes
- **⚡ Adversarial Attacks**: Implements FGA-based adversarial attack selection to test model robustness
- **📊 Real-time Progress Visualization**: Live charts and metrics for FGA selection process
- **💉 Code Injection**: Inject attack snippets into code and visualize injection points
- **🎯 Attack Success Monitoring**: Track how adversarial attacks affect model predictions
- **🎨 Professional UI**: Modern, cyberpunk-themed interface with security-focused design

## Architecture

### Backend (FastAPI)
- **Vulnerability Detection**: CodeBERT model integration for security analysis
- **FGA Selection**: Fuzzy Genetic Algorithm for optimal attack snippet selection
- **Gemini Integration**: AI-powered vulnerability explanations
- **Adversarial Learning**: Attack pool management and execution

### Frontend (React)
- **Monaco Editor**: Syntax-highlighted code editor with vulnerability highlighting
- **Real-time Charts**: Progress visualization using Recharts
- **Responsive Design**: Professional UI with Framer Motion animations
- **Component Architecture**: Modular design with styled-components

## Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- CodeBERT model files
- Gemini API key

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create .env file in backend directory
GEMINI_API_KEY=your-gemini-api-key-here
MODEL_PATH=F:\NT522-Project\eatvul-webapp\codebert-model
ATTACK_POOL_PATH=attack_pool.csv
```

4. Start the FastAPI server:
```bash
python main.py
```

The backend will run on `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install Node.js dependencies:
```bash
npm install
```

3. Set up environment variables:
```bash
# Create .env file in frontend directory
REACT_APP_API_BASE_URL=http://localhost:8000
GENERATE_SOURCEMAP=false
```

4. Start the React development server:
```bash
npm start
```

The frontend will run on `http://localhost:3000`

## Usage

### 1. Code Analysis
1. **Select Language**: Choose from C, C++, JavaScript, or Java
2. **Enter Code**: Write or paste your code in the Monaco editor
3. **Detect Vulnerabilities**: Click "Detect Vulnerability" to analyze the code
4. **View Results**: See vulnerability explanations and highlighted vulnerable lines

### 2. Adversarial Attacks
1. **Start FGA Selection**: Click "Start FGA Selection" to find the best attack snippets
2. **Monitor Progress**: Watch real-time progress visualization and metrics
3. **Load Best Snippet**: Once complete, load the optimal attack snippet
4. **Inject Code**: Inject the attack snippet into your original code
5. **Perform Attack**: Execute the adversarial attack to test model robustness
6. **Analyze Results**: Compare before/after predictions to see attack effectiveness

### 3. Understanding Results
- **Red highlighting**: Vulnerable lines in code
- **Yellow highlighting**: Injected attack code
- **Hover tooltips**: Detailed vulnerability explanations
- **Attack success**: Shows if the model's prediction changed after attack

## Components

### Backend Components

#### `train_codebert_model.py`
- CodeBERT model training and inference
- Vulnerability prediction with confidence scores
- Model loading and saving functionality

#### `fga_selection.py`
- Fuzzy Genetic Algorithm implementation
- Attack snippet selection and optimization
- Fitness scoring for adversarial examples

#### `adversarial_learning.py`
- Adversarial attack execution
- Attack pool management
- Model robustness testing

#### `main.py`
- FastAPI server with CORS configuration
- API endpoints for all functionality
- Gemini AI integration for explanations

### Frontend Components

#### `App.js`
- Main application component
- State management for all features
- API integration and error handling

#### `CodeEditor.js`
- Monaco editor with custom theme
- Vulnerability and injection highlighting
- Interactive tooltips and decorations

#### `VulnerabilityPanel.js`
- Displays analysis results
- Vulnerability details with fixes
- Confidence scores and metrics

#### `AdversarialPanel.js`
- FGA selection controls
- Attack snippet management
- Attack execution and results

#### `ProgressVisualization.js`
- Real-time FGA progress charts
- Time estimation and metrics
- Status indicators

## API Endpoints

- `GET /health` - Health check
- `POST /analyze-vulnerability` - Analyze code for vulnerabilities
- `GET /attack-pool` - Get available attack snippets
- `POST /start-fga-selection` - Start FGA selection process
- `GET /fga-progress` - Get FGA progress status
- `GET /best-attack-snippet` - Get optimal attack snippet
- `POST /adversarial-attack` - Perform adversarial attack

## Model Requirements

### CodeBERT Model
Place the trained CodeBERT model in `F:\NT522-Project\eatvul-webapp\codebert-model\` with:
- `best_model.pt` or `model.pt` - Model weights
- `model_config.json` - Model configuration
- Tokenizer files (from transformers library)

### Attack Pool
Create `attack_pool.csv` with adversarial code snippets:
```csv
adversarial_code
"// This is a harmless comment that might confuse the model"
"char buffer[100]; // Large enough buffer"
"gets(buffer); // Known buffer overflow"
```

## Configuration

### Environment Variables

**Backend:**
- `GEMINI_API_KEY` - Your Gemini API key for AI explanations
- `MODEL_PATH` - Path to CodeBERT model directory
- `ATTACK_POOL_PATH` - Path to attack pool CSV file

**Frontend:**
- `REACT_APP_API_BASE_URL` - Backend API URL (default: http://localhost:8000)
- `GENERATE_SOURCEMAP` - Disable source maps for production (false)

## Security Considerations

⚠️ **Important**: This tool is designed for security research and education purposes only.

- Keep your Gemini API key secure
- Use in controlled environments only
- Do not use on production systems
- Adversarial attacks should only be used for model testing

## Troubleshooting

### Common Issues

1. **Model not loading**: Ensure CodeBERT model files are in the correct directory
2. **Gemini API errors**: Check your API key and quota
3. **CORS issues**: Verify backend is running on port 8000
4. **Frontend build errors**: Ensure all dependencies are installed

### Development

For development, you can run both backend and frontend simultaneously:

```bash
# Terminal 1 - Backend
cd backend && python main.py

# Terminal 2 - Frontend  
cd frontend && npm start
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test thoroughly
4. Submit a pull request

## License

This project is for educational and research purposes. Please use responsibly.

## Support

For issues and questions, please check the troubleshooting section or create an issue in the repository. 