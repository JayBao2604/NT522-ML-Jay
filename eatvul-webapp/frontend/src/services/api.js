import axios from "axios";

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || "http://localhost:3002";

const api = axios.create({
	baseURL: API_BASE_URL,
	// No timeout - allow unlimited time for vulnerability detection
	headers: {
		"Content-Type": "application/json",
	},
});

// Request interceptor
api.interceptors.request.use(
	(config) => {
		console.log(`Making ${config.method?.toUpperCase()} request to ${config.url}`);
		return config;
	},
	(error) => {
		return Promise.reject(error);
	}
);

// Response interceptor
api.interceptors.response.use(
	(response) => {
		console.log(`Response from ${response.config.url}:`, response.status);
		return response;
	},
	(error) => {
		console.error("API Error:", error.response?.data || error.message);

		// Handle specific error codes
		if (error.response?.status === 503) {
			throw new Error("Model not loaded. Please check the backend.");
		} else if (error.response?.status === 500) {
			throw new Error(error.response.data?.detail || "Server error occurred.");
		} else if (error.code === "ECONNREFUSED") {
			throw new Error("Cannot connect to the backend server. Please ensure it is running.");
		}

		throw error;
	}
);

export const vulnerabilityService = {
	// Health check
	async healthCheck() {
		const response = await api.get("/health");
		return response.data;
	},

	// Analyze code for vulnerabilities
	async analyzeCode(data) {
		const response = await api.post("/analyze-vulnerability", data);
		return response.data;
	},

	// Get attack pool
	async getAttackPool() {
		const response = await api.get("/attack-pool");
		return response.data;
	},

	// Get FGA parameters and their descriptions
	async getFgaParameters() {
		const response = await api.get("/fga-parameters");
		return response.data;
	},

	// Start FGA selection with optional parameters
	async startFgaSelection(parameters = null, extractedFunction = null) {
		const payload = { parameters };
		if (extractedFunction) {
			payload.extracted_function = extractedFunction;
		}
		const response = await api.post("/start-fga-selection", payload);
		return response.data;
	},

	// Get FGA progress
	async getFgaProgress() {
		const response = await api.get("/fga-progress");
		return response.data;
	},

	// Get best attack snippet
	async getBestAttackSnippet() {
		const response = await api.get("/best-attack-snippet");
		return response.data;
	},

	// Perform adversarial attack
	async performAttack(data) {
		const response = await api.post("/adversarial-attack", data);
		return response.data;
	},

	// Upload code file
	async uploadCodeFile(file) {
		const formData = new FormData();
		formData.append("file", file);

		const response = await api.post("/upload-code", formData, {
			headers: {
				"Content-Type": "multipart/form-data",
			},
		});
		return response.data;
	},

	// Upload attack pool CSV
	async uploadAttackPool(file) {
		const formData = new FormData();
		formData.append("file", file);

		const response = await api.post("/upload-attack-pool", formData, {
			headers: {
				"Content-Type": "multipart/form-data",
			},
		});
		return response.data;
	},

	// Get attack pool format requirements
	async getAttackPoolFormat() {
		const response = await api.get("/attack-pool-format");
		return response.data;
	},

	// Get diagnostic information
	async getDiagnostic() {
		const response = await api.get("/diagnostic");
		return response.data;
	},

	async extractFunctions(data) {
		const response = await api.post("/extract-functions", data);
		return response.data;
	},
};

export default api;
