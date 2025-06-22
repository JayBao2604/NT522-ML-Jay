import React, { useState, useEffect, useRef } from "react";
import styled, { ThemeProvider, createGlobalStyle } from "styled-components";
import { motion, AnimatePresence } from "framer-motion";
import { Toaster, toast } from "react-hot-toast";
import {
	Shield,
	ShieldAlert,
	ShieldCheck,
	Bug,
	Code,
	Play,
	Zap,
	Target,
	TrendingUp,
	Clock,
	AlertTriangle,
	CheckCircle,
	XCircle,
	Eye,
	Download,
	Upload,
	Loader,
	BarChart3,
} from "lucide-react";

import CodeEditor from "./components/CodeEditor";
import VulnerabilityPanel from "./components/VulnerabilityPanel";
import AdversarialPanel from "./components/AdversarialPanel";
import ProgressVisualization from "./components/ProgressVisualization";
import StatusBadge from "./components/StatusBadge";
import FGAParameterPanel from "./components/FGAParameterPanel";
import FileUploadPanel from "./components/FileUploadPanel";
import { vulnerabilityService } from "./services/api";

// Global Styles
const GlobalStyle = createGlobalStyle`
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    background: ${(props) => props.theme.colors.background};
    color: ${(props) => props.theme.colors.text};
    line-height: 1.6;
  }

  ::-webkit-scrollbar {
    width: 8px;
  }

  ::-webkit-scrollbar-track {
    background: ${(props) => props.theme.colors.surface};
  }

  ::-webkit-scrollbar-thumb {
    background: ${(props) => props.theme.colors.border};
    border-radius: 4px;
  }

  ::-webkit-scrollbar-thumb:hover {
    background: ${(props) => props.theme.colors.primary};
  }
`;

// Theme
const theme = {
	colors: {
		primary: "#00d4ff",
		secondary: "#ff6b6b",
		success: "#00ff88",
		warning: "#ffeb3b",
		error: "#ff4757",
		background: "#0a0e1a",
		surface: "#1a1d29",
		surfaceLight: "#2a2d3a",
		text: "#ffffff",
		textSecondary: "#a0a3b1",
		border: "#3a3d4a",
		accent: "#7c3aed",
		vulnerability: "#ff4757",
		secure: "#00ff88",
	},
	gradients: {
		primary: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
		security: "linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%)",
		success: "linear-gradient(135deg, #00ff88 0%, #00b894 100%)",
		cyber: "linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%)",
	},
	shadows: {
		small: "0 2px 4px rgba(0, 0, 0, 0.2)",
		medium: "0 4px 8px rgba(0, 0, 0, 0.3)",
		large: "0 8px 16px rgba(0, 0, 0, 0.4)",
		glow: "0 0 20px rgba(0, 212, 255, 0.3)",
	},
};

// Styled Components
const AppContainer = styled.div`
	min-height: 100vh;
	background: ${(props) => props.theme.colors.background};
	display: flex;
	flex-direction: column;
`;

const Header = styled(motion.header)`
	background: linear-gradient(135deg, #1a1d29 0%, #2a2d3a 100%);
	border-bottom: 2px solid ${(props) => props.theme.colors.border};
	padding: 1rem 2rem;
	display: flex;
	align-items: center;
	justify-content: space-between;
	box-shadow: ${(props) => props.theme.shadows.medium};
	position: sticky;
	top: 0;
	z-index: 1000;
`;

const Logo = styled.div`
	display: flex;
	align-items: center;
	gap: 1rem;
	font-size: 1.5rem;
	font-weight: 700;
	color: ${(props) => props.theme.colors.primary};
`;

const LogoIcon = styled(Shield)`
	width: 2rem;
	height: 2rem;
	filter: drop-shadow(0 0 8px ${(props) => props.theme.colors.primary});
`;

const HeaderActions = styled.div`
	display: flex;
	align-items: center;
	gap: 1rem;
`;

const MainContent = styled.div`
	flex: 1;
	display: grid;
	grid-template-columns: 1fr 400px;
	gap: 1rem;
	padding: 1rem 2rem;
	max-width: 1800px;
	margin: 0 auto;
	width: 100%;

	@media (max-width: 1200px) {
		grid-template-columns: 1fr;
		max-width: 1000px;
	}
`;

const EditorSection = styled(motion.div)`
	background: ${(props) => props.theme.colors.surface};
	border-radius: 12px;
	border: 1px solid ${(props) => props.theme.colors.border};
	overflow: hidden;
	box-shadow: ${(props) => props.theme.shadows.medium};
`;

const EditorHeader = styled.div`
	padding: 1rem 1.5rem;
	border-bottom: 1px solid ${(props) => props.theme.colors.border};
	display: flex;
	align-items: center;
	justify-content: space-between;
	background: ${(props) => props.theme.colors.surfaceLight};
`;

const EditorTitle = styled.div`
	display: flex;
	align-items: center;
	gap: 0.5rem;
	font-weight: 600;
	color: ${(props) => props.theme.colors.text};
`;

const LanguageSelector = styled.select`
	background: ${(props) => props.theme.colors.surface};
	border: 1px solid ${(props) => props.theme.colors.border};
	border-radius: 6px;
	padding: 0.5rem 1rem;
	color: ${(props) => props.theme.colors.text};
	font-size: 0.9rem;
	cursor: pointer;

	&:focus {
		outline: none;
		border-color: ${(props) => props.theme.colors.primary};
	}

	option {
		background: ${(props) => props.theme.colors.surface};
		color: ${(props) => props.theme.colors.text};
	}
`;

const ActionButton = styled(motion.button)`
	background: ${(props) => {
		if (props.variant === "primary") return props.theme.gradients.cyber;
		if (props.variant === "danger") return props.theme.gradients.security;
		if (props.variant === "success") return props.theme.gradients.success;
		return props.theme.colors.surfaceLight;
	}};
	border: none;
	border-radius: 8px;
	padding: 0.75rem 1.5rem;
	color: white;
	font-weight: 600;
	cursor: pointer;
	display: flex;
	align-items: center;
	gap: 0.5rem;
	font-size: 0.9rem;
	transition: all 0.2s ease;
	box-shadow: ${(props) => props.theme.shadows.small};

	&:hover {
		transform: translateY(-2px);
		box-shadow: ${(props) => props.theme.shadows.medium};
	}

	&:disabled {
		opacity: 0.6;
		cursor: not-allowed;
		transform: none;
	}
`;

const SidePanel = styled(motion.div)`
	display: flex;
	flex-direction: column;
	gap: 1rem;

	@media (max-width: 1200px) {
		grid-column: 1;
	}
`;

const LoadingOverlay = styled(motion.div)`
	position: fixed;
	top: 0;
	left: 0;
	right: 0;
	bottom: 0;
	background: rgba(10, 14, 26, 0.9);
	display: flex;
	align-items: center;
	justify-content: center;
	z-index: 9999;
`;

const LoadingContent = styled.div`
	text-align: center;
	color: ${(props) => props.theme.colors.text};
`;

const SpinningLoader = styled(motion.div)`
	width: 4rem;
	height: 4rem;
	border: 3px solid ${(props) => props.theme.colors.border};
	border-top: 3px solid ${(props) => props.theme.colors.primary};
	border-radius: 50%;
	margin: 0 auto 1rem;
`;

function App() {
	// State
	const [code, setCode] = useState(`#include <stdio.h>
#include <string.h>

void vulnerableFunction(char* userInput) {
    char buffer[64];
    strcpy(buffer, userInput);  // Potential buffer overflow
    printf("Input: %s\\n", buffer);
}

int main() {
    char input[256];
    printf("Enter some text: ");
    gets(input);  // Dangerous function
    vulnerableFunction(input);
    return 0;
}`);

	const [language, setLanguage] = useState("c");
	const [isAnalyzing, setIsAnalyzing] = useState(false);
	const [vulnerabilityResult, setVulnerabilityResult] = useState(null);
	const [attackSnippets, setAttackSnippets] = useState([]);
	const [bestAttackSnippet, setBestAttackSnippet] = useState("");
	const [isFgaRunning, setIsFgaRunning] = useState(false);
	const [fgaProgress, setFgaProgress] = useState(null);
	const [isAttacking, setIsAttacking] = useState(false);
	const [attackResult, setAttackResult] = useState(null);
	const [injectedCode, setInjectedCode] = useState("");
	const [injectionInfo, setInjectionInfo] = useState(null);
	const [fgaParameters, setFgaParameters] = useState(null);

	const progressIntervalRef = useRef(null);

	// Effects
	useEffect(() => {
		loadAttackPool();
	}, []);

	useEffect(() => {
		if (isFgaRunning) {
			// Start polling immediately, then every 1 second
			updateFgaProgress();
			progressIntervalRef.current = setInterval(updateFgaProgress, 1000);
		} else {
			if (progressIntervalRef.current) {
				clearInterval(progressIntervalRef.current);
				progressIntervalRef.current = null;
			}
		}

		return () => {
			if (progressIntervalRef.current) {
				clearInterval(progressIntervalRef.current);
				progressIntervalRef.current = null;
			}
		};
	}, [isFgaRunning]);

	// API Functions
	const loadAttackPool = async () => {
		try {
			const response = await vulnerabilityService.getAttackPool();
			setAttackSnippets(response.attack_snippets);
			setBestAttackSnippet(response.best_snippet);
		} catch (error) {
			toast.error("Failed to load attack pool");
			console.error("Attack pool error:", error);
		}
	};

	const analyzeVulnerability = async () => {
		if (!code.trim()) {
			toast.error("Please enter some code to analyze");
			return;
		}

		setIsAnalyzing(true);
		setVulnerabilityResult(null);

		try {
			const result = await vulnerabilityService.analyzeCode({
				code: code,
				language: language,
			});

			setVulnerabilityResult(result);

			if (result.is_vulnerable) {
				toast.success("Vulnerability detected!", {
					icon: "🔍",
					style: {
						borderLeft: "4px solid #ff4757",
					},
				});
			} else {
				toast.success("Code appears secure", {
					icon: "🛡️",
					style: {
						borderLeft: "4px solid #00ff88",
					},
				});
			}
		} catch (error) {
			toast.error("Analysis failed: " + error.message);
			console.error("Analysis error:", error);
		} finally {
			setIsAnalyzing(false);
		}
	};

	const startFgaSelection = async () => {
		setIsFgaRunning(true);
		setFgaProgress(null);

		try {
			// Get extracted function from vulnerability result if available
			let extractedFunction = null;
			if (vulnerabilityResult?.extraction_info?.selected_function?.code) {
				extractedFunction = vulnerabilityResult.extraction_info.selected_function.code;
				console.log("Using extracted function for FGA:", extractedFunction.substring(0, 100) + "...");
			} else {
				console.log("No extracted function available, FGA will use default test function");
			}

			const result = await vulnerabilityService.startFgaSelection(fgaParameters, extractedFunction);

			// Show a single startup notification
			if (result && (result.status === "running" || result.message)) {
				const hasExtractedFunction = extractedFunction !== null;
				const functionName = vulnerabilityResult?.extraction_info?.selected_function?.name || "test function";

				toast.success(
					`🎯 FGA started with ${
						hasExtractedFunction ? `function: ${functionName}` : "default test function"
					}`,
					{
						duration: 4000,
						style: {
							borderLeft: "4px solid #00d4ff",
						},
					}
				);

				console.log("FGA started with parameters:", result.parameters);
			} else {
				throw new Error("Unexpected response from server");
			}
		} catch (error) {
			toast.error("Failed to start FGA selection: " + (error.message || "Unknown error"));
			setIsFgaRunning(false);
			console.error("FGA start error:", error);
		}
	};

	const updateFgaProgress = async () => {
		try {
			const progress = await vulnerabilityService.getFgaProgress();
			setFgaProgress(progress);

			// Handle completion
			if (progress.status === "completed" && isFgaRunning) {
				setIsFgaRunning(false);

				// Show completion notification with results
				const attackRate = progress.attack_success_rate || 0;
				const fitness = progress.best_fitness || 0;

				toast.success(
					`✅ FGA completed! Best fitness: ${fitness.toFixed(3)}, Attack rate: ${(attackRate * 100).toFixed(
						1
					)}%`,
					{
						duration: 6000,
						style: {
							borderLeft: "4px solid #00ff88",
						},
					}
				);

				// Automatically load the best attack snippet
				try {
					const bestSnippetResult = await vulnerabilityService.getBestAttackSnippet();
					if (bestSnippetResult?.best_snippet && bestSnippetResult.best_snippet !== bestAttackSnippet) {
						setBestAttackSnippet(bestSnippetResult.best_snippet);
						console.log(
							"✅ Best attack snippet loaded automatically:",
							bestSnippetResult.best_snippet.substring(0, 100) + "..."
						);
					}
				} catch (snippetError) {
					console.error("Failed to load best snippet after FGA completion:", snippetError);
					toast.error("FGA completed but failed to load best snippet", {
						duration: 3000,
					});
				}
			} else if (progress.status === "error" && isFgaRunning) {
				setIsFgaRunning(false);
				toast.error(`❌ FGA selection failed: ${progress.phase_description || "Unknown error"}`, {
					duration: 5000,
				});
			}
		} catch (error) {
			console.error("Progress update error:", error);
			// Only show error toast if we've been running for a while (to avoid spam on startup)
			if (isFgaRunning && fgaProgress && fgaProgress.time_elapsed > 5) {
				toast.error("Failed to update FGA progress");
			}
		}
	};

	const loadBestSnippet = async () => {
		try {
			const result = await vulnerabilityService.getBestAttackSnippet();

			if (result?.best_snippet) {
				setBestAttackSnippet(result.best_snippet);

				// Show different messages based on FGA status
				const isFromFga = result.source === "fga_algorithm";
				const fitness = result.fitness_score || 0;
				const attackRate = result.attack_success_rate || 0;

				if (isFromFga && fitness > 0) {
					toast.success(
						`📥 FGA snippet loaded! Fitness: ${fitness.toFixed(3)}, Success: ${(attackRate * 100).toFixed(
							1
						)}%`,
						{
							icon: "🎯",
							duration: 4000,
							style: {
								borderLeft: "4px solid #00d4ff",
							},
						}
					);
				} else {
					toast.success("📥 Attack snippet loaded", {
						duration: 3000,
					});
				}

				console.log("Best snippet loaded:", {
					source: result.source,
					fitness: fitness,
					attackRate: attackRate,
					snippet: result.best_snippet.substring(0, 100) + "...",
				});
			} else {
				toast.error("No attack snippet available");
			}
		} catch (error) {
			toast.error("Failed to load best snippet");
			console.error("Best snippet error:", error);
		}
	};

	const injectAttackSnippet = () => {
		if (!bestAttackSnippet.trim()) {
			toast.error("No attack snippet available");
			return;
		}

		// Simple injection logic - insert at the beginning of the function
		const lines = code.split("\n");
		let insertionPoint = 1;

		// Find a good insertion point
		for (let i = 0; i < lines.length; i++) {
			const line = lines[i].trim().toLowerCase();
			if (line.includes("void ") || line.includes("int ") || line.includes("char ")) {
				if (line.includes("{")) {
					insertionPoint = i + 1;
					break;
				}
			}
		}

		const injectedLines = [
			...lines.slice(0, insertionPoint),
			`    ${bestAttackSnippet}  // INJECTED_ATTACK_CODE`,
			...lines.slice(insertionPoint),
		];

		const injectedCodeResult = injectedLines.join("\n");
		setCode(injectedCodeResult);
		setInjectedCode(injectedCodeResult);
		setInjectionInfo({
			injection_line: insertionPoint + 1,
			attack_snippet: bestAttackSnippet,
			marker: "// INJECTED_ATTACK_CODE",
		});

		toast.success("Attack snippet injected into code", {
			icon: "💉",
			style: {
				borderLeft: "4px solid #ffeb3b",
			},
		});
	};

	const performAdversarialAttack = async () => {
		if (!injectedCode || !bestAttackSnippet) {
			toast.error("Please inject an attack snippet first");
			return;
		}

		setIsAttacking(true);
		setAttackResult(null);

		try {
			// Prepare attack request with original prediction from vulnerability analysis
			const attackRequest = {
				original_code: code.replace(/.*\/\/ INJECTED_ATTACK_CODE.*\n?/g, ""),
				language: language,
				attack_snippet: bestAttackSnippet,
			};

			// Include original prediction if we have vulnerability analysis results
			if (
				vulnerabilityResult &&
				vulnerabilityResult.probabilities &&
				vulnerabilityResult.confidence !== undefined
			) {
				attackRequest.original_prediction = {
					prediction: vulnerabilityResult.is_vulnerable ? 1 : 0,
					confidence: vulnerabilityResult.confidence,
					probabilities: vulnerabilityResult.probabilities,
				};
				console.log(
					"Using stored vulnerability analysis result for attack:",
					attackRequest.original_prediction
				);
			} else {
				console.log("No stored vulnerability result found, backend will re-predict");
			}

			const result = await vulnerabilityService.performAttack(attackRequest);

			setAttackResult(result);

			if (result.attack_success) {
				toast.success("🎯 Attack successful! Model prediction changed", {
					icon: "💥",
					style: {
						borderLeft: "4px solid #ff4757",
					},
				});
			} else {
				toast.error("Attack failed - model prediction unchanged", {
					icon: "🛡️",
				});
			}
		} catch (error) {
			toast.error("Attack failed: " + error.message);
			console.error("Attack error:", error);
		} finally {
			setIsAttacking(false);
		}
	};

	const handleFgaParametersChange = (parameters) => {
		setFgaParameters(parameters);
		toast.success("FGA parameters updated", {
			icon: "⚙️",
			style: {
				borderLeft: "4px solid #00d4ff",
			},
		});
	};

	const handleCodeFileUploaded = (uploadedCode, detectedLanguage, filename) => {
		setCode(uploadedCode);
		setLanguage(detectedLanguage);
		setInjectedCode(null);
		setInjectionInfo(null);
		setVulnerabilityResult(null);
		setAttackResult(null);

		toast.success(`Code file "${filename}" loaded successfully`, {
			icon: "📁",
			style: {
				borderLeft: "4px solid #00ff88",
			},
		});
	};

	const handleAttackPoolUploaded = async (uploadResult) => {
		// Reload the attack pool from the server to get the updated data
		try {
			await loadAttackPool();
			toast.success(`Attack pool updated with ${uploadResult.valid_entries} entries`, {
				icon: "⚡",
				style: {
					borderLeft: "4px solid #7c3aed",
				},
			});
		} catch (error) {
			console.error("Failed to reload attack pool:", error);
			toast.error("Attack pool uploaded but failed to reload");
		}
	};

	return (
		<ThemeProvider theme={theme}>
			<GlobalStyle />
			<AppContainer>
				<Toaster
					position="top-right"
					toastOptions={{
						duration: 4000,
						style: {
							background: theme.colors.surface,
							color: theme.colors.text,
							border: `1px solid ${theme.colors.border}`,
							borderRadius: "8px",
						},
					}}
				/>

				<Header initial={{ y: -100 }} animate={{ y: 0 }} transition={{ duration: 0.5 }}>
					<Logo>
						<LogoIcon />
						EatVul Security Analyzer
					</Logo>

					<HeaderActions>
						<StatusBadge
							status={vulnerabilityResult?.is_vulnerable ? "vulnerable" : "secure"}
							text={
								vulnerabilityResult
									? vulnerabilityResult.is_vulnerable
										? "Vulnerable"
										: "Secure"
									: "Ready"
							}
						/>
					</HeaderActions>
				</Header>

				<MainContent>
					<EditorSection
						initial={{ x: -50, opacity: 0 }}
						animate={{ x: 0, opacity: 1 }}
						transition={{ duration: 0.5, delay: 0.1 }}>
						<EditorHeader>
							<EditorTitle>
								<Code size={20} />
								Code Editor
							</EditorTitle>

							<div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
								<LanguageSelector value={language} onChange={(e) => setLanguage(e.target.value)}>
									<option value="c">C</option>
									<option value="cpp">C++</option>
									<option value="javascript">JavaScript</option>
									<option value="java">Java</option>
								</LanguageSelector>

								<ActionButton
									variant="primary"
									onClick={analyzeVulnerability}
									disabled={isAnalyzing}
									whileHover={{ scale: 1.05 }}
									whileTap={{ scale: 0.95 }}>
									{isAnalyzing ? <Loader className="animate-spin" size={16} /> : <Shield size={16} />}
									{isAnalyzing ? "Analyzing..." : "Detect Vulnerability"}
								</ActionButton>
							</div>
						</EditorHeader>

						<CodeEditor
							code={code}
							language={language}
							onChange={setCode}
							vulnerableLines={vulnerabilityResult?.vulnerable_lines || []}
							injectionInfo={injectionInfo}
						/>
					</EditorSection>

					<SidePanel
						initial={{ x: 50, opacity: 0 }}
						animate={{ x: 0, opacity: 1 }}
						transition={{ duration: 0.5, delay: 0.2 }}>
						<FileUploadPanel
							onCodeUploaded={handleCodeFileUploaded}
							onAttackPoolUploaded={handleAttackPoolUploaded}
						/>

						<VulnerabilityPanel result={vulnerabilityResult} isAnalyzing={isAnalyzing} />

						<FGAParameterPanel onParametersChange={handleFgaParametersChange} isRunning={isFgaRunning} />

						<AdversarialPanel
							attackSnippets={attackSnippets}
							bestAttackSnippet={bestAttackSnippet}
							onStartFga={startFgaSelection}
							onLoadBestSnippet={loadBestSnippet}
							onInjectSnippet={injectAttackSnippet}
							onPerformAttack={performAdversarialAttack}
							isFgaRunning={isFgaRunning}
							isAttacking={isAttacking}
							attackResult={attackResult}
						/>

						{(isFgaRunning || fgaProgress) && (
							<ProgressVisualization progress={fgaProgress} isRunning={isFgaRunning} />
						)}
					</SidePanel>
				</MainContent>

				<AnimatePresence>
					{(isAnalyzing || isAttacking) && (
						<LoadingOverlay initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
							<LoadingContent>
								<SpinningLoader
									animate={{ rotate: 360 }}
									transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
								/>
								<h3>
									{isAnalyzing ? "Analyzing Code Security..." : "Performing Adversarial Attack..."}
								</h3>
								<p style={{ color: theme.colors.textSecondary, marginTop: "0.5rem" }}>
									{isAnalyzing
										? "Running CodeBERT vulnerability detection"
										: "Testing attack effectiveness"}
								</p>
							</LoadingContent>
						</LoadingOverlay>
					)}
				</AnimatePresence>
			</AppContainer>
		</ThemeProvider>
	);
}

export default App;
