import React, { useRef, useEffect, useState } from "react";
import Editor from "@monaco-editor/react";
import styled from "styled-components";
import { motion } from "framer-motion";

const EditorContainer = styled.div`
	height: 600px;
	position: relative;
	border-radius: 0 0 12px 12px;
	overflow: hidden;
`;

const TooltipOverlay = styled(motion.div)`
	position: absolute;
	background: ${(props) => props.theme.colors.surface};
	border: 1px solid ${(props) => props.theme.colors.border};
	border-radius: 8px;
	padding: 1rem;
	color: ${(props) => props.theme.colors.text};
	box-shadow: ${(props) => props.theme.shadows.large};
	max-width: 300px;
	z-index: 1000;
	pointer-events: none;
`;

const TooltipTitle = styled.div`
	font-weight: 600;
	color: ${(props) => props.theme.colors.vulnerability};
	margin-bottom: 0.5rem;
	display: flex;
	align-items: center;
	gap: 0.5rem;
`;

const TooltipContent = styled.div`
	font-size: 0.9rem;
	line-height: 1.4;
	color: ${(props) => props.theme.colors.textSecondary};
`;

const TooltipFix = styled.div`
	margin-top: 0.5rem;
	padding-top: 0.5rem;
	border-top: 1px solid ${(props) => props.theme.colors.border};
	font-size: 0.8rem;
	color: ${(props) => props.theme.colors.success};
`;

const CodeEditor = ({ code, language, onChange, vulnerableLines = [], injectionInfo = null }) => {
	const editorRef = useRef(null);
	const decorationsRef = useRef([]);
	const [tooltip, setTooltip] = useState(null);
	const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

	useEffect(() => {
		if (editorRef.current) {
			updateDecorations();
		}
	}, [vulnerableLines, injectionInfo, code]);

	useEffect(() => {
		const handleMouseMove = (e) => {
			setMousePosition({ x: e.clientX, y: e.clientY });
		};

		document.addEventListener("mousemove", handleMouseMove);
		return () => document.removeEventListener("mousemove", handleMouseMove);
	}, []);

	const handleEditorDidMount = (editor, monaco) => {
		editorRef.current = editor;

		// Configure Monaco theme
		monaco.editor.defineTheme("cyberpunk", {
			base: "vs-dark",
			inherit: true,
			rules: [
				{ token: "comment", foreground: "6A9955" },
				{ token: "keyword", foreground: "569CD6" },
				{ token: "string", foreground: "CE9178" },
				{ token: "number", foreground: "B5CEA8" },
				{ token: "type", foreground: "4EC9B0" },
				{ token: "function", foreground: "DCDCAA" },
			],
			colors: {
				"editor.background": "#1a1d29",
				"editor.foreground": "#ffffff",
				"editor.lineHighlightBackground": "#2a2d3a",
				"editor.selectionBackground": "#264F78",
				"editor.inactiveSelectionBackground": "#3A3D41",
				"editorLineNumber.foreground": "#858585",
				"editorLineNumber.activeForeground": "#ffffff",
				"editorCursor.foreground": "#00d4ff",
			},
		});

		monaco.editor.setTheme("cyberpunk");

		// Add hover provider for vulnerability tooltips
		editor.onMouseMove((e) => {
			if (e.target.type === monaco.editor.MouseTargetType.CONTENT_TEXT) {
				const position = e.target.position;
				if (position) {
					const lineNumber = position.lineNumber;
					const vulnerableLine = vulnerableLines.find((vl) => vl.line_number === lineNumber);

					if (vulnerableLine) {
						setTooltip({
							line: lineNumber,
							vulnerability: vulnerableLine,
						});
					} else if (injectionInfo && lineNumber === injectionInfo.injection_line) {
						setTooltip({
							line: lineNumber,
							injection: true,
							snippet: injectionInfo.attack_snippet,
						});
					} else {
						setTooltip(null);
					}
				}
			} else {
				setTooltip(null);
			}
		});

		// Hide tooltip when mouse leaves editor
		editor.onMouseLeave(() => {
			setTooltip(null);
		});

		updateDecorations();
	};

	const updateDecorations = () => {
		if (!editorRef.current) return;

		const editor = editorRef.current;
		const model = editor.getModel();

		if (!model) return;

		// Clear existing decorations
		if (decorationsRef.current.length > 0) {
			decorationsRef.current = editor.deltaDecorations(decorationsRef.current, []);
		}

		const newDecorations = [];

		// Add vulnerability line decorations
		vulnerableLines.forEach((vuln) => {
			if (vuln.line_number && vuln.line_number <= model.getLineCount()) {
				newDecorations.push({
					range: {
						startLineNumber: vuln.line_number,
						startColumn: 1,
						endLineNumber: vuln.line_number,
						endColumn: model.getLineMaxColumn(vuln.line_number),
					},
					options: {
						className: "vulnerability-line",
						isWholeLine: true,
						glyphMarginClassName: "vulnerability-glyph",
						marginClassName: "vulnerability-margin",
						overviewRuler: {
							color: "#ff4757",
							darkColor: "#ff4757",
							position: 1,
						},
						minimap: {
							color: "#ff4757",
							position: 1,
						},
					},
				});
			}
		});

		// Add injection line decoration
		if (injectionInfo && injectionInfo.injection_line) {
			const injectionLine = injectionInfo.injection_line;
			if (injectionLine <= model.getLineCount()) {
				newDecorations.push({
					range: {
						startLineNumber: injectionLine,
						startColumn: 1,
						endLineNumber: injectionLine,
						endColumn: model.getLineMaxColumn(injectionLine),
					},
					options: {
						className: "injection-line",
						isWholeLine: true,
						glyphMarginClassName: "injection-glyph",
						marginClassName: "injection-margin",
						overviewRuler: {
							color: "#ffeb3b",
							darkColor: "#ffeb3b",
							position: 2,
						},
						minimap: {
							color: "#ffeb3b",
							position: 2,
						},
					},
				});
			}
		}

		// Apply decorations
		decorationsRef.current = editor.deltaDecorations([], newDecorations);
	};

	const getLanguageForMonaco = (lang) => {
		switch (lang) {
			case "c":
			case "cpp":
				return "cpp";
			case "javascript":
				return "javascript";
			case "java":
				return "java";
			default:
				return "cpp";
		}
	};

	return (
		<>
			<EditorContainer>
				<Editor
					height="100%"
					language={getLanguageForMonaco(language)}
					value={code}
					onChange={onChange}
					onMount={handleEditorDidMount}
					options={{
						theme: "cyberpunk",
						fontSize: 14,
						lineHeight: 20,
						fontFamily: "'Fira Code', 'Consolas', 'Monaco', monospace",
						fontLigatures: true,
						minimap: { enabled: true },
						scrollBeyondLastLine: false,
						automaticLayout: true,
						renderLineHighlight: "all",
						rulers: [80, 120],
						wordWrap: "on",
						lineNumbers: "on",
						glyphMargin: true,
						folding: true,
						foldingHighlight: true,
						showFoldingControls: "always",
						smoothScrolling: true,
						cursorBlinking: "smooth",
						cursorSmoothCaretAnimation: true,
						renderWhitespace: "selection",
						bracketPairColorization: { enabled: true },
					}}
				/>

				{/* Custom CSS for vulnerability and injection highlighting */}
				<style>{`
          .vulnerability-line {
            background-color: rgba(255, 71, 87, 0.15) !important;
            border-left: 3px solid #ff4757 !important;
          }
          
          .vulnerability-glyph {
            background-color: #ff4757;
            width: 8px !important;
            border-radius: 2px;
          }
          
          .vulnerability-margin {
            background-color: rgba(255, 71, 87, 0.1);
          }
          
          .injection-line {
            background-color: rgba(255, 235, 59, 0.15) !important;
            border-left: 3px solid #ffeb3b !important;
          }
          
          .injection-glyph {
            background-color: #ffeb3b;
            width: 8px !important;
            border-radius: 2px;
          }
          
          .injection-margin {
            background-color: rgba(255, 235, 59, 0.1);
          }
          
          .monaco-editor .current-line {
            border: 1px solid rgba(0, 212, 255, 0.3) !important;
          }
          
          .monaco-editor .view-lines {
            font-feature-settings: 'liga' 1, 'calt' 1;
          }
        `}</style>
			</EditorContainer>

			{/* Vulnerability/Injection Tooltip */}
			{tooltip && (
				<TooltipOverlay
					style={{
						left: mousePosition.x + 10,
						top: mousePosition.y - 10,
					}}
					initial={{ opacity: 0, scale: 0.8 }}
					animate={{ opacity: 1, scale: 1 }}
					exit={{ opacity: 0, scale: 0.8 }}
					transition={{ duration: 0.2 }}>
					{tooltip.vulnerability ? (
						<>
							<TooltipTitle>
								🚨 {tooltip.vulnerability.vulnerability_type || "Security Vulnerability"}
							</TooltipTitle>
							<TooltipContent>
								<strong>Line {tooltip.line}:</strong> {tooltip.vulnerability.reason}
							</TooltipContent>
							{tooltip.vulnerability.fix_suggestion && (
								<TooltipFix>
									<strong>💡 Fix:</strong> {tooltip.vulnerability.fix_suggestion}
								</TooltipFix>
							)}
						</>
					) : tooltip.injection ? (
						<>
							<TooltipTitle>💉 Injected Attack Code</TooltipTitle>
							<TooltipContent>
								<strong>Line {tooltip.line}:</strong> This line contains an injected attack snippet to
								test the model's robustness.
							</TooltipContent>
							<TooltipFix>
								<strong>Snippet:</strong> {tooltip.snippet}
							</TooltipFix>
						</>
					) : null}
				</TooltipOverlay>
			)}
		</>
	);
};

export default CodeEditor;
