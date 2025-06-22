import React from "react";
import styled from "styled-components";
import { motion } from "framer-motion";
import { Code, Settings, Shield, AlertTriangle, Info, CheckCircle, FileText, Hash, Target } from "lucide-react";

const ExtractionContainer = styled(motion.div)`
	margin-top: 1.5rem;
	border-top: 1px solid ${(props) => props.theme.colors.border};
	padding-top: 1.5rem;
`;

const SectionTitle = styled.div`
	display: flex;
	align-items: center;
	gap: 0.75rem;
	font-weight: 600;
	color: ${(props) => props.theme.colors.text};
	margin-bottom: 1.5rem;
	font-size: 1.1rem;
`;

const FunctionCard = styled.div`
	background: ${(props) => props.theme.colors.background};
	border: 1px solid ${(props) => props.theme.colors.border};
	border-radius: 8px;
	padding: 1.5rem;
	margin-bottom: 1.5rem;
	box-shadow: ${(props) => props.theme.shadows.small};
`;

const CardHeader = styled.div`
	display: flex;
	align-items: flex-start;
	gap: 0.75rem;
	margin-bottom: 1.25rem;
`;

const CardTitle = styled.div`
	font-weight: 600;
	color: ${(props) => props.theme.colors.text};
	font-size: 1rem;
`;

const CardSubtitle = styled.div`
	font-size: 0.875rem;
	color: ${(props) => props.theme.colors.textSecondary};
	margin-top: 0.25rem;
	line-height: 1.4;
`;

const ChipContainer = styled.div`
	display: flex;
	flex-wrap: wrap;
	gap: 0.75rem;
	margin-bottom: 1.25rem;
	align-items: center;
`;

const Chip = styled.div`
	display: inline-flex;
	align-items: center;
	gap: 0.375rem;
	padding: 0.375rem 0.75rem;
	border-radius: 16px;
	font-size: 0.8rem;
	font-weight: 500;
	background: ${(props) => {
		switch (props.variant) {
			case "primary":
				return props.theme.colors.primary + "15";
			case "success":
				return props.theme.colors.success + "15";
			case "warning":
				return props.theme.colors.warning + "15";
			case "error":
				return props.theme.colors.vulnerability + "15";
			case "info":
				return props.theme.colors.info + "15";
			default:
				return props.theme.colors.surfaceLight;
		}
	}};
	color: ${(props) => {
		switch (props.variant) {
			case "primary":
				return props.theme.colors.primary;
			case "success":
				return props.theme.colors.success;
			case "warning":
				return props.theme.colors.warning;
			case "error":
				return props.theme.colors.vulnerability;
			case "info":
				return props.theme.colors.info;
			default:
				return props.theme.colors.textSecondary;
		}
	}};
	border: 1px solid
		${(props) => {
			switch (props.variant) {
				case "primary":
					return props.theme.colors.primary + "30";
				case "success":
					return props.theme.colors.success + "30";
				case "warning":
					return props.theme.colors.warning + "30";
				case "error":
					return props.theme.colors.vulnerability + "30";
				case "info":
					return props.theme.colors.info + "30";
				default:
					return props.theme.colors.border;
			}
		}};
`;

const CodeBlock = styled.div`
	background: ${(props) => props.theme.colors.surface};
	border: 1px solid ${(props) => props.theme.colors.border};
	border-radius: 6px;
	padding: 1rem;
	font-family: "Consolas", "Monaco", "Courier New", monospace;
	font-size: 0.85rem;
	max-height: 300px;
	overflow-y: auto;
	white-space: pre-wrap;
	color: ${(props) => props.theme.colors.text};
	line-height: 1.4;
	margin-top: 0.75rem;
`;

const AlertBox = styled.div`
	display: flex;
	align-items: flex-start;
	gap: 0.75rem;
	padding: 1rem;
	border-radius: 8px;
	margin-bottom: 1.5rem;
	background: ${(props) => {
		switch (props.severity) {
			case "warning":
				return props.theme.colors.warning + "10";
			case "error":
				return props.theme.colors.vulnerability + "10";
			case "success":
				return props.theme.colors.success + "10";
			default:
				return props.theme.colors.info + "10";
		}
	}};
	border: 1px solid
		${(props) => {
			switch (props.severity) {
				case "warning":
					return props.theme.colors.warning + "30";
				case "error":
					return props.theme.colors.vulnerability + "30";
				case "success":
					return props.theme.colors.success + "30";
				default:
					return props.theme.colors.info + "30";
			}
		}};
`;

const AlertText = styled.div`
	font-size: 0.9rem;
	color: ${(props) => props.theme.colors.text};
	line-height: 1.5;
`;

const FunctionList = styled.div`
	display: grid;
	grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
	gap: 0.75rem;
	margin-top: 0.75rem;
`;

const FunctionItem = styled.div`
	padding: 0.75rem;
	background: ${(props) => props.theme.colors.surface};
	border-radius: 6px;
	border: 1px solid ${(props) => props.theme.colors.border};
	transition: all 0.2s ease;

	&:hover {
		border-color: ${(props) => props.theme.colors.primary};
		background: ${(props) => props.theme.colors.primary}05;
	}
`;

const FunctionName = styled.div`
	font-weight: 500;
	color: ${(props) => props.theme.colors.text};
	font-size: 0.9rem;
`;

const FunctionMeta = styled.div`
	font-size: 0.75rem;
	color: ${(props) => props.theme.colors.primary};
	margin-top: 0.25rem;
	font-weight: 500;
`;

const FunctionExtractionPanel = ({ extractionInfo, onFunctionSelect }) => {
	if (!extractionInfo) {
		return null;
	}

	const renderSelectedFunction = () => {
		const selectedFunc = extractionInfo.selected_function;
		if (!selectedFunc) return null;

		const getExtractionMethodDescription = (method) => {
			switch (method) {
				case "vulnerability_scoring":
					return "Selected based on vulnerability indicators";
				case "no_functions_found":
					return "No functions found, using full code";
				case "fallback_due_to_error":
					return "Extraction failed, using full code";
				case "full_code_non_c":
					return "Non-C language, using full code";
				default:
					return "Unknown extraction method";
			}
		};

		const getVulnerabilityVariant = (score) => {
			if (score === 0) return "default";
			if (score <= 2) return "success";
			if (score <= 5) return "warning";
			return "error";
		};

		return (
			<FunctionCard>
				<CardHeader>
					<Code size={20} color="#4F46E5" />
					<div>
						<CardTitle>Selected Function for Analysis</CardTitle>
						<CardSubtitle>{getExtractionMethodDescription(selectedFunc.extraction_method)}</CardSubtitle>
					</div>
				</CardHeader>

				<ChipContainer>
					<Chip variant="primary">
						<FileText size={12} />
						Function: {selectedFunc.name}
					</Chip>
					{selectedFunc.vulnerability_score !== undefined && (
						<Chip variant={getVulnerabilityVariant(selectedFunc.vulnerability_score)}>
							<Shield size={12} />
							Risk Score: {selectedFunc.vulnerability_score}
						</Chip>
					)}
					{selectedFunc.start_line && selectedFunc.end_line && (
						<Chip variant="info">
							<Hash size={12} />
							Lines: {selectedFunc.start_line}-{selectedFunc.end_line}
						</Chip>
					)}
					{selectedFunc.is_main && (
						<Chip variant="primary">
							<Target size={12} />
							Main Function
						</Chip>
					)}
				</ChipContainer>

				{selectedFunc.code && selectedFunc.code !== extractionInfo.original_code && (
					<div>
						<CardTitle style={{ fontSize: "0.95rem", marginBottom: "0.75rem" }}>
							Extracted Function Code:
						</CardTitle>
						<CodeBlock>{selectedFunc.code}</CodeBlock>
					</div>
				)}
			</FunctionCard>
		);
	};

	const renderAllFunctions = () => {
		const summary = extractionInfo.all_functions;
		if (!summary || summary.total_functions === 0) {
			return (
				<AlertBox severity="info">
					<Info size={16} />
					<AlertText>No functions detected in the code. The entire code will be analyzed as-is.</AlertText>
				</AlertBox>
			);
		}

		return (
			<FunctionCard>
				<CardHeader>
					<Settings size={20} color="#059669" />
					<div>
						<CardTitle>Function Analysis Summary</CardTitle>
						<CardSubtitle>{summary.total_functions} function(s) found</CardSubtitle>
					</div>
				</CardHeader>

				<ChipContainer>
					<Chip variant="primary">Total: {summary.total_functions}</Chip>
					{summary.main_function_present && <Chip variant="info">Has main()</Chip>}
					{summary.most_vulnerable && <Chip variant="warning">Most Risky: {summary.most_vulnerable}</Chip>}
					<Chip variant="default">Avg Risk: {summary.average_vulnerability_score?.toFixed(1) || 0}</Chip>
				</ChipContainer>

				{summary.function_names && summary.function_names.length > 0 && (
					<div>
						<CardTitle style={{ fontSize: "0.95rem", marginBottom: "0.75rem" }}>Functions Found:</CardTitle>
						<FunctionList>
							{summary.function_names.map((name, index) => (
								<FunctionItem key={index}>
									<FunctionName>{name}</FunctionName>
									{name === summary.most_vulnerable && (
										<FunctionMeta>Highest risk score</FunctionMeta>
									)}
								</FunctionItem>
							))}
						</FunctionList>
					</div>
				)}
			</FunctionCard>
		);
	};

	const renderErrorInfo = () => {
		if (extractionInfo.error) {
			return (
				<AlertBox severity="warning">
					<AlertTriangle size={16} />
					<div>
						<AlertText>
							<strong>Function Extraction Warning:</strong> {extractionInfo.error}
						</AlertText>
						<AlertText style={{ marginTop: "0.5rem" }}>
							Analysis will proceed using the full code instead.
						</AlertText>
					</div>
				</AlertBox>
			);
		}
		return null;
	};

	const renderNonCLanguage = () => {
		if (extractionInfo.extraction_method === "full_code_non_c") {
			return (
				<AlertBox severity="info">
					<Info size={16} />
					<AlertText>
						Function extraction is only available for C/C++ code. Analyzing the entire{" "}
						{extractionInfo.language} code.
					</AlertText>
				</AlertBox>
			);
		}
		return null;
	};

	return (
		<ExtractionContainer
			initial={{ opacity: 0, y: 20 }}
			animate={{ opacity: 1, y: 0 }}
			transition={{ duration: 0.3 }}>
			<SectionTitle>
				<Code size={20} />
				Function Extraction Results
			</SectionTitle>

			{renderErrorInfo()}
			{renderNonCLanguage()}
			{renderSelectedFunction()}
			{renderAllFunctions()}
		</ExtractionContainer>
	);
};

export default FunctionExtractionPanel;
