import React, { useState, useEffect } from "react";
import styled from "styled-components";
import { motion } from "framer-motion";
import {
	Settings,
	ChevronDown,
	ChevronRight,
	RefreshCw,
	Save,
	RotateCcw,
	HelpCircle,
	Target,
	Users,
	Layers,
	TrendingDown,
	Zap,
	Minus,
	AlertCircle,
} from "lucide-react";
import { vulnerabilityService } from "../services/api";

const PanelContainer = styled(motion.div)`
	background: ${(props) => props.theme.colors.surface};
	border-radius: 12px;
	border: 1px solid ${(props) => props.theme.colors.border};
	overflow: hidden;
	box-shadow: ${(props) => props.theme.shadows.medium};
`;

const PanelHeader = styled.div`
	padding: 1rem 1.5rem;
	border-bottom: 1px solid ${(props) => props.theme.colors.border};
	background: ${(props) => props.theme.colors.surfaceLight};
	display: flex;
	align-items: center;
	justify-content: space-between;
	cursor: pointer;
`;

const PanelTitle = styled.div`
	display: flex;
	align-items: center;
	gap: 0.5rem;
	font-weight: 600;
	color: ${(props) => props.theme.colors.text};
`;

const PanelContent = styled(motion.div)`
	padding: 1.5rem;
`;

const ParameterGroup = styled.div`
	margin-bottom: 1.5rem;

	&:last-child {
		margin-bottom: 0;
	}
`;

const ParameterLabel = styled.label`
	display: block;
	font-weight: 600;
	color: ${(props) => props.theme.colors.text};
	margin-bottom: 0.5rem;
	font-size: 0.9rem;
`;

const ParameterDescription = styled.div`
	font-size: 0.8rem;
	color: ${(props) => props.theme.colors.textSecondary};
	margin-bottom: 0.75rem;
	line-height: 1.4;
`;

const ParameterInput = styled.input`
	width: 100%;
	background: ${(props) => props.theme.colors.background};
	border: 1px solid ${(props) => props.theme.colors.border};
	border-radius: 6px;
	padding: 0.75rem;
	color: ${(props) => props.theme.colors.text};
	font-size: 0.9rem;
	transition: border-color 0.2s ease;

	&:focus {
		outline: none;
		border-color: ${(props) => props.theme.colors.primary};
	}

	&:invalid {
		border-color: ${(props) => props.theme.colors.error};
	}
`;

const ParameterRange = styled.div`
	display: flex;
	justify-content: between;
	font-size: 0.8rem;
	color: ${(props) => props.theme.colors.textSecondary};
	margin-top: 0.25rem;
`;

const ParameterSlider = styled.input`
	width: 100%;
	margin: 0.5rem 0;
	-webkit-appearance: none;
	appearance: none;
	height: 4px;
	background: ${(props) => props.theme.colors.border};
	border-radius: 2px;
	outline: none;

	&::-webkit-slider-thumb {
		-webkit-appearance: none;
		appearance: none;
		width: 16px;
		height: 16px;
		background: ${(props) => props.theme.colors.primary};
		border-radius: 50%;
		cursor: pointer;
	}

	&::-moz-range-thumb {
		width: 16px;
		height: 16px;
		background: ${(props) => props.theme.colors.primary};
		border-radius: 50%;
		cursor: pointer;
		border: none;
	}
`;

const ButtonGroup = styled.div`
	display: flex;
	gap: 0.75rem;
	margin-top: 1.5rem;
`;

const ActionButton = styled(motion.button)`
	background: ${(props) => {
		if (props.variant === "primary") return props.theme.gradients.cyber;
		if (props.variant === "secondary") return props.theme.colors.surfaceLight;
		return props.theme.colors.border;
	}};
	border: none;
	border-radius: 6px;
	padding: 0.6rem 1rem;
	color: ${(props) => (props.variant === "primary" ? "white" : props.theme.colors.text)};
	font-weight: 600;
	cursor: pointer;
	display: flex;
	align-items: center;
	gap: 0.5rem;
	font-size: 0.8rem;
	transition: all 0.2s ease;
	box-shadow: ${(props) => props.theme.shadows.small};
	flex: 1;
	justify-content: center;

	&:hover {
		transform: translateY(-1px);
		box-shadow: ${(props) => props.theme.shadows.medium};
	}

	&:disabled {
		opacity: 0.6;
		cursor: not-allowed;
		transform: none;
	}
`;

const ValidationMessage = styled.div`
	display: flex;
	align-items: center;
	gap: 0.5rem;
	padding: 0.75rem;
	background: rgba(255, 71, 87, 0.1);
	border: 1px solid ${(props) => props.theme.colors.error};
	border-radius: 6px;
	color: ${(props) => props.theme.colors.error};
	font-size: 0.8rem;
	margin-top: 1rem;
`;

const PresetButtons = styled.div`
	display: grid;
	grid-template-columns: 1fr 1fr 1fr;
	gap: 0.5rem;
	margin-bottom: 1rem;
`;

const PresetButton = styled(motion.button)`
	background: ${(props) => (props.active ? props.theme.colors.primary : props.theme.colors.surfaceLight)};
	border: 1px solid ${(props) => (props.active ? props.theme.colors.primary : props.theme.colors.border)};
	border-radius: 4px;
	padding: 0.5rem;
	color: ${(props) => (props.active ? "white" : props.theme.colors.text)};
	font-size: 0.8rem;
	cursor: pointer;
	transition: all 0.2s ease;

	&:hover {
		background: ${(props) => props.theme.colors.primary};
		color: white;
	}
`;

const FGAParameterPanel = ({ onParametersChange, isRunning = false }) => {
	const [isExpanded, setIsExpanded] = useState(false);
	const [parameters, setParameters] = useState({});
	const [parameterInfo, setParameterInfo] = useState({});
	const [validationErrors, setValidationErrors] = useState([]);
	const [currentPreset, setCurrentPreset] = useState("balanced");

	// Parameter presets
	const presets = {
		fast: {
			pop_size: 10,
			clusters: 2,
			max_generations: 20,
			decay_rate: 1.0,
			alpha: 1.5,
			penalty: 0.02,
			verbose: 1,
		},
		balanced: {
			pop_size: 20,
			clusters: 3,
			max_generations: 50,
			decay_rate: 1.5,
			alpha: 2.0,
			penalty: 0.01,
			verbose: 1,
		},
		thorough: {
			pop_size: 50,
			clusters: 5,
			max_generations: 100,
			decay_rate: 2.0,
			alpha: 2.5,
			penalty: 0.005,
			verbose: 1,
		},
	};

	useEffect(() => {
		loadParameterInfo();
	}, []);

	const loadParameterInfo = async () => {
		try {
			const data = await vulnerabilityService.getFgaParameters();
			setParameterInfo(data);
			if (data.default_parameters) {
				setParameters(data.default_parameters);
			}
		} catch (error) {
			console.error("Failed to load parameter info:", error);
		}
	};

	const handleParameterChange = (key, value) => {
		const numValue = parseFloat(value);
		const newParameters = { ...parameters, [key]: numValue };
		setParameters(newParameters);
		validateParameters(newParameters);
	};

	const validateParameters = (params) => {
		const errors = [];

		// Basic validation - just check if values are positive numbers
		Object.entries(params).forEach(([key, value]) => {
			if (isNaN(value) || value < 0) {
				errors.push(`${key}: must be a positive number`);
			}
		});

		setValidationErrors(errors);
		return errors.length === 0;
	};

	const applyPreset = (presetName) => {
		const preset = presets[presetName];
		setParameters(preset);
		setCurrentPreset(presetName);
		validateParameters(preset);
	};

	const resetToDefaults = () => {
		if (parameterInfo.default_parameters) {
			setParameters(parameterInfo.default_parameters);
			setCurrentPreset("balanced");
			setValidationErrors([]);
		}
	};

	const saveParameters = () => {
		if (validateParameters(parameters)) {
			onParametersChange(parameters);
		}
	};

	const renderParameterInput = (key, value) => {
		const description = parameterInfo.parameter_descriptions?.[key];

		return (
			<ParameterGroup key={key}>
				<ParameterLabel htmlFor={key}>
					{key.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())}
				</ParameterLabel>
				<ParameterDescription>{description}</ParameterDescription>

				{/* Number input without min/max constraints */}
				<ParameterInput
					id={key}
					type="number"
					step={key === "verbose" ? 1 : key.includes("rate") || key.includes("alpha") ? 0.1 : 1}
					value={value}
					onChange={(e) => handleParameterChange(key, e.target.value)}
					disabled={isRunning}
				/>
			</ParameterGroup>
		);
	};

	return (
		<PanelContainer
			initial={{ opacity: 0, y: 20 }}
			animate={{ opacity: 1, y: 0 }}
			transition={{ duration: 0.3, delay: 0.3 }}>
			<PanelHeader onClick={() => setIsExpanded(!isExpanded)}>
				<PanelTitle>
					<Settings size={20} />
					FGA Parameters
				</PanelTitle>
				{isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
			</PanelHeader>

			{isExpanded && (
				<PanelContent
					initial={{ height: 0 }}
					animate={{ height: "auto" }}
					exit={{ height: 0 }}
					transition={{ duration: 0.3 }}>
					{/* Preset Buttons */}
					<div style={{ marginBottom: "1rem" }}>
						<ParameterLabel>Quick Presets</ParameterLabel>
						<PresetButtons>
							{Object.keys(presets).map((presetName) => (
								<PresetButton
									key={presetName}
									active={currentPreset === presetName}
									onClick={() => applyPreset(presetName)}
									disabled={isRunning}
									whileHover={{ scale: 1.02 }}
									whileTap={{ scale: 0.98 }}>
									{presetName.charAt(0).toUpperCase() + presetName.slice(1)}
								</PresetButton>
							))}
						</PresetButtons>
					</div>

					{/* Parameter Inputs */}
					{Object.entries(parameters).map(([key, value]) => renderParameterInput(key, value))}

					{/* Validation Errors */}
					{validationErrors.length > 0 && (
						<ValidationMessage>
							<AlertCircle size={16} />
							<div>
								{validationErrors.map((error, index) => (
									<div key={index}>{error}</div>
								))}
							</div>
						</ValidationMessage>
					)}

					{/* Action Buttons */}
					<ButtonGroup>
						<ActionButton
							onClick={resetToDefaults}
							disabled={isRunning}
							whileHover={{ scale: 1.02 }}
							whileTap={{ scale: 0.98 }}>
							<RotateCcw size={14} />
							Reset
						</ActionButton>

						<ActionButton
							variant="primary"
							onClick={saveParameters}
							disabled={isRunning || validationErrors.length > 0}
							whileHover={{ scale: 1.02 }}
							whileTap={{ scale: 0.98 }}>
							<Save size={14} />
							Apply
						</ActionButton>
					</ButtonGroup>
				</PanelContent>
			)}
		</PanelContainer>
	);
};

export default FGAParameterPanel;
