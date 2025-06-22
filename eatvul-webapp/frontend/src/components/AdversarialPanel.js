import React, { useState } from "react";
import styled from "styled-components";
import { motion, AnimatePresence } from "framer-motion";
import {
	Target,
	Zap,
	Play,
	Download,
	Upload,
	ChevronDown,
	ChevronRight,
	AlertTriangle,
	CheckCircle,
	XCircle,
	TrendingUp,
	Code2,
	Loader,
	Code,
	Shield,
} from "lucide-react";

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

const Section = styled.div`
	margin-bottom: 1.5rem;

	&:last-child {
		margin-bottom: 0;
	}
`;

const SectionTitle = styled.div`
	display: flex;
	align-items: center;
	gap: 0.5rem;
	font-weight: 600;
	color: ${(props) => props.theme.colors.text};
	margin-bottom: 0.75rem;
	font-size: 0.9rem;
`;

const ActionButton = styled(motion.button)`
	background: ${(props) => {
		if (props.variant === "primary") return props.theme.gradients.cyber;
		if (props.variant === "danger") return props.theme.gradients.security;
		if (props.variant === "success") return props.theme.gradients.success;
		return props.theme.colors.surfaceLight;
	}};
	border: none;
	border-radius: 6px;
	padding: 0.6rem 1rem;
	color: white;
	font-weight: 600;
	cursor: pointer;
	display: flex;
	align-items: center;
	gap: 0.5rem;
	font-size: 0.8rem;
	transition: all 0.2s ease;
	box-shadow: ${(props) => props.theme.shadows.small};
	width: 100%;
	justify-content: center;
	margin-bottom: 0.5rem;

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

const SnippetContainer = styled.div`
	background: ${(props) => props.theme.colors.background};
	border: 1px solid ${(props) => props.theme.colors.border};
	border-radius: 6px;
	overflow: hidden;
	margin-bottom: 1rem;
`;

const SnippetHeader = styled.div`
	padding: 0.75rem;
	background: ${(props) => props.theme.colors.surfaceLight};
	border-bottom: 1px solid ${(props) => props.theme.colors.border};
	display: flex;
	align-items: center;
	justify-content: space-between;
	cursor: pointer;
`;

const SnippetTitle = styled.div`
	display: flex;
	align-items: center;
	gap: 0.5rem;
	font-weight: 600;
	color: ${(props) => props.theme.colors.text};
	font-size: 0.9rem;
`;

const SnippetContent = styled(motion.div)`
	padding: 1rem;
	font-family: "Monaco", "Consolas", monospace;
	font-size: 0.8rem;
	color: ${(props) => props.theme.colors.textSecondary};
	line-height: 1.4;
	background: ${(props) => props.theme.colors.surface};
	border-top: 1px solid ${(props) => props.theme.colors.border};
	white-space: pre-wrap;
	overflow-x: auto;
`;

const AttackResultContainer = styled(motion.div)`
	background: ${(props) => props.theme.colors.background};
	border: 1px solid ${(props) => props.theme.colors.border};
	border-radius: 8px;
	overflow: hidden;
	margin-top: 1rem;
`;

const AttackResultHeader = styled.div`
	padding: 0.75rem 1rem;
	background: ${(props) => {
		if (props.success) return "rgba(0, 255, 136, 0.1)";
		return "rgba(255, 71, 87, 0.1)";
	}};
	border-bottom: 1px solid ${(props) => props.theme.colors.border};
	display: flex;
	align-items: center;
	gap: 0.5rem;
	font-weight: 600;
	color: ${(props) => (props.success ? props.theme.colors.success : props.theme.colors.vulnerability)};
`;

const AttackResultContent = styled.div`
	padding: 1rem;
`;

const MetricRow = styled.div`
	display: flex;
	justify-content: space-between;
	align-items: center;
	padding: 0.5rem 0;
	border-bottom: 1px solid ${(props) => props.theme.colors.border};
	font-size: 0.9rem;

	&:last-child {
		border-bottom: none;
	}
`;

const MetricLabel = styled.span`
	color: ${(props) => props.theme.colors.textSecondary};
`;

const MetricValue = styled.span`
	font-weight: 600;
	color: ${(props) => props.theme.colors.text};
`;

const StatusGrid = styled.div`
	display: grid;
	grid-template-columns: 1fr 1fr;
	gap: 1rem;
	margin-bottom: 1.5rem;
`;

const StatusCard = styled.div`
	background: ${(props) => props.theme.colors.background};
	border: 1px solid ${(props) => props.theme.colors.border};
	border-radius: 8px;
	padding: 1rem;
	text-align: center;
`;

const StatusValue = styled.div`
	font-size: 1.5rem;
	font-weight: 700;
	color: ${(props) => props.theme.colors.primary};
	margin-bottom: 0.25rem;
`;

const StatusLabel = styled.div`
	font-size: 0.8rem;
	color: ${(props) => props.theme.colors.textSecondary};
`;

const ActionSection = styled.div`
	margin-bottom: 1.5rem;
`;

const ButtonGroup = styled.div`
	display: grid;
	grid-template-columns: 1fr 1fr;
	gap: 0.75rem;
	margin-bottom: 1rem;
`;

const AdversarialPanel = ({
	attackSnippets,
	bestAttackSnippet,
	onStartFga,
	onLoadBestSnippet,
	onInjectSnippet,
	onPerformAttack,
	isFgaRunning,
	isAttacking,
	attackResult,
}) => {
	const [isExpanded, setIsExpanded] = useState(true);

	const getAttackResultIcon = () => {
		if (!attackResult) return null;
		return attackResult.attack_success ? <CheckCircle size={16} /> : <XCircle size={16} />;
	};

	const getAttackResultText = () => {
		if (!attackResult) return "";
		return attackResult.attack_success ? "Attack Successful" : "Attack Failed";
	};

	return (
		<PanelContainer
			initial={{ opacity: 0, y: 20 }}
			animate={{ opacity: 1, y: 0 }}
			transition={{ duration: 0.3, delay: 0.1 }}>
			<PanelHeader onClick={() => setIsExpanded(!isExpanded)}>
				<PanelTitle>
					<Target size={20} />
					Adversarial Attack Panel
				</PanelTitle>
				{isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
			</PanelHeader>

			{isExpanded && (
				<PanelContent
					initial={{ height: 0 }}
					animate={{ height: "auto" }}
					exit={{ height: 0 }}
					transition={{ duration: 0.3 }}>
					{/* Status Grid */}
					<StatusGrid>
						<StatusCard>
							<StatusValue>{attackSnippets?.length || 0}</StatusValue>
							<StatusLabel>Attack Snippets</StatusLabel>
						</StatusCard>
						<StatusCard>
							<StatusValue>{bestAttackSnippet ? "1" : "0"}</StatusValue>
							<StatusLabel>Best Snippet Ready</StatusLabel>
						</StatusCard>
					</StatusGrid>

					{/* FGA Selection Section */}
					<ActionSection>
						<SectionTitle>
							<TrendingUp size={16} />
							Fuzzy Genetic Algorithm
						</SectionTitle>
						<ButtonGroup>
							<ActionButton
								variant="primary"
								onClick={onStartFga}
								disabled={isFgaRunning || !attackSnippets?.length}
								whileHover={{ scale: 1.02 }}
								whileTap={{ scale: 0.98 }}>
								<Play size={14} />
								{isFgaRunning ? "Running FGA..." : "Start FGA Selection"}
							</ActionButton>

							<ActionButton
								onClick={onLoadBestSnippet}
								disabled={isFgaRunning}
								whileHover={{ scale: 1.02 }}
								whileTap={{ scale: 0.98 }}>
								<Download size={14} />
								Load Best Snippet
							</ActionButton>
						</ButtonGroup>
					</ActionSection>

					{/* Best Attack Snippet Display */}
					{bestAttackSnippet && (
						<ActionSection>
							<SectionTitle>
								<Code2 size={16} />
								Best Attack Snippet
							</SectionTitle>
							<SnippetContainer>
								<SnippetHeader>
									<SnippetTitle>
										<Code size={14} />
										Selected by FGA Algorithm
									</SnippetTitle>
								</SnippetHeader>
								<SnippetContent
									initial={{ height: 0 }}
									animate={{ height: "auto" }}
									transition={{ duration: 0.3 }}>
									{bestAttackSnippet}
								</SnippetContent>
							</SnippetContainer>
						</ActionSection>
					)}

					{/* Attack Execution Section */}
					<ActionSection>
						<SectionTitle>
							<Zap size={16} />
							Attack Execution
						</SectionTitle>
						<ButtonGroup>
							<ActionButton
								variant="warning"
								onClick={onInjectSnippet}
								disabled={!bestAttackSnippet || isAttacking}
								whileHover={{ scale: 1.02 }}
								whileTap={{ scale: 0.98 }}>
								<Code size={14} />
								Inject Snippet
							</ActionButton>

							<ActionButton
								variant="success"
								onClick={onPerformAttack}
								disabled={!bestAttackSnippet || isAttacking}
								whileHover={{ scale: 1.02 }}
								whileTap={{ scale: 0.98 }}>
								<Target size={14} />
								{isAttacking ? "Attacking..." : "Perform Attack"}
							</ActionButton>
						</ButtonGroup>
					</ActionSection>

					{/* Attack Results */}
					{attackResult && (
						<AttackResultContainer
							initial={{ opacity: 0, y: 10 }}
							animate={{ opacity: 1, y: 0 }}
							transition={{ duration: 0.3 }}>
							<AttackResultHeader success={attackResult.attack_success}>
								{getAttackResultIcon()}
								{getAttackResultText()}
							</AttackResultHeader>
							<AttackResultContent>
								<MetricRow>
									<MetricLabel>Original Prediction:</MetricLabel>
									<MetricValue>
										{attackResult.original_prediction?.is_vulnerable ? "Vulnerable" : "Safe"}
									</MetricValue>
								</MetricRow>
								<MetricRow>
									<MetricLabel>Adversarial Prediction:</MetricLabel>
									<MetricValue>
										{attackResult.adversarial_prediction?.is_vulnerable ? "Vulnerable" : "Safe"}
									</MetricValue>
								</MetricRow>
								<MetricRow>
									<MetricLabel>Original Confidence:</MetricLabel>
									<MetricValue>
										{Math.round((attackResult.original_prediction?.confidence || 0) * 100)}%
									</MetricValue>
								</MetricRow>
								<MetricRow>
									<MetricLabel>Adversarial Confidence:</MetricLabel>
									<MetricValue>
										{Math.round((attackResult.adversarial_prediction?.confidence || 0) * 100)}%
									</MetricValue>
								</MetricRow>
							</AttackResultContent>
						</AttackResultContainer>
					)}

					{/* Empty State */}
					{!attackSnippets?.length && (
						<div
							style={{
								textAlign: "center",
								padding: "2rem",
								color: "#a0a3b1",
							}}>
							<Shield size={48} style={{ marginBottom: "1rem", opacity: 0.5 }} />
							<div>Load attack pool to begin adversarial testing</div>
						</div>
					)}
				</PanelContent>
			)}
		</PanelContainer>
	);
};

export default AdversarialPanel;
