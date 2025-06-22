import React, { useState } from "react";
import styled from "styled-components";
import { motion } from "framer-motion";
import { Activity, TrendingUp, Clock, Target, ChevronDown, ChevronRight, BarChart3, Zap } from "lucide-react";

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

const ProgressSection = styled.div`
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

const ProgressBar = styled.div`
	width: 100%;
	height: 8px;
	background: ${(props) => props.theme.colors.border};
	border-radius: 4px;
	overflow: hidden;
	margin-bottom: 0.5rem;
`;

const ProgressFill = styled(motion.div)`
	height: 100%;
	background: ${(props) => props.theme.gradients.cyber};
	border-radius: 4px;
	position: relative;

	&::after {
		content: "";
		position: absolute;
		top: 0;
		left: 0;
		right: 0;
		bottom: 0;
		background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
		animation: shimmer 2s infinite;
	}

	@keyframes shimmer {
		0% {
			transform: translateX(-100%);
		}
		100% {
			transform: translateX(100%);
		}
	}
`;

const ProgressText = styled.div`
	display: flex;
	justify-content: space-between;
	font-size: 0.8rem;
	color: ${(props) => props.theme.colors.textSecondary};
`;

const MetricGrid = styled.div`
	display: grid;
	grid-template-columns: 1fr 1fr;
	gap: 1rem;
	margin-bottom: 1rem;
`;

const MetricCard = styled.div`
	background: ${(props) => props.theme.colors.background};
	border: 1px solid ${(props) => props.theme.colors.border};
	border-radius: 8px;
	padding: 1rem;
	text-align: center;
`;

const MetricValue = styled.div`
	font-size: 1.2rem;
	font-weight: 700;
	color: ${(props) => {
		if (props.type === "fitness") return props.theme.colors.success;
		if (props.type === "attack") return props.theme.colors.vulnerability;
		if (props.type === "time") return props.theme.colors.primary;
		return props.theme.colors.text;
	}};
	margin-bottom: 0.25rem;
`;

const MetricLabel = styled.div`
	font-size: 0.7rem;
	color: ${(props) => props.theme.colors.textSecondary};
	text-transform: uppercase;
	letter-spacing: 0.5px;
`;

const StatusDot = styled.div`
	width: 8px;
	height: 8px;
	border-radius: 50%;
	background: ${(props) => {
		if (props.status === "running") return props.theme.colors.primary;
		if (props.status === "completed") return props.theme.colors.success;
		if (props.status === "error") return props.theme.colors.vulnerability;
		return props.theme.colors.border;
	}};
	animation: ${(props) => (props.status === "running" ? "pulse 2s infinite" : "none")};

	@keyframes pulse {
		0% {
			opacity: 1;
		}
		50% {
			opacity: 0.5;
		}
		100% {
			opacity: 1;
		}
	}
`;

const TimeEstimate = styled.div`
	display: flex;
	align-items: center;
	justify-content: space-between;
	padding: 0.75rem;
	background: ${(props) => props.theme.colors.background};
	border: 1px solid ${(props) => props.theme.colors.border};
	border-radius: 6px;
	font-size: 0.8rem;
`;

const TimeLabel = styled.span`
	color: ${(props) => props.theme.colors.textSecondary};
`;

const TimeValue = styled.span`
	color: ${(props) => props.theme.colors.text};
	font-weight: 600;
`;

const formatTime = (seconds) => {
	if (seconds < 60) {
		return `${Math.round(seconds)}s`;
	} else if (seconds < 3600) {
		const minutes = Math.floor(seconds / 60);
		const remainingSeconds = Math.round(seconds % 60);
		return `${minutes}m ${remainingSeconds}s`;
	} else {
		const hours = Math.floor(seconds / 3600);
		const minutes = Math.floor((seconds % 3600) / 60);
		return `${hours}h ${minutes}m`;
	}
};

const ProgressVisualization = ({ progress, isRunning }) => {
	const [isExpanded, setIsExpanded] = useState(true);

	const getStatusText = () => {
		if (!progress) return isRunning ? "starting" : "idle";
		return progress.status || "idle";
	};

	const getProgressPercentage = () => {
		if (!progress || !progress.max_generations) return 0;
		return (progress.current_generation / progress.max_generations) * 100;
	};

	const getSubProgressPercentage = () => {
		if (!progress || !progress.sub_total) return 0;
		return (progress.sub_progress / progress.sub_total) * 100;
	};

	const getPhaseIcon = () => {
		if (!progress || !progress.current_phase) return <Activity size={16} />;

		switch (progress.current_phase) {
			case "initialization":
				return <Target size={16} />;
			case "evolution":
				return <TrendingUp size={16} />;
			case "testing_snippets":
				return <BarChart3 size={16} />;
			case "completed":
				return <Zap size={16} />;
			case "initializing":
				return <Target size={16} />;
			default:
				return <Activity size={16} />;
		}
	};

	return (
		<PanelContainer
			initial={{ opacity: 0, y: 20 }}
			animate={{ opacity: 1, y: 0 }}
			transition={{ duration: 0.3, delay: 0.4 }}>
			<PanelHeader onClick={() => setIsExpanded(!isExpanded)}>
				<PanelTitle>
					<Activity size={20} />
					FGA Progress
				</PanelTitle>
				<div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
					<StatusDot status={progress?.status || "idle"} />
					<span
						style={{
							fontSize: "0.8rem",
							color: "#a0a3b1",
							textTransform: "capitalize",
						}}>
						{getStatusText()}
					</span>
					{isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
				</div>
			</PanelHeader>

			{isExpanded && (
				<PanelContent
					initial={{ height: 0 }}
					animate={{ height: "auto" }}
					exit={{ height: 0 }}
					transition={{ duration: 0.3 }}>
					{/* Overall Progress Bar */}
					<ProgressSection>
						<SectionTitle>
							<Activity size={16} />
							Generation Progress
						</SectionTitle>
						<ProgressBar>
							<ProgressFill
								initial={{ width: 0 }}
								animate={{ width: `${getProgressPercentage()}%` }}
								transition={{ duration: 0.5 }}
							/>
						</ProgressBar>
						<ProgressText>
							<span>
								{progress?.current_generation || 0} / {progress?.max_generations || 0} generations
							</span>
							<span>{Math.round(getProgressPercentage())}%</span>
						</ProgressText>
					</ProgressSection>

					{/* Current Phase Progress */}
					{progress && progress.current_phase && progress.current_phase !== "idle" && (
						<ProgressSection>
							<SectionTitle>
								{getPhaseIcon()}
								Current Phase
							</SectionTitle>
							<div style={{ marginBottom: "0.75rem" }}>
								<div
									style={{
										fontSize: "0.85rem",
										color: "#e0e3ed",
										marginBottom: "0.5rem",
										fontWeight: "500",
									}}>
									{progress.phase_description || "Processing..."}
								</div>
								<ProgressBar>
									<ProgressFill
										initial={{ width: 0 }}
										animate={{ width: `${getSubProgressPercentage()}%` }}
										transition={{ duration: 0.3 }}
									/>
								</ProgressBar>
								<ProgressText>
									<span>
										{progress.sub_progress || 0} / {progress.sub_total || 0} items
									</span>
									<span>{Math.round(getSubProgressPercentage())}%</span>
								</ProgressText>
							</div>

							{/* Phase Status Indicator */}
							<div
								style={{
									display: "flex",
									alignItems: "center",
									gap: "0.5rem",
									padding: "0.5rem 0.75rem",
									background: "rgba(0, 212, 255, 0.1)",
									border: "1px solid rgba(0, 212, 255, 0.2)",
									borderRadius: "6px",
									fontSize: "0.75rem",
									color: "#00d4ff",
								}}>
								<StatusDot status="running" />
								{progress.current_phase === "initialization" &&
									"Initializing population with attack snippets"}
								{progress.current_phase === "initializing" && "Setting up FGA components"}
								{progress.current_phase === "evolution" &&
									"Evolving population using genetic algorithm"}
								{progress.current_phase === "testing_snippets" &&
									"Testing attack snippets against extracted function"}
								{progress.current_phase === "completed" && "FGA selection completed successfully"}
								{![
									"initialization",
									"initializing",
									"evolution",
									"testing_snippets",
									"completed",
								].includes(progress.current_phase) &&
									(progress.phase_description || "Processing...")}
							</div>
						</ProgressSection>
					)}

					{/* Metrics */}
					{progress && (
						<ProgressSection>
							<SectionTitle>
								<BarChart3 size={16} />
								Performance Metrics
							</SectionTitle>
							<MetricGrid>
								<MetricCard>
									<MetricValue type="fitness">
										{Math.round((progress.best_fitness || 0) * 100)}%
									</MetricValue>
									<MetricLabel>Best Fitness</MetricLabel>
								</MetricCard>
								<MetricCard>
									<MetricValue type="attack">
										{Math.round((progress.attack_success_rate || 0) * 100)}%
									</MetricValue>
									<MetricLabel>Attack Success</MetricLabel>
								</MetricCard>
							</MetricGrid>
						</ProgressSection>
					)}

					{/* Time Information */}
					{progress && (progress.time_elapsed > 0 || progress.estimated_time_remaining > 0) && (
						<ProgressSection>
							<SectionTitle>
								<Clock size={16} />
								Time Information
							</SectionTitle>
							<div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
								{progress.time_elapsed > 0 && (
									<TimeEstimate>
										<TimeLabel>Elapsed:</TimeLabel>
										<TimeValue>{formatTime(progress.time_elapsed)}</TimeValue>
									</TimeEstimate>
								)}
								{progress.estimated_time_remaining > 0 && (
									<TimeEstimate>
										<TimeLabel>Remaining:</TimeLabel>
										<TimeValue>{formatTime(progress.estimated_time_remaining)}</TimeValue>
									</TimeEstimate>
								)}
							</div>
						</ProgressSection>
					)}

					{/* Empty State */}
					{!progress && !isRunning && (
						<div
							style={{
								textAlign: "center",
								padding: "2rem",
								color: "#a0a3b1",
							}}>
							<Target size={48} style={{ marginBottom: "1rem", opacity: 0.5 }} />
							<div>Start FGA selection to see progress</div>
						</div>
					)}
				</PanelContent>
			)}
		</PanelContainer>
	);
};

export default ProgressVisualization;
