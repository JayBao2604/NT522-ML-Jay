import React from "react";
import styled from "styled-components";
import { motion } from "framer-motion";
import { Shield, ShieldAlert, ShieldCheck, Activity } from "lucide-react";

const BadgeContainer = styled(motion.div)`
	display: flex;
	align-items: center;
	gap: 0.5rem;
	padding: 0.5rem 1rem;
	border-radius: 20px;
	font-size: 0.8rem;
	font-weight: 600;
	text-transform: uppercase;
	letter-spacing: 0.5px;
	border: 1px solid;
	background: ${(props) => {
		if (props.status === "vulnerable") return "rgba(255, 71, 87, 0.1)";
		if (props.status === "secure") return "rgba(0, 255, 136, 0.1)";
		if (props.status === "analyzing") return "rgba(0, 212, 255, 0.1)";
		return "rgba(160, 163, 177, 0.1)";
	}};
	border-color: ${(props) => {
		if (props.status === "vulnerable") return props.theme.colors.vulnerability;
		if (props.status === "secure") return props.theme.colors.success;
		if (props.status === "analyzing") return props.theme.colors.primary;
		return props.theme.colors.border;
	}};
	color: ${(props) => {
		if (props.status === "vulnerable") return props.theme.colors.vulnerability;
		if (props.status === "secure") return props.theme.colors.success;
		if (props.status === "analyzing") return props.theme.colors.primary;
		return props.theme.colors.textSecondary;
	}};
`;

const StatusIcon = styled(motion.div)`
	display: flex;
	align-items: center;
	justify-content: center;
`;

const StatusText = styled.span`
	white-space: nowrap;
`;

const StatusBadge = ({ status, text }) => {
	const getStatusIcon = () => {
		switch (status) {
			case "vulnerable":
				return ShieldAlert;
			case "secure":
				return ShieldCheck;
			case "analyzing":
				return Activity;
			default:
				return Shield;
		}
	};

	const StatusIconComponent = getStatusIcon();

	const getStatusColor = () => {
		switch (status) {
			case "vulnerable":
				return "#ff4757";
			case "secure":
				return "#00ff88";
			case "analyzing":
				return "#00d4ff";
			default:
				return "#a0a3b1";
		}
	};

	return (
		<BadgeContainer
			status={status}
			initial={{ scale: 0.8, opacity: 0 }}
			animate={{ scale: 1, opacity: 1 }}
			transition={{ duration: 0.3 }}
			whileHover={{ scale: 1.05 }}>
			<StatusIcon
				animate={status === "analyzing" ? { rotate: 360 } : {}}
				transition={status === "analyzing" ? { duration: 2, repeat: Infinity, ease: "linear" } : {}}>
				<StatusIconComponent size={16} />
			</StatusIcon>
			<StatusText>{text}</StatusText>
		</BadgeContainer>
	);
};

export default StatusBadge;
