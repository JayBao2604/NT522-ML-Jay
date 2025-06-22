import React, { useState, useRef, useEffect } from "react";
import styled from "styled-components";
import { motion, AnimatePresence } from "framer-motion";
import {
	Upload,
	File,
	Database,
	CheckCircle,
	XCircle,
	ChevronDown,
	ChevronRight,
	Trash2,
	Info,
	AlertCircle,
} from "lucide-react";
import { vulnerabilityService } from "../services/api";

// Styled Components
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

const Panel = styled(motion.div)`
	background: ${(props) => props.theme.colors.surface};
	border-radius: 12px;
	border: 1px solid ${(props) => props.theme.colors.border};
	padding: 1.5rem;
	margin-bottom: 1rem;
	box-shadow: ${(props) => props.theme.shadows.medium};
`;

const Section = styled.div`
	margin-bottom: 2rem;

	&:last-child {
		margin-bottom: 0;
	}
`;

const SectionHeader = styled.div`
	display: flex;
	align-items: center;
	justify-content: space-between;
	margin-bottom: 1rem;
`;

const SectionTitle = styled.div`
	display: flex;
	align-items: center;
	gap: 0.5rem;
	font-weight: 600;
	color: ${(props) => props.theme.colors.text};
	margin-bottom: 1rem;
	font-size: 0.9rem;
`;

const ActionButton = styled.button`
	background: none;
	border: none;
	color: ${(props) => (props.variant === "danger" ? props.theme.colors.error : props.theme.colors.primary)};
	font-size: 0.875rem;
	cursor: pointer;
	padding: 0.25rem 0.5rem;
	border-radius: 4px;
	transition: all 0.2s ease;

	&:hover {
		background: ${(props) =>
			props.variant === "danger" ? `${props.theme.colors.error}20` : `${props.theme.colors.primary}20`};
	}
`;

const UploadArea = styled.div`
	border: 2px dashed ${(props) => props.theme.colors.border};
	border-radius: 8px;
	padding: 2rem;
	text-align: center;
	transition: all 0.3s ease;
	cursor: pointer;
	position: relative;
	background: ${(props) => props.theme.colors.background};

	&:hover {
		border-color: ${(props) => props.theme.colors.primary};
		background: ${(props) => props.theme.colors.primary}08;
	}

	&.drag-over {
		border-color: ${(props) => props.theme.colors.success};
		background: ${(props) => props.theme.colors.success}08;
	}
`;

const UploadIcon = styled.div`
	color: ${(props) => props.theme.colors.primary};
	margin-bottom: 1rem;
`;

const UploadText = styled.div`
	color: ${(props) => props.theme.colors.text};
	font-weight: 600;
	margin-bottom: 0.5rem;
`;

const UploadSubtext = styled.div`
	color: ${(props) => props.theme.colors.textSecondary};
	font-size: 0.9rem;
`;

const HiddenInput = styled.input`
	position: absolute;
	top: 0;
	left: 0;
	width: 100%;
	height: 100%;
	opacity: 0;
	cursor: pointer;
`;

const StatusMessage = styled(motion.div)`
	display: flex;
	align-items: center;
	gap: 0.5rem;
	padding: 0.75rem 1rem;
	border-radius: 6px;
	margin-top: 1rem;
	font-size: 0.9rem;
	background: ${(props) => {
		if (props.type === "success") return props.theme.colors.success + "20";
		if (props.type === "error") return props.theme.colors.vulnerability + "20";
		return props.theme.colors.primary + "20";
	}};
	border: 1px solid
		${(props) => {
			if (props.type === "success") return props.theme.colors.success;
			if (props.type === "error") return props.theme.colors.vulnerability;
			return props.theme.colors.primary;
		}};
	color: ${(props) => {
		if (props.type === "success") return props.theme.colors.success;
		if (props.type === "error") return props.theme.colors.vulnerability;
		return props.theme.colors.primary;
	}};
`;

const FileInfo = styled.div`
	background: ${(props) => props.theme.colors.background};
	border: 1px solid ${(props) => props.theme.colors.border};
	border-radius: 6px;
	padding: 1rem;
	margin-top: 1rem;
`;

const FileInfoRow = styled.div`
	display: flex;
	justify-content: space-between;
	margin-bottom: 0.5rem;
	font-size: 0.9rem;

	&:last-child {
		margin-bottom: 0;
	}
`;

const FileInfoLabel = styled.span`
	color: ${(props) => props.theme.colors.textSecondary};
`;

const FileInfoValue = styled.span`
	color: ${(props) => props.theme.colors.text};
	font-weight: 600;
`;

const ClearButton = styled.button`
	background: none;
	border: 1px solid ${(props) => props.theme.colors.border};
	border-radius: 4px;
	color: ${(props) => props.theme.colors.textSecondary};
	padding: 0.5rem 1rem;
	cursor: pointer;
	font-size: 0.8rem;
	margin-top: 0.5rem;
	transition: all 0.2s ease;

	&:hover {
		border-color: ${(props) => props.theme.colors.vulnerability};
		color: ${(props) => props.theme.colors.vulnerability};
	}
`;

const FormatInfo = styled.div`
	background: ${(props) => props.theme.colors.background};
	border: 1px solid ${(props) => props.theme.colors.border};
	border-radius: 6px;
	padding: 1rem;
	margin-top: 1rem;
	font-size: 0.85rem;
`;

const FormatInfoToggle = styled.button`
	background: none;
	border: none;
	color: ${(props) => props.theme.colors.primary};
	cursor: pointer;
	display: flex;
	align-items: center;
	gap: 0.5rem;
	font-size: 0.85rem;
	margin-top: 0.5rem;
	padding: 0;

	&:hover {
		text-decoration: underline;
	}
`;

const FormatDetails = styled(motion.div)`
	margin-top: 1rem;
	padding-top: 1rem;
	border-top: 1px solid ${(props) => props.theme.colors.border};
`;

const FormatSection = styled.div`
	margin-bottom: 1rem;

	&:last-child {
		margin-bottom: 0;
	}
`;

const FormatSectionTitle = styled.div`
	font-weight: 600;
	color: ${(props) => props.theme.colors.text};
	margin-bottom: 0.5rem;
`;

const FormatList = styled.ul`
	margin: 0;
	padding-left: 1.5rem;
	color: ${(props) => props.theme.colors.textSecondary};
`;

const FormatListItem = styled.li`
	margin-bottom: 0.25rem;
`;

const FileUploadPanel = ({ onCodeUploaded, onAttackPoolUploaded }) => {
	const [isExpanded, setIsExpanded] = useState(true);
	const [codeUploadStatus, setCodeUploadStatus] = useState(null);
	const [attackPoolUploadStatus, setAttackPoolUploadStatus] = useState(null);
	const [uploadedCodeFile, setUploadedCodeFile] = useState(null);
	const [uploadedAttackPool, setUploadedAttackPool] = useState(null);
	const [showFormatInfo, setShowFormatInfo] = useState(false);
	const [formatInfo, setFormatInfo] = useState(null);

	const codeFileRef = useRef(null);
	const attackPoolRef = useRef(null);

	const handleCodeFileUpload = async (event) => {
		const file = event.target.files[0];
		if (!file) return;

		setCodeUploadStatus({ type: "loading", message: "Uploading file..." });

		try {
			const result = await vulnerabilityService.uploadCodeFile(file);

			setCodeUploadStatus({ type: "success", message: `File uploaded successfully: ${result.filename}` });
			setUploadedCodeFile(result);
			onCodeUploaded(result.code, result.language, result.filename);
		} catch (error) {
			setCodeUploadStatus({ type: "error", message: error.message || "Network error during upload" });
		}

		// Reset input
		event.target.value = "";
	};

	const handleAttackPoolUpload = async (event) => {
		const file = event.target.files[0];
		if (!file) return;

		setAttackPoolUploadStatus({ type: "loading", message: "Uploading attack pool..." });

		try {
			const result = await vulnerabilityService.uploadAttackPool(file);

			setAttackPoolUploadStatus({
				type: "success",
				message: `Attack pool uploaded: ${result.valid_entries} valid entries`,
			});
			setUploadedAttackPool(result);
			onAttackPoolUploaded(result);
		} catch (error) {
			setAttackPoolUploadStatus({ type: "error", message: error.message || "Network error during upload" });
		}

		// Reset input
		event.target.value = "";
	};

	const loadFormatInfo = async () => {
		try {
			const data = await vulnerabilityService.getAttackPoolFormat();
			setFormatInfo(data);
		} catch (error) {
			console.error("Failed to load format info:", error);
		}
	};

	useEffect(() => {
		loadFormatInfo();
	}, []);

	const clearCodeFile = () => {
		setUploadedCodeFile(null);
		setCodeUploadStatus(null);
	};

	const clearAttackPool = () => {
		setUploadedAttackPool(null);
		setAttackPoolUploadStatus(null);
	};

	const getSupportedExtensions = () => {
		return [
			".c",
			".cpp",
			".h",
			".hpp",
			".java",
			".py",
			".js",
			".ts",
			".php",
			".cs",
			".go",
			".rs",
			".rb",
			".swift",
			".kt",
			".scala",
			".txt",
		];
	};

	return (
		<PanelContainer
			initial={{ opacity: 0, y: 20 }}
			animate={{ opacity: 1, y: 0 }}
			transition={{ duration: 0.3, delay: 0.2 }}>
			<PanelHeader onClick={() => setIsExpanded(!isExpanded)}>
				<PanelTitle>
					<Upload size={20} />
					File Upload
				</PanelTitle>
				{isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
			</PanelHeader>

			{isExpanded && (
				<PanelContent
					initial={{ height: 0 }}
					animate={{ height: "auto" }}
					exit={{ height: 0 }}
					transition={{ duration: 0.3 }}>
					{/* Code File Upload */}
					<Section>
						<SectionHeader>
							<SectionTitle>
								<File size={16} />
								Code File Upload
							</SectionTitle>
							{uploadedCodeFile && (
								<ActionButton variant="danger" onClick={clearCodeFile}>
									Clear
								</ActionButton>
							)}
						</SectionHeader>

						<UploadArea>
							<UploadIcon>
								<File size={48} />
							</UploadIcon>
							<UploadText>Upload Code File</UploadText>
							<UploadSubtext>Supported: {getSupportedExtensions().join(", ")}</UploadSubtext>
							<HiddenInput
								type="file"
								accept={getSupportedExtensions().join(",")}
								onChange={handleCodeFileUpload}
							/>
						</UploadArea>

						{codeUploadStatus && (
							<StatusMessage
								type={codeUploadStatus.type}
								initial={{ opacity: 0, y: -10 }}
								animate={{ opacity: 1, y: 0 }}
								transition={{ duration: 0.3 }}>
								{codeUploadStatus.type === "success" && <CheckCircle size={16} />}
								{codeUploadStatus.type === "error" && <XCircle size={16} />}
								{codeUploadStatus.type === "loading" && <div className="animate-spin">⏳</div>}
								{codeUploadStatus.message}
							</StatusMessage>
						)}

						{uploadedCodeFile && (
							<FileInfo>
								<FileInfoRow>
									<FileInfoLabel>Filename:</FileInfoLabel>
									<FileInfoValue>{uploadedCodeFile.filename}</FileInfoValue>
								</FileInfoRow>
								<FileInfoRow>
									<FileInfoLabel>Language:</FileInfoLabel>
									<FileInfoValue>{uploadedCodeFile.language}</FileInfoValue>
								</FileInfoRow>
								<FileInfoRow>
									<FileInfoLabel>Size:</FileInfoLabel>
									<FileInfoValue>{uploadedCodeFile.size} bytes</FileInfoValue>
								</FileInfoRow>
								<FileInfoRow>
									<FileInfoLabel>Lines:</FileInfoLabel>
									<FileInfoValue>{uploadedCodeFile.lines}</FileInfoValue>
								</FileInfoRow>
								<ClearButton onClick={clearCodeFile}>Clear File</ClearButton>
							</FileInfo>
						)}
					</Section>

					{/* Attack Pool Upload */}
					<Section>
						<SectionHeader>
							<SectionTitle>
								<Database size={16} />
								Attack Pool Upload
							</SectionTitle>
							<div style={{ display: "flex", gap: "0.5rem" }}>
								<ActionButton onClick={() => setShowFormatInfo(!showFormatInfo)}>
									{showFormatInfo ? "Hide" : "Show"} detailed format info
								</ActionButton>
								{uploadedAttackPool && (
									<ActionButton variant="danger" onClick={clearAttackPool}>
										Clear
									</ActionButton>
								)}
							</div>
						</SectionHeader>

						{showFormatInfo && formatInfo && (
							<FormatInfo>
								<div
									style={{
										display: "flex",
										alignItems: "center",
										gap: "0.5rem",
										marginBottom: "0.5rem",
									}}>
									<Info size={14} />
									<span style={{ fontWeight: 600 }}>CSV Format Requirements</span>
								</div>
								<div style={{ color: "#a0a3b1" }}>
									Required columns: original_code, adversarial_code, label
								</div>

								<FormatInfoToggle onClick={() => setShowFormatInfo(!showFormatInfo)}>
									{showFormatInfo ? "Hide" : "Show"} detailed format info
									{showFormatInfo ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
								</FormatInfoToggle>

								{showFormatInfo && (
									<FormatDetails
										initial={{ height: 0, opacity: 0 }}
										animate={{ height: "auto", opacity: 1 }}
										exit={{ height: 0, opacity: 0 }}
										transition={{ duration: 0.3 }}>
										<FormatSection>
											<FormatSectionTitle>Required Columns:</FormatSectionTitle>
											<FormatList>
												{formatInfo.required_columns?.map((col, index) => (
													<FormatListItem key={index}>
														<strong>{col.name}:</strong> {col.description}
													</FormatListItem>
												))}
											</FormatList>
										</FormatSection>

										<FormatSection>
											<FormatSectionTitle>Requirements:</FormatSectionTitle>
											<FormatList>
												{formatInfo.requirements?.map((req, index) => (
													<FormatListItem key={index}>{req}</FormatListItem>
												))}
											</FormatList>
										</FormatSection>

										<FormatSection>
											<FormatSectionTitle>Tips:</FormatSectionTitle>
											<FormatList>
												{formatInfo.tips?.map((tip, index) => (
													<FormatListItem key={index}>{tip}</FormatListItem>
												))}
											</FormatList>
										</FormatSection>
									</FormatDetails>
								)}
							</FormatInfo>
						)}

						<UploadArea>
							<UploadIcon>
								<Database size={48} />
							</UploadIcon>
							<UploadText>Upload Attack Pool CSV</UploadText>
							<UploadSubtext>CSV file with adversarial code snippets</UploadSubtext>
							<HiddenInput type="file" accept=".csv" onChange={handleAttackPoolUpload} />
						</UploadArea>

						{attackPoolUploadStatus && (
							<StatusMessage
								type={attackPoolUploadStatus.type}
								initial={{ opacity: 0, y: -10 }}
								animate={{ opacity: 1, y: 0 }}
								transition={{ duration: 0.3 }}>
								{attackPoolUploadStatus.type === "success" && <CheckCircle size={16} />}
								{attackPoolUploadStatus.type === "error" && <XCircle size={16} />}
								{attackPoolUploadStatus.type === "loading" && <div className="animate-spin">⏳</div>}
								{attackPoolUploadStatus.message}
							</StatusMessage>
						)}

						{uploadedAttackPool && (
							<FileInfo>
								<FileInfoRow>
									<FileInfoLabel>Filename:</FileInfoLabel>
									<FileInfoValue>{uploadedAttackPool.filename}</FileInfoValue>
								</FileInfoRow>
								<FileInfoRow>
									<FileInfoLabel>Total Entries:</FileInfoLabel>
									<FileInfoValue>{uploadedAttackPool.total_entries}</FileInfoValue>
								</FileInfoRow>
								<FileInfoRow>
									<FileInfoLabel>Valid Entries:</FileInfoLabel>
									<FileInfoValue>{uploadedAttackPool.valid_entries}</FileInfoValue>
								</FileInfoRow>
								{uploadedAttackPool.removed_entries > 0 && (
									<FileInfoRow>
										<FileInfoLabel>Removed (Invalid):</FileInfoLabel>
										<FileInfoValue>{uploadedAttackPool.removed_entries}</FileInfoValue>
									</FileInfoRow>
								)}
								<ClearButton onClick={clearAttackPool}>Clear Attack Pool</ClearButton>
							</FileInfo>
						)}
					</Section>
				</PanelContent>
			)}
		</PanelContainer>
	);
};

export default FileUploadPanel;
