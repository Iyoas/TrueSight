import React, { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  Chip,
  Dialog,
  DialogContent,
  DialogTitle,
  IconButton,
  Stack,
} from "@mui/material";
import "./StepConclusion.module.css";
import {
  getDecisionLabel,
  getJointConclusionTitle,
  getJointOutcomeCallout,
  getPartialAgreementSubtitle,
} from "../../../lib/jointOutcomeUi";

export type VerdictLabel = "ai" | "human" | "uncertain";

export interface ConclusionCueSummary {
  id: number | string;
  label: string;
  score?: number; // 0–1 model attention
  source?: "model" | "user" | "both";
  note?: string;
  userJudgement?: "agree" | "not_sure" | "disagree";
}

export interface CueRecapItem {
  id: number | string;
  title: string;
  originalSrc?: string;
  overlaySrc?: string;
  modelScore?: number;
  userJudgement?: "agree" | "not_sure" | "disagree";
  isCustom?: boolean;
  observationText?: string;
}

export interface StepConclusionProps {
  // Core outcome
  modelLabel: VerdictLabel;
  modelProbability?: number; // 0–1
  userLabel?: VerdictLabel;
  aiDecision?: "AI_GENERATED" | "HUMAN_GENERATED";
  humanDecision?: "AI_GENERATED" | "HUMAN_GENERATED" | "UNCERTAIN";
  finalConclusion?: {
    kind: "AGREEMENT" | "PARTIAL_AGREEMENT" | "DISAGREEMENT";
    label: "AI_GENERATED" | "HUMAN_GENERATED" | null;
    leaning?: "AI" | "HUMAN";
  };

  // Optional metadata
  keyCues?: ConclusionCueSummary[];
  userConfidence?: number | null; // 0–1
  modelConfidenceText?: string;

  // Optional impact / context
  impactLevel?: "low" | "medium" | "high";
  cueRecapItems?: CueRecapItem[];

  // Callbacks for buttons (parent can handle navigation / saving)
  onFinish?: () => void;
  onTryAnother?: () => void;
}

const verdictLabelToText = (label: VerdictLabel): string => {
  switch (label) {
    case "ai":
      return "Likely AI-generated";
    case "human":
      return "Likely human-made";
    default:
      return "Uncertain";
  }
};

const formatProbability = (p?: number): string | null => {
  if (p == null) return null;
  const pct = Math.round(p * 100);
  return `${pct}%`;
};

const judgementLabelMap = {
  agree: "Suspicious",
  disagree: "Not suspicious",
  not_sure: "Not sure",
} as const;

const getJudgementLabel = (
  judgement?: "agree" | "not_sure" | "disagree"
): string => {
  if (!judgement) return "Not answered";
  return judgementLabelMap[judgement];
};

const StepConclusion: React.FC<StepConclusionProps> = ({
  modelLabel,
  modelProbability,
  userLabel,
  aiDecision,
  humanDecision,
  finalConclusion,
  keyCues = [],
  userConfidence,
  modelConfidenceText,
  impactLevel,
  cueRecapItems = [],
  onFinish,
  onTryAnother,
}) => {
  const modelVerdictText = verdictLabelToText(modelLabel);
  const userVerdictText = userLabel ? verdictLabelToText(userLabel) : null;
  const finalVerdictLabel: VerdictLabel = userLabel ?? modelLabel;
  const finalVerdictText = finalConclusion
    ? finalConclusion.kind === "AGREEMENT" && finalConclusion.label
      ? `Agreement - ${finalConclusion.label === "AI_GENERATED" ? "AI-generated" : "Human-generated"}`
      : finalConclusion.kind === "PARTIAL_AGREEMENT"
      ? "Partial agreement"
      : "Disagreement"
    : verdictLabelToText(finalVerdictLabel);
  const probabilityText = formatProbability(modelProbability);

  const hasUserVerdict = Boolean(userLabel);
  const agreementFlag =
    finalConclusion?.kind === "AGREEMENT"
      ? true
      : finalConclusion?.kind === "DISAGREEMENT"
      ? false
      : finalConclusion?.kind === "PARTIAL_AGREEMENT"
      ? null
      : hasUserVerdict
      ? userLabel === modelLabel
      : null;

  const impactLabel =
    impactLevel === "high"
      ? "High impact if wrong"
      : impactLevel === "medium"
      ? "Medium impact if wrong"
      : impactLevel === "low"
      ? "Low impact (exploratory)"
      : null;

  const userConfidenceText =
    typeof userConfidence === "number"
      ? userConfidence >= 0.8
        ? "Very confident"
        : userConfidence >= 0.6
        ? "Confident"
        : userConfidence >= 0.4
        ? "Somewhat unsure"
        : "Very unsure"
      : null;

  const [summaryText, setSummaryText] = useState<string | null>(null);
  const [isLoadingSummary, setIsLoadingSummary] = useState(false);
  const [summaryError, setSummaryError] = useState<string | null>(null);
  const [isSummaryReady, setIsSummaryReady] = useState(false);
  const [previewImage, setPreviewImage] = useState<{ src: string; label: string } | null>(
    null
  );
  const jointOutcomeTitle = getJointConclusionTitle(finalConclusion);
  const jointOutcomeCallout = getJointOutcomeCallout(finalConclusion);
  const partialAgreementSubtitle =
    finalConclusion?.kind === "PARTIAL_AGREEMENT" && aiDecision && humanDecision
      ? getPartialAgreementSubtitle(aiDecision, humanDecision)
      : null;

  useEffect(() => {
    // Only try to generate a summary if we have at least some meaningful info
    const hasBasicInfo =
      modelLabel !== undefined ||
      userLabel !== undefined ||
      (keyCues && keyCues.length > 0);

    setIsSummaryReady(false);
    if (!hasBasicInfo) {
      setIsSummaryReady(true);
      return;
    }

    const controller = new AbortController();

    const fetchSummary = async () => {
      try {
        setIsLoadingSummary(true);
        setSummaryError(null);

        const payload = {
          model_label: modelLabel,
          model_probability: modelProbability ?? null,
          user_label: userLabel ?? null,
          user_confidence: userConfidence ?? null,
          agreement: hasUserVerdict ? agreementFlag : null,
          cues: (keyCues || []).map((cue) => ({
            id: cue.id,
            label: cue.label,
            score: cue.score ?? null,
            source: cue.source ?? null,
            note: cue.note ?? null,
            user_judgement: cue.userJudgement ?? null,
          })),
        };

        const response = await fetch("http://localhost:8000/explain_conclusion", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch conclusion summary: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();

        if (typeof data.summary === "string" && data.summary.trim().length > 0) {
          setSummaryText(data.summary.trim());
        } else {
          setSummaryText(null);
        }
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        console.error("Error fetching conclusion summary", error);
        setSummaryError(
          "I couldn't generate a narrative summary for this conclusion. You can still use the structured details below."
        );
      } finally {
        setIsLoadingSummary(false);
        setIsSummaryReady(true);
      }
    };

    fetchSummary();

    return () => {
      controller.abort();
    };
  }, [modelLabel, modelProbability, userLabel, userConfidence, hasUserVerdict, agreementFlag, keyCues]);

  const showSkeleton = !isSummaryReady;

  type SkeletonSize = "title" | "subtitle" | "label" | "text" | "small";
  const SkeletonLine: React.FC<{ width?: string; size?: SkeletonSize }> = ({
    width = "100%",
    size = "text",
  }) => (
    <span
      aria-hidden="true"
      className={`conclusion-skeleton conclusion-skeleton--${size}`}
      style={{ width }}
    />
  );

  const ImagePreviewDialog: React.FC<{
    open: boolean;
    src: string;
    label: string;
    onClose: () => void;
  }> = ({ open, src, label, onClose }) => (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle className="recap-dialog-title">
        {label}
        <IconButton
          aria-label="Close image preview"
          onClick={onClose}
          className="recap-dialog-close"
        >
          ×
        </IconButton>
      </DialogTitle>
      <DialogContent className="recap-dialog-content">
        <img src={src} alt={label} className="recap-dialog-image" />
      </DialogContent>
    </Dialog>
  );

  const CueRecapCard: React.FC<{ item: CueRecapItem }> = ({ item }) => {
    const judgementLabel = getJudgementLabel(item.userJudgement);
    return (
      <Card className="recap-card" variant="outlined">
        <CardHeader
          title={item.title}
          className="recap-card-header"
          action={
            <Chip
              size="small"
              label={judgementLabel}
              className="recap-judgement-chip"
            />
          }
        />
        <CardContent className="recap-card-content">
          <div className="recap-card-body">
            <div className="recap-images">
              <div className="recap-image-panel">
                <span className="recap-image-label">Original</span>
                {item.originalSrc ? (
                  <button
                    type="button"
                    className="recap-image-button"
                    onClick={() =>
                      setPreviewImage({ src: item.originalSrc as string, label: "Original" })
                    }
                  >
                    <img
                      src={item.originalSrc}
                      alt={`${item.title} original`}
                      className="recap-image"
                    />
                  </button>
                ) : (
                  <div className="recap-image-placeholder">No image</div>
                )}
              </div>

              {!item.isCustom && (
                <div className="recap-image-panel">
                  <span className="recap-image-label">Model focus</span>
                  {item.overlaySrc ? (
                    <button
                      type="button"
                      className="recap-image-button"
                      onClick={() =>
                        setPreviewImage({
                          src: item.overlaySrc as string,
                          label: "Model focus",
                        })
                      }
                    >
                      <img
                        src={item.overlaySrc}
                        alt={`${item.title} model focus`}
                        className="recap-image"
                      />
                    </button>
                  ) : (
                    <div className="recap-image-placeholder">No overlay available</div>
                  )}
                </div>
              )}
            </div>

            <Stack spacing={0.6} className="recap-text">
              <span className="recap-text-line">Your judgment: {judgementLabel}</span>
              {typeof item.modelScore === "number" && (
                <span className="recap-text-line">
                  Model focus score: {Math.round(item.modelScore * 100)}%
                </span>
              )}
              {item.isCustom && item.observationText && (
                <span className="recap-text-line recap-text-observation">
                  Your observation: {item.observationText}
                </span>
              )}
            </Stack>
          </div>
        </CardContent>
      </Card>
    );
  };

  const CueRecapSection: React.FC = () => {
    if (!cueRecapItems || cueRecapItems.length === 0 || showSkeleton) {
      return null;
    }

    return (
      <section className="conclusion-section">
        <h3 className="conclusion-section-title">Cue recap</h3>
        <div className="recap-card-list">
          {cueRecapItems.map((item) => (
            <CueRecapCard key={item.id} item={item} />
          ))}
        </div>
      </section>
    );
  };

  const handleDownloadInfo = () => {
    if (typeof window === "undefined") return;

    const lines: string[] = [];

    lines.push("=== TrueSight Conclusion Summary ===");
    lines.push("");

    // Model vs user verdict
    lines.push(`Final joint conclusion: ${jointOutcomeTitle ?? finalVerdictText}`);
    lines.push(`Model prediction: ${modelVerdictText}`);
    lines.push(
      `Model confidence: ${
        probabilityText ? probabilityText : "not available"
      }`
    );
    lines.push(
      `Your judgement: ${
        userVerdictText ? userVerdictText : "not provided"
      }`
    );
    if (finalConclusion?.kind === "PARTIAL_AGREEMENT") {
      lines.push(
        "Agreement: You and the model did not fully agree. The model has a prediction while your assessment is uncertain."
      );
    } else if (hasUserVerdict && agreementFlag != null) {
      lines.push(
        `Agreement: ${
          agreementFlag
            ? "You and the model reached the same conclusion."
            : "You and the model reached different conclusions."
        }`
      );
    } else {
      lines.push("Agreement: not available (no user judgement provided).");
    }

    if (impactLabel) {
      lines.push(`Impact level: ${impactLabel}`);
    }

    lines.push("");

    // GPT narrative summary
    lines.push("Narrative summary:");
    if (summaryText) {
      lines.push(summaryText);
    } else {
      lines.push(
        "No narrative summary was generated. The structured details below still describe the decision."
      );
    }

    lines.push("");
    lines.push("Key cues:");
    if (keyCues && keyCues.length > 0) {
      keyCues.forEach((cue) => {
        const focusPct =
          cue.score != null ? `${Math.round(cue.score * 100)}% model attention` : "n/a";
        lines.push(
          `- ${cue.label} (model attention: ${focusPct}, source: ${
            cue.source ?? "model"
          })`
        );
        if (cue.userJudgement) {
          if (cue.userJudgement === "agree") {
            lines.push("  Your judgement: You found this cue suspicious or relevant.");
          } else if (cue.userJudgement === "disagree") {
            lines.push("  Your judgement: You did not find this cue suspicious.");
          } else if (cue.userJudgement === "not_sure") {
            lines.push("  Your judgement: You were unsure how to interpret this cue.");
          }
        } else if (cue.note) {
          lines.push(`  Note: ${cue.note}`);
        }
      });
    } else {
      lines.push("No specific key cues were highlighted.");
    }

    const content = lines.join("\n");
    const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "truesight-conclusion.txt";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  return (
    <section className="step step-conclusion">
      {previewImage && (
        <ImagePreviewDialog
          open={Boolean(previewImage)}
          src={previewImage.src}
          label={previewImage.label}
          onClose={() => setPreviewImage(null)}
        />
      )}
      {/* Header */}
      <header className="step-header">
        <h2 className="step-title">
          {showSkeleton ? <SkeletonLine size="title" width="55%" /> : "Step 5: Final joint conclusion"}
        </h2>
        <p className="step-subtitle">
          {showSkeleton ? (
            <SkeletonLine size="subtitle" width="70%" />
          ) : (
            "This step brings together the model's assessment and your judgement into a single, auditable conclusion."
          )}
        </p>
      </header>

      {/* Main layout */}
      <div className="step-conclusion-layout">
        {/* Hero verdict card */}
        <div className="conclusion-hero-card">
          <div className="conclusion-hero-label">
            {showSkeleton ? <SkeletonLine size="label" width="45%" /> : "Final joint conclusion"}
          </div>
          <h3 className="conclusion-hero-verdict">
            {showSkeleton ? (
              <SkeletonLine size="title" width="60%" />
            ) : (
              jointOutcomeTitle ?? finalVerdictText
            )}
          </h3>
          {!showSkeleton && partialAgreementSubtitle && (
            <div className="conclusion-text">{partialAgreementSubtitle}</div>
          )}

          <div className="conclusion-hero-pill-row">
            <div className="conclusion-pill">
              <span className="conclusion-pill-label">
                {showSkeleton ? <SkeletonLine size="label" width="35%" /> : "Model prediction"}
              </span>
              <span className="conclusion-pill-value">
                {showSkeleton ? (
                  <SkeletonLine size="text" width="65%" />
                ) : aiDecision ? (
                  `${getDecisionLabel(aiDecision)}${probabilityText ? ` • ${probabilityText} confidence` : ""}`
                ) : probabilityText ? (
                  `${probabilityText} confidence`
                ) : (
                  "Confidence not available"
                )}
              </span>
            </div>

            {hasUserVerdict && userVerdictText && (
              <div className="conclusion-pill conclusion-pill--user">
                <span className="conclusion-pill-label">
                  {showSkeleton ? <SkeletonLine size="label" width="40%" /> : "Your judgement"}
                </span>
                <span className="conclusion-pill-value">
                  {showSkeleton ? (
                    <SkeletonLine size="text" width="55%" />
                  ) : humanDecision ? (
                    getDecisionLabel(humanDecision)
                  ) : (
                    userVerdictText
                  )}
                </span>
              </div>
            )}
          </div>

          {hasUserVerdict && jointOutcomeCallout && (
            <div
              className={
                "conclusion-agreement-banner" +
                (jointOutcomeCallout.tone === "agree"
                  ? " conclusion-agreement-banner--agree"
                  : " conclusion-agreement-banner--disagree")
              }
            >
              {showSkeleton ? (
                <>
                  <span className="conclusion-agreement-icon">*</span>
                  <span style={{ flex: 1 }}>
                    <SkeletonLine size="small" width="92%" />
                    <span style={{ display: "block", marginTop: "0.35rem" }}>
                      <SkeletonLine size="small" width="70%" />
                    </span>
                  </span>
                </>
              ) : (
                <>
                  <span className="conclusion-agreement-icon">{jointOutcomeCallout.icon}</span>
                  <span>{jointOutcomeCallout.text}</span>
                </>
              )}
            </div>
          )}

          {impactLabel && (
            <div className="conclusion-impact-row">
              <span className="conclusion-impact-label">
                {showSkeleton ? <SkeletonLine size="label" width="40%" /> : "Potential impact"}
              </span>
              <span className="conclusion-impact-value">
                {showSkeleton ? <SkeletonLine size="text" width="55%" /> : impactLabel}
              </span>
            </div>
          )}
        </div>

        {/* Right-hand column: process + cues + reflection */}
        <div className="conclusion-detail-column">
          {/* GPT-powered narrative summary */}
          <section className="conclusion-section">
            <h3 className="conclusion-section-title">
              {showSkeleton ? <SkeletonLine size="subtitle" width="58%" /> : "Summary of this conclusion"}
            </h3>
            {showSkeleton ? (
              <div>
                <SkeletonLine size="text" width="95%" />
                <span style={{ display: "block", marginTop: "0.5rem" }}>
                  <SkeletonLine size="text" width="88%" />
                </span>
                <span style={{ display: "block", marginTop: "0.5rem" }}>
                  <SkeletonLine size="text" width="80%" />
                </span>
              </div>
            ) : (
              <>
                {isLoadingSummary && (
                  <p className="conclusion-text">
                    Summarizing how you and the model arrived at this conclusion…
                  </p>
                )}
                {summaryError && (
                  <p className="conclusion-text conclusion-text--muted">
                    {summaryError}
                  </p>
                )}
                {!isLoadingSummary && !summaryError && summaryText && (
                  <p className="conclusion-text">{summaryText}</p>
                )}
                {!isLoadingSummary && !summaryError && !summaryText && (
                  <p className="conclusion-text">
                    This step summarizes how the model&apos;s focus and your cue-level judgments led to the final
                    decision.
                  </p>
                )}
              </>
            )}
          </section>

          <CueRecapSection />

          {/* How you got here */}
          <section className="conclusion-section">
            <h3 className="conclusion-section-title">
              {showSkeleton ? <SkeletonLine size="subtitle" width="45%" /> : "How you got here"}
            </h3>
            {showSkeleton ? (
              <ul className="conclusion-timeline">
                {[0, 1, 2, 3].map((item) => (
                  <li key={item}>
                    <SkeletonLine size="text" width={`${90 - item * 6}%`} />
                  </li>
                ))}
              </ul>
            ) : (
              <ul className="conclusion-timeline">
                <li>1. The image was uploaded and preprocessed.</li>
                <li>2. The model produced an initial prediction.</li>
                <li>3. You inspected the model&apos;s focus and explanation cues.</li>
                <li>4. A joint conclusion was derived by comparing the model&apos;s prediction with your cue-based assessment.</li>
              </ul>
            )}
          </section>

          {/* Key cues */}
          {keyCues && keyCues.length > 0 && (
            <section className="conclusion-section">
              <h3 className="conclusion-section-title">
                {showSkeleton ? <SkeletonLine size="subtitle" width="70%" /> : "Key cues that influenced this conclusion"}
              </h3>
              {showSkeleton ? (
                <ul className="conclusion-cue-list">
                  {[0, 1].map((item) => (
                    <li key={item} className="conclusion-cue-item">
                      <div className="conclusion-cue-main-row">
                        <SkeletonLine size="text" width="60%" />
                        <SkeletonLine size="small" width="25%" />
                      </div>
                      <div className="conclusion-cue-meta-row">
                        <SkeletonLine size="small" width="30%" />
                      </div>
                      <SkeletonLine size="small" width="80%" />
                    </li>
                  ))}
                </ul>
              ) : (
                <ul className="conclusion-cue-list">
                  {keyCues.map((cue) => {
                    const focusPct =
                    cue.score != null ? `${Math.round(cue.score * 100)}% model attention` : null;

                    return (
                      <li key={cue.id} className="conclusion-cue-item">
                        <div className="conclusion-cue-main-row">
                          <span className="conclusion-cue-label">{cue.label}</span>
                          {focusPct && <span className="conclusion-cue-score">{focusPct}</span>}
                        </div>

                        <div className="conclusion-cue-meta-row">
                          {cue.source && (
                            <span className="conclusion-cue-chip">
                              {cue.source === "model" && "Model-driven"}
                              {cue.source === "user" && "You selected this"}
                              {cue.source === "both" && "Model & you"}
                            </span>
                          )}
                        </div>

                        {cue.userJudgement && (
                          <div className="conclusion-cue-user-answer">
                            Your judgement:{" "}
                            {cue.userJudgement === "agree" &&
                              "You found this cue suspicious or relevant."}
                            {cue.userJudgement === "disagree" &&
                              "You did not find this cue suspicious."}
                            {cue.userJudgement === "not_sure" &&
                              "You were unsure how to interpret this cue."}
                          </div>
                        )}

                        {!cue.userJudgement && cue.note && (
                          <span className="conclusion-cue-note">{cue.note}</span>
                        )}
                      </li>
                    );
                  })}
                </ul>
              )}
            </section>
          )}

          {/* Risk / impact explanation (text-only for now) */}
          <section className="conclusion-section">
            <h3 className="conclusion-section-title">
              {showSkeleton ? <SkeletonLine size="subtitle" width="62%" /> : "If this conclusion were wrong…"}
            </h3>
            {showSkeleton ? (
              <div>
                <SkeletonLine size="text" width="92%" />
                <span style={{ display: "block", marginTop: "0.5rem" }}>
                  <SkeletonLine size="text" width="85%" />
                </span>
                <span style={{ display: "block", marginTop: "0.5rem" }}>
                  <SkeletonLine size="text" width="78%" />
                </span>
              </div>
            ) : (
              <p className="conclusion-text">
                In a real-world scenario, the impact of a wrong decision depends heavily on the context. A misclassified
                meme might be harmless, while a misclassified news or political image could contribute to multimodal
                misinformation. This prototype step is designed to make that risk explicit and encourage careful
                reflection.
              </p>
            )}
          </section>

          {/* Actions */}
          <section className="conclusion-actions">
            <button
              type="button"
              className="btn btn-primary conclusion-action-main"
              onClick={handleDownloadInfo}
              disabled={showSkeleton}
            >
              {showSkeleton ? <SkeletonLine size="text" width="45%" /> : "Download info"}
            </button>
          </section>
        </div>
      </div>
    </section>
  );
};

export default StepConclusion;
