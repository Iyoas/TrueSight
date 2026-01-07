import React, { useEffect, useState } from "react";
import "./StepConclusion.module.css";

export type VerdictLabel = "ai" | "human" | "uncertain";

export interface ConclusionCueSummary {
  id: number | string;
  label: string;
  score?: number; // 0–1 model focus
  source?: "model" | "user" | "both";
  note?: string;
  userJudgement?: "agree" | "not_sure" | "disagree";
}

export interface StepConclusionProps {
  // Core outcome
  modelLabel: VerdictLabel;
  modelProbability?: number; // 0–1
  userLabel?: VerdictLabel;

  // Optional metadata
  keyCues?: ConclusionCueSummary[];
  userConfidence?: number | null; // 0–1
  modelConfidenceText?: string;

  // Optional impact / context
  impactLevel?: "low" | "medium" | "high";

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

const StepConclusion: React.FC<StepConclusionProps> = ({
  modelLabel,
  modelProbability,
  userLabel,
  keyCues = [],
  userConfidence,
  modelConfidenceText,
  impactLevel,
  onFinish,
  onTryAnother,
}) => {
  const modelVerdictText = verdictLabelToText(modelLabel);
  const userVerdictText = userLabel ? verdictLabelToText(userLabel) : null;
  const finalVerdictLabel: VerdictLabel = userLabel ?? modelLabel;
  const finalVerdictText = verdictLabelToText(finalVerdictLabel);
  const probabilityText = formatProbability(modelProbability);

  const hasUserVerdict = Boolean(userLabel);
  const isAgreement = hasUserVerdict && userLabel === modelLabel;

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
          agreement: hasUserVerdict ? isAgreement : null,
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
  }, [modelLabel, modelProbability, userLabel, userConfidence, hasUserVerdict, isAgreement, keyCues]);

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

  const handleDownloadInfo = () => {
    if (typeof window === "undefined") return;

    const lines: string[] = [];

    lines.push("=== TrueSight Conclusion Summary ===");
    lines.push("");

    // Model vs user verdict
    lines.push(`Final conclusion (human-led): ${finalVerdictText}`);
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
    if (hasUserVerdict) {
      lines.push(
        `Agreement: ${
          isAgreement
            ? "You and the model agree on this image."
            : "You and the model disagree on this image."
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
          cue.score != null ? `${Math.round(cue.score * 100)}% model focus` : "n/a";
        lines.push(
          `- ${cue.label} (model focus: ${focusPct}, source: ${
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
      {/* Header */}
      <header className="step-header">
        <h2 className="step-title">
          {showSkeleton ? <SkeletonLine size="title" width="55%" /> : "Step 5: Final conclusion"}
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
            {showSkeleton ? <SkeletonLine size="label" width="45%" /> : "Final conclusion (human-led)"}
          </div>
          <h3 className="conclusion-hero-verdict">
            {showSkeleton ? <SkeletonLine size="title" width="60%" /> : finalVerdictText}
          </h3>

          <div className="conclusion-hero-pill-row">
            <div className="conclusion-pill">
              <span className="conclusion-pill-label">
                {showSkeleton ? <SkeletonLine size="label" width="35%" /> : "Model prediction"}
              </span>
              <span className="conclusion-pill-value">
                {showSkeleton ? (
                  <SkeletonLine size="text" width="65%" />
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
                  {showSkeleton ? <SkeletonLine size="text" width="55%" /> : userVerdictText}
                </span>
              </div>
            )}
          </div>

          {hasUserVerdict && (
            <div
              className={
                "conclusion-agreement-banner" +
                (isAgreement ? " conclusion-agreement-banner--agree" : " conclusion-agreement-banner--disagree")
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
              ) : isAgreement ? (
                <>
                  <span className="conclusion-agreement-icon">✅</span>
                  <span>
                    You and the model <strong>agree</strong> on this image. This can increase trust, but keep in mind
                    that both human and AI can still be wrong.
                  </span>
                </>
              ) : (
                <>
                  <span className="conclusion-agreement-icon">⚠️</span>
                  <span>
                    You and the model <strong>disagree</strong>. This is a valuable example of human–AI disagreement and
                    worth a closer look.
                  </span>
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
                <li>4. You formed a final judgement based on both model and your own reasoning.</li>
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
                      cue.score != null ? `${Math.round(cue.score * 100)}% model focus` : null;

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
