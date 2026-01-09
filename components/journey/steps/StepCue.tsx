"use client";

import React, { useState, useEffect } from "react";
import { Skeleton } from "@mui/material";
import CueImageMagnifier from "./CueImageMagnifier";
import "./StepCue.module.css";

type Decision = "agree" | "not_sure" | "disagree" | null;

interface StepCueProps {
  title?: string;
  subtitle?: string;
  imageSrc?: string;
  originalImageSrc?: string;
  cueText?: string | null;
  cueTextStatus?: "idle" | "loading" | "ready" | "error";
  helperText?: {
    whereToLook: string;
    whatToCheck: string;
    bothWaysSupport: string;
    bothWaysChallenge: string;
    questions: string[];
  } | null;
  helperTextStatus?: "idle" | "loading" | "ready" | "error";
  onRetryHelperText?: () => void;
  cueType?: string;
  predictionLabel?: string; // "ai" | "hum"
  predictionConfidence?: number; // 0-1
  cueScore?: number; // 0-1
  imageContext?: string; // optional extra context text from the UI
  image_base64?: string; // original uploaded image as data URL
  heatmap_base64?: string; // cue-specific or global heatmap as data URL
  cueId?: number;
  cueSource?: "model" | "user";
  userObservationText?: string;
  onDecisionChange?: (decision: Exclude<Decision, null>) => void;
  existingAnswer?: Decision;
}

const StepCue: React.FC<StepCueProps> = ({
  title = "Lighting",
  subtitle = "This cue highlights one aspect of the model's focus in its decision.",
  imageSrc = "/heatmap-picture.png",
  originalImageSrc,
  cueText,
  cueTextStatus = "idle",
  helperText,
  helperTextStatus = "idle",
  onRetryHelperText,
  cueId,
  cueSource,
  onDecisionChange,
  existingAnswer,
}) => {
  const [decision, setDecision] = useState<Decision>(existingAnswer ?? null);
  const isUserCue = cueSource === "user";
  const isHowToReadLoading = isUserCue
    ? helperTextStatus === "loading" || helperTextStatus === "idle"
    : cueTextStatus === "loading" || cueTextStatus === "idle";

  useEffect(() => {
    setDecision(existingAnswer ?? null);
  }, [existingAnswer]);

  const handleDecision = (value: Exclude<Decision, null>) => {
    setDecision(value);
    if (onDecisionChange && cueId !== undefined) {
      onDecisionChange(value);
    }
  };

  const normalizedTitle = (title || "").toLowerCase();

  let agentIntro: string;
  switch (normalizedTitle) {
    case "texture":
      agentIntro =
        "Let's take a closer look at the textures in this image. Here is how the model attended to this aspect.";
      break;
    case "lighting":
      agentIntro =
        "Let's look closely at the lighting. Here is how this aspect influenced the model's decision.";
      break;
    case "background":
      agentIntro =
        "Let's examine the background areas. These regions often reveal where the model concentrated its attention.";
      break;
    case "geometry":
      agentIntro =
        "Let's focus on the shapes and geometry. This cue reflects how the model responded to structure in the image.";
      break;
    default:
      agentIntro =
        "Let's look closely at this cue. Here is how it contributed to the model's decision.";
      break;
  }

  return (
    <div className="journey-step journey-step-cue">
      {/* Agent bubble */}
      <div className="journey-agent-bubble journey-agent-bubble--wide">
        <p>{agentIntro}</p>
      </div>

      {/* Cue header */}
      <div className="cue-header">
        <h3 className="cue-title">{title}</h3>
        <p className="cue-subtitle">{subtitle}</p>
      </div>

      {/* Two-column layout: original image + overlay + explanation */}
      <div className="cue-content">
        <div className="cue-images">
          {originalImageSrc && (
            <div className="cue-image-panel">
              <div className="cue-image-label">Original image</div>
              <div className="cue-image-wrapper cue-image-wrapper--original">
                <CueImageMagnifier src={originalImageSrc} alt="Original uploaded image" />
              </div>
            </div>
          )}

          <div className="cue-image-panel cue-image-panel--overlay">
            <div className="cue-image-label">
              {originalImageSrc ? "Model focus overlay" : "Model focus"}
            </div>
            <div className="cue-image-wrapper cue-image-wrapper--overlay">
              <CueImageMagnifier
                src={imageSrc}
                alt="Model focus overlay for the selected cue"
              />
            </div>
            <p className="cue-image-legend">
              The overlay highlights where the model attended most when making its decision. It is not
              a direct AI-vs-human map, but a visualization of relative attention.
            </p>
          </div>
        </div>

        <div className="cue-explanation">
      {isHowToReadLoading ? (
        <div aria-hidden="true" style={{ width: "100%" }}>
          <Skeleton variant="text" height={32} style={{ width: "100%" }} />
          <Skeleton variant="text" style={{ width: "100%" }} />
          <Skeleton variant="text" style={{ width: "100%" }} />
          <Skeleton variant="text" style={{ width: "100%" }} />
        </div>
      ) : (
        <>
          <h4 className="cue-explanation-title">How to read this cue</h4>

          {isUserCue ? (
            helperTextStatus === "ready" && helperText ? (
              <>
                <p className="cue-explanation-text">{helperText.whereToLook}</p>
                <p className="cue-explanation-text">{helperText.whatToCheck}</p>
                <p className="cue-explanation-text">{helperText.bothWaysSupport}</p>
                <p className="cue-explanation-text">{helperText.bothWaysChallenge}</p>
                {helperText.questions && helperText.questions.length > 0 && (
                  <ul className="cue-explanation-list">
                    {helperText.questions.map((q, index) => (
                      <li key={index} className="cue-explanation-list-item">
                        {q}
                      </li>
                    ))}
                  </ul>
                )}
              </>
            ) : (
              <div className="cue-explanation-error">
                <p className="cue-explanation-text cue-explanation-text--error">
                  Couldn't load explanation.
                </p>
                {onRetryHelperText && (
                  <button
                    type="button"
                    className="cue-explanation-retry"
                    onClick={onRetryHelperText}
                  >
                    Retry
                  </button>
                )}
              </div>
            )
          ) : cueText ? (
            <p className="cue-explanation-text">{cueText}</p>
          ) : (
            <p className="cue-explanation-text cue-explanation-text--error">
              Couldn't load explanation.
            </p>
          )}
        </>
      )}
    </div>
      </div>

      {/* Decision section */}
      <div className="cue-decisions">
        <h4 className="cue-decisions-title">Your judgment on this cue</h4>
        <div className="cue-decision-buttons">
          <button
            type="button"
            className={
              "cue-decision-button" +
              (decision === "agree" ? " cue-decision-button--active" : "")
            }
            onClick={() => handleDecision("agree")}
          >
            I find this cue suspicious
          </button>
          <button
            type="button"
            className={
              "cue-decision-button" +
              (decision === "not_sure" ? " cue-decision-button--active" : "")
            }
            onClick={() => handleDecision("not_sure")}
          >
            I&apos;m not sure
          </button>
          <button
            type="button"
            className={
              "cue-decision-button" +
              (decision === "disagree" ? " cue-decision-button--active" : "")
            }
            onClick={() => handleDecision("disagree")}
          >
            I don&apos;t find this cue suspicious
          </button>
        </div>
      </div>
    </div>
  );
};

export default StepCue;
