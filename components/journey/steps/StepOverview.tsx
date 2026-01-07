"use client";

import React from "react";
import { Tooltip } from "@mui/material";
import "./StepOverview.module.css";

interface PredictionScore {
  label: string;
  score: number;
}

interface BackendCue {
  id: number;
  title: string;
  score: number;
}

interface PredictionResult {
  prediction: string;
  confidence: number;
  scores: PredictionScore[];
  cues: BackendCue[];
}

interface StepOverviewProps {
  selectedCueId?: number | null;
  onSelectCue?: (cueId: number) => void;
  predictionResult?: PredictionResult | null;
}

const getCueSeverityLabel = (score: number): string => {
  if (score >= 0.75) {
    return "High model attention";
  }
  if (score >= 0.4) {
    return "Medium model attention";
  }
  if (score > 0.1) {
    return "Low model attention";
  }
  return "Low model attention";
};

const getCueSeverityClass = (score: number): string => {
  if (score >= 0.75) {
    return "overview-severity overview-severity--high";
  }
  if (score >= 0.4) {
    return "overview-severity overview-severity--medium";
  }
  if (score > 0.1) {
    return "overview-severity overview-severity--low";
  }
  return "overview-severity overview-severity--minimal";
};

const StepOverview: React.FC<StepOverviewProps> = ({
  selectedCueId,
  onSelectCue,
  predictionResult,
}) => {
  const modelAttentionTooltip =
    "Model attention indicates how strongly the AI attended to this cue during its prediction, based on Grad-CAM activations. It does not indicate how important the cue is objectively.";

  const handleSelect = (cueId: number) => {
    if (onSelectCue) {
      onSelectCue(cueId);
    }
  };

  const sortedCues: BackendCue[] =
    predictionResult?.cues
      ?.slice()
      .sort((a, b) => b.score - a.score) ?? [];

  return (
    <div className="journey-step journey-step-overview">
      {/* Agent bubble */}
      <div className="journey-agent-bubble journey-agent-bubble--wide">
        <p>
          I&apos;ve completed the initial analysis. Here are the key visual cues
          we&apos;ll review together.
        </p>
      </div>

      {/* Cue overview card */}
      <div className="overview-card">
        <div className="overview-header">
          <h3 className="overview-title">Cue Overview</h3>
          <p className="overview-subtitle">
            These are the visual cues the model relied on when forming its internal assessment.
          </p>
        </div>

        <div className="overview-list">
          {sortedCues.length === 0 && (
            <div className="overview-empty">
              No visual cues were detected for this image.
            </div>
          )}

          {sortedCues.map((cue) => {
            const isActive = cue.id === selectedCueId;
            const severityLabel = getCueSeverityLabel(cue.score);
            const severityClass = getCueSeverityClass(cue.score);

            return (
              <button
                key={cue.id}
                type="button"
                className={
                  "overview-row" +
                  (isActive ? " overview-row--active" : "")
                }
                onClick={() => handleSelect(cue.id)}
              >
                <div className="overview-row-main">
                  <div className="overview-icon">
                    <span className="overview-icon-symbol">☀️</span>
                  </div>

                  <div className="overview-text">
                    <span className="overview-text-title">{cue.title}</span>
                    <span className="overview-text-description">
                      {severityLabel}
                      <Tooltip title={modelAttentionTooltip} arrow>
                        <span
                          className="overview-info-icon"
                          role="img"
                          aria-label="Model attention info"
                          tabIndex={0}
                        >
                          ⓘ
                        </span>
                      </Tooltip>
                    </span>
                  </div>
                </div>

                <div className="overview-row-meta">
                  <span className={severityClass}></span>
                  <span className="overview-link">View details</span>
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default StepOverview;
