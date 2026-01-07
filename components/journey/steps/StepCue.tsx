"use client";

import React, { useState, useEffect } from "react";
import Image from "next/image";
import "./StepCue.module.css";

type Decision = "agree" | "not_sure" | "disagree" | null;

interface StepCueProps {
  title?: string;
  subtitle?: string;
  imageSrc?: string;
  originalImageSrc?: string;
  explanation?: string; // optional fallback explanation from the parent
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

/**
 * Explanation object returned by the backend /explain_cue endpoint.
 */
type CueExplanation = {
  summary: string;
  bullets: string[];
  questions: string[];
};

const StepCue: React.FC<StepCueProps> = ({
  title = "Lighting",
  subtitle = "This cue highlights one aspect of the model's focus in its decision.",
  imageSrc = "/heatmap-picture.png",
  originalImageSrc,
  explanation,
  cueType,
  predictionLabel,
  predictionConfidence,
  cueScore,
  imageContext,
  image_base64,
  heatmap_base64,
  cueId,
  cueSource,
  userObservationText,
  onDecisionChange,
  existingAnswer,
}) => {
  const [decision, setDecision] = useState<Decision>(existingAnswer ?? null);
  const [isLoadingExplanation, setIsLoadingExplanation] = useState(false);
  const [explanationError, setExplanationError] = useState<string | null>(null);
  const [explanationData, setExplanationData] = useState<CueExplanation | null>(null);

  useEffect(() => {
    setDecision(existingAnswer ?? null);
  }, [existingAnswer]);

  // Prefer an explicit heatmap_base64 from props; otherwise, if imageSrc looks like a data URL,
  // treat it as a heatmap image for the multimodal explanation.
  const effectiveHeatmapBase64 =
    heatmap_base64 || (imageSrc && imageSrc.startsWith("data:image") ? imageSrc : undefined);

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
        "Let's take a closer look at the textures in this image. Here is how the model focused on this aspect.";
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

  const getFallbackExplanation = (): string => {
    if (explanation) {
      return explanation;
    }

    switch (normalizedTitle) {
      case "texture":
        return "This texture cue highlights where the model was sensitive to surface detail. You can use it to check whether surfaces look naturally detailed, or unusually smooth or repetitive.";
      case "lighting":
        return "This lighting cue highlights how light and shadow are distributed in the image. You can use it to check whether illumination and shadows look consistent with a real scene.";
      case "background":
        return "This background cue shows where the model paid attention to regions behind the main subject. You can check whether the background looks coherent, with realistic depth and structure.";
      case "geometry":
        return "This geometry cue emphasizes shapes, edges, and alignment. It can help you inspect whether object proportions and contours look plausible.";
      default:
        return "This visual cue summarizes one aspect of what the model focused on. Use it to inspect the highlighted regions more carefully.";
    }
  };

  useEffect(() => {
    const controller = new AbortController();

    const fetchExplanation = async () => {
      if (cueSource === "user") {
        if (!userObservationText || !image_base64) {
          return;
        }
        try {
          setIsLoadingExplanation(true);
          setExplanationError(null);
          setExplanationData(null);

          const payload = {
            image_base64,
            user_observation: userObservationText,
            model_prediction: predictionLabel ?? null,
            model_confidence: predictionConfidence ?? null,
          };

          const response = await fetch("http://localhost:8000/explain_user_cue", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
            signal: controller.signal,
          });

          if (!response.ok) {
            throw new Error(`Failed to fetch user cue explanation: ${response.status} ${response.statusText}`);
          }

          const data = await response.json();

          setExplanationData({
            summary: data.summary ?? "",
            bullets: Array.isArray(data.bullets) ? data.bullets : [],
            questions: Array.isArray(data.questions) ? data.questions : [],
          });
        } catch (error) {
          if (controller.signal.aborted) {
            return;
          }
          console.error("Error fetching user cue explanation", error);
          setExplanationError("Could not generate guidance for this observation.");
        } finally {
          setIsLoadingExplanation(false);
        }
        return;
      }

      if (
        !predictionLabel ||
        predictionConfidence === undefined ||
        predictionConfidence === null ||
        cueScore === undefined ||
        cueScore === null
      ) {
        return;
      }

      try {
        setIsLoadingExplanation(true);
        setExplanationError(null);
        setExplanationData(null);

        const payload = {
          cue_type: (cueType || normalizedTitle || "cue"),
          cue_title: title,
          prediction_label: predictionLabel,
          prediction_confidence: predictionConfidence,
          cue_score: cueScore,
          image_context: imageContext ?? undefined,
          image_base64: image_base64 ?? undefined,
          heatmap_base64: effectiveHeatmapBase64 ?? undefined,
        };

        const response = await fetch("http://localhost:8000/explain_cue", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch cue explanation: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();

        setExplanationData({
          summary: data.summary ?? "",
          bullets: Array.isArray(data.bullets) ? data.bullets : [],
          questions: Array.isArray(data.questions) ? data.questions : [],
        });
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        console.error("Error fetching cue explanation", error);
        setExplanationError("I couldn't generate a detailed explanation for this cue. You can still use the highlight to inspect the image yourself.");
      } finally {
        setIsLoadingExplanation(false);
      }
    };

    fetchExplanation();

    return () => {
      controller.abort();
    };
  }, [
    cueSource,
    cueType,
    cueScore,
    imageContext,
    normalizedTitle,
    predictionConfidence,
    predictionLabel,
    title,
    image_base64,
    effectiveHeatmapBase64,
    userObservationText,
  ]);

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
                <Image
                  src={originalImageSrc}
                  alt="Original uploaded image"
                  fill
                  sizes="(max-width: 768px) 100vw, 50vw"
                  className="cue-image"
                />
              </div>
            </div>
          )}

          <div className="cue-image-panel cue-image-panel--overlay">
            <div className="cue-image-label">
              {originalImageSrc ? "Model focus overlay" : "Model focus"}
            </div>
            <div className="cue-image-wrapper cue-image-wrapper--overlay">
              <Image
                src={imageSrc}
                alt="Model focus overlay for the selected cue"
                fill
                sizes="(max-width: 768px) 100vw, 50vw"
                className="cue-image"
              />
            </div>
            <p className="cue-image-legend">
              The overlay highlights where the model focused most when making its decision. It is not
              a direct AI-vs-human map, but a visualization of relative attention.
            </p>
          </div>
        </div>

        <div className="cue-explanation">
          <h4 className="cue-explanation-title">How to read this cue</h4>

          {isLoadingExplanation && (
            <p className="cue-explanation-text cue-explanation-text--muted">
              Analyzing this cue in more detail...
            </p>
          )}

          {explanationError && (
            <p className="cue-explanation-text cue-explanation-text--error">
              {explanationError}
            </p>
          )}

          {explanationData ? (
            <>
              {explanationData.summary && (
                <p className="cue-explanation-text">{explanationData.summary}</p>
              )}

              {explanationData.bullets && explanationData.bullets.length > 0 && (
                <ul className="cue-explanation-list">
                  {explanationData.bullets.map((item, index) => (
                    <li key={index} className="cue-explanation-list-item">
                      {item}
                    </li>
                  ))}
                </ul>
              )}

              {explanationData.questions && explanationData.questions.length > 0 && (
                <div className="cue-questions">
                  <h5 className="cue-questions-title">Questions to guide your review</h5>
                  <ul className="cue-questions-list">
                    {explanationData.questions.map((q, index) => (
                      <li key={index} className="cue-questions-list-item">
                        {q}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          ) : (
            !isLoadingExplanation &&
            !explanationError && (
              <p className="cue-explanation-text">{getFallbackExplanation()}</p>
            )
          )}
        </div>
      </div>

      {/* Decision section */}
      <div className="cue-decisions">
        <h4 className="cue-decisions-title">Your judgment on this cue</h4>
        <p className="cue-decisions-description">
          Use the questions above to reflect, then choose whether you consider this cue suspicious
          or not. Your answer does not change the model&apos;s prediction, but it helps you reason
          about how much weight to give this cue.
        </p>

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
