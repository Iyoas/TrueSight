"use client";

import React, { useState, ChangeEvent, useRef } from "react";

import "./JourneySection.module.css";
import StepAnalyzing from "./steps/StepAnalyzing";
import StepOverview from "./steps/StepOverview";
import StepCue from "./steps/StepCue";
import StepConclusion, { type VerdictLabel, type ConclusionCueSummary } from "./steps/StepConclusion";
import {
  computeFinalConclusion,
  computeHumanDecision,
  type AIDecision,
  type CanonicalCueValue,
  type HumanDecision,
} from "../../lib/decisionLogic";

type StepId = 1 | 2 | 3 | 4 | 5;

interface CustomCueHelperText {
  whereToLook: string;
  whatToCheck: string;
  bothWaysSupport: string;
  bothWaysChallenge: string;
  questions: string[];
}

interface Cue {
  id: number;
  title: string;
  description: string;
  source?: "model" | "user";
  cueType?: "CUSTOM";
  userObservationText?: string;
  helperTextStatus?: "idle" | "loading" | "ready" | "error";
  helperText?: CustomCueHelperText | null;
  userJudgment?: "AI" | "HUMAN" | "UNCERTAIN" | null;
  isUserProvided?: boolean;
}

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
  model: string;
  prediction: string;
  confidence: number;
  scores: PredictionScore[];
  cues: BackendCue[];
  heatmap: string;
  cue_heatmaps?: { [key: string]: string };
}

const CUE_TYPES = ["texture", "lighting", "background", "geometry"] as const;
type CueType = (typeof CUE_TYPES)[number];

const TOTAL_STEPS: StepId = 5;

function getStepSubtitle(step: StepId): string {
  switch (step) {
    case 1:
      return "Upload View";
    case 2:
      return "Analyzing View";
    case 3:
      return "Cue Overview";
    case 4:
      return "Cue Deep Dive";
    case 5:
      return "Results";
    default:
      return "";
  }
}

const JourneySection: React.FC = () => {
  const [currentStep, setCurrentStep] = useState<StepId>(1);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadedPreviewUrl, setUploadedPreviewUrl] = useState<string | null>(null);
  const [uploadedBase64, setUploadedBase64] = useState<string | null>(null);
  const [selectedCueId, setSelectedCueId] = useState<number>(1);
  // userDecision removed — decisions are stored per cue now
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(
    null
  );
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [userCues, setUserCues] = useState<Cue[]>([]);
  const [userCueText, setUserCueText] = useState<string>("");
  const [cueTextByType, setCueTextByType] = useState<Record<CueType, string | null>>({
    texture: null,
    lighting: null,
    background: null,
    geometry: null,
  });
  const [cueTextStatusByType, setCueTextStatusByType] = useState<
    Record<CueType, "idle" | "loading" | "ready" | "error">
  >({
    texture: "idle",
    lighting: "idle",
    background: "idle",
    geometry: "idle",
  });
  const cueTextCacheRef = useRef<Map<string, string>>(new Map());

  const [cueAnswers, setCueAnswers] = useState<Record<number, "agree" | "not_sure" | "disagree">>({});

  const handleCueAnswer = (cueId: number, decision: "agree" | "not_sure" | "disagree") => {
    setCueAnswers((prev) => ({
      ...prev,
      [cueId]: decision,
    }));
  };

  const deriveModelLabel = (): VerdictLabel => {
    const raw = predictionResult?.prediction?.toLowerCase() ?? "";

    if (raw.includes("human") || raw.includes("real")) {
      return "human";
    }
    if (raw.includes("ai") || raw.includes("generated") || raw.includes("fake")) {
      return "ai";
    }
    return "uncertain";
  };

  const modelLabel: VerdictLabel = deriveModelLabel();
  const modelProbability = predictionResult?.confidence;


  const modelCueSummaries: ConclusionCueSummary[] = predictionResult?.cues
    ? [...predictionResult.cues]
        .sort((a, b) => b.score - a.score)
        .slice(0, 3)
        .map((c) => {
          const userAnswer = cueAnswers[c.id];

          let note: string | undefined;
          if (userAnswer === "agree") {
            note = "You found this cue suspicious or relevant for AI detection.";
          } else if (userAnswer === "disagree") {
            note = "You did not find this cue suspicious and chose to rely more on other signals.";
          } else if (userAnswer === "not_sure") {
            note = "You were unsure how to interpret this cue.";
          }

          const source: ConclusionCueSummary["source"] =
            userAnswer != null ? "both" : "model";

          return {
            id: c.id,
            label: c.title,
            score: c.score,
            source,
            note,
            userJudgement: userAnswer,
          };
        })
    : [];

  const userCueSummaries: ConclusionCueSummary[] = userCues
    .map((cue) => {
      const userAnswer = cueAnswers[cue.id];
      if (!userAnswer) return undefined;

      return {
        id: cue.id,
        label: cue.title || "Your observation",
        source: "user" as const,
        note: cue.description,
        userJudgement: userAnswer,
      };
    })
    .filter((c): c is ConclusionCueSummary => Boolean(c));

  const keyCues: ConclusionCueSummary[] = [...modelCueSummaries, ...userCueSummaries];
  
  const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  const hashString = (value?: string | null): string => {
    if (!value) return "none";
    const step = Math.max(1, Math.floor(value.length / 500));
    let hash = 0;
    for (let i = 0; i < value.length; i += step) {
      hash = (hash * 31 + value.charCodeAt(i)) | 0;
    }
    return `${hash}_${value.length}`;
  };

  const buildCueCacheKey = (
    cueType: CueType,
    heatmapBase64: string,
    imageBase64: string | null,
    modelId: string
  ): string => {
    return `${modelId}|${cueType}|${hashString(imageBase64)}|${hashString(heatmapBase64)}`;
  };

  const getCueType = (title?: string): CueType | null => {
    const normalized = (title || "").toLowerCase();
    return CUE_TYPES.find((type) => type === normalized) ?? null;
  };

  const ensureCueTextMap = (
    value: Record<string, unknown>
  ): value is Record<CueType, string> => {
    return CUE_TYPES.every(
      (type) => typeof value[type] === "string" && (value[type] as string).trim().length > 0
    );
  };

  const modelCues: Cue[] =
    predictionResult?.cues && predictionResult.cues.length > 0
      ? predictionResult.cues.map((c) => ({
          id: c.id,
          title: c.title,
          description: `${c.title} score: ${c.score.toFixed(2)}`,
          source: "model" as const,
        }))
      : [
          {
            id: 1,
            title: "Texture",
            description: "Texture analysis placeholder.",
            source: "model" as const,
          },
          {
            id: 2,
            title: "Lighting",
            description: "Lighting analysis placeholder.",
            source: "model" as const,
          },
          {
            id: 3,
            title: "Background",
            description: "Background analysis placeholder.",
            source: "model" as const,
          },
          {
            id: 4,
            title: "Geometry",
            description: "Geometry analysis placeholder.",
            source: "model" as const,
          },
        ];

  const decisionToCanonicalCueValue = (
    decision?: "agree" | "not_sure" | "disagree"
  ): CanonicalCueValue => {
    if (decision === "agree") return "AI";
    if (decision === "disagree") return "HUMAN";
    return "UNCERTAIN";
  };

  const decisionCues = modelCues.slice(0, 4);
  const humanCueValues = decisionCues.map((cue) =>
    decisionToCanonicalCueValue(cueAnswers[cue.id])
  );
  const humanDecision: HumanDecision = computeHumanDecision(humanCueValues);

  const deriveAIDecision = (): AIDecision => {
    const raw = predictionResult?.prediction?.toLowerCase() ?? "";
    if (raw.includes("ai") || raw.includes("generated") || raw.includes("fake")) {
      return "AI_GENERATED";
    }
    if (raw.includes("human") || raw.includes("real")) {
      return "HUMAN_GENERATED";
    }
    return modelLabel === "ai" ? "AI_GENERATED" : "HUMAN_GENERATED";
  };

  const aiDecision: AIDecision = deriveAIDecision();
  const finalConclusion = computeFinalConclusion(aiDecision, humanDecision);

  const userVerdictLabel: VerdictLabel | undefined =
    humanDecision === "AI_GENERATED"
      ? "ai"
      : humanDecision === "HUMAN_GENERATED"
      ? "human"
      : "uncertain";

  const derivedCues: Cue[] = [...userCues, ...modelCues];
  const allCuesAnswered = derivedCues.length > 0 && derivedCues.every((cue) => cueAnswers[cue.id] !== undefined);

  const getCueIndex = (cueId: number) => derivedCues.findIndex((cue) => cue.id === cueId);
  const getNextCueId = (cueId: number): number | null => {
    if (derivedCues.length === 0) return null;
    const currentIndex = getCueIndex(cueId);
    if (currentIndex === -1) {
      return derivedCues[0]?.id ?? null;
    }
    if (currentIndex < derivedCues.length - 1) {
      return derivedCues[currentIndex + 1]?.id ?? null;
    }
    const firstUnanswered = derivedCues.find((cue) => cueAnswers[cue.id] === undefined);
    if (firstUnanswered && firstUnanswered.id !== cueId) {
      return firstUnanswered.id;
    }
    return null;
  };

  const getPreviousCueId = (cueId: number): number | null => {
    if (derivedCues.length === 0) return null;
    const currentIndex = getCueIndex(cueId);
    if (currentIndex > 0) {
      return derivedCues[currentIndex - 1]?.id ?? null;
    }
    return null;
  };

  const requestCustomCueHelper = async (cueId: number, observationText: string) => {
    if (!predictionResult || !uploadedBase64) {
      setUserCues((prev) =>
        prev.map((cue) =>
          cue.id === cueId ? { ...cue, helperTextStatus: "error" } : cue
        )
      );
      return;
    }

    setUserCues((prev) =>
      prev.map((cue) =>
        cue.id === cueId
          ? { ...cue, helperTextStatus: "loading", helperText: null }
          : cue
      )
    );

    try {
      const response = await fetch("/api/explain-custom-cue", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prediction_label: predictionResult.prediction,
          prediction_confidence: predictionResult.confidence,
          cue_type: "CUSTOM",
          cue_title: "Your observation",
          user_observation: observationText,
          image_base64: uploadedBase64 ?? undefined,
        }),
      });

      if (!response.ok) {
        throw new Error(`Custom cue helper request failed: ${response.status}`);
      }

      const data = await response.json();

      const helperText: CustomCueHelperText = {
        whereToLook: data.where_to_look ?? "",
        whatToCheck: data.what_to_check ?? "",
        bothWaysSupport: data.both_ways_support ?? "",
        bothWaysChallenge: data.both_ways_challenge ?? "",
        questions: Array.isArray(data.questions) ? data.questions : [],
      };

      setUserCues((prev) =>
        prev.map((cue) =>
          cue.id === cueId
            ? { ...cue, helperTextStatus: "ready", helperText }
            : cue
        )
      );
    } catch (error) {
      console.error("Error generating custom cue helper text", error);
      setUserCues((prev) =>
        prev.map((cue) =>
          cue.id === cueId ? { ...cue, helperTextStatus: "error" } : cue
        )
      );
    }
  };

  const handleAddUserCue = () => {
    const text = userCueText.trim();
    if (text.length < 12) return;

    const existingCustomCue = userCues.find((cue) => cue.cueType === "CUSTOM");
    if (
      existingCustomCue &&
      existingCustomCue.userObservationText === text
    ) {
      setSelectedCueId(existingCustomCue.id);
      setUserCueText("");
      return;
    }

    const nextId =
      existingCustomCue?.id ??
      (userCues.length > 0 ? Math.max(...userCues.map((c) => c.id)) + 1 : 10000);

    const newCue: Cue = {
      id: nextId,
      title: "Your observation",
      description: text,
      source: "user",
      cueType: "CUSTOM",
      userObservationText: text,
      helperTextStatus: "loading",
      helperText: null,
      userJudgment: null,
      isUserProvided: true,
    };

    setUserCues((prev) =>
      existingCustomCue
        ? prev.map((cue) => (cue.id === nextId ? newCue : cue))
        : [newCue, ...prev]
    );
    setUserCueText("");
    setSelectedCueId(nextId);
    requestCustomCueHelper(nextId, text);
  };

  const selectedCue =
    derivedCues.find((c) => c.id === selectedCueId) ?? derivedCues[0];

  const selectedBackendCue =
    predictionResult?.cues?.find((c) => c.id === selectedCueId) ?? null;
  const selectedCueType = getCueType(selectedBackendCue?.title);
  const selectedCustomCue =
    selectedCue?.cueType === "CUSTOM" ? selectedCue : null;

  const handlePrevious = () => {
    if (currentStep === 4) {
      const previousCueId = getPreviousCueId(selectedCueId);
      if (previousCueId !== null) {
        setSelectedCueId(previousCueId);
        return;
      }
      setCurrentStep(3);
      return;
    }
    setCurrentStep((prev) => (prev > 1 ? ((prev - 1) as StepId) : prev));
  };

  const handleNext = () => {
    // Block navigation while the backend is still analyzing
    if (currentStep === 2 && isAnalyzing) {
      return;
    }

    // On the upload step, if a file is present, start the analysis flow
    if (currentStep === 1) {
      if (uploadedFile) {
        handleStartAnalyzing();
      }
      return;
    }

    if (currentStep === 4) {
      const nextCueId = getNextCueId(selectedCueId);
      if (nextCueId !== null) {
        setSelectedCueId(nextCueId);
        return;
      }
      if (allCuesAnswered) {
        setCurrentStep(5);
        return;
      }
      alert("Please answer this cue before continuing.");
      return;
    }

    // Generic logic for later steps
    setCurrentStep((prev) =>
      prev < TOTAL_STEPS ? ((prev + 1) as StepId) : prev
    );
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // Revoke old preview URL if it exists
      if (uploadedPreviewUrl) {
        URL.revokeObjectURL(uploadedPreviewUrl);
      }

      setUploadedFileName(file.name);
      setUploadedFile(file);
      setUploadedPreviewUrl(URL.createObjectURL(file));
      fileToBase64(file).then(setUploadedBase64).catch(console.error);
      // later kun je hier analyse / API-call starten
      // setCurrentStep(2);
    }
  };

  const handleStartAnalyzing = () => {
    // Alleen doorgaan als er daadwerkelijk een bestand is gekozen
    if (!uploadedFile) {
      return;
    }

    // Markeer dat de analyse bezig is en ga naar de analyzing view
    setIsAnalyzing(true);
    setCueTextByType({
      texture: null,
      lighting: null,
      background: null,
      geometry: null,
    });
    setCueTextStatusByType({
      texture: "idle",
      lighting: "idle",
      background: "idle",
      geometry: "idle",
    });
    setCurrentStep(2 as StepId);

    // Fire-and-forget async flow
    (async () => {
      try {
        const formData = new FormData();
        formData.append("file", uploadedFile);

        const response = await fetch("/api/predict", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          console.error("Prediction request failed:", response.statusText);
          return;
        }

        const data = await response.json();
        setPredictionResult(data);
        await preloadCueTexts(data);
        console.log("Prediction result:", data);
      } catch (error) {
        console.error("Error during prediction:", error);
      } finally {
        // Analyse is klaar: flag uitzetten en doorgaan naar de cue overview (stap 3)
        setIsAnalyzing(false);
        setCurrentStep(3 as StepId);
      }
    })();
  };

  const handleStartOver = () => {
    setCurrentStep(1);
    setUploadedFileName(null);
    setUploadedFile(null);
    if (uploadedPreviewUrl) {
      URL.revokeObjectURL(uploadedPreviewUrl);
    }
    setUploadedPreviewUrl(null);
    setUploadedBase64(null);
    setSelectedCueId(1);
    setPredictionResult(null);
    setIsAnalyzing(false);
    setCueTextByType({
      texture: null,
      lighting: null,
      background: null,
      geometry: null,
    });
    setCueTextStatusByType({
      texture: "idle",
      lighting: "idle",
      background: "idle",
      geometry: "idle",
    });
    setCueAnswers({});
    setUserCues([]);
    setUserCueText("");
  };

  const preloadCueTexts = async (result: PredictionResult) => {
    if (!result.cues || result.cues.length === 0) {
      return;
    }

    const modelId = result.model || "unknown";
    const cuePayloads = CUE_TYPES.map((cueType) => {
      const cue = result.cues.find((item) => item.title.toLowerCase() === cueType);
      return {
        cue_type: cueType,
        cue_title: cue?.title ?? cueType,
        cue_score: cue?.score ?? 0,
        heatmap_base64: result.cue_heatmaps?.[cueType] || result.heatmap || "",
      };
    });

    const cachedCueText: Record<CueType, string | null> = {
      texture: null,
      lighting: null,
      background: null,
      geometry: null,
    };
    const nextStatus: Record<CueType, "idle" | "loading" | "ready" | "error"> = {
      texture: "idle",
      lighting: "idle",
      background: "idle",
      geometry: "idle",
    };
    let needsFetch = false;

    cuePayloads.forEach((cue) => {
      const cueType = cue.cue_type as CueType;
      const cacheKey = buildCueCacheKey(
        cueType,
        cue.heatmap_base64,
        uploadedBase64,
        modelId
      );
      const cached = cueTextCacheRef.current.get(cacheKey) ?? null;
      cachedCueText[cueType] = cached;
      if (cached) {
        nextStatus[cueType] = "ready";
      } else {
        nextStatus[cueType] = "loading";
        needsFetch = true;
      }
    });

    setCueTextByType(cachedCueText);
    setCueTextStatusByType(nextStatus);

    if (!needsFetch) {
      return;
    }

    try {
      const response = await fetch("/api/explain-cues-batch", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prediction_label: result.prediction,
          prediction_confidence: result.confidence,
          image_context: `The model predicted ${result.prediction.toUpperCase()} with ${(
            result.confidence * 100
          ).toFixed(1)}% confidence for this image.`,
          image_base64: uploadedBase64 ?? undefined,
          cues: cuePayloads,
        }),
      });

      if (!response.ok) {
        throw new Error(`Batch cue explanation request failed: ${response.status}`);
      }

      const data = await response.json();
      const cueTextMap = data?.cue_text_by_type ?? data ?? {};

      if (!ensureCueTextMap(cueTextMap)) {
        throw new Error("Cue text response missing expected cue types.");
      }

      const nextCueText: Record<CueType, string | null> = {
        texture: cueTextMap.texture,
        lighting: cueTextMap.lighting,
        background: cueTextMap.background,
        geometry: cueTextMap.geometry,
      };

      const readyStatus: Record<CueType, "idle" | "loading" | "ready" | "error"> = {
        texture: "ready",
        lighting: "ready",
        background: "ready",
        geometry: "ready",
      };

      cuePayloads.forEach((cue) => {
        const cueType = cue.cue_type as CueType;
        const cacheKey = buildCueCacheKey(
          cueType,
          cue.heatmap_base64,
          uploadedBase64,
          modelId
        );
        const cueText = nextCueText[cueType];
        if (cueText) {
          cueTextCacheRef.current.set(cacheKey, cueText);
        } else {
          readyStatus[cueType] = "error";
        }
      });

      setCueTextByType(nextCueText);
      setCueTextStatusByType(readyStatus);
    } catch (error) {
      console.error("Error preloading cue explanations", error);
      setCueTextStatusByType({
        texture: "error",
        lighting: "error",
        background: "error",
        geometry: "error",
      });
    }
  };

  return (
    <section className="journey-section">
      <div className="journey-container">
        {/* Titel + subtitel */}
        <div className="journey-heading">
          <h2 className="journey-title">Your TrueSight Journey</h2>
          <p className="journey-subtitle">{getStepSubtitle(currentStep)}</p>
        </div>

        {/* Hoofdkaart */}
        <div className="journey-card">
          {/* Header met agent + stepper controls */}
          <header className="journey-card-header">
            <div className="journey-agent">
              <div className="journey-agent-avatar">TS</div>
              <div className="journey-agent-text">
                <div className="journey-agent-name">TrueSight Agent</div>
                <div className="journey-agent-helper">
                  Here to help you decide what&apos;s real.
                </div>
              </div>
            </div>

            <div className="journey-step-controls">
              <span className="journey-step-indicator">
                Step {currentStep} of {TOTAL_STEPS}
              </span>
              <div className="journey-step-buttons">
                <button
                  type="button"
                  className="journey-step-button journey-step-button-prev"
                  onClick={handlePrevious}
                  disabled={currentStep === 1}
                >
                  ‹
                </button>
                <button
                  type="button"
                  className="journey-step-button journey-step-button-next"
                  onClick={handleNext}
                  disabled={
                    (currentStep === 1 && !uploadedFileName) ||
                    (currentStep === 2 && isAnalyzing) ||
                    currentStep === TOTAL_STEPS
                  }
                >
                  ›
                </button>
              </div>
            </div>
          </header>

          {/* Content die wisselt per stap */}
          <div className="journey-card-body">
            {currentStep === 1 && (
              <StepUpload
                fileName={uploadedFileName}
                onFileChange={handleFileChange}
                onStartAnalyzing={handleStartAnalyzing}
              />
            )}

            {currentStep === 2 && <StepAnalyzing />}

            {currentStep === 3 && (
              <div>
                <StepOverview
                  selectedCueId={selectedCueId}
                  onSelectCue={setSelectedCueId}
                  predictionResult={predictionResult}
                />

                <div style={{ marginTop: "1.25rem" }}>
                  <div style={{ fontWeight: 600, marginBottom: "0.5rem" }}>
                    Add your own observation
                  </div>
                  <div style={{ fontSize: "0.9rem", color: "#6b7280", marginBottom: "0.75rem" }}>
                    If you notice something suspicious that isn't covered by the model cues, write it here.
                  </div>

                  <div style={{ display: "flex", gap: "0.75rem", alignItems: "flex-start" }}>
                    <textarea
                      value={userCueText}
                      onChange={(e) => setUserCueText(e.target.value)}
                      placeholder="Example: the hands look unnatural and the shadows don't match the light source..."
                      rows={3}
                      style={{
                        flex: 1,
                        resize: "vertical",
                        padding: "0.75rem",
                        borderRadius: "10px",
                        border: "1px solid #e5e7eb",
                        fontSize: "0.95rem",
                      }}
                    />
                    <button
                      type="button"
                      className="journey-footer-button journey-footer-button-primary"
                      onClick={handleAddUserCue}
                      disabled={userCueText.trim().length < 12}
                      style={{ whiteSpace: "nowrap" }}
                    >
                      Add cue
                    </button>
                  </div>

                  {userCues.length > 0 && (
                    <div style={{ marginTop: "1rem" }}>
                      <div style={{ fontWeight: 600, marginBottom: "0.5rem" }}>
                        Your cues
                      </div>
                      <div style={{ display: "grid", gap: "0.5rem" }}>
                        {userCues.map((cue) => (
                          <button
                            key={cue.id}
                            type="button"
                            onClick={() => setSelectedCueId(cue.id)}
                            style={{
                              textAlign: "left",
                              padding: "0.75rem",
                              borderRadius: "10px",
                              border: selectedCueId === cue.id ? "1px solid #4338ca" : "1px solid #e5e7eb",
                              background: selectedCueId === cue.id ? "rgba(67, 56, 202, 0.06)" : "#ffffff",
                              cursor: "pointer",
                            }}
                          >
                            <div style={{ fontWeight: 600, marginBottom: "0.25rem" }}>
                              Your observation
                            </div>
                            <div style={{ fontSize: "0.9rem", color: "#4b5563" }}>
                              {cue.description}
                            </div>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {currentStep === 4 && (
              <StepCue
                title={selectedCue.title}
                subtitle={selectedCue.description}
                imageSrc={
                  predictionResult?.cue_heatmaps?.[
                    selectedCue.title.toLowerCase()
                  ] || predictionResult?.heatmap || "/heatmap-picture.png"
                }
                originalImageSrc={uploadedPreviewUrl || undefined}
                image_base64={uploadedBase64 || undefined}
                cueText={selectedCueType ? cueTextByType[selectedCueType] : null}
                cueTextStatus={
                  selectedCueType ? cueTextStatusByType[selectedCueType] : "idle"
                }
                helperText={selectedCustomCue?.helperText ?? null}
                helperTextStatus={selectedCustomCue?.helperTextStatus ?? "idle"}
                onRetryHelperText={
                  selectedCustomCue
                    ? () =>
                        requestCustomCueHelper(
                          selectedCustomCue.id,
                          selectedCustomCue.userObservationText ||
                            selectedCustomCue.description
                        )
                    : undefined
                }
                cueType={selectedBackendCue?.title.toLowerCase()}
                predictionLabel={predictionResult?.prediction}
                predictionConfidence={predictionResult?.confidence}
                cueScore={selectedBackendCue?.score}
                cueSource={selectedCue.source}
                userObservationText={selectedCue.source === "user" ? selectedCue.description : undefined}
                imageContext={
                  predictionResult
                    ? `The model predicted ${predictionResult.prediction.toUpperCase()} with ${(predictionResult.confidence * 100).toFixed(
                        1
                      )}% confidence for this image.`
                    : undefined
                }
                cueId={selectedCue.id}
                existingAnswer={cueAnswers[selectedCue.id]}
                onDecisionChange={(decision) => handleCueAnswer(selectedCue.id, decision)}
              />
            )}

            {currentStep === 5 && (
              <StepConclusion
                modelLabel={modelLabel}
                modelProbability={modelProbability}
                userLabel={userVerdictLabel}
                aiDecision={aiDecision}
                humanDecision={humanDecision}
                finalConclusion={finalConclusion}
                keyCues={keyCues}
                cueRecapItems={derivedCues.map((cue) => {
                  const isCustom = cue.cueType === "CUSTOM";
                  const overlaySrc = isCustom
                    ? undefined
                    : predictionResult?.cue_heatmaps?.[
                        cue.title.toLowerCase()
                      ] || predictionResult?.heatmap || undefined;
                  const score =
                    predictionResult?.cues?.find((c) => c.id === cue.id)?.score;

                  return {
                    id: cue.id,
                    title: cue.title,
                    originalSrc: uploadedPreviewUrl || undefined,
                    overlaySrc,
                    modelScore: score,
                    userJudgement: cueAnswers[cue.id],
                    isCustom,
                    observationText: isCustom
                      ? cue.userObservationText || cue.description
                      : undefined,
                  };
                })}
                impactLevel={undefined}
                onFinish={handleStartOver}
                onTryAnother={handleStartOver}
              />
            )}
          </div>

          {/* Footer / navigatie onderin */}
          <footer className="journey-card-footer">
            <button
              type="button"
              className="journey-footer-button journey-footer-button-secondary"
              onClick={handlePrevious}
              disabled={currentStep === 1}
            >
              Previous
            </button>

            <div className="journey-footer-actions">
              {currentStep === TOTAL_STEPS && (
                <button
                  type="button"
                  className="journey-footer-button journey-footer-button-secondary"
                  onClick={handleStartOver}
                >
                  Start over
                </button>
              )}

              <button
                type="button"
                className="journey-footer-button journey-footer-button-primary"
                onClick={
                  currentStep === TOTAL_STEPS ? handleStartOver : handleNext
                }
                disabled={
                  (currentStep === 1 && !uploadedFileName) ||
                  (currentStep === 2 && isAnalyzing) ||
                  currentStep === TOTAL_STEPS
                }
              >
                {currentStep === TOTAL_STEPS
                  ? "Analyze another image"
                  : "Next step"}
              </button>
            </div>
          </footer>
        </div>
      </div>
    </section>
  );
};

/* ---------- STEP 1: UPLOAD VIEW ---------- */

interface StepUploadProps {
  fileName: string | null;
  onFileChange: (event: ChangeEvent<HTMLInputElement>) => void;
  onStartAnalyzing: () => void;
}

const StepUpload: React.FC<StepUploadProps> = ({
  fileName,
  onFileChange,
  onStartAnalyzing,
}) => {
  return (
    <div className="journey-step journey-step-upload">
      <div className="journey-agent-bubble">
        <p>
          Upload the image that you want to find out if it is AI or not. Your
          image is processed securely and never stored.
        </p>
      </div>

      <div className="journey-upload-wrapper">
        <label className="journey-upload-dropzone">
          <div className="journey-upload-icon">⬆️</div>
          <div className="journey-upload-title">Upload an image to start</div>
          <p className="journey-upload-description">
            Drag &amp; drop an image here or click the button below.
          </p>
          <button
            type="button"
            className="journey-upload-button"
            onClick={(event) => {
              event.preventDefault();
              onStartAnalyzing();
            }}
          >
            Upload image
          </button>
          <input
            type="file"
            accept="image/*"
            className="journey-upload-input"
            onChange={onFileChange}
          />
        </label>

        {fileName && (
          <p className="journey-upload-filename">
            Selected file: <span>{fileName}</span>
          </p>
        )}
      </div>
    </div>
  );
};

export default JourneySection;
