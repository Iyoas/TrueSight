export type CanonicalCueValue = "AI" | "HUMAN" | "UNCERTAIN";
export type HumanDecision = "AI_GENERATED" | "HUMAN_GENERATED" | "UNCERTAIN";
export type AIDecision = "AI_GENERATED" | "HUMAN_GENERATED";
export type FinalConclusionKind = "AGREEMENT" | "PARTIAL_AGREEMENT" | "DISAGREEMENT";

export type FinalConclusion = {
  kind: FinalConclusionKind;
  label: AIDecision | null;
  leaning?: "AI" | "HUMAN";
};

export const computeHumanDecision = (cues: CanonicalCueValue[]): HumanDecision => {
  const aiCount = cues.filter((cue) => cue === "AI").length;
  const humanCount = cues.filter((cue) => cue === "HUMAN").length;

  if (aiCount >= 3) {
    return "AI_GENERATED";
  }
  if (humanCount >= 3) {
    return "HUMAN_GENERATED";
  }
  return "UNCERTAIN";
};

export const computeFinalConclusion = (
  aiDecision: AIDecision,
  humanDecision: HumanDecision
): FinalConclusion => {
  if (humanDecision !== "UNCERTAIN" && aiDecision === humanDecision) {
    return {
      kind: "AGREEMENT",
      label: aiDecision,
      leaning: aiDecision === "AI_GENERATED" ? "AI" : "HUMAN",
    };
  }

  if (humanDecision === "UNCERTAIN") {
    return {
      kind: "PARTIAL_AGREEMENT",
      label: aiDecision,
      leaning: aiDecision === "AI_GENERATED" ? "AI" : "HUMAN",
    };
  }

  return {
    kind: "DISAGREEMENT",
    label: null,
  };
};
