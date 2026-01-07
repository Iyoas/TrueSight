import type { AIDecision, FinalConclusion, HumanDecision } from "./decisionLogic";

export const getDecisionLabel = (decision: AIDecision | HumanDecision): string => {
  if (decision === "AI_GENERATED") return "AI-generated";
  if (decision === "HUMAN_GENERATED") return "Human-generated";
  return "Uncertain";
};

export const getJointConclusionTitle = (
  finalConclusion?: FinalConclusion
): string | null => {
  if (!finalConclusion) return null;
  if (finalConclusion.kind === "AGREEMENT" && finalConclusion.label) {
    return `Agreement - ${getDecisionLabel(finalConclusion.label)}`;
  }
  if (finalConclusion.kind === "PARTIAL_AGREEMENT") {
    return "Partial agreement";
  }
  if (finalConclusion.kind === "DISAGREEMENT") {
    return "Disagreement";
  }
  return null;
};

export const getPartialAgreementSubtitle = (
  aiDecision: AIDecision,
  humanDecision: HumanDecision
): string | null => {
  if (humanDecision !== "UNCERTAIN") return null;
  if (aiDecision === "AI_GENERATED") {
    return "The model classifies this image as AI-generated, while your assessment is uncertain.";
  }
  return "The model classifies this image as human-generated, while your assessment is uncertain.";
};

export const getJointOutcomeCallout = (
  finalConclusion?: FinalConclusion
): { tone: "agree" | "warn"; icon: string; text: string } | null => {
  if (!finalConclusion) return null;
  if (finalConclusion.kind === "AGREEMENT") {
    return {
      tone: "agree",
      icon: "✅",
      text: "You and the model reached the same conclusion. This can increase trust, but both human and AI can still be wrong.",
    };
  }
  if (finalConclusion.kind === "PARTIAL_AGREEMENT") {
    return {
      tone: "warn",
      icon: "⚠️",
      text: "You and the model did not fully agree. The model indicates a likely origin, while your assessment is uncertain. Treat this case with caution.",
    };
  }
  return {
    tone: "warn",
    icon: "⚠️",
    text: "You and the model reached different conclusions. Treat this case as high-risk and consider additional verification.",
  };
};
