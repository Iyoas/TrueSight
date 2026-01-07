import assert from "node:assert/strict";
import { test } from "node:test";

import {
  getDecisionLabel,
  getJointConclusionTitle,
  getJointOutcomeCallout,
  getPartialAgreementSubtitle,
} from "./jointOutcomeUi";

test("getDecisionLabel", () => {
  assert.equal(getDecisionLabel("AI_GENERATED"), "AI-generated");
  assert.equal(getDecisionLabel("HUMAN_GENERATED"), "Human-generated");
  assert.equal(getDecisionLabel("UNCERTAIN"), "Uncertain");
});

test("getJointConclusionTitle", () => {
  assert.equal(
    getJointConclusionTitle({ kind: "AGREEMENT", label: "AI_GENERATED", leaning: "AI" }),
    "Agreement - AI-generated"
  );
  assert.equal(
    getJointConclusionTitle({ kind: "AGREEMENT", label: "HUMAN_GENERATED", leaning: "HUMAN" }),
    "Agreement - Human-generated"
  );
  assert.equal(
    getJointConclusionTitle({ kind: "PARTIAL_AGREEMENT", label: "AI_GENERATED", leaning: "AI" }),
    "Partial agreement"
  );
  assert.equal(
    getJointConclusionTitle({ kind: "DISAGREEMENT", label: null }),
    "Disagreement"
  );
});

test("getPartialAgreementSubtitle", () => {
  assert.equal(
    getPartialAgreementSubtitle("AI_GENERATED", "UNCERTAIN"),
    "The model classifies this image as AI-generated, while your assessment is uncertain."
  );
  assert.equal(
    getPartialAgreementSubtitle("HUMAN_GENERATED", "UNCERTAIN"),
    "The model classifies this image as human-generated, while your assessment is uncertain."
  );
  assert.equal(getPartialAgreementSubtitle("AI_GENERATED", "AI_GENERATED"), null);
});

test("getJointOutcomeCallout", () => {
  assert.deepEqual(
    getJointOutcomeCallout({ kind: "AGREEMENT", label: "AI_GENERATED", leaning: "AI" }),
    {
      tone: "agree",
      icon: "✅",
      text: "You and the model reached the same conclusion. This can increase trust, but both human and AI can still be wrong.",
    }
  );
  assert.deepEqual(
    getJointOutcomeCallout({ kind: "PARTIAL_AGREEMENT", label: "AI_GENERATED", leaning: "AI" }),
    {
      tone: "warn",
      icon: "⚠️",
      text: "You and the model did not fully agree. The model indicates a likely origin, while your assessment is uncertain. Treat this case with caution.",
    }
  );
  assert.deepEqual(
    getJointOutcomeCallout({ kind: "DISAGREEMENT", label: null }),
    {
      tone: "warn",
      icon: "⚠️",
      text: "You and the model reached different conclusions. Treat this case as high-risk and consider additional verification.",
    }
  );
});
