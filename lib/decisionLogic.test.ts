import assert from "node:assert/strict";
import { test } from "node:test";

import { computeHumanDecision, computeFinalConclusion } from "./decisionLogic";

test("computeHumanDecision - AI majority", () => {
  assert.equal(
    computeHumanDecision(["AI", "AI", "AI", "HUMAN"]),
    "AI_GENERATED"
  );
  assert.equal(
    computeHumanDecision(["AI", "AI", "AI", "UNCERTAIN"]),
    "AI_GENERATED"
  );
  assert.equal(
    computeHumanDecision(["AI", "AI", "AI", "AI"]),
    "AI_GENERATED"
  );
});

test("computeHumanDecision - HUMAN majority", () => {
  assert.equal(
    computeHumanDecision(["HUMAN", "HUMAN", "HUMAN", "AI"]),
    "HUMAN_GENERATED"
  );
  assert.equal(
    computeHumanDecision(["HUMAN", "HUMAN", "HUMAN", "UNCERTAIN"]),
    "HUMAN_GENERATED"
  );
  assert.equal(
    computeHumanDecision(["HUMAN", "HUMAN", "HUMAN", "HUMAN"]),
    "HUMAN_GENERATED"
  );
});

test("computeHumanDecision - uncertain mixes", () => {
  assert.equal(
    computeHumanDecision(["AI", "AI", "HUMAN", "HUMAN"]),
    "UNCERTAIN"
  );
  assert.equal(
    computeHumanDecision(["AI", "HUMAN", "UNCERTAIN", "UNCERTAIN"]),
    "UNCERTAIN"
  );
  assert.equal(
    computeHumanDecision(["UNCERTAIN", "UNCERTAIN", "UNCERTAIN", "UNCERTAIN"]),
    "UNCERTAIN"
  );
});

test("computeFinalConclusion - agreement", () => {
  assert.deepEqual(
    computeFinalConclusion("AI_GENERATED", "AI_GENERATED"),
    { kind: "AGREEMENT", label: "AI_GENERATED", leaning: "AI" }
  );
  assert.deepEqual(
    computeFinalConclusion("HUMAN_GENERATED", "HUMAN_GENERATED"),
    { kind: "AGREEMENT", label: "HUMAN_GENERATED", leaning: "HUMAN" }
  );
});

test("computeFinalConclusion - partial agreement", () => {
  assert.deepEqual(
    computeFinalConclusion("AI_GENERATED", "UNCERTAIN"),
    { kind: "PARTIAL_AGREEMENT", label: "AI_GENERATED", leaning: "AI" }
  );
  assert.deepEqual(
    computeFinalConclusion("HUMAN_GENERATED", "UNCERTAIN"),
    { kind: "PARTIAL_AGREEMENT", label: "HUMAN_GENERATED", leaning: "HUMAN" }
  );
});

test("computeFinalConclusion - disagreement", () => {
  assert.deepEqual(
    computeFinalConclusion("AI_GENERATED", "HUMAN_GENERATED"),
    { kind: "DISAGREEMENT", label: null }
  );
  assert.deepEqual(
    computeFinalConclusion("HUMAN_GENERATED", "AI_GENERATED"),
    { kind: "DISAGREEMENT", label: null }
  );
});
