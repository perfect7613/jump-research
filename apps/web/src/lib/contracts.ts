import { createHash } from "node:crypto";
import { z } from "zod";

export const QUESTION_VERSION = "jump.thought-experiment-question/v2";
export const SPEC_RESPONSE_VERSION = "jump.thought-experiment-spec-response/v2";
export const CONFIRMATION_VERSION = "jump.thought-experiment-confirmation/v2";
export const RUN_RESPONSE_VERSION = "jump.thought-experiment-run-response/v2";
export const SPEC_VERSION = "jump.thought-experiment-spec/v2";
export const RUN_VERSION = "jump.thought-experiment-run/v2";
export const ENGINE_ID = "jump.declarative-visual-engine/v2";
export const SPEC_SCHEMA_SHA256 = "fa7674dc3c5f759dc74ff723cef7a194edc4186069496e631e65b4d0ebd84ab5";
export const RUN_SCHEMA_SHA256 = "55d1fd3fdef215abfb1a148080cc01aea3fff118ba1e779e02e6841f43941166";
export const CODE_VERSION = "a484c59cad54e78772a36291c2a6c7b3e9eff51a";

const id = z.string().regex(/^[a-z][a-z0-9_-]{0,63}$/);
const sha = z.string().regex(/^[0-9a-f]{64}$/);
const boundedNumber = z.number().finite().min(-1_000_000).max(1_000_000);
const model = z.object({
  repo_id: z.string().min(1), revision: z.string().min(1), transformers_revision: z.string().min(1),
  frozen: z.literal(true), adapter_id: z.null(),
}).strict();

const point = z.object({
  entity_id: id, type_id: id, x: boundedNumber, y: boundedNumber,
  shape: z.enum(["circle", "square", "triangle"]), color: z.string().regex(/^#[0-9a-fA-F]{6}$/),
  size: z.number().positive().max(20), category: z.string().max(40).nullable(),
}).strict();

const conditionSpec = z.object({
  id, label: z.string().min(1).max(80), kind: z.enum(["baseline", "counterfactual"]),
  interventions: z.array(z.object({
    time: z.number().int().min(0).max(500),
    operation: z.enum(["set_rule_parameter", "scale_rule_parameter", "set_numeric_state", "set_categorical_state"]),
    target: id, field: id, value: z.union([z.string(), z.number(), z.boolean()]),
  }).strict()).max(8),
}).strict();

export const experimentSpecSchema = z.object({
  schema_version: z.literal(SPEC_VERSION), spec_id: z.string().regex(/^spec-[0-9a-f]{24}$/),
  intent: z.string().min(1).max(2000), question: z.string().min(1).max(500), hypothesis: z.string().min(1).max(500),
  world: z.object({
    bounds: z.object({ width: z.number().positive().max(1000), height: z.number().positive().max(1000), boundary: z.enum(["wrap", "reflect", "clamp"]) }).strict(),
    entities: z.array(z.object({
      id, label: z.string().min(1).max(80), count: z.number().int().min(1).max(32),
      appearance: z.object({ shape: z.enum(["circle", "square", "triangle"]), color: z.string().regex(/^#[0-9a-fA-F]{6}$/), size: z.number().positive().max(20) }).strict(),
      initial_state: z.object({ numeric: z.record(id, boundedNumber), categorical: z.record(id, z.string().min(1).max(40)) }).strict(),
      initial_layout: z.object({ kind: z.enum(["uniform", "grid", "ring", "line"]), center: z.tuple([boundedNumber, boundedNumber]), spread: z.number().min(0).max(1000) }).strict(),
    }).strict()).min(1).max(8),
    graph: z.object({ kind: z.enum(["none", "ring", "grid", "random"]), edge_probability: z.number().min(0).max(1), directed: z.boolean() }).strict(),
  }).strict(),
  dynamics: z.object({ rules: z.array(z.object({
    id, op: z.enum(["move_2d", "random_walk_2d", "pairwise_force_2d", "graph_diffusion", "graph_contagion", "predator_prey_2d", "lane_traffic_2d", "queue_agents_2d"]),
    target_type: id.nullable(), parameters: z.record(id, z.union([z.string(), z.number(), z.boolean(), z.null()])),
  }).strict()).min(1).max(12) }).strict(),
  conditions: z.array(conditionSpec).min(2).max(4),
  schedule: z.object({ duration_steps: z.number().int().min(2).max(500), dt: z.number().positive().max(10), seed: z.number().int().min(0).max(2147483647), repetitions: z.number().int().min(1).max(20) }).strict(),
  measurements: z.array(z.object({ id, label: z.string().min(1).max(100), op: z.enum(["mean_state", "sum_state", "variance_state", "count_category", "population_count"]), entity_type: id.nullable(), state: id.nullable(), category: z.string().min(1).max(40).nullable() }).strict()).min(1).max(12),
  visualization: z.object({ kind: z.literal("animated_2d"), frame_stride: z.number().int().min(1).max(100), max_frames: z.number().int().min(2).max(40), chart_measurement_ids: z.array(id).min(1).max(12) }).strict(),
  spec_sha256: sha,
}).strict();

const conditionResult = z.object({
  condition_id: id,
  frames: z.array(z.object({ step: z.number().int().min(0).max(500), points: z.array(point).max(256) }).strict()).min(2).max(40),
  series: z.array(z.object({ measurement_id: id, values: z.array(z.object({ step: z.number().int(), value: boundedNumber }).strict()).min(2).max(501) }).strict()).min(1).max(12),
  summary: z.record(id, boundedNumber),
}).strict();

export const thoughtRunSchema = z.object({
  schema_version: z.literal(RUN_VERSION), run_id: z.string().regex(/^visual-run-[0-9a-f]{24}$/),
  spec_id: z.string().regex(/^spec-[0-9a-f]{24}$/), spec_sha256: sha, status: z.enum(["completed", "failed"]),
  execution: z.object({
    engine_id: z.literal(ENGINE_ID), code_version: z.string().min(1).max(80), modal_call_id: z.string().min(1).max(128),
    prediction: z.object({ summary: z.string().min(1).max(500), expected_direction: z.enum(["increase", "decrease", "change", "no_change"]), measurement_id: id }).strict(),
    prediction_recorded_at: z.iso.datetime({ offset: true }), started_at: z.iso.datetime({ offset: true }), completed_at: z.iso.datetime({ offset: true }), error: z.string().max(500).nullable(),
  }).strict(),
  conditions: z.array(conditionResult).min(2).max(4),
  comparisons: z.array(z.object({ measurement_id: id, baseline_condition_id: id, counterfactual_condition_id: id, baseline_final: boundedNumber, counterfactual_final: boundedNumber, difference: boundedNumber }).strict()).min(1).max(36),
  revision: z.object({ disposition: z.enum(["retain", "revise", "reject"]), interpretation: z.string().min(1).max(500) }).strict(),
  evidence: z.object({ spec_sha256: sha, engine_id: z.literal(ENGINE_ID), code_version: z.string().min(1).max(80), modal_call_id: z.string().min(1).max(128), result_sha256: sha, sealed_payload_sha256: sha }).strict(),
  run_sha256: sha,
}).strict();

export const questionSchema = z.object({
  schema_version: z.literal(QUESTION_VERSION), request_id: z.string().min(1).max(128), session_id: z.string().min(1).max(128),
  intent: z.string().trim().min(1).max(2000), seed: z.number().int().min(0).max(2147483647), repetitions: z.number().int().min(1).max(20),
}).strict().superRefine((value, ctx) => {
  if (/(?:https?:\/\/|www\.|file:\/\/|(?:^|\s)[~/][^\s]+|```|\bimport\s+|\bdef\s+|clinical trial|download|upload)/i.test(value.intent)) {
    ctx.addIssue({ code: "custom", message: "Only bounded simulations are supported; URLs, files, code, and real-world actions are rejected." });
  }
});

export const confirmationSchema = z.object({
  schema_version: z.literal(CONFIRMATION_VERSION), request_id: z.string().min(1).max(128), session_id: z.string().min(1).max(128),
  spec_id: z.string().regex(/^spec-[0-9a-f]{24}$/), spec_sha256: sha, confirmation_token: z.string().regex(/^[0-9a-f]{64}$/), confirmed: z.literal(true),
}).strict();

export type ExperimentSpec = z.infer<typeof experimentSpecSchema>;
export type ThoughtRun = z.infer<typeof thoughtRunSchema>;
export type Question = z.infer<typeof questionSchema>;
export type Confirmation = z.infer<typeof confirmationSchema>;

export function validateSpecResponse(input: unknown, request: Question) {
  const parsed = z.object({ schema_version: z.literal(SPEC_RESPONSE_VERSION), status: z.literal("awaiting_confirmation"), request_id: z.string(), session_id: z.string(), spec: experimentSpecSchema, model, confirmation: z.object({ schema_version: z.literal(CONFIRMATION_VERSION), request_id: z.string(), session_id: z.string(), spec_id: z.string(), spec_sha256: sha, confirmation_token: z.string().regex(/^[0-9a-f]{64}$/), confirmed: z.literal(false) }).strict() }).strict().parse(input);
  validateSpec(parsed.spec);
  if (parsed.request_id !== request.request_id || parsed.session_id !== request.session_id) throw new Error("Spec response does not bind the request");
  const c = parsed.confirmation;
  if (c.request_id !== request.request_id || c.session_id !== request.session_id || c.spec_id !== parsed.spec.spec_id || c.spec_sha256 !== parsed.spec.spec_sha256) throw new Error("Confirmation does not bind the exact spec and session");
  return parsed;
}

export function validateRunResponse(input: unknown, confirmation: Confirmation) {
  const parsed = z.object({ schema_version: z.literal(RUN_RESPONSE_VERSION), status: z.literal("completed"), request_id: z.string(), session_id: z.string(), spec: experimentSpecSchema, run: thoughtRunSchema, model }).strict().parse(input);
  validateSpec(parsed.spec); validateRun(parsed.run, parsed.spec);
  if (parsed.spec.spec_sha256 !== confirmation.spec_sha256 || parsed.spec.spec_id !== confirmation.spec_id || parsed.request_id !== confirmation.request_id || parsed.session_id !== confirmation.session_id) throw new Error("Run changed the confirmed spec or session");
  return parsed;
}

export function validateSpec(spec: ExperimentSpec) {
  const digest = contentHash(spec, ["spec_id", "spec_sha256"]);
  if (spec.spec_sha256 !== digest || spec.spec_id !== `spec-${digest.slice(0, 24)}`) throw new Error("Spec identity hash mismatch");
  const baseline = spec.conditions.filter((c) => c.kind === "baseline");
  if (baseline.length !== 1 || baseline[0].interventions.length !== 0 || !spec.conditions.some((c) => c.kind === "counterfactual" && c.interventions.length > 0)) throw new Error("Spec conditions are not a baseline/counterfactual pair");
}

export function validateRun(run: ThoughtRun, spec: ExperimentSpec) {
  const digest = contentHash(run, ["run_id", "run_sha256"]);
  if (run.run_sha256 !== digest || run.run_id !== `visual-run-${digest.slice(0, 24)}`) throw new Error("Run identity hash mismatch");
  if (run.spec_id !== spec.spec_id || run.spec_sha256 !== spec.spec_sha256) throw new Error("Run does not bind the confirmed spec");
  if (run.execution.code_version !== CODE_VERSION || run.evidence.code_version !== CODE_VERSION) throw new Error("Run code pin mismatch");
  if (run.evidence.spec_sha256 !== spec.spec_sha256 || run.evidence.engine_id !== ENGINE_ID || run.evidence.modal_call_id !== run.execution.modal_call_id) throw new Error("Run evidence does not bind execution");
  if (Date.parse(run.execution.prediction_recorded_at) >= Date.parse(run.execution.started_at)) throw new Error("Prediction was not recorded before execution");
  const resultHash = sha256(canonicalJson({ conditions: run.conditions, comparisons: run.comparisons }));
  if (run.evidence.result_sha256 !== resultHash) throw new Error("Visual result hash mismatch");
  const payload = Object.fromEntries(["spec_id", "spec_sha256", "status", "execution", "conditions", "comparisons", "revision"].map((key) => [key, run[key as keyof ThoughtRun]]));
  if (run.evidence.sealed_payload_sha256 !== sha256(canonicalJson(payload))) throw new Error("Sealed payload hash mismatch");
}

function contentHash(value: Record<string, unknown>, excluded: string[]) {
  return sha256(canonicalJson(Object.fromEntries(Object.entries(value).filter(([key]) => !excluded.includes(key)))));
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.entries(value as Record<string, unknown>).sort(([a], [b]) => a.localeCompare(b)).map(([key, item]) => `${JSON.stringify(key)}:${canonicalJson(item)}`).join(",")}}`;
  return JSON.stringify(value);
}

function sha256(value: string) { return createHash("sha256").update(value, "utf8").digest("hex"); }
