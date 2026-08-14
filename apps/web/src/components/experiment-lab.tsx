"use client";

import { useEffect, useMemo, useState } from "react";
import { ArrowRight, Check, Pause, Play, RotateCcw, ShieldCheck } from "lucide-react";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import type { Confirmation, ExperimentSpec, ThoughtRun } from "@/lib/contracts";
import { ComparisonChart, SimulationCanvas } from "@/components/simulation-canvas";

type SpecResponse = { request_id: string; session_id: string; spec: ExperimentSpec; confirmation: Omit<Confirmation, "confirmed"> & { confirmed: false } };
type RunResponse = { spec: ExperimentSpec; run: ThoughtRun };
type Stage = "question" | "planning" | "confirm" | "running" | "result" | "error";

const examples = [
  "What if twelve particles start with x velocity 0 and y velocity 0, then repulsion begins halfway through?",
  "How does an infection spread on a ring network if transmission is stopped halfway through?",
  "What happens to predator and prey populations after available food drops?",
];

async function streamRequest<T>(url: string, body: unknown, onProgress: (message: string) => void): Promise<T> {
  const response = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!response.ok || !response.body) { const error = await response.json().catch(() => ({})); throw new Error(error.detail || "Request failed closed"); }
  const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = "";
  while (true) {
    const { done, value } = await reader.read(); buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n"); buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line) continue; const event = JSON.parse(line);
      if (event.type === "progress") onProgress(event.message);
      if (event.type === "error") throw new Error(event.detail);
      if (event.type === "result") return event.payload as T;
    }
    if (done) break;
  }
  throw new Error("The experiment ended without a validated result");
}

export function ExperimentLab() {
  const [question, setQuestion] = useState(examples[0]);
  const [stage, setStage] = useState<Stage>("question");
  const [progress, setProgress] = useState("Ready");
  const [error, setError] = useState("");
  const [planned, setPlanned] = useState<SpecResponse | null>(null);
  const [completed, setCompleted] = useState<RunResponse | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(true);

  async function buildPlan() {
    setStage("planning"); setError(""); setPlanned(null); setCompleted(null); setProgress("Checking the question");
    try {
      const payload = await streamRequest<SpecResponse>("/api/thought-experiments/spec", {
        schema_version: "jump.thought-experiment-question/v2", request_id: `req-${crypto.randomUUID().replaceAll("-", "").slice(0, 20)}`,
        session_id: `web-${crypto.randomUUID().replaceAll("-", "").slice(0, 20)}`, intent: question, seed: 7613, repetitions: 2,
      }, setProgress);
      setPlanned(payload); setStage("confirm"); setProgress("Plan ready for review");
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Plan rejected"); setStage("error"); }
  }

  async function runExperiment() {
    if (!planned) return;
    setStage("running"); setError(""); setProgress("Recording the prediction before execution");
    try {
      const confirmation = { ...planned.confirmation, confirmed: true };
      const payload = await streamRequest<RunResponse>("/api/thought-experiments/confirm", confirmation, setProgress);
      setCompleted(payload); setStage("result"); setProgress("Validated result ready"); setFrameIndex(0); setPlaying(true);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Run rejected"); setStage("error"); }
  }

  const maxFrames = completed ? Math.max(...completed.run.conditions.slice(0, 2).map((item) => item.frames.length)) : 1;
  useEffect(() => {
    if (!playing || stage !== "result") return;
    const timer = window.setInterval(() => setFrameIndex((index) => (index + 1) % maxFrames), 650);
    return () => window.clearInterval(timer);
  }, [playing, stage, maxFrames]);

  const conditionLabels = useMemo(() => Object.fromEntries((completed?.spec.conditions || []).map((condition) => [condition.id, condition.label])), [completed]);

  return (
    <main className="lab-shell">
      <section className="question-stage" aria-labelledby="question-title">
        <div className="folio">01 / QUESTION</div>
        <h2 id="question-title">Change one rule.<br />Watch both worlds.</h2>
        <div className="experiment-explainer">
          <p><strong>Why run this?</strong> We compare an unchanged world with a world where one rule changes. This tests whether the result gives a model a reason to revise an inadequate explanation.</p>
          <p><strong>What are the twelve particles?</strong> They are twelve simulated dots in a small 2D world. They begin still. Halfway through, repulsion makes the dots push apart in the changed world. Twelve is only a small number that is easy to see; it has no special scientific meaning.</p>
          <small>The dots are not atoms, real observations, or model activations. This is a bounded thought experiment—not a claim about the real world.</small>
        </div>
        <Textarea value={question} onChange={(event) => setQuestion(event.target.value)} rows={5} className="question-input" aria-label="Thought experiment question" />
        <div className="example-row">{examples.map((example, index) => <button key={example} onClick={() => setQuestion(example)}><span>0{index + 1}</span>{example}</button>)}</div>
        <Button size="lg" onClick={buildPlan} disabled={stage === "planning" || stage === "running"} className="primary-action">Build the plan <ArrowRight /></Button>
        {(stage === "planning" || stage === "running") && <div className="progress-line"><i /><span>{progress}</span></div>}
        {stage === "error" && <div className="error-sheet" role="alert"><strong>Experiment stopped safely.</strong><p>{error}</p><Button variant="outline" onClick={() => setStage(planned ? "confirm" : "question")}><RotateCcw /> Edit and try again</Button><small>No cached plan or result was substituted.</small></div>}
      </section>

      {planned && <section className="paper-sheet plan-sheet" aria-labelledby="plan-title">
        <div className="section-mark"><span>02</span><p>PLAN CONFIRMATION</p></div>
        <div className="sheet-body"><Badge variant="outline">Validated spec · {planned.spec.spec_id}</Badge><h2 id="plan-title">{planned.spec.question}</h2><p className="hypothesis"><span>Hypothesis</span>{planned.spec.hypothesis}</p>
          <div className="plan-grid"><PlanDatum label="Baseline" value={planned.spec.conditions.find((item) => item.kind === "baseline")?.label || "Baseline"} /><PlanDatum label="Counterfactual" value={planned.spec.conditions.find((item) => item.kind === "counterfactual")?.label || "Changed world"} /><PlanDatum label="Duration" value={`${planned.spec.schedule.duration_steps} steps`} /><PlanDatum label="Frames" value={`Up to ${planned.spec.visualization.max_frames} per world`} /></div>
          <p className="limitation">The compiler selected only allowlisted declarative rules. No code was generated or executed.</p>
          {stage === "confirm" && <Button size="lg" onClick={runExperiment} className="primary-action">Confirm and run both worlds <ArrowRight /></Button>}
        </div>
      </section>}

      {completed && <>
        <section className="result-lead" aria-labelledby="result-title"><div className="folio light">03 / RESULT</div><Badge><Check /> Validated run</Badge><h2 id="result-title">{completed.run.execution.prediction.summary}</h2><div className="result-columns"><div><span>Prediction</span><p>Expected {completed.run.execution.prediction.expected_direction.replaceAll("_", " ")} in {completed.run.execution.prediction.measurement_id}.</p></div><div><span>Interpretation · {completed.run.revision.disposition}</span><p>{completed.run.revision.interpretation}</p></div></div></section>
        <section className="simulation-section"><div className="section-head"><div><p className="eyebrow">DETERMINISTIC SIMULATION FRAMES</p><h2>Baseline / counterfactual</h2></div><div className="playback"><Button variant="outline" size="icon" onClick={() => setPlaying((value) => !value)} aria-label={playing ? "Pause animation" : "Play animation"}>{playing ? <Pause /> : <Play />}</Button><span>Frame {frameIndex + 1} / {maxFrames}</span></div></div>
          <div className="world-grid">{completed.run.conditions.slice(0, 2).map((condition, index) => <Card key={condition.condition_id} className="world-card"><CardContent><div className="world-label"><span>{index === 0 ? "A" : "B"}</span><div><p>{index === 0 ? "BASELINE" : "COUNTERFACTUAL"}</p><h3>{conditionLabels[condition.condition_id]}</h3></div></div><SimulationCanvas condition={condition} spec={completed.spec} frameIndex={frameIndex} /></CardContent></Card>)}</div>
          <input className="frame-scrubber" type="range" min="0" max={maxFrames - 1} value={frameIndex} onChange={(event) => { setPlaying(false); setFrameIndex(Number(event.target.value)); }} aria-label="Simulation frame" />
          <p className="limitation centered">These are deterministic engine states—not learned-latent reconstructions.</p>
        </section>
        <section className="paper-sheet chart-section"><div className="section-mark"><span>04</span><p>MEASUREMENT</p></div><div className="sheet-body"><h2>What changed?</h2><ComparisonChart run={completed.run} spec={completed.spec} /><div className="comparison-list">{completed.run.comparisons.map((comparison) => <div key={`${comparison.measurement_id}-${comparison.counterfactual_condition_id}`}><span>{comparison.measurement_id}</span><strong>{comparison.baseline_final.toFixed(3)} → {comparison.counterfactual_final.toFixed(3)}</strong><em>{comparison.difference >= 0 ? "+" : ""}{comparison.difference.toFixed(3)}</em></div>)}</div>
          <Accordion type="single" collapsible><AccordionItem value="evidence"><AccordionTrigger><span className="evidence-trigger"><ShieldCheck /> Evidence & limitations</span></AccordionTrigger><AccordionContent><dl className="evidence-list"><div><dt>Run</dt><dd>{completed.run.run_id}</dd></div><div><dt>Engine</dt><dd>{completed.run.execution.engine_id}</dd></div><div><dt>Prediction recorded</dt><dd>{completed.run.execution.prediction_recorded_at}</dd></div><div><dt>Execution started</dt><dd>{completed.run.execution.started_at}</dd></div></dl><Separator /><p className="limitation">This simulator tests consequences inside the confirmed model. It does not validate the assumptions or establish effects outside that model.</p></AccordionContent></AccordionItem></Accordion>
        </div></section>
      </>}
    </main>
  );
}

function PlanDatum({ label, value }: { label: string; value: string }) { return <div><span>{label}</span><strong>{value}</strong></div>; }
