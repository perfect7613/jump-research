"use client";

import { useEffect, useRef } from "react";
import type { ExperimentSpec, ThoughtRun } from "@/lib/contracts";

type Condition = ThoughtRun["conditions"][number];

export function SimulationCanvas({ condition, spec, frameIndex }: { condition: Condition; spec: ExperimentSpec; frameIndex: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const frame = condition.frames[Math.min(frameIndex, condition.frames.length - 1)];
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !frame) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = width * ratio; canvas.height = height * ratio;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.scale(ratio, ratio);
    context.fillStyle = "#f7f0e2"; context.fillRect(0, 0, width, height);
    context.strokeStyle = "rgba(28,35,48,.13)"; context.lineWidth = 1;
    for (let x = 0; x <= width; x += width / 8) { context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke(); }
    for (let y = 0; y <= height; y += height / 5) { context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke(); }
    for (const point of frame.points) {
      const x = (point.x / spec.world.bounds.width) * width;
      const y = (point.y / spec.world.bounds.height) * height;
      const size = Math.max(3, point.size * 1.5);
      context.fillStyle = point.color; context.beginPath();
      if (point.shape === "square") context.rect(x - size, y - size, size * 2, size * 2);
      else if (point.shape === "triangle") { context.moveTo(x, y - size); context.lineTo(x + size, y + size); context.lineTo(x - size, y + size); context.closePath(); }
      else context.arc(x, y, size, 0, Math.PI * 2);
      context.fill();
    }
  }, [frame, spec]);
  return <canvas ref={ref} className="simulation-canvas" aria-label={`Deterministic simulation at step ${frame?.step ?? 0}`} />;
}

export function ComparisonChart({ run, spec }: { run: ThoughtRun; spec: ExperimentSpec }) {
  const wanted = new Set(spec.visualization.chart_measurement_ids);
  const tracks = run.conditions.flatMap((condition) => condition.series.filter((item) => wanted.has(item.measurement_id)).map((item) => ({ condition: condition.condition_id, ...item })));
  if (!tracks.length) return null;
  const values = tracks.flatMap((track) => track.values.map((item) => item.value));
  const low = Math.min(...values); const high = Math.max(...values); const span = high - low || 1;
  return (
    <div className="chart-wrap">
      <svg viewBox="0 0 720 250" role="img" aria-label="Baseline and counterfactual measurement chart">
        {[0, 1, 2, 3, 4].map((line) => <line key={line} x1="44" y1={30 + line * 44} x2="700" y2={30 + line * 44} className="chart-grid" />)}
        {tracks.slice(0, 4).map((track, index) => {
          const maxStep = Math.max(...track.values.map((item) => item.step), 1);
          const points = track.values.map((item) => `${44 + (item.step / maxStep) * 656},${206 - ((item.value - low) / span) * 176}`).join(" ");
          return <polyline key={`${track.condition}-${track.measurement_id}`} points={points} className={index % 2 ? "chart-line counter" : "chart-line baseline"} />;
        })}
      </svg>
      <div className="chart-key"><span><i className="key-baseline" />Baseline</span><span><i className="key-counter" />Counterfactual</span></div>
    </div>
  );
}
