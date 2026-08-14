import "server-only";
import { confirmationSchema, questionSchema, validateRunResponse, validateSpecResponse } from "@/lib/contracts";
import { serverConfig } from "@/lib/server-config";

export const runtime = "nodejs";
export const maxDuration = 300;

export async function POST(request: Request, context: { params: Promise<{ action: string }> }) {
  const { action } = await context.params;
  if (action !== "spec" && action !== "confirm") return Response.json({ detail: "Unknown action" }, { status: 404 });
  let config: ReturnType<typeof serverConfig>;
  try { config = serverConfig(); } catch { return Response.json({ detail: "Visual experiment service is unavailable" }, { status: 503 }); }
  let body: unknown;
  try { body = await request.json(); } catch { return Response.json({ detail: "Request must be JSON" }, { status: 400 }); }
  const parsed = action === "spec" ? questionSchema.safeParse(body) : confirmationSchema.safeParse(body);
  if (!parsed.success) return Response.json({ detail: parsed.error.issues[0]?.message ?? "Request rejected" }, { status: 400 });

  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      const emit = (value: unknown) => controller.enqueue(encoder.encode(`${JSON.stringify(value)}\n`));
      emit({ type: "progress", message: action === "spec" ? "Writing a bounded plan" : "Recording the prediction before simulation" });
      try {
        const upstream = await fetch(`${config.endpoint}/v2/thought-experiments/${action}`, {
          method: "POST", headers: { Authorization: `Bearer ${config.token}`, "Content-Type": "application/json", Accept: "application/json" },
          body: JSON.stringify(parsed.data), cache: "no-store", signal: request.signal,
        });
        const payload = await upstream.json();
        if (!upstream.ok) { emit({ type: "error", detail: typeof payload?.detail === "string" ? payload.detail : "Experiment rejected" }); return; }
        if (action === "spec") {
          emit({ type: "result", payload: validateSpecResponse(payload, questionSchema.parse(parsed.data)) });
        } else {
          emit({ type: "result", payload: validateRunResponse(payload, confirmationSchema.parse(parsed.data)) });
        }
      } catch (error) {
        emit({ type: "error", detail: error instanceof Error ? error.message : "Experiment failed closed" });
      } finally { controller.close(); }
    },
  });
  return new Response(stream, { headers: { "Content-Type": "application/x-ndjson", "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff" } });
}
