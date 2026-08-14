import "server-only";
import { CODE_VERSION, ENGINE_ID, QUESTION_VERSION, RUN_SCHEMA_SHA256, SPEC_SCHEMA_SHA256 } from "@/lib/contracts";
import { serverConfig } from "@/lib/server-config";

export const runtime = "nodejs";

export async function GET() {
  try {
    const config = serverConfig();
    const response = await fetch(`${config.endpoint}/health`, { headers: { Authorization: `Bearer ${config.token}`, Accept: "application/json" }, cache: "no-store" });
    const payload = await response.json();
    const visual = payload?.thought_experiments_v2;
    if (!response.ok || payload?.status !== "available" || payload?.code_version !== CODE_VERSION || visual?.question_schema_version !== QUESTION_VERSION || visual?.spec_schema_sha256 !== SPEC_SCHEMA_SHA256 || visual?.run_schema_sha256 !== RUN_SCHEMA_SHA256 || visual?.engine_id !== ENGINE_ID || visual?.generated_code !== false || visual?.learned_decoder !== false) throw new Error("Health pins do not match");
    return Response.json({ status: "available", code_version: CODE_VERSION, engine_id: ENGINE_ID, generated_code: false, learned_decoder: false });
  } catch {
    return Response.json({ status: "unavailable" }, { status: 503 });
  }
}
