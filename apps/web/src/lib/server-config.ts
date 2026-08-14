import "server-only";
import { CODE_VERSION, ENGINE_ID, RUN_SCHEMA_SHA256, SPEC_SCHEMA_SHA256 } from "./contracts";

export const MODAL_ENDPOINT = "https://ameymuke252003--jump-general-experiment-workbench-genera-d81606.modal.run";

export function serverConfig() {
  const expected = {
    JUMP_MODAL_ENDPOINT: MODAL_ENDPOINT,
    JUMP_MODAL_CODE_VERSION: CODE_VERSION,
    JUMP_VISUAL_SPEC_SCHEMA_SHA256: SPEC_SCHEMA_SHA256,
    JUMP_VISUAL_RUN_SCHEMA_SHA256: RUN_SCHEMA_SHA256,
    JUMP_VISUAL_ENGINE_ID: ENGINE_ID,
  };
  for (const [key, value] of Object.entries(expected)) {
    if (process.env[key] !== value) throw new Error(`Server deployment pin mismatch: ${key}`);
  }
  if (!process.env.JUMP_MODAL_TOKEN) throw new Error("Server bearer token is unavailable");
  return { endpoint: MODAL_ENDPOINT, token: process.env.JUMP_MODAL_TOKEN };
}
