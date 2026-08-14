import { describe, expect, it } from "vitest";
import { confirmationSchema, questionSchema, SPEC_SCHEMA_SHA256, RUN_SCHEMA_SHA256 } from "./contracts";

describe("v2 ingress contracts", () => {
  it("accepts only the exact bounded question fields", () => {
    const question = {
      schema_version: "jump.thought-experiment-question/v2",
      request_id: "req-test",
      session_id: "session-test",
      intent: "What happens if repulsion begins halfway through?",
      seed: 7613,
      repetitions: 2,
    };
    expect(questionSchema.parse(question)).toEqual(question);
    expect(() => questionSchema.parse({ ...question, source: "not accepted" })).toThrow();
    expect(() => questionSchema.parse({ ...question, intent: "Download https://example.com/data.csv" })).toThrow(/URLs/);
  });

  it("requires the token-bound seven-field confirmation", () => {
    const confirmation = {
      schema_version: "jump.thought-experiment-confirmation/v2",
      request_id: "req-test",
      session_id: "session-test",
      spec_id: "spec-0123456789abcdef01234567",
      spec_sha256: "a".repeat(64),
      confirmation_token: "b".repeat(64),
      confirmed: true,
    };
    expect(confirmationSchema.parse(confirmation)).toEqual(confirmation);
    const oldConfirmation = { ...confirmation } as Partial<typeof confirmation>;
    delete oldConfirmation.request_id;
    expect(() => confirmationSchema.parse(oldConfirmation)).toThrow();
  });

  it("pins the frozen schema identities", () => {
    expect(SPEC_SCHEMA_SHA256).toBe("fa7674dc3c5f759dc74ff723cef7a194edc4186069496e631e65b4d0ebd84ab5");
    expect(RUN_SCHEMA_SHA256).toBe("55d1fd3fdef215abfb1a148080cc01aea3fff118ba1e779e02e6841f43941166");
  });
});
