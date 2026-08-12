# JUMP Plan — Independent Red-Team Review

Reviewed artifact: `1786528309242-JUMP_COMPLETE_RESEARCH_PLAN.md` (1,797 lines)
Reviewer: independent red-team worker (scratch-5)
Date: 2026-08-12
Scope: drawbacks, security/privacy, legal/compliance, operational risk, cost, scalability, failure modes, hidden assumptions, go/no-go.

Severity scale: Critical / High / Medium / Low.
Likelihood scale: High / Medium / Low.

---

## 0. Executive summary

The plan is unusually self-aware (explicit non-claims, controls A–J, phase gates, cost ledger, truthfulness rules). Its dominant defect is not any single technical risk but a **structural contradiction: it binds a 28–50 day gated research program (Phases 0–7, Section 11) to a 2-day hackathon with a $188 compute ceiling**, and its public product framing ("ask any scientific question") vastly exceeds what one toy 2D world can deliver. Most other risks are downstream of that compression: budget overrun, demo fragility, rushed causal claims, and truthfulness commitments that are expensive to actually honor under deadline pressure.

**Recommendation: CONDITIONAL GO** — go for the hackathon deliverable (vertical slice + cached demos + honest labeling), **NO-GO for shipping any mechanistic/causal research claim within the hackathon window**. Details in Section 9.

---

## 1. Structural / planning risks

### R1. Timeline contradiction between research plan and hackathon plan
- **Severity: Critical. Likelihood: High (near certain). Impact:** Phases 0–7 sum to 28–50 working days; the hackathon allows 2. If the team attempts both simultaneously, neither the research gates nor the demo polish gets adequate time, and the submission slips or ships broken. Nothing in the document reconciles the two schedules — Section 11 and Section 17 are two different projects sharing one name.
- **Mitigations:** Explicitly declare the hackathon scope to be Phase 0 + a thin slice of Phases 1–3 (behavioural baseline, one trained latent, one adapter injection, one swap demo). Pre-write the demo script assuming the *minimum viable* result only. Treat Phases 4–7 (localization, causal surgery, mediation) as post-hackathon work and say so publicly.
- **Hidden assumption:** that "Day 2 morning: implement Hidden Law Discovery + latent-only readout + matched A/B swap + benchmarks" is a morning of work. Each of these is days of work in the research plan itself (Phases 3–5 are budgeted 13–24 days).

### R2. Product promise vs. actual capability gap
- **Severity: High. Likelihood: High. Impact:** The product is framed as "ask JUMP a scientific question as naturally as you would ask ChatGPT" with a domain router and 8 adapter families — but only one toy synthetic-2D adapter will exist. Real users/judges will immediately ask out-of-domain questions; the system will fall back to qualitative LLM prose. If the decoded-latent visual appears anywhere near such answers, the demo *implies* grounding it doesn't have — precisely the failure the plan's own truthfulness section forbids. Reputational risk if a judge catches an unlabeled qualitative answer next to a "simulated" artifact.
- **Mitigations:** Constrain the public composer to the supported domain (visible scope banner: "JUMP currently executes thought experiments only in synthetic 2D dynamics"). Make the unsupported-domain refusal path the *most* tested path, not an afterthought. Default the suggested prompt chips to the three flagship demos only.

### R3. The "scientific query compiler" is an unbounded NLP problem on the critical path
- **Severity: High. Likelihood: High. Impact:** Translating free-form conversation into a valid `ScientificQueryPlan` (assumptions, hypotheses, interventions, solver selection) is itself a hard, failure-prone LLM task — arguably harder than the research question. It sits between every user message and every demo. Compiler misfires (wrong assumption inferred, invalid DSL emitted) will dominate observed failures and burn Day 1/2 debugging time.
- **Mitigations:** For the hackathon, hard-code the compiler output for the three flagship conversations (template matching with slot-filling), with the general compiler as best-effort behind them. Log every compiled plan; surface the "Experiment JUMP constructed" card so failures are visible rather than silent.

---

## 2. Cost / budget risks

### R4. $188 ceiling is realistic only if everything works the first time
- **Severity: High. Likelihood: Medium-High. Impact:** The ledger allocates $15 for "debugging and necessary reruns" against $145 of planned experimentation — a ~10% rerun allowance. Real ML projects on new fusion architectures typically consume 2–5× planned GPU time in reruns (bad hyperparameters, hook bugs, OOMs, corrupted activation dumps). One failed activation-extraction run on A100-80GB ($2.50/hr) with a bug discovered after 6 hours eats the entire debug budget. Activation storage is also underpriced: residual streams for a 12B model across layers/timepoints/episodes can reach hundreds of GB even sharded in bf16, and Modal storage/egress is explicitly excluded from the GPU-hour table.
- **Mitigations:** The per-phase cost gates are good — enforce them mechanically (a wrapper that refuses to launch when the ledger exceeds the gate). Extract activations only at the preregistered layers/timepoints (plan says this; make it a hard allowlist in code). Smoke-test every job at 1% scale on T4/L4 before any A100 launch. Consider dropping the localization/causal phases entirely from the paid window (see R1).

### R5. Denial-of-wallet through the public Space
- **Severity: High. Likelihood: Medium. Impact:** A public HF Space whose backend calls a metered Modal GPU is a credit-drain endpoint. Any visitor (or a script) can submit chat turns; each triggers a 12B-model container. With scaledown 3–5 min raised "during the judging window," a handful of griefers or even organic hackathon traffic can exhaust the remaining ~$15 serving budget mid-judging, taking the live demo down at the worst moment. The plan mentions rate limits and validation but gives no concrete mechanism.
- **Mitigations:** Per-IP/session rate limiting *in the Space backend* plus a global daily token/GPU-second cap enforced Modal-side. Serve anonymous visitors the cached/precomputed conversations by default with a "run live" button behind the rate limiter. Keep the demo-video path fully independent of live infra (the plan's cached-mode fallback is correct — make it the default, not the fallback).

### R6. Credit terms are unverified assumptions
- **Severity: Medium. Likelihood: Medium. Impact:** The plan assumes $188 is available, non-expiring during the event, and that workspace/environment budget caps exist as described. If credits expire, are promotional-tier-limited (e.g., no A100/H100 access), or budgets bill with delay (plan acknowledges delay), the ledger silently breaks.
- **Mitigations:** Verify credit balance, expiry, GPU-class eligibility, and budget-cap behavior with a $1 smoke test on Aug 11 preflight (the plan already schedules preflight — add this item explicitly).

---

## 3. Security risks

### R7. Untrusted natural language compiled into an executable DSL
- **Severity: High. Likelihood: Medium. Impact:** The pipeline is: attacker-controlled chat → LLM → generated equations/programs → "whitelisted DSL" → numerical execution on paid GPU/CPU. This is a classic prompt-injection-to-tool-execution chain. Even a genuinely whitelisted DSL is exposed to resource-exhaustion payloads (stiff ODEs, near-singular systems, adversarial coefficients driving max timesteps/memory), and DSL validators written in a 2-day sprint routinely have gaps (eval-adjacent parsing, unbounded recursion in expression trees, exponent bombs like `r^(10^9)`).
- **Mitigations:** Validate *post-compilation artifacts*, not prompts: hard caps on objects, dims, coefficient magnitude, exponent range, timesteps, wall time, memory — enforced in the executor, not just the compiler (plan states this; ensure the enforcement lives Modal-side). Parse the DSL with a real grammar, never `eval`/`exec`. Fuzz the compiler→executor path with adversarial strings before going public. Timeout + kill at the Modal function level as final backstop (plan has this).

### R8. Secret handling across HF Space / GitHub / Modal
- **Severity: High. Likelihood: Medium. Impact:** Modal credentials live in HF Space secrets; `HF_TOKEN` lives in GitHub Actions; the Space repo is public and synced automatically. Common failure modes under deadline pressure: secrets echoed into build logs or Gradio error tracebacks, tokens committed to the public repo during a hasty sync fix, over-scoped `HF_TOKEN` (write access to the whole org), Modal token embedded in a client-visible config. The plan's guidance is correct but has no verification step.
- **Mitigations:** Pre-deadline secret scan (gitleaks/trufflehog) in CI on every push. Minimum-scope, per-purpose tokens; treat the Space→Modal token as revocable and rotate after the event. Catch-all exception handler in the Space that never surfaces raw tracebacks (plan's "friendly errors" — enforce it as a wrapper). Confirm Modal endpoint requires auth even if the URL leaks.

### R9. Untrusted file/image/table attachments
- **Severity: Medium. Likelihood: Medium. Impact:** The composer accepts file/image/table uploads for "observations." Parsing untrusted CSVs/images server-side (pandas, PIL) exposes decompression bombs, malformed-file DoS, and — if any parsed content flows back into prompts — indirect prompt injection into the query compiler.
- **Mitigations:** Size caps, MIME/extension allowlist, parse inside the sandboxed Modal function with timeouts, never render uploaded content as raw HTML, treat file-derived text as untrusted data (delimit clearly in prompts). For the hackathon, consider disabling uploads entirely — they serve no flagship demo.

### R10. Supply-chain and CI risk
- **Severity: Low-Medium. Likelihood: Low-Medium. Impact:** Pinned requirements are planned (good), but a 2-day sprint invites `pip install` of unvetted helper packages, and the GitHub→Space sync action runs with a write token on every push to main.
- **Mitigations:** Lockfile from Day 1; restrict the sync workflow to tagged releases or a protected branch; review action versions (pin by SHA).

---

## 4. Privacy / legal / compliance risks

### R11. Conversation data handling has no stated policy
- **Severity: Medium. Likelihood: High (that data gets collected) / Medium (that it causes harm). Impact:** The UI includes conversation history and multi-turn memory; the plan says "make user-run logging opt-in" and "strip personal data," but a public chat product with server-side conversation state *is* processing user content. There is no privacy notice, retention policy, or deletion mechanism. Users may paste proprietary data or personal information into "observations." Under GDPR-style regimes, even a demo processing EU visitors' chat content without notice is noncompliant; HF/Modal act as subprocessors nobody has papered.
- **Mitigations:** Session-scoped, in-memory conversation state only; no server persistence of user chats for the hackathon. A one-line privacy notice in the About panel ("conversations are processed transiently on Hugging Face and Modal infrastructure; do not submit personal or confidential data; anonymous demos may be logged for debugging"). Make any logging opt-in as planned, and actually implement the opt-in.

### R12. Gemma license and derivative-artifact compliance
- **Severity: Medium. Likelihood: Medium. Impact:** The plan releases adapters, decoders, and checkpoints derived from a Gemma checkpoint. Gemma-family weights ship under Google's Terms of Use with a Prohibited Use Policy and pass-through obligations for derivatives. Publishing `jump-gemma-adapter` etc. without the required notice/terms propagation, or serving a public endpoint without use restrictions, breaches the license. Also unverified: whether the assumed checkpoint is gated (requiring per-user acceptance that the public Space's server-side download must respect).
- **Mitigations:** Read the exact license of the chosen checkpoint during Aug 11 preflight; include the required terms/notice in every released model card; verify gated-access mechanics work from Modal with the team's token; state use restrictions in the Space README.

### R13. Scientific-assistant liability and dual-use surface
- **Severity: Medium. Likelihood: Low-Medium. Impact:** The product invites arbitrary scientific questions. The plan wisely bans "presenting toy simulation as validation of medical/safety-critical conclusions," but the public chat will still *receive* medical, chemical, or hazardous "what-if" questions and answer them with base-model prose wearing a scientific-instrument UI. The instrument framing increases perceived authority of unvalidated answers. Future chemistry/biology adapters (mentioned as extensions) raise a genuine dual-use question that is currently unaddressed.
- **Mitigations:** Visible disclaimer ("research demo; not scientific, medical, or safety advice"). Refusal/deflection templates for hazardous or real-world-consequential domains. Defer chem/bio adapters and remove them from public roadmap language until a safety review exists.

### R14. Citation integrity
- **Severity: Medium. Likelihood: Medium. Impact:** Reference 5 (R-lens) explicitly lacks a primary source; several other references are recent enough that details may be misremembered. The plan bans fabricated citations, yet the write-up phase is 3–5 days at the end of an exhausted sprint — the classic moment unverified citations slip into a README/paper. A hackathon judge or researcher spotting one broken/nonexistent citation discounts the whole credibility story the project depends on.
- **Mitigations:** Verify every URL/arXiv ID resolves before the README ships (scriptable). Drop or clearly caveat anything unverifiable (the plan already does this for R-lens — apply the same standard universally).

### R15. Hackathon-rules compliance gray zone
- **Severity: Medium. Likelihood: Low-Medium. Impact:** The plan schedules substantial pre-hackathon preparation (schemas designed, architecture fixed, task board, acceptance criteria) and says "without claiming pre-hackathon implementation." Extensive design-complete prep can still be judged as rule-bending if the rules require work *built* during the window; also the plan itself flags that deadline timezone and rules must be confirmed. Disqualification risk is low but the cost is total.
- **Mitigations:** Keep all pre-window artifacts as documents/notes, zero code. Confirm rules in `#orchestra-announcements` (planned) and get an explicit read on what prep is permitted if ambiguous. Commit history should tell an honest story.

---

## 5. Scientific-validity risks (research-side drawbacks)

### R16. Causal-claim standards cannot be met in the available time — and partial results invite overclaiming
- **Severity: High. Likelihood: High. Impact:** The plan's own bar for a "serious research claim" is a preregistered intervention surviving matched controls, cross-seed replication, and mediation analysis (Phases 4–6, 13–24 days). The demo narrative ("causal proof," 0:35–0:55 in the video script) will be recorded before any of that exists. A single cherry-picked latent-swap example presented as "causal proof" is exactly the overclaim Section 2 forbids. The most likely bad outcome is not fraud but *drift*: demo language hardens into claims the evidence doesn't support.
- **Mitigations:** Rename the video segment from "causal proof" to "causal test" or "swap demonstration." Script the demo with the plan's own non-claims language. Ship the benchmark + infrastructure as the hackathon contribution and the mechanism study as future work.

### R17. Confound between "world latent helps" and "any structured signal helps"
- **Severity: Medium. Likelihood: Medium. Impact:** The plan's own failure-mode table admits the language-hint condition (B) may equal the world latent (E), and control I (same-parameter adapter without dynamics) is the key discriminator — but adapter training itself can leak task structure: the adapter is trained on episodes from the same generative family, so improvements may reflect distribution-specific fine-tuning rather than counterfactual execution. With one checkpoint, one world family, and validation-set architecture selection, the headline comparison is fragile.
- **Mitigations:** Already partially planned (controls G/I, held-out law family, second checkpoint "if compute permits"). Elevate the OOD law-family evaluation from optional to required for any public "the latent causes X" statement. Report B vs E prominently even if unflattering.

### R18. "Decoded from the same latent" is a strong product claim with subtle ways to become false
- **Severity: Medium. Likelihood: Medium. Impact:** The flagship credibility feature promises the displayed image derives from the exact latent Gemma consumed. Quantization of the serving path (mentioned as an option), caching of precomputed demos, batching, or a decoder retrained after the adapter was frozen can all silently break the "same latent" invariant, making the provenance badge false — the single most damaging possible truthfulness failure given how heavily it is marketed.
- **Mitigations:** Hash the latent tensor and stamp the hash into both the decoder call and the answer manifest; the provenance badge should display the hash match, not an assertion. Regression-test the invariant in CI. If cached demos are shown, the cache must store latent + image + answer as one sealed artifact (the plan's single-result-object rule helps — enforce it with the hash).

### R19. The non-saturated difficulty regime (20–70%) may not exist for the chosen checkpoint
- **Severity: Medium. Likelihood: Medium. Impact:** Phase 1's gate assumes a difficulty dial can place the base model in a mixed-success regime. Small open multimodal models often sit at ~0% on genuine abduction tasks regardless of evidence tuning (or the task becomes trivially solvable by symbolic-text leakage once evidence is revealing enough). If Phase 1 gates fail, everything downstream — including the hackathon's Hidden Law Discovery demo — has no interesting behaviour to show.
- **Mitigations:** The pivot table covers this (ontology-supplied condition first). For the hackathon, make Invisible Orbit / Counterfactual Orbit (which don't require abduction success) the load-bearing demos, with Hidden Law Discovery as stretch — the Day 2 schedule currently assumes the opposite priority.

---

## 6. Operational / deployment failure modes

### R20. Cold-start latency vs. judge experience
- **Severity: Medium. Likelihood: High. Impact:** Scale-to-zero with 60–90s scaledown means most judge visits hit a cold container: image pull + 12B weight load + adapter/hook init, plausibly 1–3 minutes before first token. A judge waiting 2 minutes on a spinner concludes the demo is broken. Raising scaledown "during the judging window" assumes the team knows when judges visit — they don't, for asynchronous judging.
- **Mitigations:** Instant cached responses for the three flagship conversations while the container warms (with honest "warming live backend…" status). Weight snapshotting/fast-load if Modal supports it for the image size. Accept the cost of one warm container for the full judging day within the serving budget — recompute the ledger accordingly (~L40S at $1.95/hr × 12h ≈ $23 exceeds the $15 serving line: this is an unresolved conflict in the plan).

### R21. End-to-end integration is scheduled with zero slack
- **Severity: High. Likelihood: High. Impact:** The Day 1 target is a full conversational pipeline (NL → compiler → simulator → latent → injection → decode → answer → manifest) plus first Space and Modal deploys. That is 6+ integration boundaries in one day, across a team coordinating via AO for the first time under event conditions. Historically, the Space↔Modal auth path and the Gradio streaming/queue interplay alone consume hours. Slippage cascades into Day 2's research-credibility work (R1).
- **Mitigations:** The plan's mocked-result-object UI strategy is the right call — extend it: the Space must be fully demo-able against recorded result objects with *zero* live backend, so the deadline-day submission never depends on live integration. Define the Day 1 "known-good tag" as the submission floor.

### R22. Deadline-window single points of failure
- **Severity: Medium. Likelihood: Medium. Impact:** GitHub→Space sync action failing at 6 PM, HF Space build queue delays, Space build breaking on a transitive dependency, demo-video host hiccups, Discord/X posts forgotten — each individually small, jointly likely. The plan's checklist is good but has no owner or rehearsal.
- **Mitigations:** Assign one person as submission owner with a rehearsed dry run (post to a private channel, build the Space from scratch once on Day 1). Freeze main at T-3h, not "well before" (quantify it). Keep the direct-push-to-Space path tested as backup to the Action.

### R23. Team/coordination assumptions
- **Severity: Medium. Likelihood: Medium. Impact:** The plan assumes up to 4 people with clean ownership of integration, model, product, evaluation, and submission — 5 roles for ≤4 people — plus fluency with AO, Modal, Gradio, HF Hub, and mechanistic-interp tooling simultaneously. It also assumes AO orchestration overhead is negative (accelerates) rather than positive (new tool learned during a deadline).
- **Mitigations:** Collapse roles explicitly (evaluation folds into model; submission into product). Do the AO workflow rehearsal in preflight with a trivial task so Day 1 isn't the first AO experience. Decide in advance what gets cut if only 2–3 people show up.

---

## 7. Scalability (product trajectory) concerns

- **Severity: Low for the hackathon, Medium if the product framing persists. Impact:** The adapter-registry architecture is sound in principle, but each new scientific domain requires a validated simulator, schema, guardrails, and verification method — i.e., every domain is its own project. The conversational compiler's accuracy also degrades as the domain space grows (routing + formalization errors compound). The per-conversation structured memory, saved latent worlds, and activation artifacts grow unboundedly with users; nothing in the plan addresses multi-user state, quotas, or storage lifecycle beyond "clean bounded per-run directories."
- **Mitigations:** None needed for the event. For any continuation: per-domain launch checklist (validation suite + red-team pass), storage TTLs, and a compiler evaluation set per domain before routing to it.

---

## 8. Hidden assumptions register

| # | Assumption | Risk if false |
|---|---|---|
| A1 | `google/gemma-4-12B` exists, is open-weight, multimodal, ungated-or-accessible, and exposes usable activation hooks | Entire model plan re-planned on Day 1; fallback checkpoint changes memory/GPU math (Critical-path; verify in preflight) |
| A2 | 12B multimodal inference + hooks + adapter fits L40S 48GB in bf16 with headroom for activation capture | Forced onto A100-80GB, roughly +28% cost per hour and tighter hour budget |
| A3 | $188 credits valid, non-expiring, all GPU classes eligible | Ledger and phase gates invalid |
| A4 | Latent dynamics module trains to useful multi-step prediction from 50–100k transitions within the $25 line | Phase 2 gate fails inside hackathon window; decoded-latent demos impossible |
| A5 | A 20–70% base-success difficulty regime exists (R19) | Hidden Law Discovery demo has nothing to show |
| A6 | The query compiler can be made reliable enough for live judge input in 2 days (R3) | Live demo embarrassments; mitigated by templated flagship paths |
| A7 | HF Space ↔ Modal round trip latency acceptable for streaming chat | Sluggish UX; needs measured early Day 1 |
| A8 | Judges evaluate within the window when scaledown is raised (R20) | Cold-start impressions |
| A9 | AO orchestration is net-positive time under deadline (R23) | Coordination overhead consumes build time |
| A10 | Pre-hackathon design prep is rules-compliant (R15) | Disqualification (low likelihood, total impact) |
| A11 | "Preregistration" without an external registry will be credited as such | Reviewers treat analyses as exploratory; soften language or use OSF/timestamped commit |
| A12 | The same team can honor the plan's extensive truthfulness/labeling commitments while sprinting | The most-marketed integrity features (provenance badges, cached-run labels) are the first casualties of time pressure |

---

## 9. Go / no-go assessment for the orchestrator

**Overall: CONDITIONAL GO**, with a mandatory descope decision before Day 1.

**GO — with conditions:**
1. **Hackathon deliverable (GO):** Space + simulator + one working latent-injection vertical slice + cached flagship conversations + honest labeling. This is achievable, differentiated, and low-liability *if* R2/R5/R18 mitigations are in place.
2. **Research program (GO, but post-hackathon):** The model-organism design, controls, and gates are genuinely good science. Run Phases 1–7 on their own 4–7 week timeline after the event; the hackathon should only claim to ship the *benchmark and instrument*, not the mechanism.

**NO-GO items (do not ship these in the hackathon window):**
1. Any public statement of a **causal or mechanistic result** (inadequacy-state mediation, "causal proof") — the plan's own evidentiary bar cannot be met in 2 days (R16).
2. **Open live GPU chat without rate limiting and a global spend cap** — denial-of-wallet risk to the demo itself (R5).
3. **General-scientific-assistant framing without a visible domain-scope banner** (R2).
4. **Releasing Gemma-derived checkpoints before license pass-through terms are confirmed** (R12).
5. **Server-side persistence of user conversations** without a privacy notice (R11) — simplest fix: don't persist.

**Top three decisions needed from the orchestrator now:**
1. Pick the descope: which of Day 2's research features (Hidden Law Discovery, latent-only readout, A/B swap, benchmarks) are cut-line vs. must-have. Recommendation: swap demo = must-have; Hidden Law Discovery = stretch (inverts the plan's current priority, per R19).
2. Resolve the serving-budget conflict: warm container for judging (~$23) vs. the $15 serving line (R20) — either fund it from the reserve by written decision (the plan's own reserve rule) or commit to cached-first UX.
3. Assign a single owner for the truthfulness/provenance invariants (latent hash, cached-run labels, scope banner) with authority to block the demo recording if they're not green (A12, R18).
