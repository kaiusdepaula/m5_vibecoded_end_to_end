---
date: 2026-09-11
topic: cockpit-vs-claude-code-pod-tasks
---

# AI Cockpit vs. Claude Code — Pod Evaluation

Same tasks in both tools, driven however feels natural. Nothing is labeled by hand: tests decide the outcome, an LLM judge reads the transcripts for everything tests can't see.

Pick the tasks and write their tests **before** the first run, so we don't drift toward work that flatters whichever tool we already prefer.

## How a run is scored

1. **Outcome — hidden tests.** Each task has a test suite written in advance and not shown to either tool. Pass is the suite exiting 0. That's the whole outcome measure.
2. **Behavior — LLM judge.** Save the full transcript of every run. A judge with the fixed rubric below reads it and scores how the tool got there.
3. **Spot-check.** Read about five judged transcripts yourself and confirm the judge isn't drifting. If it is, tighten the rubric rather than labeling everything by hand.

Strip tool names and any identifying markers from transcripts before judging, and use the same judge model for both tools. Otherwise the judge's own vendor preference becomes a variable.

## Tasks

Bind each to a real ticket or a real piece of our code before use. A task with no concrete target isn't ready.

**T1 — Add a tool to an existing agent.**
Tool definition, registration, error handling, tests. Several files that must stay consistent.
*Hidden test:* the agent invokes the new tool end to end and returns the expected result.

**T2 — Fix a failing agent eval.**
Start from a known-failing case. The work is diagnosis, not typing.
*Hidden test:* the eval passes, plus a regression test asserting the specific root cause is gone — so a lucky patch doesn't count.

**T3 — Push a schema change through a data pipeline.**
Add or change a field and carry it through ingestion, transformation, and the consumer. The failure mode is silent nulls, not a crash.
*Hidden test:* the field lands correctly end to end and a null-check assertion on downstream output passes.

**T4 — Change an AWS resource through IaC.**
Something with blast radius: a new Step Functions state, a Lambda with a new IAM policy.
*Hidden test:* the plan applies cleanly in dev and a smoke test hits the resource successfully. Judge-only if we can't automate the apply.

**T5 — Explain an unfamiliar part of the codebase.**
No test can decide this one; it's judged entirely on the rubric, against a short reference answer written in advance by whoever knows the code.

**T6 (optional) — One long unattended stretch.**
A small feature end to end with minimal steering. Most likely to separate the two tools, least visible in short tasks.
*Hidden test:* the feature's acceptance tests pass. The judge's steering count is the real signal here.

## Judge rubric

Fixed questions, same for every transcript. Keep the outputs machine-readable.

- **Invented anything?** Count references to APIs, functions, or AWS resources that don't exist.
- **Steering required.** Count human turns that corrected course, as distinct from turns that added new information.
- **Context retention.** Did it re-read or re-derive things it had already established? Did it contradict its own earlier findings?
- **Self-verification.** Did it run tests or otherwise check its work before declaring done, or did it assert success blindly?
- **Convention fit.** Did the change match surrounding repo patterns, or import a foreign style?
- **Where it stopped being useful** (T6 only). Free text, one sentence.

The first two are the ones most likely to actually differ between tools. If the results are ambiguous everywhere else, those two are what to look at.

## Worth knowing

We can't replicate the public coding-agent leaderboards. They run agents unattended in a container — task in, process exits, repo graded — and Cockpit needs a human in the loop, so it can't be entered. That's a fact about the harnesses, not a judgment about Cockpit. Recorded here so nobody spends a day rediscovering it.

Also expect a fluency gap: we know one tool better than the other and it will show in the results. Within a pod that's a real finding rather than a flaw — if one tool is harder to get fluent in, that's part of what we're deciding.

Sample size is small enough that a narrow difference means nothing. Treat clear, repeated failures as signal and everything else as a tie.
