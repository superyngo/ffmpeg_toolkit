# 0001 — Existing Chinese-language debug record is kept, not translated

Date: 2026-11-20

## Context

repo-docs 4 expects documents in English. One record in this repo is substantially Chinese:
[`docs/debug/2026-01-24-motion-detection-semantics.md`](../debug/2026-01-24-motion-detection-semantics.md)
(moved from root `MOTION_DETECTION_FIX.md`). It is a resolved, frozen record. Deviations from the
documentation system are recorded as an ADR (repo-docs 15).

## Decision

Keep that debug record in Chinese as written. Scope: that file only. Every new document, and all
living docs (`CONTEXT.md`, `docs/reference/`, `docs/plan/BACKLOG.md`), are written in English.
The Chinese `## [0.2.1]` CHANGELOG section is released history and is not edited (no ADR needed).

## Alternatives rejected

- Translate on move: rewrites a frozen record's content; the owner chose to keep Chinese docs
  (rollout decision 2026-11-19).

## Consequences

English-only readers need machine translation for that one record; the glossary and
`architecture.md` cover the same semantics in English.
