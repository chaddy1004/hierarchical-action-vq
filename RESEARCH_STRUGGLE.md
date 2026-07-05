# RESEARCH_STRUGGLE.md

A doc to reread when the fear hits. Not about havq specifically — about the
two recurring fears, why they're miscalibrated, and the protocol that converts
each one into work. Written 2026-07-05, mid-project, right after a literature
sweep triggered fear #1.

---

## Fear 1: "Someone already took my idea"

**The reframe.** Ideas that nobody is near are usually ideas that don't work
or don't matter. Finding three groups converging on your neighborhood means
you independently identified where the field is going — *that is the skill*,
and it repeats. The fear comes from a mental model where the **idea** is the
contribution. It isn't. The **claim + evidence + positioning** is the
contribution, and it's much harder to collide on that than on an idea.
Simultaneity is the norm in research history, not the exception.

**The empirical pattern (verified today, 2026-07-05):** the dread almost
never survives close contact with the actual abstracts. Three "they did my
idea" papers turned out to be: a flat segmenter with no hierarchy and no
benchmarks; a fixed two-level quantizer on skeletons; and a games/RL paper.
The exact claim was untouched. This will usually be the outcome.

**The protocol (do this instead of ruminating):**

1. Read the scary paper's abstract (and intro if needed). Actually read it.
2. Write ONE sentence: what question do they answer?
3. Write ONE sentence: what question do I answer?
4. Write the differentiation sentence: "Unlike X, which does A for purpose B,
   we do C to answer D."
5. Put all three into `RELATED_WORKS.md` and decide: citation, baseline,
   or imported ablation. Every scary paper becomes one of those three things.

If after step 3 the questions are *actually identical* — rare — then you've
learned it early and cheap, and the pivot menu (different claim, different
eval, their-method-as-component) is still yours. Either way the fear has been
converted into a related-work entry instead of a lost week.

**Corollary — velocity beats brooding.** When the neighborhood is crowded,
the risk is not that the idea is taken; it's moving slowly enough that it
becomes taken. Cut scope, run the kill-test experiments first, plant a flag
(arXiv/workshop) when the core result lands.

## Fear 2: "I don't know if it will work / if the results will be publishable"

**You cannot know in advance, and neither can anyone else.** The people with
strong papers are not better at predicting outcomes — they're better at
**engineering for information**: making every week produce evidence that
either advances the claim or kills it cheaply. The fear is a demand for
certainty that doesn't exist; the antidote is a process that makes *both*
outcomes valuable.

**The process (this repo is the live example):**

- **Decompose the hypothesis** into independently testable parts (H1, H2,
  H3 in `RESEARCH.md`). "Will it work?" is unanswerable; "does H2 hold at
  1-second scale?" is a week of work.
- **Order experiments by information-per-hour**, cheapest kill-test first
  (`PLAN.md` Exp 1 needs no training at all).
- **Write the decision gate before running.** If you only decide what
  "success" means after seeing the numbers, you will negotiate with yourself.
- **A clean negative with a diagnosis is a result.** v1 "failed" — and
  produced the exact insight (temporal scale, flicker) that defines the next
  two experiments. Archived failures become ablations and appendix tables;
  nothing gated properly is wasted.

**What "publishable" actually means.** Papers are not accepted because the
idea worked; they're accepted because the evidence **teaches the reader
something they didn't know**. A strong paper = one crisp claim + an
evaluation that isolates it + baselines that are fair-but-strong + ablations
showing which pieces matter + positioning that names the neighbors precisely.
Notice: every item on that list is *engineerable*. None of it requires the
initial idea to have been magic. SOTA numbers are one way to teach something;
they are not the only way, and hierarchy-as-evaluation was chosen precisely
because it teaches something benchmarks-chasing papers don't.

## The operating loop (each file is an antidote to a failure mode)

| File | Failure mode it prevents |
|---|---|
| `PLAN.md` (goal + next 1–2 experiments + gates) | "lost in the sauce" — planning 10 steps past current evidence |
| `RESEARCH.md` (hypothesis decomposed, pitfalls with guards) | vague hypotheses that can't fail, self-deceiving evals |
| `RELATED_WORKS.md` (dated sweeps + differentiation sentences) | fear #1 rumination; surprise at writing time |
| `archive/` + git snapshots | sunk-cost dragging of dead code; losing the lesson with the failure |
| Matched baselines on every metric | shipping numbers that are secretly random (v1's lesson) |

## When the fear hits mid-project — checklist

1. Name which fear it is (#1 or #2). They have different antidotes.
2. Fear #1 → run the 5-step protocol above. Timebox: one hour.
3. Fear #2 → reread `PLAN.md`. Is there a next experiment with a written
   gate? If yes: the plan already absorbs the uncertainty — go run it. If
   no: writing that gate IS the work; do that instead of worrying.
4. Do not redesign the project after 6pm. Log the worry, decide in the
   morning.
5. Remember the base rate: v1 "failed" and produced the field-defining
   insight for the project. The last "they scooped me" scare dissolved on a
   one-hour read. Update on your own history.
