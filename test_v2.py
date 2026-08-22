"""
Checks for the V2 logic that can be exercised without a network call.

Deliberately no framework and no fixtures — one file, `python test_v2.py`.
What is covered is what fails SILENTLY if it breaks: a router that
misclassifies, a JSON extractor that drops an escape, a numeric check that
misses a wrong figure, a marker sort that puts s-8-10 before s-8-2. None of
those raise; they just quietly produce a worse answer.

Not covered here (needs a live corpus or provider, so it belongs in eval/):
retrieval quality, template output against real provisions, and anything
that costs an embedding request.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend import numeric, streaming, temporal, understanding   # noqa: E402
from backend.graph_store import _marker_key                       # noqa: E402
from backend.indexing import Chunk                                # noqa: E402
from backend.ratelimit import RateLimiter                         # noqa: E402

PASS = FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ok    {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f"  -> {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
print("\nstreaming — JSON answer extraction")

def stream(fragments):
    s = streaming.AnswerStream()
    out = []
    for f in fragments:
        out.extend(s.feed(f))
    return "".join(out), s.finish()


tricky = 'He said "250 crore".\nUse {this} \\ that. Cost ₹250.'
raw = json.dumps({"answer": tricky,
                  "citations": [{"node_id": "pen-1", "quote": "q"}],
                  "confidence": "high"})

emitted, done = stream([raw])
check("streams quotes/newlines/backslashes/braces/unicode", emitted == tricky,
      repr(emitted))
check("parses citations", done["citations"] == [{"node_id": "pen-1", "quote": "q"}])
check("reports structured", done["structured"] is True)

# Every possible fragment boundary must give the same result. This is the
# check that actually protects the extractor: a provider can split its stream
# anywhere, including mid-escape.
splits_ok = all(stream([raw[:i], raw[i:]])[0] == tricky for i in range(1, len(raw)))
check(f"identical across all {len(raw) - 1} fragment boundaries", splits_ok)

emitted, done = stream(["Section 8(5) requires safeguards."])
check("non-compliant output degrades to free text",
      done["structured"] is False and done["answer"].startswith("Section 8(5)"))

emitted, done = stream(['{"answer": "half wri'])
check("truncated stream keeps what streamed",
      emitted == "half wri" and done["structured"] is False)

emitted, _ = stream(['x```json\n' + json.dumps({"answer": "F", "citations": []})])
check("survives a code fence and preamble", emitted == "F")


# --------------------------------------------------------------------------- #
print("\nnumeric — figures must exist in the evidence")

evidence = ["May extend to two hundred and fifty crore rupees.",
            "A Data Fiduciary shall retain logs for one year."]
amounts = ["May extend to two hundred and fifty crore rupees."]

claims = numeric.check("The penalty may extend to 250 crore rupees.",
                       evidence, amounts)
check("correct figure is supported",
      claims and all(c.verdict == "supported" for c in claims),
      str([c.to_dict() for c in claims]))

# The V1 error this exists to catch.
claims = numeric.check("The penalty may extend to 200 crore rupees.",
                       evidence, amounts)
check("WRONG figure is flagged", numeric.has_contradiction(claims),
      str([c.to_dict() for c in claims]))

claims = numeric.check("Logs must be kept for one year.", evidence, amounts)
check("duration in evidence is supported",
      not numeric.has_contradiction(claims))

claims = numeric.check("Logs must be kept for five years.", evidence, amounts)
check("WRONG duration is flagged", numeric.has_contradiction(claims))

claims = numeric.check("See section 8(5) and rule 6(1).", evidence, amounts)
check("citation numbers are not treated as figures",
      not numeric.has_contradiction(claims),
      str([c.to_dict() for c in claims]))

check("word and digit forms are the same value",
      numeric.extract("two hundred and fifty crore")[0][1]
      == numeric.extract("250 crore")[0][1])


# --------------------------------------------------------------------------- #
print("\nunderstanding — routing (regex tier, no embeddings)")

def route(question):
    return understanding.understand(question, None, None)


check("GDPR question is foreign", route("What does Article 33 of the GDPR require?").should_abstain)
check("HIPAA question is foreign", route("What does HIPAA say about patient data?").should_abstain)
check("DPDP question is not foreign",
      not route("What is the penalty under the DPDP Act for a data breach?").should_abstain)
check("Indian marker beats shared vocabulary",
      not route("What must a Data Fiduciary do about a personal data breach?").should_abstain)

check("penalty intent", route("What's the fine for a data leak?").intent == "penalty")
check("retention intent", route("How long can we keep customer records?").intent == "retention")
check("definition intent", route("What is a Data Principal?").intent == "definition")
check("direct lookup intent", route("What does section 8 say?").intent == "direct_lookup")
check("obligation intent",
      route("What must we do when a breach occurs?").intent == "obligation")

check("section reference extracted",
      understanding.provision_reference("What does section 8 say?") == "s-8")
check("sub-section reference extracted",
      understanding.provision_reference("Tell me about section 8(5)") == "s-8-5")
check("rule reference extracted",
      understanding.provision_reference("Show me rule 6(1)") == "r-6-1")
check("no false provision reference",
      understanding.provision_reference("What is a Data Principal?") == "")

# A named provision plus a penalty question is a penalty question.
plan = route("What is the penalty under section 8?")
check("penalty beats bare lookup when both match", plan.intent == "penalty")

check("anaphora detected", route("And its penalty?").has_anaphora)
check("standalone question has no anaphora",
      not route("What is the penalty for a data breach under the DPDP Act?").has_anaphora)


# --------------------------------------------------------------------------- #
print("\ngraph_store — document ordering")

ordered = sorted(["s-8-10", "s-8-2", "s-8-1", "s-8-11", "s-8-9"], key=_marker_key)
check("s-8-10 sorts after s-8-9, not after s-8-1",
      ordered == ["s-8-1", "s-8-2", "s-8-9", "s-8-10", "s-8-11"], str(ordered))
check("clause letters order alphabetically",
      sorted(["r-6-1-c", "r-6-1-a", "r-6-1-b"], key=_marker_key)
      == ["r-6-1-a", "r-6-1-b", "r-6-1-c"])


# --------------------------------------------------------------------------- #
print("\ntemporal — commencement")

commencement = temporal.load(Path(__file__).resolve().parent / "data" / "commencement.yaml")
if commencement.dates:
    check("rule 4 not in force in 2025", not commencement.in_force_on("r-4", date(2025, 12, 1)))
    check("rule 4 in force in 2027", commencement.in_force_on("r-4", date(2027, 1, 1)))
    check("rule 1 in force on publication", commencement.in_force_on("r-1", date(2025, 11, 14)))
    check("rule 6 not in force in 2026", not commencement.in_force_on("r-6", date(2026, 6, 1)))
    check("clause inherits its rule's date",
          not commencement.in_force_on("r-6-1-a", date(2026, 6, 1)))
    check("Act provisions default to in force",
          commencement.in_force_on("s-8-5", date(2025, 1, 1)))
else:
    print("  skip  commencement.yaml not found in backend/data/")

check("as_of parses an ISO date",
      temporal.resolve_as_of("2026-01-15") == date(2026, 1, 15))
check("as_of falls back to today on junk",
      temporal.resolve_as_of("not-a-date") == date.today())


# --------------------------------------------------------------------------- #
print("\nratelimit")

limiter = RateLimiter(limit=3, window=3600)
results = [limiter.allow("user-a")[0] for _ in range(4)]
check("allows up to the limit then blocks", results == [True, True, True, False],
      str(results))
check("a different user is unaffected", limiter.allow("user-b")[0])
check("retry_after is reported", limiter.allow("user-a")[2] > 0)
check("limit 0 disables the limiter", RateLimiter(limit=0, window=60).allow("x")[0])
limiter.sweep()
check("sweep keeps active users", limiter.tracked_users == 2)


# --------------------------------------------------------------------------- #
print("\nindexing — embedding text budget")

chunk = Chunk(id="c", node_id="s-8-5", kind="SubSection", label="Section 8(5)",
              verbatim="A" * 6000, headnote="Security safeguards",
              plain_english="You must protect the data you hold.")
text = chunk.embedding_text(1800)
check("embedding text respects the budget", len(text) <= 1800, str(len(text)))
check("semantic layer survives truncation",
      "Security safeguards" in text and "protect the data" in text)

short = Chunk(id="c2", node_id="s-1", kind="Section", label="Section 1",
              verbatim="Short text.", headnote="Title")
check("short chunk keeps its verbatim text",
      "Short text." in short.embedding_text(1800))


# --------------------------------------------------------------------------- #
print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
