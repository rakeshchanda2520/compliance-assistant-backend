"""
Making a real question tractable before anything tries to classify it.

**The governing assumption of this module: the user's phrasing will not match
our data.** People do not write "processing of personal data of children" —
they write "a kid signed up on our app", "child user hai to kya karna hoga",
"CHILDREN DATA???", or a 300-word paragraph about their onboarding flow with
the actual question in the middle.

Regex tier 1 exists because it is free and instant when it hits. It is NOT the
primary path and must never be treated as one — on real traffic it misses far
more than it catches. Everything here exists so that a miss is cheap and the
embedding tier gets clean input.

Nothing in this module changes the question that reaches the model or the
answer. Normalisation feeds *classification and retrieval only*; the verbatim
question is what gets logged, embedded for the record, and shown back.
"""
from __future__ import annotations

import re
import unicodedata

# Devanagari, and the Latin range used by Hinglish. A question in Hindi script
# is a real input: the product is Indian and its users code-switch freely.
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# Romanised Hindi that carries interrogative or obligation force. Mapped to the
# English the corpus and the exemplars are written in. Deliberately small and
# high-precision — a big fuzzy map would mistranslate English words that happen
# to collide (e.g. "kya" vs "kyc").
_HINGLISH = {
    "kya": "what", "kyaa": "what", "kaise": "how", "kaise": "how",
    "kitna": "how much", "kitne": "how many", "kab": "when", "kahan": "where",
    "kaun": "who", "kyun": "why", "kyon": "why",
    "karna": "do", "karna hoga": "must do", "hoga": "will",
    "chahiye": "should", "sakte": "can", "sakta": "can",
    "jurmana": "penalty", "saza": "penalty", "niyam": "rule",
    "kanoon": "law", "adhiniyam": "act", "dhara": "section",
    "aankda": "data", "jaankari": "information", "suraksha": "security",
    "bachcha": "child", "bachche": "children", "sehmati": "consent",
}

# Common misspellings of the terms this corpus turns on. A typo in a defined
# term is the difference between the definition template firing and not.
_TYPOS = {
    "fiduciary": ("fiduciory", "fidiciary", "fiduciry", "feduciary",
                  "fidusiary", "fiduaciary"),
    "principal": ("principle", "prinicipal", "principel"),
    "personal": ("personnel", "persnal", "personel"),
    "consent": ("concent", "consnet", "consant"),
    "breach": ("breech", "brech", "braech"),
    "penalty": ("penality", "penalt", "pentalty"),
    "retention": ("retension", "retenion"),
    "grievance": ("greivance", "grievence"),
    "significant": ("significiant", "signficant"),
    "processing": ("procesing", "proccessing"),
    "obligations": ("obligatons", "obligtions", "obligaitons"),
    "notification": ("notifcation", "notifiction"),
    "safeguards": ("safegaurds", "safeguads"),
    "erasure": ("erasre", "errasure"),
    "compliance": ("complaince", "compliace", "compliancy"),
}
_TYPO_LOOKUP = {bad: good for good, bads in _TYPOS.items() for bad in bads}

# Filler that carries no retrieval signal but does dilute BM25 term frequency
# and push a rambling question past the embedder's input cap.
_PREAMBLE = re.compile(
    r"^\s*(?:hi|hey|hello|namaste|good\s+(?:morning|afternoon|evening))\b[\s,.!-]*"
    r"|^\s*(?:i\s+(?:have|had)\s+a\s+question|quick\s+question|"
    r"can\s+(?:you|u)\s+(?:please\s+)?(?:tell|help|explain)(?:\s+me)?|"
    r"i\s+wanted\s+to\s+(?:know|ask)|"
    r"please\s+(?:tell|help|explain)(?:\s+me)?)\b[\s,.:-]*",
    re.I)

_POLITE_TAIL = re.compile(
    r"[\s,]*(?:thanks?(?:\s+(?:a\s+lot|so\s+much|in\s+advance))?|"
    r"thank\s+you|pls|plz|please|tia|regards?)[\s.!]*$", re.I)

# Chat shorthand. Expanded because the exemplars and the corpus are written out.
_SHORTHAND = {
    r"\bu\b": "you", r"\bur\b": "your", r"\br\b": "are", r"\bpls\b": "please",
    r"\bplz\b": "please", r"\bthx\b": "thanks", r"\bwat\b": "what",
    r"\bwot\b": "what", r"\bhw\b": "how", r"\bcz\b": "because",
    r"\bbcz\b": "because", r"\bcud\b": "could", r"\bshud\b": "should",
    r"\bwud\b": "would", r"\bdnt\b": "don't", r"\bcnt\b": "can't",
    r"\bcompany's\b": "company", r"\bco\.\b": "company",
    r"\bdpdpa\b": "DPDP Act", r"\bdpdp act\b": "DPDP Act",
    r"\bpii\b": "personal data", r"\bdf\b": "Data Fiduciary",
    r"\bdp\b": "Data Principal", r"\bsdf\b": "Significant Data Fiduciary",
    r"\bdpb\b": "Data Protection Board", r"\bcm\b": "Consent Manager",
}

# Hard ceiling on what reaches a classifier or an embedder. Long rambling
# questions are real; the signal in them is almost never in the last paragraph.
MAX_CLASSIFY_CHARS = 1200


def script_of(text: str) -> str:
    """"devanagari", "latin", or "mixed" — recorded so a routing miss on a
    non-English question is attributable rather than mysterious."""
    has_dev = bool(_DEVANAGARI.search(text))
    has_lat = bool(re.search(r"[A-Za-z]", text))
    if has_dev and has_lat:
        return "mixed"
    return "devanagari" if has_dev else "latin"


def strip_noise(text: str) -> str:
    """Greetings, sign-offs and filler. Applied before anything else.

    Looped, because these stack: "Hi, can you please tell me ..." is a
    greeting AND a request preamble, and one pass anchored at `^` only ever
    removes the outermost. Bounded so a pathological input cannot spin.
    """
    out = text
    for _ in range(4):
        stripped = _POLITE_TAIL.sub("", _PREAMBLE.sub("", out)).strip()
        if stripped == out.strip():
            break
        out = stripped
    # Never return an empty question: if the whole message was pleasantries,
    # the original is more useful to a classifier than "".
    return out.strip() or text.strip()


def expand_shorthand(text: str) -> str:
    out = text
    for pattern, full in _SHORTHAND.items():
        out = re.sub(pattern, full, out, flags=re.I)
    return out


def fix_typos(text: str) -> str:
    """Only exact whole-word matches from a hand-checked table.

    Deliberately not fuzzy: edit-distance correction against a legal
    vocabulary confidently turns "principle" into "principal" in a sentence
    that meant "principle", and a wrong defined term routes to a wrong
    template. A closed table cannot surprise anyone.
    """
    def swap(m: re.Match) -> str:
        word = m.group(0)
        fixed = _TYPO_LOOKUP.get(word.lower())
        if not fixed:
            return word
        return fixed.upper() if word.isupper() else (
            fixed.capitalize() if word[0].isupper() else fixed)
    return re.sub(r"\b[A-Za-z']+\b", swap, text)


def translate_hinglish(text: str) -> str:
    """Romanised-Hindi interrogatives and legal nouns → English.

    Additive in effect: the English words are appended to the question rather
    than replacing it, so a mistranslation cannot delete a term the user
    actually typed. Same union-safety property required of query rewriting.
    """
    words = re.findall(r"\b[a-z]+\b", text.lower())
    hits = [_HINGLISH[w] for w in words if w in _HINGLISH]
    return " ".join(dict.fromkeys(hits))       # de-duped, order preserved


def collapse(text: str) -> str:
    """Whitespace, quotes, dashes and repeated punctuation.

    "CHILDREN DATA???" and "children data?" must classify identically. Casing
    is left alone — `Data Fiduciary` capitalisation is a real signal that the
    user means the defined term.
    """
    out = unicodedata.normalize("NFKC", text)
    out = out.replace("’", "'").replace("‘", "'")
    out = out.replace("“", '"').replace("”", '"')
    out = re.sub(r"[–—]", "-", out)
    # Any RUN of terminal punctuation collapses to its first mark. Matching
    # only repeats of the SAME character left "??!!" as "?!", and mixed runs
    # are at least as common as repeated ones in real questions.
    out = re.sub(r"([?!.])[?!.]+", r"\1", out)
    out = re.sub(r",{2,}", ",", out)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def shouting(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 8 and sum(c.isupper() for c in letters) / len(letters) > 0.8


class Normalized:
    """The forms of one question, each for a specific consumer.

    Kept as distinct fields rather than one "cleaned" string, because the
    consumers genuinely differ: BM25 wants the widest surface, the classifier
    wants the tightest, and the audit log wants exactly what was typed.
    """

    __slots__ = ("original", "clean", "classify", "retrieval", "script",
                 "was_shouting", "truncated")

    def __init__(self, original: str) -> None:
        self.original = original
        self.script = script_of(original)
        self.was_shouting = shouting(original)

        clean = collapse(strip_noise(original))
        if self.was_shouting:
            # All-caps defeats the capitalisation signal for defined terms; a
            # lowercase form classifies better than a shouted one.
            clean = clean.lower()
        clean = fix_typos(expand_shorthand(clean))
        self.clean = clean

        # What the classifier and embedder see: bounded, because the signal is
        # rarely in the tail of a 300-word message.
        self.truncated = len(clean) > MAX_CLASSIFY_CHARS
        self.classify = clean[:MAX_CLASSIFY_CHARS]

        # What the lexical retriever sees: the widest surface. Hinglish
        # translations are APPENDED, never substituted.
        extra = translate_hinglish(original)
        self.retrieval = f"{clean} {extra}".strip() if extra else clean

    def to_dict(self) -> dict:
        return {"script": self.script, "shouting": self.was_shouting,
                "truncated": self.truncated,
                "changed": self.clean != self.original.strip()}


def normalize(question: str) -> Normalized:
    return Normalized(question)
