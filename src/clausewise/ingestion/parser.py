"""Structural parser: contract text → Section tree.

Legal contracts use wildly inconsistent numbering (ARTICLE IV, Section 2.1,
1., 2.1.3, (a), (i), ALL-CAPS headings). This parser is deliberately
rule-based and *tolerant*: it classifies line-start markers into ranks and
builds a well-nested tree with a stack, accepting imperfect hierarchies
rather than failing.

Invariants:
- Pure function: text in, tree out. No I/O, no mutation of inputs.
- Offsets index into the exact input string (the canonical text) — never
  normalize or re-write the text here.
- Every returned Section satisfies: children lie within the parent's span,
  siblings are ordered and non-overlapping.

Rank model (lower rank = shallower):
    0  ARTICLE X / SECTION 1 (word-prefixed) and ALL-CAPS heading lines
    1  1.  /  1)          (single-component decimal)
    2  1.1                (two components)
    3  1.1.1              (three or more components)
    4  (a) (b) ...        (parenthesized letters)
    5  (i) (ii) ...       (parenthesized romans)

The classic "(i)" ambiguity — letter after (h), or roman i — is resolved by
context: "(i)" is treated as a letter continuation iff the immediately
preceding marker was the letter "(h)".
"""

import re
from dataclasses import dataclass, field

from clausewise.domain import Contract, Section

# --- Marker patterns (all anchored at line start, after optional indent) ---

_ARTICLE_RE = re.compile(
    r"^\s{0,30}(?P<label>(?:ARTICLE|Article|SECTION|Section)\s+"
    # separator: hyphen, en dash (u2013), em dash (u2014), period, colon
    "(?:[IVXLCDM]+|[ivxlcdm]+|\\d+[A-Z]?))(?:\\s*[-\\u2013\\u2014.:]\\s*|\\s+|$)(?P<heading>.*)$"
)
_DECIMAL_RE = re.compile(
    r"^\s{0,8}(?P<label>\d{1,3}(?:\.\d{1,3}){0,4})(?P<punct>[.)])?\s+(?P<heading>\S.*)$"
)
_PAREN_RE = re.compile(r"^\s{0,8}\((?P<label>[a-z]{1,4}|\d{1,3})\)\s+(?P<heading>\S.*)$")
# Top-level roman markers: "I. DEFINITIONS", "IV) Term". Requires punctuation
# after the numeral so a prose line starting with the word "I" never matches.
_ROMAN_TOP_RE = re.compile(r"^\s{0,30}(?P<label>[IVXLCDM]{1,7})[.)]\s+(?P<heading>\S.*)$")
# ALL-CAPS headings may be centered (deep indent) and end with . or :
_ALLCAPS_RE = re.compile(r"^\s{0,30}(?P<heading>[A-Z][A-Z0-9 ,&/'\-()]{2,70})[.:]?\s*$")
# Title Case standalone headings ("Included Monthly Services") — fallback rules
# for unnumbered contracts only (second parse pass), gated on blank-line context.
_TITLE_WORD = r"[A-Z][A-Za-z'&/\-]*"
_TITLE_SMALL = r"(?:of|and|the|for|to|in|on|a|an|with)"
_TITLECASE_RE = re.compile(
    rf"^\s{{0,30}}(?P<heading>{_TITLE_WORD}(?:\s+(?:{_TITLE_SMALL}|{_TITLE_WORD})){{0,7}})\s*$"
)

_ROMAN_CHARS = frozenset("ivxlcdm")
_MAX_HEADING_LEN = 120

# --- Inline markers, for "flattened" documents ------------------------------
# Some CUAD texts lost their line structure in PDF extraction: section markers
# appear mid-line in flowing text. These patterns find them inside lines.
# Top-level is safe because it demands an ALL-CAPS heading after "N. ".
_INLINE_TOP_RE = re.compile(
    r"(?<![A-Za-z0-9.])(?P<label>\d{1,2})\.\s+(?P<heading>[A-Z][A-Z][A-Z ,&/'\-]{2,60})"
)
# Sub-level "1.1 " / "6.3.1 " followed by a capital or opening quote.
_INLINE_SUB_RE = re.compile(
    r"(?<![A-Za-z0-9.(])(?P<label>\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\s+(?=[A-Z\"“(])"
)
# Guard: skip matches that are cross-references ("under Section 6.3.2").
_REF_BEFORE_RE = re.compile(r"(?:[Ss]ections?|[Ss]ubsections?|under|in|to|and|or|of)\s+$|,\s*$")


@dataclass(frozen=True, slots=True)
class DefinedTerm:
    """A defined term ("Confidential Information") and where it's defined."""

    term: str
    char_start: int


@dataclass(slots=True)
class _Node:
    """Mutable builder node; frozen into a domain Section at the end."""

    number: str
    heading: str
    rank: int
    char_start: int
    char_end: int = -1
    children: list["_Node"] = field(default_factory=list)

    def freeze(self, level: int) -> Section:
        return Section(
            number=self.number,
            heading=self.heading[:_MAX_HEADING_LEN],
            level=level,
            char_start=self.char_start,
            char_end=self.char_end,
            children=tuple(child.freeze(level + 1) for child in self.children),
        )


def _is_roman(label: str) -> bool:
    return all(c in _ROMAN_CHARS for c in label)


def _classify(
    line: str,
    prev_label: str | None,
    *,
    allow_titlecase: bool = False,
    prev_blank: bool = True,
    next_blank: bool = True,
) -> tuple[int, str, str] | None:
    """Classify one line as a section marker: (rank, number, heading) or None."""
    m = _ARTICLE_RE.match(line)
    if m:
        return 0, m.group("label").strip(), m.group("heading").strip()

    m = _ROMAN_TOP_RE.match(line)
    if m:
        return 0, m.group("label"), m.group("heading").strip()

    m = _DECIMAL_RE.match(line)
    if m:
        label = m.group("label")
        components = label.count(".") + 1
        if components == 1 and m.group("punct") is None:
            # "30 days of invoice." at line start is prose, not a marker;
            # single-component numbers must be punctuated ("1." or "1)").
            return None
        rank = min(components, 3)
        return rank, label, m.group("heading").strip()

    m = _PAREN_RE.match(line)
    if m:
        label = m.group("label")
        if label.isdigit():
            return 5, f"({label})", m.group("heading").strip()
        if len(label) == 1 and not (label == "i" and prev_label != "(h)"):
            # single letters are letters — except "(i)" not following "(h)"
            return 4, f"({label})", m.group("heading").strip()
        if _is_roman(label):
            return 5, f"({label})", m.group("heading").strip()
        if len(label) <= 2:  # (aa), (bb) — double-letter continuation
            return 4, f"({label})", m.group("heading").strip()
        return None

    m = _ALLCAPS_RE.match(line)
    if m:
        heading = m.group("heading").strip()
        # Require at least two letters and reject pure numerics/dates.
        if sum(c.isalpha() for c in heading) >= 3:
            return 0, heading, heading

    if allow_titlecase and prev_blank and next_blank:
        m = _TITLECASE_RE.match(line)
        if m:
            heading = m.group("heading").strip()
            if len(heading) <= 60 and sum(c.isalpha() for c in heading) >= 4:
                return 0, heading, heading
    return None


def _collect_markers(text: str, *, allow_titlecase: bool) -> list[_Node]:
    lines = text.splitlines(keepends=True)
    stripped = [ln.rstrip("\r\n") for ln in lines]
    markers: list[_Node] = []
    prev_label: str | None = None
    offset = 0
    for i, line in enumerate(lines):
        prev_blank = i == 0 or not stripped[i - 1].strip()
        next_blank = i == len(lines) - 1 or not stripped[i + 1].strip()
        classified = _classify(
            stripped[i],
            prev_label,
            allow_titlecase=allow_titlecase,
            prev_blank=prev_blank,
            next_blank=next_blank,
        )
        if classified is not None:
            rank, number, heading = classified
            markers.append(_Node(number=number, heading=heading, rank=rank, char_start=offset))
            prev_label = number
        offset += len(line)
    return markers


def _collect_inline_markers(text: str) -> list[_Node]:
    """Find section markers inside flowing text (flattened documents)."""
    markers: list[_Node] = []
    for m in _INLINE_TOP_RE.finditer(text):
        markers.append(
            _Node(
                number=m.group("label"),
                heading=m.group("heading").strip(),
                rank=1,
                char_start=m.start(),
            )
        )
    for m in _INLINE_SUB_RE.finditer(text):
        before = text[max(0, m.start() - 14) : m.start()]
        if _REF_BEFORE_RE.search(before):
            continue
        label = m.group("label")
        rank = min(label.count(".") + 1, 3)
        markers.append(_Node(number=label, heading="", rank=rank, char_start=m.start()))
    markers.sort(key=lambda n: n.char_start)
    return markers


def parse_contract(contract: Contract) -> tuple[Section, ...]:
    """Parse canonical text into a tree of top-level Sections.

    Text before the first marker (title block, preamble, recitals) is not
    covered by any Section; the chunker treats uncovered spans explicitly.
    """
    text = contract.text
    markers = _collect_markers(text, allow_titlecase=False)
    if len(markers) < 3:
        # Unnumbered contract — retry with Title Case heading detection.
        # Multi-pass keeps looser rules from polluting well-numbered parses.
        markers = _collect_markers(text, allow_titlecase=True)
    if len(markers) < 3:
        # Flattened document (PDF extraction lost line breaks) — look for
        # markers inside flowing text instead of at line starts.
        inline = _collect_inline_markers(text)
        if len(inline) > len(markers):
            markers = inline

    if not markers:
        return ()

    # Assign char_end: a section ends where the next marker of equal-or-lower
    # rank begins (or at end of text).
    for i, node in enumerate(markers):
        node.char_end = len(text)
        for later in markers[i + 1 :]:
            if later.rank <= node.rank:
                node.char_end = later.char_start
                break

    # Build the tree with a stack; tolerate rank jumps in either direction.
    roots: list[_Node] = []
    stack: list[_Node] = []
    for node in markers:
        while stack and stack[-1].rank >= node.rank:
            stack.pop()
        if stack:
            stack[-1].children.append(node)
        else:
            roots.append(node)
        stack.append(node)

    # Clamp children to parent bounds (defensive: rank model isn't perfect).
    def clamp(node: _Node) -> None:
        for child in node.children:
            child.char_end = min(child.char_end, node.char_end)
            child.char_start = max(child.char_start, node.char_start)
            clamp(child)

    for root in roots:
        clamp(root)

    return tuple(root.freeze(0) for root in roots)


_DEFINED_TERM_RE = re.compile(
    r"[\"“](?P<term>[A-Z][A-Za-z0-9 \-/]{1,60}?)[\"”]\s*(?:\)|,)?\s*"
    r"(?:shall\s+mean|means|shall\s+have\s+the\s+meaning|has\s+the\s+meaning)",
)


def extract_defined_terms(text: str) -> tuple[DefinedTerm, ...]:
    """Find defined terms: '"Term" means/shall mean ...' patterns.

    Deduplicated on first definition; offsets point at the opening quote.
    """
    seen: dict[str, int] = {}
    for m in _DEFINED_TERM_RE.finditer(text):
        term = m.group("term").strip()
        if term and term not in seen:
            seen[term] = m.start()
    return tuple(DefinedTerm(term=t, char_start=pos) for t, pos in seen.items())
