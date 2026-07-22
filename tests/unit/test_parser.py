"""Unit tests for the structural contract parser."""

from itertools import pairwise

from clausewise.domain import Contract, Section
from clausewise.ingestion.parser import extract_defined_terms, parse_contract

# ruff: noqa: E501  — fixture texts read better unwrapped


def _contract(text: str) -> Contract:
    return Contract(id="k1", title="T", text=text)


def _flatten(sections: tuple[Section, ...]) -> list[Section]:
    out: list[Section] = []
    for s in sections:
        out.append(s)
        out.extend(_flatten(s.children))
    return out


FIXTURE = """PREAMBLE TEXT before any section.

ARTICLE I - DEFINITIONS

1.1 Defined Terms. As used herein, "Confidential Information" means all information disclosed.

1.2 Interpretation. Headings are for convenience only.

ARTICLE II - LICENSE

2.1 Grant. Licensor grants Licensee a license:

(a) to use the Software;

(b) to make copies as follows:

(i) one backup copy; and

(ii) archival copies.

(c) to sublicense.

ARTICLE III - TERMINATION

3.1 Term. This Agreement continues until terminated.
"""


class TestTreeStructure:
    def test_articles_are_roots(self) -> None:
        roots = parse_contract(_contract(FIXTURE))
        assert [r.number for r in roots] == ["ARTICLE I", "ARTICLE II", "ARTICLE III"]
        assert all(r.level == 0 for r in roots)

    def test_decimals_nest_under_articles(self) -> None:
        roots = parse_contract(_contract(FIXTURE))
        art1 = roots[0]
        assert [c.number for c in art1.children] == ["1.1", "1.2"]
        assert all(c.level == 1 for c in art1.children)

    def test_letters_nest_under_decimals_and_romans_under_letters(self) -> None:
        roots = parse_contract(_contract(FIXTURE))
        grant = roots[1].children[0]
        assert grant.number == "2.1"
        assert [c.number for c in grant.children] == ["(a)", "(b)", "(c)"]
        b = grant.children[1]
        assert [c.number for c in b.children] == ["(i)", "(ii)"]

    def test_offsets_are_well_nested(self) -> None:
        roots = parse_contract(_contract(FIXTURE))
        for section in _flatten(roots):
            for child in section.children:
                assert section.char_start <= child.char_start
                assert child.char_start <= child.char_end
                assert child.char_end <= section.char_end

    def test_siblings_do_not_overlap(self) -> None:
        roots = parse_contract(_contract(FIXTURE))
        for section in _flatten(roots):
            kids = section.children
            for left, right in pairwise(kids):
                assert left.char_end <= right.char_start

    def test_section_text_starts_with_its_marker(self) -> None:
        text = FIXTURE
        for section in _flatten(parse_contract(_contract(text))):
            assert (
                text[section.char_start : section.char_end]
                .lstrip()
                .startswith(section.number.split()[0])
            )


class TestAmbiguousI:
    def test_i_after_h_is_a_letter(self) -> None:
        text = "\n".join(f"({c}) clause {c} text here.\n" for c in "abcdefghi")
        roots = parse_contract(_contract(text))
        flat = _flatten(roots)
        i_node = next(s for s in flat if s.number == "(i)")
        # letter rank: sibling of (h), same level, not nested beneath it
        h_node = next(s for s in flat if s.number == "(h)")
        assert i_node.level == h_node.level

    def test_i_after_b_is_roman_and_nests(self) -> None:
        text = "(a) first item.\n\n(b) second item:\n\n(i) sub one.\n\n(ii) sub two.\n"
        roots = parse_contract(_contract(text))
        b_node = next(s for s in _flatten(roots) if s.number == "(b)")
        assert [c.number for c in b_node.children] == ["(i)", "(ii)"]


class TestEdgeCases:
    def test_no_markers_returns_empty(self) -> None:
        assert parse_contract(_contract("just prose, no structure at all.")) == ()

    def test_empty_text(self) -> None:
        assert parse_contract(_contract(" ")) == ()

    def test_roman_top_level_markers(self) -> None:
        text = "I. DEFINITIONS\n\nWords mean things.\n\nII. TERM\n\nOne year.\n\nIV. NOTICES\n\nIn writing.\n"
        roots = parse_contract(_contract(text))
        assert [r.number for r in roots] == ["I", "II", "IV"]

    def test_prose_starting_with_word_i_is_not_a_marker(self) -> None:
        text = "SECTION 1 Intro\n\nI will deliver the goods\nas agreed.\n"
        roots = parse_contract(_contract(text))
        assert len(_flatten(roots)) == 1

    def test_centered_allcaps_with_colon(self) -> None:
        text = "                    WITNESSETH:\n\nWhereas the parties agree.\n\n                    TERMINATION\n\nEither party may terminate.\n"
        roots = parse_contract(_contract(text))
        assert [r.heading for r in roots] == ["WITNESSETH", "TERMINATION"]

    def test_allcaps_heading_is_top_level(self) -> None:
        text = "CONFIDENTIALITY\n\nEach party agrees to keep secrets.\n\nGOVERNING LAW\n\nDelaware law applies.\n"
        roots = parse_contract(_contract(text))
        assert [r.heading for r in roots] == ["CONFIDENTIALITY", "GOVERNING LAW"]

    def test_prose_line_with_number_is_not_a_marker(self) -> None:
        # "30." mid-sentence at line start must not become a section.
        text = "SECTION 1 Payment\n\nPayment is due within\n30 days of invoice.\n"
        roots = parse_contract(_contract(text))
        assert len(_flatten(roots)) == 1  # only SECTION 1

    def test_last_section_extends_to_end_of_text(self) -> None:
        roots = parse_contract(_contract(FIXTURE))
        assert roots[-1].char_end == len(FIXTURE)


class TestTitleCaseFallback:
    UNNUMBERED = (
        "Premium Managed Hosting Agreement\n\n"
        "This is a managed hosting agreement between two parties.\n\n"
        "Included Monthly Services\n\n"
        "Management of SMTP and DNS for the domain.\n\n"
        "Terms of Agreement\n\n"
        "Fees are $175 per month for a period of 12 months.\n"
    )

    def test_unnumbered_contract_uses_titlecase_pass(self) -> None:
        roots = parse_contract(_contract(self.UNNUMBERED))
        headings = [r.heading for r in roots]
        assert "Included Monthly Services" in headings
        assert "Terms of Agreement" in headings

    def test_numbered_contract_never_uses_titlecase(self) -> None:
        # Same Title Case line exists, but the contract has real numbering —
        # first pass finds >=3 markers, so the loose rule must stay off.
        text = (
            "1. Definitions. Terms are defined here.\n\n"
            "Included Monthly Services\n\n"
            "2. Services. Provider will perform.\n\n"
            "3. Fees. Client will pay.\n"
        )
        roots = parse_contract(_contract(text))
        assert [r.number for r in roots] == ["1", "2", "3"]


class TestInlineFallback:
    FLATTENED = (
        "THIS AGREEMENT is made by the parties who agree as follows: "
        '1. DEFINITIONS. For the purposes of this Agreement: 1.1 "Affiliates" means '
        'any entity under common control. 1.2 "Claim" shall have the meaning ascribed '
        "to the term in Section 13.2 of this Agreement. 2. GRANT OF RIGHTS 2.1 Licensor "
        "grants Licensee an exclusive license. 2.2 Licensee accepts the grant. "
        "3. TERMINATION 3.1 Either party may terminate under Section 2.1 above."
    )

    def test_flattened_document_finds_inline_sections(self) -> None:
        roots = parse_contract(_contract(self.FLATTENED))
        assert [r.number for r in roots] == ["1", "2", "3"]
        assert roots[0].heading.startswith("DEFINITIONS")
        assert [c.number for c in roots[1].children] == ["2.1", "2.2"]

    def test_cross_references_are_not_sections(self) -> None:
        roots = parse_contract(_contract(self.FLATTENED))
        flat_numbers = [s.number for s in _flatten(roots)]
        # "Section 13.2" and "Section 2.1 above" are references, not markers.
        assert "13.2" not in flat_numbers
        assert flat_numbers.count("2.1") == 1

    def test_inline_pass_not_used_for_structured_documents(self) -> None:
        text = "1. First\n\nBody one.\n\n2. Second\n\nBody two.\n\n3. Third\n\nBody three.\n"
        roots = parse_contract(_contract(text))
        assert [r.number for r in roots] == ["1", "2", "3"]


class TestDefinedTerms:
    def test_extracts_means_pattern(self) -> None:
        text = 'As used herein, "Confidential Information" means all non-public data.'
        terms = extract_defined_terms(text)
        assert [t.term for t in terms] == ["Confidential Information"]
        assert text[terms[0].char_start] == '"'

    def test_extracts_shall_mean_and_dedupes(self) -> None:
        text = (
            '"Agreement" shall mean this document. Later, "Agreement" means the same thing. '
            '"Effective Date" shall have the meaning set forth above.'
        )
        terms = extract_defined_terms(text)
        assert [t.term for t in terms] == ["Agreement", "Effective Date"]

    def test_no_terms(self) -> None:
        assert extract_defined_terms("nothing defined here") == ()
