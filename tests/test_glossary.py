from pathlib import Path

import pytest
import yaml

from fablens.glossary import Glossary, load_glossary
from fablens.ingest import TABLE_FILES


@pytest.fixture(scope="module")
def glossary() -> Glossary:
    return load_glossary()


def test_glossary_describes_every_table(glossary: Glossary) -> None:
    assert set(glossary.tables) == set(TABLE_FILES)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Which machine model had the most failures in 2015?", {"failure"}),
        ("Which machines are critical?", {"critical_machine"}),
        ("What is the MTBF of model3?", {"mtbf"}),
        ("How much did the repairs of machine 17 cost?", {"maintenance", "repair_cost"}),
        ("Will machine 42 fail next week?", {"failure_prediction"}),
    ],
)
def test_search_finds_terms_of_demo_questions(
    glossary: Glossary, question: str, expected: set[str]
) -> None:
    assert {term.id for term in glossary.search(question)} == expected


def test_search_matches_whole_words_only(glossary: Glossary) -> None:
    assert glossary.search("Is the vibration of machine 5 alarming?") == []


def write_glossary(path: Path, terms: list[dict]) -> Path:
    tables = {"failures": {"description": "Failures.", "columns": {"machineID": "Machine."}}}
    path.write_text(yaml.safe_dump({"tables": tables, "terms": terms}), encoding="utf-8")
    return path


def term(term_id: str, tables: list[str]) -> dict:
    return {
        "id": term_id,
        "name": term_id,
        "category": "concept",
        "definition": "Test.",
        "tables": tables,
    }


def test_load_glossary_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = write_glossary(tmp_path / "g.yaml", [term("a", []), term("a", [])])

    with pytest.raises(ValueError, match="duplicate term ids: a"):
        load_glossary(path)


def test_load_glossary_rejects_unknown_tables(tmp_path: Path) -> None:
    path = write_glossary(tmp_path / "g.yaml", [term("a", ["costs"])])

    with pytest.raises(ValueError, match="unknown tables: costs"):
        load_glossary(path)
