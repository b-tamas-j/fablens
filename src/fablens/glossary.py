"""Business glossary: table descriptions, KPI definitions and known data gaps."""

import re
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

GLOSSARY_PATH = Path(__file__).with_name("glossary.yaml")


class Table(BaseModel):
    """Business description of a table and its columns."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    description: str
    columns: dict[str, str]


class Term(BaseModel):
    """A glossary entry: a concept, a KPI definition or a known data gap."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str
    category: Literal["concept", "kpi", "missing_data"]
    definition: str
    aliases: tuple[str, ...] = ()
    tables: tuple[str, ...] = ()

    def matches(self, query: str) -> bool:
        """Whether the query contains the id, the name or an alias as a whole word."""
        text = query.lower()
        return any(
            re.search(rf"(?<!\w){re.escape(key.lower())}(?!\w)", text)
            for key in (self.id, self.name, *self.aliases)
        )


class Glossary(BaseModel):
    """All tables and terms of the glossary file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tables: dict[str, Table]
    terms: tuple[Term, ...]

    @model_validator(mode="after")
    def check_references(self) -> Self:
        """Reject duplicate term ids and terms that reference unknown tables."""
        ids = [term.id for term in self.terms]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate term ids: {', '.join(duplicates)}")
        unknown = sorted({t for term in self.terms for t in term.tables} - self.tables.keys())
        if unknown:
            raise ValueError(f"terms reference unknown tables: {', '.join(unknown)}")
        return self

    def search(self, query: str) -> list[Term]:
        """Terms that the query mentions."""
        return [term for term in self.terms if term.matches(query)]


def load_glossary(path: Path = GLOSSARY_PATH) -> Glossary:
    """Load and validate the glossary YAML file."""
    return Glossary.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
