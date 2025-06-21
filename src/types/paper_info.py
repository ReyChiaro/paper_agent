from dataclasses import dataclass
from pathlib import Path


@dataclass
class Author:

    name: str
    affiliation: str
    email: str


@dataclass
class Citation:

    title: str
    authors: list[Author]
    platform: str
    year: int


@dataclass
class Contents:

    introduction: str | None
    related_work: str | None
    methodology: str | None
    experiments: str | None
    results_discussion: str | None
    conclusion: str | None
    limitations: str | None


@dataclass
class Paper:

    # File properties
    fileid: str
    filename: str
    filesize: int

    # The originsl paper contents (static)
    markdown: Path

    # The original paper contents (runtime)
    contents: Contents
    contents_json: Path

    # Paper properties
    title: str
    authors: list[Author]
    abstract: str
    tags: list[str]

    # Short contents of the paper
    ranking: int
    summary: str
    comment: str

    
    