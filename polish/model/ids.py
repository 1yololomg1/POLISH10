"""Identity keys for a file, a dataset, a curve, and a sample.

Implements D30 (with F9) and D46. A mnemonic is not an identity: LAS allows
duplicate mnemonics, so a curve is (dataset id, column index).

Structure
---------
FileHash
    SHA-256 of the input file, streamed.
DatasetId
    Source ids are the file hash. Child ids add a new token and the parent id.
CurveRef, SampleRef
    Column and sample indexes into that dataset. All four are frozen and hashable.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

# Streaming read size. Not a petrophysical threshold.
_HASH_CHUNK_BYTES = 65536
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def _nonempty(text: str) -> bool:
    return isinstance(text, str) and text.strip() != ""


@dataclass(frozen=True)
class FileHash:
    """SHA-256 hex digest of an input file. D46, D63.

    The digest is lowercase hex, which is the stable serialisation.
    """

    hex_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.hex_digest, str) or _SHA256_HEX.fullmatch(self.hex_digest) is None:
            raise ValueError("FileHash must be 64 lowercase hex characters (SHA-256).")

    def __str__(self) -> str:
        return self.hex_digest

    @classmethod
    def from_path(cls, path: Path | str) -> FileHash:
        """Hash the file in streaming chunks. The file is opened read-only. D17, C5."""

        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            while True:
                chunk = handle.read(_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
        return cls(digest.hexdigest())


@dataclass(frozen=True)
class DatasetId:
    """Identity of one dataset. D30, F9, OD-03.

    A source dataset's token is the input file's SHA-256. A child dataset has
    its own token plus the parent DatasetId. This module does not resample.
    """

    token: str
    parent_id: DatasetId | None = None

    def __post_init__(self) -> None:
        if self.parent_id is None:
            if not isinstance(self.token, str) or _SHA256_HEX.fullmatch(self.token) is None:
                raise ValueError("A source DatasetId token must be the file SHA-256 hex.")
        else:
            if not isinstance(self.parent_id, DatasetId):
                raise TypeError("parent_id must be a DatasetId or None")
            if not _nonempty(self.token):
                raise ValueError("A child DatasetId needs a non-empty token.")

    def __str__(self) -> str:
        return self.serialise()

    @classmethod
    def for_source(cls, file_hash: FileHash) -> DatasetId:
        """Source dataset id, derived from the file hash. D30, F9, D46."""

        if not isinstance(file_hash, FileHash):
            raise TypeError("file_hash must be a FileHash")
        return cls(token=file_hash.hex_digest, parent_id=None)

    @classmethod
    def for_child(cls, parent_id: DatasetId, token: str) -> DatasetId:
        """A new id that records its parent. OD-03. No resampling is performed."""

        if not isinstance(parent_id, DatasetId):
            raise TypeError("parent_id must be a DatasetId")
        return cls(token=token, parent_id=parent_id)

    def serialise(self) -> str:
        """Stable text form. A child token is hex-encoded so the parent parse is unique."""

        if self.parent_id is None:
            return f"source:{self.token}"
        encoded = self.token.encode("utf-8").hex()
        return f"child:{encoded}:{self.parent_id.serialise()}"

    @classmethod
    def parse(cls, text: str) -> DatasetId:
        """Inverse of ``serialise``. D30."""

        if not isinstance(text, str):
            raise TypeError("dataset id text must be a str")
        if text.startswith("source:"):
            return cls(token=text[len("source:") :], parent_id=None)
        if text.startswith("child:"):
            rest = text[len("child:") :]
            encoded, separator, parent_text = rest.partition(":")
            if separator != ":" or encoded == "":
                raise ValueError(f"Cannot parse dataset id {text!r}")
            try:
                token = bytes.fromhex(encoded).decode("utf-8")
            except ValueError as exc:
                raise ValueError(f"Cannot parse dataset id {text!r}") from exc
            return cls(token=token, parent_id=cls.parse(parent_text))
        raise ValueError(f"Cannot parse dataset id {text!r}")


@dataclass(frozen=True)
class CurveRef:
    """One curve in one dataset. D30, F9.

    ``column_index`` is the 0-based position in that dataset's curve tuple.
    The mnemonic is not part of the key.
    """

    dataset_id: DatasetId
    column_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, DatasetId):
            raise TypeError("dataset_id must be a DatasetId")
        if type(self.column_index) is not int or self.column_index < 0:
            raise ValueError("column_index must be an int >= 0")

    def __str__(self) -> str:
        return f"curve:{self.dataset_id.serialise()}#{self.column_index}"


@dataclass(frozen=True)
class SampleRef:
    """One sample on one curve. D43. The overlay store keys on this."""

    curve_ref: CurveRef
    sample_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.curve_ref, CurveRef):
            raise TypeError("curve_ref must be a CurveRef")
        if type(self.sample_index) is not int or self.sample_index < 0:
            raise ValueError("sample_index must be an int >= 0")

    def __str__(self) -> str:
        return f"sample:{self.curve_ref}@{self.sample_index}"
