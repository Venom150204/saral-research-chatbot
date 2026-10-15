"""Binary asset storage boundary used by visual processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class BinaryAsset:
    """An immutable image asset; bytes never enter document JSON."""

    asset_hash: str
    media_type: str
    data: bytes
    original_filename: str | None = None

    def __post_init__(self) -> None:
        if not self.asset_hash or not self.data:
            raise ValueError("binary assets require a hash and non-empty data")

    @property
    def byte_size(self) -> int:
        return len(self.data)


class AssetStore(Protocol):
    """Minimal interface that object storage can implement later."""

    def put(self, asset: BinaryAsset) -> str:
        """Persist or deduplicate an asset and return an internal URI."""

    def get(self, asset_hash: str) -> BinaryAsset:
        """Return an authorized asset by its content hash."""


class InMemoryAssetStore:
    """Development/test store with content-hash deduplication."""

    def __init__(self) -> None:
        self._assets: dict[str, BinaryAsset] = {}

    def put(self, asset: BinaryAsset) -> str:
        existing = self._assets.get(asset.asset_hash)
        if existing is not None and existing.data != asset.data:
            raise ValueError("asset hash collision")
        self._assets.setdefault(asset.asset_hash, asset)
        return f"memory://assets/{asset.asset_hash}"

    def get(self, asset_hash: str) -> BinaryAsset:
        try:
            return self._assets[asset_hash]
        except KeyError as exc:
            raise KeyError(f"asset not found: {asset_hash}") from exc

    def __len__(self) -> int:
        return len(self._assets)
