class MemoryListCache:
    def __init__(self, default_ttl_seconds: int | None = None):
        import time

        self._data: dict[str, list] = {}
        self._expires: dict[str, float | None] = {}
        self._time = time
        self.default_ttl_seconds = default_ttl_seconds

    def _is_expired(self, key: str) -> bool:
        exp = self._expires.get(key)
        return exp is not None and exp < self._time.time()

    def _prune_if_needed(self, key: str) -> None:
        if self._is_expired(key):
            self.delete(key)

    def set(self, key: str, records: list[dict], ttl: int | None = None) -> None:
        expire_at = None
        if ttl is not None or self.default_ttl_seconds is not None:
            expire_at = self._time.time() + (ttl or self.default_ttl_seconds or 0)
        self._data[key] = records
        self._expires[key] = expire_at

    def get(self, key: str) -> list[dict] | None:
        self._prune_if_needed(key)
        return self._data.get(key)

    def append(self, key: str, record: dict) -> None:
        existing = self.get(key) or []
        existing.append(record)
        self.set(key, existing)

    def exists(self, key: str) -> bool:
        self._prune_if_needed(key)
        return key in self._data

    def delete(self, key: str) -> None:
        self._data.pop(key, None)
        self._expires.pop(key, None)

    def clear(self) -> None:
        self._data.clear()
        self._expires.clear()

    def keys(self) -> list[str]:
        return [k for k in list(self._data.keys()) if not self._is_expired(k)]

    def get_length(self, key: str) -> int:
        records = self.get(key)
        return len(records) if records else 0

    def __contains__(self, key: str) -> bool:
        return self.exists(key)

    def __len__(self) -> int:
        return len(self.keys())
