from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterator, List, Sequence

from torch.utils.data import Sampler


class SessionBatchSampler(Sampler[List[int]]):
    """Batch sampler grouping indices by session.

    Each yielded batch contains indices from a single session. The session
    order and the order of indices within each session are shuffled every
    iteration when ``shuffle`` is ``True``.
    """

    def __init__(self, sessions: Sequence[str], batch_size: int, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._session_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, session in enumerate(sessions):
            self._session_to_indices[session].append(idx)
        self._sessions = list(self._session_to_indices.keys())

    def __iter__(self) -> Iterator[List[int]]:
        sessions = self._sessions.copy()
        if self.shuffle:
            random.shuffle(sessions)
        for session in sessions:
            indices = self._session_to_indices[session]
            if self.shuffle:
                random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                yield indices[i : i + self.batch_size]

    def __len__(self) -> int:
        total = 0
        for indices in self._session_to_indices.values():
            total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total
