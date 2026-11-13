"""Answer counts and rule-based evidence states (the human-centered mechanism's first half).

Definitions are fixed in docs/DESIGN_LOCK.md section 2:
  * 5 true-frame questions (Yes is consistent with the claim being true) and 5 false-frame questions.
  * An answer SUPPORTS the claim if (true-frame and Yes) or (false-frame and No);
    it REFUTES the claim if (true-frame and No) or (false-frame and Yes); otherwise it is Unverified.
  * State(m): Supports if sup>=m and ref<m; Refutes if ref>=m and sup<m; Mixed if sup>=m and ref>=m; else Insufficient.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

FRAMES = ("true", "false")
ANSWERS = ("Yes", "No", "Unverified")
STATES = ("Supports", "Refutes", "Mixed", "Insufficient")
FEATURE_NAMES = ["t_yes", "t_no", "t_unv", "f_yes", "f_no", "f_unv"]


@dataclass(frozen=True)
class Counts:
    t_yes: int
    t_no: int
    t_unv: int
    f_yes: int
    f_no: int
    f_unv: int

    def as_list(self) -> list[int]:
        return [self.t_yes, self.t_no, self.t_unv, self.f_yes, self.f_no, self.f_unv]

    @property
    def sup(self) -> int:
        return self.t_yes + self.f_no

    @property
    def ref(self) -> int:
        return self.t_no + self.f_yes

    @property
    def unv(self) -> int:
        return self.t_unv + self.f_unv

    @property
    def frames_sum_ok(self) -> bool:
        """Each frame should hold exactly 5 answers (violations are logged by the caller)."""
        return (self.t_yes + self.t_no + self.t_unv == 5) and (self.f_yes + self.f_no + self.f_unv == 5)


def counts_from_answers(answers: Iterable[tuple[str, str]]) -> Counts:
    """answers: iterable of (frame, label) with frame in {'true','false'} and label in {'Yes','No','Unverified'}."""
    c = {k: 0 for k in FEATURE_NAMES}
    for frame, label in answers:
        if frame not in FRAMES:
            raise ValueError(f"bad frame {frame!r}")
        if label not in ANSWERS:
            raise ValueError(f"bad answer label {label!r}")
        prefix = "t" if frame == "true" else "f"
        suffix = {"Yes": "yes", "No": "no", "Unverified": "unv"}[label]
        c[f"{prefix}_{suffix}"] += 1
    return Counts(**c)


def state_from_counts(counts: Counts, m: int) -> str:
    if m < 1:
        raise ValueError("m must be >= 1")
    sup, ref = counts.sup, counts.ref
    if sup >= m and ref >= m:
        return "Mixed"
    if sup >= m:
        return "Supports"
    if ref >= m:
        return "Refutes"
    return "Insufficient"


def flip_stance(counts: Counts) -> Counts:
    """Reverse stance wording: every Yes becomes No and vice versa (Unverified unchanged)."""
    return Counts(counts.t_no, counts.t_yes, counts.t_unv, counts.f_no, counts.f_yes, counts.f_unv)


def swap_frames(counts: Counts) -> Counts:
    """Exchange the roles of the true-frame and false-frame question sets."""
    return Counts(counts.f_yes, counts.f_no, counts.f_unv, counts.t_yes, counts.t_no, counts.t_unv)
