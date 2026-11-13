import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.states import (  # noqa: E402
    STATES, Counts, counts_from_answers, flip_stance, state_from_counts, swap_frames,
)


def all_counts():
    """Every valid count vector: each frame has 5 answers split into (yes, no, unverified)."""
    splits = [(y, n, 5 - y - n) for y in range(6) for n in range(6 - y)]
    assert len(splits) == 21
    for (ty, tn, tu), (fy, fn, fu) in itertools.product(splits, splits):
        yield Counts(ty, tn, tu, fy, fn, fu)


ALL = list(all_counts())
MS = (1, 2, 3)


def test_enumeration_size():
    assert len(ALL) == 441
    assert all(c.frames_sum_ok for c in ALL)


@pytest.mark.parametrize("m", MS)
def test_states_partition_the_space(m):
    seen = {s: 0 for s in STATES}
    for c in ALL:
        s = state_from_counts(c, m)
        assert s in STATES
        seen[s] += 1
        sup, ref = c.sup, c.ref
        # state is exactly determined by the (sup>=m, ref>=m) pair
        assert s == {(True, True): "Mixed", (True, False): "Supports", (False, True): "Refutes", (False, False): "Insufficient"}[(sup >= m, ref >= m)]
    assert all(v > 0 for v in seen.values()), seen        # all four states reachable for m in {1,2,3}


@pytest.mark.parametrize("m", MS)
def test_reversing_stance_swaps_supports_and_refutes(m):
    swap = {"Supports": "Refutes", "Refutes": "Supports", "Mixed": "Mixed", "Insufficient": "Insufficient"}
    for c in ALL:
        assert state_from_counts(flip_stance(c), m) == swap[state_from_counts(c, m)]
        assert flip_stance(flip_stance(c)) == c


@pytest.mark.parametrize("m", MS)
def test_swapping_frames_also_swaps_supports_and_refutes(m):
    swap = {"Supports": "Refutes", "Refutes": "Supports", "Mixed": "Mixed", "Insufficient": "Insufficient"}
    for c in ALL:
        assert state_from_counts(swap_frames(c), m) == swap[state_from_counts(c, m)]


@pytest.mark.parametrize("m", MS)
def test_removing_all_supporting_answers_moves_toward_insufficient(m):
    """Turning every supporting answer into Unverified: sup becomes 0, so the state is never Supports or Mixed;
    a Supports claim becomes Insufficient (it had ref < m), a Mixed claim becomes Refutes."""
    for c in ALL:
        removed = Counts(0, c.t_no, c.t_unv + c.t_yes, c.f_yes, 0, c.f_unv + c.f_no)
        assert removed.sup == 0 and removed.ref == c.ref
        before, after = state_from_counts(c, m), state_from_counts(removed, m)
        assert after in ("Insufficient", "Refutes")
        if before == "Supports":
            assert after == "Insufficient"
        if before == "Mixed":
            assert after == "Refutes"


@pytest.mark.parametrize("m", MS)
def test_adding_a_refuting_answer_to_supports_at_threshold_gives_mixed(m):
    """Supports with ref == m-1: converting one Unverified answer into a refuting answer must give Mixed."""
    checked = 0
    for c in ALL:
        if state_from_counts(c, m) != "Supports" or c.ref != m - 1:
            continue
        if c.t_unv > 0:
            new = Counts(c.t_yes, c.t_no + 1, c.t_unv - 1, c.f_yes, c.f_no, c.f_unv)
        elif c.f_unv > 0:
            new = Counts(c.t_yes, c.t_no, c.t_unv, c.f_yes + 1, c.f_no, c.f_unv - 1)
        else:
            continue
        assert new.ref == m and new.sup == c.sup
        assert state_from_counts(new, m) == "Mixed"
        checked += 1
    assert checked > 0


@pytest.mark.parametrize("m", MS)
def test_monotone_in_supporting_evidence(m):
    """Converting an Unverified answer into a supporting one never reduces sup and never creates Refutes from Supports."""
    for c in ALL:
        if c.t_unv > 0:
            new = Counts(c.t_yes + 1, c.t_no, c.t_unv - 1, c.f_yes, c.f_no, c.f_unv)
        elif c.f_unv > 0:
            new = Counts(c.t_yes, c.t_no, c.t_unv, c.f_yes, c.f_no + 1, c.f_unv - 1)
        else:
            continue
        assert new.sup == c.sup + 1 and new.ref == c.ref
        if state_from_counts(c, m) == "Supports":
            assert state_from_counts(new, m) == "Supports"


def test_all_unverified_is_insufficient_for_every_m():
    c = Counts(0, 0, 5, 0, 0, 5)
    for m in MS:
        assert state_from_counts(c, m) == "Insufficient"


def test_counts_from_answers_and_stance_mapping():
    answers = [("true", "Yes")] * 3 + [("true", "No")] + [("true", "Unverified")] + \
              [("false", "No")] * 2 + [("false", "Yes")] + [("false", "Unverified")] * 2
    c = counts_from_answers(answers)
    assert c.as_list() == [3, 1, 1, 1, 2, 2]
    assert c.sup == 3 + 2 and c.ref == 1 + 1 and c.unv == 3
    assert c.frames_sum_ok
    assert state_from_counts(c, 2) == "Mixed" and state_from_counts(c, 3) == "Supports"


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        counts_from_answers([("maybe", "Yes")])
    with pytest.raises(ValueError):
        counts_from_answers([("true", "Maybe")])
    with pytest.raises(ValueError):
        state_from_counts(Counts(5, 0, 0, 0, 5, 0), 0)
