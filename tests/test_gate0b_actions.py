from collections import Counter

from evaluation.sketchalloc.build_candidates import SwapCandidate, Unit
from evaluation.sketchalloc.build_gate0b_actions import (
    select_with_cap,
    stable_id,
)


def test_gate0b_action_selector_keeps_unique_actions_and_category():
    pool = [
        SwapCandidate(
            donor=Unit(i, 0),
            receiver=Unit(i + 10, 1),
            amount=16,
            lu_remove_cost=0.1 + i,
            lu_next_gain=1.0 + i,
            lu_marginal_delta=0.9,
        )
        for i in range(6)
    ]

    selected_ids = set()
    donor_use = Counter()
    receiver_use = Counter()
    chosen = select_with_cap(
        pool,
        count=4,
        selected_ids=selected_ids,
        donor_use=donor_use,
        receiver_use=receiver_use,
        unit_use_cap=1,
        category="lu_promising",
    )

    assert len(chosen) == 4
    assert len({stable_id(c) for c in chosen}) == 4
    assert all(c.category == "lu_promising" for c in chosen)
