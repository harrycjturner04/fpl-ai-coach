from orchestration.appdb import block_reason


def test_allows_exactly_at_cap():
    assert block_reason(6.0, 5, 1.25, False, estimate_usd=0.25) is None


def test_blocks_over_cap():
    assert block_reason(6.01, 5, 1.25, False, estimate_usd=0.25).startswith("Monthly AI cap reached")


def test_ai_off_wins():
    assert block_reason(0, 5, 1.25, True) == "AI is switched off in Settings."
