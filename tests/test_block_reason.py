from orchestration.appdb import block_reason


def test_allows_exactly_at_cap():
    assert block_reason(6.0, 5, 1.25, False, estimate_usd=0.25) is None


def test_blocks_over_cap():
    assert block_reason(6.01, 5, 1.25, False, estimate_usd=0.25) ==         "Monthly AI cap: £4.81 of £5.00 spent; a run (about £0.20) would go over."


def test_blocks_at_cap():
    assert block_reason(6.25, 5, 1.25, False, estimate_usd=0.25) == "Monthly AI cap reached: £5.00 of £5.00 spent."


def test_ai_off_wins():
    assert block_reason(0, 5, 1.25, True) == "AI is switched off in Settings."
