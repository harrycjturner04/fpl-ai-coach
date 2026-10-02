import pandas as pd

from prediction.scoring import points_from_stats, rules_for_season


def _row(**kw):
    base = dict(position="MID", minutes=90, goals=0, assists=0, clean_sheets=0, goals_conceded=0,
                own_goals=0, penalties_saved=0, penalties_missed=0, saves=0, bonus=0,
                defensive_contribution=0, yellow_cards=0, red_cards=0)
    base.update(kw)
    return base


def test_rules_change_in_2025_26():
    old, new = rules_for_season("2024-25"), rules_for_season("2025-26")
    assert old.goal["GKP"] == 6 and new.goal["GKP"] == 10
    assert old.defensive_contribution == 0 and new.defensive_contribution == 2


def test_points_from_stats_known_rows():
    rows = pd.DataFrame([
        _row(position="DEF", goals=1, clean_sheets=1, bonus=3),                 # 2 + 6 + 4 + 3
        _row(position="GKP", saves=7, goals_conceded=3, yellow_cards=1),        # 2 + 2 - 1 - 1
        _row(position="FWD", minutes=20, assists=1),                            # 1 + 3
        _row(position="MID", minutes=0),                                        # 0
        _row(position="DEF", defensive_contribution=10),                        # 2 + 2 (2025-26 only)
    ])
    assert points_from_stats(rows, rules_for_season("2025-26")).tolist() == [15, 2, 4, 0, 4]
    assert points_from_stats(rows, rules_for_season("2024-25")).tolist() == [15, 2, 4, 0, 2]
