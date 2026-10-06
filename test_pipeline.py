import pytest
import numpy as np
from parser import parse_metar_to_numeric

def test_feature_engineering_integrity():
    sample_metars = [
        "KJFK 010000Z 24010KT 10SM CLR 20/15 A2992",
        "KJFK 010100Z 24009KT 10SM CLR 19/15 A2991",
        "KJFK 010200Z 24008KT 8SM SCT040 18/15 A2990",
        "KJFK 010300Z 24006KT 6SM BKN030 17/15 A2989",
        "KJFK 010400Z 20004KT 3SM BR BKN015 16/15 A2988",
        "KJFK 010500Z 00000KT 1SM FG OVC005 15/15 A2987",
    ]
    parsed = [parse_metar_to_numeric(m) for m in sample_metars]
    assert len(parsed) == 6
    # Check that dewpoint depression converges towards 0 as fog moves in
    assert parsed[0].dewpoint_depression == 5.0
    assert parsed[-1].dewpoint_depression == 0.0