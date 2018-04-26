import pytest

import model.parser

to_int = [("0", True), ("12", True), ("-12", True), ("1.3", False), ("-1.5", False), ("asd", False)]


@pytest.fixture(params=to_int)
def to_int_loop(request):
    return request.param


def test_is_int(to_int_loop):
    arg, expected = to_int_loop
    res = model.parser.is_int(arg)
    assert res == expected


