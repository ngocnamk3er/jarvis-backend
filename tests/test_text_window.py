"""read_file's windowed mode, shaped like DeepSeek Harness's `read`."""

import pytest

from app.agents.tools import text_window as tw


def doc(n):
    return "\n".join(f"line {i}" for i in range(1, n + 1))


def test_a_window_is_numbered_and_says_where_to_continue():
    out = tw.read_window(doc(50), offset=11, limit=10)
    assert out.startswith("11: line 11\n12: line 12") and "20: line 20" in out and "21: line 21" not in out
    assert out.endswith("(Showing lines 11-20 of 50. Use offset=21 to continue.)")


def test_the_last_window_says_the_file_ended():
    assert tw.read_window(doc(5), offset=3).endswith("(End of file - total 5 lines)")
    assert tw.read_window(doc(5), limit=5).endswith("(End of file - total 5 lines)")


def test_the_default_window_stops_at_the_line_cap():
    out = tw.read_window(doc(5000))
    assert "2000: line 2000" in out and "2001:" not in out and "Use offset=2001 to continue" in out


def test_a_huge_window_is_capped_by_bytes_and_resumes_after_the_last_line_shown():
    text = "\n".join("x" * 1500 for _ in range(200))                 # ~300 KB
    out = tw.read_window(text, limit=200)
    assert len(out.encode()) < tw.MAX_BYTES + 400
    shown = int(out.split("(Output capped. Showing lines 1-")[1].split(".")[0])
    assert f"Use offset={shown + 1} to continue" in out


def test_a_very_long_line_is_cut():
    out = tw.read_window("a" * 5000)
    assert out.split("\n")[0] == "1: " + "a" * tw.MAX_LINE_CHARS + "..."


def test_bad_arguments_are_explained():
    with pytest.raises(tw.WindowError, match="offset must be at least 1"):
        tw.read_window(doc(5), offset=0)
    with pytest.raises(tw.WindowError, match="less than or equal to 2000"):
        tw.read_window(doc(5), limit=2001)
    with pytest.raises(tw.WindowError, match="past the end of the file, which has 5 lines"):
        tw.read_window(doc(5), offset=9)


def test_an_empty_text_reads_as_empty():
    assert tw.read_window("") == "(End of file - total 0 lines)"
