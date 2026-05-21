"""Tests for the tengri ASCII-art logo."""

import pytest

pytestmark = pytest.mark.contract


def test_logo_is_nonempty_string():
    """LOGO should be a non-empty string with substantial content."""
    from tengri import LOGO

    assert isinstance(LOGO, str)
    assert len(LOGO) > 500  # 37-line solid-block art


def test_logo_banner_is_plain_text():
    """LOGO_BANNER should be a single line of plain text 'tengri'."""
    from tengri import LOGO_BANNER

    assert "\n" not in LOGO_BANNER
    assert "tengri" in LOGO_BANNER


def test_print_logo_respects_env(monkeypatch, capsys):
    """print_logo() should be silent when TENGRI_NO_LOGO is set."""
    from tengri import print_logo

    monkeypatch.setenv("TENGRI_NO_LOGO", "1")
    print_logo()
    out = capsys.readouterr().out
    assert out == ""


def test_print_logo_compact(capsys, monkeypatch):
    """print_logo(size='compact') should print plain 'tengri'."""
    monkeypatch.delenv("TENGRI_NO_LOGO", raising=False)
    from tengri import print_logo

    print_logo(size="compact")
    out = capsys.readouterr().out
    assert "tengri" in out
    assert out.endswith("\n")
    # Compact is plain text — no logo block glyphs
    assert "█" not in out


def test_print_logo_default_is_solid_block(capsys, monkeypatch):
    """Default print_logo() should print the 37-line solid-block logo."""
    monkeypatch.delenv("TENGRI_NO_LOGO", raising=False)
    from tengri import print_logo

    print_logo()
    out = capsys.readouterr().out
    assert "\n" in out
    assert "█" in out
    # ~37 lines × ~65 cols ≈ 2400 chars
    assert 1500 < len(out) < 4000


def test_print_logo_small(capsys, monkeypatch):
    """print_logo(size='small') should print the 21-line compact rendering."""
    monkeypatch.delenv("TENGRI_NO_LOGO", raising=False)
    from tengri import print_logo

    print_logo(size="small")
    out = capsys.readouterr().out
    assert "\n" in out
    # 21-line stipple uses Unicode block-element glyphs
    assert any(ch in out for ch in ("▌", "▐", "▛", "▜", "▞", "▟"))
    assert 400 < len(out) < 1500


def test_print_logo_stipple(capsys, monkeypatch):
    """print_logo(size='stipple') should print the textured 37-line variant."""
    monkeypatch.delenv("TENGRI_NO_LOGO", raising=False)
    from tengri import print_logo

    print_logo(size="stipple")
    out = capsys.readouterr().out
    # Stippled variant uses punctuation glyphs
    assert any(ch in out for ch in ("[", "]", "{", "}"))
    assert len(out) > 1500


def test_print_logo_unknown_size_raises():
    """Unknown size should raise ValueError."""
    import pytest

    from tengri import print_logo

    with pytest.raises(ValueError):
        print_logo(size="bogus")


def test_logo_str_respects_env(monkeypatch):
    """logo_str() should return empty string when TENGRI_NO_LOGO is set."""
    from tengri._logo import logo_str

    monkeypatch.setenv("TENGRI_NO_LOGO", "1")
    assert logo_str() == ""


def test_logo_str_compact(monkeypatch):
    """logo_str(size='compact') should return plain 'tengri'."""
    monkeypatch.delenv("TENGRI_NO_LOGO", raising=False)
    from tengri._logo import logo_str

    result = logo_str(size="compact")
    assert "\n" not in result
    assert "tengri" in result


def test_doctor_includes_logo(monkeypatch, capsys):
    """doctor() output should include logo block characters when enabled."""
    monkeypatch.delenv("TENGRI_NO_LOGO", raising=False)
    from tengri import doctor

    doctor()
    out = capsys.readouterr().out
    # Default logo uses solid blocks
    assert "█" in out


def test_doctor_no_logo_env(monkeypatch, capsys):
    """doctor() should skip logo but still print report when TENGRI_NO_LOGO set."""
    monkeypatch.setenv("TENGRI_NO_LOGO", "1")
    from tengri import doctor

    doctor()
    out = capsys.readouterr().out
    assert "Environment Health Check" in out
    # Logo blocks absent when suppressed
    assert "█" not in out
