"""Tests for the tengri ASCII-art logo."""


def test_logo_is_nonempty_string():
    """LOGO should be a non-empty string with substantial content."""
    from tengri import LOGO

    assert isinstance(LOGO, str)
    assert len(LOGO) > 100


def test_logo_banner_is_one_line():
    """LOGO_BANNER should be a single line containing 'tengri'."""
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


def test_print_logo_without_env(capsys, monkeypatch):
    """print_logo(compact=True) should print banner when env var not set."""
    monkeypatch.delenv("TENGRI_NO_LOGO", raising=False)
    from tengri import print_logo

    print_logo(compact=True)
    out = capsys.readouterr().out
    assert "tengri" in out
    assert out.endswith("\n")


def test_print_logo_full(capsys, monkeypatch):
    """print_logo(compact=False) should print full logo."""
    monkeypatch.delenv("TENGRI_NO_LOGO", raising=False)
    from tengri import print_logo

    print_logo(compact=False)
    out = capsys.readouterr().out
    # Full logo is much longer than banner
    assert len(out) > 1000


def test_logo_str_respects_env(monkeypatch):
    """logo_str() should return empty string when TENGRI_NO_LOGO is set."""
    from tengri._logo import logo_str

    monkeypatch.setenv("TENGRI_NO_LOGO", "1")
    result = logo_str()
    assert result == ""


def test_logo_str_compact(monkeypatch):
    """logo_str(compact=True) should return one-line banner."""
    monkeypatch.delenv("TENGRI_NO_LOGO", raising=False)
    from tengri._logo import logo_str

    result = logo_str(compact=True)
    assert "\n" not in result
    assert "tengri" in result


def test_doctor_includes_logo(monkeypatch, capsys):
    """doctor() output should include logo characters when env var not set."""
    monkeypatch.delenv("TENGRI_NO_LOGO", raising=False)
    from tengri import doctor

    doctor()
    out = capsys.readouterr().out
    # Logo contains Unicode block characters — check for one
    assert "▜" in out or "█" in out or "▗" in out


def test_doctor_no_logo_env(monkeypatch, capsys):
    """doctor() should skip logo but still print report when TENGRI_NO_LOGO set."""
    monkeypatch.setenv("TENGRI_NO_LOGO", "1")
    from tengri import doctor

    doctor()
    out = capsys.readouterr().out
    # Report should still print (with header and checks)
    assert "Environment Health Check" in out
    # But logo should be absent (no block characters from logo)
    assert "▜" not in out
