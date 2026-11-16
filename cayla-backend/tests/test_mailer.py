from app import mailer


def make_invite():
    return mailer.invite_email(
        "Nur Aisyah",
        "nur.aisyah@towera.sg",
        "K7m-9qXp-2Lw",
        "Sam Lee",
        "staff",
        168,
        "https://insight.example/login",
    )


def test_invite_matches_design():
    e = make_invite()
    assert e.subject == "You've been invited to InSight"
    assert "Hi Nur," in e.text
    assert "Sam Lee added you to InSight for Tower A as Security staff." in e.text
    assert "You'll be asked to set your own password straight after." in e.text
    assert (
        "This password expires in 7 days. If you weren't expecting this, you can ignore this email."
        in e.text
    )
    for part in (
        "K7m-9qXp-2Lw",
        "nur.aisyah@towera.sg",
        "Sign in to InSight",
        "https://insight.example/login",
    ):
        assert part in e.html


def test_html_escapes_names():
    e = mailer.invite_email(
        "<b>x</b>", "a@b.com", "K7m-9qXp-2Lw", "Sam", "staff", 168, "http://x"
    )
    assert "<b>x</b>" not in e.html and "&lt;b&gt;x&lt;/b&gt;" in e.html


def test_reset_email():
    e = mailer.reset_email(
        "Nur Aisyah", "nur@x.sg", "K7m-9qXp-2Lw", "Sam Lee", 168, "http://x"
    )
    assert (
        e.subject == "Your InSight password was reset"
        and "Sam Lee reset your InSight password" in e.text
    )


def test_without_smtp_prints_instead_of_sending(capsys, monkeypatch):
    monkeypatch.setattr(mailer, "SMTP_HOST", "")
    mailer.send_email("nur.aisyah@towera.sg", make_invite())
    out = capsys.readouterr().out
    assert "To: nur.aisyah@towera.sg" in out and "K7m-9qXp-2Lw" in out


def test_smtp_failure_raises_email_error(monkeypatch):
    monkeypatch.setattr(mailer, "SMTP_HOST", "127.0.0.1")
    monkeypatch.setattr(mailer, "SMTP_PORT", 1)  # nothing listens here
    try:
        mailer.send_email("a@b.com", mailer.Email("s", "b"))
    except mailer.EmailError:
        return
    raise AssertionError("expected EmailError")
