import re

from app.password_policy import new_temp_password, password_problem


def test_temp_password_matches_invite_email_format():
    # The design shows K7m-9qXp-2Lw: 3-4-3 characters with dashes
    for _ in range(200):
        pw = new_temp_password()
        assert re.fullmatch(r"[A-Za-z2-9]{3}-[A-Za-z2-9]{4}-[A-Za-z2-9]{3}", pw), pw
        assert any(c.isdigit() for c in pw) and any(c.isalpha() for c in pw)
        assert not set(pw) & set("0O1lI")  # no look-alike characters
        assert password_problem(pw) is None  # a temp password itself passes the rules


def test_temp_passwords_differ():
    assert len({new_temp_password() for _ in range(100)}) == 100


def test_rules_match_the_set_password_screen():
    assert password_problem("short1") == "Password must be at least 8 characters"
    assert password_problem("ABCDEFG1!") == "Password must include a lower-case letter"
    assert password_problem("abcdefg1!") == "Password must include a capital letter"
    assert password_problem("Abcdefgh!") == "Password must include a number"
    assert (
        password_problem("Abcdefg12")
        == "Password must include a symbol, such as ! @ # or -"
    )
    assert password_problem("Sunflower42!") is None
    assert (
        password_problem("K7m-9qXp-2Lw", "K7m-9qXp-2Lw")
        == "New password must be different from the temporary one"
    )
