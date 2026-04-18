"""Tests for identity scoping enforcement in single-user credential resolution.

Covers three defenses introduced to prevent identity bleed (an agent
authenticated under a different email than the one declared in
``USER_GOOGLE_EMAIL``):

1. ``LocalDirectoryCredentialStore.list_users`` rejects files with parked
   suffixes (``.disabled``, ``.bak*``, ``.old``, ``.backup``, ``.parked``)
   and warns when such files are detected.
2. ``_find_any_credentials`` warns when multiple users are present and
   ``USER_GOOGLE_EMAIL`` is declared.
3. ``get_credentials`` in single-user mode refuses to fall back to
   ``_find_any_credentials`` when a specific user was requested but not
   found — serving a different identity than the one requested is an
   identity scoping violation.
"""

import logging
from unittest.mock import MagicMock

import pytest

from auth import google_auth
from auth.credential_store import LocalDirectoryCredentialStore


@pytest.fixture
def cred_store(tmp_path):
    return LocalDirectoryCredentialStore(base_dir=str(tmp_path / "creds"))


class TestListUsersRejectsParkedSuffixes:
    def test_accepts_plain_email_json(self, cred_store, tmp_path):
        base = tmp_path / "creds"
        base.mkdir(parents=True, exist_ok=True)
        (base / "real@example.com.json").write_text("{}")

        assert cred_store.list_users() == ["real@example.com"]

    @pytest.mark.parametrize(
        "parked_filename",
        [
            "ghost@example.com.json.disabled",
            "ghost@example.com.json.bak",
            "ghost@example.com.json.bak2",
            "ghost@example.com.json.old",
            "ghost@example.com.json.backup",
            "ghost@example.com.json.parked",
        ],
    )
    def test_rejects_parked_suffix(self, cred_store, tmp_path, parked_filename):
        base = tmp_path / "creds"
        base.mkdir(parents=True, exist_ok=True)
        (base / "real@example.com.json").write_text("{}")
        (base / parked_filename).write_text("{}")

        assert cred_store.list_users() == ["real@example.com"]

    def test_warns_when_parked_files_detected(
        self, cred_store, tmp_path, caplog
    ):
        base = tmp_path / "creds"
        base.mkdir(parents=True, exist_ok=True)
        (base / "real@example.com.json").write_text("{}")
        (base / "ghost@example.com.json.disabled").write_text("{}")

        with caplog.at_level(logging.WARNING, logger="auth.credential_store"):
            cred_store.list_users()

        assert any(
            "parked credential file" in rec.message
            and "ghost@example.com.json.disabled" in rec.message
            for rec in caplog.records
        )

    def test_rejects_non_credential_files(self, cred_store, tmp_path):
        base = tmp_path / "creds"
        base.mkdir(parents=True, exist_ok=True)
        (base / "real@example.com.json").write_text("{}")
        (base / "oauth_states.json").write_text("{}")
        (base / "notes.txt").write_text("hello")

        assert cred_store.list_users() == ["real@example.com"]


class TestFindAnyCredentialsWarnsOnAmbiguity:
    def test_warns_when_multiple_users_and_user_email_declared(
        self, monkeypatch, caplog
    ):
        fake_store = MagicMock()
        fake_store.list_users.return_value = [
            "alice@example.com",
            "bob@example.com",
        ]
        fake_store.get_credential.return_value = MagicMock()

        monkeypatch.setattr(
            google_auth, "get_credential_store", lambda: fake_store
        )
        monkeypatch.setenv("USER_GOOGLE_EMAIL", "alice@example.com")

        with caplog.at_level(logging.WARNING, logger="auth.google_auth"):
            google_auth._find_any_credentials()

        assert any(
            "USER_GOOGLE_EMAIL=alice@example.com" in rec.message
            and "2 users" in rec.message
            for rec in caplog.records
        )

    def test_no_warning_when_single_user(self, monkeypatch, caplog):
        fake_store = MagicMock()
        fake_store.list_users.return_value = ["alice@example.com"]
        fake_store.get_credential.return_value = MagicMock()

        monkeypatch.setattr(
            google_auth, "get_credential_store", lambda: fake_store
        )
        monkeypatch.setenv("USER_GOOGLE_EMAIL", "alice@example.com")

        with caplog.at_level(logging.WARNING, logger="auth.google_auth"):
            google_auth._find_any_credentials()

        assert not any(
            "users" in rec.message and rec.levelno >= logging.WARNING
            for rec in caplog.records
        )


class TestGetCredentialsRefusesFallback:
    """The identity scoping fix: no fallback to any-credential when a
    specific user was requested but not found."""

    def test_returns_none_when_requested_user_has_no_credentials(
        self, monkeypatch
    ):
        fake_store = MagicMock()
        fake_store.get_credential.return_value = None

        monkeypatch.setattr(
            google_auth, "get_credential_store", lambda: fake_store
        )
        monkeypatch.setenv("MCP_SINGLE_USER_MODE", "1")

        def fail_if_called(*args, **kwargs):
            raise AssertionError(
                "_find_any_credentials must not be invoked when a specific "
                "user was requested (identity scoping enforcement)"
            )

        monkeypatch.setattr(google_auth, "_find_any_credentials", fail_if_called)

        result = google_auth.get_credentials(
            user_google_email="requested@example.com",
            required_scopes=["scope1"],
            session_id=None,
        )

        assert result is None
        fake_store.get_credential.assert_called_once_with("requested@example.com")

    def test_returns_credentials_when_requested_user_present(self, monkeypatch):
        fake_credentials = MagicMock()
        fake_credentials.valid = True
        fake_credentials.expired = False
        fake_credentials.has_scopes = MagicMock(return_value=True)

        fake_store = MagicMock()
        fake_store.get_credential.return_value = fake_credentials

        monkeypatch.setattr(
            google_auth, "get_credential_store", lambda: fake_store
        )
        monkeypatch.setenv("MCP_SINGLE_USER_MODE", "1")

        result = google_auth.get_credentials(
            user_google_email="requested@example.com",
            required_scopes=[],
            session_id=None,
        )

        assert result is fake_credentials
