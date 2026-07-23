"""Tests for AWS profile resolution."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.aws import profiles
from src.aws.session import runtime_profile_name


def test_resolve_all_uses_runtime_profile_when_no_config_profiles(monkeypatch):
    monkeypatch.setattr(profiles, "list_profiles", lambda: ())
    runtime = profiles.ProfileInfo(profile="ecs-task-role", account_id="123456789012")

    with patch.object(profiles, "resolve_runtime", return_value=runtime):
        assert profiles.resolve_all() == [runtime]


def test_resolve_all_prefers_configured_profiles():
    configured = [
        profiles.ProfileInfo(profile="dev", account_id="111111111111"),
        profiles.ProfileInfo(profile="broken", account_id=None, error="denied"),
    ]

    with patch.object(profiles, "list_profiles", return_value=("dev", "broken")):
        with patch.object(profiles, "resolve", side_effect=configured):
            with patch.object(profiles, "resolve_runtime") as runtime_mock:
                result = profiles.resolve_all()

    assert result == [configured[0]]
    runtime_mock.assert_not_called()


def test_resolve_runtime_uses_sts_identity():
    sts = MagicMock()
    sts.get_caller_identity.return_value = {"Account": "609400232087"}
    session = MagicMock()
    session.client.return_value = sts

    with patch.object(profiles, "make_session", return_value=session):
        info = profiles.resolve_runtime()

    assert info == profiles.ProfileInfo(
        profile=runtime_profile_name(),
        account_id="609400232087",
    )
