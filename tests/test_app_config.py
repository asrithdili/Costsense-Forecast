"""Tests for JSON application config."""

from __future__ import annotations

import json

from src import config


def test_load_config_merges_local_override(tmp_path, monkeypatch):
    base = {
        "aws": {"region": "us-west-2", "runtime_profile": "ecs-task-role"},
        "github": {"token": ""},
    }
    local = {
        "aws": {"runtime_profile": "dev-profile"},
        "github": {"token": "ghp_test"},
    }
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "costsense.json").write_text(json.dumps(base), encoding="utf-8")
    (config_dir / "costsense.local.json").write_text(json.dumps(local), encoding="utf-8")

    monkeypatch.setenv("COSTSENSE_CONFIG_DIR", str(config_dir))
    config.reload_config()

    assert config.get_str("aws.region") == "us-west-2"
    assert config.get_str("aws.runtime_profile") == "dev-profile"
    assert config.get_str("github.token") == "ghp_test"


def test_get_returns_default_for_missing_path(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "costsense.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("COSTSENSE_CONFIG_DIR", str(config_dir))
    config.reload_config()

    assert config.get("bedrock.region", "us-west-2") == "us-west-2"
