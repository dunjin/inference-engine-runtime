# -*- coding: utf-8 -*-
"""
Tests for the MindIE Motor Coordinator integration client.
"""
from importlib import reload
from unittest.mock import MagicMock, patch

from patio import envs
from patio.integration.motor.coordinator_client import (
    MotorCoordinatorClient,
    _build_body,
    _derive_instance_id,
    _get_mgmt_endpoint,
    STANDALONE_ID_NAMESPACE,
    STANDALONE_ID_MASK,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _patch_retry():
    """Bypass the 60-attempt retry loop in MotorCoordinatorClient methods."""
    return patch(
        "patio.integration.motor.coordinator_client._retry",
        side_effect=lambda func, **_: func(),
    )


def _set_motor_env(monkeypatch, endpoint="motor-coordinator",
                   mgmt_port="1026", api_key_file=None):
    monkeypatch.setenv("MOTOR_COORDINATOR_ENDPOINT", endpoint)
    monkeypatch.setenv("MOTOR_MGMT_PORT", mgmt_port)
    if api_key_file:
        monkeypatch.setenv("MOTOR_MGMT_API_KEY_FILE", api_key_file)


# ---------------------------------------------------------------------------
# _derive_instance_id
# ---------------------------------------------------------------------------

def test_derive_instance_id_prefill():
    """Derives a deterministic instance id within the namespace range."""
    instance_id = _derive_instance_id("prefill", "10.0.0.1:8000")
    assert instance_id >= STANDALONE_ID_NAMESPACE
    assert instance_id <= STANDALONE_ID_NAMESPACE | STANDALONE_ID_MASK


def test_derive_instance_id_decode():
    instance_id = _derive_instance_id("decode", "10.0.0.2:8000")
    assert instance_id >= STANDALONE_ID_NAMESPACE


def test_derive_instance_id_different_roles_yield_different_ids():
    prefill_id = _derive_instance_id("prefill", "10.0.0.1:8000")
    decode_id = _derive_instance_id("decode", "10.0.0.1:8000")
    assert prefill_id != decode_id


# ---------------------------------------------------------------------------
# _get_mgmt_endpoint
# ---------------------------------------------------------------------------

def test_get_mgmt_endpoint(monkeypatch):
    monkeypatch.setenv("MOTOR_COORDINATOR_ENDPOINT", "coordinator-svc")
    monkeypatch.setenv("MOTOR_MGMT_PORT", "9999")
    reload(envs)

    endpoint = _get_mgmt_endpoint()
    assert endpoint == "coordinator-svc:9999"

def test_get_mgmt_endpoint_uses_default_port(monkeypatch):
    monkeypatch.setenv("MOTOR_COORDINATOR_ENDPOINT", "coordinator-svc")
    monkeypatch.delenv("MOTOR_MGMT_PORT", raising=False)
    reload(envs)

    endpoint = _get_mgmt_endpoint()
    assert endpoint == "coordinator-svc:1026"


def test_get_mgmt_endpoint_raises_when_not_set(monkeypatch):
    monkeypatch.delenv("MOTOR_COORDINATOR_ENDPOINT", raising=False)
    reload(envs)

    import pytest
    with pytest.raises(RuntimeError, match="MOTOR_COORDINATOR_ENDPOINT is not set"):
        _get_mgmt_endpoint()


# ---------------------------------------------------------------------------
# _build_body
# ---------------------------------------------------------------------------

def test_build_body_add():
    body = _build_body(event="add", role="prefill", address="10.0.0.1:8000",
                       instance_id=1073741825, model_name="Qwen-7B")
    assert body["event"] == "add"
    assert body["engine_type"] == "vllm"
    assert body["dispatch_capabilities"] == "prefill_handoff_decode"
    assert body["model_name"] == "Qwen-7B"
    assert len(body["instances"]) == 1
    inst = body["instances"][0]
    assert inst["id"] == 1073741825
    assert inst["role"] == "prefill"
    assert inst["endpoints"][0]["address"] == "10.0.0.1:8000"


def test_build_body_del_without_model_name():
    body = _build_body(event="del", role="decode", address="10.0.0.2:8000",
                       instance_id=1073741826)
    assert body["event"] == "del"
    assert "model_name" not in body


# ---------------------------------------------------------------------------
# MotorCoordinatorClient instantiation
# ---------------------------------------------------------------------------

def test_init_prefill(monkeypatch):
    _set_motor_env(monkeypatch)
    monkeypatch.setenv("POD_IP", "10.0.0.1")
    reload(envs)

    client = MotorCoordinatorClient(
        {"type": "prefill", "port": 8000}
    )
    assert client.role == "prefill"
    assert client.address == "10.0.0.1:8000"
    assert client.instance_id >= STANDALONE_ID_NAMESPACE


def test_init_decode(monkeypatch):
    _set_motor_env(monkeypatch)
    monkeypatch.setenv("POD_IP", "10.0.0.2")
    reload(envs)

    client = MotorCoordinatorClient(
        {"worker_type": "decode", "port": 8001}
    )
    assert client.role == "decode"
    assert client.instance_id >= STANDALONE_ID_NAMESPACE


def test_init_uses_explicit_instance(monkeypatch):
    _set_motor_env(monkeypatch)
    monkeypatch.setenv("POD_IP", "10.0.0.1")
    reload(envs)

    client = MotorCoordinatorClient(
        {"type": "prefill", "instance": "custom-svc:8500", "port": 8500}
    )
    assert client.address == "custom-svc:8500"


def test_init_raises_without_worker_type(monkeypatch):
    _set_motor_env(monkeypatch)
    monkeypatch.setenv("POD_IP", "10.0.0.1")
    reload(envs)

    import pytest
    with pytest.raises(ValueError, match="worker type is not set"):
        MotorCoordinatorClient({"port": 8000})


# ---------------------------------------------------------------------------
# wait_engine_ready
# ---------------------------------------------------------------------------

@patch("patio.integration.motor.coordinator_client.requests.get")
def test_wait_engine_ready_ok(mock_get, monkeypatch):
    _set_motor_env(monkeypatch)
    monkeypatch.setenv("POD_IP", "10.0.0.1")
    monkeypatch.setenv("TOPO_CONNECT_TIMEOUT", "4")
    monkeypatch.setenv("TOPO_HEALTH_CHECK_TIMEOUT", "35")
    reload(envs)

    response = MagicMock()
    response.status_code = 200
    mock_get.return_value = response

    with _patch_retry():
        client = MotorCoordinatorClient({"type": "prefill", "port": 8000})
        assert client.wait_engine_ready({"port": 8000})
        mock_get.assert_called_once_with(
            "http://10.0.0.1:8000/health",
            timeout=(4.0, 35.0),
        )


@patch("patio.integration.motor.coordinator_client.requests.get")
def test_wait_engine_ready_fails_on_non_200(mock_get, monkeypatch):
    _set_motor_env(monkeypatch)
    monkeypatch.setenv("POD_IP", "10.0.0.1")
    reload(envs)

    response = MagicMock()
    response.status_code = 503
    response.text = "not ready"
    mock_get.return_value = response

    with _patch_retry():
        client = MotorCoordinatorClient({"type": "prefill", "port": 8000})
        assert client.wait_engine_ready({"port": 8000}) is False


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------

@patch("patio.integration.motor.coordinator_client.requests.post")
def test_register_posts_instances_refresh(mock_post, monkeypatch):
    _set_motor_env(monkeypatch)
    monkeypatch.setenv("POD_IP", "10.0.0.1")
    monkeypatch.setenv("TOPO_CONNECT_TIMEOUT", "4")
    monkeypatch.setenv("TOPO_REGISTER_TIMEOUT", "12")
    monkeypatch.setenv("MOTOR_MODEL_NAME", "test-model")
    reload(envs)

    response = MagicMock()
    response.status_code = 200
    response.text = "ok"
    mock_post.return_value = response

    with _patch_retry():
        client = MotorCoordinatorClient({"type": "prefill", "port": 8000})
        assert client.register("", {"type": "prefill", "port": 8000})
        # Verify the POST went to the right URL with the right body
        call_args = mock_post.call_args
        assert call_args is not None
        url = call_args[0][0] if call_args[0] else call_args[1]["url"]
        assert "/instances/refresh" in url
        assert "motor-coordinator:1026" in url


@patch("patio.integration.motor.coordinator_client.requests.post")
def test_register_returns_false_on_error(mock_post, monkeypatch):
    _set_motor_env(monkeypatch)
    monkeypatch.setenv("POD_IP", "10.0.0.1")
    reload(envs)

    response = MagicMock()
    response.status_code = 500
    response.text = "bad"
    mock_post.return_value = response

    with _patch_retry():
        client = MotorCoordinatorClient({"type": "prefill", "port": 8000})
        assert client.register("", {"type": "prefill", "port": 8000}) is False


# ---------------------------------------------------------------------------
# unregister
# ---------------------------------------------------------------------------

@patch("patio.integration.motor.coordinator_client.requests.post")
def test_unregister_posts_instance_del(mock_post, monkeypatch):
    _set_motor_env(monkeypatch)
    monkeypatch.setenv("POD_IP", "10.0.0.1")
    reload(envs)

    response = MagicMock()
    response.status_code = 200
    response.text = "ok"
    mock_post.return_value = response

    with _patch_retry():
        client = MotorCoordinatorClient({"type": "prefill", "port": 8000})
        assert client.unregister()
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        body = call_kwargs["json"]
        assert body["event"] == "del"