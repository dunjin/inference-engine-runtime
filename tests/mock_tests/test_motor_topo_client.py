# -*- coding: utf-8 -*-
"""
Tests for the MindIE Motor topo client (MotorTopoClient).
"""
from importlib import reload
from unittest.mock import MagicMock, patch

from patio import envs
from patio.topo.client.motor_topo_client import (
    MotorTopoClient,
    build_body,
    derive_instance_id,
    get_motor_endpoint,
    STANDALONE_ID_NAMESPACE,
    STANDALONE_ID_MASK,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _patch_retry():
    """Bypass the 60-attempt retry loop in MotorTopoClient methods."""
    return patch(
        "patio.topo.client.motor_topo_client.utils.retry",
        side_effect=lambda func, **_: func(),
    )


def _set_motor_env(monkeypatch, group="demo", router_role="router",
                   router_port="9000", api_key_file=None):
    monkeypatch.setenv("RBG_GROUP_NAME", group)
    monkeypatch.setenv("ROUTER_ROLE_NAME", router_role)
    monkeypatch.setenv("ROUTER_PORT", router_port)
    if api_key_file:
        monkeypatch.setenv("MOTOR_MGMT_API_KEY_FILE", api_key_file)


# ---------------------------------------------------------------------------
# derive_instance_id
# ---------------------------------------------------------------------------

def test_derive_instance_id_prefill():
    """Derives a deterministic instance id within the namespace range."""
    instance_id = derive_instance_id("prefill", "10.0.0.1:8000")
    assert instance_id >= STANDALONE_ID_NAMESPACE
    assert instance_id <= STANDALONE_ID_NAMESPACE | STANDALONE_ID_MASK


def test_derive_instance_id_decode():
    instance_id = derive_instance_id("decode", "10.0.0.2:8000")
    assert instance_id >= STANDALONE_ID_NAMESPACE


def test_derive_instance_id_different_roles_yield_different_ids():
    prefill_id = derive_instance_id("prefill", "10.0.0.1:8000")
    decode_id = derive_instance_id("decode", "10.0.0.1:8000")
    assert prefill_id != decode_id


# ---------------------------------------------------------------------------
# get_motor_endpoint (RBG DNS derivation, same shape as sgl/vllm clients)
# ---------------------------------------------------------------------------

def test_get_motor_endpoint(monkeypatch):
    _set_motor_env(monkeypatch, group="demo", router_role="router", router_port="9000")
    reload(envs)

    endpoint = get_motor_endpoint()
    assert endpoint == "demo-router-0.s-demo-router:9000"


def test_get_motor_endpoint_raises_when_group_missing(monkeypatch):
    monkeypatch.delenv("RBG_GROUP_NAME", raising=False)
    monkeypatch.setenv("ROUTER_ROLE_NAME", "router")
    monkeypatch.setenv("ROUTER_PORT", "9000")
    reload(envs)

    import pytest
    with pytest.raises(RuntimeError, match="RBG_GROUP_NAME is not set"):
        get_motor_endpoint()


def test_get_motor_endpoint_raises_when_router_role_missing(monkeypatch):
    monkeypatch.setenv("RBG_GROUP_NAME", "demo")
    monkeypatch.delenv("ROUTER_ROLE_NAME", raising=False)
    monkeypatch.setenv("ROUTER_PORT", "9000")
    reload(envs)

    import pytest
    with pytest.raises(RuntimeError, match="ROUTER_ROLE_NAME is not set"):
        get_motor_endpoint()


# ---------------------------------------------------------------------------
# build_body
# ---------------------------------------------------------------------------

def test_build_body_add():
    body = build_body(event="add", role="prefill", address="10.0.0.1:8000",
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
    body = build_body(event="del", role="decode", address="10.0.0.2:8000",
                      instance_id=1073741826)
    assert body["event"] == "del"
    assert "model_name" not in body


# ---------------------------------------------------------------------------
# MotorTopoClient instantiation
# ---------------------------------------------------------------------------

def test_init_prefill(monkeypatch):
    _set_motor_env(monkeypatch)
    monkeypatch.setenv("POD_IP", "10.0.0.1")
    reload(envs)

    client = MotorTopoClient(
        {"type": "prefill", "port": 8000}
    )
    assert client.role == "prefill"
    assert client.address == "10.0.0.1:8000"
    assert client.instance_id >= STANDALONE_ID_NAMESPACE


def test_init_decode(monkeypatch):
    _set_motor_env(monkeypatch)
    monkeypatch.setenv("POD_IP", "10.0.0.2")
    reload(envs)

    client = MotorTopoClient(
        {"worker_type": "decode", "port": 8001}
    )
    assert client.role == "decode"
    assert client.instance_id >= STANDALONE_ID_NAMESPACE


def test_init_uses_explicit_instance(monkeypatch):
    _set_motor_env(monkeypatch)
    monkeypatch.setenv("POD_IP", "10.0.0.1")
    reload(envs)

    client = MotorTopoClient(
        {"type": "prefill", "instance": "custom-svc:8500", "port": 8500}
    )
    assert client.address == "custom-svc:8500"


def test_init_raises_without_worker_type(monkeypatch):
    _set_motor_env(monkeypatch)
    monkeypatch.setenv("POD_IP", "10.0.0.1")
    reload(envs)

    import pytest
    with pytest.raises(ValueError, match="worker type is not set"):
        MotorTopoClient({"port": 8000})


def test_init_raises_without_rbg_group(monkeypatch):
    monkeypatch.delenv("RBG_GROUP_NAME", raising=False)
    monkeypatch.setenv("ROUTER_ROLE_NAME", "router")
    monkeypatch.setenv("ROUTER_PORT", "9000")
    monkeypatch.setenv("POD_IP", "10.0.0.1")
    reload(envs)

    import pytest
    with pytest.raises(RuntimeError, match="RBG_GROUP_NAME is not set"):
        MotorTopoClient({"type": "prefill", "port": 8000})


# ---------------------------------------------------------------------------
# wait_engine_ready
# ---------------------------------------------------------------------------

@patch("patio.topo.client.motor_topo_client.requests.get")
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
        client = MotorTopoClient({"type": "prefill", "port": 8000})
        assert client.wait_engine_ready({"port": 8000})
        mock_get.assert_called_once_with(
            "http://10.0.0.1:8000/health",
            timeout=(4.0, 35.0),
        )


@patch("patio.topo.client.motor_topo_client.requests.get")
def test_wait_engine_ready_fails_on_non_200(mock_get, monkeypatch):
    _set_motor_env(monkeypatch)
    monkeypatch.setenv("POD_IP", "10.0.0.1")
    reload(envs)

    response = MagicMock()
    response.status_code = 503
    response.text = "not ready"
    mock_get.return_value = response

    with _patch_retry():
        client = MotorTopoClient({"type": "prefill", "port": 8000})
        assert client.wait_engine_ready({"port": 8000}) is False


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------

@patch("patio.topo.client.motor_topo_client.requests.post")
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
        client = MotorTopoClient({"type": "prefill", "port": 8000})
        assert client.register("", {"type": "prefill", "port": 8000})
        # Verify the POST went to the right URL with the right body
        call_args = mock_post.call_args
        assert call_args is not None
        url = call_args[0][0] if call_args[0] else call_args[1]["url"]
        assert "/instances/refresh" in url
        assert "demo-router-0.s-demo-router:9000" in url
        # env-configured model name is carried (no /v1/models probe needed)
        body = call_args[1]["json"]
        assert body["model_name"] == "test-model"


@patch("patio.topo.client.motor_topo_client.requests.post")
def test_register_resolves_model_name_from_engine(mock_post, monkeypatch):
    """register() probes the engine /v1/models when MOTOR_MODEL_NAME is unset,
    caches the name, and sends it in the request body."""
    _set_motor_env(monkeypatch)
    monkeypatch.setenv("POD_IP", "10.0.0.1")
    reload(envs)

    response = MagicMock()
    response.status_code = 200
    response.text = "ok"
    mock_post.return_value = response

    with _patch_retry(), patch(
        "patio.topo.client.motor_topo_client.requests.get"
    ) as mock_get:
        probe = MagicMock()
        probe.status_code = 200
        probe.json.return_value = {"data": [{"id": "Qwen2.5-7B"}]}
        mock_get.return_value = probe

        client = MotorTopoClient({"type": "prefill", "port": 8000})
        assert client.register("", {"type": "prefill", "port": 8000})

        # engine /v1/models probed once (health check not part of this test)
        model_urls = [
            c[0][0] for c in mock_get.call_args_list if "/v1/models" in c[0][0]
        ]
        assert len(model_urls) == 1
        assert model_urls[0] == "http://10.0.0.1:8000/v1/models"

        body = mock_post.call_args[1]["json"]
        assert body["model_name"] == "Qwen2.5-7B"
        # cached for unregister after the engine dies
        assert client._model_name == "Qwen2.5-7B"


@patch("patio.topo.client.motor_topo_client.requests.post")
def test_register_omits_model_name_when_probe_fails(mock_post, monkeypatch):
    """If the /v1/models probe fails, register proceeds without model_name and
    lets Motor fall back to its own probe."""
    _set_motor_env(monkeypatch)
    monkeypatch.setenv("POD_IP", "10.0.0.1")
    reload(envs)

    response = MagicMock()
    response.status_code = 200
    response.text = "ok"
    mock_post.return_value = response

    with _patch_retry(), patch(
        "patio.topo.client.motor_topo_client.requests.get"
    ) as mock_get:
        probe = MagicMock()
        probe.status_code = 404
        mock_get.return_value = probe

        client = MotorTopoClient({"type": "prefill", "port": 8000})
        assert client.register("", {"type": "prefill", "port": 8000})

        body = mock_post.call_args[1]["json"]
        assert "model_name" not in body
        assert client._model_name is None


@patch("patio.topo.client.motor_topo_client.requests.post")
def test_register_returns_false_on_error(mock_post, monkeypatch):
    _set_motor_env(monkeypatch)
    monkeypatch.setenv("POD_IP", "10.0.0.1")
    reload(envs)

    response = MagicMock()
    response.status_code = 500
    response.text = "bad"
    mock_post.return_value = response

    with _patch_retry():
        client = MotorTopoClient({"type": "prefill", "port": 8000})
        assert client.register("", {"type": "prefill", "port": 8000}) is False


# ---------------------------------------------------------------------------
# unregister
# ---------------------------------------------------------------------------

@patch("patio.topo.client.motor_topo_client.requests.post")
def test_unregister_posts_instance_del(mock_post, monkeypatch):
    _set_motor_env(monkeypatch)
    monkeypatch.setenv("POD_IP", "10.0.0.1")
    monkeypatch.setenv("MOTOR_MODEL_NAME", "test-model")
    reload(envs)

    response = MagicMock()
    response.status_code = 200
    response.text = "ok"
    mock_post.return_value = response

    with _patch_retry():
        client = MotorTopoClient({"type": "prefill", "port": 8000})
        assert client.unregister()
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        body = call_kwargs["json"]
        assert body["event"] == "del"
        # Motor requires model_name for del too; carried from env/cache
        assert body["model_name"] == "test-model"


@patch("patio.topo.client.motor_topo_client.requests.post")
def test_unregister_carries_cached_model_name_after_engine_death(mock_post, monkeypatch):
    """unregister() reuses the name resolved at register time even though the
    engine is now dead (Motor's own fallback probe would 400 against it)."""
    _set_motor_env(monkeypatch)
    monkeypatch.setenv("POD_IP", "10.0.0.1")
    reload(envs)

    response = MagicMock()
    response.status_code = 200
    response.text = "ok"
    mock_post.return_value = response

    with _patch_retry(), patch(
        "patio.topo.client.motor_topo_client.requests.get"
    ) as mock_get:
        probe = MagicMock()
        probe.status_code = 200
        probe.json.return_value = {"data": [{"id": "Qwen2.5-7B"}]}
        mock_get.return_value = probe

        client = MotorTopoClient({"type": "prefill", "port": 8000})
        # register() resolves and caches the model name (engine alive then)
        assert client.register("", {"type": "prefill", "port": 8000})
        mock_get.reset_mock()
        # engine dies; unregister must NOT probe /v1/models again
        assert client.unregister()
        mock_get.assert_not_called()

        body = mock_post.call_args[1]["json"]
        assert body["event"] == "del"
        assert body["model_name"] == "Qwen2.5-7B"
