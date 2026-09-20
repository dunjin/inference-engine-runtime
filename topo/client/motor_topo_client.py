# -*- coding: utf-8 -*-
# @Author: dunjin

"""
MotorTopoClient — register/unregister inference engine instances with a
standalone MindIE Motor Coordinator via its /instances/refresh management protocol.

MindIE Motor Coordinator is a routing/control-plane component of the SAME
abstraction class as the SGLang router and the vLLM proxy, so this client lives
under topo/client/ and implements the shared GroupTopoClient interface, exactly
like SGLangGroupTopoClient and VLLMProxyTopoClient.
"""

import os
import traceback
import zlib
from typing import Optional

import requests

from patio import envs
from patio.logger import init_logger
from patio.topo.client.base_topo_client import GroupTopoClient
from patio.topo import utils

logger = init_logger(__name__)

# Motor CLI-compatible instance-id derivation constants
STANDALONE_ID_NAMESPACE = 0x40000000
STANDALONE_ID_MASK = 0x3FFFFFFF

# Default dispatch plan accepted by Motor's external protocol.
# Only "prefill_handoff_decode" and "concurrent_engine_sync" are valid.
EXTERNAL_DISPATCH_PLAN = "prefill_handoff_decode"

# Engine type as accepted by Motor's ExternalInsEventMsg protocol.
# Motor standalone currently accepts "vllm" only.
MOTOR_ENGINE_TYPE = "vllm"

# Motor management API key header (see mgmt_server_modified.py MGMT_API_KEY_HEADER)
MOTOR_MGMT_API_KEY_HEADER = "X-Motor-Management-Key"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def get_motor_endpoint() -> str:
    """Derive the Motor Coordinator management endpoint from RBG env vars.

    Uses the same headless-service DNS shape as the SGLang router / vLLM proxy
    endpoints (ROUTER_ROLE_NAME / ROUTER_PORT), keeping Motor a first-class
    routing component rather than a side-channel integration.
    """
    rbg_group_name = envs.GROUP_NAME
    if rbg_group_name is None:
        raise RuntimeError("RBG_GROUP_NAME is not set")

    router_role_name = envs.ROUTER_ROLE_NAME
    if router_role_name is None:
        raise RuntimeError("ROUTER_ROLE_NAME is not set")

    router_port = envs.ROUTER_PORT
    if router_port is None:
        raise RuntimeError("ROUTER_PORT is not set")

    return f"{rbg_group_name}-{router_role_name}-0.s-{rbg_group_name}-{router_role_name}:{router_port}"


def get_mgmt_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    api_key_file = envs.MOTOR_MGMT_API_KEY_FILE
    if api_key_file:
        try:
            with open(api_key_file, "r") as f:
                api_key = f.read().strip()
            if api_key:
                headers[MOTOR_MGMT_API_KEY_HEADER] = api_key
        except OSError as e:
            logger.warning("failed to read motor mgmt api key file %s: %s", api_key_file, e)
    return headers


def detect_worker_type(worker_info: dict) -> str:
    """Extract the worker role (prefill / decode) from worker_info."""
    worker_type = worker_info.get("type") or worker_info.get("worker_type")
    if worker_type is None:
        raise ValueError("worker type is not set; provide 'type' or 'worker_type' in instance info")
    return worker_type


def get_worker_address(worker_info: dict) -> str:
    """Return the worker's routable address (host:port).

    Resolution order:
    1. 'instance' field (from --instance-info JSON)
    2. POD_IP env var + worker port
    3. RBG headless DNS (group-role-index.s-group-role:port)
    """
    if worker_info.get("instance"):
        return worker_info["instance"]

    port = str(worker_info.get("port", "8000"))
    pod_ip = os.getenv("POD_IP")
    if pod_ip:
        return "{}:{}".format(pod_ip, port)

    group_name = os.getenv("RBG_GROUP_NAME")
    role_name = os.getenv("RBG_ROLE_NAME")
    role_index = os.getenv("RBG_ROLE_INDEX")
    if group_name and role_name and role_index:
        return "{}-{}-{}.s-{}-{}:{}".format(group_name, role_name, role_index, group_name, role_name, port)

    raise RuntimeError("cannot determine worker address: set 'instance' field, POD_IP, or RBG env vars")


def derive_instance_id(role: str, address: str) -> int:
    """Deterministic instance ID matching Motor's CLI scheme.

    Motor CLI (register.py, derive_instance_id):
        identity = "role|ip:port"
        digest = crc32(identity) & 0x3FFFFFFF
        return 0x40000000 | digest
    """
    identity = "{}|{}".format(role, address.strip())
    digest = zlib.crc32(identity.encode()) & STANDALONE_ID_MASK
    return STANDALONE_ID_NAMESPACE | digest


def build_body(event: str, role: str, address: str, instance_id: int,
               model_name: Optional[str] = None) -> dict:
    """Build an ExternalInsEventMsg-compatible request body."""
    body = {
        "event": event,
        "engine_type": MOTOR_ENGINE_TYPE,
        "dispatch_capabilities": EXTERNAL_DISPATCH_PLAN,
        "instances": [
            {
                "id": instance_id,
                "role": role,
                "endpoints": [
                    {
                        "id": 0,
                        "address": address.strip(),
                    }
                ],
            }
        ],
    }
    if model_name:
        body["model_name"] = model_name
    return body


def _build_health_check_endpoint(worker_info: dict) -> str:
    """Determine the local health-check address.

    Uses POD_IP if available, otherwise falls back to localhost.
    """
    port = str(worker_info.get("port", "8000"))
    pod_ip = os.getenv("POD_IP")
    return "{}:{}".format(pod_ip, port) if pod_ip else "localhost:{}".format(port)


# ---------------------------------------------------------------------------
# MotorTopoClient
# ---------------------------------------------------------------------------


class MotorTopoClient(GroupTopoClient):
    """Register / unregister / health-check an inference engine instance with
    a standalone MindIE Motor Coordinator.

    Motor Coordinator is a routing/control-plane component of the same class
    as the SGLang router and the vLLM proxy, so this is a GroupTopoClient just
    like its topo/client/ siblings.
    """

    def __init__(self, worker_info: dict):
        self.worker_info = worker_info
        self.mgmt_endpoint = get_motor_endpoint()
        self.mgmt_headers = get_mgmt_headers()
        self.role = detect_worker_type(worker_info).strip().lower()
        self.address = get_worker_address(worker_info)
        self.instance_id = derive_instance_id(self.role, self.address)
        self.health_check_endpoint = _build_health_check_endpoint(worker_info)

        self._model_name = envs.MOTOR_MODEL_NAME
        logger.info(
            "MotorTopoClient: role=%s address=%s instance_id=%d endpoint=%s",
            self.role, self.address, self.instance_id, self.mgmt_endpoint,
        )

    def _resolve_model_name(self) -> Optional[str]:
        """Resolve the engine model name for Coordinator registration.

        Priority: MOTOR_MODEL_NAME env > probe the local engine /v1/models
        (OpenAI-compatible, same endpoint Motor itself falls back to). The
        resolved name is cached in self._model_name so unregister() still
        carries it after the engine goes away — Motor's own fallback probe
        would fail against a dead engine (del would 400).
        """
        if self._model_name:
            return self._model_name
        try:
            resp = requests.get(
                "http://{}/v1/models".format(self.health_check_endpoint),
                timeout=(envs.TOPO_CONNECT_TIMEOUT, envs.TOPO_HEALTH_CHECK_TIMEOUT),
            )
            if resp.status_code != 200:
                logger.warning(
                    "engine /v1/models returned status=%s, keep model_name unset",
                    resp.status_code,
                )
                return None
            data = resp.json().get("data") or []
            ids = [m["id"] for m in data if isinstance(m, dict) and m.get("id")]
            if len(ids) == 1:
                self._model_name = ids[0]
                logger.info("resolved engine model name from /v1/models: %s", self._model_name)
                return self._model_name
            logger.warning(
                "engine /v1/models returned %d model ids, expected exactly 1; "
                "keep model_name unset", len(ids),
            )
        except Exception as e:
            logger.warning("failed to resolve model name from engine /v1/models: %s", e)
        return None

    # -----------------------------------------------------------------------
    # GroupTopoClient interface
    # -----------------------------------------------------------------------

    def wait_engine_ready(self, worker_info: dict) -> bool:
        """Poll the engine /health endpoint until it returns 200."""
        def _check():
            try:
                resp = requests.get(
                    "http://" + self.health_check_endpoint + "/health",
                    timeout=(envs.TOPO_CONNECT_TIMEOUT, envs.TOPO_HEALTH_CHECK_TIMEOUT),
                )
                if resp.status_code == 200:
                    logger.info("Health check OK, inference engine is now ready.")
                    return True
                raise RuntimeError(
                    "health check failed, url: {}, status_code: {}, content: {}".format(
                        self.health_check_endpoint, resp.status_code, resp.text
                    )
                )
            except requests.RequestException as e:
                raise RuntimeError("health check request failed: {}".format(e))

        try:
            utils.retry(_check, retry_times=60, interval=3)
            return True
        except Exception as e:
            logger.error("failed to wait for engine ready: %s", e)
            traceback.print_exc()
            return False

    def register(self, url: str, worker_info: dict,
                 file_path: Optional[str] = None) -> bool:
        """Register the engine instance with the Motor Coordinator.

        Posts to /instances/refresh with event=add.
        Idempotent when the instance-id already exists on the Coordinator.
        """
        # Engine is guaranteed alive here (wait_engine_ready ran first), so
        # resolve/cache the model name now — unregister() will reuse it after
        # the engine is gone.
        self._resolve_model_name()
        body = build_body(
            event="add",
            role=self.role,
            address=self.address,
            instance_id=self.instance_id,
            model_name=self._model_name,
        )

        def _do_register():
            mgmt_url = "http://{}/instances/refresh".format(self.mgmt_endpoint)
            logger.info(
                "registering instance to Motor Coordinator: %s  body=%s",
                mgmt_url, body,
            )
            resp = requests.post(
                mgmt_url,
                json=body,
                headers=self.mgmt_headers,
                timeout=(envs.TOPO_CONNECT_TIMEOUT, envs.TOPO_REGISTER_TIMEOUT),
            )
            if not 200 <= resp.status_code < 300:
                raise RuntimeError(
                    "register failed, url: {}, status_code: {}, content: {}".format(
                        mgmt_url, resp.status_code, resp.text
                    )
                )
            logger.info("registered instance %d (%s) successfully", self.instance_id, self.role)

        try:
            utils.retry(_do_register, retry_times=60, interval=3)
            return True
        except Exception as e:
            logger.error("failed to register instance: %s", e)
            traceback.print_exc()
            return False

    def unregister(self):
        """Deregister the engine instance from the Motor Coordinator.

        Posts to /instances/refresh with event=del.
        Carries the cached model name (resolved at register time): Motor
        requires model_name for del too, and its fallback /v1/models probe
        would fail against the already-dead engine.
        """
        body = build_body(
            event="del",
            role=self.role,
            address=self.address,
            instance_id=self.instance_id,
            model_name=self._model_name,
        )

        def _do_unregister():
            mgmt_url = "http://{}/instances/refresh".format(self.mgmt_endpoint)
            logger.info(
                "unregistering instance from Motor Coordinator: %s  body=%s",
                mgmt_url, body,
            )
            resp = requests.post(
                mgmt_url,
                json=body,
                headers=self.mgmt_headers,
                timeout=(envs.TOPO_CONNECT_TIMEOUT, envs.TOPO_REGISTER_TIMEOUT),
            )
            if not 200 <= resp.status_code < 300:
                raise RuntimeError(
                    "unregister failed, url: {}, status_code: {}, content: {}".format(
                        mgmt_url, resp.status_code, resp.text
                    )
                )
            logger.info("unregistered instance %d (%s) successfully", self.instance_id, self.role)

        try:
            utils.retry(_do_unregister, retry_times=60, interval=3)
            return True
        except Exception as e:
            logger.error("failed to unregister instance: %s", e)
            traceback.print_exc()
            return False
