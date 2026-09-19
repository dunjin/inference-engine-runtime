# -*- coding: utf-8 -*-
# @Author: dunjin

"""
MotorCoordinatorClient — register/unregister inference engine instances with a
standalone MindIE Motor Coordinator via its /instances/refresh management protocol.

This client lives under a dedicated integration/ package rather than topo/client/
because MindIE Motor Coordinator is a cross-engine control plane, not an inference-engine
topology router — the abstraction layers differ.
"""

import json
import os
import threading
import time
import traceback
import zlib
from typing import Optional

import requests

from patio import envs
from patio.logger import init_logger

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

# Retry policy for registration / health checks
_DEFAULT_RETRY_TIMES = 60
_DEFAULT_RETRY_INTERVAL = 3


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _detect_worker_type(worker_info: dict) -> str:
    """Extract the worker role (prefill / decode) from worker_info."""
    worker_type = worker_info.get("type") or worker_info.get("worker_type")
    if worker_type is None:
        raise ValueError("worker type is not set; provide 'type' or 'worker_type' in instance info")
    return worker_type


def _get_worker_address(worker_info: dict) -> str:
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


def _derive_instance_id(role: str, address: str) -> int:
    """Deterministic instance ID matching Motor's CLI scheme.

    Motor CLI (register.py, derive_instance_id):
        identity = "role|ip:port"
        digest = crc32(identity) & 0x3FFFFFFF
        return 0x40000000 | digest
    """
    identity = "{}|{}".format(role, address)
    digest = zlib.crc32(identity.encode()) & STANDALONE_ID_MASK
    return STANDALONE_ID_NAMESPACE | digest


def _get_mgmt_endpoint() -> str:
    endpoint = envs.MOTOR_COORDINATOR_ENDPOINT
    if not endpoint:
        raise RuntimeError("MOTOR_COORDINATOR_ENDPOINT is not set")
    port = envs.MOTOR_MGMT_PORT
    return "{}:{}".format(endpoint, port)


def _get_mgmt_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    api_key_file = envs.MOTOR_MGMT_API_KEY_FILE
    if api_key_file:
        try:
            with open(api_key_file, "r") as f:
                api_key = f.read().strip()
            if api_key:
                headers["X-Motor-Management-Key"] = api_key
        except OSError as e:
            logger.warning("failed to read motor mgmt api key file %s: %s", api_key_file, e)
    return headers


def _build_body(event: str, role: str, address: str, instance_id: int,
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
                        "address": address,
                    }
                ],
            }
        ],
    }
    if model_name:
        body["model_name"] = model_name
    return body


# ---------------------------------------------------------------------------
# retry helper
# ---------------------------------------------------------------------------


def _retry(fn, retry_times=_DEFAULT_RETRY_TIMES, interval=_DEFAULT_RETRY_INTERVAL):
    last_exc = None
    for attempt in range(1, retry_times + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < retry_times:
                logger.warning("attempt %d/%d failed: %s; retrying in %ds", attempt, retry_times, e, interval)
                time.sleep(interval)
    raise last_exc


# ---------------------------------------------------------------------------
# MotorCoordinatorClient
# ---------------------------------------------------------------------------


class MotorCoordinatorClient(object):
    """Register / unregister / health-check an inference engine instance with
    a standalone MindIE Motor Coordinator.

    This is NOT a GroupTopoClient: Motor Coordinator is a cross-engine control
    plane, not an inference-engine topology router.  The method signatures
    happen to overlap (wait_engine_ready, register, unregister) because both
    models need lifecycle hooks.
    """

    def __init__(self, worker_info: dict):
        self.worker_info = worker_info
        self.mgmt_endpoint = _get_mgmt_endpoint()
        self.mgmt_headers = _get_mgmt_headers()
        self.role = _detect_worker_type(worker_info).strip().lower()
        self.address = _get_worker_address(worker_info)
        self.instance_id = _derive_instance_id(self.role, self.address)
        self.health_check_endpoint = self._build_health_check_endpoint(worker_info)

        self._model_name = envs.MOTOR_MODEL_NAME
        logger.info(
            "MotorCoordinatorClient: role=%s address=%s instance_id=%d endpoint=%s",
            self.role, self.address, self.instance_id, self.mgmt_endpoint,
        )

    # -----------------------------------------------------------------------
    # public interface
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
            _retry(_check, retry_times=60, interval=3)
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
        body = _build_body(
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
                mgmt_url, json.dumps(body),
            )
            resp = requests.post(
                mgmt_url,
                json=body,
                headers=self.mgmt_headers,
                timeout=(envs.TOPO_CONNECT_TIMEOUT, envs.TOPO_REGISTER_TIMEOUT),
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    "register failed, url: {}, status_code: {}, content: {}".format(
                        mgmt_url, resp.status_code, resp.text
                    )
                )
            logger.info("registered instance %d (%s) successfully", self.instance_id, self.role)

        try:
            _retry(_do_register, retry_times=_DEFAULT_RETRY_TIMES, interval=_DEFAULT_RETRY_INTERVAL)
            return True
        except Exception as e:
            logger.error("failed to register instance: %s", e)
            traceback.print_exc()
            return False

    def unregister(self):
        """Deregister the engine instance from the Motor Coordinator.

        Posts to /instances/refresh with event=del.
        """
        body = _build_body(
            event="del",
            role=self.role,
            address=self.address,
            instance_id=self.instance_id,
        )

        def _do_unregister():
            mgmt_url = "http://{}/instances/refresh".format(self.mgmt_endpoint)
            logger.info(
                "unregistering instance from Motor Coordinator: %s  body=%s",
                mgmt_url, json.dumps(body),
            )
            resp = requests.post(
                mgmt_url,
                json=body,
                headers=self.mgmt_headers,
                timeout=(envs.TOPO_CONNECT_TIMEOUT, envs.TOPO_REGISTER_TIMEOUT),
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    "unregister failed, url: {}, status_code: {}, content: {}".format(
                        mgmt_url, resp.status_code, resp.text
                    )
                )
            logger.info("unregistered instance %d (%s) successfully", self.instance_id, self.role)

        try:
            _retry(_do_unregister, retry_times=_DEFAULT_RETRY_TIMES, interval=_DEFAULT_RETRY_INTERVAL)
            return True
        except Exception as e:
            logger.error("failed to unregister instance: %s", e)
            traceback.print_exc()
            return False

    # -----------------------------------------------------------------------
    # internals
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_health_check_endpoint(worker_info: dict) -> str:
        """Determine the local health-check address.

        Uses POD_IP if available, otherwise falls back to localhost.
        """
        port = str(worker_info.get("port", "8000"))
        pod_ip = os.getenv("POD_IP")
        return "{}:{}".format(pod_ip, port) if pod_ip else "localhost:{}".format(port)