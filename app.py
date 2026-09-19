# -*- coding: utf-8 -*-
# @Author: zibai.gj
import argparse
import json
import signal
import sys
import threading
import time
import traceback
from urllib.parse import urljoin
from prometheus_client import REGISTRY

from patio import envs
from patio.api.server_router import server_router
from patio.api.lora_router import lora_router
from patio.config import DEFAULT_PATIO_PORT, EXCLUDE_ENDPOINTS, EXIT_EVENT
from patio.logger import configure_logging, init_logger

from patio.metrics.engine_collector import EngineCollector
from patio.metrics.engine_metric_rules import get_metric_standard_rules
from patio.metrics.metrics import REQUEST_COUNT, REQUEST_LATENCY, patio_registry

from patio.topo.client.base_topo_client import GroupTopoClient

from fastapi import FastAPI, Request
import uvicorn

from patio.topo.factory import create_topo_client

app = FastAPI(title="Patio Server", debug=False)

# add router
app.include_router(server_router)
app.include_router(lora_router)

topo_client: GroupTopoClient = None
motor_client = None

# Middleware to collect metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    method = request.method
    endpoint = request.scope.get("path")

    # skip metrics endpoints
    for exclude_endpoint in EXCLUDE_ENDPOINTS:
        if endpoint.startswith(exclude_endpoint):
            response = await call_next(request)
            return response

    start_time = time.perf_counter()
    response = await call_next(request)
    status = response.status_code
    duration = time.perf_counter() - start_time

    # Update metrics
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
    return response


def _init_metrics(scrape_metrics: bool):
    if scrape_metrics and envs.INFERENCE_ENGINE_ENDPOINT:
        scrape_endpoint = urljoin(envs.INFERENCE_ENGINE_ENDPOINT, envs.METRIC_SCRAPE_PATH)
        engine_collector = EngineCollector(scrape_endpoint, get_metric_standard_rules(envs.INFERENCE_ENGINE))
        patio_registry.register(engine_collector)


def stop_topo_client_signal_handler(signal, frame):
    global topo_client
    if topo_client is not None:
        topo_client.unregister()
    sys.exit(0)

def run_topo_client(worker_instance_info: str) -> GroupTopoClient:
    logger = init_logger(__name__)
    logger.info("Starting Patio TopoClient...")
    logger.info(f"worker instance info: {worker_instance_info}")

    worker_dict = None
    try:
        worker_dict = json.loads(worker_instance_info)
    except json.decoder.JSONDecodeError as e:
        logger.error(f"Failed to decode worker instance info: {worker_instance_info}: {e}")
        sys.exit(1)

    if not worker_dict.get("topo_type"):
        topo_type = envs.TOPO_TYPE
        if not topo_type:
            logger.error(f"No topo_type defined for worker instance, either --instance-info or env variable TOPO_TYPE should be set")
            sys.exit(1)
        logger.info(f"Found topo type from env variable TOPO_TYPE: {topo_type}")
        worker_dict["topo_type"] = topo_type

    try:
        global topo_client
        worker_info = worker_dict["data"]
        if worker_info is None:
            logger.error(f"No worker info found for worker instance info, please set \"data\" field in instance info")
            sys.exit(1)
        topo_client = create_topo_client(worker_dict["topo_type"], worker_info)
        topo_client.wait_engine_ready(worker_info)
        topo_client.register("", worker_info)
        signal.signal(signal.SIGTERM, stop_topo_client_signal_handler)
        signal.signal(signal.SIGINT, stop_topo_client_signal_handler)

        # Start a heartbeat thread to periodically re-register with the router,
        # ensuring workers automatically recover registration after a router restart.
        def _heartbeat_loop():
            interval = int(envs.HEARTBEAT_INTERVAL)
            while not EXIT_EVENT.wait(timeout=interval):
                try:
                    topo_client.register("", worker_info)
                    logger.debug("heartbeat register ok")
                except Exception as e:
                    logger.warning(f"heartbeat register failed: {e}")

        t = threading.Thread(target=_heartbeat_loop, daemon=True, name="patio-heartbeat")
        t.start()
        logger.info(f"heartbeat thread started, interval={envs.HEARTBEAT_INTERVAL}s")

        return topo_client
    except Exception as e:
        logger.error(f"Failed to start Patio TopoClient: {e}")
        sys.exit(1)


def stop_motor_client_signal_handler(signal, frame):
    global motor_client
    if motor_client is not None:
        motor_client.unregister()
    sys.exit(0)


def run_motor_integration(worker_instance_info: str):
    """Start the MindIE Motor Coordinator integration client.

    Parses the same --instance-info JSON, creates a MotorCoordinatorClient,
    waits for the engine to be ready, registers the instance, and maintains
    a heartbeat-driven registration loop.

    This is an independent control-plane integration (not engine topology):
    it is enabled separately via the MOTOR_COORDINATOR_ENDPOINT env var and
    can coexist with the engine-native router registration.
    """
    logger = init_logger(__name__)
    logger.info("Starting MindIE Motor integration...")

    worker_dict = None
    try:
        worker_dict = json.loads(worker_instance_info)
    except json.decoder.JSONDecodeError as e:
        logger.error(f"Failed to decode worker instance info: {worker_instance_info}: {e}")
        sys.exit(1)

    worker_info = worker_dict.get("data")
    if worker_info is None:
        logger.error(f"No worker info found, please set \"data\" field in instance info")
        sys.exit(1)

    try:
        global motor_client
        from patio.integration.motor import MotorCoordinatorClient
        motor_client = MotorCoordinatorClient(worker_info)
        motor_client.wait_engine_ready(worker_info)
        motor_client.register("", worker_info)
        signal.signal(signal.SIGTERM, stop_motor_client_signal_handler)
        signal.signal(signal.SIGINT, stop_motor_client_signal_handler)

        # Heartbeat: periodic re-register. This also covers Coordinator
        # restart recovery (standalone mode has no Controller to re-push state).
        def _motor_heartbeat_loop():
            interval = int(envs.HEARTBEAT_INTERVAL)
            while not EXIT_EVENT.wait(timeout=interval):
                try:
                    motor_client.register("", worker_info)
                    logger.debug("motor heartbeat register ok")
                except Exception as e:
                    logger.warning(f"motor heartbeat register failed: {e}")

        t = threading.Thread(target=_motor_heartbeat_loop, daemon=True, name="motor-heartbeat")
        t.start()
        logger.info(f"motor heartbeat thread started, interval={envs.HEARTBEAT_INTERVAL}s")

        return motor_client
    except Exception as e:
        logger.error(f"Failed to start Motor integration: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run patio runtime server")
    parser.add_argument(
        "--enable-fastapi-docs",
        action="store_true",
        default=False,
        help="Enable FastAPI docs",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to listen on")
    parser.add_argument("--port", type=int, default=DEFAULT_PATIO_PORT, help="Port to listen on")
    parser.add_argument("--instance-info", type=str, default="", help="instance info")
    parser.add_argument("--scrape-engine-metrics", type=bool, default=True, help="Enable to scrape engine metrics")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    args = parser.parse_args()
    configure_logging(log_level=args.log_level)
    logger = init_logger(__name__)
    logger.info("Use %s to start up runtime server", args)

    _init_metrics(args.scrape_engine_metrics)

    if args.instance_info:
        run_topo_client(args.instance_info)
        # If a Motor Coordinator endpoint is configured, also register the
        # engine instance with it. Both registrations can coexist: the engine
        # topology goes to the native router, the Motor integration registers
        # with a separate control plane.
        if envs.MOTOR_COORDINATOR_ENDPOINT:
            run_motor_integration(args.instance_info)

    # Run the server
    try:
        uvicorn.run(
            app=app,
            host=args.host,
            port=args.port,
            reload=False
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        traceback.print_exception(e)
        sys.exit(1)


if __name__ == "__main__":
    main()