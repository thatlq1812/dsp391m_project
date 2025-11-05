"""
Page 9: API & Integration
Provide commands and guidance for running the FastAPI service and managing webhooks.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import streamlit as st

from dashboard.utils.command_blocks import show_command_block

st.set_page_config(page_title="API & Integration", page_icon="🔌", layout="wide")

st.title("API & Integration")
st.markdown("Run the FastAPI server, inspect endpoints, and manage webhooks.")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_NAME = os.environ.get("CONDA_ENV", "dsp")

TAB_SERVER, TAB_ENDPOINTS, TAB_WEBHOOKS, TAB_DOCS = st.tabs(
    [
        "API Server",
        "Endpoints",
        "Webhooks",
        "Documentation",
    ]
)

with TAB_SERVER:
    st.markdown("### FastAPI Server Control")
    col1, col2 = st.columns([2, 1])

    with col1:
        host = st.text_input("Host", "0.0.0.0")
        port = st.number_input("Port", min_value=1024, max_value=65535, value=8000)
        workers = st.slider("Workers", 1, 8, 4)
        reload_flag = st.checkbox("Enable reload (dev only)", value=False)
        base_command = [
            "uvicorn",
            "traffic_forecast.api.main:app",
            "--host",
            host,
            "--port",
            str(port),
            "--workers",
            str(workers),
        ]
        if reload_flag:
            base_command.append("--reload")
        st.code(" ".join(base_command), language="bash")

    with col2:
        if st.button("Prepare start command", type="primary", width='stretch'):
            show_command_block(
                [
                    "conda",
                    "run",
                    "-n",
                    ENV_NAME,
                    "--no-capture-output",
                    *base_command,
                ],
                cwd=PROJECT_ROOT,
                description="Run the FastAPI server from the project root.",
                success_hint="Keep the server terminal open while the API is in use.",
            )
            st.success("Command prepared. Execute it manually to start the server.")

        if st.button("Prepare stop instructions", width='stretch'):
            st.info("Focus the terminal running uvicorn and press Ctrl+C once to stop.")

    st.divider()
    st.markdown("#### Logs")
    if st.button("Tail server logs", width='stretch'):
        show_command_block(
            [
                "tail",
                "-n",
                "200",
                "logs/api_server.log",
            ],
            cwd=PROJECT_ROOT,
            description="Inspect the latest API server logs (if logging to file is enabled).",
        )

with TAB_ENDPOINTS:
    st.markdown("### REST Endpoints")
    endpoints = [
        {"Method": "GET", "Path": "/api/v1/health", "Description": "Health check", "Auth": "None"},
        {"Method": "GET", "Path": "/api/v1/models", "Description": "List model runs", "Auth": "API Key"},
        {"Method": "POST", "Path": "/api/v1/predict", "Description": "Run inference", "Auth": "API Key"},
        {"Method": "GET", "Path": "/api/v1/data/stats", "Description": "Dataset stats", "Auth": "API Key"},
        {"Method": "POST", "Path": "/api/v1/collect", "Description": "Trigger collection", "Auth": "API Key"},
        {"Method": "GET", "Path": "/api/v1/metrics", "Description": "Service metrics", "Auth": "Admin"},
    ]
    st.dataframe(endpoints, hide_index=True, width='stretch')

    st.divider()
    st.markdown("#### Request Builder")
    endpoint = st.selectbox("Endpoint", [e["Path"] for e in endpoints])
    method = next(e["Method"] for e in endpoints if e["Path"] == endpoint)
    st.code(f"{method} http://localhost:{port}{endpoint}", language="http")

    if method == "POST":
        payload = st.text_area("Request body (JSON)", "{}")
    else:
        payload = None

    if st.button("Prepare curl command", width='stretch'):
        command = [
            "curl",
            "-X",
            method,
            f"http://localhost:{port}{endpoint}",
            "-H",
            "X-API-Key: <your-key>",
        ]
        if payload:
            command.extend(["-H", "Content-Type: application/json", "-d", payload])
        show_command_block(
            command,
            cwd=PROJECT_ROOT,
            description="Sample curl command (replace `<your-key>` before executing).",
        )

with TAB_WEBHOOKS:
    st.markdown("### Webhook Management")
    st.info("Store webhook definitions in `configs/webhooks.json` for automation.")

    webhook_name = st.text_input("Webhook name", "Slack Notifications")
    events = st.multiselect(
        "Events",
        [
            "model.trained",
            "data.collected",
            "prediction.completed",
            "error.occurred",
            "alert.triggered",
        ],
        default=["model.trained", "error.occurred"],
    )
    url = st.text_input("Webhook URL", "https://hooks.slack.com/...")

    if st.button("Print webhook payload", width='stretch'):
        payload = {
            "name": webhook_name,
            "url": url,
            "events": events,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        st.code(json.dumps(payload, indent=2), language="json")

    st.markdown("#### Test Command")
    if st.button("Prepare webhook test", width='stretch'):
        show_command_block(
            [
                "python",
                "tools/test_webhook.py",
                "--name",
                webhook_name,
            ],
            cwd=PROJECT_ROOT,
            description="Replace with the actual webhook test script when available.",
        )

with TAB_DOCS:
    st.markdown("### API Documentation Links")
    st.markdown(
        "- Swagger UI: http://localhost:8000/docs\n"
        "- ReDoc: http://localhost:8000/redoc\n"
        "- OpenAPI JSON: http://localhost:8000/openapi.json"
    )

    if st.button("Generate API key", width='stretch'):
        import secrets

        api_key = secrets.token_urlsafe(32)
        st.code(api_key)
        st.success("API key generated. Store it securely in your secret manager.")

    st.markdown("#### Python Client Template")
    st.code(
        """
import requests

headers = {"X-API-Key": "<your-key>"}
response = requests.post(
    "http://localhost:8000/api/v1/predict",
    headers=headers,
    json={"edge_ids": [0, 1, 2], "timesteps": 12},
)
response.raise_for_status()
print(response.json())
        """.strip(),
        language="python",
    )

st.divider()
st.caption("Tip: run the API server in a dedicated terminal before starting the Streamlit dashboard.")
