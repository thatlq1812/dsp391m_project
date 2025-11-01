"""
Page 12: API & Integration
FastAPI endpoints, webhooks, and external integrations
"""

import streamlit as st
from pathlib import Path
import subprocess
import json
from datetime import datetime

st.set_page_config(page_title="API & Integration", page_icon="🔌", layout="wide")

st.title("🔌 API & Integration")
st.markdown("Manage API endpoints and external integrations")

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "API Server",
    "Endpoints",
    "Webhooks",
    "API Documentation"
])

with tab1:
    st.markdown("### FastAPI Server Control")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Server Configuration")
        
        host = st.text_input("Host", "0.0.0.0")
        port = st.number_input("Port", min_value=1000, max_value=9999, value=8000)
        workers = st.slider("Workers", 1, 8, 4)
        reload = st.checkbox("Auto-reload (Development)", value=False)
        
        st.code(f"""
uvicorn traffic_forecast.api.main:app \\
    --host {host} \\
    --port {port} \\
    --workers {workers} \\
    {'--reload' if reload else ''}
        """, language="bash")
    
    with col2:
        st.markdown("#### Server Control")
        
        if st.button("Start API Server", width='stretch', type="primary"):
            st.success("API Server starting...")
            st.info(f"Server will be available at http://{host}:{port}")
            
            # In production, actually start the server
            # subprocess.Popen([...])
        
        if st.button("STOP Stop Server", width='stretch'):
            st.warning("WARNING API Server stopped")
        
        if st.button("Restart Server", width='stretch'):
            st.info("Restarting API Server...")
        
        st.divider()
        
        st.markdown("#### Server Status")
        st.metric("Status", "Offline")
        st.metric("Uptime", "0h 0m")
        st.metric("Requests", "0")
    
    st.divider()
    
    st.markdown("### API Logs")
    
    if st.button("Refresh Logs"):
        st.code("""
[2025-11-01 14:30:15] INFO: API Server started
[2025-11-01 14:30:20] INFO: GET /api/v1/predict - 200 OK (245ms)
[2025-11-01 14:30:25] INFO: POST /api/v1/predict - 200 OK (1.2s)
[2025-11-01 14:30:30] ERROR: POST /api/v1/train - 500 Error
        """, language="text")

with tab2:
    st.markdown("### API Endpoints")
    
    st.info("Available REST API endpoints for external integration")
    
    endpoints = [
        {
            "Method": "GET",
            "Path": "/api/v1/health",
            "Description": "Health check",
            "Auth": "No"
        },
        {
            "Method": "GET",
            "Path": "/api/v1/models",
            "Description": "List available models",
            "Auth": "API Key"
        },
        {
            "Method": "POST",
            "Path": "/api/v1/predict",
            "Description": "Generate predictions",
            "Auth": "API Key"
        },
        {
            "Method": "GET",
            "Path": "/api/v1/data/stats",
            "Description": "Data statistics",
            "Auth": "API Key"
        },
        {
            "Method": "POST",
            "Path": "/api/v1/collect",
            "Description": "Trigger data collection",
            "Auth": "API Key"
        },
        {
            "Method": "GET",
            "Path": "/api/v1/metrics",
            "Description": "System metrics",
            "Auth": "Admin"
        }
    ]
    
    st.dataframe(endpoints, hide_index=True, width='stretch')
    
    st.divider()
    
    # Endpoint testing
    st.markdown("### Test Endpoints")
    
    col1, col2 = st.columns(2)
    
    with col1:
        endpoint = st.selectbox("Select Endpoint", [e["Path"] for e in endpoints])
        method = next(e["Method"] for e in endpoints if e["Path"] == endpoint)
        
        st.code(f"{method} http://localhost:8000{endpoint}", language="http")
        
        if method == "POST":
            request_body = st.text_area("Request Body (JSON)", "{}")
        
        api_key = st.text_input("API Key", type="password")
    
    with col2:
        if st.button("Send Request", width='stretch'):
            st.info("Sending request...")
            
            # Simulated response
            response = {
                "status": "success",
                "data": {
                    "predictions": [45.2, 43.8, 42.1],
                    "confidence": [0.85, 0.82, 0.79]
                },
                "timestamp": datetime.now().isoformat()
            }
            
            st.json(response)

with tab3:
    st.markdown("### Webhook Management")
    
    st.info("Configure webhooks to receive notifications about system events")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Add Webhook")
        
        webhook_name = st.text_input("Webhook Name", "Slack Notifications")
        webhook_url = st.text_input("Webhook URL", "https://hooks.slack.com/...")
        
        webhook_events = st.multiselect(
            "Events to Subscribe",
            [
                "model.trained",
                "data.collected",
                "prediction.completed",
                "error.occurred",
                "alert.triggered"
            ],
            default=["model.trained", "error.occurred"]
        )
        
        if st.button("➕ Add Webhook"):
            st.success(f"Webhook '{webhook_name}' added")
    
    with col2:
        st.markdown("#### Active Webhooks")
        
        webhooks = [
            {
                "Name": "Slack Notifications",
                "URL": "https://hooks.slack.com/...",
                "Events": 2,
                "Status": "Active"
            },
            {
                "Name": "Discord Alerts",
                "URL": "https://discord.com/api/webhooks/...",
                "Events": 3,
                "Status": "Paused"
            }
        ]
        
        st.dataframe(webhooks, hide_index=True, width='stretch')
        
        if st.button("🧪 Test Webhook"):
            st.info("Sending test payload...")
            st.success("Test webhook sent successfully")

with tab4:
    st.markdown("### API Documentation")
    
    st.info("Interactive API documentation powered by OpenAPI/Swagger")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Quick Links")
        
        st.markdown("""
        - [Swagger UI](http://localhost:8000/docs) - Interactive API docs
        - [ReDoc](http://localhost:8000/redoc) - Alternative docs
        - [OpenAPI JSON](http://localhost:8000/openapi.json) - API specification
        """)
        
        if st.button("Open Swagger UI"):
            st.info("Opening Swagger UI in browser...")
    
    with col2:
        st.markdown("#### Authentication")
        
        st.code("""
# API Key Authentication
headers = {
    "X-API-Key": "your-api-key-here"
}

response = requests.get(
    "http://localhost:8000/api/v1/predict",
    headers=headers
)
        """, language="python")
        
        if st.button("Generate API Key"):
            import secrets
            api_key = secrets.token_urlsafe(32)
            st.code(api_key)
            st.success("API Key generated (save it securely)")
    
    st.divider()
    
    # Example usage
    st.markdown("### Example Usage")
    
    tab_py, tab_curl, tab_js = st.tabs(["Python", "cURL", "JavaScript"])
    
    with tab_py:
        st.code("""
import requests

# Make prediction
response = requests.post(
    "http://localhost:8000/api/v1/predict",
    headers={"X-API-Key": "your-api-key"},
    json={
        "edge_ids": [0, 1, 2],
        "timesteps": 12
    }
)

predictions = response.json()
print(predictions)
        """, language="python")
    
    with tab_curl:
        st.code("""
curl -X POST "http://localhost:8000/api/v1/predict" \\
  -H "X-API-Key: your-api-key" \\
  -H "Content-Type: application/json" \\
  -d '{
    "edge_ids": [0, 1, 2],
    "timesteps": 12
  }'
        """, language="bash")
    
    with tab_js:
        st.code("""
const response = await fetch(
  'http://localhost:8000/api/v1/predict',
  {
    method: 'POST',
    headers: {
      'X-API-Key': 'your-api-key',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      edge_ids: [0, 1, 2],
      timesteps: 12
    })
  }
);

const predictions = await response.json();
console.log(predictions);
        """, language="javascript")

st.divider()
st.caption("Tip: Secure your API with proper authentication and rate limiting")
