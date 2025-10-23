#!/usr/bin/env python3
import json
from monitor import OpenAIStatusMonitor


def test_incident_detection():
    print("Testing incident detection...")
    
    monitor = OpenAIStatusMonitor(check_interval=60, alert_method="console")
    
    mock_status = {
        "page": {
            "id": "test",
            "name": "OpenAI",
            "url": "https://status.openai.com/",
            "updated_at": "2025-10-23T00:00:00Z"
        },
        "status": {
            "description": "Partial System Outage",
            "indicator": "major"
        },
        "components": [
            {
                "id": "1",
                "name": "Chat Completions",
                "status": "degraded_performance"
            },
            {
                "id": "2",
                "name": "Images",
                "status": "operational"
            }
        ]
    }
    
    mock_incidents = {
        "page": {
            "id": "test",
            "name": "OpenAI",
            "url": "https://status.openai.com/"
        },
        "incidents": [
            {
                "id": "test-incident-1",
                "name": "Elevated error rates on Chat Completions API",
                "status": "investigating",
                "created_at": "2025-10-23T08:00:00Z",
                "impact": "major"
            }
        ]
    }
    
    alerts = monitor.check_for_issues(mock_status, mock_incidents)
    
    print(f"\nFound {len(alerts)} alerts:")
    for alert in alerts:
        print(f"  - {alert}")
    
    if len(alerts) > 0:
        print("\n✅ Test passed: Incident detection is working correctly!")
    else:
        print("\n❌ Test failed: No alerts detected")


def test_operational_status():
    print("\nTesting operational status (no alerts expected)...")
    
    monitor = OpenAIStatusMonitor(check_interval=60, alert_method="console")
    
    mock_status = {
        "page": {
            "id": "test",
            "name": "OpenAI",
            "url": "https://status.openai.com/"
        },
        "status": {
            "description": "All Systems Operational",
            "indicator": "none"
        },
        "components": [
            {
                "id": "1",
                "name": "Chat Completions",
                "status": "operational"
            },
            {
                "id": "2",
                "name": "Images",
                "status": "operational"
            }
        ]
    }
    
    mock_incidents = {
        "page": {
            "id": "test",
            "name": "OpenAI"
        },
        "incidents": []
    }
    
    alerts = monitor.check_for_issues(mock_status, mock_incidents)
    
    if len(alerts) == 0:
        print("✅ Test passed: No false alerts when system is operational!")
    else:
        print(f"❌ Test failed: Unexpected alerts: {alerts}")


if __name__ == "__main__":
    test_operational_status()
    test_incident_detection()
