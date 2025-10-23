#!/usr/bin/env python3
import requests
import time
import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional
import argparse


class OpenAIStatusMonitor:
    def __init__(self, check_interval: int = 60, alert_method: str = "console"):
        self.base_url = "https://status.openai.com/api/v2"
        self.check_interval = check_interval
        self.alert_method = alert_method
        self.last_status = None
        self.last_incidents = set()
        
    def get_status(self) -> Optional[Dict]:
        try:
            response = requests.get(f"{self.base_url}/summary.json", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching status: {e}")
            return None
    
    def get_incidents(self) -> Optional[Dict]:
        try:
            response = requests.get(f"{self.base_url}/incidents.json", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching incidents: {e}")
            return None
    
    def check_for_issues(self, status_data: Dict, incidents_data: Dict) -> List[str]:
        alerts = []
        
        overall_status = status_data.get("status", {})
        indicator = overall_status.get("indicator", "none")
        description = overall_status.get("description", "Unknown")
        
        if indicator != "none":
            alerts.append(f"⚠️  OVERALL STATUS ALERT: {description} (Indicator: {indicator})")
        
        components = status_data.get("components", [])
        for component in components:
            if component.get("status") != "operational":
                component_name = component.get("name", "Unknown")
                component_status = component.get("status", "unknown")
                alerts.append(f"🔴 Component Issue: {component_name} is {component_status}")
        
        incidents = incidents_data.get("incidents", [])
        for incident in incidents:
            incident_id = incident.get("id")
            incident_status = incident.get("status")
            
            if incident_status in ["investigating", "identified", "monitoring"] and incident_id not in self.last_incidents:
                incident_name = incident.get("name", "Unknown incident")
                impact = incident.get("impact", "unknown")
                created_at = incident.get("created_at", "")
                alerts.append(f"🚨 NEW INCIDENT: {incident_name}")
                alerts.append(f"   Impact: {impact} | Status: {incident_status}")
                alerts.append(f"   Created: {created_at}")
                self.last_incidents.add(incident_id)
        
        return alerts
    
    def send_console_alert(self, alerts: List[str]):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*60}")
        print(f"ALERT at {timestamp}")
        print(f"{'='*60}")
        for alert in alerts:
            print(alert)
        print(f"{'='*60}\n")
    
    def send_email_alert(self, alerts: List[str]):
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        sender_email = os.getenv("SENDER_EMAIL")
        sender_password = os.getenv("SENDER_PASSWORD")
        recipient_email = os.getenv("RECIPIENT_EMAIL")
        
        if not all([sender_email, sender_password, recipient_email]):
            print("Email configuration missing. Set SENDER_EMAIL, SENDER_PASSWORD, and RECIPIENT_EMAIL environment variables.")
            return
        
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = "OpenAI Status Alert"
        
        body = "\n".join(alerts)
        msg.attach(MIMEText(body, "plain"))
        
        try:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            server.quit()
            print(f"Email alert sent to {recipient_email}")
        except Exception as e:
            print(f"Failed to send email: {e}")
    
    def send_slack_alert(self, alerts: List[str]):
        webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        
        if not webhook_url:
            print("Slack webhook URL not configured. Set SLACK_WEBHOOK_URL environment variable.")
            return
        
        message = "\n".join(alerts)
        payload = {
            "text": f"*OpenAI Status Alert*\n{message}"
        }
        
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            print("Slack alert sent successfully")
        except Exception as e:
            print(f"Failed to send Slack alert: {e}")
    
    def send_alert(self, alerts: List[str]):
        if self.alert_method == "console":
            self.send_console_alert(alerts)
        elif self.alert_method == "email":
            self.send_email_alert(alerts)
        elif self.alert_method == "slack":
            self.send_slack_alert(alerts)
        elif self.alert_method == "all":
            self.send_console_alert(alerts)
            self.send_email_alert(alerts)
            self.send_slack_alert(alerts)
    
    def run(self):
        print(f"Starting OpenAI Status Monitor...")
        print(f"Check interval: {self.check_interval} seconds")
        print(f"Alert method: {self.alert_method}")
        print(f"Monitoring started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)
        
        while True:
            try:
                status_data = self.get_status()
                incidents_data = self.get_incidents()
                
                if status_data and incidents_data:
                    alerts = self.check_for_issues(status_data, incidents_data)
                    
                    if alerts:
                        self.send_alert(alerts)
                    else:
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"[{current_time}] ✅ All systems operational")
                    
                    self.last_status = status_data
                
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                time.sleep(self.check_interval)


def main():
    parser = argparse.ArgumentParser(description="Monitor OpenAI status page for outages and degradations")
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Check interval in seconds (default: 60)"
    )
    parser.add_argument(
        "--alert",
        choices=["console", "email", "slack", "all"],
        default="console",
        help="Alert method: console, email, slack, or all (default: console)"
    )
    
    args = parser.parse_args()
    
    monitor = OpenAIStatusMonitor(
        check_interval=args.interval,
        alert_method=args.alert
    )
    monitor.run()


if __name__ == "__main__":
    main()
