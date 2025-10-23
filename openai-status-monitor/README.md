# OpenAI Status Monitor

A simple Python application that monitors the OpenAI status page and alerts you when there's an outage or degradation event.

## Features

- 🔍 Monitors OpenAI's official status page via API
- 🚨 Detects overall system status changes
- 🔴 Tracks individual component status (Chat Completions, Images, Audio, etc.)
- 📊 Monitors active incidents and their status
- 📢 Multiple alert methods: Console, Email, Slack webhook
- ⏱️ Configurable check intervals
- 🔄 Continuous monitoring with automatic retries

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (Console Alerts)

Run the monitor with default settings (checks every 60 seconds, outputs to console):

```bash
python monitor.py
```

### Custom Check Interval

Check every 30 seconds:

```bash
python monitor.py --interval 30
```

### Email Alerts

To receive email alerts, set the following environment variables:

```bash
export SENDER_EMAIL="your-email@gmail.com"
export SENDER_PASSWORD="your-app-password"
export RECIPIENT_EMAIL="recipient@example.com"
export SMTP_SERVER="smtp.gmail.com"  # Optional, defaults to Gmail
export SMTP_PORT="587"  # Optional, defaults to 587
```

Then run:

```bash
python monitor.py --alert email
```

**Note for Gmail users:** You'll need to use an [App Password](https://support.google.com/accounts/answer/185833) instead of your regular password.

### Slack Alerts

To receive Slack alerts, create an [Incoming Webhook](https://api.slack.com/messaging/webhooks) and set:

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```

Then run:

```bash
python monitor.py --alert slack
```

### All Alert Methods

To use all alert methods simultaneously:

```bash
python monitor.py --alert all
```

## Command Line Options

```
--interval SECONDS    Check interval in seconds (default: 60)
--alert METHOD        Alert method: console, email, slack, or all (default: console)
```

## What Gets Monitored

The monitor tracks:

1. **Overall Status**: Detects when the overall system indicator changes from "none" (operational)
2. **Component Status**: Monitors all OpenAI services including:
   - Chat Completions
   - Images
   - Audio
   - Voice mode
   - Video viewing/generation
   - Deep Research
   - Realtime API
   - And more...
3. **Active Incidents**: Tracks new incidents with status:
   - Investigating
   - Identified
   - Monitoring
   - Resolved

## Example Output

When all systems are operational:
```
[2025-10-23 10:30:15] ✅ All systems operational
```

When an issue is detected:
```
============================================================
ALERT at 2025-10-23 10:35:22
============================================================
🚨 NEW INCIDENT: Elevated 503 errors when using GPT-4.1-Nano model
   Impact: minor | Status: investigating
   Created: 2025-10-22T20:38:01Z
🔴 Component Issue: Chat Completions is degraded_performance
============================================================
```

## Running as a Background Service

### Using screen (Linux/Mac)

```bash
screen -S openai-monitor
python monitor.py --interval 60 --alert slack
# Press Ctrl+A then D to detach
```

To reattach:
```bash
screen -r openai-monitor
```

### Using nohup (Linux/Mac)

```bash
nohup python monitor.py --interval 60 --alert email > monitor.log 2>&1 &
```

### Using systemd (Linux)

Create a service file at `/etc/systemd/system/openai-monitor.service`:

```ini
[Unit]
Description=OpenAI Status Monitor
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/openai-status-monitor
Environment="SLACK_WEBHOOK_URL=your-webhook-url"
ExecStart=/usr/bin/python3 /path/to/openai-status-monitor/monitor.py --interval 60 --alert slack
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable openai-monitor
sudo systemctl start openai-monitor
```

## API Endpoints Used

The monitor uses OpenAI's public status API:
- `https://status.openai.com/api/v2/summary.json` - Overall status and components
- `https://status.openai.com/api/v2/incidents.json` - Active and recent incidents

## Requirements

- Python 3.6+
- `requests` library

## License

MIT License - Feel free to use and modify as needed.
