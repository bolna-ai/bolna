#!/usr/bin/env python3
from monitor import OpenAIStatusMonitor


monitor = OpenAIStatusMonitor(
    check_interval=60,
    alert_method="console"
)

monitor.run()
