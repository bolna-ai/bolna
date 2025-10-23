#!/bin/bash

echo "OpenAI Status Monitor - Quick Start"
echo "===================================="
echo ""

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

echo "✅ Python 3 found"

if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

echo "🔧 Activating virtual environment..."
source venv/bin/activate

echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "Choose an option:"
echo "  1) Run with console alerts (default)"
echo "  2) Run with Slack alerts"
echo "  3) Run with email alerts"
echo "  4) Run with all alert methods"
echo "  5) Exit"
echo ""
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo "Starting monitor with console alerts..."
        python monitor.py
        ;;
    2)
        if [ -z "$SLACK_WEBHOOK_URL" ]; then
            echo ""
            read -p "Enter your Slack webhook URL: " webhook
            export SLACK_WEBHOOK_URL="$webhook"
        fi
        echo "Starting monitor with Slack alerts..."
        python monitor.py --alert slack
        ;;
    3)
        if [ -z "$SENDER_EMAIL" ]; then
            echo ""
            read -p "Enter sender email: " sender
            read -p "Enter sender password/app password: " -s password
            echo ""
            read -p "Enter recipient email: " recipient
            export SENDER_EMAIL="$sender"
            export SENDER_PASSWORD="$password"
            export RECIPIENT_EMAIL="$recipient"
        fi
        echo "Starting monitor with email alerts..."
        python monitor.py --alert email
        ;;
    4)
        echo "Starting monitor with all alert methods..."
        python monitor.py --alert all
        ;;
    5)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Starting with console alerts..."
        python monitor.py
        ;;
esac
