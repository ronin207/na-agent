#!/bin/bash

# Change to the mattermost directory
cd "$(dirname "$0")"

# Stop ngrok if pid file exists
if [ -f pids/ngrok.pid ]; then
    echo "Stopping ngrok..."
    kill $(cat pids/ngrok.pid) 2>/dev/null || true
    rm pids/ngrok.pid
fi

echo "Ngrok stopped." 