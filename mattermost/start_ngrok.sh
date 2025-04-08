#!/bin/bash

# Change to the mattermost directory
cd "$(dirname "$0")"

# Create a pid directory if it doesn't exist
mkdir -p pids

# Kill any existing ngrok processes and remove pid file
if [ -f pids/ngrok.pid ]; then
    kill $(cat pids/ngrok.pid) 2>/dev/null || true
    rm pids/ngrok.pid
fi

# Start ngrok in the background with setsid
(setsid ngrok start agent > ngrok.log 2>&1 & echo $! > pids/ngrok.pid)

echo "Ngrok started in background. Check ngrok.log for output."
echo "To stop ngrok, run: ./stop_ngrok.sh" 