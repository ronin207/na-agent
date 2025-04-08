#!/bin/bash

# Change to the mattermost directory
cd "$(dirname "$0")"

# Create a pid directory if it doesn't exist
mkdir -p pids

# Kill any existing processes
if [ -f pids/app.pid ]; then
    kill $(cat pids/app.pid) 2>/dev/null || true
    rm pids/app.pid
fi

if [ -f pids/bot.pid ]; then
    kill $(cat pids/bot.pid) 2>/dev/null || true
    rm pids/bot.pid
fi

# Start the RAG pipeline server
(setsid python3 app.py > app.log 2>&1 & echo $! > pids/app.pid)

# Start the Mattermost bot server
(setsid python3 mattermost_bot.py > bot.log 2>&1 & echo $! > pids/bot.pid)

echo "Servers started in background. Check app.log and bot.log for output."
echo "To stop the servers, run: ./stop_servers.sh" 