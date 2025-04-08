#!/bin/bash

# Change to the mattermost directory
cd "$(dirname "$0")"

# Stop the servers if pid files exist
if [ -f pids/app.pid ]; then
    echo "Stopping RAG pipeline server..."
    kill $(cat pids/app.pid) 2>/dev/null || true
    rm pids/app.pid
fi

if [ -f pids/bot.pid ]; then
    echo "Stopping Mattermost bot server..."
    kill $(cat pids/bot.pid) 2>/dev/null || true
    rm pids/bot.pid
fi

echo "Servers stopped." 