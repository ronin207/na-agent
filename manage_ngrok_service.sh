#!/bin/bash

# Configuration
SERVICE_NAME="com.ngrok.mattermost"
PLIST_FILE="com.ngrok.mattermost.plist"
LAUNCHD_DIR="$HOME/Library/LaunchAgents"

# Ensure logs directory exists
mkdir -p logs

# Function to check if service is running
is_service_running() {
    launchctl list | grep -q "$SERVICE_NAME"
    return $?
}

# Function to load the service
load_service() {
    echo "Loading ngrok service..."
    cp "$PLIST_FILE" "$LAUNCHD_DIR/"
    launchctl load "$LAUNCHD_DIR/$PLIST_FILE"
    sleep 2
    if is_service_running; then
        echo "✅ Service loaded successfully"
    else
        echo "❌ Failed to load service"
    fi
}

# Function to unload the service
unload_service() {
    echo "Unloading ngrok service..."
    launchctl unload "$LAUNCHD_DIR/$PLIST_FILE" 2>/dev/null
    rm -f "$LAUNCHD_DIR/$PLIST_FILE"
    sleep 2
    if ! is_service_running; then
        echo "✅ Service unloaded successfully"
    else
        echo "❌ Failed to unload service"
    fi
}

# Function to restart the service
restart_service() {
    unload_service
    load_service
}

# Function to check service status
check_status() {
    if is_service_running; then
        echo "✅ ngrok service is running"
    else
        echo "❌ ngrok service is not running"
    fi
}

# Main script
case "$1" in
    "start")
        load_service
        ;;
    "stop")
        unload_service
        ;;
    "restart")
        restart_service
        ;;
    "status")
        check_status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac 