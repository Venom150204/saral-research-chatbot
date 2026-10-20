#!/bin/sh
# Installs the entrypoint that keeps SARAL running. Runs once, at build time, as root.
set -e
user="${_REMOTE_USER:-vscode}"
cat > /usr/local/share/saral-autostart.sh <<EOF
#!/bin/sh
# Runs at every container start. Every 60 seconds, if SARAL does not answer on
# port 8000, run the repository's start script as the codespace user.
(
  sleep 30
  while true; do
    if ! curl -fs -m 5 -o /dev/null http://localhost:8000/health; then
      for dir in /workspaces/*/; do
        if [ -f "\${dir}.devcontainer/start.sh" ]; then
          su - "$user" -c "bash '\${dir}.devcontainer/start.sh'" >> /tmp/saral-autostart.log 2>&1
          break
        fi
      done
    fi
    sleep 60
  done
) > /dev/null 2>&1 &
exec "\$@"
EOF
chmod 755 /usr/local/share/saral-autostart.sh
