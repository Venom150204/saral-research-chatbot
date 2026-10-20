#!/bin/sh
# Installs the entrypoint that keeps SARAL running. Runs once, at build time, as root.
set -e
user="${_REMOTE_USER:-vscode}"
cat > /usr/local/share/saral-autostart.sh <<EOF
#!/bin/sh
# Runs at every container start. Every 60 seconds, if SARAL does not answer on
# port 8000, run the repository's start script as the codespace user.
mkdir -p /tmp/saral-diag
echo "entrypoint ran at \$(date -u)" > /tmp/saral-autostart.log
(cd /tmp/saral-diag && python3 -m http.server 8001 --bind 0.0.0.0 >/dev/null 2>&1 &)
(
  sleep 30
  while true; do
    if ! curl -fs -m 5 -o /dev/null http://localhost:8000/health; then
      for dir in /workspaces/*/; do
        if [ -f "\${dir}.devcontainer/start.sh" ]; then
          echo "--- \$(date -u) running \${dir}.devcontainer/start.sh as $user" >> /tmp/saral-autostart.log
          su - "$user" -c "bash '\${dir}.devcontainer/start.sh'" >> /tmp/saral-autostart.log 2>&1
          break
        fi
      done
    fi
    {
      echo "time: \$(date -u)"
      echo "user for start: $user"
      for v in GEMINI_API_KEY SARAL_DATABASE_URL DATABASE_URL_UNPOOLED DATABASE_URL; do
        if tr '\0' '\n' < /proc/1/environ | grep -q "^\$v="; then echo "pid1 env has \$v: yes"; else echo "pid1 env has \$v: no"; fi
      done
      echo "env-secrets file: \$(ls -la /workspaces/.codespaces/shared/.env-secrets 2>&1 | cut -c1-60)"
      if [ -r /workspaces/.codespaces/shared/.env-secrets ]; then echo "names in env-secrets: \$(cut -d= -f1 /workspaces/.codespaces/shared/.env-secrets | tr '\n' ' ')"; fi
      echo "uv: \$(ls /home/$user/.local/bin/uv 2>&1)"
      echo "workspaces: \$(ls /workspaces 2>&1 | tr '\n' ' ')"
      echo "local health: \$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://localhost:8000/health)"
      echo "===== saral-autostart.log"; tail -n 40 /tmp/saral-autostart.log
      echo "===== saral.log"; tail -n 40 /tmp/saral.log 2>&1
    } | sed -E 's#postgres(ql)?://[^ ]+#postgres://[hidden]#g; s#AIza[0-9A-Za-z_-]+#[hidden-key]#g' > /tmp/saral-diag/index.txt
    sleep 60
  done
) > /dev/null 2>&1 &
exec "\$@"
EOF
chmod 755 /usr/local/share/saral-autostart.sh
