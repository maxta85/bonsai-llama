#!/bin/bash
# Manage multiple Google Colab CLI accounts.
#
# Usage:
#   ./scripts/colab_account.sh save <N>    Save current token as account N
#   ./scripts/colab_account.sh <N>         Switch to account N
#   ./scripts/colab_account.sh list        List all saved accounts
#   ./scripts/colab_account.sh current     Show which account is active
#   ./scripts/colab_account.sh new <N>     Start auth flow for account N
#
# Tokens are stored in ~/.config/colab-cli/tokens/account_N.json
# The active token is symlinked at ~/.config/colab-cli/token.json

set -euo pipefail

COLAB_DIR="$HOME/.config/colab-cli"
TOKEN_DIR="$COLAB_DIR/tokens"
TOKEN_FILE="$COLAB_DIR/token.json"
ENV_FILE="$(dirname "$0")/../.env"

mkdir -p "$TOKEN_DIR"

cmd="${1:-list}"

case "$cmd" in
    list)
        echo "Saved Colab accounts:"
        echo "───────────────────────────────────────"
        if [ ! -d "$TOKEN_DIR" ] || [ -z "$(ls -A "$TOKEN_DIR" 2>/dev/null)" ]; then
            echo "  (none yet)"
        else
            for f in "$TOKEN_DIR"/account_*.json; do
                [ -f "$f" ] || continue
                acct=$(basename "$f" .json)
                # Get refresh token (truncated) and check if it's the active one
                rt=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('refresh_token','')[:20])" 2>/dev/null || echo "?")
                active=""
                if [ -L "$TOKEN_FILE" ] && [ "$(readlink -f "$TOKEN_FILE")" = "$(readlink -f "$f")" ]; then
                    active=" ← ACTIVE"
                fi
                echo "  $acct  token=${rt}...${active}"
            done
        fi
        echo ""
        # Also check if there's an unsaved active token
        if [ -f "$TOKEN_FILE" ] && [ ! -L "$TOKEN_FILE" ]; then
            echo "  ⚠ token.json exists but is NOT saved to any account"
            echo "    Run: ./scripts/colab_account.sh save <N>"
        fi
        ;;

    current)
        if [ -L "$TOKEN_FILE" ]; then
            target=$(readlink "$TOKEN_FILE")
            acct=$(basename "$target" .json)
            echo "Active account: $acct"
        elif [ -f "$TOKEN_FILE" ]; then
            echo "Active token: token.json (not saved to any account)"
            echo "Run: ./scripts/colab_account.sh save <N>"
        else
            echo "No active token"
        fi
        ;;

    save)
        acct_num="${2:?Usage: colab_account.sh save <N>}"
        target="$TOKEN_DIR/account_${acct_num}.json"
        if [ ! -f "$TOKEN_FILE" ]; then
            echo "Error: No active token at $TOKEN_FILE"
            echo "Authenticate first: colab new --gpu T4 -s acct${acct_num}"
            exit 1
        fi
        # Copy current token to account file
        cp "$TOKEN_FILE" "$target"
        # Symlink so it's now the active account
        rm -f "$TOKEN_FILE"
        ln -s "$target" "$TOKEN_FILE"
        echo "Saved current token as account_${acct_num}"
        echo "Active account: account_${acct_num}"
        ;;

    new)
        acct_num="${2:?Usage: colab_account.sh new <N>}"
        # Remove symlink so CLI creates a fresh token
        rm -f "$TOKEN_FILE"
        echo "Starting auth flow for account ${acct_num}..."
        echo "After authenticating, run: ./scripts/colab_account.sh save ${acct_num}"
        echo ""
        # Now run colab new which will prompt for auth
        exec colab new --gpu T4 -s "acct${acct_num}"
        ;;

    [0-9]*)
        acct_num="$cmd"
        target="$TOKEN_DIR/account_${acct_num}.json"
        if [ ! -f "$target" ]; then
            echo "Error: account_${acct_num} not found"
            echo "To create it: ./scripts/colab_account.sh new ${acct_num}"
            echo ""
            echo "Available accounts:"
            exec "$0" list
            exit 1
        fi
        rm -f "$TOKEN_FILE"
        ln -s "$target" "$TOKEN_FILE"
        echo "Switched to account_${acct_num}"
        # Verify it works
        echo "Active sessions:"
        colab sessions 2>/dev/null || echo "(CLI will refresh token on next use)"
        ;;

    *)
        echo "Usage:"
        echo "  $0 save <N>     Save current token as account N"
        echo "  $0 <N>          Switch to account N"
        echo "  $0 list         List all saved accounts"
        echo "  $0 current      Show active account"
        echo "  $0 new <N>      Start auth flow for account N"
        exit 1
        ;;
esac
