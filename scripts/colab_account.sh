#!/bin/bash
# Switch the Colab CLI to a different Google account.
# Usage: ./scripts/colab_account.sh <account_number>
#   ./scripts/colab_account.sh 1    # switch to account 1
#   ./scripts/colab_account.sh 2    # switch to account 2
#
# Tokens are stored in .env (gitignored).
# The script swaps the refresh token in ~/.config/colab-cli/token.json

set -euo pipefail

ACCT="${1:-1}"
ENV_FILE="$(dirname "$0")/../.env"
TOKEN_FILE="$HOME/.config/colab-cli/token.json"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env not found at $ENV_FILE"
    exit 1
fi

# Load the token for this account
TOKEN_VAR="COLAB_REFRESH_TOKEN_${ACCT}"
REFRESH_TOKEN=$(grep "^${TOKEN_VAR}=" "$ENV_FILE" | cut -d'=' -f2-)

if [ -z "$REFRESH_TOKEN" ]; then
    echo "Error: $TOKEN_VAR not set in .env"
    echo "To add account $ACCT:"
    echo "  1. Run: colab new --gpu T4 -s acct${ACCT}"
    echo "  2. Authenticate in browser"
    echo "  3. Copy refresh_token from $TOKEN_FILE"
    echo "  4. Add to .env: COLAB_REFRESH_TOKEN_${ACCT}=<token>"
    exit 1
fi

# Load client ID/secret
CLIENT_ID=$(grep "^COLAB_CLIENT_ID=" "$ENV_FILE" | cut -d'=' -f2-)
CLIENT_SECRET=$(grep "^COLAB_CLIENT_SECRET=" "$ENV_FILE" | cut -d'=' -f2-)

if [ -z "$CLIENT_ID" ] || [ -z "$CLIENT_SECRET" ]; then
    echo "Error: COLAB_CLIENT_ID or COLAB_CLIENT_SECRET not set in .env"
    exit 1
fi

# Write the token file for this account
mkdir -p "$(dirname "$TOKEN_FILE")"
cat > "$TOKEN_FILE" <<JSON
{
  "token": "",
  "refresh_token": "$REFRESH_TOKEN",
  "token_uri": "https://oauth2.googleapis.com/token",
  "client_id": "$CLIENT_ID",
  "client_secret": "$CLIENT_SECRET",
  "scopes": ["openid", "https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/cloud-platform", "https://www.googleapis.com/auth/colaboratory", "https://www.googleapis.com/auth/drive.file"],
  "universe_domain": "googleapis.com",
  "account": "",
  "expiry": "2020-01-01T00:00:00Z"
}
JSON

echo "Switched to Colab account $ACCT"
echo "The CLI will refresh the access token on next use."
