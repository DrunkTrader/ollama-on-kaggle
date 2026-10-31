# Ollama on Kaggle

Run **Qwen3.8-27B (IQ4_XS)** on Kaggle GPU T4x2 with Ollama, then access it remotely through an authenticated, OpenAI-compatible API.

## Setup

1. Open `ollama-vps.ipynb` on Kaggle, enable **GPU T4x2** and Internet, and keep the notebook private.
2. Create a Kaggle Secret named `OLLAMA_CLIENT_API_KEY` and enable notebook access. The notebook uses this secret for the proxy's bearer-token authentication.
3. Run the notebook cells in order. It starts Ollama, checks inference and GPU placement, starts the authenticated proxy, and creates a Cloudflare Quick Tunnel.
4. Download the generated `ollama-client.env`, `gpu_client.py`, and `requirements-client.txt`. Rename `ollama-client.env` to `.env` beside the client script.

## Run the client

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-client.txt
python gpu_client.py --stream "Explain how a lock-free queue works in C++20."
```

The client defaults to 8192 output tokens and disables model thinking. Add `--think` to enable thinking, or `--max-tokens N` to change the output limit.

## Restarting a Kaggle session

The Kaggle Secret persists, but the Ollama process, proxy, and Quick Tunnel do not. Start the notebook again and download a fresh `ollama-client.env`; the `OLLAMA_BASE_URL` will change, while `OLLAMA_API_KEY` remains the same as long as the Kaggle Secret is unchanged. To avoid downloading the model again, save the notebook output containing `ollama-models` and attach that output (or a private dataset containing it) to the next session.

## Security

- Ollama listens on `127.0.0.1:11434`; do **not** expose this port directly. Only the bearer-authenticated proxy is tunneled.
- Keep `.env` and downloaded credentials private. Never commit them or paste the API key into notebook output.
- The Quick Tunnel URL is temporary and publicly routable; bearer authentication protects API requests, but this is not a private network connection. Do not send sensitive prompts.
- Kaggle sessions are temporary and subject to platform limits. This is not an always-on VPS.
