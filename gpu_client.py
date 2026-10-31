import argparse
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI


def get_field(obj, *names):
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    elif not isinstance(obj, dict):
        obj = vars(obj) if hasattr(obj, "__dict__") else {}

    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else None
        if value is not None:
            return value
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Send a prompt to the Kaggle-hosted Ollama model."
    )
    parser.add_argument("prompt", nargs="+", help="Prompt text")
    parser.add_argument("--system", default=None, help="Optional system message")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--stream", action="store_true", help="Stream output")
    parser.add_argument(
        "--think",
        action="store_true",
        help="Enable model thinking; disabled by default to reserve tokens for the answer",
    )
    args = parser.parse_args()

    load_dotenv()
    base_url = os.getenv("OLLAMA_BASE_URL")
    api_key = os.getenv("OLLAMA_API_KEY")
    model = os.getenv("OLLAMA_MODEL")

    missing = [
        name for name, value in {
            "OLLAMA_BASE_URL": base_url,
            "OLLAMA_API_KEY": api_key,
            "OLLAMA_MODEL": model,
        }.items() if not value
    ]
    if missing:
        raise SystemExit("Missing .env values: " + ", ".join(missing))

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": " ".join(args.prompt)})

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=1800.0,
        max_retries=1,
    )

    print(f"Requesting model: {model}", file=sys.stderr)
    print(
        f"Streaming: {args.stream} | Thinking: {args.think} | "
        f"Max output tokens: {args.max_tokens}",
        file=sys.stderr,
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stream=args.stream,
        extra_body={
            "reasoning_effort": "none" if not args.think else "high"
        },
    )

    if args.stream:
        content_seen = False
        reasoning_seen = False
        finish_reason = None

        for chunk in response:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            finish_reason = choice.finish_reason or finish_reason
            delta = choice.delta
            content = get_field(delta, "content")
            reasoning = get_field(delta, "reasoning", "reasoning_content")

            if content:
                print(content, end="", flush=True)
                content_seen = True
            if reasoning:
                reasoning_seen = True

        print()
    else:
        if not response.choices:
            raise RuntimeError("The API returned no completion choices.")

        choice = response.choices[0]
        message = choice.message
        content = get_field(message, "content")
        reasoning = get_field(message, "reasoning", "reasoning_content")
        finish_reason = choice.finish_reason
        content_seen = bool(content)
        reasoning_seen = bool(reasoning)

        if content:
            print(content)

    if not content_seen:
        print("No final answer content was returned.", file=sys.stderr)
        print(f"Finish reason: {finish_reason or 'not provided'}", file=sys.stderr)

        if reasoning_seen:
            print(
                "The model emitted reasoning but no final answer. "
                "Try a larger --max-tokens value or keep thinking disabled.",
                file=sys.stderr,
            )
        else:
            print(
                "The response contained neither visible content nor recognized "
                "reasoning fields. Check the proxy/model response format.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nRequest failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
