"""
Test client for the MCP server defined in main.py.

Spawns the server as a subprocess via stdio and exercises all
registered tools, resources, and prompts.

Run:
    uv run python test_client.py
"""

import asyncio
import logging
import sys

from client.base_client import BaseMCPClient

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)

SERVER_SCRIPT = "./main.py"


def print_section(title: str):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


async def run():
    print(f"Connecting to server: {SERVER_SCRIPT}")

    async with BaseMCPClient(server_script_path=SERVER_SCRIPT) as client:
        await client.discover()

        # ── Tools ──────────────────────────────────────────────
        print_section(f"Tools ({len(client.tools)})")
        for tool in client.list_tools():
            print(f"  • {tool.name}: {tool.description}")

        if client.tools:
            print("\nCalling each tool with sample arguments...")
            for tool in client.list_tools():
                # Build minimal sample args from the schema's required fields
                sample_args = _build_sample_args(tool.input_schema)
                try:
                    result = await client.call_tool(tool.name, sample_args)
                    print(f"  ✓ {tool.name}({sample_args!r}) → {result!r}")
                except Exception as exc:
                    print(f"  ✗ {tool.name}: {exc}")

        # ── Resources ──────────────────────────────────────────
        print_section(f"Resources ({len(client.resources)})")
        for res in client.list_resources():
            print(f"  • [{res.mime_type}] {res.uri} — {res.name}")

        if client.resources:
            print("\nReading each resource...")
            for res in client.list_resources():
                try:
                    content = await client.read_resource(res.uri)
                    preview = (content or "")[:120].replace("\n", " ")
                    print(f"  ✓ {res.uri} → {preview!r}")
                except Exception as exc:
                    print(f"  ✗ {res.uri}: {exc}")

        # ── Prompts ────────────────────────────────────────────
        print_section(f"Prompts ({len(client.prompts)})")
        for prompt in client.list_prompts():
            args_info = ", ".join(
                a["name"] if isinstance(a, dict) else str(a)
                for a in (prompt.arguments or [])
            )
            print(f"  • {prompt.name}({args_info}): {prompt.description}")

        if client.prompts:
            print("\nFetching each prompt with sample arguments...")
            for prompt in client.list_prompts():
                sample_args = {
                    a["name"]: f"sample_{a['name']}"
                    for a in (prompt.arguments or [])
                    if isinstance(a, dict)
                }
                try:
                    messages = await client.get_prompt(prompt.name, sample_args)
                    print(f"  ✓ {prompt.name} → {len(messages)} message(s)")
                    for msg in messages:
                        preview = str(msg.get("content", ""))[:80].replace("\n", " ")
                        print(f"      [{msg.get('role')}] {preview!r}")
                except Exception as exc:
                    print(f"  ✗ {prompt.name}: {exc}")

        print("\nDone.\n")


def _build_sample_args(schema: dict) -> dict:
    """Build a minimal set of arguments from a JSON Schema definition."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    args = {}
    for name, prop in properties.items():
        if name not in required:
            continue
        prop_type = prop.get("type", "string")
        if prop_type == "string":
            args[name] = f"sample_{name}"
        elif prop_type == "integer":
            args[name] = 1
        elif prop_type == "number":
            args[name] = 1.0
        elif prop_type == "boolean":
            args[name] = True
        elif prop_type == "array":
            args[name] = []
        elif prop_type == "object":
            args[name] = {}
        else:
            args[name] = None
    return args


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        sys.exit(0)
