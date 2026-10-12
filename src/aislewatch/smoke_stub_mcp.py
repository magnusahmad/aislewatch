"""S3 stub MCP server: one canned tool so Hermes (on the Codex subscription) can be tested end to end."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("aislewatch")


@mcp.tool()
def get_alerts(since: str = "today") -> list[dict]:
    """List containment and temperature alerts since a time (ISO timestamp or 'today')."""
    return [
        {
            "alert_id": "alert-0001",
            "type": "temperature_excursion",
            "camera_id": "cam-B",
            "aisle": "hall-2-aisle-B",
            "started_at": "2026-10-12T14:20:00-07:00",
            "detail": "Inlet temperature +4.1 C above baseline",
            "window_ids": ["cam-B-1402-00", "cam-B-1402-05"],
        }
    ]


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
