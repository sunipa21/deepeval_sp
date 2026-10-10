"""
mcp_server.py
=============
MCP server exposing one tool used by agent_instrumented.py.

Run standalone to verify:
    python mcp_server.py
"""

from mcp.server.fastmcp import FastMCP

from mock_data import SHIPPING_OPTIONS

mcp = FastMCP("ShopEasy Shipping")


@mcp.tool()
def get_shipping_options() -> str:
    """Return available shipping methods with delivery times and costs."""
    return "; ".join(f"{name}: {days}, {cost}" for name, days, cost in SHIPPING_OPTIONS)


if __name__ == "__main__":
    mcp.run()
