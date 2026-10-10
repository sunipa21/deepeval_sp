"""
mcp_server.py
=============
MCP server exposing one tool used by agent_instrumented.py.

Run standalone to verify:
    python mcp_server.py
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ShopEasy Shipping")


@mcp.tool()
def get_shipping_options() -> str:
    """Return available shipping methods with delivery times and costs."""
    return (
        "Standard (3-5 days, free on orders over $50), "
        "Express (1-2 days, $12.99), "
        "Overnight (next business day, $24.99)."
    )


if __name__ == "__main__":
    mcp.run()
