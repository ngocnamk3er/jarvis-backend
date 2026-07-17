from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool


@tool
def get_current_time(timezone: str, label: str) -> str:
    """Get the current date and time in a specific IANA timezone.

    Args:
        timezone: IANA timezone name, e.g. "Asia/Ho_Chi_Minh", "America/New_York", "Europe/London", "UTC".
        label: Brief human-readable description shown to the user.
    """
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        return f"Error: unknown timezone '{timezone}'. Use an IANA timezone name (e.g. 'Asia/Tokyo')."
    now = datetime.now(tz)
    return now.strftime("%A, %Y-%m-%d %H:%M:%S %Z%z")
