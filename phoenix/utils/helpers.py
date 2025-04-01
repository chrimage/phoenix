# phoenix/utils/helpers.py - General utility functions
# (Currently empty, add helpers as needed)

def format_timestamp(ts: float, fmt: str = '%Y-%m-%d %H:%M:%S') -> str:
    """Formats a Unix timestamp into a human-readable string."""
    import datetime
    try:
        return datetime.datetime.fromtimestamp(ts).strftime(fmt)
    except Exception:
        return str(ts) # Fallback to string representation

# Add other general-purpose functions here
