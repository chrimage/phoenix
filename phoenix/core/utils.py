# phoenix/core/utils.py - Shared Utility Functions

def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.
    Uses a simple whitespace split as a basic approximation.
    """
    if not isinstance(text, str):
        return 0
    return len(text.split())
