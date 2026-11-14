"""REST routes under ``/api``."""

from fastapi import HTTPException


def not_implemented(step: str) -> HTTPException:
    """HTTP 501 for routes that exist but are implemented in a later step."""
    return HTTPException(status_code=501, detail=f"not implemented ({step})")


NOT_IMPLEMENTED = {501: {"description": "Not implemented yet"}}
