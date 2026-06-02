"""Import-safe Streamlit UI package for DEX Scout page renderers.

The modules in this package expose render functions only. They intentionally do
not import app.py or perform storage/scanner work at import time; app.py remains
the single Streamlit entrypoint and passes explicit render actions through the
UI context.
"""

__all__ = [
    "layout",
    "pages_archive",
    "pages_monitoring",
    "pages_portfolio",
    "pages_runtime",
    "pages_scout",
]
