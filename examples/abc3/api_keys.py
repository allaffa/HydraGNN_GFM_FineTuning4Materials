"""Materials Project API key loader for the abc3 example.

The key is intentionally NOT hard-coded here.  It is read from the
``MP_API_KEY`` environment variable so the secret never lives in the repo.

Usage:
    export MP_API_KEY="your-materials-project-key"
    python examples/abc3/abc3_getData_API.py
"""

import os

MP_API_KEY = os.environ.get("MP_API_KEY")

if not MP_API_KEY:
    raise RuntimeError(
        "MP_API_KEY is not set. Export your Materials Project API key first:\n"
        "    export MP_API_KEY='your-key'\n"
        "Get a key at https://next-gen.materialsproject.org/api"
    )
