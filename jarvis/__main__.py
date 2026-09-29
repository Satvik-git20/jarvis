"""Allow `python -m jarvis`.

Without this, Python refuses because `jarvis` is a package rather than a
module, so the documented command in the README would not work.
"""

from .main import main

if __name__ == "__main__":
    raise SystemExit(main())
