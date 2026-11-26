"""Install the same guards in ordinary Python subprocesses of the unit suite."""

import os

if os.environ.get("AF_OFFLINE_TESTS") == "1":
    from unit_isolation import install

    install()
