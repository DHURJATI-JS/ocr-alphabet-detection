"""Start the server:  python -m app  [--port 8000] [--host 127.0.0.1] [--reload]

If the port is taken (e.g. by a Docker container), the next free port is used and the URL
to open is printed.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys

PORT_SEARCH = 20


def port_in_use(host: str, port: int) -> bool:
    # Something already accepting connections on localhost (IPv4 or IPv6)?
    for family, addr in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        with socket.socket(family, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            try:
                if s.connect_ex((addr, port)) == 0:
                    return True
            except OSError:
                pass
    # Can we bind it ourselves?
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m app", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--reload", action="store_true", help="restart on code changes (development)")
    args = ap.parse_args()

    port = next((p for p in range(args.port, args.port + PORT_SEARCH) if not port_in_use(args.host, p)), None)
    if port is None:
        print(f"Ports {args.port}-{args.port + PORT_SEARCH - 1} are all in use; pass --port.", file=sys.stderr)
        return 1
    if port != args.port:
        print(f"Port {args.port} is already in use by another program; using {port} instead.")
    browse_host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0", "::", "::1") else args.host
    print(f"\n  Alphabet OCR  ->  http://{browse_host}:{port}\n"
          "  (camera access needs localhost or HTTPS; Ctrl+C to stop)\n", flush=True)

    import uvicorn

    uvicorn.run("app.main:app", host=args.host, port=port, reload=args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
