"""Optional MongoDB storage for removal history.

Only metadata is stored. Uploaded images and cutouts stay on this machine, in
keeping with the app's local-first design.

Everything here is fail-soft: with no connection string configured, or with the
cluster unreachable, the functions do nothing and background removal is never
affected. Writes run on a background thread so a slow cluster cannot delay a
response.
"""

import logging
import os
import threading
from datetime import datetime, timezone
from urllib.parse import quote_plus

try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
except ImportError:  # pragma: no cover - pymongo is optional at runtime
    MongoClient = None
    PyMongoError = Exception

log = logging.getLogger("cutout-studio.db")

PASSWORD_PLACEHOLDER = "<db_password>"
DEFAULT_DATABASE = "cutout_studio"
REMOVALS = "removals"

_client = None
_client_uri = None
_client_lock = threading.Lock()
_last_ok = None  # outcome of the most recent operation; None until one runs
_warned = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        log.warning(message)


def connection_uri() -> str | None:
    """Return the usable connection string, or None when MongoDB is not set up.

    Atlas hands out a string containing ``<db_password>``. It can be edited in
    place, or left as it is with the password in MONGODB_PASSWORD, which is
    percent-encoded here so passwords containing ``@`` or ``/`` still work.
    """
    uri = os.environ.get("MONGODB_URI", "").strip()
    if not uri:
        return None
    if PASSWORD_PLACEHOLDER in uri:
        password = os.environ.get("MONGODB_PASSWORD", "")
        if not password:
            _warn_once(
                "password",
                "MONGODB_URI still contains <db_password>; set MONGODB_PASSWORD "
                "in .env to enable MongoDB. Continuing without it.",
            )
            return None
        uri = uri.replace(PASSWORD_PLACEHOLDER, quote_plus(password))
    if MongoClient is None:
        _warn_once("driver", "pymongo is not installed; MongoDB is disabled.")
        return None
    return uri


def configured() -> bool:
    return connection_uri() is not None


def database_name() -> str:
    return os.environ.get("MONGODB_DB", "").strip() or DEFAULT_DATABASE


def _database():
    """Create the client on first use; connecting itself is deferred by pymongo."""
    global _client, _client_uri
    uri = connection_uri()
    if uri is None:
        return None
    with _client_lock:
        if _client is None or _client_uri != uri:
            _client = MongoClient(
                uri,
                serverSelectionTimeoutMS=3000,
                appname="cutout-studio",
                tz_aware=True,
            )
            _client_uri = uri
        return _client[database_name()]


def _run(operation) -> None:
    """Run a write on a daemon thread so callers never wait on the network."""
    threading.Thread(target=operation, daemon=True).start()


def _insert(collection: str, document: dict) -> None:
    global _last_ok
    database = _database()
    if database is None:
        return
    try:
        database[collection].insert_one(document)
        _last_ok = True
    except PyMongoError as exc:
        _last_ok = False
        log.warning("Could not write to MongoDB collection %s: %s", collection, exc)


def _record(collection: str, fields: dict) -> None:
    if not configured():
        return
    document = {"created_at": datetime.now(timezone.utc), **fields}
    _run(lambda: _insert(collection, document))


def record_removal(**fields) -> None:
    """Log one completed background removal."""
    _record(REMOVALS, fields)


def ping() -> bool:
    """Contact the cluster now. Used at startup and by the history endpoint."""
    global _last_ok
    database = _database()
    if database is None:
        return False
    try:
        database.client.admin.command("ping")
    except PyMongoError as exc:
        _last_ok = False
        log.warning("MongoDB is not reachable: %s", exc)
        return False
    _last_ok = True
    return True


def recent_removals(limit: int = 20) -> list[dict] | None:
    """Return the newest removal records, or None if the cluster cannot be read."""
    global _last_ok
    database = _database()
    if database is None:
        return None
    try:
        rows = list(database[REMOVALS].find().sort("_id", -1).limit(limit))
    except PyMongoError as exc:
        _last_ok = False
        log.warning("Could not read MongoDB history: %s", exc)
        return None
    _last_ok = True
    for row in rows:
        row["id"] = str(row.pop("_id"))
        row["created_at"] = row["created_at"].isoformat()
    return rows


def status() -> dict:
    """Describe the connection without touching the network."""
    return {
        "configured": configured(),
        "connected": _last_ok,
        "database": database_name() if configured() else None,
    }
