"""pyrpod.logging_utils — centralized, opt-in operational logging for PyRPOD.

PyRPOD is primarily a reusable library, so importing ``pyrpod`` (or this
module) has **no logging side effects**: it does not create files or
directories, does not add handlers, does not touch the root logger, and does
not emit output. Production modules obtain a logger with the standard idiom::

    import logging
    logger = logging.getLogger(__name__)

and never configure handlers themselves.

An application turns logging on explicitly at its boundary::

    from pyrpod.logging_utils import configure_logging

    session = configure_logging(case_dir)
    try:
        ...  # run the PyRPOD workflow
        session.finalize("successful")
    finally:
        session.close()

``configure_logging`` attaches PyRPOD-owned handlers to the ``pyrpod`` package
logger (never the root logger), so records from ``pyrpod.*`` module loggers
propagate up to those handlers. Handlers belonging to an embedding application
are never inspected or modified; calling ``configure_logging`` again closes and
replaces only the handlers PyRPOD itself installed, so messages are never
duplicated.

Configuration precedence (highest first):

1. Explicit Python API arguments to ``configure_logging``
2. Environment variables (``PYRPOD_LOG_LEVEL``, ``PYRPOD_LOG_FORMAT``)
3. ``<case_dir>/logging.ini`` ``[logging]`` section
4. Built-in defaults (see :class:`LoggingSettings`)
"""
from __future__ import annotations

import configparser
import hashlib
import logging
import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, TypeVar

import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
PACKAGE_LOGGER_NAME = "pyrpod"
DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_OWNED_HANDLER_ATTR = "_pyrpod_owned"

# The repo-level shared data directory, mirroring pyrpod.util.io.fs so asset
# provenance ("case-local" vs "shared") can be reported consistently.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SHARED_DATA_DIR = os.path.join(_REPO_ROOT, "data")

# The single active PyRPOD logging session, if any. Kept module-global so
# workflow code can read run-scoped settings (progress interval, array-stats
# toggle, ...) without threading a session object through every call site.
_ACTIVE_SESSION: Optional["LoggingSession"] = None


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
@dataclass
class LoggingSettings:
    """Resolved logging configuration for a run.

    These are the built-in defaults; ``logging.ini``, environment variables,
    and explicit API arguments override them per the documented precedence.
    """

    enabled: bool = True
    console: bool = True
    file: bool = True
    level: str = "INFO"
    log_format: str = DEFAULT_LOG_FORMAT
    progress_every_n_firings: int = 1
    log_array_stats: bool = True
    log_performance: bool = True
    snapshot_config: bool = True
    checksum_inputs: bool = True


_INI_BOOL_KEYS = (
    "enabled",
    "console",
    "file",
    "log_array_stats",
    "log_performance",
    "snapshot_config",
    "checksum_inputs",
)


def _read_logging_ini(
    case_dir: str,
) -> tuple[configparser.SectionProxy | None, str]:
    """Return (``[logging]`` section proxy or None, ini path)."""
    ini_path = os.path.join(case_dir, "logging.ini")
    if not os.path.isfile(ini_path):
        return None, ini_path
    parser = configparser.ConfigParser()
    try:
        parser.read(ini_path)
    except configparser.Error:
        return None, ini_path
    if not parser.has_section("logging"):
        return None, ini_path
    return parser["logging"], ini_path


def _apply_ini(settings: LoggingSettings,
               section: configparser.SectionProxy) -> None:
    for key in _INI_BOOL_KEYS:
        if key in section:
            try:
                setattr(settings, key, section.getboolean(key))
            except ValueError:
                pass
    # SectionProxy.get()/getint() are typed as returning ``| None`` because
    # configparser supports allow_no_value=True. _read_logging_ini builds a
    # plain ConfigParser() (allow_no_value=False), so an option that passes the
    # ``in section`` guard above always carries a value; the stub cannot say so.
    if "level" in section:
        settings.level = section.get("level")  # type: ignore[assignment]
    if "format" in section:
        settings.log_format = section.get("format")  # type: ignore[assignment]
    if "progress_every_n_firings" in section:
        try:
            settings.progress_every_n_firings = (
                section.getint("progress_every_n_firings")  # type: ignore[assignment]
            )
        except ValueError:
            pass


def _resolve_settings(
    case_dir: str, overrides: Dict[str, Any]
) -> tuple[LoggingSettings, bool, str]:
    """Resolve :class:`LoggingSettings` following the documented precedence."""
    settings = LoggingSettings()

    # 3. logging.ini
    ini_section, ini_path = _read_logging_ini(case_dir)
    used_ini = ini_section is not None
    if ini_section is not None:
        _apply_ini(settings, ini_section)

    # 2. environment variables (only level and format are supported)
    env_level = os.environ.get("PYRPOD_LOG_LEVEL")
    if env_level:
        settings.level = env_level
    env_format = os.environ.get("PYRPOD_LOG_FORMAT")
    if env_format:
        settings.log_format = env_format

    # 1. explicit API arguments
    for key, value in overrides.items():
        if value is not None:
            setattr(settings, key, value)

    settings.level = str(settings.level).upper()
    if settings.progress_every_n_firings < 1:
        settings.progress_every_n_firings = 1
    return settings, used_ini, ini_path


# --------------------------------------------------------------------------- #
# Handler ownership helpers
# --------------------------------------------------------------------------- #
class _LevelCounter(logging.Handler):
    """Owned handler that tallies WARNING/ERROR records for the run summary."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.warning_count = 0
        self.error_count = 0

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        if record.levelno >= logging.ERROR:
            self.error_count += 1
        elif record.levelno >= logging.WARNING:
            self.warning_count += 1


# Marking a handler returns the *same* handler, so the concrete subclass must
# survive the call: configure_logging feeds the result straight into
# LoggingSession(_counter=...), which is typed _LevelCounter.
_HandlerT = TypeVar("_HandlerT", bound=logging.Handler)


def _mark_owned(handler: _HandlerT) -> _HandlerT:
    setattr(handler, _OWNED_HANDLER_ATTR, True)
    return handler


def _remove_owned_handlers(logger: logging.Logger) -> None:
    """Detach and close only handlers PyRPOD installed; leave app handlers."""
    for handler in list(logger.handlers):
        if getattr(handler, _OWNED_HANDLER_ATTR, False):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass


def _level_to_int(level: int | str) -> int:
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)


def _safe_case_name(case_dir: str) -> str:
    name = os.path.basename(os.path.normpath(case_dir)) or "case"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _unique_log_path(log_dir: str, case_name: str, timestamp: str) -> str:
    base = os.path.join(log_dir, f"{case_name}_{timestamp}.log")
    if not os.path.exists(base):
        return base
    # Two runs in the same wall-clock second: keep the convention but stay
    # unique by appending microseconds, then the PID as a last resort.
    for _ in range(1000):
        micro = datetime.now().strftime("%f")
        candidate = os.path.join(log_dir, f"{case_name}_{timestamp}_{micro}.log")
        if not os.path.exists(candidate):
            return candidate
    return os.path.join(log_dir, f"{case_name}_{timestamp}_{os.getpid()}.log")


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #
@dataclass
class LoggingSession:
    """Handle for one configured PyRPOD logging run.

    Returned by :func:`configure_logging`. Carries the resolved settings, the
    active runtime-log path, and a run-scoped input-checksum cache. Use
    :meth:`finalize` to record the final status and :meth:`close` to release
    the PyRPOD-owned handlers.
    """

    case_dir: str
    case_name: str
    settings: LoggingSettings
    log_path: Optional[str]
    used_ini: bool
    ini_path: str
    timestamp: str
    logger: logging.Logger
    _counter: _LevelCounter
    _hash_cache: Dict[str, str] = field(default_factory=dict)
    _closed: bool = False
    _start_wall: float = field(default_factory=time.perf_counter)
    _start_dt: datetime = field(default_factory=datetime.now)

    @property
    def warning_count(self) -> int:
        return self._counter.warning_count

    @property
    def error_count(self) -> int:
        return self._counter.error_count

    def finalize(self, status: str = "successful", *,
                 summary: Optional[Dict[str, Any]] = None) -> None:
        """Log the final run status and an optional summary once.

        ``status`` is one of ``successful``, ``successful_with_warning``,
        ``partial``, or ``failed``. This does not close handlers; call
        :meth:`close` (typically from a ``finally`` block) for that.
        """
        elapsed = time.perf_counter() - self._start_wall
        fields = {
            "status": status,
            "warnings": self.warning_count,
            "errors": self.error_count,
            "runtime_s": round(elapsed, 3),
            "runtime_log": self.log_path,
        }
        if summary:
            fields.update(summary)
        self.logger.info("run finalized: %s", _format_fields(fields))

    def close(self) -> None:
        """Detach and close PyRPOD-owned handlers installed for this session."""
        global _ACTIVE_SESSION
        if self._closed:
            return
        _remove_owned_handlers(self.logger)
        self._closed = True
        if _ACTIVE_SESSION is self:
            _ACTIVE_SESSION = None


# --------------------------------------------------------------------------- #
# Public configuration API
# --------------------------------------------------------------------------- #
def configure_logging(case_dir: str, *,
                      console: Optional[bool] = None,
                      file: Optional[bool] = None,
                      level: Optional[str] = None,
                      log_format: Optional[str] = None,
                      enabled: Optional[bool] = None,
                      progress_every_n_firings: Optional[int] = None,
                      log_array_stats: Optional[bool] = None,
                      log_performance: Optional[bool] = None,
                      snapshot_config: Optional[bool] = None,
                      checksum_inputs: Optional[bool] = None,
                      write_startup: bool = True) -> LoggingSession:
    """Configure PyRPOD logging for a case and return a :class:`LoggingSession`.

    Resolves the absolute case directory, reads an optional
    ``<case_dir>/logging.ini``, applies environment-variable and explicit-API
    overrides, installs PyRPOD-owned console/file handlers on the ``pyrpod``
    package logger, opens a fresh timestamped runtime log under
    ``<case_dir>/results/logs/``, writes the startup-metadata block, and (when
    enabled) snapshots the case configuration files.

    Calling this repeatedly is safe: previously installed PyRPOD handlers are
    closed and replaced, never duplicated, and handlers owned by an embedding
    application are left untouched.
    """
    global _ACTIVE_SESSION

    abs_case_dir = os.path.abspath(case_dir)
    overrides = {
        "console": console,
        "file": file,
        "level": level,
        "log_format": log_format,
        "enabled": enabled,
        "progress_every_n_firings": progress_every_n_firings,
        "log_array_stats": log_array_stats,
        "log_performance": log_performance,
        "snapshot_config": snapshot_config,
        "checksum_inputs": checksum_inputs,
    }
    settings, used_ini, ini_path = _resolve_settings(abs_case_dir, overrides)

    pkg_logger = logging.getLogger(PACKAGE_LOGGER_NAME)

    # Replace only PyRPOD-owned handlers so repeated configuration never
    # duplicates output and never disturbs the embedding application.
    _remove_owned_handlers(pkg_logger)

    case_name = _safe_case_name(abs_case_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    level_value = _level_to_int(settings.level)

    counter = _mark_owned(_LevelCounter())
    pkg_logger.addHandler(counter)

    log_path: Optional[str] = None
    if settings.enabled:
        pkg_logger.setLevel(level_value)
        formatter = logging.Formatter(settings.log_format,
                                      datefmt=DEFAULT_DATE_FORMAT)
        if settings.console:
            console_handler = _mark_owned(logging.StreamHandler())
            console_handler.setLevel(level_value)
            console_handler.setFormatter(formatter)
            pkg_logger.addHandler(console_handler)
        if settings.file:
            log_dir = os.path.join(abs_case_dir, "results", "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = _unique_log_path(log_dir, case_name, timestamp)
            file_handler = _mark_owned(
                logging.FileHandler(log_path, encoding="utf-8"))
            file_handler.setLevel(level_value)
            file_handler.setFormatter(formatter)
            pkg_logger.addHandler(file_handler)

    session = LoggingSession(
        case_dir=abs_case_dir,
        case_name=case_name,
        settings=settings,
        log_path=log_path,
        used_ini=used_ini,
        ini_path=ini_path,
        timestamp=timestamp,
        logger=pkg_logger,
        _counter=counter,
    )
    _ACTIVE_SESSION = session

    if settings.enabled and write_startup:
        _write_startup_metadata(session)
        if settings.snapshot_config:
            _write_config_snapshots(session)

    return session


def get_active_session() -> Optional[LoggingSession]:
    """Return the currently active :class:`LoggingSession`, or None."""
    return _ACTIVE_SESSION


def get_settings() -> LoggingSettings:
    """Return the active run's settings, or built-in defaults if unconfigured."""
    if _ACTIVE_SESSION is not None:
        return _ACTIVE_SESSION.settings
    return LoggingSettings()


def performance_logging_enabled() -> bool:
    """True only when a session is active and performance logging is on.

    tracemalloc has real overhead, so it is engaged only for an explicitly
    configured run — never merely because a workflow function was called.
    """
    return _ACTIVE_SESSION is not None and _ACTIVE_SESSION.settings.log_performance


# --------------------------------------------------------------------------- #
# Startup metadata and configuration snapshots
# --------------------------------------------------------------------------- #
def git_dirty_status(path: str) -> str:
    """Return 'clean', 'dirty', or 'unknown' for the repo containing ``path``.

    Uses a short, safe standard-library subprocess call. Any failure (git
    missing, not a repository, timeout) yields 'unknown' rather than raising —
    logging must never take down a simulation.
    """
    work_dir = path if os.path.isdir(path) else os.path.dirname(path) or "."
    try:
        result = subprocess.run(
            ["git", "-C", work_dir, "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return "dirty" if result.stdout.strip() else "clean"


def _write_startup_metadata(session: LoggingSession) -> None:
    logger = session.logger
    settings = session.settings
    logger.info("=== PyRPOD run start ===")
    fields = {
        "case_path": session.case_dir,
        "start_time": session._start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "git_dirty": git_dirty_status(session.case_dir),
        "log_level": settings.level,
        "console_logging": "on" if settings.console else "off",
        "file_logging": "on" if settings.file else "off",
        "runtime_log": session.log_path,
        "logging_config": "logging.ini" if session.used_ini else "built-in defaults",
    }
    logger.info("startup: %s", _format_fields(fields))
    if not session.used_ini:
        logger.info("No logging.ini found in case; using built-in default "
                    "logging configuration.")


def _write_config_snapshots(session: LoggingSession) -> None:
    """Copy case config files into the log dir and log their SHA-256 sums."""
    logger = session.logger
    if session.log_path:
        log_dir = os.path.dirname(session.log_path)
    else:
        log_dir = os.path.join(session.case_dir, "results", "logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not create log dir for config snapshots: %s",
                       exc, exc_info=True)
        return

    prefix = f"{session.case_name}_{session.timestamp}"

    config_src = os.path.join(session.case_dir, "config.ini")
    if os.path.isfile(config_src):
        try:
            dst = os.path.join(log_dir, f"{prefix}_config.ini")
            shutil.copyfile(config_src, dst)
            logger.info("config snapshot: src=%s snapshot=%s sha256=%s",
                        config_src, dst, sha256_file(dst))
        except OSError as exc:
            logger.warning("Failed to snapshot config.ini: %s", exc,
                           exc_info=True)
    else:
        logger.warning("No config.ini found at %s; snapshot skipped.",
                       config_src)

    if session.used_ini and os.path.isfile(session.ini_path):
        try:
            dst = os.path.join(log_dir, f"{prefix}_logging.ini")
            shutil.copyfile(session.ini_path, dst)
            logger.info("logging.ini snapshot: src=%s snapshot=%s sha256=%s",
                        session.ini_path, dst, sha256_file(dst))
        except OSError as exc:
            logger.warning("Failed to snapshot logging.ini: %s", exc,
                           exc_info=True)
    else:
        logger.info("No logging.ini present; built-in defaults used "
                    "(no logging.ini snapshot).")


# --------------------------------------------------------------------------- #
# Input-asset tracking
# --------------------------------------------------------------------------- #
def sha256_file(path: str, chunk_size: int = 65536) -> str:
    """Streaming SHA-256 so large assets are not read fully into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_source(abs_path: str, case_dir: str) -> str:
    abs_case = os.path.abspath(case_dir)
    try:
        if os.path.commonpath([abs_path, abs_case]) == abs_case:
            return "case-local"
    except ValueError:
        pass
    try:
        if os.path.commonpath([abs_path, _SHARED_DATA_DIR]) == _SHARED_DATA_DIR:
            return "shared-data"
    except ValueError:
        pass
    return "external"


def _cached_checksum(session: Optional[LoggingSession], abs_path: str) -> Optional[str]:
    if not os.path.isfile(abs_path):
        return None
    if session is not None:
        cache = session._hash_cache
        if abs_path in cache:
            return cache[abs_path]
        digest = sha256_file(abs_path)
        cache[abs_path] = digest
        return digest
    return sha256_file(abs_path)


def log_asset(category: str, filename: str, resolved_path: str, case_dir: str,
              *, logger: Optional[logging.Logger] = None) -> None:
    """Record a loaded input asset at INFO with provenance and checksum.

    Logs the asset category, filename, absolute resolved path, whether it came
    from the case-local directory or the shared repository data, the file size
    in bytes, and (unless disabled via ``checksum_inputs``) its SHA-256. The
    checksum is cached per run so an unchanged asset is not rehashed.
    """
    logger = logger or logging.getLogger(PACKAGE_LOGGER_NAME)
    abs_path = os.path.abspath(resolved_path)
    source = _asset_source(abs_path, case_dir)
    try:
        size = os.path.getsize(abs_path)
    except OSError:
        size = -1

    digest = None
    if get_settings().checksum_inputs:
        digest = _cached_checksum(get_active_session(), abs_path)

    logger.info(
        "asset loaded: %s",
        _format_fields({
            "category": category,
            "file": filename,
            "path": abs_path,
            "source": source,
            "size_bytes": size,
            "sha256": digest if digest is not None else "n/a",
        }),
    )


# --------------------------------------------------------------------------- #
# Array summaries
# --------------------------------------------------------------------------- #
def summarize_array(name: str, array: Any) -> str:
    """Return a compact one-line summary of a NumPy-like array.

    Reports name, shape, dtype, min/max/mean, nonzero/zero/NaN counts, and
    memory in bytes. Handles empty, non-numeric, integer, boolean, NaN-bearing,
    and non-array inputs without raising — a summary must never break a run.
    Full array contents are never logged.
    """
    try:
        arr = np.asarray(array)
    except Exception:
        return _format_fields({"name": name,
                               "note": f"unsummarizable {type(array).__name__}"})

    parts: Dict[str, Any] = {
        "name": name,
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "nbytes": int(arr.nbytes),
    }
    if arr.size == 0:
        parts["empty"] = True
        return _format_fields(parts)

    is_numeric = np.issubdtype(arr.dtype, np.number)
    is_bool = np.issubdtype(arr.dtype, np.bool_)
    if not (is_numeric or is_bool):
        parts["numeric"] = False
        return _format_fields(parts)

    try:
        numeric = arr if is_numeric else arr.astype(np.int64)
        if np.issubdtype(numeric.dtype, np.floating):
            nan_count = int(np.isnan(numeric).sum())
            finite = numeric[np.isfinite(numeric)]
        else:
            nan_count = 0
            finite = numeric.ravel()
        if finite.size:
            parts["min"] = f"{float(finite.min()):.6g}"
            parts["max"] = f"{float(finite.max()):.6g}"
            parts["mean"] = f"{float(finite.mean()):.6g}"
        else:
            parts["min"] = parts["max"] = parts["mean"] = "nan"
        nonzero = int(np.count_nonzero(arr))
        parts["nonzero"] = nonzero
        parts["zero"] = int(arr.size - nonzero)
        parts["nan"] = nan_count
    except Exception as exc:  # pragma: no cover - defensive
        parts["note"] = f"summary error: {exc}"
    return _format_fields(parts)


def log_array_summary(logger: logging.Logger, name: str, array: Any,
                      level: int = logging.DEBUG) -> None:
    """Log :func:`summarize_array` output, honoring the array-stats toggle.

    Skips the (cheap but non-zero) summary work entirely when the level is not
    enabled or ``log_array_stats`` is off.
    """
    if not get_settings().log_array_stats:
        return
    if not logger.isEnabledFor(level):
        return
    logger.log(level, "array: %s", summarize_array(name, array))


# --------------------------------------------------------------------------- #
# Small formatting helper
# --------------------------------------------------------------------------- #
def _format_fields(fields: Dict[str, Any]) -> str:
    """Render an ordered mapping as ``key=value`` pairs for concise records."""
    return " ".join(f"{key}={value}" for key, value in fields.items())
