"""FastAPI app factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI

from . import __version__
from .auth_state import load_auth_state
from .config import ConfigStore
from .dependencies import require_authorized, require_operator
from .engine import DataEngine
from .routes import admin as admin_routes
from .routes import config as config_routes
from .routes import data as data_routes
from .routes import health as health_routes
from .settings import Settings, load_settings

log = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    config_store = ConfigStore(settings.config_file)
    if settings.safe_mode:
        log.warning(
            "starting in SAFE MODE (EUGENE_PLEXUS_DAT_SAFE_MODE=1); ignoring "
            "%s and running on defaults. Fix config via /v1/config, then "
            "restart without the env var.",
            settings.config_file,
        )
    else:
        config_store.load()
    app.state.config_store = config_store
    app.state.safe_mode = settings.safe_mode

    # v0.2 auth state. Tests can pre-populate `app.state.auth_state` to
    # exercise authed paths; the default lifespan build reads env vars
    # via Settings and produces an auth-disabled state when the watchdog
    # didn't supply AUTH_SIGNING_KEY.
    if not hasattr(app.state, "auth_state"):
        app.state.auth_state = load_auth_state(
            signing_key_b64=settings.auth_signing_key,
            service_token=settings.service_token,
            master_key_b64=settings.master_key,
        )

    # Build the data-preparation engine. In safe mode we deliberately skip
    # it: app.state.engine stays None, mutating routes return 503, and
    # /healthz reports degraded — config endpoints remain reachable so an
    # operator can repair a config that breaks the engine. A build failure
    # surfaces as degraded mode rather than crashing the process, per
    # feedback_degraded_mode_required.md. Tests may pre-inject an engine.
    if not hasattr(app.state, "engine"):
        if settings.safe_mode:
            app.state.engine = None
            app.state.engine_error = (
                "safe mode active (EUGENE_PLEXUS_DAT_SAFE_MODE=1); data "
                "operations are refused until it is cleared"
            )
        else:
            try:
                app.state.engine = DataEngine(
                    data_root=Path(config_store.get("dataRoot")),
                    inbox_dir=Path(config_store.get("inboxDir")),
                    archive_dir=Path(config_store.get("archiveDir")),
                    default_block_size=int(config_store.get("defaultBlockSize")),
                )
                app.state.engine_error = None
            except Exception as e:  # degrade instead of crashing the process
                log.exception("failed to build the data engine; entering degraded mode")
                app.state.engine = None
                app.state.engine_error = f"data engine failed to initialize: {e}"

    yield

    engine = getattr(app.state, "engine", None)
    if engine is not None:
        engine.shutdown()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a FastAPI app with all routers mounted."""
    settings = settings or load_settings()

    app = FastAPI(
        title="Eugene Plexus — data",
        description=(
            "Dataset preparation and tokenizer training engine. v0.3 "
            "skeleton ships the control-plane wire shape; the preparation "
            "engine is future work."
        ),
        version=__version__,
        lifespan=_lifespan,
    )
    app.state.settings = settings

    # Health stays unauthenticated — supervisors and load balancers need
    # to probe it without holding credentials.
    app.include_router(health_routes.router)

    # Config edits are operator-only — service tokens are rejected so a
    # compromised peer can't reconfigure the data component (e.g. repoint
    # the data root).
    operator = [Depends(require_operator)]
    app.include_router(config_routes.router, dependencies=operator)
    app.include_router(admin_routes.router, dependencies=operator)

    # Dataset / tokenizer operations: peer services (trainer, coordinator)
    # drive reads via service tokens; operators may also drive imports and
    # tokenizer training through the UI.
    app.include_router(data_routes.router, dependencies=[Depends(require_authorized)])

    return app
