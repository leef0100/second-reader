"""`secondreader` command line. For research use only. Not for diagnostic use."""

from __future__ import annotations

import typer

from secondreader.core.constants import RESEARCH_USE_LABEL

app = typer.Typer(help=f"Second Reader CLI — {RESEARCH_USE_LABEL}", no_args_is_help=True)
db_app = typer.Typer(help="Database management")
worker_app = typer.Typer(help="Background workers")
users_app = typer.Typer(help="Local user accounts")
app.add_typer(db_app, name="db")
app.add_typer(worker_app, name="worker")
app.add_typer(users_app, name="users")

from secondreader.modalities.cxr.cli import app as cxr_app  # noqa: E402

app.add_typer(cxr_app, name="cxr")


@db_app.command("upgrade")
def db_upgrade() -> None:
    """Apply Alembic migrations (uses SR_DATABASE_URL)."""
    from alembic import command
    from alembic.config import Config

    command.upgrade(Config("alembic.ini"), "head")
    typer.echo("migrated to head")


@db_app.command("create-all")
def db_create_all() -> None:
    """Dev shortcut: create tables directly from models (no migration history)."""
    from secondreader.core.db import create_all
    from secondreader.core.serving.deps import get_container

    create_all(get_container().engine)
    typer.echo("tables created")


@worker_app.command("run")
def worker_run(
    once: bool = typer.Option(False, help="process at most one job then exit"),
    poll_seconds: float = 2.0,
) -> None:
    """Process queued jobs (inference, result push)."""
    from secondreader.core.serving.worker import run_worker

    run_worker(once=once, poll_seconds=poll_seconds)


@users_app.command("create")
def users_create(
    username: str,
    password: str = typer.Option(..., prompt=True, hide_input=True),
    role: str = "clinician",
) -> None:
    from secondreader.core.security.auth import create_user
    from secondreader.core.serving.deps import get_container

    c = get_container()
    with c.sessions() as s:
        u = create_user(s, username, password, role)
        s.commit()
        typer.echo(f"created user {u.username} ({u.role.value})")


@app.command("ingest")
def ingest(paths: list[str], source: str = "cli") -> None:
    """De-identify and ingest DICOM files from disk."""
    from pathlib import Path

    from secondreader.core.serving.deps import get_container

    c = get_container()
    with c.sessions() as s:
        for p in paths:
            for f in sorted(Path(p).rglob("*")) if Path(p).is_dir() else [Path(p)]:
                if f.is_file():
                    r = c.ingest.ingest_bytes(s, f.read_bytes(), source=source)
                    typer.echo(f"{f}: {r.status} {' '.join(r.reasons)}")
        s.commit()


if __name__ == "__main__":
    app()
