"""Per-stage CLI (typer) for debugging: `uv run ber prepare --split train`, `uv run ber block --split test` ...
The end-to-end runner is `python -m business_entity_resolution.blocking_main`."""
from __future__ import annotations

from pathlib import Path

import typer

from .config import BlockingConfig

app = typer.Typer(add_completion=False, help="Business entity resolution: blocking stages")


def _cfg(sample: int | None, pool_fraction: float | None, threads: int | None, cap: int | None) -> BlockingConfig:
    cfg = BlockingConfig()
    if sample is not None: cfg.sample_s1 = sample
    if pool_fraction is not None: cfg.pool_fraction = pool_fraction
    if threads is not None: cfg.threads = threads
    if cap is not None: cfg.cap = cap
    return cfg


@app.command()
def prepare(split: str = typer.Option(..., help="train|test")):
    from .prepare import stage_prepare
    stage_prepare(BlockingConfig(), split)


@app.command("learn-aliases")
def learn_aliases():
    from .aliases_stage import stage_aliases
    stage_aliases(BlockingConfig())


@app.command()
def block(split: str = typer.Option(...), sample: int | None = None, pool_fraction: float | None = None,
          threads: int | None = None, cap: int | None = None):
    from .blocking.pipeline import stage_block
    stage_block(_cfg(sample, pool_fraction, threads, cap), split)


@app.command()
def evaluate():
    from .evaluate import stage_evaluate
    stage_evaluate(BlockingConfig())


@app.command()
def export(split: str = "test"):
    from .export import stage_export
    stage_export(BlockingConfig(), split)


@app.command()
def normalize(text: str, kind: str = "name"):
    """Show how one string is normalised (debug helper)."""
    from .normalize import AddressNormalizer, NameNormalizer, split_components
    if kind == "name":
        full, core = NameNormalizer()(text)
        typer.echo(f"full: {full!r}\ncore: {core!r}")
    else:
        comps = split_components(text)
        addr, state = AddressNormalizer()(comps)
        typer.echo(f"components: {comps}\naddr_norm: {addr!r}\nstate: {state}")


if __name__ == "__main__":
    app()
