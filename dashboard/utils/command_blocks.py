"""Streamlit helpers for presenting shell commands instead of executing them."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Iterable, Sequence

import streamlit as st


def _format_command(command: Sequence[object]) -> str:
    parts = []
    for item in command:
        text = str(item)
        parts.append(shlex.quote(text))
    return " ".join(parts)


def show_command_block(
    command: Sequence[object],
    *,
    description: str | None = None,
    cwd: str | Path | None = None,
    success_hint: str | None = None,
    show_hint: bool = True,
) -> None:
    """Render a copyable command snippet with optional context."""

    if description:
        st.markdown(description)

    if cwd:
        st.caption(f"Run inside: `{Path(cwd)}`")

    st.code(_format_command(command), language="bash")

    if show_hint:
        hint_lines = ["Copy the command above and run it manually in your terminal."]
        if success_hint:
            hint_lines.append(success_hint)

        st.info(" ".join(hint_lines))


def show_command_list(
    commands: Iterable[Sequence[object]],
    *,
    description: str | None = None,
    cwd: str | Path | None = None,
) -> None:
    """Render multiple commands sequentially with copyable blocks."""

    if description:
        st.markdown(description)

    commands = list(commands)
    for index, command in enumerate(commands):
        show_command_block(
            command,
            cwd=cwd,
            show_hint=index == len(commands) - 1,
        )