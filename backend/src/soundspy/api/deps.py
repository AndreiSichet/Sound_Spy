"""FastAPI dependencies: app config and database sessions."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from soundspy.config import AppConfig


def get_config(request: Request) -> AppConfig:
    return request.app.state.config


def get_db(request: Request) -> Iterator[Session]:
    session: Session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()


ConfigDep = Annotated[AppConfig, Depends(get_config)]
DbDep = Annotated[Session, Depends(get_db)]
