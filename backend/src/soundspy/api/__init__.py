"""FastAPI layer: a thin HTTP/WebSocket shell around the ``soundspy`` library.

This is the only package that may import FastAPI, Starlette or SQLAlchemy. The library never
imports from here.
"""
