"""
neo4j/connection.py
====================
Driver singleton — reads credentials from environment / .env file.

Environment variables required:
    NEO4J_URI       e.g. neo4j+s://xxxxxxxx.databases.neo4j.io
    NEO4J_USER      e.g. neo4j
    NEO4J_PASSWORD  your Aura password
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j import Driver

load_dotenv()

_driver: Driver | None = None


def get_driver() -> Driver:
    """
    Return (or create) the module-level Neo4j driver singleton.
    Credentials are read from environment variables.
    """
    global _driver
    if _driver is None:
        uri      = os.environ.get("NEO4J_URI", "")
        user     = os.environ.get("NEO4J_USER", "")
        password = os.environ.get("NEO4J_PASSWORD", "")
        if not uri:
            raise EnvironmentError(
                "NEO4J_URI is not set. "
                "Add it to your .env file or set it as an environment variable."
            )
        _driver = GraphDatabase.driver(uri, auth=(user, password))
    return _driver


def close_driver() -> None:
    """Close the driver and release the singleton so the next call re-creates it."""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


def reset_driver() -> None:
    """
    Force-reset the singleton without calling close() (useful in tests where
    the driver is a mock).
    """
    global _driver
    _driver = None


def test_connection() -> bool:
    """
    Ping the database and print server info.
    Returns True if the connection is healthy.
    """
    driver = get_driver()
    try:
        with driver.session() as session:
            result = session.run("RETURN 1 AS ping")
            record = result.single()
            if record and record["ping"] == 1:
                info = driver.get_server_info()
                print(f"Neo4j OK  address={info.address}  agent={info.agent}")
                return True
    except Exception as exc:
        print(f"Neo4j connection failed: {exc}")
    return False
