
"""
Database models for the Smart Entry face-recognition service.

Tables:
    persons          - registered people
    face_encodings   - legacy and migrated face embeddings
    face_encodings_512 - InsightFace 512-dimensional embeddings
    entry_logs       - attendance records

Run this file directly to create the tables:
    python model.py
"""

import os
import sys

from pathlib import Path
from datetime import datetime

from peewee import *
from pgvector.peewee import VectorField


# ============================================================
# APPLICATION PATHS
# ============================================================

if getattr(sys, "frozen", False):

    # Running as a packaged EXE
    APP_ROOT = (
        Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData"))
        / "SmartEntry"
    )

else:

    # Running in development mode
    APP_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# DATABASE CONFIGURATION FILE
# ============================================================

# Default configuration file
DEFAULT_CONFIG_FILE = APP_ROOT / "smart-entry.env"

# Allow an explicit configuration path through environment variable
CONFIG_FILE = Path(
    os.environ.get(
        "SMART_ENTRY_CONFIG_FILE",
        str(DEFAULT_CONFIG_FILE),
    )
).expanduser()


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

def load_env_file(file_path: Path):
    """
    Load environment variables from smart-entry.env.

    Existing system environment variables have priority
    over values stored in the configuration file.
    """

    if not file_path.is_file():
        return

    with file_path.open("r", encoding="utf-8") as config:

        for line in config:

            line = line.strip()

            if (
                not line
                or line.startswith("#")
                or "=" not in line
            ):
                continue

            key, value = line.split("=", 1)

            key = key.strip()
            value = value.strip()

            if key and key not in os.environ:
                os.environ[key] = value


load_env_file(CONFIG_FILE)


# ============================================================
# VALIDATE DATABASE SETTINGS
# ============================================================

required_settings = (
    "SMART_ENTRY_DB_NAME",
    "SMART_ENTRY_DB_USER",
    "SMART_ENTRY_DB_PASSWORD",
    "SMART_ENTRY_DB_HOST",
    "SMART_ENTRY_DB_PORT",
)

missing_settings = [
    key
    for key in required_settings
    if not os.environ.get(key)
]

if missing_settings:

    raise RuntimeError(
        "Missing database settings: "
        + ", ".join(missing_settings)
        + f". Check configuration file: {CONFIG_FILE}"
    )


# ============================================================
# POSTGRESQL DATABASE CONNECTION
# ============================================================

db = PostgresqlDatabase(
    os.environ["SMART_ENTRY_DB_NAME"],
    user=os.environ["SMART_ENTRY_DB_USER"],
    password=os.environ["SMART_ENTRY_DB_PASSWORD"],
    host=os.environ["SMART_ENTRY_DB_HOST"],
    port=int(os.environ["SMART_ENTRY_DB_PORT"]),
)


# ============================================================
# BASE MODEL
# ============================================================

class BaseModel(Model):

    class Meta:
        database = db


# ============================================================
# PERSON MODEL
# ============================================================

class Person(BaseModel):
    """A person registered in the Smart Entry system."""

    name = CharField(
        max_length=100,
        unique=True,
    )

    created_at = DateTimeField(
        default=datetime.now,
    )

    profile_image = CharField(
        max_length=255,
        null=True,
    )

    class Meta:
        table_name = "persons"


# ============================================================
# LEGACY AND MIGRATION FACE EMBEDDINGS
# ============================================================

class FaceEmbedding(BaseModel):
    """
    Stores old and migrated face embeddings.

    Old embeddings:
        face_recognition / dlib (128 dimensions)

    New embeddings:
        InsightFace (512 dimensions)
    """

    person = ForeignKeyField(
        Person,
        backref="embeddings",
        on_delete="CASCADE",
    )

    embedding = VectorField(
        dimensions=128,
    )

    embedding_512 = VectorField(
        dimensions=512,
        null=True,
    )

    created_at = DateTimeField(
        default=datetime.now,
    )

    class Meta:
        table_name = "face_encodings"


# ============================================================
# INSIGHTFACE 512-DIMENSIONAL EMBEDDINGS
# ============================================================

class FaceEmbedding512(BaseModel):
    """Stores InsightFace 512-dimensional face embeddings."""

    person = ForeignKeyField(
        Person,
        backref="embeddings_512",
        on_delete="CASCADE",
    )

    image_name = CharField(
        max_length=255,
    )

    embedding = VectorField(
        dimensions=512,
    )

    created_at = DateTimeField(
        default=datetime.now,
    )

    class Meta:
        table_name = "face_encodings_512"

        indexes = (
            (("person", "image_name"), True),
        )


# ============================================================
# ATTENDANCE LOG MODEL
# ============================================================

class EntryLog(BaseModel):
    """Stores a person's entry and exit records."""

    person = ForeignKeyField(
        Person,
        backref="entry_logs",
        on_delete="CASCADE",
    )

    entry_date = DateField()

    entry_time = TimeField()

    exit_time = TimeField(
        null=True,
    )

    class Meta:
        table_name = "entry_logs"


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_db():
    """
    Connect to PostgreSQL, enable pgvector
    and create missing database tables.

    Existing tables and records are preserved.
    """

    db.connect()

    try:

        # Enable pgvector extension
        db.execute_sql(
            "CREATE EXTENSION IF NOT EXISTS vector;"
        )

        # Create missing tables without deleting existing tables
        db.create_tables(
            [
                Person,
                FaceEmbedding,
                FaceEmbedding512,
                EntryLog,
            ],
            safe=True,
        )

        print("Database initialized successfully.")

    finally:

        if not db.is_closed():
            db.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    init_db()

    print(
        "Tables ready: "
        "persons, face_encodings, "
        "face_encodings_512, entry_logs"
    )
