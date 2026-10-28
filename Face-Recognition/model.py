"""
Database models for the Smart Entry face-recognition service.

Tables:
    persons          - one row per known person
    face_encodings   - one row per face embedding (pgvector type)

Run this file directly to create the tables:
    python model.py
"""

from datetime import datetime

from peewee import *
from pgvector.peewee import VectorField

import os
import sys
from pathlib import Path
# =========================
# Database
# =========================

# PostgreSQL database names are case-sensitive.
# The database was created as lowercase "face-reco" (not "face-Recog").
'''
db = PostgresqlDatabase(
    "face-reco",
    user="postgres",
    password="123456789",
    host="localhost",
    port=5432,
)

'''

# =========================
# Database Configuration
# =========================

if getattr(sys, "frozen", False):
    APP_ROOT = Path(sys.executable).resolve().parent
else:
    APP_ROOT = Path(__file__).resolve().parent.parent

#CONFIG_FILE = APP_ROOT / "smart-entry.env"
CONFIG_FILE = Path(
    os.environ.get(
        "SMART_ENTRY_CONFIG_FILE",
        str(APP_ROOT / "smart-entry.env"),
    )
).expanduser()


def load_env_file(file_path):
    if not file_path.is_file():
        return

    with file_path.open("r", encoding="utf-8") as config:
        for line in config:
            line = line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key and key not in os.environ:
                os.environ[key] = value


load_env_file(CONFIG_FILE)

required_settings = (
    "SMART_ENTRY_DB_NAME",
    "SMART_ENTRY_DB_USER",
    "SMART_ENTRY_DB_PASSWORD",
    "SMART_ENTRY_DB_HOST",
    "SMART_ENTRY_DB_PORT",
)

missing_settings = [
    key for key in required_settings
    if not os.environ.get(key)
]

if missing_settings:
    raise RuntimeError(
        "Missing database settings: "
        + ", ".join(missing_settings)
        + ". Check smart-entry.env."
    )

db = PostgresqlDatabase(
    os.environ["SMART_ENTRY_DB_NAME"],
    user=os.environ["SMART_ENTRY_DB_USER"],
    password=os.environ["SMART_ENTRY_DB_PASSWORD"],
    host=os.environ["SMART_ENTRY_DB_HOST"],
    port=int(os.environ["SMART_ENTRY_DB_PORT"]),
)

class BaseModel(Model):

    class Meta:
        database = db


# =========================
# Models
# =========================

class Person(BaseModel):
    """A person registered in the system."""

    name = CharField(max_length=100, unique=True)
    
    created_at = DateTimeField(default=datetime.now)
    profile_image = CharField(max_length=255, null=True)

    class Meta:
        table_name = "persons"



class FaceEmbedding(BaseModel):
    """Stores old and new face embeddings during migration."""

    person = ForeignKeyField(
        Person,
        backref="embeddings",
        on_delete="CASCADE",
    )

    # Old embeddings: face_recognition / dlib
    embedding = VectorField(dimensions=128)

    # New embeddings: InsightFace buffalo_l
    embedding_512 = VectorField(dimensions=512, null=True)

    created_at = DateTimeField(default=datetime.now)

    class Meta:
        table_name = "face_encodings"


class FaceEmbedding512(BaseModel):
    """InsightFace 512-dimensional embeddings."""

    person = ForeignKeyField(
        Person,
        backref="embeddings_512",
        on_delete="CASCADE",
    )

    image_name = CharField(max_length=255)

    embedding = VectorField(dimensions=512)

    created_at = DateTimeField(default=datetime.now)

    class Meta:
        table_name = "face_encodings_512"
        indexes = (
            (("person", "image_name"), True),
        )

class EntryLog(BaseModel):
    """Log of a person's entry."""

    person = ForeignKeyField(
        Person,
        backref="entry_logs",
        on_delete="CASCADE",
    )

    entry_date = DateField()
    entry_time = TimeField()
    exit_time = TimeField(null=True)

    class Meta:
        table_name = "entry_logs"


# =========================
# Init
# =========================

def init_db():
    """Connect, enable the pgvector extension and create the tables."""
    db.connect()

    # The "vector" column type only exists after this extension.
    db.execute_sql("CREATE EXTENSION IF NOT EXISTS vector;")

    # Both tables are listed: peewee does NOT create the Person
    # table automatically just because it is referenced by a FK.
    db.create_tables([Person, FaceEmbedding, FaceEmbedding512, EntryLog], safe=True)

    db.close()


if __name__ == "__main__":
    init_db()
    print("Tables ready: persons, face_encodings, entrylogs")
