import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool
from sqlalchemy import text as sa_text
from alembic import context

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import settings
from app.db.models import Base, SCHEMA

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

db_url = settings.DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
config.set_main_option("sqlalchemy.url", db_url)

target_metadata = Base.metadata

# target_metadata.tables keys are schema-qualified ("jarvis.conversations"),
# but include_object's `name` arg is always the bare table name — compare
# against the unqualified set so this still matches regardless of schema.
_known_tables = {t.split(".")[-1] for t in target_metadata.tables}


def include_object(obj, name, type_, reflected, compare_to):
    """Only manage tables defined in our models; ignore LangGraph checkpoint tables."""
    if type_ == "table" and name not in _known_tables:
        return False
    return True


def run_migrations_offline() -> None:
    context.configure(
        url=db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
        include_schemas=True,
        version_table_schema=SCHEMA,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        # Alembic's own version_table_schema=SCHEMA (below) needs the schema to
        # exist before it can even check/create its tracking table — bootstrap
        # it here rather than in a tracked migration, so this works the same on
        # a brand-new database as on one that already ran earlier migrations.
        connection.execute(sa_text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))

        # One-time move of Alembic's own version-tracking table out of public,
        # for databases that already ran migrations before version_table_schema
        # was introduced (their current revision is recorded in
        # public.alembic_version). A brand-new database has neither table yet —
        # Alembic just creates jarvis.alembic_version itself in that case.
        public_has_version = connection.execute(
            sa_text("SELECT to_regclass('public.alembic_version') IS NOT NULL")
        ).scalar()
        schema_has_version = connection.execute(
            sa_text(f"SELECT to_regclass('{SCHEMA}.alembic_version') IS NOT NULL")
        ).scalar()
        if public_has_version and not schema_has_version:
            connection.execute(sa_text(f"ALTER TABLE public.alembic_version SET SCHEMA {SCHEMA}"))
        connection.commit()
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
            include_schemas=True,
            version_table_schema=SCHEMA,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
