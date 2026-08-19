"""move_app_tables_to_jarvis_schema

Revision ID: 7340737df168
Revises: 9bbbb1ab57aa
Create Date: 2026-07-27 08:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '7340737df168'
down_revision: Union[str, Sequence[str], None] = '9bbbb1ab57aa'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Move conversations/subagent_traces from public into their own schema.

    The schema itself is created in migrations/env.py (not here) since
    Alembic's own version_table_schema needs it to exist before it can even
    check/create its tracking table — by the time this migration body runs,
    `jarvis` is already guaranteed to exist.

    Only applies to databases that ran earlier migrations while these tables
    still lived in `public` — a brand-new database never has them there (the
    two migrations before this one already create everything straight into
    `jarvis` via version_table_schema), so this is a no-op there. Same
    defensive shape as the LangGraph checkpoint-table move in
    app/db/connection.py and the alembic_version move in migrations/env.py.
    """
    for table in ("conversations", "subagent_traces"):
        op.execute(
            f"""
            DO $$
            BEGIN
                IF to_regclass('public.{table}') IS NOT NULL
                   AND to_regclass('jarvis.{table}') IS NULL THEN
                    ALTER TABLE public.{table} SET SCHEMA jarvis;
                END IF;
            END $$;
            """
        )


def downgrade() -> None:
    """Move the tables back to public."""
    op.execute("ALTER TABLE jarvis.subagent_traces SET SCHEMA public")
    op.execute("ALTER TABLE jarvis.conversations SET SCHEMA public")
