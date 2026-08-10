"""add_user_id_to_conversations

Revision ID: 1d81879ee43c
Revises: 7340737df168
Create Date: 2026-08-10 17:41:47.244102

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1d81879ee43c'
down_revision: Union[str, Sequence[str], None] = '7340737df168'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # No pre-existing conversation can have an owner (auth didn't exist
    # before this migration) — truncate rather than backfill a fake user_id.
    # Cascades to subagent_traces via its FK.
    op.execute("TRUNCATE TABLE jarvis.conversations CASCADE")
    op.add_column('conversations', sa.Column('user_id', sa.Text(), nullable=False), schema='jarvis')
    op.create_index(op.f('ix_jarvis_conversations_user_id'), 'conversations', ['user_id'], unique=False, schema='jarvis')


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_jarvis_conversations_user_id'), table_name='conversations', schema='jarvis')
    op.drop_column('conversations', 'user_id', schema='jarvis')
