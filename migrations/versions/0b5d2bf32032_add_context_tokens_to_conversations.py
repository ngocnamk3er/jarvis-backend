"""add_context_tokens_to_conversations

Revision ID: 0b5d2bf32032
Revises: 1d81879ee43c
Create Date: 2026-08-14 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0b5d2bf32032'
down_revision: Union[str, Sequence[str], None] = '1d81879ee43c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('conversations', sa.Column('context_tokens', sa.Integer(), nullable=False, server_default='0'))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('conversations', 'context_tokens')
