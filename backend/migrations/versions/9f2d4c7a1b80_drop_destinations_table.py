"""drop destinations table

Revision ID: 9f2d4c7a1b80
Revises: c4eccab51eac
Create Date: 2026-09-08

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "9f2d4c7a1b80"
down_revision: str | Sequence[str] | None = "c4eccab51eac"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_table("destinations")


def downgrade() -> None:
    op.create_table(
        "destinations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("country", sa.String(length=100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("climate", sa.String(length=100), nullable=True),
        sa.Column("attractions", sa.JSON(), nullable=True),
        sa.Column("best_seasons", sa.JSON(), nullable=True),
        sa.Column(
            "avg_cost_per_day",
            sa.Numeric(precision=10, scale=2),
            nullable=True,
        ),
        sa.Column("image_url", sa.String(length=500), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_destinations")),
    )
