"""create incidents, detections, vlm results, reviews and settlements

Revision ID: ba92658850ab
Revises: c06a65ed31cd
Create Date: 2026-11-11 02:45:10.300610

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "ba92658850ab"
down_revision: str | Sequence[str] | None = "c06a65ed31cd"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "incidents",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("camera_id", sa.String(length=128), nullable=False),
        sa.Column("location_id", sa.Integer(), nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("severity", sa.String(length=8), nullable=False),
        sa.Column("first_anomaly_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_anomaly_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("evidence_start_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("evidence_end_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("evidence_clip_path", sa.String(length=1024), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "severity IN ('minor', 'moderate', 'critical')",
            name="ck_incidents_severity",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_incidents_first_anomaly_at_id", "incidents", ["first_anomaly_at", "id"]
    )
    op.create_index("ix_incidents_camera_id", "incidents", ["camera_id"])
    op.create_index("ix_incidents_location_id", "incidents", ["location_id"])
    op.create_index("ix_incidents_event_type", "incidents", ["event_type"])
    op.create_index("ix_incidents_severity", "incidents", ["severity"])

    op.create_table(
        "detections",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("incident_id", sa.Integer(), nullable=False),
        sa.Column("segment_id", sa.Integer(), nullable=True),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("class_label", sa.String(length=64), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("bbox_x1", sa.Float(), nullable=False),
        sa.Column("bbox_y1", sa.Float(), nullable=False),
        sa.Column("bbox_x2", sa.Float(), nullable=False),
        sa.Column("bbox_y2", sa.Float(), nullable=False),
        sa.Column("is_flagged", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1", name="ck_detections_confidence"
        ),
        sa.CheckConstraint(
            "bbox_x1 >= 0 AND bbox_y1 >= 0 AND bbox_x2 <= 1 AND bbox_y2 <= 1"
            " AND bbox_x1 <= bbox_x2 AND bbox_y1 <= bbox_y2",
            name="ck_detections_bbox",
        ),
        sa.ForeignKeyConstraint(["incident_id"], ["incidents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["segment_id"], ["video_segments.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_detections_incident_detected_at",
        "detections",
        ["incident_id", "detected_at"],
    )

    op.create_table(
        "vlm_results",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("incident_id", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("assessment", sa.String(length=9), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "assessment IN ('confirmed', 'rejected', 'uncertain')",
            name="ck_vlm_results_assessment",
        ),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1", name="ck_vlm_results_confidence"
        ),
        sa.ForeignKeyConstraint(["incident_id"], ["incidents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_vlm_results_incident_processed_at",
        "vlm_results",
        ["incident_id", "processed_at"],
    )

    op.create_table(
        "reviews",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("incident_id", sa.Integer(), nullable=False),
        sa.Column("outcome", sa.String(length=9), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("corrected_type", sa.String(length=64), nullable=True),
        sa.Column("reviewer_id", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_by", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "outcome IN ('accepted', 'dismissed', 'escalated')",
            name="ck_reviews_outcome",
        ),
        sa.ForeignKeyConstraint(["incident_id"], ["incidents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("incident_id"),
    )
    op.create_index("ix_reviews_outcome", "reviews", ["outcome"])

    op.create_table(
        "settlements",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("incident_id", sa.Integer(), nullable=False),
        sa.Column("outcome", sa.String(length=9), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("settled_by", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_by", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "outcome IN ('resolved', 'dismissed')", name="ck_settlements_outcome"
        ),
        sa.ForeignKeyConstraint(["incident_id"], ["incidents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("incident_id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("settlements")
    op.drop_index("ix_reviews_outcome", table_name="reviews")
    op.drop_table("reviews")
    op.drop_index("ix_vlm_results_incident_processed_at", table_name="vlm_results")
    op.drop_table("vlm_results")
    op.drop_index("ix_detections_incident_detected_at", table_name="detections")
    op.drop_table("detections")
    op.drop_index("ix_incidents_severity", table_name="incidents")
    op.drop_index("ix_incidents_event_type", table_name="incidents")
    op.drop_index("ix_incidents_location_id", table_name="incidents")
    op.drop_index("ix_incidents_camera_id", table_name="incidents")
    op.drop_index("ix_incidents_first_anomaly_at_id", table_name="incidents")
    op.drop_table("incidents")
