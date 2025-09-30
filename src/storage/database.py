"""
Database management and operations.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from loguru import logger

from .schema import Base, SchemaVersion

# Current schema version
CURRENT_SCHEMA_VERSION = 1


class Database:
    """Database manager for Playlister."""

    def __init__(self, db_path: str, backup_count: int = 5):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
            backup_count: Number of backups to keep
        """
        self.db_path = Path(db_path)
        self.backup_count = backup_count
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None

    def init(self, backup_on_startup: bool = True):
        """
        Initialize database, create tables if needed, run migrations.

        Args:
            backup_on_startup: Whether to backup database on startup
        """
        # Create directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Backup existing database
        if backup_on_startup and self.db_path.exists():
            self._backup_database()

        # Create engine
        db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(
            db_url,
            echo=False,
            connect_args={"check_same_thread": False}
        )

        # Enable foreign keys for SQLite
        @event.listens_for(Engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for better concurrency
            cursor.close()

        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

        # Create tables
        Base.metadata.create_all(bind=self.engine)

        # Run migrations
        self._run_migrations()

        logger.info(f"Database initialized at {self.db_path}")

    @contextmanager
    def session_scope(self):
        """
        Provide a transactional scope for database operations.

        Usage:
            with db.session_scope() as session:
                session.add(obj)
                session.commit()
        """
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized. Call init() first.")

        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()

    def get_session(self) -> Session:
        """
        Get a new database session.
        Caller is responsible for closing the session.
        """
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized. Call init() first.")
        return self.SessionLocal()

    def _backup_database(self):
        """Create a backup of the database."""
        if not self.db_path.exists():
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.db_path.with_suffix(f".backup_{timestamp}.db")

        try:
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to {backup_path}")

            # Clean up old backups
            self._cleanup_old_backups()
        except Exception as e:
            logger.error(f"Failed to backup database: {e}")

    def _cleanup_old_backups(self):
        """Remove old backup files, keeping only the most recent ones."""
        backup_pattern = f"{self.db_path.stem}.backup_*.db"
        backup_dir = self.db_path.parent
        backups = sorted(backup_dir.glob(backup_pattern), key=os.path.getmtime, reverse=True)

        # Remove old backups
        for backup in backups[self.backup_count:]:
            try:
                backup.unlink()
                logger.debug(f"Removed old backup: {backup}")
            except Exception as e:
                logger.warning(f"Failed to remove old backup {backup}: {e}")

    def _run_migrations(self):
        """Run database migrations if needed."""
        with self.session_scope() as session:
            # Get current schema version
            current_version = self._get_schema_version(session)

            if current_version < CURRENT_SCHEMA_VERSION:
                logger.info(f"Running migrations from version {current_version} to {CURRENT_SCHEMA_VERSION}")

                # Run migrations
                for version in range(current_version + 1, CURRENT_SCHEMA_VERSION + 1):
                    self._apply_migration(session, version)

                logger.info("Migrations completed successfully")
            else:
                logger.debug(f"Database schema is up to date (version {current_version})")

    def _get_schema_version(self, session: Session) -> int:
        """Get current schema version from database."""
        try:
            version_record = session.query(SchemaVersion).order_by(
                SchemaVersion.version.desc()
            ).first()
            return version_record.version if version_record else 0
        except Exception:
            # Table doesn't exist yet, this is a new database
            return 0

    def _apply_migration(self, session: Session, version: int):
        """Apply a specific migration version."""
        logger.info(f"Applying migration version {version}")

        # Add migration logic here as needed
        if version == 1:
            # Initial schema - already created by Base.metadata.create_all
            pass

        # Record migration
        version_record = SchemaVersion(
            version=version,
            description=f"Migration to version {version}",
            applied_at=datetime.utcnow()
        )
        session.add(version_record)
        session.commit()

    def optimize(self):
        """Optimize database (vacuum, analyze)."""
        if not self.engine:
            raise RuntimeError("Database not initialized. Call init() first.")

        try:
            with self.engine.connect() as conn:
                conn.execute("VACUUM")
                conn.execute("ANALYZE")
            logger.info("Database optimized")
        except Exception as e:
            logger.error(f"Failed to optimize database: {e}")

    def get_stats(self) -> dict:
        """Get database statistics."""
        with self.session_scope() as session:
            from .schema import Song, Playlist, SongClassification

            stats = {
                "total_songs": session.query(Song).count(),
                "total_playlists": session.query(Playlist).count(),
                "classified_songs": session.query(SongClassification).filter(
                    SongClassification.manual_override == False
                ).count(),
                "manually_classified": session.query(SongClassification).filter(
                    SongClassification.manual_override == True
                ).count(),
                "database_size_mb": self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
            }
            return stats

    def close(self):
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
            logger.debug("Database connections closed")