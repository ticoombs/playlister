"""
Database schema for Playlister.
Uses SQLAlchemy ORM for database management.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean,
    ForeignKey, Text, JSON, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Song(Base):
    """Represents a music file in the library."""
    __tablename__ = 'songs'

    id = Column(Integer, primary_key=True)
    file_path = Column(String(512), unique=True, nullable=False, index=True)
    file_hash = Column(String(64), index=True)  # SHA-256 hash for duplicate detection
    file_size = Column(Integer)  # Size in bytes
    mtime = Column(DateTime)  # Last modification time

    # Metadata
    title = Column(String(256))
    artist = Column(String(256), index=True)
    album = Column(String(256), index=True)
    album_artist = Column(String(256))
    year = Column(Integer, index=True)
    genre = Column(String(128))
    track_number = Column(Integer)
    duration = Column(Float)  # Duration in seconds

    # File format
    format = Column(String(16))  # mp3, flac, wav, etc.
    bitrate = Column(Integer)
    sample_rate = Column(Integer)

    # Processing metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    scanned_at = Column(DateTime)

    # Relationships
    features = relationship("SongFeatures", back_populates="song", uselist=False, cascade="all, delete-orphan")
    classifications = relationship("SongClassification", back_populates="song", cascade="all, delete-orphan")
    playlist_entries = relationship("PlaylistSong", back_populates="song", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index('idx_artist_album', 'artist', 'album'),
        Index('idx_artist_year', 'artist', 'year'),
    )

    def __repr__(self):
        return f"<Song(id={self.id}, title='{self.title}', artist='{self.artist}')>"


class SongFeatures(Base):
    """Audio features extracted from a song."""
    __tablename__ = 'song_features'

    id = Column(Integer, primary_key=True)
    song_id = Column(Integer, ForeignKey('songs.id'), unique=True, nullable=False, index=True)

    # Basic audio features
    tempo = Column(Float)  # BPM
    energy = Column(Float, index=True)  # 0-1
    valence = Column(Float, index=True)  # 0-1 (positivity)
    danceability = Column(Float, index=True)  # 0-1
    acousticness = Column(Float)  # 0-1
    instrumentalness = Column(Float)  # 0-1
    loudness = Column(Float)  # dB

    # Musical key
    key = Column(Integer)  # 0-11 (C, C#, D, etc.)
    mode = Column(Integer)  # 0=minor, 1=major
    camelot_key = Column(String(4))  # Camelot notation (e.g., "8A", "5B")

    # Spectral features (stored as JSON for flexibility)
    spectral_features = Column(JSON)  # MFCC, spectral centroid, etc.

    # Full feature vector (for similarity search)
    feature_vector = Column(JSON)  # Complete feature vector for cosine similarity

    # Extraction metadata
    extraction_version = Column(String(32))  # Version of extraction algorithm
    extracted_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    song = relationship("Song", back_populates="features")

    def __repr__(self):
        return f"<SongFeatures(song_id={self.song_id}, tempo={self.tempo}, energy={self.energy})>"


class SongClassification(Base):
    """Mood classification for a song."""
    __tablename__ = 'song_classifications'

    id = Column(Integer, primary_key=True)
    song_id = Column(Integer, ForeignKey('songs.id'), nullable=False, index=True)

    # Classification
    mood = Column(String(64), nullable=False, index=True)
    confidence = Column(Float)  # 0-1 confidence score

    # Classification method
    method = Column(String(64))  # 'rule_based', 'clustering', 'pretrained', 'manual'
    manual_override = Column(Boolean, default=False, index=True)

    # Metadata
    classified_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    song = relationship("Song", back_populates="classifications")

    # Constraints
    __table_args__ = (
        UniqueConstraint('song_id', 'mood', name='uq_song_mood'),
        Index('idx_mood_confidence', 'mood', 'confidence'),
    )

    def __repr__(self):
        return f"<SongClassification(song_id={self.song_id}, mood='{self.mood}', confidence={self.confidence})>"


class Playlist(Base):
    """Generated playlist."""
    __tablename__ = 'playlists'

    id = Column(Integer, primary_key=True)
    name = Column(String(256), nullable=False)
    description = Column(Text)

    # Generation parameters
    mood = Column(String(64), index=True)
    seed_song_id = Column(Integer, ForeignKey('songs.id'))
    generation_method = Column(String(64))  # 'mood_based', 'seed_based', 'custom'
    parameters = Column(JSON)  # Generation parameters used

    # Playlist metadata
    song_count = Column(Integer)
    total_duration = Column(Float)  # Total duration in seconds
    avg_transition_score = Column(Float)  # Average transition quality

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    exported_at = Column(DateTime)

    # Relationships
    songs = relationship("PlaylistSong", back_populates="playlist", cascade="all, delete-orphan", order_by="PlaylistSong.position")
    seed_song = relationship("Song", foreign_keys=[seed_song_id])

    def __repr__(self):
        return f"<Playlist(id={self.id}, name='{self.name}', mood='{self.mood}', songs={self.song_count})>"


class PlaylistSong(Base):
    """Songs in a playlist with ordering and transition scores."""
    __tablename__ = 'playlist_songs'

    id = Column(Integer, primary_key=True)
    playlist_id = Column(Integer, ForeignKey('playlists.id'), nullable=False, index=True)
    song_id = Column(Integer, ForeignKey('songs.id'), nullable=False, index=True)

    # Ordering
    position = Column(Integer, nullable=False)

    # Transition quality
    transition_score = Column(Float)  # Quality of transition to next song
    transition_notes = Column(Text)  # Details about transition

    # Relationship
    playlist = relationship("Playlist", back_populates="songs")
    song = relationship("Song", back_populates="playlist_entries")

    # Constraints
    __table_args__ = (
        UniqueConstraint('playlist_id', 'position', name='uq_playlist_position'),
        Index('idx_playlist_position', 'playlist_id', 'position'),
    )

    def __repr__(self):
        return f"<PlaylistSong(playlist_id={self.playlist_id}, position={self.position}, song_id={self.song_id})>"


class UserPreference(Base):
    """User preferences and settings."""
    __tablename__ = 'user_preferences'

    id = Column(Integer, primary_key=True)
    key = Column(String(128), unique=True, nullable=False)
    value = Column(JSON)
    description = Column(Text)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<UserPreference(key='{self.key}')>"


class ScanProgress(Base):
    """Track scanning progress for resumption capability."""
    __tablename__ = 'scan_progress'

    id = Column(Integer, primary_key=True)
    scan_id = Column(String(36), unique=True, nullable=False, index=True)  # UUID
    root_path = Column(String(512), nullable=False)
    current_directory = Column(String(512))  # Last directory being processed
    directories_completed = Column(JSON)  # List of completed directory paths
    files_processed = Column(Integer, default=0)
    files_added = Column(Integer, default=0)
    files_updated = Column(Integer, default=0)
    files_skipped = Column(Integer, default=0)
    files_errored = Column(Integer, default=0)
    status = Column(String(32), default='running', index=True)  # 'running', 'completed', 'interrupted', 'failed'
    started_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<ScanProgress(scan_id='{self.scan_id}', status='{self.status}', files={self.files_processed})>"


class SchemaVersion(Base):
    """Track database schema version for migrations."""
    __tablename__ = 'schema_version'

    id = Column(Integer, primary_key=True)
    version = Column(Integer, nullable=False, unique=True)
    description = Column(String(256))
    applied_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<SchemaVersion(version={self.version})>"