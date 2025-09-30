"""
Main CLI for Playlister.
"""

import sys
from pathlib import Path

import click
from loguru import logger

from src.utils.config import Config
from src.storage.database import Database
from src.scanner.scanner import MusicScanner


def setup_logging(config: Config):
    """Configure logging based on config."""
    log_config = config.get('logging', default={})
    level = log_config.get('level', 'INFO')
    log_format = log_config.get('format', '[{time:YYYY-MM-DD HH:mm:ss}] {level: <8} | {message}')
    log_file = log_config.get('file', '/data/playlister.log')
    rotation = log_config.get('rotation', '10 MB')
    retention = log_config.get('retention', '1 week')

    # Remove default handler
    logger.remove()

    # Add console handler
    logger.add(
        sys.stderr,
        format=log_format,
        level=level,
        colorize=True
    )

    # Add file handler if path provided
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logger.add(
                log_file,
                format=log_format,
                level=level,
                rotation=rotation,
                retention=retention
            )
        except Exception as e:
            logger.warning(f"Could not set up file logging: {e}")


@click.group()
@click.option('--config', default='/config/config.yaml', help='Path to configuration file')
@click.pass_context
def cli(ctx, config):
    """Playlister - Offline mood-based playlist generator."""
    # Load configuration
    config_path = Path(config)
    if config_path.exists():
        ctx.obj = Config(str(config_path))
    else:
        logger.warning(f"Config file not found at {config}, using defaults")
        ctx.obj = Config()

    # Setup logging
    setup_logging(ctx.obj)


@cli.command()
@click.pass_context
def init(ctx):
    """Initialize database and configuration."""
    config = ctx.obj
    logger.info("Initializing Playlister...")

    # Initialize database
    db_path = config.get('database', 'path', default='/data/playlister.db')
    backup_on_startup = config.get('database', 'backup_on_startup', default=True)
    backup_count = config.get('database', 'backup_count', default=5)

    db = Database(db_path, backup_count=backup_count)
    db.init(backup_on_startup=backup_on_startup)

    logger.info("✓ Database initialized")

    # Create config file if it doesn't exist
    config_path = config.get('paths', 'config', default='/config')
    config_file = Path(config_path) / 'config.yaml'

    if not config_file.exists():
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config.save_to_file(str(config_file))
        logger.info(f"✓ Configuration saved to {config_file}")
    else:
        logger.info(f"Configuration file already exists at {config_file}")

    logger.info("Initialization complete!")


@cli.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--no-extract', is_flag=True, help='Only scan files, do not extract features')
@click.option('--stream', is_flag=True, help='Stream mode for very large libraries (saves memory, no progress bar)')
@click.pass_context
def scan(ctx, path, no_extract, stream):
    """Scan music library and extract features."""
    config = ctx.obj
    logger.info(f"Scanning music library: {path}")

    # Initialize database
    db_path = config.get('database', 'path', default='/data/playlister.db')
    db = Database(db_path)
    db.init(backup_on_startup=False)

    # Initialize scanner
    audio_formats = config.get('audio_formats', default=['.mp3', '.flac', '.wav', '.m4a', '.ogg', '.wma'])
    exclude_patterns = config.get('scanner', 'exclude_patterns', default=['Playlists', 'playlists'])
    follow_symlinks = config.get('scanner', 'follow_symlinks', default=False)
    min_file_size = config.get('scanner', 'min_file_size', default=1024)

    scanner = MusicScanner(
        audio_formats=audio_formats,
        exclude_patterns=exclude_patterns,
        follow_symlinks=follow_symlinks,
        min_file_size=min_file_size
    )

    if exclude_patterns:
        logger.info(f"Excluding directories: {', '.join(exclude_patterns)}")

    if stream:
        logger.info("Using stream mode - suitable for 100k+ file libraries")

    # Scan directory
    scanned_count = 0
    error_count = 0

    with db.session_scope() as session:
        from src.storage.schema import Song

        for scanned_file in scanner.scan_directory(path, recursive=True, show_progress=not stream, stream_mode=stream):
            if scanned_file.error:
                error_count += 1
                logger.error(f"Error scanning {scanned_file.file_path}: {scanned_file.error}")
                continue

            # Check if song already exists
            existing = session.query(Song).filter_by(file_path=scanned_file.file_path).first()

            if existing:
                # Check if file has been modified
                if existing.mtime and existing.mtime >= scanned_file.mtime:
                    logger.debug(f"Skipping unchanged file: {scanned_file.file_path}")
                    continue
                else:
                    logger.info(f"Updating modified file: {scanned_file.file_path}")
                    # Update existing record
                    existing.file_hash = scanned_file.file_hash
                    existing.file_size = scanned_file.file_size
                    existing.mtime = scanned_file.mtime
                    existing.title = scanned_file.title
                    existing.artist = scanned_file.artist
                    existing.album = scanned_file.album
                    existing.album_artist = scanned_file.album_artist
                    existing.year = scanned_file.year
                    existing.genre = scanned_file.genre
                    existing.track_number = scanned_file.track_number
                    existing.duration = scanned_file.duration
                    existing.format = scanned_file.format
                    existing.bitrate = scanned_file.bitrate
                    existing.sample_rate = scanned_file.sample_rate
            else:
                # Create new song record
                song = Song(
                    file_path=scanned_file.file_path,
                    file_hash=scanned_file.file_hash,
                    file_size=scanned_file.file_size,
                    mtime=scanned_file.mtime,
                    title=scanned_file.title,
                    artist=scanned_file.artist,
                    album=scanned_file.album,
                    album_artist=scanned_file.album_artist,
                    year=scanned_file.year,
                    genre=scanned_file.genre,
                    track_number=scanned_file.track_number,
                    duration=scanned_file.duration,
                    format=scanned_file.format,
                    bitrate=scanned_file.bitrate,
                    sample_rate=scanned_file.sample_rate
                )
                session.add(song)
                logger.debug(f"Added: {scanned_file.title or scanned_file.file_path}")

            scanned_count += 1

            # Commit in batches
            if scanned_count % 100 == 0:
                session.commit()

    logger.info(f"✓ Scan complete: {scanned_count} files processed, {error_count} errors")

    if not no_extract:
        logger.info("Starting feature extraction...")
        ctx.invoke(extract)


@cli.command()
@click.option('--force', is_flag=True, help='Force re-extraction of all features')
@click.pass_context
def extract(ctx, force):
    """Extract audio features from scanned songs."""
    config = ctx.obj
    logger.info("Extracting audio features...")

    # Initialize database
    db_path = config.get('database', 'path', default='/data/playlister.db')
    db = Database(db_path)
    db.init(backup_on_startup=False)

    # Import extractor
    try:
        from src.extractor.extractor import FeatureExtractor
    except ImportError as e:
        logger.error(f"Could not import feature extractor: {e}")
        logger.error("Make sure Essentia is installed correctly")
        sys.exit(1)

    # Initialize extractor
    workers = config.get('extraction', 'workers', default=4)
    extractor = FeatureExtractor(db, workers=workers)

    # Extract features
    extracted_count = extractor.extract_all(force_reextract=force)

    logger.info(f"✓ Feature extraction complete: {extracted_count} songs processed")


@cli.command()
@click.pass_context
def classify(ctx):
    """Classify songs by mood."""
    config = ctx.obj
    logger.info("Classifying songs by mood...")

    # Initialize database
    db_path = config.get('database', 'path', default='/data/playlister.db')
    db = Database(db_path)
    db.init(backup_on_startup=False)

    # Import classifier
    try:
        from src.classifier.classifier import MoodClassifier
    except ImportError as e:
        logger.error(f"Could not import mood classifier: {e}")
        sys.exit(1)

    # Initialize classifier
    mood_config = config.get('moods', default={})
    confidence_threshold = config.get('advanced', 'confidence_threshold', default=0.85)
    strict_matching = config.get('advanced', 'strict_matching', default=True)

    classifier = MoodClassifier(
        db,
        mood_config,
        confidence_threshold=confidence_threshold,
        strict_matching=strict_matching
    )

    # Classify songs
    classified_count = classifier.classify_all()

    logger.info(f"✓ Classification complete: {classified_count} songs classified")
    logger.info(f"Using confidence threshold: {confidence_threshold}, strict matching: {strict_matching}")


@cli.command()
@click.option('--mood', help='Mood for playlist (e.g., chill, uplifting)')
@click.option('--count', default=50, help='Number of songs in playlist')
@click.option('--name', help='Playlist name')
@click.pass_context
def generate(ctx, mood, count, name):
    """Generate a playlist."""
    config = ctx.obj

    if not mood:
        logger.error("Please specify a mood with --mood")
        sys.exit(1)

    logger.info(f"Generating {mood} playlist with {count} songs...")

    # Initialize database
    db_path = config.get('database', 'path', default='/data/playlister.db')
    db = Database(db_path)
    db.init(backup_on_startup=False)

    # Import generator
    try:
        from src.generator.generator import PlaylistGenerator
    except ImportError as e:
        logger.error(f"Could not import playlist generator: {e}")
        sys.exit(1)

    # Initialize generator
    generator = PlaylistGenerator(db, config)

    # Generate playlist
    playlist = generator.generate_by_mood(mood, count, name)

    if playlist:
        logger.info(f"✓ Playlist created: {playlist.name} ({playlist.song_count} songs)")
        logger.info(f"  ID: {playlist.id}")
        logger.info(f"  Duration: {playlist.total_duration / 60:.1f} minutes")
        logger.info(f"  Avg transition score: {playlist.avg_transition_score:.2f}")
    else:
        logger.error("Failed to generate playlist")
        sys.exit(1)


@cli.command()
@click.argument('playlist_id', type=int)
@click.option('--format', 'fmt', default='m3u8', type=click.Choice(['m3u', 'm3u8', 'pls', 'json']))
@click.option('--output', help='Output directory (default: from config)')
@click.pass_context
def export(ctx, playlist_id, fmt, output):
    """Export a playlist to file."""
    config = ctx.obj

    # Initialize database
    db_path = config.get('database', 'path', default='/data/playlister.db')
    db = Database(db_path)
    db.init(backup_on_startup=False)

    # Import exporter
    try:
        from src.generator.exporter import PlaylistExporter
    except ImportError as e:
        logger.error(f"Could not import playlist exporter: {e}")
        sys.exit(1)

    # Initialize exporter
    output_dir = output or config.get('paths', 'playlists', default='/playlists')
    music_path = config.get('paths', 'music', default='/music')
    exporter = PlaylistExporter(db, output_dir, music_base_path=music_path)

    # Export playlist
    output_file = exporter.export(playlist_id, fmt)

    if output_file:
        logger.info(f"✓ Playlist exported to: {output_file}")
    else:
        logger.error("Failed to export playlist")
        sys.exit(1)


@cli.command()
@click.pass_context
def stats(ctx):
    """Show library statistics."""
    config = ctx.obj

    # Initialize database
    db_path = config.get('database', 'path', default='/data/playlister.db')
    db = Database(db_path)
    db.init(backup_on_startup=False)

    # Get stats
    stats = db.get_stats()

    logger.info("Library Statistics:")
    logger.info(f"  Total songs: {stats['total_songs']}")
    logger.info(f"  Total playlists: {stats['total_playlists']}")
    logger.info(f"  Classified songs: {stats['classified_songs']}")
    logger.info(f"  Manually classified: {stats['manually_classified']}")
    logger.info(f"  Database size: {stats['database_size_mb']:.2f} MB")

    # Get mood distribution
    with db.session_scope() as session:
        from src.storage.schema import SongClassification
        from sqlalchemy import func

        mood_dist = session.query(
            SongClassification.mood,
            func.count(SongClassification.id).label('count')
        ).group_by(SongClassification.mood).all()

        if mood_dist:
            logger.info("\nMood Distribution:")
            for mood, count in mood_dist:
                logger.info(f"  {mood}: {count}")


@cli.command()
@click.pass_context
def history(ctx):
    """Show playlist generation history."""
    config = ctx.obj

    # Initialize database
    db_path = config.get('database', 'path', default='/data/playlister.db')
    db = Database(db_path)
    db.init(backup_on_startup=False)

    with db.session_scope() as session:
        from src.storage.schema import Playlist

        playlists = session.query(Playlist).order_by(Playlist.created_at.desc()).limit(20).all()

        if not playlists:
            logger.info("No playlists created yet")
            return

        logger.info("Recent Playlists:")
        for pl in playlists:
            logger.info(f"  [{pl.id}] {pl.name}")
            logger.info(f"      Mood: {pl.mood}, Songs: {pl.song_count}, Created: {pl.created_at}")


def main():
    """Entry point."""
    cli(obj=None)


if __name__ == '__main__':
    main()