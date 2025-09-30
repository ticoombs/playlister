"""
Music library scanner for file discovery and metadata extraction.
"""

import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Generator, Tuple
from dataclasses import dataclass

from mutagen import File as MutagenFile
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from mutagen.wave import WAVE
from mutagen.mp4 import MP4
from mutagen.oggvorbis import OggVorbis
from mutagen.asf import ASF
from loguru import logger
from tqdm import tqdm


@dataclass
class ScannedFile:
    """Represents a scanned music file with metadata."""
    file_path: str
    file_hash: str
    file_size: int
    mtime: datetime

    # Metadata
    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    album_artist: Optional[str] = None
    year: Optional[int] = None
    genre: Optional[str] = None
    track_number: Optional[int] = None
    duration: Optional[float] = None

    # File format
    format: Optional[str] = None
    bitrate: Optional[int] = None
    sample_rate: Optional[int] = None

    # Errors
    error: Optional[str] = None


class MusicScanner:
    """Scanner for music library files."""

    # Supported audio formats
    AUDIO_EXTENSIONS = {'.mp3', '.flac', '.wav', '.m4a', '.ogg', '.wma', '.opus'}

    def __init__(
        self,
        audio_formats: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        follow_symlinks: bool = False,
        min_file_size: int = 0
    ):
        """
        Initialize scanner.

        Args:
            audio_formats: List of audio format extensions (e.g., ['.mp3', '.flac'])
                          If None, uses default AUDIO_EXTENSIONS
            exclude_patterns: List of directory name patterns to exclude from scanning
            follow_symlinks: Whether to follow symbolic links during scanning
            min_file_size: Minimum file size in bytes (files smaller are skipped)
        """
        if audio_formats:
            self.audio_extensions = {ext.lower() for ext in audio_formats}
        else:
            self.audio_extensions = self.AUDIO_EXTENSIONS

        self.exclude_patterns = set(exclude_patterns) if exclude_patterns else set()
        self.follow_symlinks = follow_symlinks
        self.min_file_size = min_file_size

    def scan_directory(
        self,
        root_path: str,
        recursive: bool = True,
        show_progress: bool = True,
        stream_mode: bool = False
    ) -> Generator[ScannedFile, None, None]:
        """
        Scan directory for music files.

        Args:
            root_path: Root directory to scan
            recursive: Whether to scan subdirectories recursively
            show_progress: Whether to show progress bar
            stream_mode: If True, stream files without counting first (saves memory for huge libraries)

        Yields:
            ScannedFile objects for each discovered audio file
        """
        root = Path(root_path)

        if not root.exists():
            logger.error(f"Path does not exist: {root_path}")
            return

        if not root.is_dir():
            logger.error(f"Path is not a directory: {root_path}")
            return

        if stream_mode:
            # Stream mode: don't load file list into memory, but can't show progress percentage
            logger.info("Streaming scan mode (no file count available)")
            file_count = 0

            for file_path in self._find_audio_files(root, recursive):
                file_count += 1
                if show_progress and file_count % 100 == 0:
                    logger.info(f"Processed {file_count} files...")

                scanned = self.scan_file(file_path)
                yield scanned

            if show_progress:
                logger.info(f"Total files processed: {file_count}")

        else:
            # Standard mode: load file list for progress bar
            # For large libraries (100k+ files), this uses ~10-50 MB RAM for paths
            audio_files = list(self._find_audio_files(root, recursive))

            if not audio_files:
                logger.warning(f"No audio files found in {root_path}")
                return

            logger.info(f"Found {len(audio_files)} audio files")

            # Process files with progress bar
            iterator = tqdm(audio_files, desc="Scanning files", disable=not show_progress)

            for file_path in iterator:
                if show_progress:
                    iterator.set_postfix({"file": file_path.name[:30]})

                scanned = self.scan_file(file_path)
                yield scanned

    def _find_audio_files(self, root: Path, recursive: bool) -> Generator[Path, None, None]:
        """
        Find all audio files in directory, respecting exclusion patterns.

        Args:
            root: Root directory
            recursive: Whether to search recursively

        Yields:
            Path objects for audio files
        """
        if recursive:
            # Use os.walk for better control over directory traversal
            for dirpath, dirnames, filenames in os.walk(root, followlinks=self.follow_symlinks):
                # Filter out excluded directories (modifies dirnames in-place to prune traversal)
                if self.exclude_patterns:
                    dirnames[:] = [
                        d for d in dirnames
                        if d not in self.exclude_patterns
                    ]

                # Find matching audio files in this directory
                current_dir = Path(dirpath)
                for filename in filenames:
                    filepath = current_dir / filename

                    # Check file extension
                    if filepath.suffix.lower() in self.audio_extensions:
                        # Check file size if minimum specified
                        if self.min_file_size > 0:
                            try:
                                if filepath.stat().st_size < self.min_file_size:
                                    logger.debug(f"Skipping small file: {filepath}")
                                    continue
                            except OSError:
                                continue

                        yield filepath
        else:
            # Non-recursive: just scan immediate directory
            for ext in self.audio_extensions:
                for filepath in root.glob(f"*{ext}"):
                    # Check file size if minimum specified
                    if self.min_file_size > 0:
                        try:
                            if filepath.stat().st_size < self.min_file_size:
                                logger.debug(f"Skipping small file: {filepath}")
                                continue
                        except OSError:
                            continue

                    yield filepath

    def scan_file(self, file_path: Path) -> ScannedFile:
        """
        Scan a single audio file and extract metadata.

        Args:
            file_path: Path to audio file

        Returns:
            ScannedFile object with metadata
        """
        try:
            # Get file stats
            stat = file_path.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime)

            # Calculate file hash (for duplicate detection)
            file_hash = self._calculate_hash(file_path)

            # Extract metadata
            metadata = self._extract_metadata(file_path)

            scanned = ScannedFile(
                file_path=str(file_path.absolute()),
                file_hash=file_hash,
                file_size=stat.st_size,
                mtime=mtime,
                **metadata
            )

            return scanned

        except Exception as e:
            logger.error(f"Error scanning {file_path}: {e}")
            return ScannedFile(
                file_path=str(file_path.absolute()),
                file_hash="",
                file_size=0,
                mtime=datetime.now(),
                error=str(e)
            )

    def _calculate_hash(self, file_path: Path, chunk_size: int = 8192) -> str:
        """
        Calculate SHA-256 hash of file.

        Args:
            file_path: Path to file
            chunk_size: Size of chunks to read

        Returns:
            Hexadecimal hash string
        """
        sha256 = hashlib.sha256()

        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(chunk_size):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            logger.warning(f"Could not calculate hash for {file_path}: {e}")
            return ""

    def _extract_metadata(self, file_path: Path) -> Dict:
        """
        Extract metadata from audio file using mutagen.

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with metadata fields
        """
        metadata = {
            'title': None,
            'artist': None,
            'album': None,
            'album_artist': None,
            'year': None,
            'genre': None,
            'track_number': None,
            'duration': None,
            'format': file_path.suffix.lower()[1:],  # Remove leading dot
            'bitrate': None,
            'sample_rate': None
        }

        try:
            audio = MutagenFile(file_path)

            if audio is None:
                logger.warning(f"Could not read file: {file_path}")
                return metadata

            # Extract common metadata
            metadata['duration'] = getattr(audio.info, 'length', None)
            metadata['bitrate'] = getattr(audio.info, 'bitrate', None)
            metadata['sample_rate'] = getattr(audio.info, 'sample_rate', None)

            # Extract tags based on file type
            if isinstance(audio, MP3):
                metadata.update(self._extract_id3_tags(audio))
            elif isinstance(audio, FLAC):
                metadata.update(self._extract_vorbis_tags(audio))
            elif isinstance(audio, OggVorbis):
                metadata.update(self._extract_vorbis_tags(audio))
            elif isinstance(audio, MP4):
                metadata.update(self._extract_mp4_tags(audio))
            elif isinstance(audio, ASF):
                metadata.update(self._extract_asf_tags(audio))
            elif isinstance(audio, WAVE):
                # WAV files may have ID3 tags
                if hasattr(audio, 'tags') and audio.tags:
                    metadata.update(self._extract_id3_tags(audio))

        except Exception as e:
            logger.warning(f"Error extracting metadata from {file_path}: {e}")

        return metadata

    def _extract_id3_tags(self, audio) -> Dict:
        """Extract metadata from ID3 tags (MP3)."""
        tags = {}

        if not hasattr(audio, 'tags') or audio.tags is None:
            return tags

        try:
            tags['title'] = str(audio.tags.get('TIT2', [''])[0]) or None
            tags['artist'] = str(audio.tags.get('TPE1', [''])[0]) or None
            tags['album'] = str(audio.tags.get('TALB', [''])[0]) or None
            tags['album_artist'] = str(audio.tags.get('TPE2', [''])[0]) or None
            tags['genre'] = str(audio.tags.get('TCON', [''])[0]) or None

            # Year
            year_tag = audio.tags.get('TDRC') or audio.tags.get('TYER')
            if year_tag:
                year_str = str(year_tag[0])
                tags['year'] = int(year_str[:4]) if len(year_str) >= 4 else None

            # Track number
            track_tag = audio.tags.get('TRCK')
            if track_tag:
                track_str = str(track_tag[0]).split('/')[0]
                try:
                    tags['track_number'] = int(track_str)
                except ValueError:
                    pass

        except Exception as e:
            logger.debug(f"Error extracting ID3 tags: {e}")

        return {k: v for k, v in tags.items() if v is not None}

    def _extract_vorbis_tags(self, audio) -> Dict:
        """Extract metadata from Vorbis comments (FLAC, OGG)."""
        tags = {}

        if not hasattr(audio, 'tags') or audio.tags is None:
            return tags

        try:
            tags['title'] = audio.tags.get('title', [None])[0]
            tags['artist'] = audio.tags.get('artist', [None])[0]
            tags['album'] = audio.tags.get('album', [None])[0]
            tags['album_artist'] = audio.tags.get('albumartist', [None])[0]
            tags['genre'] = audio.tags.get('genre', [None])[0]

            # Year
            date_tag = audio.tags.get('date', [None])[0]
            if date_tag:
                tags['year'] = int(str(date_tag)[:4])

            # Track number
            track_tag = audio.tags.get('tracknumber', [None])[0]
            if track_tag:
                track_str = str(track_tag).split('/')[0]
                try:
                    tags['track_number'] = int(track_str)
                except ValueError:
                    pass

        except Exception as e:
            logger.debug(f"Error extracting Vorbis tags: {e}")

        return {k: v for k, v in tags.items() if v is not None}

    def _extract_mp4_tags(self, audio) -> Dict:
        """Extract metadata from MP4/M4A tags."""
        tags = {}

        if not hasattr(audio, 'tags') or audio.tags is None:
            return tags

        try:
            tags['title'] = audio.tags.get('\xa9nam', [None])[0]
            tags['artist'] = audio.tags.get('\xa9ART', [None])[0]
            tags['album'] = audio.tags.get('\xa9alb', [None])[0]
            tags['album_artist'] = audio.tags.get('aART', [None])[0]
            tags['genre'] = audio.tags.get('\xa9gen', [None])[0]

            # Year
            date_tag = audio.tags.get('\xa9day', [None])[0]
            if date_tag:
                tags['year'] = int(str(date_tag)[:4])

            # Track number
            track_tag = audio.tags.get('trkn', [None])[0]
            if track_tag:
                tags['track_number'] = track_tag[0]

        except Exception as e:
            logger.debug(f"Error extracting MP4 tags: {e}")

        return {k: v for k, v in tags.items() if v is not None}

    def _extract_asf_tags(self, audio) -> Dict:
        """Extract metadata from ASF/WMA tags."""
        tags = {}

        if not hasattr(audio, 'tags') or audio.tags is None:
            return tags

        try:
            tags['title'] = audio.tags.get('Title', [None])[0]
            tags['artist'] = audio.tags.get('Author', [None])[0]
            tags['album'] = audio.tags.get('WM/AlbumTitle', [None])[0]
            tags['album_artist'] = audio.tags.get('WM/AlbumArtist', [None])[0]
            tags['genre'] = audio.tags.get('WM/Genre', [None])[0]

            # Year
            year_tag = audio.tags.get('WM/Year', [None])[0]
            if year_tag:
                tags['year'] = int(str(year_tag))

            # Track number
            track_tag = audio.tags.get('WM/TrackNumber', [None])[0]
            if track_tag:
                try:
                    tags['track_number'] = int(str(track_tag))
                except ValueError:
                    pass

        except Exception as e:
            logger.debug(f"Error extracting ASF tags: {e}")

        return {k: v for k, v in tags.items() if v is not None}

    def validate_file(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate if a file is a readable audio file.

        Args:
            file_path: Path to file

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not file_path.exists():
            return False, "File does not exist"

        if not file_path.is_file():
            return False, "Not a regular file"

        if file_path.suffix.lower() not in self.audio_extensions:
            return False, f"Unsupported format: {file_path.suffix}"

        try:
            audio = MutagenFile(file_path)
            if audio is None:
                return False, "Could not read audio file"
            return True, None
        except Exception as e:
            return False, f"Error reading file: {str(e)}"