"""
Music library scanner for file discovery and metadata extraction.
"""

import hashlib
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Generator, Tuple
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count

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

    # Supported audio formats (all formats mutagen can handle)
    AUDIO_EXTENSIONS = {
        # Lossless
        '.flac', '.wav', '.aiff', '.aif', '.ape', '.wv', '.tta', '.tak',
        # Lossy
        '.mp3', '.ogg', '.opus', '.m4a', '.mp4', '.mpc', '.wma', '.asf', '.spx'
    }

    def __init__(
        self,
        audio_formats: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        follow_symlinks: bool = False,
        min_file_size: int = 0,
        workers: int = 4,
        database_session=None
    ):
        """
        Initialize scanner.

        Args:
            audio_formats: List of audio format extensions (e.g., ['.mp3', '.flac'])
                          If None, uses default AUDIO_EXTENSIONS
            workers: Number of parallel workers for scanning (default: 4)
            exclude_patterns: List of directory name patterns to exclude from scanning
            follow_symlinks: Whether to follow symbolic links during scanning
            min_file_size: Minimum file size in bytes (files smaller are skipped)
            database_session: Database session for checkpoint operations
        """
        if audio_formats:
            self.audio_extensions = {ext.lower() for ext in audio_formats}
        else:
            self.audio_extensions = self.AUDIO_EXTENSIONS

        self.exclude_patterns = set(exclude_patterns) if exclude_patterns else set()
        self.follow_symlinks = follow_symlinks
        self.min_file_size = min_file_size
        self.workers = max(1, min(workers, cpu_count()))
        self.database_session = database_session

    def scan_directory_with_resumption(
        self,
        root_path: str,
        recursive: bool = True,
        show_progress: bool = True,
        resume_scan_id: Optional[str] = None
    ) -> Generator[Tuple[ScannedFile, str], None, None]:
        """
        Scan directory with checkpoint-based resumption capability.

        Args:
            root_path: Root directory to scan
            recursive: Whether to scan subdirectories recursively
            show_progress: Whether to show progress bar
            resume_scan_id: Scan ID to resume from (if None, starts new scan)

        Yields:
            Tuple of (ScannedFile, scan_id) for each discovered audio file
        """
        if not self.database_session:
            logger.error("Database session required for resumption-capable scanning")
            return

        root = Path(root_path).resolve()
        if not root.exists() or not root.is_dir():
            logger.error(f"Invalid path: {root_path}")
            return

        # Initialize or resume scan progress
        scan_progress = self._initialize_scan_progress(str(root), resume_scan_id)
        scan_id = scan_progress.scan_id

        try:
            # Get completed directories for skipping
            completed_dirs = set(scan_progress.directories_completed or [])
            current_dir = scan_progress.current_directory
            
            logger.info(f"{'Resuming' if resume_scan_id else 'Starting'} scan {scan_id}")
            if completed_dirs:
                logger.info(f"Skipping {len(completed_dirs)} completed directories")

            # Directory-level scanning with checkpointing
            for scanned_file in self._scan_with_checkpoints(
                root, recursive, scan_progress, completed_dirs, current_dir, show_progress
            ):
                yield scanned_file, scan_id

            # Mark scan as completed
            self._update_scan_status(scan_id, 'completed')
            logger.info(f"✓ Scan {scan_id} completed successfully")

        except Exception as e:
            self._update_scan_status(scan_id, 'failed')
            logger.error(f"Scan {scan_id} failed: {e}")
            raise

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
            logger.info(f"Scanning with {self.workers} parallel workers")

            # Process files with progress bar
            pool = Pool(processes=self.workers)
            try:
                iterator = pool.imap_unordered(self.scan_file, audio_files, chunksize=10)
                
                pbar = tqdm(
                    total=len(audio_files),
                    desc="Scanning files",
                    disable=not show_progress,
                    dynamic_ncols=False,
                    ncols=100
                )
                
                for scanned in iterator:
                    yield scanned
                    pbar.update(1)
                
                pbar.close()
                
            finally:
                pool.close()
                pool.join()

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
        """Extract metadata from Vorbis comments (FLAC, OGG, Opus)."""
        tags = {}

        if not hasattr(audio, 'tags') or audio.tags is None:
            return tags

        try:
            # Vorbis tags return strings, but convert to be safe
            title = audio.tags.get('title', [None])[0]
            tags['title'] = str(title) if title else None

            artist = audio.tags.get('artist', [None])[0]
            tags['artist'] = str(artist) if artist else None

            album = audio.tags.get('album', [None])[0]
            tags['album'] = str(album) if album else None

            album_artist = audio.tags.get('albumartist', [None])[0]
            tags['album_artist'] = str(album_artist) if album_artist else None

            genre = audio.tags.get('genre', [None])[0]
            tags['genre'] = str(genre) if genre else None

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
            # MP4 tags can return various types, convert to str
            title = audio.tags.get('\xa9nam', [None])[0]
            tags['title'] = str(title) if title else None

            artist = audio.tags.get('\xa9ART', [None])[0]
            tags['artist'] = str(artist) if artist else None

            album = audio.tags.get('\xa9alb', [None])[0]
            tags['album'] = str(album) if album else None

            album_artist = audio.tags.get('aART', [None])[0]
            tags['album_artist'] = str(album_artist) if album_artist else None

            genre = audio.tags.get('\xa9gen', [None])[0]
            tags['genre'] = str(genre) if genre else None

            # Year
            date_tag = audio.tags.get('\xa9day', [None])[0]
            if date_tag:
                tags['year'] = int(str(date_tag)[:4])

            # Track number (trkn returns tuple like (1, 12))
            track_tag = audio.tags.get('trkn', [None])[0]
            if track_tag:
                tags['track_number'] = int(track_tag[0]) if isinstance(track_tag, tuple) else int(track_tag)

        except Exception as e:
            logger.debug(f"Error extracting MP4 tags: {e}")

        return {k: v for k, v in tags.items() if v is not None}

    def _extract_asf_tags(self, audio) -> Dict:
        """Extract metadata from ASF/WMA tags."""
        tags = {}

        if not hasattr(audio, 'tags') or audio.tags is None:
            return tags

        try:
            # ASF tags return ASFUnicodeAttribute objects, extract .value attribute
            title = audio.tags.get('Title', [None])[0]
            tags['title'] = title.value if title and hasattr(title, 'value') else (str(title) if title else None)

            artist = audio.tags.get('Author', [None])[0]
            tags['artist'] = artist.value if artist and hasattr(artist, 'value') else (str(artist) if artist else None)

            album = audio.tags.get('WM/AlbumTitle', [None])[0]
            tags['album'] = album.value if album and hasattr(album, 'value') else (str(album) if album else None)

            album_artist = audio.tags.get('WM/AlbumArtist', [None])[0]
            tags['album_artist'] = album_artist.value if album_artist and hasattr(album_artist, 'value') else (str(album_artist) if album_artist else None)

            genre = audio.tags.get('WM/Genre', [None])[0]
            tags['genre'] = genre.value if genre and hasattr(genre, 'value') else (str(genre) if genre else None)

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

    def _initialize_scan_progress(self, root_path: str, resume_scan_id: Optional[str] = None):
        """Initialize or resume scan progress."""
        from src.storage.schema import ScanProgress
        
        if resume_scan_id:
            # Resume existing scan
            progress = self.database_session.query(ScanProgress).filter_by(
                scan_id=resume_scan_id, status='running'
            ).first()
            
            if not progress:
                logger.warning(f"Cannot resume scan {resume_scan_id}, starting new scan")
                resume_scan_id = None
            else:
                logger.info(f"Resuming scan from {len(progress.directories_completed or [])} completed directories")
        
        if not resume_scan_id:
            # Start new scan
            progress = ScanProgress(
                scan_id=str(uuid.uuid4()),
                root_path=root_path,
                directories_completed=[],
                status='running'
            )
            self.database_session.add(progress)
            self.database_session.commit()
        
        return progress

    def _scan_with_checkpoints(
        self, 
        root: Path, 
        recursive: bool, 
        scan_progress, 
        completed_dirs: set, 
        resume_from_dir: Optional[str],
        show_progress: bool
    ) -> Generator[ScannedFile, None, None]:
        """Scan directories with checkpoint management."""
        
        directories_to_scan = []
        
        # Collect directories to scan
        if recursive:
            for dirpath, dirnames, filenames in os.walk(root, followlinks=self.follow_symlinks):
                # Filter out excluded directories
                if self.exclude_patterns:
                    dirnames[:] = [d for d in dirnames if d not in self.exclude_patterns]
                
                # Skip completed directories
                if dirpath in completed_dirs:
                    continue
                
                # If resuming, skip directories until we reach the resume point
                if resume_from_dir and dirpath < resume_from_dir:
                    continue
                
                # Only process directories with audio files
                audio_files = [f for f in filenames if self._is_audio_file(Path(dirpath) / f)]
                if audio_files:
                    directories_to_scan.append((dirpath, audio_files))
        else:
            # Non-recursive: just process root directory
            if str(root) not in completed_dirs:
                audio_files = [f.name for f in root.iterdir() if self._is_audio_file(f)]
                if audio_files:
                    directories_to_scan.append((str(root), audio_files))

        if not directories_to_scan:
            logger.info("No directories to scan")
            return

        logger.info(f"Processing {len(directories_to_scan)} directories")
        
        # Process directories with progress tracking
        pbar = tqdm(
            directories_to_scan,
            desc="Scanning directories",
            disable=not show_progress,
            unit="dir"
        )
        
        for dirpath, audio_files in pbar:
            pbar.set_postfix({"dir": os.path.basename(dirpath)})
            
            # Update current directory in progress
            self._update_current_directory(scan_progress.scan_id, dirpath)
            
            # Process all files in this directory
            for filename in audio_files:
                filepath = Path(dirpath) / filename
                scanned = self.scan_file(filepath)
                yield scanned
                
                # Update file counters
                if scanned.error:
                    scan_progress.files_errored += 1
                else:
                    scan_progress.files_processed += 1
            
            # Mark directory as completed
            self._mark_directory_completed(scan_progress.scan_id, dirpath)
            completed_dirs.add(dirpath)
            
            # Commit progress periodically
            self.database_session.commit()

    def _is_audio_file(self, file_path: Path) -> bool:
        """Check if file is a supported audio file."""
        if not file_path.is_file():
            return False
        
        if file_path.suffix.lower() not in self.audio_extensions:
            return False
        
        if self.min_file_size > 0:
            try:
                if file_path.stat().st_size < self.min_file_size:
                    return False
            except OSError:
                return False
        
        return True

    def _update_current_directory(self, scan_id: str, directory: str):
        """Update the current directory being processed."""
        from src.storage.schema import ScanProgress
        
        self.database_session.query(ScanProgress).filter_by(scan_id=scan_id).update({
            'current_directory': directory,
            'updated_at': datetime.utcnow()
        })

    def _mark_directory_completed(self, scan_id: str, directory: str):
        """Mark a directory as completed."""
        from src.storage.schema import ScanProgress
        
        progress = self.database_session.query(ScanProgress).filter_by(scan_id=scan_id).first()
        if progress:
            completed = progress.directories_completed or []
            if directory not in completed:
                completed.append(directory)
                progress.directories_completed = completed
                progress.updated_at = datetime.utcnow()

    def _update_scan_status(self, scan_id: str, status: str):
        """Update scan status."""
        from src.storage.schema import ScanProgress
        
        self.database_session.query(ScanProgress).filter_by(scan_id=scan_id).update({
            'status': status,
            'updated_at': datetime.utcnow()
        })
        self.database_session.commit()

    def get_interrupted_scans(self) -> List[Dict]:
        """Get list of interrupted scans that can be resumed."""
        if not self.database_session:
            return []
        
        from src.storage.schema import ScanProgress
        
        interrupted = self.database_session.query(ScanProgress).filter_by(status='running').all()
        return [
            {
                'scan_id': scan.scan_id,
                'root_path': scan.root_path,
                'started_at': scan.started_at,
                'files_processed': scan.files_processed,
                'directories_completed': len(scan.directories_completed or [])
            }
            for scan in interrupted
        ]

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