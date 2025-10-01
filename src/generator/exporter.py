"""
Playlist export to various formats.
"""

import json
import os
from pathlib import Path
from typing import Optional

from loguru import logger

from src.storage.database import Database
from src.storage.schema import Playlist, PlaylistSong


class PlaylistExporter:
    """Export playlists to various formats."""

    def __init__(self, db: Database, output_dir: str, music_base_path: Optional[str] = None, path_prefix: Optional[str] = None):
        """
        Initialize playlist exporter.

        Args:
            db: Database instance
            output_dir: Output directory for playlists
            music_base_path: Base path for music library (for path substitution)
            path_prefix: Optional prefix to override in playlist paths (e.g., "/music" for Gonic)
        """
        self.db = db
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.music_base_path = music_base_path
        self.path_prefix = path_prefix

    def export(self, playlist_id: int, format: str = 'm3u8') -> Optional[str]:
        """
        Export playlist to file.

        Args:
            playlist_id: Playlist ID
            format: Export format ('m3u', 'm3u8', 'pls', 'json')

        Returns:
            Path to exported file or None on failure
        """
        # Get playlist
        with self.db.session_scope() as session:
            playlist = session.query(Playlist).filter_by(id=playlist_id).first()

            if not playlist:
                logger.error(f"Playlist {playlist_id} not found")
                return None

            # Get songs in order
            playlist_songs = session.query(PlaylistSong).filter_by(
                playlist_id=playlist_id
            ).order_by(PlaylistSong.position).all()

            if not playlist_songs:
                logger.error(f"Playlist {playlist_id} has no songs")
                return None

            # Export based on format
            if format == 'm3u':
                output_path = self._export_m3u(playlist, playlist_songs, extended=False)
            elif format == 'm3u8':
                output_path = self._export_m3u(playlist, playlist_songs, extended=True)
            elif format == 'pls':
                output_path = self._export_pls(playlist, playlist_songs)
            elif format == 'json':
                output_path = self._export_json(playlist, playlist_songs)
            else:
                logger.error(f"Unknown format: {format}")
                return None

            if output_path:
                logger.info(f"Playlist exported to {output_path}")
                # Update exported timestamp
                from datetime import datetime
                playlist.exported_at = datetime.utcnow()
                session.commit()

            return str(output_path) if output_path else None

    def _convert_path(self, file_path: str) -> str:
        """
        Convert file path for playlist output.
        Makes paths relative to the playlist directory.

        Handles multiple scenarios:
        1. Playlists outside music dir: ../music/Artist/Song.mp3
        2. Playlists inside music dir: ../Artist/Song.mp3
        3. Docker container paths: converts /app/music/... to relative paths
        4. Path prefix override: /music/Artist/Song.mp3 (for Gonic compatibility)

        Args:
            file_path: Original file path from database

        Returns:
            Converted path suitable for playlist
        """
        path = Path(file_path)

        # If path_prefix is set, use it to override the default relative path logic
        if self.path_prefix:
            # Extract the relative path from music directory
            if self.music_base_path:
                try:
                    music_base = Path(self.music_base_path)
                    # Try to get relative path from music base
                    if path.is_absolute() and music_base.is_absolute():
                        try:
                            rel_path = path.relative_to(music_base)
                            return str(Path(self.path_prefix) / rel_path)
                        except ValueError:
                            pass

                    # Try finding 'music' in path hierarchy
                    parts = path.parts
                    if 'music' in parts:
                        music_idx = parts.index('music')
                        if len(parts) > music_idx + 1:
                            rel_path = Path(*parts[music_idx+1:])
                            return str(Path(self.path_prefix) / rel_path)
                except Exception as e:
                    logger.debug(f"Path prefix conversion failed: {e}")

            # Fallback: just use filename with prefix
            return str(Path(self.path_prefix) / path.name)

        # If music_base_path is specified, try to make relative to it
        if self.music_base_path:
            try:
                music_base = Path(self.music_base_path)
                playlist_dir = Path(self.output_dir)

                # First, extract the path relative to the music directory
                # This handles Docker paths like /app/music/Artist/Song.mp3
                rel_from_music = None

                # Try direct relative_to if paths share a common base
                try:
                    if path.is_absolute() and music_base.is_absolute():
                        # Both are absolute, try to make relative
                        rel_from_music = path.relative_to(music_base)
                except ValueError:
                    # Paths don't share common base, try pattern matching
                    pass

                # If that didn't work, try finding 'music' in the path hierarchy
                if rel_from_music is None:
                    parts = path.parts
                    if 'music' in parts:
                        music_idx = parts.index('music')
                        # Get everything after the 'music' directory
                        if len(parts) > music_idx + 1:
                            rel_from_music = Path(*parts[music_idx+1:])

                # If we still couldn't extract it, just use the filename
                if rel_from_music is None:
                    rel_from_music = Path(path.name)

                # Now calculate the path from playlist directory to music directory
                # Check if playlist directory is inside music directory
                try:
                    if playlist_dir.is_absolute() and music_base.is_absolute():
                        playlist_dir.relative_to(music_base)
                        # Playlist is inside music directory
                        # Path is relative going up from playlist to music, then into the file location
                        playlist_to_music = os.path.relpath(music_base, playlist_dir)
                    else:
                        # Can't determine relationship, check if both use 'music' in path
                        playlist_parts = playlist_dir.parts
                        if 'music' in playlist_parts:
                            # Both are under music, calculate relative path
                            playlist_to_music = os.path.relpath(music_base, playlist_dir)
                        else:
                            # Playlist is outside music directory
                            playlist_to_music = os.path.relpath(music_base, playlist_dir)
                except (ValueError, OSError):
                    # Default to assuming playlist is outside music dir
                    playlist_to_music = os.path.relpath(music_base, playlist_dir)

                # Combine the paths
                final_path = Path(playlist_to_music) / rel_from_music
                return str(final_path)

            except (ValueError, OSError, Exception) as e:
                logger.debug(f"Path conversion failed for {file_path}: {e}, using fallback")

        # Fallback: try to make relative to output directory
        try:
            return str(path.relative_to(self.output_dir))
        except ValueError:
            # Can't make relative, return absolute path
            return str(path.absolute())

    def _export_m3u(
        self,
        playlist: Playlist,
        playlist_songs: list,
        extended: bool = True
    ) -> Optional[Path]:
        """Export to M3U/M3U8 format."""
        ext = 'm3u8' if extended else 'm3u'
        filename = self._sanitize_filename(playlist.name) + f'.{ext}'
        output_path = self.output_dir / filename

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                if extended:
                    f.write('#EXTM3U\n')

                for ps in playlist_songs:
                    song = ps.song

                    if extended:
                        # Write extended info
                        duration = int(song.duration) if song.duration else -1
                        artist = song.artist or 'Unknown'
                        title = song.title or Path(song.file_path).stem
                        f.write(f'#EXTINF:{duration},{artist} - {title}\n')

                    # Write file path (converted to relative)
                    converted_path = self._convert_path(song.file_path)
                    f.write(f'{converted_path}\n')

            return output_path

        except Exception as e:
            logger.error(f"Error exporting M3U: {e}")
            return None

    def _export_pls(self, playlist: Playlist, playlist_songs: list) -> Optional[Path]:
        """Export to PLS format."""
        filename = self._sanitize_filename(playlist.name) + '.pls'
        output_path = self.output_dir / filename

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('[playlist]\n')
                f.write(f'NumberOfEntries={len(playlist_songs)}\n\n')

                for idx, ps in enumerate(playlist_songs, start=1):
                    song = ps.song

                    # Write file path (converted to relative)
                    converted_path = self._convert_path(song.file_path)
                    f.write(f'File{idx}={converted_path}\n')

                    if song.title:
                        artist = song.artist or 'Unknown'
                        f.write(f'Title{idx}={artist} - {song.title}\n')

                    if song.duration:
                        f.write(f'Length{idx}={int(song.duration)}\n')

                    f.write('\n')

                f.write('Version=2\n')

            return output_path

        except Exception as e:
            logger.error(f"Error exporting PLS: {e}")
            return None

    def _export_json(self, playlist: Playlist, playlist_songs: list) -> Optional[Path]:
        """Export to JSON format with full metadata."""
        filename = self._sanitize_filename(playlist.name) + '.json'
        output_path = self.output_dir / filename

        try:
            data = {
                'playlist': {
                    'id': playlist.id,
                    'name': playlist.name,
                    'description': playlist.description,
                    'mood': playlist.mood,
                    'song_count': playlist.song_count,
                    'total_duration': playlist.total_duration,
                    'avg_transition_score': playlist.avg_transition_score,
                    'created_at': playlist.created_at.isoformat() if playlist.created_at else None,
                    'generation_method': playlist.generation_method,
                    'parameters': playlist.parameters
                },
                'songs': []
            }

            for ps in playlist_songs:
                song = ps.song

                song_data = {
                    'position': ps.position,
                    'transition_score': ps.transition_score,
                    'file_path': song.file_path,
                    'title': song.title,
                    'artist': song.artist,
                    'album': song.album,
                    'year': song.year,
                    'genre': song.genre,
                    'duration': song.duration,
                    'format': song.format
                }

                # Add features if available
                if song.features:
                    song_data['features'] = {
                        'tempo': song.features.tempo,
                        'energy': song.features.energy,
                        'valence': song.features.valence,
                        'danceability': song.features.danceability,
                        'key': song.features.camelot_key
                    }

                data['songs'].append(song_data)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return output_path

        except Exception as e:
            logger.error(f"Error exporting JSON: {e}")
            return None

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize playlist name for use as filename."""
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, '_')

        # Limit length
        if len(name) > 200:
            name = name[:200]

        return name.strip()