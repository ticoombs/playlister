"""
Playlist generation with smooth transitions.
"""

import random
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

import numpy as np
from loguru import logger

from src.storage.database import Database
from src.storage.schema import Song, SongFeatures, SongClassification, Playlist, PlaylistSong


class PlaylistGenerator:
    """Generate playlists with smooth transitions."""

    def __init__(self, db: Database, config: Any):
        """
        Initialize playlist generator.

        Args:
            db: Database instance
            config: Configuration object
        """
        self.db = db
        self.config = config

    def generate_by_mood(
        self,
        mood: str,
        count: int = 50,
        name: Optional[str] = None
    ) -> Optional[Playlist]:
        """
        Generate a playlist for a specific mood.

        Args:
            mood: Mood name
            count: Number of songs
            name: Optional playlist name

        Returns:
            Playlist object or None on failure
        """
        logger.info(f"Generating {mood} playlist with {count} songs")

        # Get songs with this mood classification
        with self.db.session_scope() as session:
            candidates = session.query(Song).join(SongFeatures).join(
                SongClassification
            ).filter(
                SongClassification.mood == mood
            ).order_by(
                SongClassification.confidence.desc()
            ).all()

            if not candidates:
                logger.error(f"No songs found with mood '{mood}'")
                return None

            if len(candidates) < count:
                logger.warning(f"Only {len(candidates)} songs available for mood '{mood}', generating shorter playlist")
                count = len(candidates)

            logger.info(f"Found {len(candidates)} candidate songs")

            # Generate playlist ordering
            ordered_songs = self._order_songs_for_flow(candidates, count)

            if not ordered_songs:
                logger.error("Failed to order songs")
                return None

            # Create playlist
            playlist_name = name or f"{mood.title()} Playlist - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

            playlist = Playlist(
                name=playlist_name,
                mood=mood,
                generation_method='mood_based',
                parameters={'mood': mood, 'count': count},
                song_count=len(ordered_songs),
                created_at=datetime.utcnow()
            )
            session.add(playlist)
            session.flush()  # Get playlist ID

            # Add songs to playlist
            total_duration = 0
            transition_scores = []

            for position, (song, transition_score) in enumerate(ordered_songs):
                playlist_song = PlaylistSong(
                    playlist_id=playlist.id,
                    song_id=song.id,
                    position=position,
                    transition_score=transition_score
                )
                session.add(playlist_song)

                if song.duration:
                    total_duration += song.duration

                if transition_score is not None:
                    transition_scores.append(transition_score)

            # Update playlist stats
            playlist.total_duration = total_duration
            playlist.avg_transition_score = sum(transition_scores) / len(transition_scores) if transition_scores else 0.0

            session.commit()
            session.refresh(playlist)

            logger.info(f"✓ Created playlist '{playlist_name}' with {len(ordered_songs)} songs")

            # Store essential attributes before session closes
            result = type('obj', (object,), {
                'id': playlist.id,
                'name': playlist.name,
                'mood': playlist.mood,
                'song_count': playlist.song_count,
                'total_duration': playlist.total_duration,
                'avg_transition_score': playlist.avg_transition_score
            })()

            return result

    def _order_songs_for_flow(
        self,
        songs: List[Song],
        count: int
    ) -> List[Tuple[Song, Optional[float]]]:
        """
        Order songs to create smooth flow.

        Args:
            songs: List of candidate songs
            count: Number of songs to select

        Returns:
            List of (song, transition_score) tuples in order
        """
        if not songs:
            return []

        # Get configuration
        bpm_tolerance = self.config.get('playlist', 'transitions', 'bpm_tolerance', default=10)
        energy_max_jump = self.config.get('playlist', 'transitions', 'energy_max_jump', default=0.2)
        avoid_same_artist = self.config.get('playlist', 'smart_shuffle', 'avoid_same_artist_within', default=5)
        avoid_same_album = self.config.get('playlist', 'smart_shuffle', 'avoid_same_album_within', default=10)

        # Start with a random song
        ordered = [random.choice(songs)]
        remaining = [s for s in songs if s.id != ordered[0].id]

        # Keep track of recently used artists/albums
        recent_artists = [ordered[0].artist] if ordered[0].artist else []
        recent_albums = [ordered[0].album] if ordered[0].album else []

        # Build playlist by selecting best transitions
        while len(ordered) < count and remaining:
            current_song = ordered[-1]
            best_song = None
            best_score = -1

            # Find best next song
            for candidate in remaining:
                # Check smart shuffle constraints
                if avoid_same_artist > 0 and candidate.artist:
                    if candidate.artist in recent_artists[-avoid_same_artist:]:
                        continue

                if avoid_same_album > 0 and candidate.album:
                    if candidate.album in recent_albums[-avoid_same_album:]:
                        continue

                # Calculate transition score
                score = self._calculate_transition_score(
                    current_song,
                    candidate,
                    bpm_tolerance,
                    energy_max_jump
                )

                if score > best_score:
                    best_score = score
                    best_song = candidate

            if best_song is None:
                # No candidates passed constraints, relax them
                best_song = random.choice(remaining)
                best_score = 0.5

            ordered.append(best_song)
            remaining = [s for s in remaining if s.id != best_song.id]

            # Update recent lists
            if best_song.artist:
                recent_artists.append(best_song.artist)
            if best_song.album:
                recent_albums.append(best_song.album)

        # Create result with transition scores
        result = [(ordered[0], None)]  # First song has no transition score

        for i in range(1, len(ordered)):
            prev_song = ordered[i - 1]
            curr_song = ordered[i]

            score = self._calculate_transition_score(
                prev_song,
                curr_song,
                bpm_tolerance,
                energy_max_jump
            )

            result.append((curr_song, score))

        return result

    def _calculate_transition_score(
        self,
        song1: Song,
        song2: Song,
        bpm_tolerance: float,
        energy_max_jump: float
    ) -> float:
        """
        Calculate how well two songs transition.

        Args:
            song1: Current song
            song2: Next song
            bpm_tolerance: BPM tolerance
            energy_max_jump: Maximum energy difference

        Returns:
            Transition score (0-1, higher is better)
        """
        if not song1.features or not song2.features:
            return 0.5  # Neutral score if features missing

        f1 = song1.features
        f2 = song2.features

        scores = []

        # BPM matching (weight: 2.0)
        if f1.tempo and f2.tempo:
            bpm_diff = abs(f1.tempo - f2.tempo)
            if bpm_diff <= bpm_tolerance:
                bpm_score = 1.0 - (bpm_diff / bpm_tolerance) * 0.3
            else:
                bpm_score = max(0.0, 1.0 - (bpm_diff / (bpm_tolerance * 3)))
            scores.append((bpm_score, 2.0))

        # Energy gradient (weight: 2.5)
        if f1.energy is not None and f2.energy is not None:
            energy_diff = abs(f1.energy - f2.energy)
            if energy_diff <= energy_max_jump:
                energy_score = 1.0 - (energy_diff / energy_max_jump) * 0.2
            else:
                energy_score = max(0.0, 1.0 - (energy_diff / energy_max_jump))
            scores.append((energy_score, 2.5))

        # Key compatibility (weight: 1.5)
        if f1.camelot_key and f2.camelot_key:
            key_score = self._calculate_key_compatibility(f1.camelot_key, f2.camelot_key)
            scores.append((key_score, 1.5))

        # Feature vector similarity (weight: 1.0)
        if f1.feature_vector and f2.feature_vector:
            similarity = self._cosine_similarity(f1.feature_vector, f2.feature_vector)
            # Convert similarity to score (0.5-1.0 range)
            vector_score = 0.5 + (similarity * 0.5)
            scores.append((vector_score, 1.0))

        if not scores:
            return 0.5

        # Calculate weighted average
        total_weight = sum(w for _, w in scores)
        weighted_sum = sum(s * w for s, w in scores)

        return weighted_sum / total_weight

    def _calculate_key_compatibility(self, key1: str, key2: str) -> float:
        """
        Calculate key compatibility using Camelot wheel.

        Args:
            key1: Camelot key (e.g., "8A")
            key2: Camelot key (e.g., "9A")

        Returns:
            Compatibility score (0-1)
        """
        if not key1 or not key2:
            return 0.5

        try:
            # Extract number and letter
            num1 = int(key1[:-1])
            letter1 = key1[-1]
            num2 = int(key2[:-1])
            letter2 = key2[-1]

            # Perfect match
            if key1 == key2:
                return 1.0

            # Same number, different mode (relative major/minor)
            if num1 == num2:
                return 0.9

            # Adjacent numbers, same mode
            if letter1 == letter2:
                diff = min(abs(num1 - num2), 12 - abs(num1 - num2))
                if diff == 1:
                    return 0.85
                elif diff == 2:
                    return 0.6

            # Not compatible
            return 0.4

        except (ValueError, IndexError):
            return 0.5

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two feature vectors.

        Args:
            vec1: First feature vector
            vec2: Second feature vector

        Returns:
            Similarity score (0-1)
        """
        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)

            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)

            # Clamp to [0, 1]
            return max(0.0, min(1.0, similarity))

        except Exception:
            return 0.5