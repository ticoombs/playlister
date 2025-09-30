"""
Mood classification for songs based on audio features.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime

from loguru import logger
from tqdm import tqdm

from src.storage.database import Database
from src.storage.schema import Song, SongFeatures, SongClassification


class MoodClassifier:
    """
    Classify songs by mood based on audio features.

    The mood system is fully extensible. You can define any mood in config.yaml
    by specifying feature ranges (tempo, energy, valence, etc.). The moods below
    are just defaults used when no config file is provided.

    Custom moods can be added without modifying code - just edit config/config.yaml.
    """

    # Default moods - used only as fallback when no config provided
    # Feel free to customize these in config.yaml instead of editing code
    DEFAULT_MOOD_RULES = {
        'uplifting': {
            'valence': (0.65, 1.0),
            'energy': (0.55, 0.85),
            'tempo': (110, 170)
        },
        'rock_out': {
            'energy': (0.75, 1.0),
            'valence': (0.35, 0.75),
            'tempo': (130, 180),
            'loudness': (0.65, 1.0)
        },
        'dance': {
            'danceability': (0.65, 1.0),
            'energy': (0.65, 1.0),
            'tempo': (115, 135)
        },
        'chill': {
            'energy': (0.25, 0.45),
            'valence': (0.45, 0.75),
            'tempo': (85, 115)
        },
        'relax': {
            'energy': (0.0, 0.25),
            'valence': (0.35, 0.65),
            'tempo': (65, 95),
            'acousticness': (0.5, 1.0)
        }
    }

    DEFAULT_CONFIDENCE_THRESHOLD = 0.85

    def __init__(self, db: Database, mood_config: Optional[Dict] = None,
                 confidence_threshold: float = None, strict_matching: bool = True):
        """
        Initialize mood classifier.

        Args:
            db: Database instance
            mood_config: Optional mood configuration (uses defaults if not provided)
                        Can contain ANY mood definitions - not limited to the 5 defaults.
                        Load from config.yaml to add custom moods.
            confidence_threshold: Minimum confidence for classification (default 0.85)
            strict_matching: If True, require all features to be in range (default True)
        """
        self.db = db
        # mood_rules can contain ANY moods defined in config - fully extensible
        self.mood_rules = mood_config if mood_config else self.DEFAULT_MOOD_RULES
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else self.DEFAULT_CONFIDENCE_THRESHOLD
        self.strict_matching = strict_matching

    def classify_all(self) -> int:
        """
        Classify all songs that have features but no classification.

        Returns:
            Number of songs classified
        """
        # Get songs with features but no classification
        with self.db.session_scope() as session:
            songs_with_features = session.query(Song).join(SongFeatures).outerjoin(
                SongClassification
            ).filter(SongClassification.id == None).all()

            if not songs_with_features:
                logger.info("No songs need classification")
                return 0

            logger.info(f"Classifying {len(songs_with_features)} songs")

            classified_count = 0

            for song in tqdm(songs_with_features, desc="Classifying songs"):
                moods = self.classify_song(song)

                if moods:
                    for mood, confidence in moods:
                        classification = SongClassification(
                            song_id=song.id,
                            mood=mood,
                            confidence=confidence,
                            method='rule_based',
                            manual_override=False,
                            classified_at=datetime.utcnow()
                        )
                        session.add(classification)

                    classified_count += 1

                    # Commit in batches
                    if classified_count % 100 == 0:
                        session.commit()

            session.commit()

        return classified_count

    def classify_song(self, song: Song) -> List[Tuple[str, float]]:
        """
        Classify a single song into one or more moods.

        Args:
            song: Song object with features loaded

        Returns:
            List of (mood, confidence) tuples
        """
        if not song.features:
            logger.warning(f"Song {song.id} has no features")
            return []

        features = song.features
        moods = []

        # Check each mood
        for mood_name, rules in self.mood_rules.items():
            confidence = self._calculate_mood_match(features, rules)

            # Only classify if confidence meets threshold
            if confidence >= self.confidence_threshold:
                moods.append((mood_name, confidence))

        # Sort by confidence
        moods.sort(key=lambda x: x[1], reverse=True)

        # No fallback - if song doesn't match any mood strongly enough, don't classify it
        return moods

    def _calculate_mood_match(self, features: SongFeatures, rules: Dict) -> float:
        """
        Calculate how well a song's features match mood rules.

        Args:
            features: SongFeatures object
            rules: Dictionary of feature ranges

        Returns:
            Confidence score (0-1)
        """
        matches = []
        weights = []
        features_in_range = 0
        total_features = 0

        for feature_name, (min_val, max_val) in rules.items():
            feature_value = self._get_feature_value(features, feature_name)

            if feature_value is None:
                # Feature not available, skip
                continue

            total_features += 1

            # Calculate match score
            if min_val <= feature_value <= max_val:
                # Feature is in range
                features_in_range += 1

                # Calculate how centered it is in the range
                range_size = max_val - min_val
                center = (min_val + max_val) / 2
                distance_from_center = abs(feature_value - center)
                score = 1.0 - (distance_from_center / (range_size / 2)) * 0.2
                matches.append(score)
            else:
                # Out of range - strict mode gives zero score
                if self.strict_matching:
                    matches.append(0.0)
                else:
                    # Lenient mode - small penalty for being close
                    if feature_value < min_val:
                        distance = min_val - feature_value
                    else:
                        distance = feature_value - max_val

                    # Only give partial credit if very close (within 20% of range)
                    range_size = max_val - min_val
                    tolerance = range_size * 0.2

                    if distance <= tolerance:
                        score = max(0.0, 1.0 - (distance / tolerance))
                    else:
                        score = 0.0

                    matches.append(score)

            # Weight different features
            weight = self._get_feature_weight(feature_name)
            weights.append(weight)

        if not matches:
            return 0.0

        # In strict matching mode, require ALL features to be in range
        if self.strict_matching and total_features > 0:
            if features_in_range < total_features:
                return 0.0

        # Calculate weighted average
        total_weight = sum(weights)
        weighted_sum = sum(m * w for m, w in zip(matches, weights))

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _get_feature_value(self, features: SongFeatures, feature_name: str) -> Optional[float]:
        """Get feature value by name."""
        feature_map = {
            'tempo': features.tempo,
            'energy': features.energy,
            'valence': features.valence,
            'danceability': features.danceability,
            'acousticness': features.acousticness,
            'instrumentalness': features.instrumentalness,
            'loudness': features.loudness,
        }

        return feature_map.get(feature_name)

    def _get_feature_weight(self, feature_name: str) -> float:
        """
        Get importance weight for a feature.

        Different features have different importance for mood classification.
        """
        weights = {
            'energy': 1.5,      # Very important
            'valence': 1.5,     # Very important
            'tempo': 1.2,       # Important
            'danceability': 1.0,
            'acousticness': 0.8,
            'instrumentalness': 0.6,
            'loudness': 0.8
        }

        return weights.get(feature_name, 1.0)

    def reclassify_song(self, song_id: int, mood: str, confidence: float = 1.0):
        """
        Manually reclassify a song.

        Args:
            song_id: Song ID
            mood: Mood name
            confidence: Confidence score (default 1.0 for manual)
        """
        with self.db.session_scope() as session:
            # Remove existing classifications
            session.query(SongClassification).filter_by(song_id=song_id).delete()

            # Add new classification
            classification = SongClassification(
                song_id=song_id,
                mood=mood,
                confidence=confidence,
                method='manual',
                manual_override=True,
                classified_at=datetime.utcnow()
            )
            session.add(classification)

        logger.info(f"Song {song_id} reclassified as '{mood}'")

    def get_song_moods(self, song_id: int) -> List[Tuple[str, float]]:
        """
        Get all mood classifications for a song.

        Args:
            song_id: Song ID

        Returns:
            List of (mood, confidence) tuples
        """
        with self.db.session_scope() as session:
            classifications = session.query(SongClassification).filter_by(
                song_id=song_id
            ).order_by(SongClassification.confidence.desc()).all()

            return [(c.mood, c.confidence) for c in classifications]