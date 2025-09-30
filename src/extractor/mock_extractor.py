"""
Mock feature extractor for testing without Essentia.
Generates plausible random features based on filename heuristics.
"""

import random
import hashlib
from typing import Dict, Any
from pathlib import Path

from loguru import logger


class MockFeatureExtractor:
    """Mock feature extractor for testing."""

    def extract_features(self, file_path: str) -> Dict[str, Any]:
        """
        Generate mock features based on filename heuristics.

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary of mock features
        """
        # Use filename as seed for consistent results
        filename = Path(file_path).stem.lower()
        seed = int(hashlib.md5(filename.encode()).hexdigest()[:8], 16)
        random.seed(seed)

        # Guess characteristics from filename
        is_chill = any(word in filename for word in ['chill', 'relax', 'calm', 'slow', 'lofi', 'moonlight', 'evening'])
        is_energetic = any(word in filename for word in ['power', 'hard', 'energetic', 'trap', 'rock'])
        is_uplifting = any(word in filename for word in ['uplifting', 'inspiring', 'optimistic', 'happy', 'summer'])
        is_dance = any(word in filename for word in ['dance', 'edm', 'house', 'electronic'])

        # Generate features based on heuristics
        if is_chill:
            tempo = random.uniform(70, 100)
            energy = random.uniform(0.1, 0.4)
            valence = random.uniform(0.4, 0.7)
            danceability = random.uniform(0.3, 0.6)
            acousticness = random.uniform(0.5, 0.9)
        elif is_energetic:
            tempo = random.uniform(130, 160)
            energy = random.uniform(0.7, 0.95)
            valence = random.uniform(0.4, 0.7)
            danceability = random.uniform(0.5, 0.8)
            acousticness = random.uniform(0.1, 0.4)
        elif is_uplifting:
            tempo = random.uniform(110, 140)
            energy = random.uniform(0.6, 0.85)
            valence = random.uniform(0.7, 0.95)
            danceability = random.uniform(0.6, 0.9)
            acousticness = random.uniform(0.3, 0.7)
        elif is_dance:
            tempo = random.uniform(115, 135)
            energy = random.uniform(0.65, 0.9)
            valence = random.uniform(0.5, 0.8)
            danceability = random.uniform(0.7, 0.95)
            acousticness = random.uniform(0.1, 0.3)
        else:
            # Generic/neutral
            tempo = random.uniform(90, 130)
            energy = random.uniform(0.4, 0.7)
            valence = random.uniform(0.4, 0.7)
            danceability = random.uniform(0.4, 0.7)
            acousticness = random.uniform(0.3, 0.7)

        # Generate key and mode
        key = random.randint(0, 11)
        mode = random.randint(0, 1)

        # Camelot wheel mapping
        CAMELOT_WHEEL = {
            (0, 0): '5A', (0, 1): '8B',
            (1, 0): '12A', (1, 1): '3B',
            (2, 0): '7A', (2, 1): '10B',
            (3, 0): '2A', (3, 1): '5B',
            (4, 0): '9A', (4, 1): '12B',
            (5, 0): '4A', (5, 1): '7B',
            (6, 0): '11A', (6, 1): '2B',
            (7, 0): '6A', (7, 1): '9B',
            (8, 0): '1A', (8, 1): '4B',
            (9, 0): '8A', (9, 1): '11B',
            (10, 0): '3A', (10, 1): '6B',
            (11, 0): '10A', (11, 1): '1B'
        }

        camelot_key = CAMELOT_WHEEL.get((key, mode), '1A')

        # Generate other features
        loudness = random.uniform(60, 95)
        instrumentalness = random.uniform(0.3, 0.8)
        spectral_centroid = random.uniform(1000, 4000)

        features = {
            'tempo': tempo,
            'energy': energy,
            'valence': valence,
            'danceability': danceability,
            'acousticness': acousticness,
            'instrumentalness': instrumentalness,
            'loudness': loudness,
            'key': key,
            'mode': mode,
            'camelot_key': camelot_key,
            'spectral_features': {
                'centroid_mean': spectral_centroid
            },
            'feature_vector': [
                tempo / 200.0,
                energy,
                valence,
                danceability,
                acousticness,
                key / 11.0,
                float(mode),
                loudness / 100.0,
                spectral_centroid / 5000.0
            ]
        }

        logger.debug(f"Mock features for {Path(file_path).name}: tempo={tempo:.1f}, energy={energy:.2f}, valence={valence:.2f}")

        return features