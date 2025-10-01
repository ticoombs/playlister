"""
Feature extraction using Essentia.
"""

import json
from datetime import datetime
from typing import Optional, Dict, Any
from multiprocessing import Pool, cpu_count
from pathlib import Path

from loguru import logger
from tqdm import tqdm

try:
    import essentia
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False
    logger.warning("Essentia not available, feature extraction will be limited")

from src.storage.database import Database
from src.storage.schema import Song, SongFeatures


# Camelot wheel mapping
CAMELOT_WHEEL = {
    (0, 0): '5A', (0, 1): '8B',   # C
    (1, 0): '12A', (1, 1): '3B',  # C#/Db
    (2, 0): '7A', (2, 1): '10B',  # D
    (3, 0): '2A', (3, 1): '5B',   # D#/Eb
    (4, 0): '9A', (4, 1): '12B',  # E
    (5, 0): '4A', (5, 1): '7B',   # F
    (6, 0): '11A', (6, 1): '2B',  # F#/Gb
    (7, 0): '6A', (7, 1): '9B',   # G
    (8, 0): '1A', (8, 1): '4B',   # G#/Ab
    (9, 0): '8A', (9, 1): '11B',  # A
    (10, 0): '3A', (10, 1): '6B', # A#/Bb
    (11, 0): '10A', (11, 1): '1B' # B
}


def _worker_extract_features(args):
    """
    Worker function for multiprocessing feature extraction.
    Must be at module level to be picklable.

    Args:
        args: Tuple of (db_path, song_id, file_path)

    Returns:
        True if successful, False otherwise
    """
    db_path, song_id, file_path = args

    # Create new database connection in worker process (suppress init log to avoid spam)
    db = Database(db_path)
    db.init(backup_on_startup=False, log_init=False)

    try:
        # Extract features using module-level function
        features = _extract_audio_features(file_path)

        if not features:
            logger.warning(f"No features extracted for {file_path}")
            return False

        # Save to database
        with db.session_scope() as session:
            # Check if features already exist
            existing = session.query(SongFeatures).filter_by(song_id=song_id).first()

            if existing:
                # Update existing
                existing.tempo = features['tempo']
                existing.energy = features['energy']
                existing.valence = features.get('valence')
                existing.danceability = features['danceability']
                existing.acousticness = features.get('acousticness')
                existing.instrumentalness = features.get('instrumentalness')
                existing.loudness = features['loudness']
                existing.key = features['key']
                existing.mode = features['mode']
                existing.camelot_key = features['camelot_key']
                existing.spectral_features = features.get('spectral_features')
                existing.feature_vector = features.get('feature_vector')
                existing.extraction_version = '1.0'
                existing.extracted_at = datetime.utcnow()
            else:
                # Create new
                song_features = SongFeatures(
                    song_id=song_id,
                    tempo=features['tempo'],
                    energy=features['energy'],
                    valence=features.get('valence'),
                    danceability=features['danceability'],
                    acousticness=features.get('acousticness'),
                    instrumentalness=features.get('instrumentalness'),
                    loudness=features['loudness'],
                    key=features['key'],
                    mode=features['mode'],
                    camelot_key=features['camelot_key'],
                    spectral_features=features.get('spectral_features'),
                    feature_vector=features.get('feature_vector'),
                    extraction_version='1.0',
                    extracted_at=datetime.utcnow()
                )
                session.add(song_features)

        return True

    except Exception as e:
        logger.error(f"Error extracting features for {file_path}: {e}")
        return False
    finally:
        db.close()


def _extract_audio_features(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Extract audio features from file using Essentia or mock extractor.
    Module-level function for use by workers.

    Args:
        file_path: Path to audio file

    Returns:
        Dictionary of features or None on error
    """
    if not ESSENTIA_AVAILABLE:
        # Use mock extractor for testing
        from src.extractor.mock_extractor import MockFeatureExtractor
        mock = MockFeatureExtractor()
        return mock.extract_features(file_path)

    try:
        # Load audio
        loader = es.MonoLoader(filename=file_path)
        audio = loader()

        if len(audio) == 0:
            logger.warning(f"Empty audio file: {file_path}")
            return None

        # Extract features
        features = {}

        # Rhythm features
        rhythm_extractor = es.RhythmExtractor2013()
        bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
        features['tempo'] = float(bpm)

        # Energy
        energy = es.Energy()
        features['energy'] = float(energy(audio))

        # Normalize energy to 0-1 range (typical range is 0-1000000)
        features['energy'] = min(features['energy'] / 1000000.0, 1.0)

        # Danceability
        danceability = es.Danceability()
        dance_value, _ = danceability(audio)
        features['danceability'] = float(dance_value)

        # Key and scale
        key_extractor = es.KeyExtractor()
        key, scale, strength = key_extractor(audio)

        # Convert key to numeric (C=0, C#=1, ..., B=11)
        key_map = {
            'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
            'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
        }
        key_num = key_map.get(key.split()[0], 0)
        mode = 1 if scale == 'major' else 0

        features['key'] = key_num
        features['mode'] = mode
        features['camelot_key'] = CAMELOT_WHEEL.get((key_num, mode), '1A')

        # Loudness
        loudness_extractor = es.Loudness()
        features['loudness'] = float(loudness_extractor(audio))

        # Spectral features (for similarity)
        spectral_features = {}

        # Spectral centroid
        centroid = es.Centroid()
        spectrum = es.Spectrum()
        spectral_centroid_values = []

        # Process in frames
        frame_size = 2048
        hop_size = 1024

        for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size):
            spec = spectrum(frame)
            spectral_centroid_values.append(centroid(spec))

        if spectral_centroid_values:
            spectral_features['centroid_mean'] = float(sum(spectral_centroid_values) / len(spectral_centroid_values))

        features['spectral_features'] = spectral_features

        # Try to extract high-level features if available
        try:
            # Valence (happiness/sadness) - estimate from key and mode
            if mode == 1:  # Major
                features['valence'] = 0.6 + (strength * 0.2)  # 0.6-0.8
            else:  # Minor
                features['valence'] = 0.4 - (strength * 0.2)  # 0.2-0.4

            features['valence'] = max(0.0, min(1.0, features['valence']))

            # Acousticness - estimate from spectral features
            if spectral_features.get('centroid_mean'):
                norm_centroid = min(spectral_features['centroid_mean'] / 5000.0, 1.0)
                features['acousticness'] = 1.0 - norm_centroid
            else:
                features['acousticness'] = 0.5

            # Instrumentalness - placeholder
            features['instrumentalness'] = 0.5

        except Exception as e:
            logger.debug(f"Could not extract high-level features: {e}")
            features['valence'] = 0.5
            features['acousticness'] = 0.5
            features['instrumentalness'] = 0.5

        # Create feature vector for similarity
        feature_vector = [
            features['tempo'] / 200.0,
            features['energy'],
            features.get('valence', 0.5),
            features['danceability'],
            features.get('acousticness', 0.5),
            features['key'] / 11.0,
            float(features['mode']),
            features['loudness'] / 100.0,
            spectral_features.get('centroid_mean', 0) / 5000.0
        ]

        features['feature_vector'] = feature_vector

        return features

    except Exception as e:
        logger.error(f"Essentia extraction failed for {file_path}: {e}")
        return None


class FeatureExtractor:
    """Extract audio features using Essentia."""

    def __init__(self, db: Database, workers: int = 4):
        """
        Initialize feature extractor.

        Args:
            db: Database instance
            workers: Number of parallel workers (0 = auto-detect)
        """
        self.db = db
        self.db_path = str(db.db_path)  # Store path as string for worker processes
        self.workers = workers if workers > 0 else cpu_count()

        if not ESSENTIA_AVAILABLE:
            logger.warning("Essentia is not installed. Using mock feature extractor for testing.")
            logger.warning("For production use, please install Essentia in your environment.")

    def extract_all(self, force_reextract: bool = False) -> int:
        """
        Extract features for all songs in database.

        Args:
            force_reextract: If True, re-extract even if features exist

        Returns:
            Number of songs processed
        """
        # Continue even without Essentia - we'll use mock extractor

        # Get songs that need feature extraction
        with self.db.session_scope() as session:
            query = session.query(Song)

            if not force_reextract:
                # Only extract songs without features
                query = query.outerjoin(SongFeatures).filter(SongFeatures.id == None)

            songs = query.all()

            if not songs:
                logger.info("No songs need feature extraction")
                return 0

            song_paths = [(song.id, song.file_path) for song in songs]

        logger.info(f"Extracting features for {len(song_paths)} songs using {self.workers} workers")

        # Extract features in parallel
        extracted_count = 0

        if self.workers == 1:
            # Single-threaded for debugging
            for song_id, file_path in tqdm(song_paths, desc="Extracting features"):
                features = _extract_audio_features(file_path)
                if features and self._save_features(song_id, features):
                    extracted_count += 1
        else:
            # Multi-threaded - prepare arguments with db_path
            worker_args = [(self.db_path, song_id, file_path) for song_id, file_path in song_paths]

            with Pool(processes=self.workers) as pool:
                results = list(tqdm(
                    pool.imap(_worker_extract_features, worker_args),
                    total=len(worker_args),
                    desc="Extracting features"
                ))
                extracted_count = sum(1 for r in results if r)

        return extracted_count

    def _save_features(self, song_id: int, features: Dict[str, Any]) -> bool:
        """
        Save extracted features to database.

        Args:
            song_id: Song database ID
            features: Extracted features dictionary

        Returns:
            True if successful
        """
        try:
            with self.db.session_scope() as session:
                # Check if features already exist
                existing = session.query(SongFeatures).filter_by(song_id=song_id).first()

                if existing:
                    # Update existing
                    existing.tempo = features['tempo']
                    existing.energy = features['energy']
                    existing.valence = features.get('valence')
                    existing.danceability = features['danceability']
                    existing.acousticness = features.get('acousticness')
                    existing.instrumentalness = features.get('instrumentalness')
                    existing.loudness = features['loudness']
                    existing.key = features['key']
                    existing.mode = features['mode']
                    existing.camelot_key = features['camelot_key']
                    existing.spectral_features = features.get('spectral_features')
                    existing.feature_vector = features.get('feature_vector')
                    existing.extraction_version = '1.0'
                    existing.extracted_at = datetime.utcnow()
                else:
                    # Create new
                    song_features = SongFeatures(
                        song_id=song_id,
                        tempo=features['tempo'],
                        energy=features['energy'],
                        valence=features.get('valence'),
                        danceability=features['danceability'],
                        acousticness=features.get('acousticness'),
                        instrumentalness=features.get('instrumentalness'),
                        loudness=features['loudness'],
                        key=features['key'],
                        mode=features['mode'],
                        camelot_key=features['camelot_key'],
                        spectral_features=features.get('spectral_features'),
                        feature_vector=features.get('feature_vector'),
                        extraction_version='1.0',
                        extracted_at=datetime.utcnow()
                    )
                    session.add(song_features)

            return True

        except Exception as e:
            logger.error(f"Error saving features: {e}")
            return False



def extract_single_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Extract features from a single file (utility function).

    Args:
        file_path: Path to audio file

    Returns:
        Dictionary of features or None
    """
    extractor = FeatureExtractor(None, workers=1)
    return extractor._extract_features(file_path)