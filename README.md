# Playlister

Generate mood-based playlists from your local music library with smooth transitions between songs. Runs completely offline using Docker.

## Why?

Because I wanted automatic playlists so I can just hit play. No streaming service algorithms, no internet dependency, just my music organized the way I want it.

## What it does

Analyzes your local music files and generates playlists based on mood. Songs are ordered with smooth transitions using BPM matching, key compatibility, and energy flow. Like having a DJ that knows your entire library.

**Default moods (fully customizable):**

- Uplifting
- Rock Out
- Dance
- Chill
- Relax

You can define any mood you want by specifying audio feature ranges in the configuration.

**Audio formats:**
Lossless: FLAC, WAV, AIFF, APE, WavPack, TTA, TAK
Lossy: MP3, Ogg Vorbis, Opus, AAC (M4A/MP4), Musepack, WMA, Speex

**Playlist formats:**
M3U8, PLS, JSON

## Requirements

- Docker and Docker Compose
- A music library
- About 3GB disk space for the Docker image (it's beefy)

## Installation

### Option 0: Setup

Clone the repository:

```bash
git clone https://github.com/ticoombs/playlister.git
cd playlister
```

Set up your environment:

```bash
cp .env.example .env
```

Edit `.env` and set `MUSIC_PATH` to your music library location:

```bash
MUSIC_PATH=/path/to/your/music
UID=1000  # Your user ID/the owner of your music (run: id -u)
GID=1000  # Your group ID/the group of your music (run: id -g)
```

### Option 1: Build from Source (Recommended)

Build your image locally (takes 10-15 minutes):

```bash
docker-compose build
```

### Option 2: Pull pre-built image (it's huge. So don't and I'm not supporting it)

Pull the pre-built image: (if you are lucky)

```bash
docker-compose pull
```

## Usage

Initialize the database:

```bash
docker-compose run --rm playlister init
```

Edit the configuration as desired. (See Below for explanation)

```bash
vim config/config.yaml

```

Scan your music library (automatically extracts features):

```bash
docker-compose run --rm playlister scan /music
```

The scan command automatically extracts audio features (tempo, energy, key, etc.) for all new files. To skip feature extraction and only scan file metadata:

```bash
docker-compose run --rm playlister scan /music --no-extract
```

Or run feature extraction separately:

```bash
docker-compose run --rm playlister extract
```

Classify songs by mood:

```bash
docker-compose run --rm playlister classify
```

Generate a playlist:

```bash
docker-compose run --rm playlister generate --mood chill --count 50 --name "Sunday Morning"
```

Playlists are automatically exported to M3U8 format in the `playlists/` directory.

Export an existing playlist to different formats:

```bash
# Export playlist by ID (see history command for IDs)
docker-compose run --rm playlister export 1 --format m3u8
docker-compose run --rm playlister export 1 --format json
docker-compose run --rm playlister export 1 --format pls
```

View playlist history:

```bash
docker-compose run --rm playlister history
```

View your library statistics:

```bash
docker-compose run --rm playlister stats
```

Generated playlists are saved to the `playlists/` directory and can be opened in any media player.

## How it works

**Scanner:** Walks through your music directory recursively, scanning any depth of nested folders (e.g., `Artists/Album/Song.mp3` or `Artists/Album/Disk/Song.flac`). Extracts metadata (artist, title, album, year, etc) using the mutagen library. Supports directory exclusion patterns to skip playlist folders, backups, and other non-music directories.

**Feature Extractor:** Analyzes audio files using Essentia to extract musical features including tempo (BPM), energy level, valence (musical positivity), danceability, acousticness, musical key, and spectral characteristics.

**Classifier:** Uses rule-based classification to assign songs to moods based on their audio features. Each mood has defined ranges for energy, tempo, and other attributes. Classification is strict to prevent songs from appearing in incompatible playlists.

**Playlist Generator:** Creates playlists optimized for smooth transitions. Uses multi-factor scoring combining BPM matching, energy gradients, key compatibility (Camelot wheel), and spectral similarity. Smart shuffle prevents artist and album repetition while maintaining flow.

**Storage:** SQLite database stores all metadata, features, classifications, and playlists. No external database required. All computations are cached - rescans only process new or modified files.

## Configuration

Mood definitions and playlist settings can be customized in `config/config.yaml`:

```yaml
moods:
  chill:
    energy: [0.25, 0.45]
    valence: [0.45, 0.75]
    tempo: [85, 115]

playlist:
  transitions:
    bpm_tolerance: 10
    energy_max_jump: 0.2
    key_compatibility: true

  smart_shuffle:
    avoid_same_artist_within: 5
    avoid_same_album_within: 10

advanced:
  confidence_threshold: 0.85
  strict_matching: true
```

The confidence threshold controls how strictly songs must match mood criteria. Higher values mean more accurate but fewer classified songs.

### Directory Exclusion

If you have playlists, backups, or other directories you want to skip:

```yaml
scanner:
  exclude_patterns:
    - "Playlists"  # Skip /music/Playlists directory
    - "playlists"
    - ".backup"
    - "@eaDir"  # Synology thumbnails
```

This prevents scanning playlist directories and avoids circular references when playlists are stored inside the music directory.

### Custom Moods

You can define any mood you want based on audio features. Add custom moods to `config/config.yaml`:

```yaml
moods:
  # Your custom moods
  aggressive:
    energy: [0.8, 1.0]
    valence: [0.0, 0.4]
    tempo: [140, 200]
    loudness: [0.7, 1.0]

  melancholy:
    energy: [0.2, 0.5]
    valence: [0.0, 0.3]
    tempo: [70, 100]
    acousticness: [0.4, 1.0]

  party:
    energy: [0.7, 1.0]
    danceability: [0.7, 1.0]
    tempo: [120, 140]
    valence: [0.6, 1.0]

  study:
    energy: [0.2, 0.4]
    instrumentalness: [0.5, 1.0]
    tempo: [80, 110]
    acousticness: [0.3, 0.8]
```

**Available features for mood definitions:**

- `tempo`: BPM (beats per minute)
- `energy`: Intensity level (0-1)
- `valence`: Musical positivity (0-1, low = sad, high = happy)
- `danceability`: Rhythmic stability (0-1)
- `acousticness`: Acoustic vs electronic (0-1)
- `instrumentalness`: Vocal vs instrumental (0-1)
- `loudness`: Overall loudness (0-1)

After adding custom moods, reclassify your library:

```bash
docker-compose run --rm playlister classify
docker-compose run --rm playlister generate --mood aggressive --count 50
```

## Performance

Processing times on a typical system:

- Scanning: 100 files/second
- Feature extraction: 1-5 files/second (only runs on new files)
- Classification: 1000 files/second
- Playlist generation: Under 1 second for 100 songs

### Incremental Processing

The system is optimized for large libraries (10,000+ songs):

- **Scanning:** Only processes new or modified files (checks file modification time)
- **Feature extraction:** Skips files that already have extracted features
- **Database caching:** All computations stored in SQLite database
- **Force re-extraction:** Use `--force` flag if you need to recompute features

Example workflow for adding 100 new songs to a 10,000 song library:

```bash
# Only the 100 new songs will be processed
docker-compose run --rm playlister scan /music
docker-compose run --rm playlister classify
```

The existing 9,900 songs are skipped automatically - processing takes seconds instead of hours.

### Very Large Libraries (100k+ files)

For extremely large libraries, use stream mode to avoid loading all file paths into memory:

```bash
# Stream mode: constant memory usage, no progress bar
docker-compose run --rm playlister scan /music --stream
```

Or enable in config:

```yaml
scanner:
  stream_mode: true
```

**Memory usage:**

- Normal mode: ~10-50 MB for 100k file paths (shows progress bar)
- Stream mode: ~constant memory (shows count every 100 files)

## Transition Quality

Playlists achieve 0.80-0.90 average transition scores through:

- BPM matching within ±10 BPM
- Smooth energy transitions
- Harmonic mixing using the Camelot wheel
- Spectral similarity for sonic cohesion

## Nested Directory Support

The scanner automatically handles any directory structure:

```
/music/
  Artists/
    Led Zeppelin/
      Led Zeppelin IV/
        01 Black Dog.flac
    Pink Floyd/
      The Wall/
        Disc 1/
          01 In The Flesh.mp3
  Compilations/
    song.mp3
  Playlists/  # This directory is excluded by default
```

All audio files are discovered recursively at any depth.

## Playlists Location

You can store playlists anywhere:

**Outside music directory** (default):

- Config: `playlists: /playlists`
- Paths in playlists: `../music/Artist/Song.mp3`

**Inside music directory** (optional):

- Config: `playlists: /music/Playlists`
- Paths in playlists: `../Artist/Song.mp3`
- Add `Playlists` to `scanner.exclude_patterns` to prevent scanning playlist files

## Troubleshooting

**No songs found with mood:**
Run the classify command first: `docker-compose run --rm playlister classify`

**Permission denied errors:**
Make sure UID and GID in `.env` match your user (run `id -u` and `id -g`)

**Database locked:**
Stop all running containers and try again

**Essentia not available:**
You are using the mock extractor for testing. For production accuracy, build with `docker-compose build` instead of the test configuration.

**Playlist directory being scanned:**
Add it to `scanner.exclude_patterns` in `config/config.yaml`

## Project Structure

```
src/scanner/     - File discovery and metadata extraction
src/extractor/   - Audio feature extraction
src/classifier/  - Mood classification
src/generator/   - Playlist generation and export
src/storage/     - Database models
src/cli/         - Command-line interface
```

Database tables:

- songs: File paths and metadata
- song_features: Extracted audio features
- song_classifications: Mood assignments
- playlists: Generated playlists
- playlist_songs: Song ordering with transition scores

## Development

The project uses Docker for all operations. All commands must be run inside the container using `docker-compose run --rm playlister <command>`.

See `CLAUDE.md` for detailed development documentation.

## License

MIT

## Credits

Built using:

- Essentia (audio analysis)
- mutagen (metadata extraction)
- SQLAlchemy (database)
- Click (CLI framework)
