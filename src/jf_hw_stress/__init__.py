"""Jellyfin hardware transcode stress tester.

Runs Jellyfin-style HLS transcodes with ffmpeg and adds streams until one can no longer keep
up with real time, to find how many concurrent transcodes a machine's media engine sustains.
"""

__version__ = "2.1.0"
APP_NAME = "jf-hw-stress"
APP_TITLE = "Jellyfin Hardware Transcode Stress Tester"
