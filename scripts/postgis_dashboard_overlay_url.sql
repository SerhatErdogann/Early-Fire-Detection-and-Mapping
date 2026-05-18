-- Adds image URL columns used by the Flask/GeoServer dashboard.
-- Run this once on the PostGIS database backing the GeoServer workspace.

ALTER TABLE IF EXISTS fire_observations
    ADD COLUMN IF NOT EXISTS overlay_url TEXT;

ALTER TABLE IF EXISTS active_fire_tracks
    ADD COLUMN IF NOT EXISTS overlay_url TEXT;
