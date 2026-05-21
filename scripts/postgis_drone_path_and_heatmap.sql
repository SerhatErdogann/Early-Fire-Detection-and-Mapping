-- Drone flight path as LineString from drone_frame_points
-- Creates a time-ordered line showing where the drone flew

DROP VIEW IF EXISTS drone_flight_path;

CREATE VIEW drone_flight_path AS
SELECT
    1 AS id,
    ST_MakeLine(
        ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)
        ORDER BY video_time_s ASC
    ) AS geom,
    COUNT(*) AS total_points,
    MIN(video_time_s) AS start_time,
    MAX(video_time_s) AS end_time,
    AVG(altitude_m) AS avg_altitude_m
FROM drone_frame_points
WHERE latitude IS NOT NULL AND longitude IS NOT NULL;

-- Active fire heatmap view (aggregated fire observations)
DROP VIEW IF EXISTS fire_heatmap_points;

CREATE VIEW fire_heatmap_points AS
SELECT
    fire_track_id,
    ST_SetSRID(ST_MakePoint(longitude, latitude), 4326) AS geom,
    last_probability,
    (COALESCE(last_probability, 0) * LN(1 + GREATEST(COALESCE(observations, 0), 0))) AS heatmap_weight,
    last_video_time_s,
    observations,
    max_area_m2,
    overlay_url
FROM active_fire_tracks
WHERE fire_track_id IS NOT NULL
  AND latitude IS NOT NULL
  AND longitude IS NOT NULL
ORDER BY last_probability DESC;
