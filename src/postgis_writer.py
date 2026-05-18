# src/postgis_writer.py

import os
from urllib.parse import quote, urlparse

import psycopg2

from env_utils import load_project_env


load_project_env(__file__)


class PostgisWriter:
    def __init__(
        self,
        host=None,
        port=None,
        database=None,
        user=None,
        password=None,
    ):
        self.conn = psycopg2.connect(
            host=host or os.getenv("POSTGIS_HOST", "localhost"),
            port=port or int(os.getenv("POSTGIS_PORT", "5432")),
            dbname=database or os.getenv("POSTGIS_DB", "fire_mapping"),
            user=user or os.getenv("POSTGIS_USER", "postgres"),
            password=password or os.getenv("POSTGIS_PASSWORD", "postgres"),
        )

        self.conn.autocommit = True
        self._ensure_dashboard_columns()

    def close(self):
        if self.conn:
            self.conn.close()

    def _ensure_dashboard_columns(self):
        """Add optional dashboard URL columns when the PostGIS tables already exist."""
        statements = [
            "ALTER TABLE IF EXISTS fire_observations ADD COLUMN IF NOT EXISTS overlay_url TEXT;",
            "ALTER TABLE IF EXISTS active_fire_tracks ADD COLUMN IF NOT EXISTS overlay_url TEXT;",
        ]
        with self.conn.cursor() as cur:
            for sql in statements:
                cur.execute(sql)

    @staticmethod
    def _is_url(value):
        parsed = urlparse(str(value or ""))
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def _overlay_url(self, result):
        explicit_url = result.get("overlay_url") or result.get("image_url") or result.get("frame_url")
        if explicit_url and self._is_url(explicit_url):
            return str(explicit_url)

        overlay_path = result.get("overlay_path") or result.get("fire_frame_path")
        if overlay_path and self._is_url(overlay_path):
            return str(overlay_path)

        base_url = os.getenv("DASHBOARD_FRAME_BASE_URL", "").rstrip("/")
        if not base_url or not overlay_path:
            return None

        filename = os.path.basename(str(overlay_path))
        if not filename:
            return None
        return f"{base_url}/{quote(filename)}"

    def insert_drone_frame_point(self, result):
        """
        Her işlenen frame için drone konumunu yazar.
        no_fire ise yeşil, fire ise kırmızı gösterilebilir.
        """

        latitude = result.get("latitude")
        longitude = result.get("longitude")

        if latitude is None or longitude is None:
            return

        sql = """
        INSERT INTO drone_frame_points (
            frame_idx,
            video_time_s,
            prediction,
            fire_probability,
            latitude,
            longitude,
            altitude_m,
            drone_yaw,
            gimbal_pitch,
            gimbal_yaw,
            simulated,
            location_source,
            geom
        )
        VALUES (
            %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s,
            %s, %s,
            ST_SetSRID(ST_MakePoint(%s, %s), 4326)
        );
        """

        values = (
            result.get("frame_idx"),
            result.get("video_time_s"),
            result.get("prediction"),
            result.get("fire_probability"),
            latitude,
            longitude,
            result.get("altitude_m"),
            result.get("drone_yaw"),
            result.get("gimbal_pitch"),
            result.get("gimbal_yaw"),
            result.get("simulated", True),
            result.get("location_source"),
            longitude,
            latitude
        )

        with self.conn.cursor() as cur:
            cur.execute(sql, values)

    def insert_fire_observation(self, result):
        """
        Her yangın bölgesi için ham gözlem kaydı yazar.
        """

        fire_latitude = result.get("fire_latitude")
        fire_longitude = result.get("fire_longitude")

        if fire_latitude is None or fire_longitude is None:
            return

        sql = """
        INSERT INTO fire_observations (
            fire_track_id,
            frame_idx,
            video_time_s,
            region_id,
            pixel_area,
            approx_area_m2,
            fire_probability,
            fire_latitude,
            fire_longitude,
            drone_latitude,
            drone_longitude,
            altitude_m,
            drone_yaw,
            gimbal_pitch,
            fire_frame_path,
            mask_path,
            overlay_path,
            overlay_url,
            simulated,
            location_source,
            geom
        )
        VALUES (
            %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s,
            %s, %s,
            %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s,
            ST_SetSRID(ST_MakePoint(%s, %s), 4326)
        );
        """

        values = (
            result.get("fire_track_id"),
            result.get("frame_idx"),
            result.get("video_time_s"),
            result.get("region_id"),
            result.get("pixel_area"),
            result.get("approx_area_m2"),
            result.get("fire_probability"),
            fire_latitude,
            fire_longitude,
            result.get("latitude"),
            result.get("longitude"),
            result.get("altitude_m"),
            result.get("drone_yaw"),
            result.get("gimbal_pitch"),
            result.get("fire_frame_path"),
            result.get("mask_path"),
            result.get("overlay_path"),
            self._overlay_url(result),
            result.get("simulated", True),
            result.get("location_source"),
            fire_longitude,
            fire_latitude
        )

        with self.conn.cursor() as cur:
            cur.execute(sql, values)

    def upsert_active_fire_track(self, result):
        """
        Aynı fire_track_id varsa günceller.
        Yoksa yeni aktif yangın kaydı açar.
        """

        fire_track_id = result.get("fire_track_id")
        fire_latitude = result.get("fire_latitude")
        fire_longitude = result.get("fire_longitude")

        if fire_track_id is None or fire_latitude is None or fire_longitude is None:
            return

        sql = """
        INSERT INTO active_fire_tracks (
            fire_track_id,
            first_frame_idx,
            last_frame_idx,
            last_video_time_s,
            observations,
            last_area_m2,
            max_area_m2,
            last_probability,
            latitude,
            longitude,
            overlay_path,
            overlay_url,
            simulated,
            location_source,
            geom
        )
        VALUES (
            %s, %s, %s, %s,
            1,
            %s, %s, %s,
            %s, %s,
            %s,
            %s,
            %s, %s,
            ST_SetSRID(ST_MakePoint(%s, %s), 4326)
        )
        ON CONFLICT (fire_track_id)
        DO UPDATE SET
            last_frame_idx = EXCLUDED.last_frame_idx,
            last_video_time_s = EXCLUDED.last_video_time_s,
            observations = active_fire_tracks.observations + 1,
            last_area_m2 = EXCLUDED.last_area_m2,
            max_area_m2 = GREATEST(
                COALESCE(active_fire_tracks.max_area_m2, 0),
                COALESCE(EXCLUDED.max_area_m2, 0)
            ),
            last_probability = EXCLUDED.last_probability,
            latitude = EXCLUDED.latitude,
            longitude = EXCLUDED.longitude,
            overlay_path = EXCLUDED.overlay_path,
            overlay_url = EXCLUDED.overlay_url,
            simulated = EXCLUDED.simulated,
            location_source = EXCLUDED.location_source,
            last_seen_at = NOW(),
            geom = EXCLUDED.geom;
        """

        approx_area_m2 = result.get("approx_area_m2")

        values = (
            fire_track_id,
            result.get("frame_idx"),
            result.get("frame_idx"),
            result.get("video_time_s"),
            approx_area_m2,
            approx_area_m2,
            result.get("fire_probability"),
            fire_latitude,
            fire_longitude,
            result.get("overlay_path"),
            self._overlay_url(result),
            result.get("simulated", True),
            result.get("location_source"),
            fire_longitude,
            fire_latitude
        )

        with self.conn.cursor() as cur:
            cur.execute(sql, values)
