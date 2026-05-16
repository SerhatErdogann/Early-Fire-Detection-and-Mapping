# src/postgis_writer.py

import psycopg2


class PostgisWriter:
    def __init__(
        self,
        host="localhost",
        port=5432,
        database="fire_mapping",
        user="postgres",
        password="postgres"
    ):
        self.conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=database,
            user=user,
            password=password
        )

        self.conn.autocommit = True

    def close(self):
        if self.conn:
            self.conn.close()

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
            %s, %s, %s,
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

        if not fire_track_id or fire_latitude is None or fire_longitude is None:
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
            result.get("simulated", True),
            result.get("location_source"),
            fire_longitude,
            fire_latitude
        )

        with self.conn.cursor() as cur:
            cur.execute(sql, values)