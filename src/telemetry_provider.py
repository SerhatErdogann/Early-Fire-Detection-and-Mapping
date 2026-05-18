# src/telemetry_provider.py

import pandas as pd


class DjiCsvTelemetryProvider:
    """
    DJI CSV dosyasını canlı telemetry geliyormuş gibi simüle eder.

    Gerçek sistemde bunun yerine:
    - drone SDK
    - socket
    - MQTT
    - REST API
    - serial telemetry
    kullanılabilir.

    Ama pipeline aynı kalır:
        telemetry_provider.get_current(video_time_s)
    """

    def __init__(self, csv_path, video_duration_s, telemetry_offset_s=None):
        self.csv_path = csv_path
        self.video_duration_s = video_duration_s
        self.telemetry_offset_s = telemetry_offset_s

        self.df = self._read_csv(csv_path)

        self.telemetry_start = self.df["OSD.flyTime [s]"].min()
        self.telemetry_end = self.df["OSD.flyTime [s]"].max()
        self.telemetry_duration = self.telemetry_end - self.telemetry_start

    def _read_csv(self, csv_path):
        required_columns = [
            "OSD.flyTime [s]",
            "OSD.latitude",
            "OSD.longitude",
            "OSD.height [ft]",
            "OSD.altitude [ft]",
            "OSD.yaw [360]",
            "GIMBAL.pitch",
            "GIMBAL.yaw [360]"
        ]

        aliases = {
            "OSD.flyTime [s]": ["OSD.flyTime", "flyTime [s]", "time_s", "saniye"],
            "OSD.latitude": ["latitude", "lat", "enlem"],
            "OSD.longitude": ["longitude", "lon", "lng", "boylam"],
            "OSD.height [ft]": ["height_ft", "height [ft]", "OSD.height"],
            "OSD.altitude [ft]": ["altitude_ft", "altitude [ft]", "OSD.altitude"],
            "OSD.yaw [360]": ["yaw", "drone_yaw", "OSD.yaw"],
            "GIMBAL.pitch": ["gimbal_pitch", "pitch"],
            "GIMBAL.yaw [360]": ["gimbal_yaw", "GIMBAL.yaw", "gimbal yaw"],
        }

        candidates = []
        for skiprows in (0, 1):
            try:
                df = pd.read_csv(csv_path, skiprows=skiprows)
                df = self._apply_aliases(df, aliases)
                missing = [col for col in required_columns if col not in df.columns]
                candidates.append((missing, df, skiprows))
                if not missing:
                    break
            except Exception:
                continue
        else:
            raise ValueError(f"Telemetry CSV could not be read: {csv_path}")

        missing, df, skiprows_used = min(candidates, key=lambda item: len(item[0]))

        if missing:
            raise ValueError(f"Missing telemetry columns after CSV parse (skiprows={skiprows_used}): {missing}")

        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(
            subset=[
                "OSD.flyTime [s]",
                "OSD.latitude",
                "OSD.longitude"
            ]
        )

        df = df.sort_values("OSD.flyTime [s]").reset_index(drop=True)

        return df

    @staticmethod
    def _apply_aliases(df, aliases):
        rename = {}
        lower_to_col = {str(col).strip().lower(): col for col in df.columns}
        for canonical, names in aliases.items():
            if canonical in df.columns:
                continue
            for name in names:
                match = lower_to_col.get(name.lower())
                if match is not None:
                    rename[match] = canonical
                    break
        return df.rename(columns=rename)

    def get_current(self, video_time_s):
        """
        Canlı sistemde bu fonksiyon o anki drone telemetry bilgisini döndürür.

        Şu an CSV simülasyonunda:
            video zamanı telemetry uçuş zamanına normalize edilir.
        """

        if self.telemetry_offset_s is not None:
            sim_flight_time_s = self.telemetry_start + float(video_time_s) + float(self.telemetry_offset_s)
        elif self.video_duration_s <= 0 or self.telemetry_duration <= 0:
            sim_flight_time_s = self.telemetry_start
        else:
            video_ratio = video_time_s / self.video_duration_s
            video_ratio = max(0.0, min(1.0, video_ratio))

            sim_flight_time_s = (
                self.telemetry_start +
                video_ratio * self.telemetry_duration
            )

        closest_idx = (
            self.df["OSD.flyTime [s]"] - sim_flight_time_s
        ).abs().idxmin()

        row = self.df.loc[closest_idx]

        height_ft = row["OSD.height [ft]"]

        if pd.isna(height_ft) or height_ft <= 0:
            height_ft = row["OSD.altitude [ft]"]

        altitude_m = height_ft * 0.3048 if pd.notna(height_ft) else None

        return {
            "sim_flight_time_s": float(sim_flight_time_s),
            "telemetry_fly_time_s": float(row["OSD.flyTime [s]"]),
            "latitude": float(row["OSD.latitude"]),
            "longitude": float(row["OSD.longitude"]),
            "altitude_m": float(altitude_m) if altitude_m is not None else None,
            "height_ft": float(row["OSD.height [ft]"]) if pd.notna(row["OSD.height [ft]"]) else None,
            "altitude_ft": float(row["OSD.altitude [ft]"]) if pd.notna(row["OSD.altitude [ft]"]) else None,
            "drone_yaw": float(row["OSD.yaw [360]"]) if pd.notna(row["OSD.yaw [360]"]) else None,
            "gimbal_pitch": float(row["GIMBAL.pitch"]) if pd.notna(row["GIMBAL.pitch"]) else None,
            "gimbal_yaw": float(row["GIMBAL.yaw [360]"]) if pd.notna(row["GIMBAL.yaw [360]"]) else None,
            "location_source": "simulated_live_dji_csv",
            "simulated": True
        }
