# src/geospatial/fire_tracker.py

import math


def haversine_distance_m(lat1, lon1, lat2, lon2):
    """
    İki lat/lon noktası arası yaklaşık metre mesafesi.
    """

    r = 6371000.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return r * c


class FireTracker:
    """
    Arka arkaya gelen frame'lerde aynı yangın odaklarını tek track altında toplar.
    Konum olarak en şiddetli (en büyük alan) gözlem kullanılır.
    """

    def __init__(self, match_distance_m=25.0, max_missing_frames=10):
        self.match_distance_m = match_distance_m
        self.max_missing_frames = max_missing_frames
        self.tracks = {}
        self.next_id = 1

    def update(self, fire_lat, fire_lon, frame_idx, approx_area_m2=None, fire_probability=None):
        """
        Yeni fire point'i mevcut track ile eşleştirir veya yeni track açar.
        Konum olarak en büyük alana sahip gözlem saklanır.
        Returns: (track_id, best_lat, best_lon)
        """

        if fire_lat is None or fire_lon is None:
            return None, None, None

        best_track_id = None
        best_distance = None

        for track_id, track in self.tracks.items():
            missing = frame_idx - track["last_frame_idx"]

            if missing > self.max_missing_frames:
                continue

            distance = haversine_distance_m(
                fire_lat,
                fire_lon,
                track["latitude"],
                track["longitude"]
            )

            if distance <= self.match_distance_m:
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_track_id = track_id

        if best_track_id is None:
            track_id = f"FIRE_{self.next_id}"
            self.next_id += 1

            self.tracks[track_id] = {
                "track_id": track_id,
                "latitude": fire_lat,
                "longitude": fire_lon,
                "first_frame_idx": frame_idx,
                "last_frame_idx": frame_idx,
                "observations": 1,
                "max_area_m2": approx_area_m2,
                "last_area_m2": approx_area_m2,
                "max_probability": fire_probability,
                "last_probability": fire_probability
            }

            return track_id, fire_lat, fire_lon

        track = self.tracks[best_track_id]

        # En şiddetli gözlem konumunu koru (en büyük alan)
        if approx_area_m2 is not None and (track["max_area_m2"] is None or approx_area_m2 > track["max_area_m2"]):
            track["latitude"] = fire_lat
            track["longitude"] = fire_lon
            track["max_area_m2"] = approx_area_m2

        track["last_frame_idx"] = frame_idx
        track["observations"] += 1
        track["last_area_m2"] = approx_area_m2
        track["last_probability"] = fire_probability

        if fire_probability is not None:
            if track["max_probability"] is None or fire_probability > track["max_probability"]:
                track["max_probability"] = fire_probability

        return best_track_id, track["latitude"], track["longitude"]