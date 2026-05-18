from __future__ import annotations

import math
import os
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd

from env_utils import load_project_env


load_project_env(__file__)


class FuelScorer:
    """
    Bitki yanicilik modelini calistirir.

    Model egitimde ndvi, ndmi ve one-hot Dynamic World land-cover kolonlari ile
    kaydedildi. Lat/lon dogrudan modele verilmez; sadece GEE ozelliklerini
    cekmek icin kullanilir.
    """

    def __init__(
        self,
        model_path: str | os.PathLike | None = None,
        default_score: float = 0.50,
        use_gee: bool = False,
        gee_project: str | None = None,
        lookback_days: int = 60,
        cloud_pct: int = 30,
        buffer_m: int = 100,
        cache_distance_km: float = 1.5,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.default_score = float(default_score)
        self.use_gee = bool(use_gee)
        self.gee_project = gee_project or os.getenv("GEE_PROJECT_ID") or None
        self.lookback_days = int(lookback_days)
        self.cloud_pct = int(cloud_pct)
        self.buffer_m = int(buffer_m)
        self.cache_distance_km = float(cache_distance_km)

        self.model = None
        self.features: list[str] = []
        self._last_lat: float | None = None
        self._last_lon: float | None = None
        self._last_date: str | None = None
        self._last_score: float | None = None
        self._gee_unavailable = False
        self._gee_initialized = False

        if self.model_path and self.model_path.exists():
            self._load_model(self.model_path)
        elif self.model_path:
            print(f"[FuelScorer] Model bulunamadi: {self.model_path}")

    def _load_model(self, model_path: Path) -> None:
        try:
            model_data = joblib.load(model_path)
            if isinstance(model_data, dict):
                self.model = model_data.get("model")
                self.features = list(model_data.get("features") or [])
            else:
                self.model = model_data
                self.features = list(getattr(model_data, "feature_names_in_", []))

            if self.model is None:
                raise ValueError("Model dosyasinda 'model' nesnesi bulunamadi.")
            if not self.features:
                raise ValueError("Model dosyasinda beklenen ozellik listesi bulunamadi.")

            print(f"[FuelScorer] Model yuklendi: {model_path}")
            print(f"[FuelScorer] Beklenen ozellikler: {self.features}")
        except Exception as exc:
            self.model = None
            self.features = []
            print(f"[FuelScorer] Model yuklenirken hata olustu: {exc}")

    def get_score(
        self,
        lat: float | None = None,
        lon: float | None = None,
        ndvi: float | None = None,
        ndmi: float | None = None,
        land_cover: int | str | None = None,
        flight_date: datetime | str | None = None,
    ) -> float:
        """0.0-1.0 arasi bitki yanicilik skoru dondurur."""
        if self.model is None:
            return self.default_score

        if ndvi is not None and ndmi is not None and land_cover is not None:
            return self.predict_from_features(ndvi=ndvi, ndmi=ndmi, land_cover=land_cover)

        if not self.use_gee or lat is None or lon is None:
            return self.default_score

        date_key = self._date_key(flight_date)
        if self._can_reuse_cached_score(lat, lon, date_key):
            return float(self._last_score)

        features = self._fetch_gee_features(lat, lon, flight_date)
        if features is None:
            return self.default_score

        score = self.predict_from_features(**features)
        self._last_lat = float(lat)
        self._last_lon = float(lon)
        self._last_date = date_key
        self._last_score = score
        return score

    def predict_from_features(self, ndvi: float, ndmi: float, land_cover: int | str) -> float:
        if self.model is None:
            return self.default_score

        try:
            row = self._build_feature_row(ndvi=ndvi, ndmi=ndmi, land_cover=land_cover)
            score = self.model.predict_proba(row)[0][1]
            return float(max(0.0, min(1.0, score)))
        except Exception as exc:
            print(f"[FuelScorer] Skor hesaplanamadi: {exc}")
            return self.default_score

    def _build_feature_row(self, ndvi: float, ndmi: float, land_cover: int | str) -> pd.DataFrame:
        row = pd.DataFrame({"ndvi": [float(ndvi)], "ndmi": [float(ndmi)]})
        lc_col = f"LC_{int(float(land_cover))}"

        for feature in self.features:
            if feature.startswith("LC_"):
                row[feature] = 1 if feature == lc_col else 0

        for feature in self.features:
            if feature not in row.columns:
                row[feature] = 0

        return row[self.features]

    def _fetch_gee_features(
        self,
        lat: float,
        lon: float,
        flight_date: datetime | str | None,
    ) -> dict[str, float | int] | None:
        if self._gee_unavailable:
            return None

        try:
            import ee

            if not self._gee_initialized:
                if self.gee_project:
                    ee.Initialize(project=self.gee_project)
                else:
                    ee.Initialize()
                self._gee_initialized = True
        except Exception as exc:
            self._gee_unavailable = True
            print(f"[FuelScorer] Earth Engine kullanilamiyor: {exc}")
            return None

        try:
            end_date = self._parse_date(flight_date)
            start_date = end_date - timedelta(days=self.lookback_days)
            start = start_date.strftime("%Y-%m-%d")
            end = end_date.strftime("%Y-%m-%d")

            point_area = ee.Geometry.Point([float(lon), float(lat)]).buffer(self.buffer_m)

            dw = (
                ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                .filterBounds(point_area)
                .filterDate(start, end)
            )
            if dw.size().getInfo() <= 0:
                raise ValueError("Dynamic World verisi bulunamadi.")
            dw_img = dw.sort("system:time_start", False).first()
            lc_data = dw_img.select("label").reduceRegion(
                reducer=ee.Reducer.mode(),
                geometry=point_area,
                scale=10,
            ).getInfo()
            land_cover = lc_data.get("label")
            if land_cover is None:
                raise ValueError("Land-cover degeri bulunamadi.")

            s2 = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(point_area)
                .filterDate(start, end)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", self.cloud_pct))
            )
            if s2.size().getInfo() <= 0:
                raise ValueError("Bulutsuz Sentinel-2 verisi bulunamadi.")

            composite = s2.median()
            ndvi = composite.normalizedDifference(["B8", "B4"]).rename("NDVI")
            ndmi = composite.normalizedDifference(["B8", "B11"]).rename("NDMI")
            values = ndvi.addBands(ndmi).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point_area,
                scale=10,
            ).getInfo()
            ndvi_val = values.get("NDVI")
            ndmi_val = values.get("NDMI")
            if ndvi_val is None or ndmi_val is None:
                raise ValueError("NDVI/NDMI degeri bulunamadi.")

            return {"ndvi": float(ndvi_val), "ndmi": float(ndmi_val), "land_cover": int(land_cover)}
        except Exception as exc:
            print(f"[FuelScorer] GEE ozellikleri alinamadi: {exc}")
            return None

    def _can_reuse_cached_score(self, lat: float, lon: float, date_key: str) -> bool:
        if self._last_score is None or self._last_lat is None or self._last_lon is None:
            return False
        if self._last_date != date_key:
            return False
        return self._haversine_km(self._last_lat, self._last_lon, float(lat), float(lon)) <= self.cache_distance_km

    def _date_key(self, value: datetime | str | None) -> str:
        return self._parse_date(value).strftime("%Y-%m-%d")

    @staticmethod
    def _parse_date(value: datetime | str | None) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and value.strip():
            for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y"):
                try:
                    return datetime.strptime(value.strip(), fmt)
                except ValueError:
                    continue
        return datetime.utcnow()

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        radius_km = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        )
        return radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
