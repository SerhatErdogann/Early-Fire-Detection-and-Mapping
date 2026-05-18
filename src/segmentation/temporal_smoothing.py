# src/segmentation/temporal_smoothing.py

from collections import deque
import numpy as np


class TemporalMaskSmoother:
    """
    Son N maskeyi tutar ve stabil bir mask üretir.

    Amaç:
    - Tek frame'lik gürültüleri azaltmak
    - Aynı bölgede birkaç frame boyunca görünen sıcak alanları korumak
    """

    def __init__(self, history_size=5, vote_threshold=0.4, min_history=2):
        """
        history_size:
            Kaç maskenin hafızada tutulacağı.

        vote_threshold:
            Bir pikselin stabil maskede kalması için son maskelerin kaçında
            görünmesi gerektiği.

            0.4 ve history_size=5 ise:
            piksel yaklaşık en az 2 frame'de varsa kalır.

            0.6 ve history_size=5 ise:
            piksel yaklaşık en az 3 frame'de varsa kalır.
        """

        self.history_size = history_size
        self.vote_threshold = vote_threshold
        self.min_history = max(1, int(min_history))
        self.mask_history = deque(maxlen=history_size)

    def reset(self):
        """
        Mask geçmişini temizler.
        No-fire frame geldiğinde eski yangın maskesi taşmasın diye kullanılır.
        """

        self.mask_history.clear()

    def update(self, mask):
        """
        Yeni maskeyi ekler ve stabilize edilmiş mask döndürür.
        """

        binary_mask = (mask > 0).astype(np.float32)

        self.mask_history.append(binary_mask)

        if len(self.mask_history) < self.min_history:
            return np.zeros_like(mask, dtype=np.uint8)

        stacked = np.stack(list(self.mask_history), axis=0)

        mean_mask = np.mean(stacked, axis=0)

        stable_mask = (mean_mask >= self.vote_threshold).astype(np.uint8) * 255

        return stable_mask
