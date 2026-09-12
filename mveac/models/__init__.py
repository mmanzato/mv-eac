"""Base recommendation models: Most Popular, ItemKNN, and NRMS.

Every model exposes a ``score_all_impressions(behaviors, ...) -> {impression_id:
[(article_id, score), ...] sorted by descending score}`` method. This is the
common interface the calibration and EAC rerankers (``mveac.calibration``)
consume as their "Original" (uncalibrated) input.
"""
