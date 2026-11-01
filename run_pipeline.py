"""Run the full MLOps pipeline end to end: ingest -> preprocess -> features -> train -> monitor -> evaluate -> export."""
import time

from src import data_ingestion, evaluate, export_web, features, monitor, preprocess, train
from src.utils import get_logger

log = get_logger("pipeline")

STAGES = [
    ("ingestion", data_ingestion.main),
    ("preprocess", preprocess.main),
    ("features", features.main),
    ("train", train.main),
    ("monitor", monitor.main),
    ("evaluate", evaluate.main),
    ("export_web", export_web.main),
]

if __name__ == "__main__":
    for name, fn in STAGES:
        t0 = time.perf_counter()
        log.info("=== Stage: %s ===", name)
        fn()
        log.info("Stage %s finished in %.1fs", name, time.perf_counter() - t0)
