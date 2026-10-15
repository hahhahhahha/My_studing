"""Engineering-stage interface tests; no inference, policy, fitting or test read."""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from execution_service_catalog import ExecutionServiceCatalog
from metadata_service_predictor import MetadataServicePredictor, ObservableImage, PROFILES

BASE = Path(__file__).resolve().parents[1]


def run():
    predictor = MetadataServicePredictor()
    calibration = ExecutionServiceCatalog('calibration')
    validation = ExecutionServiceCatalog('validation')
    if set(calibration.image_ids()) & set(validation.image_ids()):
        raise AssertionError('Calibration/validation image overlap')
    try:
        ExecutionServiceCatalog('test')
    except ValueError:
        test_access_rejected = True
    else:
        raise AssertionError('Engineering entry accessed retained test')
    queries = validation.image_ids()[:100]  # fixed ID order, never timing selection
    largest_difference = 0.
    for image in queries:
        metadata = validation.observable_metadata(image)
        got = predictor.predict_seconds(metadata)
        neighbors, distance = predictor.neighbors(metadata)
        expected = {profile: math.fsum(calibration.measured_pair_seconds(known, session)[profile]
                    for known in neighbors for session in range(1, 6)) / (len(neighbors) * 5)
                    for profile in PROFILES}
        largest_difference = max(largest_difference, *(abs(got[p] - expected[p]) for p in PROFILES))
        if largest_difference > 1e-14 or distance > 2.0:
            raise AssertionError('Metadata predictor differs from frozen calibration pairing')
        # Execution durations are obtained through a different interface, not
        # handed to predict_seconds. Mutating a returned pair/cache copy cannot
        # modify either catalog or predictor.
        pair = validation.measured_pair_seconds(image, 1)
        pair['CPU-1T'] = 1000.
        got['CPU-1T'] = 1000.
        if validation.measured_pair_seconds(image, 1)['CPU-1T'] == 1000. or predictor.predict_seconds(metadata)['CPU-1T'] == 1000.:
            raise AssertionError('Mutable alias across execution/prediction boundary')
    try:
        ObservableImage(width=500, height=500, payload_bytes=100000, service_ms=42.)
    except TypeError:
        actual_duration_argument_rejected = True
    else:
        raise AssertionError('Duration entered observable metadata')
    report = dict(at=datetime.now().astimezone().isoformat(), passed=True,
        calibration_rows=calibration.row_count, validation_rows=validation.row_count,
        query_images=len(queries), model_sha256=predictor.model_sha256,
        source_sha256={**calibration.source_sha256, **validation.source_sha256},
        implementation_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), BASE/'scripts/metadata_service_predictor.py', BASE/'scripts/execution_service_catalog.py')},
        predictor_raw_calibration_max_abs_difference_seconds=largest_difference,
        retained_test_access_rejected=test_access_rejected,
        actual_duration_argument_rejected=actual_duration_argument_rejected,
        policy_training_or_testing=False, model_refitted=False,
        scope='Interface/unit engineering check only; future integrated actor/critic/baseline information flow still needs independent audit.')
    destination = BASE/'protocol'/('VISUAL_SERVICE_INTERFACE_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json')
    with destination.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print('VISUAL_SERVICE_INTERFACE_FOUNDATION_OK', destination, flush=True)


if __name__ == '__main__':
    run()
