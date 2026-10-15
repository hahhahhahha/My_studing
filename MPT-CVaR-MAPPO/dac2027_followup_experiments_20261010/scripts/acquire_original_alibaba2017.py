"""Source-path repair in a new attempt; original official SHA gate unchanged."""
from datetime import datetime
from pathlib import Path
import acquire_alibaba2017 as original

if __name__ == '__main__':
    folder = original.BASE / 'data_acquisition/original2017_attempt_20261010'
    folder.mkdir(parents=True, exist_ok=True)
    probe = original.BASE / 'data_acquisition/original_source_probe_20261010_233159_417364.json'
    failed = original.BASE / 'data_acquisition/DOWNLOAD_STATE.json'
    # This preserves the earlier archive/worker/source. The historical HTTPS
    # endpoint was read-only probed, not selected using any policy outcomes.
    original.save(folder / 'SOURCE_PATH_REPAIR.json', dict(
        at=datetime.now().astimezone().isoformat(),
        wrapper_sha256=original.sha(Path(__file__)),
        acquisition_source_sha256=original.sha(Path(original.__file__)),
        historical_failure_sha256=original.sha(failed),
        probe_sha256=original.sha(probe),
        url='https://clusterdata.oss-cn-beijing.aliyuncs.com/trace_201708.tgz',
        approved_HEAD_bytes=243323702,
        expected_sha256=original.EXPECTED,
        checksum_gate_changed=False, dataset_release_changed=False,
        prior_failed_archive_preserved=True), exclusive=True)
    original.FOLDER = folder
    original.STATE = folder / 'DOWNLOAD_STATE.json'
    original.LOCK = folder / 'DOWNLOAD_STARTED.json'
    original.URL = 'https://clusterdata.oss-cn-beijing.aliyuncs.com/trace_201708.tgz'
    original.EXPECTED_BYTES = 243323702
    original.main()
