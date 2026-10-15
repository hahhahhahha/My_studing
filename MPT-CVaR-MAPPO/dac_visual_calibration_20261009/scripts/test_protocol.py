import unittest
from common import safe_member, split_images, validate_roles, session_blocks, cross_role_near_duplicates


def fixture():
    rows = []
    for cls in range(10):
        for official, count in (("train", 300), ("val", 120)):
            for image in range(count):
                key = f"imagenette2/{official}/class{cls}/{image}.JPEG"
                rows.append({"image_id": key, "image_sha256": key, "class_id": str(cls), "official_split": official})
    return rows


class ProtocolTests(unittest.TestCase):
    def test_archive_paths(self):
        self.assertEqual(safe_member("imagenette2/train/class/file.JPEG"), "imagenette2/train/class/file.JPEG")
        for name in ("../bad", "imagenette2/../bad", "/imagenette2/bad", "imagenette2/C:/bad"):
            with self.assertRaises(ValueError):
                safe_member(name)

    def test_disjoint_stratified_counts(self):
        roles, _ = split_images(fixture())
        self.assertEqual({k: len(v) for k, v in roles.items()}, {"pilot": 100, "calibration": 2000, "validation": 500, "test": 1000})
        hashes = [r["image_sha256"] for rows in roles.values() for r in rows]
        self.assertEqual(len(hashes), len(set(hashes)))

    def test_reproducibility(self):
        self.assertEqual(split_images(fixture()), split_images(fixture()))

    def test_each_block_uses_paired_order(self):
        roles, _ = split_images(fixture())
        jobs = session_blocks(roles["validation"], "validation", 1)
        self.assertEqual(len(jobs), 6)
        for a, b in zip(jobs[::2], jobs[1::2]):
            self.assertEqual(a["image_ids"], b["image_ids"])
            self.assertNotEqual(a["profile"], b["profile"])

    def test_session_order_counterbalanced(self):
        roles, _ = split_images(fixture())
        self.assertNotEqual(session_blocks(roles["calibration"], "calibration", 1)[0]["profile"],
                            session_blocks(roles["calibration"], "calibration", 2)[0]["profile"])

    def test_near_duplicate_cross_roles_detected(self):
        hits = cross_role_near_duplicates({"calibration": [{"image_id": "a", "dhash64": "0000000000000000"}],
                                           "test": [{"image_id": "b", "dhash64": "0000000000000003"}]})
        self.assertEqual(len(hits), 1)

    def test_sessions_guarded(self):
        with self.assertRaises(ValueError):
            session_blocks([], "test", 6)

    def test_frozen_roles_revalidate(self):
        roles, _ = split_images(fixture())
        validate_roles(roles)
        roles["test"][0]["image_sha256"] = roles["calibration"][0]["image_sha256"]
        with self.assertRaises(ValueError):
            validate_roles(roles)


if __name__ == "__main__":
    unittest.main()
