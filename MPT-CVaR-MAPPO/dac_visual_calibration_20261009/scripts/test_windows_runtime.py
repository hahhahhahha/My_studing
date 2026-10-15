from windows_runtime import initialize_windows_runtime

initialize_windows_runtime()

import os
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.name == "nt", "Windows runtime paths only")
class RuntimeTests(unittest.TestCase):
    def test_missing_paths_recovered(self):
        names = ("SystemRoot", "windir", "ProgramFiles", "ProgramW6432",
                 "USERNAME", "TEMP", "TMP")
        names_upper = {name.upper() for name in names}
        stripped = {key: value for key, value in os.environ.items() if key.upper() not in names_upper}
        with patch.dict(os.environ, stripped, clear=True):
            added = initialize_windows_runtime()
            self.assertEqual(set(added), set(names))
            self.assertTrue(os.path.isdir(added["SystemRoot"]))
            self.assertTrue(os.path.isdir(added["ProgramFiles"]))
            self.assertTrue(os.path.isdir(added["TEMP"]))
            self.assertTrue(added["USERNAME"])
            self.assertEqual(initialize_windows_runtime(), {})

    def test_existing_values_preserved(self):
        names = ("SystemRoot", "windir", "ProgramFiles", "ProgramW6432",
                 "USERNAME", "TEMP", "TMP")
        existing = {name: f"existing_{name}" for name in names}
        with patch.dict(os.environ, existing):
            self.assertEqual(initialize_windows_runtime(), {})
            self.assertEqual({name: os.environ[name] for name in names}, existing)


if __name__ == "__main__":
    unittest.main()
