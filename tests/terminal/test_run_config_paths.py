"""A run config names the paths inside its install folder, and the data folder beside it, from the config's folder (./x,
../x), so a renamed or moved install still finds its engine: cfg_paths.dumps writes those paths, cfg_paths.read_config
makes them absolute again, and the run script names its own files from its folder.  The person's own keys (sampling,
mcp_servers) and values without ./ or ../ are never rewritten."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from strata_common import cfg_paths  # noqa: E402
import setup  # noqa: E402


class RelativeRunConfigPaths(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.beside = Path(tmp.name)
        self.install = self.beside / "Strata"
        self.data = self.beside / "Strata-data" / "packs" / "iq3_s"      # the default data folder, beside the install
        (self.install / "build").mkdir(parents=True)
        (self.install / "build" / "strata.exe").write_bytes(b"")
        self.data.mkdir(parents=True)
        self.cfg = {
            "exe": str(self.install / "build" / "strata.exe"),
            "args": ["--pack", str(self.data), "--max-context", "8192", "--kv", "int8"],
            "cwd": str(self.install),
            "tokenizer": str(self.data / "tokenizer"),
            "log": str(self.install / "strata-x.log"),
            "lib_dirs": [str(self.install / "build")],
            "env": {"STRATA_TUNING": str(self.install / "tuning.txt"), "STRATA_PF": "1"},
            "sampling": {"temperature": 0.7},
            "mcp_servers": {"files": {"args": [str(self.install / "notes")]}},
        }

    def test_paths_inside_the_install_or_beside_it_are_written_relative_to_it(self):
        out = cfg_paths.to_relative(self.cfg, self.install)
        self.assertEqual(out["exe"], "./build/strata.exe")
        self.assertEqual(out["cwd"], ".")
        self.assertEqual(out["log"], "./strata-x.log")
        self.assertEqual(out["lib_dirs"], ["./build"])
        self.assertEqual(out["args"], ["--pack", "../Strata-data/packs/iq3_s", "--max-context", "8192", "--kv", "int8"])
        self.assertEqual(out["tokenizer"], "../Strata-data/packs/iq3_s/tokenizer")
        self.assertEqual(out["env"], {"STRATA_TUNING": "./tuning.txt", "STRATA_PF": "1"})

    def test_the_persons_own_keys_and_paths_outside_the_parent_are_left_as_they_are(self):
        elsewhere = self.beside.parent / "elsewhere" / "model.gguf"
        out = cfg_paths.to_relative({**self.cfg, "tokenizer": str(elsewhere)}, self.install)
        self.assertEqual(out["sampling"], {"temperature": 0.7})
        self.assertEqual(out["mcp_servers"], self.cfg["mcp_servers"])
        self.assertEqual(out["tokenizer"], str(elsewhere))

    def test_reading_the_written_file_gives_the_same_absolute_paths_back(self):
        path = self.install / "strata-x.json"
        path.write_text(cfg_paths.dumps(self.cfg, path, indent=1), encoding="utf-8")
        self.assertEqual(cfg_paths.read_config(path), self.cfg)

    def test_only_values_marked_with_dot_slash_are_read_as_paths_from_the_folder(self):
        cfg = {"exe": "component:strata-engine", "args": ["auto", "owner/repo", "build", "../x"], "cwd": "."}
        back = cfg_paths.to_absolute(cfg, self.install)
        self.assertEqual(back["exe"], "component:strata-engine")
        self.assertEqual(back["args"], ["auto", "owner/repo", "build", str(self.beside / "x")])
        self.assertEqual(back["cwd"], str(self.install))

    def test_run_script_names_its_config_and_server_from_its_own_folder(self):
        with mock.patch.object(setup, "home", lambda: self.install), mock.patch.object(setup, "ROOT", self.install), \
                mock.patch.object(setup, "WIN", True):
            script = setup.write_run_script("iq3_s", self.install / "strata-iq3_s.json", 8080, True)
        text = script.read_text(encoding="utf-8")
        self.assertIn('cd /d "%~dp0"', text)
        self.assertIn('"%~dp0strata-iq3_s.json"', text)
        self.assertIn('"%~dp0strata_server' + os.sep + 'server.py"', text)
        self.assertNotIn(str(self.install), text)


if __name__ == "__main__":
    unittest.main()
