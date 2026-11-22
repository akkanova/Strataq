"""strata_server/test_runconfig.py - #564: the web page's Settings view of the run config (GET / POST /config), against the
mock engine: only the listed keys change, every other key of the file stays, a bad value changes nothing, and only
Strata's own page may write it (JSON, no foreign Origin, the API key when one is set).

    python -m unittest strata_server.test_runconfig
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from strata_server import runconfig
from strata_server.frontend import ChatTemplate
from strata_server.server import ByteTokenizer, Service, serve
from strata_common.engine_client import MockEngine

ROOT = Path(__file__).resolve().parents[1]
CFG = {"exe": "engine/strata.exe", "args": ["--pack", "data/packs/q2_0", "--kv", "int8", "--spec-min-p", "0.5"],
       "model_name": "qwen3.8-flash-next-q2_0", "port": 8080, "host": "0.0.0.0", "api_key": "",
       "sampling": {"temperature": 0.7, "presence_penalty": 1.5},
       "mcp_servers": {"files": {"command": "npx", "args": ["-y", "server-filesystem", "."]}},
       "cors_origins": ["http://localhost:3000"]}


class Apply(unittest.TestCase):
    def test_stale_conversation_age_is_saved_independently_of_residency_capacity(self):
        values = runconfig.check_live({"conversation_idle_seconds": 66}, True)
        new, changed = runconfig.apply_live({"args": ["--vram-conversations", "3"]}, values)
        self.assertEqual(new["args"], ["--vram-conversations", "3", "--conversation-idle-seconds", "66"])
        self.assertEqual(changed, ["conversation_idle_seconds"])
        with self.assertRaises(ValueError):
            runconfig.check_live({"conversation_idle_seconds": -1}, True)

    def test_prompt_read_sharing_is_saved_and_takes_whole_numbers(self):
        values = runconfig.check_live({"decode_share_percent": 0, "prompt_turn_ms": 20000}, True)
        new, changed = runconfig.apply_live({"args": []}, values)
        self.assertEqual(new["args"], ["--decode-share-percent", "0", "--prompt-turn-ms", "20000"])
        self.assertEqual(changed, ["decode_share_percent", "prompt_turn_ms"])
        with self.assertRaises(ValueError):
            runconfig.check_live({"prompt_turn_ms": -1}, True)

    def test_batch_groups_are_live_and_saved_with_0_for_auto(self):
        values = runconfig.check_live({"batch_groups": 0}, True)
        new, changed = runconfig.apply_live({"args": ["--batch-groups", "auto"]}, values)
        self.assertEqual((new["args"], changed), (["--batch-groups", "0"], ["batch_groups"]))
        with self.assertRaises(ValueError):
            runconfig.check_live({"batch_groups": -1}, True)

    def test_expert_and_draft_tunings_are_live_and_saved_as_their_flags(self):
        values = runconfig.check_live({"route_tail_skip": 0, "draft_confidence_pct": 40, "draft_cumulative_pct": 15,
                                       "expert_half_life": 50, "copy_back": "off", "route_prior_tenths": 15}, True)
        new, changed = runconfig.apply_live({"args": ["--copy-back", "auto"]}, values)
        self.assertEqual(new["args"], ["--copy-back", "off", "--route-tail-skip", "0", "--draft-confidence-pct", "40",
                                       "--draft-cumulative-pct", "15", "--expert-half-life", "50",
                                       "--route-prior-tenths", "15"])
        self.assertEqual(len(changed), 6)
        for bad in ({"route_tail_skip": 10}, {"draft_confidence_pct": 101}, {"copy_back": True},
                    {"copy_back": "sometimes"}, {"expert_half_life": -1}, {"expert_half_life": 2147483648},
                    {"route_prior_tenths": 101}):
            with self.assertRaises(ValueError):
                runconfig.check_live(bad, True)

    def test_step_profile_is_a_switch_saved_as_on_or_off(self):
        new, changed = runconfig.apply_live({"args": ["--step-profile", "1"]},
                                            runconfig.check_live({"step_profile": False}, True))
        self.assertEqual((new["args"], changed), (["--step-profile", "off"], ["step_profile"]))

    def test_saving_conversations_on_shutdown_is_a_switch_saved_as_on_or_off(self):
        values = runconfig.check_live({"save_on_shutdown": True}, True)
        new, changed = runconfig.apply_live({"args": []}, values)
        self.assertEqual(new["args"], ["--save-on-shutdown", "on"])
        self.assertEqual(changed, ["save_on_shutdown"])
        with self.assertRaises(ValueError):
            runconfig.check_live({"save_on_shutdown": 1}, True)

    def test_the_reserve_is_live_and_saved_as_the_flag_both_new_engines_take(self):
        values = runconfig.check_live({"vram_reserve_mib": 1024}, True)
        new, changed = runconfig.apply_live({"args": ["--vram-reserve-mib", "700"]}, values)
        self.assertEqual((new["args"], changed), (["--vram-reserve-mib", "1024"], ["vram_reserve_mib"]))
        self.assertEqual(runconfig.read_live_defaults(None)["vram_reserve_mib"], None)
        with self.assertRaises(ValueError):
            runconfig.check_live({"vram_reserve_mib": -1}, True)
        with self.assertRaises(ValueError):                              # the Settings view leaves it to the panel
            runconfig.apply(CFG, {"vram_reserve_mib": 2048})

    def test_saving_the_reserve_replaces_its_deepseek_alias(self):
        cfg = {"args": ["--vram-reserve-mib", "700", "--reserve-mib", "600"],
               "user_args": ["--reserve-mib", "600"]}
        self.assertEqual(runconfig._arg(cfg, "--vram-reserve-mib"), 600)
        new, changed = runconfig.apply_live(cfg, {"vram_reserve_mib": 1024})
        self.assertEqual(changed, ["vram_reserve_mib"])
        self.assertEqual(new["args"], ["--vram-reserve-mib", "1024"])
        self.assertEqual(new["user_args"], ["--vram-reserve-mib", "1024"])
        self.assertEqual(runconfig._arg(new, "--vram-reserve-mib"), 1024)

    def test_shared_memory_holds_for_one_change_and_is_never_saved(self):
        values = runconfig.check_live({"memory_mib.gpu0": 30720, "shared_memory": True}, True, ["memory_mib.gpu0"])
        self.assertEqual(values, {"memory_mib.gpu0": 30720, "shared_memory": "on"})
        new, changed = runconfig.apply_live({"args": []}, values, {"memory_mib.gpu0": 30208})
        self.assertEqual((new["args"], changed), (["--gpu-mib", "30720"], ["memory_mib.gpu0"]))

    def test_live_residency_default_survives_a_restart_without_changing_concurrency(self):
        values = runconfig.check_live({"vram_conversations": 3}, True)
        new, changed = runconfig.apply_live({"args": ["--concurrent", "2"]}, values)
        self.assertEqual(new["args"], ["--concurrent", "2", "--vram-conversations", "3"])
        self.assertEqual(changed, ["vram_conversations"])
        with self.assertRaises(ValueError):
            runconfig.check_live({"vram_conversations": 0}, True)

    def test_live_token_defaults_keep_the_loaded_capacity_separate(self):
        cfg = {"args": ["--max-context", "204800", "--tokens-max", "401000"]}
        values = runconfig.check_live({"context": 180000, "tokens_max": 350000}, True)
        new, changed = runconfig.apply_live(cfg, values)
        self.assertEqual(changed, ["context", "tokens_max"])
        self.assertEqual(new["args"], ["--max-context", "204800", "--tokens-max", "350000",
                                       "--conversation-tokens", "180000"])
        self.assertEqual(cfg["args"], ["--max-context", "204800", "--tokens-max", "401000"])
        self.assertEqual(runconfig._arg(new, "--conversation-tokens"), 180000)

    def test_only_the_named_keys_change(self):
        new, changed = runconfig.apply(CFG, {"sampling.top_p": 0.95, "reasoning_budget_tokens": 4096,
                                             "aliases": "qwen, local", "lazy_load": True})
        self.assertEqual(changed, ["sampling.top_p", "reasoning_budget_tokens", "aliases", "lazy_load"])
        self.assertEqual(new["sampling"], {"temperature": 0.7, "presence_penalty": 1.5, "top_p": 0.95})
        self.assertEqual(new["args"], CFG["args"])
        self.assertEqual(new["aliases"], ["qwen", "local"])
        for k in ("exe", "model_name", "port", "host", "api_key", "mcp_servers", "cors_origins"):
            self.assertEqual(new[k], CFG[k], k)
        self.assertEqual(CFG["sampling"], {"temperature": 0.7, "presence_penalty": 1.5})   # the input is left alone

    def test_null_is_the_default_again(self):
        cfg, _ = runconfig.apply(CFG, {"reasoning_budget_tokens": 2048, "fit_max_tokens": True})
        new, changed = runconfig.apply(cfg, {"reasoning_budget_tokens": None, "fit_max_tokens": None,
                                             "sampling.temperature": None, "idle_unload_s": None})
        self.assertEqual(changed, ["reasoning_budget_tokens", "fit_max_tokens", "sampling.temperature"])
        self.assertNotIn("reasoning_budget_tokens", new)
        self.assertNotIn("fit_max_tokens", new)
        self.assertEqual(new["sampling"], {"presence_penalty": 1.5})
        new, _ = runconfig.apply({"sampling": {"top_k": 20}}, {"sampling.top_k": None})
        self.assertNotIn("sampling", new)                                 # an empty block goes

    def test_refused(self):
        for changes in ({"api_key": "x"}, {"host": "0.0.0.0"}, {"mcp_servers": {}}, {"before_load": "calc.exe"},
                        {"args": []}, {"exe": "evil.exe"}, {"trusted_origins": ["*"]}, {"allowed_hosts": ["*"]},
                        {"sampling.temperature": -1}, {"sampling.top_p": 0}, {"sampling.top_k": 65},
                        {"sampling.min_p": 2}, {"reasoning_budget_tokens": 1.5}, {"fit_max_tokens": "yes"},
                        {"effort_position": "middle"}, {"reasoning_budget_tokens": -5}, {"idle_unload_s": True},
                        {"aliases": [1]}, {}, None, ["x"]):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                runconfig.apply(CFG, changes)
        with self.assertRaises(ValueError):                              # one bad value: nothing is applied
            runconfig.apply(CFG, {"fit_max_tokens": True, "sampling.top_k": 0})
        with self.assertRaises(ValueError):
            runconfig.apply({**CFG, "vision": {"gpu": True}}, {"lazy_load": True})

    def test_view(self):
        v = runconfig.view({**CFG, "args": CFG["args"] + ["--vram-reserve-mib", "1500"]}, "x/strata-q2_0.json")
        self.assertEqual(v["file"], "strata-q2_0.json")
        got = {k["key"]: k for k in v["keys"]}
        self.assertEqual(got["sampling.temperature"]["value"], 0.7)
        self.assertNotIn("vram_reserve_mib", got)                        # the Memory panel's (LIVE_FLAGS)
        self.assertIsNone(got["fit_max_tokens"]["value"])
        self.assertEqual(got["anthropic_thinking"]["choices"], ["model", "on_request"])
        self.assertNotIn("api_key", got)
        self.assertNotIn("mcp_servers", got)


class Http(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "strata-q2_0.json"
        self.path.write_text(json.dumps(CFG, indent=1), encoding="utf-8")
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate(ROOT / "strata_server/chat_template.jinja"))
        self.svc.config_path = str(self.path)
        self.httpd = serve(self.svc, port=0)
        self.host = f"127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.dir.cleanup()

    def call(self, method, body=None, headers=None):
        h = {"Content-Type": "application/json", **(headers or {})}
        req = urllib.request.Request(f"http://{self.host}/config", method=method, headers=h,
                                     data=None if body is None else (body if isinstance(body, bytes)
                                                                     else json.dumps(body).encode()))
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def saved(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def test_read_and_write_from_the_page(self):
        status, v = self.call("GET")
        self.assertEqual(status, 200, v)
        self.assertEqual({k["key"]: k["value"] for k in v["keys"]}["sampling.temperature"], 0.7)
        with contextlib.redirect_stdout(io.StringIO()):
            status, v = self.call("POST", {"set": {"sampling.temperature": 1.0, "open_browser": False}},
                                  {"Origin": f"http://{self.host}"})
        self.assertEqual(status, 200, v)
        self.assertEqual(v["changed"], ["sampling.temperature", "open_browser"])
        cfg = self.saved()
        self.assertEqual(cfg["sampling"]["temperature"], 1.0)
        self.assertIs(cfg["open_browser"], False)
        for k in ("mcp_servers", "cors_origins", "host", "args", "exe"):
            self.assertEqual(cfg[k], CFG[k], k)
        self.assertEqual(json.loads((self.path.parent / "strata-q2_0.json.bak").read_text(encoding="utf-8")), CFG)
        status, v = self.call("POST", {"set": {"open_browser": False}})   # no Origin (a script on this PC): allowed
        self.assertEqual((status, v["changed"]), (200, []))

    def test_only_from_strata_s_own_page(self):
        before = self.path.read_bytes()
        status, _ = self.call("POST", {"set": {"fit_max_tokens": True}}, {"Origin": "https://evil.example"})
        self.assertEqual(status, 403)
        status, _ = self.call("POST", b'{"set": {"fit_max_tokens": true}}', {"Content-Type": "text/plain"})
        self.assertEqual(status, 415)
        for body in ({"set": {"api_key": ""}}, {"set": {"mcp_servers": {"x": {"command": "calc"}}}},
                     {"set": {"sampling.top_k": 0}}, {"nothing": 1}, b"not json"):
            status, b = self.call("POST", body, {"Origin": f"http://{self.host}"})
            self.assertEqual(status, 400, (body, b))
        self.assertEqual(self.path.read_bytes(), before)                 # nothing was written
        self.assertFalse((self.path.parent / "strata-q2_0.json.bak").exists())

    def test_the_api_key(self):
        self.svc.api_key = "secret"
        self.assertEqual(self.call("GET")[0], 401)
        self.assertEqual(self.call("POST", {"set": {"fit_max_tokens": True}})[0], 401)
        with contextlib.redirect_stdout(io.StringIO()):
            status, _ = self.call("POST", {"set": {"fit_max_tokens": True}}, {"Authorization": "Bearer secret"})
        self.assertEqual(status, 200)
        self.assertIs(self.saved()["fit_max_tokens"], True)

    def test_without_a_run_config(self):
        self.svc.config_path = None
        self.assertEqual(self.call("GET")[0], 404)
        self.assertEqual(self.call("POST", {"set": {"fit_max_tokens": True}})[0], 404)

    def test_the_page_has_the_view(self):
        web = ROOT / "strata_server" / "web"
        html, js = (web / "index.html").read_text(encoding="utf-8"), (web / "app.js").read_text(encoding="utf-8")
        for el in ("cfg-card", "cfg-form", "cfg-save", "cfg-msg"):
            self.assertIn(f'id="{el}"', html)
        self.assertIn('fetch("config", {method: "POST", headers: headers(true)', js)


if __name__ == "__main__":
    unittest.main()
