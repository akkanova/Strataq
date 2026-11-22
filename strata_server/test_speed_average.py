"""Checks the Monitor's moving speed averages and the API monitor's and Terminal's run averages.

The run-average tests originated in keinebenutzer/STRATA-average-patch (MIT, v0.1.55 c62d5e3a).
Browser arithmetic runs in Node without a browser or network.
"""
import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

WEB = Path(__file__).resolve().parent / "web"
NODE = shutil.which("node")


def run_node(source: str) -> object:
    done = subprocess.run([NODE, "-e", source], capture_output=True, text=True, encoding="utf-8", timeout=60, check=True)   # node writes UTF-8; the locale code page (cp1252, cp932 ...) is not what it speaks
    return json.loads(done.stdout)


def function_source(text: str, name: str) -> str:
    """One top-level function of a browser script, by its opening line to its closing brace in column 0."""
    start = text.index(f"function {name}(")
    end = text.index("\n}\n", start) + 3
    return text[start:end]


def cards_block(text: str) -> str:
    """The code that makes the Generation and Prefill cards: their moving averages and what the cards show."""
    start = text.index("class FadingSums {")
    return "\n".join([text[start:text.index("\n}\n", start) + 3],
                      *(function_source(text, name) for name in ("trackSpeeds", "showSpeeds", "describeDecode",
                                                                  "describePrefill"))])


@unittest.skipUnless(NODE, "node is needed to run the page's own code")
class MonitorTab(unittest.TestCase):
    """strata_server/web/app.js: the two cards, run with stand-ins for the page's helpers."""

    def polls(self, polls, last=None, totals=None, wait_ms=0):
        """Runs the cards' code over /metrics `polls` (live objects) `wait_ms` apart; returns each poll's cards, by
        key: [value, sub, idle, tooltip, right-hand lines]."""
        block = cards_block((WEB / "app.js").read_text(encoding="utf-8"))
        script = ("let shown = {}; const fmt = (n, d = 0) => Number(n).toFixed(d);\n"
                  "const setMetric = (k, v, unit, sub, o = {}) => { shown[k] = [v, sub, !!o.idle, o.tip || \"\"]; };\n"
                  "const setSide = (k, lines) => shown[k].push(lines.map((l) => l[0]));\n"
                  f"{block}\nconst speeds = {{gen: new FadingSums(), decode: new FadingSums(), "
                  "prefill: new FadingSums(), stats: null, at: null};\nconst out = [];\nlet clock = 0; performance.now = () => clock;\n"
                  f"for (const live of {json.dumps(polls)}) {{ shown = {{}}; showSpeeds(live, {json.dumps(last)}, "
                  f"{json.dumps(totals)}, trackSpeeds(live), 8); out.push(shown); clock += {wait_ms}; }}\n"
                  "console.log(JSON.stringify(out));")
        return run_node(script)

    def cards(self, live, last, totals):
        return {k: (v[0], v[1]) for k, v in self.polls([live], last, totals)[0].items()}

    TOTALS = {"requests": 12, "prompt_tokens": 90000, "reused": 5000, "output_tokens": 40000, "prompt_ms": 120000.0, "decode_ms": 1000000.0}
    LAST = {"decode_tok_s": 50.5, "prompt_ms": 2000.0, "prompt_tokens": 1000, "reused": 0}

    def test_idle_shows_the_run_average_for_both_cards(self):
        cards = self.cards({"state": "idle"}, self.LAST, self.TOTALS)
        self.assertEqual(cards["gen"], ("40.0", "Run average"))
        self.assertEqual(cards["prefill"], ("708", "Run average"), "(90000 - 5000 reused) tokens in 120 s")

    def test_live_values_stay_live(self):
        cards = self.cards({"state": "generating", "tok_s": 31.5, "prefill_tok_s_mean": 600}, self.LAST, self.TOTALS)
        self.assertEqual(cards["gen"], ("31.5", "Now"))
        self.assertEqual(cards["prefill"], ("708", "Run average"), "no prompt is read: the past, in grey")
        self.assertEqual(self.cards({"state": "reading", "prefill_tok_s_mean": 820}, self.LAST, self.TOTALS)["prefill"],
                         ("820", "Now"))

    def test_decode_stays_live_beside_a_prompt_read(self):
        cards = self.cards({"state": "reading", "tok_s": 12.0, "prefill_tok_s_mean": 820}, self.LAST, self.TOTALS)
        self.assertEqual(cards["gen"], ("12.0", "Now"))
        self.assertEqual(cards["prefill"], ("820", "Now"))

    def test_after_the_work_the_card_greys_and_keeps_the_recent_average(self):
        polls = self.polls([{"state": "generating", "tok_s": 30.0}, {"state": "generating", "tok_s": 20.0},
                            {"state": "idle"}], self.LAST, self.TOTALS, wait_ms=1000)
        self.assertEqual(polls[1]["gen"][:3], ["20.0", "Now", False], "the first poll has no interval to weigh")
        self.assertEqual(polls[2]["gen"][:4], ["20.0", "Recent average", True, "Recent average: 20.0 tokens/sec"])
        polls = self.polls([{"state": "generating", "tok_s": 30.0}, {"state": "generating", "tok_s": 30.0},
                            {"state": "generating", "tok_s": 20.0}, {"state": "idle"}], wait_ms=1000)
        self.assertEqual(polls[2]["gen"][0], "23.8", "60% of the older second is kept: (30 * .6 + 20) / 1.6")
        self.assertEqual(polls[3]["gen"][0], "24.9", "96% of it is kept: (30 * .96 + 20) / 1.96")

    @staticmethod
    def stats(seconds, steps, rows, experts, routings, cpu, guesses, kept, tokens, ms):
        return {"seconds": seconds, "gpus": [], "prefill": {"tokens": tokens, "ms": ms},
                "decode": {"steps": steps, "rows": rows, "routed_rows": rows, "routings": routings,
                           "experts": experts, "cpu_experts": cpu, "guesses": guesses, "kept": kept}}

    def test_a_paged_engine_s_totals_give_the_step_lines_and_a_prefill_rate_without_decode_turns(self):
        first = self.stats(10, 100, 200, 90000, 105600, 0, 100, 50, 4000, 3000)
        second = self.stats(11, 110, 226, 102350, 119328, 0, 116, 59, 5000, 3500)
        polls = self.polls([{"state": "reading", "tok_s": 20.0, "engine_stats": first},
                            {"state": "reading", "tok_s": 20.0, "engine_stats": second}], wait_ms=1000)
        gen, prefill = polls[1]["gen"], polls[1]["prefill"]
        self.assertEqual(gen[4], ["2.6 tokens/step", "475 experts/token", "0 on CPU, 10% PRES", "56% MTP acceptance"],
                         "26 rows in 10 steps; 12350 experts for 26 rows that alone use 528 each; 9 of 16 kept")
        self.assertEqual(prefill[:2], ["2000", "Now"], "1000 tokens in the 500 ms spent reading, not in the second")
        self.assertIn("50% of time reading", prefill[4])

    def test_an_engine_restart_is_not_a_negative_interval(self):
        polls = self.polls([{"state": "generating", "tok_s": 5.0,
                             "engine_stats": self.stats(49, 8, 8, 8, 8, 8, 8, 8, 8, 8)},
                            {"state": "generating", "tok_s": 5.0,
                             "engine_stats": self.stats(50, 9, 9, 9, 9, 9, 9, 9, 9, 9)},
                            {"state": "generating", "tok_s": 5.0, "engine_stats": self.stats(1, 1, 1, 1, 1, 0, 0, 0, 0, 0)}],
                           wait_ms=1000)
        self.assertTrue(polls[1]["gen"][4])
        self.assertEqual(polls[2]["gen"][4], [])
        self.assertIsNone(polls[2]["prefill"][0])

    def test_the_last_prefill_interval_updates_the_average_without_keeping_the_card_live(self):
        polls = self.polls([{"state": "reading", "engine_stats": self.stats(10, 0, 0, 0, 0, 0, 0, 0, 1000, 1000)},
                            {"state": "generating", "tok_s": 20.0,
                             "engine_stats": self.stats(11, 1, 1, 1, 1, 0, 0, 0, 2000, 1500)}], wait_ms=1000)
        self.assertEqual(polls[1]["prefill"][:3], ["2000", "Recent average", True])

    def test_repainting_the_same_poll_does_not_add_another_speed_sample(self):
        block = cards_block((WEB / "app.js").read_text(encoding="utf-8"))
        script = (f"{block}\nconst speeds = {{gen: new FadingSums(), decode: new FadingSums(), "
                  "prefill: new FadingSums(), stats: null, at: null};\n"
                  "let clock = 0; performance.now = () => clock;\n"
                  "trackSpeeds({state: 'generating', tok_s: 30}); clock = 1000;\n"
                  "const live = {state: 'generating', tok_s: 30}; trackSpeeds(live);\n"
                  "clock = 2000; trackSpeeds(live);\n"
                  "console.log(JSON.stringify(speeds.gen.now.seconds));")
        self.assertEqual(run_node(script), 1)

    def test_without_totals_it_is_the_last_request_and_never_an_invented_zero(self):
        for totals in (None, {}, {"output_tokens": 0, "decode_ms": 0, "prompt_tokens": 0, "prompt_ms": 0}):
            cards = self.cards({"state": "idle"}, self.LAST, totals)
            self.assertEqual(cards["gen"], ("50.5", "Last request"), totals)
            self.assertEqual(cards["prefill"], ("500", "Last request"), totals)
        cards = self.cards({"state": "idle"}, None, None)
        self.assertEqual(cards["gen"], (None, ""))
        self.assertEqual(cards["prefill"], (None, ""))


@unittest.skipUnless(NODE, "node is needed to run the page's own code")
class MonitorPage(unittest.TestCase):
    def average(self, rows):
        source = function_source((WEB / "monitor.js").read_text(encoding="utf-8"), "averageDecodeSpeed")
        return run_node(f"{source}\nconsole.log(JSON.stringify(averageDecodeSpeed({json.dumps(rows)})));")

    def test_it_is_token_weighted_over_the_finished_requests(self):
        long_one = {"timings": {"predicted_n": 19511, "predicted_ms": 386000}}
        short_one = {"timings": {"predicted_n": 11, "predicted_ms": 410}}
        self.assertAlmostEqual(self.average([long_one, short_one]), (19511 + 11) / (386410 / 1000), places=6)
        self.assertGreater(self.average([long_one, short_one]), 50, "a plain mean of the two (about 38) would let the short one count as much as the long one")

    def test_requests_without_timings_or_with_one_token_are_left_out(self):
        self.assertIsNone(self.average([]))
        self.assertIsNone(self.average([{}, {"timings": None}, {"timings": {"predicted_n": 1, "predicted_ms": 10}}, {"timings": {"predicted_n": 5, "predicted_ms": 0}},
                                        {"timings": {"predicted_n": "9", "predicted_ms": 5}}]))
        self.assertEqual(self.average([{"timings": {"predicted_n": 100, "predicted_ms": 2000}}, {}]), 50)

    def test_the_card_says_what_it_shows(self):
        html = (WEB / "monitor.html").read_text(encoding="utf-8")
        self.assertIn("Average decode speed", html)
        self.assertIn("Run average", html)
        self.assertNotIn("Last decode speed", html)
        js = (WEB / "monitor.js").read_text(encoding="utf-8")
        self.assertIn('text("speed", speed(averageDecodeSpeed(records.filter(r => !active(r)))))', js)


class AddedCodeIsDisplayOnly(unittest.TestCase):
    """What the patch added reads numbers the page already has and writes text with textContent: no request, no eval, no markup, no address."""

    def test_nothing_but_arithmetic_and_text(self):
        pieces = [cards_block((WEB / "app.js").read_text(encoding="utf-8")), function_source((WEB / "monitor.js").read_text(encoding="utf-8"), "averageDecodeSpeed"),
                  (Path(__file__).resolve().parents[1] / "strata_cli" / "overview.py").read_text(encoding="utf-8")]
        pieces[2] = pieces[2][pieces[2].index("def run_average_speed"):pieces[2].index("def _cards")]
        forbidden = re.compile(r"\b(fetch|XMLHttpRequest|WebSocket|sendBeacon|eval|Function|import|require|innerHTML|insertAdjacentHTML|document\.write|"
                               r"subprocess|socket|urllib|requests|os\.system|open)\s*\(|https?://|\.src\s*=|new Image")
        for piece in pieces:
            self.assertIsNone(forbidden.search(piece), forbidden.search(piece) and forbidden.search(piece).group(0))
        self.assertIn("textContent", (WEB / "monitor.js").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
