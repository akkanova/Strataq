"""GET_MEMORY_USE's EXPERT_TRAFFIC lines reach /v1/memory as each expert pool's counters."""
from unittest.mock import patch

from strata_common.engine_client import StrataEngine

REPORT = """MEMORY gpu0 total=1000 free=300 experts=450
MEMORY ram pool_pages=10 allocated_pages=5 fragmented_pages=0
EXPERT_FILL loaded=1 pending=0
EXPERT_TRAFFIC gpu0 hits=7 misses=5 loads=3 evictions=1 swaps=2
EXPERT_TRAFFIC ram hits=1 loads=4 evictions=0 swaps=0 ssd_reads=2 ssd_bytes=4096 ssd_wait_us=150
SETTINGS context=32768"""


def ask_report(text: str) -> dict:
    """Parses `text` as an engine's answer to GET_MEMORY_USE."""
    engine = StrataEngine.__new__(StrataEngine)
    engine.info = {}
    with patch.object(StrataEngine, "_ask", return_value=[line.split() for line in text.splitlines()]):
        return engine.memory()


def test_traffic_is_parsed_per_memory():
    report = ask_report(REPORT)
    assert report["expert_traffic"] == {
        "gpu0": {"hits": 7, "misses": 5, "loads": 3, "evictions": 1, "swaps": 2},
        "ram": {"hits": 1, "loads": 4, "evictions": 0, "swaps": 0, "ssd_reads": 2, "ssd_bytes": 4096,
                "ssd_wait_us": 150}}
    assert [m["id"] for m in report["memories"]] == ["gpu0", "ram"]
    assert report["memories"][0]["uses"] == {"experts": 450}


def test_report_without_traffic_has_no_key():
    assert "expert_traffic" not in ask_report("MEMORY gpu0 experts=1\nSETTINGS")
