import json
import base64
import tempfile
import unittest
from pathlib import Path

from rendered_snapshot import CLEANUP_SCRIPT, build_backend_marker_map, capture_rendered_snapshot


class _DOM:
    async def getDocument(self, params, *, session_id):
        return {
            "root": {
                "backendNodeId": 1,
                "attributes": ["data-gnn-node-id", "0"],
                "children": [
                    {
                        "backendNodeId": 2,
                        "attributes": ["class", "hero", "data-gnn-node-id", "1"],
                    }
                ],
            }
        }


class _Accessibility:
    async def getFullAXTree(self, params, *, session_id):
        return {"nodes": [{"nodeId": "ax-1", "backendDOMNodeId": 2}]}


class _CDPPage:
    async def captureScreenshot(self, params, *, session_id):
        assert params == {"format": "png", "captureBeyondViewport": True}
        assert session_id == "session-1"
        return {"data": base64.b64encode(b"PNG").decode("ascii")}


class _Send:
    DOM = _DOM()
    Accessibility = _Accessibility()
    Page = _CDPPage()


class _Client:
    send = _Send()


class _Page:
    _client = _Client()

    def __init__(self):
        self.cleaned = False

    async def _ensure_session(self):
        return "session-1"

    async def evaluate(self, script):
        if script == CLEANUP_SCRIPT:
            self.cleaned = True
            return None
        return {
            "html": '<html data-gnn-node-id="0"><body data-gnn-node-id="1"></body></html>',
            "visual": {
                "version": 1,
                "captured_url": "https://example.test/",
                "viewport": {"width": 800, "height": 600},
                "nodes": [],
            },
        }


class SnapshotTests(unittest.IsolatedAsyncioTestCase):
    def test_backend_marker_mapping(self):
        mapping = build_backend_marker_map(
            {
                "backendNodeId": 7,
                "attributes": ["data-gnn-node-id", "3"],
                "shadowRoots": [
                    {"backendNodeId": 8, "attributes": ["data-gnn-node-id", "4"]}
                ],
            }
        )
        self.assertEqual(mapping, {"7": "3", "8": "4"})

    async def test_writes_complete_aligned_bundle_and_cleans_markers(self):
        page = _Page()
        with tempfile.TemporaryDirectory() as directory:
            html_path = Path(directory) / "0.html"
            paths = await capture_rendered_snapshot(page, html_path)

            self.assertEqual(set(paths), {"html", "visual", "ax", "screenshot"})
            for path in paths.values():
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)
            payload = json.loads(paths["ax"].read_text(encoding="utf-8"))
            self.assertEqual(payload["backend_dom_to_snapshot_node"], {"1": "0", "2": "1"})
            self.assertEqual(payload["mapping_stats"]["ax_nodes_mapped_to_snapshot"], 1)
            self.assertTrue(page.cleaned)


if __name__ == "__main__":
    unittest.main()
