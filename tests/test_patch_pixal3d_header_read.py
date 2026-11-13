"""Tests for the Windows `post-replay` header read (issue #192).

What matters: the header is read in one call (the operand order of `fread(...)+fread(...)`
is unspecified, and MSVC runs it right to left), a short read now says which dump and
which header it was, nothing else in the file changes, re-running is a no-op, and a moved
anchor is refused rather than guessed.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "patch_pixal3d_header_read.py"
VENDOR_SOURCE = REPO / "vendor" / "pixal3d-cpp" / "src" / "post_replay.cpp"


def _load():
    spec = importlib.util.spec_from_file_location("patch_pixal3d_header_read", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patch = _load()

# `main`'s dump loader as raven38/pixal3d.cpp writes it at the pinned commit c89126d.
ORIGINAL = '''int main(int argc, char** argv) {
    FILE* f = fopen(dump, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\\n", dump); return 1; }
    int V, F, Mv, res;
    if (fread(&V,4,1,f)+fread(&F,4,1,f)+fread(&Mv,4,1,f)+fread(&res,4,1,f) != 4) return 1;
    std::vector<float> verts((size_t)V*3);
    std::vector<int32_t> faces((size_t)F*3);
    std::vector<std::array<int,3>> coords((size_t)Mv);
    std::vector<float> pbr6((size_t)Mv*6);
    if (fread(verts.data(),4,verts.size(),f) != verts.size()) return 1;
    if (fread(faces.data(),4,faces.size(),f) != faces.size()) return 1;
    for (auto& c : coords) if (fread(c.data(),4,3,f) != 3) return 1;
    if (fread(pbr6.data(),4,pbr6.size(),f) != pbr6.size()) return 1;
    fclose(f);
    printf("loaded: V=%d F=%d voxels=%d res=%d\\n", V, F, Mv, res);
    return 0;
}
'''


def test_the_header_is_read_in_one_call():
    """Four freads added together ran right to left on MSVC: Windows read the header backwards."""
    out = patch.apply(ORIGINAL)
    assert "if (fread(hdr,4,4,f) != 4)" in out
    assert not re.search(r"fread\([^)]*\)\s*\+\s*fread", out)
    assert "V = hdr[0]; F = hdr[1]; Mv = hdr[2]; res = hdr[3];" in out


def test_a_short_header_says_it_is_short():
    assert '"%s: header is short\\n"' in patch.apply(ORIGINAL)


def test_a_short_body_names_the_dump_and_its_header():
    """The Windows log was empty, so the exit code was all the Finish had to report."""
    out = patch.apply(ORIGINAL)
    assert '"%s ends early: its header says V=%d F=%d Mv=%d\\n", dump, V, F, Mv' in out
    assert patch.BINARY_MARKER in out


def test_no_body_read_is_still_a_bare_return():
    """All four go through `ok`, so the one that fell short is named rather than ignored."""
    out = patch.apply(ORIGINAL)
    body = out.split("std::vector<float> pbr6((size_t)Mv*6);", 1)[1]
    body = body.split("fclose(f);", 1)[0]
    for line in body.splitlines():
        assert not ("fread(" in line and "return 1" in line), line
    assert 'if (!ok) { fprintf(stderr, "%s ends early' in body


def test_the_rest_of_the_file_is_untouched():
    patched = patch.apply(ORIGINAL)
    for line in ('    std::vector<float> verts((size_t)V*3);',
                 '    fclose(f);',
                 '    printf("loaded: V=%d F=%d voxels=%d res=%d\\n", V, F, Mv, res);'):
        assert line in patched


def test_it_still_returns_one_on_a_failure():
    """Every failure path is a `return 1` out of `main`, as before: the caller reads the code."""
    out = patch.apply(ORIGINAL)
    assert " exit(1)" not in out
    assert out.count("return 1;") == 3


def test_rerunning_changes_nothing():
    once = patch.apply(ORIGINAL)
    assert patch.apply(once) == once
    assert patch.is_patched(once) and not patch.is_patched(ORIGINAL)


@pytest.mark.parametrize("broken", [
    ORIGINAL.replace("fread(&res,4,1,f)", "fread(&res,4,1,fp)"),
    ORIGINAL.replace("for (auto& c : coords) if (fread(c.data(),4,3,f) != 3) return 1;\n", ""),
])
def test_a_moved_anchor_is_refused(broken):
    with pytest.raises(SystemExit, match="anchor"):
        patch.apply(broken)


def test_a_doubled_anchor_is_refused():
    """A second dump reader means upstream grew another path this must not silently miss."""
    with pytest.raises(SystemExit, match="expected 1"):
        patch.apply(ORIGINAL + ORIGINAL.split("int main", 1)[1])


@pytest.mark.skipif(not VENDOR_SOURCE.is_file(), reason="no vendored pixal3d.cpp checkout")
def test_the_vendored_source_takes_the_patch():
    source = VENDOR_SOURCE.read_text(encoding="utf-8")
    assert patch.is_patched(source) or patch.apply(source) != source
