"""Orient each separate piece of a mesh outward by its own signed volume.

``trimesh``'s ``fix_normals`` decides inversion globally: on a mesh of several pieces the
volumes cancel, or one inverted piece flips the correct ones. Here every piece is decided
on its own, and only face winding changes.
"""

from __future__ import annotations


def component_labels(mesh):
    """Per-face connected-component labels, via face adjacency."""
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    n = len(mesh.faces)
    adjacency = mesh.face_adjacency
    if len(adjacency) == 0:
        return np.arange(n), n
    graph = sp.coo_matrix(
        (np.ones(len(adjacency)), (adjacency[:, 0], adjacency[:, 1])), shape=(n, n)
    )
    count, labels = connected_components(graph, directed=False)
    return labels, count


def orient_pieces(mesh) -> bool:
    """Orient each piece by its own volume; True if any face was flipped.

    Pieces are found on a position weld, so triangles split at UV seams join their piece.
    The weld is a scratch copy: ``mesh`` keeps its vertices and only its face winding changes.
    """
    import numpy as np
    import trimesh

    welded = trimesh.Trimesh(mesh.vertices.copy(), mesh.faces.copy(), process=False)
    welded.merge_vertices()
    original = welded.faces.copy()

    trimesh.repair.fix_winding(welded)  # consistent within each piece
    labels, count = component_labels(welded)
    verts, faces = welded.vertices, welded.faces
    per_face = (
        np.einsum(
            "ij,ij->i",
            verts[faces[:, 0]],
            np.cross(verts[faces[:, 1]], verts[faces[:, 2]]),
        )
        / 6.0
    )
    per_piece = np.bincount(labels, weights=per_face, minlength=count)
    welded_flip = per_piece[labels] < 0
    if welded_flip.any():
        welded.faces[welded_flip] = welded.faces[welded_flip][:, ::-1]

    flip = ~np.all(welded.faces == original, axis=1)
    if flip.any():
        mesh.faces = np.where(flip[:, None], mesh.faces[:, ::-1], mesh.faces)
    return bool(flip.any())
