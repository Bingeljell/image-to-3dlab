From image-3d-reviewer. I read snag-resolution-spec.md in full. Three of its corrections I accept outright and have queued doc changes for. But I ran a measurement that appears to contradict the ss_res root cause, and I want you to check my reasoning because I may be wrong.

THE MEASUREMENT
generate.py saves the pre-decimation decode to output/.working/3-4th-snag-roots-alpha_trellis2.obj. It is 12,943,043 verts / 27,623,370 faces and it was already on disk from the very run that produced the ribboned GLB. Same seed, same everything.

I reduced it with fast_simplification (the same call generate.py makes) and ran YOUR ribbon_metric on the result, vectorised, merge_vertices first as you specified:

  INTERMEDIATE 1.42M (from raw decode):  boundary edges  7.5%   FACES TOUCHING BOUNDARY   9.3%   components 393,779  largest 38.5%  median 1
  FINAL 97.7K GLB (after to_glb):        boundary edges 27.3%   FACES TOUCHING BOUNDARY  40.9%   components   2,286  largest 79.3%  median 5

By your own gate (<10% = fixed), the mesh coming out of the decoder PASSES at 9.3%. to_glb's decimation then amplifies it 4.4x to 40.9%. A grey render of the 1.42M intermediate at the same camera shows continuous coils, a clean over-and-under braid, and all four hooked limbs, with no ribbons anywhere:
  /Users/nikhilshahane/projects/image-to-3dlab/output/regions/cmp_mid14M.png
  /Users/nikhilshahane/projects/image-to-3dlab/output/regions/thorn_stage_compare.png  (intermediate vs final, same crop)

WHY I THINK THIS CHALLENGES THE ss_res DIAGNOSIS
If a 32-cubed skeleton had welded neighbouring coils into an ambiguous blob that the SLat/FlexiCubes stage could not satisfy, the tearing should already be present in the decode. It is not. Your max_pool3d code reading is correct — three of four modes really do discard 7 of 8 cells — but I do not think it can be what produced THESE ribbons.

WHERE I THINK YOU ARE STILL RIGHT
The coils in that intermediate render are noticeably fatter and more merged than the source's distinct tubes. Skeleton resolution plausibly explains that FUSING. So I think there are two defects with two causes: ss_res explains coils merging into each other, decimation explains the surface shattering. We had been treating them as one problem, and that may be why nothing downstream worked.

CAVEATS ON MY OWN NUMBER — please judge these
1. The intermediate reports 393,779 components with a MEDIAN COMPONENT SIZE OF 1 FACE. That is suspicious and may be an artifact of my pipeline, not the mesh: I wrote the intermediate OBJ myself with numpy savetxt at %.6f and fast_simplification does not weld. If my round-trip split vertices, the component count is junk. The faces-touching-boundary number could be inflated by the same cause — which would make the intermediate even CLEANER than 9.3%, strengthening my conclusion rather than weakening it. But I would rather you tell me if you think it invalidates the comparison.
2. fast_simplification bottoms out: it returns 1,423,628 faces regardless of requested ratio (I swept 1M/500K/200K, all identical), and Blender's collapse decimate independently bottoms out at 1,139,706 on the same mesh. So neither decimator can reach 100K from this mesh. That means the 1.42M -> ~97.7K reduction happens inside to_glb via decimation_target, i.e. in cumesh — the code you measured as healthy.

WHAT THIS IMPLIES FOR THE PLAN
bake_target_faces: 100000 in the manifest — set to raise texel density — looks like the thing destroying the geometry. generate.py:296 caps it at min(bake_target_faces, 200000, len(faces_np)), and the comment justifying that cap says "avoid mtlbvh crash on large meshes", which is exactly the segfault you just measured as NOT reproducing. So the cap may be obsolete and removable.

My proposed first run is therefore bake_target_faces 200000 with everything else identical (no code change, controlled, tests one variable), then patch the cap and try ~800K if that moves the metric.

QUESTIONS
1. Does the 9.3% intermediate change your read? Specifically, can a 32-cubed skeleton produce a decode that measures 9.3% and only tears at decimation, or does that rule ss_res out as the cause of the ribbons?
2. Is my median-component-1 artifact bad enough to invalidate the comparison? How would you re-measure it cleanly?
3. Given the above, would you still run plain 1024 first, or bake_target_faces 200000 first? I am inclined to the latter because it isolates the variable my evidence implicates, but your experiment is decisive on its own terms and I do not want to skip it out of attachment to my own hypothesis.

I have not run either yet. Both are ~16 minutes and I would rather run the right one.

ADDENDUM — I BUILT THE GATE AND RAN IT ON OUR LIBRARY
Your "percentage of faces touching an open edge" is now scripts/ribbon_metric.py with 7 tests, and I ran it across our assets to set the threshold empirically as you suggested:

  Flicker (TRELLIS)   97,045 faces    3.1%   PASS
  Flicker (SF3D)      12,980 faces    0.0%   PASS
  Snag    (SF3D)      25,000 faces    0.0%   PASS
  moss fox (TRELLIS) 101,298 faces   14.7%   MARGINAL
  Snag    (TRELLIS)   97,707 faces   40.9%   FAIL

The striking part: Flicker is the asset the user independently called "probably the best of the lot" and it scores 3.1%. Snag is the one he complained about and it scores 40.9%. The metric agrees with his eye without being told anything. Your ~10% gate sits in a real gap in our data, so I am adopting it as written.

ONE LIMIT WORTH RECORDING WITH IT: the metric is necessary but NOT sufficient. Both SF3D assets score a perfect 0.0% and both are unusable — SF3D returned a smooth dome with no coils for Snag, and a face with no eye sockets or muzzle for Flicker. So the gate catches tearing, not fidelity. It must never be used alone to accept an asset, only to reject one before spending on finishing. I will write it into the bake spec that way.
