# Incremental Voronoi Window Optimization Technical Report

**Project:** `inter-nav-project`

**Date:** 2026-09-21

**Scope:** Incremental semantic-Voronoi recomputation, seam validation, fallback diagnostics, and live smoke-test evaluation

## 1. Executive summary

The semantic-Voronoi updater was performing full-map rebuilds much more often than expected. Small, spatially separated map changes were frequently represented by one large bounding window, and the window was then expanded conservatively enough to cover most or all of the occupancy grid. The updater consequently crossed its full-rebuild threshold even when the directly changed area was small.

This change replaces that behavior with component-aware windows, union-based window accounting, clearance-aware reach estimation, and seam-validated adaptive expansion. A fixed safety margin is no longer paid unconditionally: it is introduced on the first seam mismatch and then increased geometrically when necessary. Full recomputation remains the correctness-preserving fallback whenever the union of local windows is too large or a safe splice cannot be established.

Across four live runs, the full-fallback rate fell from **83.9% to 33.9%**. In the final two like-for-like runs, the number of successful incremental updates increased from **32 to 41**, while full fallbacks decreased from **30 to 21**. The final run retained the same trajectory, changed-cell total, target, map occupancy, and final navigation state, so the improvement is attributable to the incremental-window decision rather than to an easier scenario.

The optimization is correctness-preserving in the exercised cases: 250 randomized sequential updates and 63 real-trajectory replay updates matched fresh full recomputation exactly. The focused test module passes all 22 tests. On the latest clean `origin/main` worktree, the full suite runs 130 tests: 124 pass, three are skipped, and three encounter pre-existing repository/resource errors listed in Section 7.4.

## 2. Problem statement

### 2.1 Observed symptom

The live run reported a large gap between the number of directly changed cells and the number of cells included in the incremental update window. The seam band and safety expansion often grew enough that the updater chose a full rebuild.

The initial live observation was:

- 62 incremental evaluations;
- 10 successful incremental updates;
- 52 full fallbacks;
- 83.9% fallback rate;
- 54 full graph builds.

The central issue was not that map changes were necessarily large. It was that the representation of their possible influence was overly conservative.

### 2.2 Meaning of a full fallback

A full fallback is not a navigation failure and does not restart the simulator. It means that an update which initially entered the incremental path is recomputed over the complete map because the local path is judged unsafe or uneconomical.

For the current 124 × 178 grid, the map contains 22,072 cells. With the configured maximum incremental window fraction of 0.75, a union larger than approximately 16,554 cells is rebuilt globally.

The current decision flow is:

1. Compare the current and previous traversability/observation state.
2. Identify directly changed cells and their connected components.
3. Construct guarded local windows around those components.
4. Reject the incremental path if the union of the windows exceeds the configured map fraction.
5. Recompute and splice the skeleton inside the windows.
6. Validate the splice in the seam band.
7. Accept the result if the seam agrees; otherwise expand and retry.
8. Fall back to a full rebuild if a safe bounded splice cannot be established.

This mechanism deliberately prefers a correct full rebuild over an incorrect local splice.

## 3. Root cause analysis

### 3.1 One bounding box connected unrelated changes

When distant changes were enclosed by one minimum/maximum row-column box, the empty space between them became part of the update window. Two small changes on opposite sides of the map could therefore produce an almost full-map rectangle.

### 3.2 Window work was measured less precisely than it was executed

Summing or enclosing windows can over-count overlaps. The fallback threshold should be based on the unique union of affected cells, while a separate work counter should expose repeated computation caused by overlapping retries or windows.

### 3.3 Fixed padding was paid before it was shown to be necessary

Clearance reach, spur pruning reach, seam guards, and a fixed incremental margin all contributed to initial expansion. Because the window bounds apply the reach on both sides, a small additive margin can have a large area effect on a 124-row map.

### 3.4 A fixed retry limit could force premature fallback

A fixed number of reach-expansion iterations does not directly express the property that matters: whether the seam is consistent and whether the window remains meaningfully local. Adaptive expansion tied to seam mismatch is more direct.

## 4. Implementation

### 4.1 Component-aware windows

Changed cells are divided into connected components. Each component gets its own candidate window instead of contributing to one global bounding box. Distant components therefore remain separate.

Windows are merged only when their write/seam influence overlaps. This preserves independent work while preventing conflicting writes in overlapping influence bands.

### 4.2 Union-based thresholding and accounting

The maximum-window decision uses the unique union area of the candidate windows. The implementation also records both:

- unique window cells, representing map coverage; and
- window work cells, representing the total amount of window processing.

This separates spatial coverage from duplicate computational work.

### 4.3 Clearance-aware reach

The reach estimate uses clearance values at the actual changed cells. It considers both the current and previous clearance state so that removal of a formerly influential obstacle cannot be incorrectly treated as a short-range change merely because its new clearance is small.

The base reach combines:

- the relevant clearance expressed in cells;
- the configured spur-pruning reach; and
- any adaptive retry boost.

### 4.4 Seam-driven adaptive expansion

After local recomputation, the new skeleton is compared with the retained skeleton in the seam band. If they agree, the splice is accepted. If they disagree, the reach is expanded and the local result is recomputed.

The fixed incremental margin is deferred until the first seam mismatch. The first retry adds at least the larger of the seam guard and configured margin; later retries double the boost. This lets small valid windows remain small without weakening the seam correctness check.

### 4.5 Fallback and diagnostic state

The updater records the reason for every fallback and exposes per-update and cumulative diagnostics, including:

- changed component count;
- window count;
- unique window cells and fraction;
- window work cells and fraction;
- seam-band cells;
- window-to-changed-cell ratio;
- cumulative changed, window, and work cells;
- maximum window count and fraction;
- last fallback reason; and
- fallback counts grouped by reason.

These counters make it possible to distinguish genuine wide-area influence from window-construction inflation.

## 5. Live-run results

All runs below used 62 incremental evaluations.

| Stage | Incremental successes | Full fallbacks | Fallback rate | Average window fraction | Full builds |
|---|---:|---:|---:|---:|---:|
| Initial observation | 10 | 52 | 83.9% | Not recorded | 54 |
| Component/window accounting iteration | 15 | 47 | 75.8% | 84.1% | 49 |
| Clearance and seam-adaptive expansion | 32 | 30 | 48.4% | 66.5% | 32 |
| Deferred fixed margin, current version | 41 | 21 | 33.9% | 64.9% | 23 |

### 5.1 Final like-for-like comparison

The final two runs are directly comparable:

- both evaluated 62 incremental map changes;
- both accumulated 13,084 directly changed cells;
- both produced 2,459 trajectory points;
- both ended with 11,825 observed and 7,443 inflated occupancy cells;
- both found `object:refrigerator:19` from the Isaac semantic source;
- both ended at the same position and active goal.

The final margin deferral changed the metrics as follows:

| Metric | Previous | Current | Change |
|---|---:|---:|---:|
| Incremental successes | 32 | 41 | +9 |
| Full fallbacks | 30 | 21 | -30.0% |
| Fallback rate | 48.4% | 33.9% | -14.5 percentage points |
| Unique window cells | 909,779 | 888,136 | -2.38% |
| Window work cells | 939,727 | 916,987 | -2.42% |
| Average window cells | 14,673.9 | 14,324.8 | -2.38% |
| Full builds | 32 | 23 | -9 |

The average area reduction is modest, but nine evaluations moved from just above the 75% cutoff to below it. This produces a discontinuously larger reduction in full rebuilds.

### 5.2 Remaining fallbacks

All 21 remaining fallbacks in the final run have reason `window_fraction`. There were no seam-mismatch fallbacks. The maximum window fraction remains 1.0 and the maximum window count is five, showing that some updates still have near-global estimated influence.

The result JSON does not record per-update wall-clock duration. Fewer full builds and fewer processed window cells are strong work-reduction indicators, but this report does not claim a measured end-to-end speedup.

## 6. Correctness strategy

The optimization does not assume that a small changed set always has a small Voronoi effect. Voronoi structure depends on nearest-obstacle relationships, which can change beyond the directly modified cells.

Correctness is protected by:

1. estimating reach from both old and new clearance;
2. retaining explicit seam bands;
3. accepting a local splice only after seam agreement;
4. expanding on disagreement; and
5. retaining full recomputation as the terminal fallback.

The tests compare incremental skeleton cells against a newly constructed full `SemanticVoronoiGraph`, rather than checking only counters or approximate graph properties.

## 7. Verification

### 7.1 Focused unit tests

Command:

```bash
$ISAAC_PYTHON -m unittest discover -s tests -p 'test_semantic_voronoi.py'
```

Result: **22/22 passed**.

New coverage includes:

- distant changes use separate windows;
- incremental diagnostics reset on an unchanged-traversability shortcut;
- fallback reasons are recorded;
- previous clearance contributes to the reach estimate; and
- incremental output matches a fresh full recomputation.

### 7.2 Randomized sequential equivalence

A deterministic 250-update sequence added small observed regions and periodic occupied cells to an 80 × 120 map. Every incremental result was compared with a fresh full recomputation.

Result:

- 250/250 exact skeleton matches;
- 247 incremental attempts succeeded;
- 0 fallbacks;
- average window fraction: 17.28%;
- maximum window fraction: 62.34%.

### 7.3 Real-trajectory progressive reveal replay

The recorded final occupancy map was progressively revealed around every 40th trajectory position. Each incremental result was compared with a fresh full recomputation.

Result:

- 63/63 exact skeleton matches;
- 61 incremental attempts succeeded;
- 0 fallbacks;
- average window fraction: 39.53%;
- maximum window fraction: 58.43%.

### 7.4 Full repository suite

Command:

```bash
$ISAAC_PYTHON -m unittest discover -s tests -p 'test_*.py'
```

Result: **130 tests run: 124 passed, three skipped, and three errors**. The errors are unrelated to this change:

1. `grutopia/demo/go2_walk_demo.py` is absent.
2. The MV7 benchmark metadata file `object_dict.json` is absent from the clean checkout.
3. `grutopia/demo/profiles/g1_grscene_mv7_navigation.json` is absent.

Compilation, line-length, and `git diff --check` validation passed for the changed implementation and test files.

## 8. Navigation outcome boundary

The final live run found the refrigerator target, produced two Voronoi plans, and performed one blocked-path replan. It nevertheless ended in `navigate_to_semantic_target`, with the robot approximately 1.099 m from the active goal.

The final position, target, trajectory length, and navigation state are unchanged from the preceding run. The incremental-window optimization therefore improved graph-update behavior without changing the navigation outcome. Arrival/termination behavior should be investigated as a separate issue.

## 9. Limitations and recommended follow-up

The present change optimizes window selection and local skeleton recomputation. It does not make every downstream graph-construction and semantic-topology stage local.

Before undertaking a larger architectural change, the next measurement step should record per-update timing and the distribution of window fractions, especially around the 0.75 cutoff. If additional improvement is required after profiling, likely directions include tiled/local clearance maintenance or a skeleton representation designed for bounded dynamic updates.

Changing the fallback threshold alone is not recommended: it can trade full-rebuild count for expensive near-global local work without establishing a real latency improvement.
