"""Bounded RGB-D object observations; no models or simulator labels."""
from collections import OrderedDict
from dataclasses import dataclass, field
import math

import numpy as np


def bounds_overlap(left, right, padding=0.08):
    """Intersection relative to the smaller observed volume, not object identity."""
    if left is None or right is None:
        return 0.0
    a, b = np.asarray(left, float), np.asarray(right, float)
    if a.shape != (2, 3) or b.shape != (2, 3) or not np.isfinite([a, b]).all():
        return 0.0
    if np.any(a[0] > a[1]) or np.any(b[0] > b[1]):
        return 0.0
    lower = np.maximum(a[0] - padding, b[0] - padding)
    upper = np.minimum(a[1] + padding, b[1] + padding)
    intersection = np.prod(np.maximum(0, upper - lower))
    volume = min(np.prod(np.maximum(0.02, a[1] - a[0] + 2 * padding)),
                 np.prod(np.maximum(0.02, b[1] - b[0] + 2 * padding)))
    return float(intersection / volume)


def robust_object_geometry(mask, depth, point_image, min_points=1, depth_gap=0.25, min_fraction=0.55):
    """Reject ambiguous depth layers; retain the dominant layer and its original pixels."""
    valid = mask & np.isfinite(depth) & (depth > 0) & np.isfinite(point_image).all(axis=-1)
    indices = np.flatnonzero(valid)
    if len(indices) < min_points:
        raise ValueError('insufficient_valid_depth')
    ordered = indices[np.argsort(depth.ravel()[indices], kind='stable')]
    splits = np.flatnonzero(np.diff(depth.ravel()[ordered]) > depth_gap) + 1
    groups = np.split(ordered, splits)
    selected = max(groups, key=len)
    if len(selected) / len(indices) < min_fraction:
        raise ValueError('ambiguous_depth_layers')
    points = point_image.reshape(-1, 3)[selected]
    center = np.median(points, axis=0)
    deviation = np.median(np.abs(points - center), axis=0)
    keep = np.all(np.abs(points - center) <= np.maximum(0.15, 4 * 1.4826 * deviation), axis=1)
    selected = selected[keep]
    if len(selected) < min_points or len(selected) / len(indices) < min_fraction:
        raise ValueError('insufficient_geometry_support')
    ratio = len(selected) / max(1, int(mask.sum()))
    if ratio < 0.35:
        raise ValueError('sparse_mask_depth')
    clean = np.zeros(mask.shape, bool)
    clean.ravel()[selected] = True
    points = point_image[clean]
    bounds = np.percentile(points, (5, 95), axis=0)
    metadata = {'bounds': tuple(tuple(float(x) for x in row) for row in bounds),
                'surface_points': points[np.linspace(0,len(points)-1,min(128,len(points)),dtype=int)].copy(),
                'quality': float(ratio), 'valid_fraction': float(ratio),
                'rejected_points': int(mask.sum()) - len(selected),
                'centroid': tuple(float(x) for x in np.median(points, axis=0))}
    return clean, points, metadata


def distinct_view(left, right, target, min_angle=20.0, min_baseline=0.20):
    if left is None or right is None:
        return False
    left, right, target = (np.asarray(x, float)[:2] for x in (left, right, target))
    a, b = left - target, right - target
    norms = np.linalg.norm(a) * np.linalg.norm(b)
    if norms <= 1e-9 or np.linalg.norm(left - right) < min_baseline:
        return False
    angle = math.degrees(math.acos(float(np.clip(np.dot(a, b) / norms, -1, 1))))
    return angle >= min_angle


def union_bounds(left, right):
    if left is None:
        return right
    if right is None:
        return left
    a,b = np.asarray(left,float),np.asarray(right,float)
    return tuple(tuple(float(x) for x in row) for row in (np.minimum(a[0],b[0]), np.maximum(a[1],b[1])))


@dataclass
class _ObjectViews:
    instance_id: str
    position: tuple
    bounds: tuple
    appearance: np.ndarray
    last_step: int
    surface_points: object = None
    views: list = field(default_factory=list)


class MultiViewMemory:
    """Associate conservative geometry/appearance matches, keeping at most three views."""

    def __init__(self, min_angle=20.0, max_age=2400, capacity=64):
        self.min_angle, self.max_age, self.capacity = min_angle, max_age, capacity
        self.objects = OrderedDict()
        self.next_id = 0
        self._step = None
        self._claimed = set()

    def begin_frame(self, step):
        self._step = step
        self._claimed.clear()

    def observe(self, crop, foreground, mask, geometry, camera_position, step):
        if self._step != step:
            self.begin_frame(step)
        for key in list(self.objects):
            if step - self.objects[key].last_step > self.max_age:
                self.objects[key].views.clear()  # Keep geometry identity, expire image evidence.
        colors = foreground[mask]
        histogram = np.concatenate([np.histogram(colors[:, channel], bins=8, range=(0, 256))[0]
                                    for channel in range(3)]).astype(float)
        histogram /= max(1e-12, np.linalg.norm(histogram))
        quality = geometry['quality'] * min(1.0, len(colors) / 4096.0)
        candidates = []
        for item in self.objects.values():
            if item.instance_id in self._claimed:
                continue
            distance = np.linalg.norm(np.asarray(item.position) - geometry['centroid'])
            overlap = bounds_overlap(item.bounds, geometry['bounds'])
            appearance = float(np.dot(item.appearance, histogram))
            extent = max(np.linalg.norm(np.subtract(item.bounds[1],item.bounds[0])),
                         np.linalg.norm(np.subtract(geometry['bounds'][1],geometry['bounds'][0])))
            if overlap<0.25 or distance>=max(0.75,0.5*extent+0.25):
                continue
            surface = geometry.get('surface_points')
            agreement = None
            if surface is not None and item.surface_points is not None:
                matrix = np.linalg.norm(np.asarray(surface)[:,None,:]-item.surface_points[None,:,:],axis=2)
                agreement = max(float(np.mean(matrix.min(axis=0)<0.12)),float(np.mean(matrix.min(axis=1)<0.12)))
            same_geometry = overlap >= 0.80 and distance < 0.35
            limit = max(0.75,0.5*extent+0.25) if agreement is not None and agreement>=0.20 else 0.75
            if distance < limit and (agreement is None or agreement>=0.20) and (appearance>=0.55 or same_geometry):
                candidates.append((overlap + 0.2 * appearance - 0.1 * distance, item))
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        item = candidates[0][1] if candidates else None
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.10:
            item = None  # Ambiguous association cannot mix reference images.
        if item is None:
            self.next_id += 1
            item = _ObjectViews(f'view-object:{self.next_id}', geometry['centroid'], geometry['bounds'],
                               histogram, int(step))
            self.objects[item.instance_id] = item
        if step < item.last_step:
            return item.instance_id, [], []
        item.views = [view for view in item.views if step - view['step'] <= self.max_age]
        references = [view for view in item.views if view['step'] < step and
                      view['quality'] >= quality * 0.25 and
                      distinct_view(view['position'], camera_position, geometry['centroid'], self.min_angle)]
        references.sort(key=lambda view: view['quality'], reverse=True)
        diverse = []
        for view in references:
            if all(distinct_view(view['position'],old['position'],geometry['centroid'],self.min_angle) for old in diverse):
                diverse.append(view)
            if len(diverse)==2:break
        references = diverse
        if camera_position is not None:
            position = tuple(float(x) for x in camera_position)
            view = {'image': crop.copy(), 'position': position, 'step': int(step),
                    'quality': quality}
            similar = [i for i, old in enumerate(item.views)
                       if not distinct_view(old['position'], position, geometry['centroid'], self.min_angle)]
            if similar:
                index = similar[0]
                if view['quality'] >= item.views[index]['quality']:
                    item.views[index] = view
            else:
                item.views.append(view)
            item.views.sort(key=lambda view: view['quality'], reverse=True)
            item.views = item.views[:3]
        item.last_step, item.position, item.bounds = int(step), geometry['centroid'], union_bounds(item.bounds, geometry['bounds'])
        self._claimed.add(item.instance_id)
        item.appearance = histogram
        surface = geometry.get('surface_points')
        if surface is not None:
            combined = surface if item.surface_points is None else np.vstack((item.surface_points,surface))
            item.surface_points = combined[np.linspace(0,len(combined)-1,min(128,len(combined)),dtype=int)].copy()
        self.objects.move_to_end(item.instance_id)
        while len(self.objects) > self.capacity:
            self.objects.popitem(last=False)
        return item.instance_id, [view['image'] for view in references], [
            {'step': view['step'], 'camera_position': list(view['position']), 'quality': view['quality']}
            for view in references]
