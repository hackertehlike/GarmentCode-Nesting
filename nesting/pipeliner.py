"""Utility module for assembling nesting experiments.

This module provides two tiers of helpers:

1. :class:`PlacementPipeline` — a light wrapper around the decoder registry
   that can (a) reorder pieces according to an initial sorting criterion and
   (b) execute one of the registered decoders while collecting common metrics.

2. :class:`NestingPipeliner` — a higher level orchestrator that keeps an
   immutable copy of the original layout/container pair and runs batches of
   experiments defined via :class:`ExperimentConfig`.

The goal is to make comparison studies (e.g. *initial sorting* + *decoder*) easy
to configure and repeat across many patterns.  The abstractions here are
agnostic to the underlying decoder implementation – anything registered in
``placement_engine.DECODER_REGISTRY`` works out of the box, and additional
heuristics (e.g. simulated annealing, naive evolution) can be plugged in by
calling :meth:`PlacementPipeline.register_decoder` with a factory callable.
"""

from __future__ import annotations

import copy
import csv
import random
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from shapely.geometry import Polygon

import nesting.config as config
import nesting.utils as utils
from .layout import Container, Layout, LayoutView, Piece
from .placement_engine import DECODER_REGISTRY, PlacementMode
from .simulated_annealing import SimulatedAnnealing


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SortSpec:
	"""Descriptor for a sorting criterion.

	Attributes
	----------
	description:
		Human-readable summary shown in logs/UI lists.
	metric:
		Callable that maps a :class:`Piece` to a numeric score. ``None`` means
		that the sorter handles the ordering internally (e.g. ``random`` or
		``original``).
	default_reverse:
		If True, higher metric values appear earlier in the order.
	supports_seed:
		Whether the sorter honours a ``seed`` keyword argument.
	"""

	description: str
	metric: Optional[Callable[[Piece], float]] = None
	default_reverse: bool = True
	supports_seed: bool = False


@dataclass
class ExperimentConfig:
	"""Configuration for a single pipeline run."""

	name: Optional[str]
	decoder_name: str
	sorting_criterion: str = "original"
	sorting_kwargs: Dict[str, Any] = field(default_factory=dict)
	decoder_kwargs: Dict[str, Any] = field(default_factory=dict)
	config_overrides: Dict[str, Any] = field(default_factory=dict)
	metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineRun:
	"""Result of a single placement pipeline execution."""

	decoder_name: str
	sorting_criterion: str
	placements: List[Tuple[str, float, float, float]]
	piece_order: List[str]
	success: bool
	metrics: Dict[str, float]
	runtime_sec: float
	experiment_name: Optional[str] = None
	metadata: Dict[str, Any] = field(default_factory=dict)
	error: Optional[str] = None

	@property
	def num_pieces(self) -> int:
		return len(self.piece_order)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


@contextmanager
def temporary_config(overrides: Dict[str, Any]):
	"""Temporarily override attributes on ``nesting.config``.

	Parameters
	----------
	overrides:
		Mapping of attribute names to temporary values.
	"""

	if not overrides:
		yield
		return

	sentinel = object()
	previous: Dict[str, Any] = {}
	try:
		for key, value in overrides.items():
			previous[key] = getattr(config, key, sentinel)
			setattr(config, key, value)
		yield
	finally:
		for key, old_value in previous.items():
			if old_value is sentinel:
				delattr(config, key)
			else:
				setattr(config, key, old_value)


def _convex_hull_area(piece: Piece) -> float:
	poly = Polygon(utils.clean_polygon_coordinates(piece.get_outer_path()))
	return poly.convex_hull.area if not poly.is_empty else 0.0


def _polygon_area(piece: Piece) -> float:
	return utils.polygon_area(piece.get_outer_path())


def _concavity(piece: Piece) -> float:
	hull = _convex_hull_area(piece)
	area = max(_polygon_area(piece), 0.0)
	if hull <= 1e-9:
		return 0.0
	return (hull - area) / hull


def _rectangularity(piece: Piece) -> float:
	rect_area = max(piece.width * piece.height, 1e-9)
	return _polygon_area(piece) / rect_area


def _aspect_ratio(piece: Piece) -> float:
	w, h = piece.width, piece.height
	if w <= 1e-9 or h <= 1e-9:
		return 1.0
	ratio = w / h
	return ratio if ratio >= 1.0 else 1.0 / ratio


# ---------------------------------------------------------------------------
# Placement pipeline
# ---------------------------------------------------------------------------


class PlacementPipeline:
	"""Utility for running a single (sort → decode) experiment."""

	def __init__(
		self,
		*,
		sort_registry: Optional[Dict[str, SortSpec]] = None,
		decoder_registry: Optional[Dict[str, Any]] = None,
	) -> None:
		self.sort_registry: Dict[str, SortSpec] = sort_registry or self._build_default_sort_registry()
		self.decoder_registry: Dict[str, Any] = dict(decoder_registry or DECODER_REGISTRY)
		self._register_builtin_adapters()

	def _register_builtin_adapters(self) -> None:
		"""Register additional high-level decoders that aren't part of DECODER_REGISTRY."""

		if "SA" not in self.decoder_registry:
			self.register_decoder("SA", SimulatedAnnealingPlacementAdapter)

	# -- registry inspection -------------------------------------------------

	def list_sort_criteria(self) -> List[str]:
		return sorted(self.sort_registry.keys())

	def list_decoders(self) -> List[str]:
		return sorted(self.decoder_registry.keys())

	# -- registry mutation ---------------------------------------------------

	def register_sorting_criterion(self, name: str, spec: SortSpec) -> None:
		self.sort_registry[name] = spec

	def register_decoder(self, name: str, factory: Any) -> None:
		self.decoder_registry[name] = factory

	# -- core execution ------------------------------------------------------

	def run(
		self,
		*,
		layout: Layout,
		container: Container,
		sort_criteria: str = "original",
		decoder: str = "BL",
		sort_kwargs: Optional[Dict[str, Any]] = None,
		decoder_kwargs: Optional[Dict[str, Any]] = None,
		config_overrides: Optional[Dict[str, Any]] = None,
		return_full_result: bool = False,
		copy_layout: bool = True,
		metadata: Optional[Dict[str, Any]] = None,
	) -> Union[PipelineRun, List[Tuple[str, float, float, float]]]:
		"""Execute the selected sorter + decoder pipeline."""

		sort_kwargs = sort_kwargs or {}
		decoder_kwargs = decoder_kwargs or {}
		metadata = metadata or {}

		work_layout = copy.deepcopy(layout) if copy_layout else layout
		work_container = copy.deepcopy(container)

		start_total = time.perf_counter()
		try:
			sorted_layout = self._apply_sort(work_layout, sort_criteria, **sort_kwargs)
		except Exception as exc:
			result = PipelineRun(
				decoder_name=decoder,
				sorting_criterion=sort_criteria,
				placements=[],
				piece_order=[],
				success=False,
				metrics={},
				runtime_sec=time.perf_counter() - start_total,
				metadata=metadata,
				error=str(exc),
			)
			return result if return_full_result else []

		try:
			with temporary_config(config_overrides or {}):
				decoder_instance = self._instantiate_decoder(decoder, sorted_layout, work_container, decoder_kwargs)
				placements = decoder_instance.decode()
			success = decoder_instance.layout_is_valid()
			metrics = self._collect_metrics(decoder_instance)
			piece_order = [pid for pid, *_ in placements]
			runtime = time.perf_counter() - start_total
			result = PipelineRun(
				decoder_name=decoder,
				sorting_criterion=sort_criteria,
				placements=placements,
				piece_order=piece_order,
				success=success,
				metrics=metrics,
				runtime_sec=runtime,
				metadata=metadata,
			)
		except Exception as exc:  # noqa: BLE001 - returning failure details is useful here
			runtime = time.perf_counter() - start_total
			result = PipelineRun(
				decoder_name=decoder,
				sorting_criterion=sort_criteria,
				placements=[],
				piece_order=[],
				success=False,
				metrics={},
				runtime_sec=runtime,
				metadata=metadata,
				error=str(exc),
			)

		return result if return_full_result else result.placements

	# -- internals -----------------------------------------------------------

	def _build_default_sort_registry(self) -> Dict[str, SortSpec]:
		return {
			"original": SortSpec("Preserve original order", metric=None, default_reverse=False),
			"random": SortSpec("Random shuffle", metric=None, default_reverse=False, supports_seed=True),
			"bbox_area": SortSpec("Bounding box area", metric=lambda p: p.bbox_area, default_reverse=True),
			"area": SortSpec("Polygon area", metric=_polygon_area, default_reverse=True),
			"length": SortSpec("Axis-aligned width", metric=lambda p: p.width, default_reverse=True),
			"height": SortSpec("Axis-aligned height", metric=lambda p: p.height, default_reverse=True),
			"hull_area": SortSpec("Convex hull area", metric=_convex_hull_area, default_reverse=True),
			"aspect_ratio": SortSpec("Aspect ratio (>=1)", metric=_aspect_ratio, default_reverse=True),
			"concavity": SortSpec("Concavity (higher → more concave)", metric=_concavity, default_reverse=True),
			"rectangularity": SortSpec("Rectangularity (area / bbox)", metric=_rectangularity, default_reverse=False),
		}

	def _apply_sort(self, layout: Layout, criterion: str, **kwargs: Any) -> Layout:
		if criterion not in self.sort_registry:
			available = ", ".join(sorted(self.sort_registry))
			raise ValueError(f"Unknown sorting criterion '{criterion}'. Available: {available}")

		spec = self.sort_registry[criterion]

		items = list(layout.order.items())
		if criterion == "original":
			sorted_items = items
		elif criterion == "random":
			seed = kwargs.get("seed") if spec.supports_seed else None
			rng = random.Random(seed)
			rng.shuffle(items)
			sorted_items = items
		else:
			reverse = spec.default_reverse if "reverse" not in kwargs else bool(kwargs["reverse"])
			sorted_items = sorted(items, key=lambda item: spec.metric(item[1]), reverse=reverse)  # type: ignore[arg-type]

		sorted_layout = Layout(OrderedDict(sorted_items))
		return sorted_layout

	def _instantiate_decoder(
		self,
		decoder_name: str,
		layout: Layout,
		container: Container,
		kwargs: Dict[str, Any],
	) -> Any:
		if decoder_name not in self.decoder_registry:
			available = ", ".join(sorted(self.decoder_registry))
			raise ValueError(f"Unknown decoder '{decoder_name}'. Available: {available}")

		factory = self.decoder_registry[decoder_name]
		if callable(factory):
			return factory(layout, container, **kwargs)
		raise TypeError(f"Decoder registry entry for '{decoder_name}' is not callable: {factory!r}")

	@staticmethod
	def _collect_metrics(decoder_instance: Any) -> Dict[str, float]:
		metrics = {}
		try:
			metrics["usage_bb"] = decoder_instance.usage_BB()
		except Exception:
			metrics["usage_bb"] = 0.0

		for name, fn in (
			("rest_length", getattr(decoder_instance, "rest_length", None)),
			("rest_height", getattr(decoder_instance, "rest_height", None)),
			("bbox_area", getattr(decoder_instance, "bbox_area", None)),
			("concave_hull_area", getattr(decoder_instance, "concave_hull_area", None)),
			("concave_hull_utilization", getattr(decoder_instance, "concave_hull_utilization", None)),
		):
			try:
				metrics[name] = fn() if callable(fn) else 0.0
			except Exception:
				metrics[name] = 0.0

		return metrics


# ---------------------------------------------------------------------------
# High-level decoder adapters
# ---------------------------------------------------------------------------


class SimulatedAnnealingPlacementAdapter:
	"""Adapter that wraps ``SimulatedAnnealing`` so it behaves like a decoder.

	The annealer optimizes piece orderings using local mutations; once complete
	we re-run a traditional decoder (``evaluation_decoder``) to obtain actual
	placements and metrics.  This keeps compatibility with
	:class:`PlacementPipeline` while allowing SA to explore the search space.
	"""

	def __init__(
		self,
		layout: Layout,
		container: Container,
		*,
		initial_temperature: float = 20.0,
		cooling_rate: float = 0.85,
		iterations_per_temp: int = 10,
		evaluation_decoder: Optional[str] = None,
		evaluation_decoder_kwargs: Optional[Dict[str, Any]] = None,
		design_params: Optional[dict] = None,
		body_params: Optional[Any] = None,
		disable_logging: bool = True,
		additional_sa_kwargs: Optional[Dict[str, Any]] = None,
	) -> None:
		self.layout = layout
		self.container = container
		self.initial_temperature = initial_temperature
		self.cooling_rate = cooling_rate
		self.iterations_per_temp = iterations_per_temp
		self.evaluation_decoder = evaluation_decoder
		self.evaluation_decoder_kwargs = evaluation_decoder_kwargs or {}
		self.design_params = design_params
		self.body_params = body_params
		self.disable_logging = disable_logging
		self.additional_sa_kwargs = additional_sa_kwargs or {}

		self._evaluation_decoder_instance: Optional[Any] = None
		self._annealer: Optional[SimulatedAnnealing] = None
		self.placed: List[Piece] = []

	def decode(self) -> List[Tuple[str, float, float, float]]:
		pieces = [copy.deepcopy(p) for p in self.layout.order.values()]
		if not pieces:
			self.placed = []
			self._evaluation_decoder_instance = None
			return []

		sa_kwargs = dict(self.additional_sa_kwargs)
		if "iterations_per_temp" not in sa_kwargs:
			sa_kwargs["iterations_per_temp"] = self.iterations_per_temp

		overrides: Dict[str, Any] = {}
		if self.disable_logging:
			overrides.update({
				"SAVE_LOGS": False,
				"SAVE_GENERATION_SVGS": False,
				"LOG_TIME": False,
			})
		if self.evaluation_decoder:
			overrides["SELECTED_DECODER"] = self.evaluation_decoder

		with temporary_config(overrides):
			annealer = SimulatedAnnealing(
				pieces=pieces,
				container=self.container,
				cooling_rate=self.cooling_rate,
				initial_temperature=self.initial_temperature,
				design_params=self.design_params,
				body_params=self.body_params,
				**sa_kwargs,
			)
			self._annealer = annealer
			annealer.run()

			best_pieces = [copy.deepcopy(p) for p in annealer.current_state]
			placements, decoder_instance = self._evaluate(best_pieces)

		self._evaluation_decoder_instance = decoder_instance
		self.placed = getattr(decoder_instance, "placed", [])
		return placements

	def _evaluate(
		self, pieces: List[Piece]
	) -> Tuple[List[Tuple[str, float, float, float]], Any]:
		eval_name = self.evaluation_decoder or getattr(config, "SELECTED_DECODER", "BL")
		if eval_name not in DECODER_REGISTRY:
			raise ValueError(f"Evaluation decoder '{eval_name}' is not registered")

		layout_view = LayoutView([copy.deepcopy(p) for p in pieces])
		factory = DECODER_REGISTRY[eval_name]
		decoder_instance = factory(layout_view, self.container, **self.evaluation_decoder_kwargs)
		placements = decoder_instance.decode()
		return placements, decoder_instance

	def layout_is_valid(self) -> bool:
		if self._evaluation_decoder_instance and hasattr(self._evaluation_decoder_instance, "layout_is_valid"):
			return self._evaluation_decoder_instance.layout_is_valid()
		return False

	def _delegate_metric(self, name: str) -> float:
		if self._evaluation_decoder_instance is None:
			return 0.0
		attr = getattr(self._evaluation_decoder_instance, name, None)
		if callable(attr):
			try:
				return float(attr())
			except Exception:
				return 0.0
		return 0.0

	def usage_BB(self) -> float:
		return self._delegate_metric("usage_BB")

	def rest_length(self) -> float:
		return self._delegate_metric("rest_length")

	def rest_height(self) -> float:
		return self._delegate_metric("rest_height")

	def bbox_area(self) -> float:
		return self._delegate_metric("bbox_area")

	def concave_hull_area(self) -> float:
		return self._delegate_metric("concave_hull_area")

	def concave_hull_utilization(self) -> float:
		return self._delegate_metric("concave_hull_utilization")

	@property
	def annealer(self) -> Optional[SimulatedAnnealing]:
		"""Expose the underlying annealer for debugging/inspection."""
		return self._annealer

# ---------------------------------------------------------------------------
# Experiment orchestrator
# ---------------------------------------------------------------------------


class NestingPipeliner:
	"""Run multiple experiments against a fixed layout/container pair."""

	def __init__(self, layout: Layout, container: Container, *, pipeline: Optional[PlacementPipeline] = None) -> None:
		self._base_layout = copy.deepcopy(layout)
		self._base_container = copy.deepcopy(container)
		self.pipeline = pipeline or PlacementPipeline()

	def run_experiment(self, config: ExperimentConfig) -> PipelineRun:
		layout_copy = copy.deepcopy(self._base_layout)
		container_copy = copy.deepcopy(self._base_container)

		result = self.pipeline.run(
			layout=layout_copy,
			container=container_copy,
			sort_criteria=config.sorting_criterion,
			decoder=config.decoder_name,
			sort_kwargs=config.sorting_kwargs,
			decoder_kwargs=config.decoder_kwargs,
			config_overrides=config.config_overrides,
			metadata=config.metadata,
			return_full_result=True,
		)

		if config.name:
			result.experiment_name = config.name
		else:
			result.experiment_name = f"{config.decoder_name}|{config.sorting_criterion}"

		# Ensure metadata is merged (pipeline may add details later)
		if config.metadata:
			merged = {**config.metadata, **result.metadata}
			result.metadata = merged

		return result

	def run_batch(self, experiments: Sequence[ExperimentConfig], *, progress: bool = False) -> List[PipelineRun]:
		results: List[PipelineRun] = []
		total = len(experiments)
		for idx, exp in enumerate(experiments, start=1):
			if progress:
				label = exp.name or f"{exp.decoder_name}|{exp.sorting_criterion}"
				print(f"[NestingPipeliner] {idx}/{total}: {label}")
			results.append(self.run_experiment(exp))
		return results


# ---------------------------------------------------------------------------
# Convenience helpers for experiment setup & logging
# ---------------------------------------------------------------------------


def create_experiment_configurations(*, quick_mode: bool = False) -> List[ExperimentConfig]:
	"""Return a curated list of experiment configs covering common scenarios."""

	sorting_options = [
		"original",
		"bbox_area",
		"area",
		"length",
		"hull_area",
		"aspect_ratio",
		"concavity",
		"rectangularity",
		"random",
	]

	if quick_mode:
		sorting_options = ["original", "bbox_area", "length", "random"]

	experiments: List[ExperimentConfig] = []

	# Baseline BL decoder across sorts
	for criterion in sorting_options:
		experiments.append(
			ExperimentConfig(
				name=f"BL|{criterion}",
				decoder_name="BL",
				sorting_criterion=criterion,
			)
		)

	# NFP variations: placement modes × gravity toggle
	nfp_modes = [
		PlacementMode.BOTTOM_LEFT,
		PlacementMode.MAX_OVERLAP,
		PlacementMode.MIN_BBOX_LENGTH,
		PlacementMode.MIN_BBOX_AREA,
	]

	gravity_options = [True] if quick_mode else [True, False]

	for criterion in sorting_options:
		for mode in nfp_modes:
			for gravity_on in gravity_options:
				metadata = {
					"placement_mode": mode.value,
					"nfp_gravity_on": gravity_on,
				}
				experiments.append(
					ExperimentConfig(
						name=f"NFP|{mode.value}|{'grav' if gravity_on else 'nograv'}|{criterion}",
						decoder_name="NFP",
						sorting_criterion=criterion,
						decoder_kwargs={"placement_mode": mode},
						config_overrides={"NFP_GRAVITATE_ON": gravity_on},
						metadata=metadata,
					)
				)

	# Simulated annealing experiments — share same sorts, configurable decoder
	# NOTE: SA experiments temporarily disabled per user request.
	# Keep adapter code available elsewhere; re-enable by uncommenting below.
	# sa_default_kwargs = {
	# 	"initial_temperature": 20.0,
	# 	"cooling_rate": 0.85,
	# 	"iterations_per_temp": 10,
	# 	"disable_logging": True,
	# 	"evaluation_decoder": "BL",
	# }
	# for criterion in sorting_options:
	# 	metadata = {
	# 		"annealing_initial_temp": sa_default_kwargs["initial_temperature"],
	# 		"annealing_cooling_rate": sa_default_kwargs["cooling_rate"],
	# 	}
	# 	experiments.append(
	# 		ExperimentConfig(
	# 			name=f"SA|{criterion}",
	# 			decoder_name="SA",
	# 			sorting_criterion=criterion,
	# 			decoder_kwargs=dict(sa_default_kwargs),
	# 			metadata=metadata,
	# 		)
	# 	)

	# Random decoder – ignore initial sorting but keep configs for consistency
	for criterion in sorting_options:
		experiments.append(
			ExperimentConfig(
				name=f"RandomDecoder|{criterion}",
				decoder_name="Random",
				sorting_criterion=criterion,
			)
		)

	return experiments


def save_results_to_csv(results: Iterable[PipelineRun], file_path: str) -> None:
	"""Persist pipeline results into a CSV file."""

	results = list(results)
	if not results:
		return

	base_fields = [
		"experiment_name",
		"decoder_name",
		"sorting_criterion",
		"success",
		"error",
		"runtime_sec",
		"num_pieces",
		"piece_order",
		"placements",
	]

	metric_keys = sorted({key for res in results for key in res.metrics})
	metadata_keys = sorted({key for res in results for key in res.metadata})

	fieldnames = base_fields + metric_keys + [f"metadata_{k}" for k in metadata_keys]

	with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()

		for res in results:
			row = {
				"experiment_name": res.experiment_name,
				"decoder_name": res.decoder_name,
				"sorting_criterion": res.sorting_criterion,
				"success": res.success,
				"error": res.error,
				"runtime_sec": f"{res.runtime_sec:.6f}",
				"num_pieces": res.num_pieces,
				"piece_order": " | ".join(res.piece_order),
				"placements": " | ".join(
					f"{pid}:{x:.3f},{y:.3f},{rot:.1f}" for pid, x, y, rot in res.placements
				),
			}

			for key in metric_keys:
				row[key] = res.metrics.get(key, 0.0)

			for key in metadata_keys:
				row[f"metadata_{key}"] = res.metadata.get(key)

			writer.writerow(row)

