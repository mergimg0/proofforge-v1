"""Protein hyperparameter sweep: GP regression + Pareto optimization.

Adapted from PufferLib 4.0 sweep.py (lines 1-966). All PufferLib-specific
imports (pufferlib, _C) removed. Depends on: gpytorch, scikit-learn, scipy.

Core idea: Bayesian optimization using dual GPs (one for score, one for cost)
with Pareto front tracking and early stopping via quantile regression.
"""

from __future__ import annotations

import math
import random
import warnings
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

import numpy as np

# Heavy dependencies are imported lazily to allow the module to be importable
# without gpytorch/sklearn installed (tests can mock them).
_GPYTORCH_AVAILABLE = False
_SKLEARN_AVAILABLE = False

try:
    import torch
    import gpytorch
    from gpytorch.models import ExactGP
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.kernels import MaternKernel, PolynomialKernel, ScaleKernel, AdditiveKernel
    from gpytorch.means import ConstantMean
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.priors import LogNormalPrior
    _GPYTORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from scipy.optimize import minimize as scipy_minimize
    from scipy.stats.qmc import Sobol
    from scipy.spatial import KDTree
except ImportError:
    pass

try:
    from sklearn.linear_model import LogisticRegression
    _SKLEARN_AVAILABLE = True
except ImportError:
    pass


EPSILON = 1e-6


# ---------------------------------------------------------------------------
# Utility helpers (adapted from PufferLib sweep.py:26-44)
# ---------------------------------------------------------------------------

def unroll_nested_dict(d):
    """Flatten a nested dict into (dotted_key, value) pairs."""
    if not isinstance(d, dict):
        return d
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in unroll_nested_dict(v):
                yield f"{k}/{k2}", v2
        else:
            yield k, v


@contextmanager
def default_tensor_dtype(dtype):
    """Temporarily set torch default dtype."""
    import torch as _torch
    old_dtype = _torch.get_default_dtype()
    try:
        _torch.set_default_dtype(dtype)
        yield
    finally:
        _torch.set_default_dtype(old_dtype)


# ---------------------------------------------------------------------------
# Parameter space classes (adapted from PufferLib sweep.py:46-139)
# ---------------------------------------------------------------------------

class Space:
    """Base parameter space with normalize/unnormalize mapping to [-1, 1]."""

    def __init__(self, min, max, scale, is_integer=False):
        self.min = min
        self.max = max
        self.scale = scale
        self.norm_min = self.normalize(min)
        self.norm_max = self.normalize(max)
        self.norm_mean = 0
        self.is_integer = is_integer


class Linear(Space):
    """Uniform linear space. Adapted from PufferLib sweep.py:57-74."""

    def __init__(self, min, max, scale, is_integer=False):
        if scale == "auto":
            scale = 0.5
        super().__init__(min, max, scale, is_integer)

    def normalize(self, value):
        zero_one = (value - self.min) / (self.max - self.min)
        return 2 * zero_one - 1

    def unnormalize(self, value):
        zero_one = (value + 1) / 2
        value = zero_one * (self.max - self.min) + self.min
        if self.is_integer:
            value = round(value)
        return value


class Pow2(Space):
    """Power-of-2 space. Adapted from PufferLib sweep.py:76-94."""

    def __init__(self, min, max, scale, is_integer=False):
        if scale == "auto":
            scale = 0.5
        super().__init__(min, max, scale, is_integer)

    def normalize(self, value):
        zero_one = (math.log(value, 2) - math.log(self.min, 2)) / (
            math.log(self.max, 2) - math.log(self.min, 2)
        )
        return 2 * zero_one - 1

    def unnormalize(self, value):
        zero_one = (value + 1) / 2
        log_spaced = zero_one * (math.log(self.max, 2) - math.log(self.min, 2)) + math.log(
            self.min, 2
        )
        rounded = round(log_spaced)
        return 2**rounded


class Log(Space):
    """Log-normal space. Adapted from PufferLib sweep.py:96-120."""

    base: int = 10

    def __init__(self, min, max, scale, is_integer=False):
        if scale == "time":
            scale = 1 / (np.log2(max) - np.log2(min))
        elif scale == "auto":
            scale = 0.5
        super().__init__(min, max, scale, is_integer)

    def normalize(self, value):
        zero_one = (math.log(value, self.base) - math.log(self.min, self.base)) / (
            math.log(self.max, self.base) - math.log(self.min, self.base)
        )
        return 2 * zero_one - 1

    def unnormalize(self, value):
        zero_one = (value + 1) / 2
        log_spaced = zero_one * (
            math.log(self.max, self.base) - math.log(self.min, self.base)
        ) + math.log(self.min, self.base)
        value = self.base**log_spaced
        if self.is_integer:
            value = round(value)
        return value


class Logit(Space):
    """Logit-normal space. Adapted from PufferLib sweep.py:122-139."""

    base: int = 10

    def __init__(self, min, max, scale, is_integer=False):
        if scale == "auto":
            scale = 0.5
        super().__init__(min, max, scale, is_integer)

    def normalize(self, value):
        value = max(self.min, min(value, self.max))
        zero_one = (math.log(1 - value, self.base) - math.log(1 - self.min, self.base)) / (
            math.log(1 - self.max, self.base) - math.log(1 - self.min, self.base)
        )
        return 2 * zero_one - 1

    def unnormalize(self, value):
        zero_one = (value + 1) / 2
        log_spaced = zero_one * (
            math.log(1 - self.max, self.base) - math.log(1 - self.min, self.base)
        ) + math.log(1 - self.min, self.base)
        return 1 - self.base**log_spaced


# ---------------------------------------------------------------------------
# Config parsing (adapted from PufferLib sweep.py:141-182)
# ---------------------------------------------------------------------------

SWEEP_META_KEYS = frozenset({
    "method", "metric", "metric_distribution", "goal", "downsample",
    "use_gpu", "prune_pareto", "sweep_only", "max_suggestion_cost",
    "early_stop_quantile", "gpus", "max_runs",
})


def params_from_sweep_config(sweep_config: dict, only_include=None) -> dict:
    """Parse a sweep config dict into a dict of Space objects."""
    param_spaces: dict[str, Any] = {}

    if "sweep_only" in sweep_config:
        only_include = [p.strip() for p in sweep_config["sweep_only"].split(",")]

    for name, param in sweep_config.items():
        if name in SWEEP_META_KEYS:
            continue

        assert isinstance(param, dict), f"Param {name} is not a dict"
        if any(isinstance(param[k], dict) for k in param):
            param_spaces[name] = params_from_sweep_config(param, only_include)
            continue

        if only_include and not any(k in name for k in only_include):
            continue

        assert "distribution" in param
        distribution = param["distribution"]
        kwargs = dict(min=param["min"], max=param["max"], scale=param["scale"])

        if distribution == "uniform":
            space = Linear(**kwargs)
        elif distribution == "int_uniform":
            space = Linear(**kwargs, is_integer=True)
        elif distribution == "uniform_pow2":
            space = Pow2(**kwargs, is_integer=True)
        elif distribution == "log_normal":
            space = Log(**kwargs)
        elif distribution == "logit_normal":
            space = Logit(**kwargs)
        else:
            raise ValueError(f"Invalid distribution: {distribution}")

        param_spaces[name] = space

    return param_spaces


# ---------------------------------------------------------------------------
# Hyperparameters (adapted from PufferLib sweep.py:184-254)
# ---------------------------------------------------------------------------

class Hyperparameters:
    """Manages search space normalization and sampling."""

    def __init__(self, config: dict, verbose: bool = False):
        self.spaces = params_from_sweep_config(config)
        self.flat_spaces = dict(unroll_nested_dict(self.spaces))
        self.num = len(self.flat_spaces)

        self.metric = config["metric"]
        goal = config["goal"]
        assert goal in ("maximize", "minimize")
        self.optimize_direction = 1 if goal == "maximize" else -1

        self.search_centers = np.array([e.norm_mean for e in self.flat_spaces.values()])
        self.min_bounds = np.array([e.norm_min for e in self.flat_spaces.values()])
        self.max_bounds = np.array([e.norm_max for e in self.flat_spaces.values()])
        self.search_scales = np.array([e.scale for e in self.flat_spaces.values()])

        if verbose:
            print("Min random sample:")
            for name, space in self.flat_spaces.items():
                print(f"\t{name}: {space.unnormalize(max(space.norm_mean - space.scale, space.norm_min))}")
            print("Max random sample:")
            for name, space in self.flat_spaces.items():
                print(f"\t{name}: {space.unnormalize(min(space.norm_mean + space.scale, space.norm_max))}")

    def sample(self, n: int, mu=None, scale: float = 1):
        """Sample n points around mu (or search centers)."""
        if mu is None:
            mu = self.search_centers
        if len(mu.shape) == 1:
            mu = mu[None, :]
        n_input, n_dim = mu.shape
        s = scale * self.search_scales
        mu_idxs = np.random.randint(0, n_input, n)
        samples = s * (2 * np.random.rand(n, n_dim) - 1) + mu[mu_idxs]
        return np.clip(samples, self.min_bounds, self.max_bounds)

    def from_dict(self, params: dict) -> np.ndarray:
        """Convert a parameter dict to normalized array."""
        flat_params = dict(unroll_nested_dict(params))
        values = []
        for key, space in self.flat_spaces.items():
            assert key in flat_params, f"Missing hyperparameter {key}"
            val = flat_params[key]
            normed = space.normalize(val)
            values.append(normed)
        return np.array(values)

    def to_dict(self, sample: np.ndarray, fill=None) -> dict:
        """Convert a normalized array back to a parameter dict."""
        params = deepcopy(self.spaces) if fill is None else fill
        self._fill(params, self.spaces, sample)
        return params

    def _fill(self, params, spaces, flat_sample, idx=0):
        for name, space in spaces.items():
            if isinstance(space, dict):
                idx = self._fill(params[name], spaces[name], flat_sample, idx=idx)
            else:
                params[name] = spaces[name].unnormalize(flat_sample[idx])
                idx += 1
        return idx

    def get_flat_idx(self, flat_key: str):
        keys = list(self.flat_spaces.keys())
        return keys.index(flat_key) if flat_key in keys else None


# ---------------------------------------------------------------------------
# Pareto front utilities (adapted from PufferLib sweep.py:256-306)
# ---------------------------------------------------------------------------

def pareto_points(observations: list) -> tuple[list, list]:
    """Find Pareto-optimal observations (maximize score, minimize cost)."""
    if not observations:
        return [], []

    scores = np.array([e["output"] for e in observations])
    costs = np.array([e["cost"] for e in observations])

    sorted_indices = np.argsort(costs)
    pareto = []
    pareto_idxs = []
    max_score_so_far = -np.inf

    for idx in sorted_indices:
        if scores[idx] > max_score_so_far + EPSILON:
            pareto.append(observations[idx])
            pareto_idxs.append(idx)
            max_score_so_far = scores[idx]

    return pareto, pareto_idxs


def prune_pareto_front(
    pareto: list, efficiency_threshold: float = 0.5, pruning_stop_score_fraction: float = 0.98
) -> list:
    """Prune the high-cost long tail of a Pareto front.

    Adapted from PufferLib sweep.py:278-306.
    """
    if not pareto or len(pareto) < 2:
        return pareto

    sorted_pareto = sorted(pareto, key=lambda x: x["cost"])
    scores = np.array([e["output"] for e in sorted_pareto])
    costs = np.array([e["cost"] for e in sorted_pareto])
    score_range = max(scores.max() - scores.min(), EPSILON)
    cost_range = max(costs.max() - costs.min(), EPSILON)
    max_pareto_score = scores[-1] if scores.size > 0 else -np.inf

    for i in range(len(sorted_pareto) - 1, 1, -1):
        if scores[i - 1] < pruning_stop_score_fraction * max_pareto_score:
            break
        norm_score_gain = (scores[i] - scores[i - 1]) / score_range
        norm_cost_increase = (costs[i] - costs[i - 1]) / cost_range
        efficiency = norm_score_gain / (norm_cost_increase + EPSILON)
        if efficiency < efficiency_threshold:
            sorted_pareto.pop(i)
        else:
            break

    return sorted_pareto


# ---------------------------------------------------------------------------
# GP Model (adapted from PufferLib sweep.py:399-447)
# ---------------------------------------------------------------------------

if _GPYTORCH_AVAILABLE:
    class ExactGPModel(ExactGP):
        """Exact GP with Matern 3/2 + Polynomial kernels.

        Adapted from PufferLib sweep.py:399-417.
        """

        def __init__(self, train_x, train_y, likelihood, x_dim):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean()
            matern_kernel = MaternKernel(nu=1.5, ard_num_dims=x_dim)
            linear_kernel = PolynomialKernel(power=1)
            self.covar_module = ScaleKernel(AdditiveKernel(linear_kernel, matern_kernel))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        @property
        def lengthscale_range(self):
            lengthscale = self.covar_module.base_kernel.kernels[1].lengthscale.tolist()[0]
            return min(lengthscale), max(lengthscale)


def train_gp_model(model, likelihood, mll, optimizer, train_x, train_y, training_iter=50):
    """Train a GP model. Adapted from PufferLib sweep.py:425-446."""
    model.train()
    likelihood.train()
    model.set_train_data(inputs=train_x, targets=train_y, strict=False)

    loss = None
    for _ in range(training_iter):
        try:
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            loss = loss.detach()
        except gpytorch.utils.errors.NotPSDError:
            break

    model.eval()
    likelihood.eval()
    return loss.item() if loss is not None else 0


# ---------------------------------------------------------------------------
# RobustLogCostModel (adapted from PufferLib sweep.py:449-518)
# ---------------------------------------------------------------------------

class RobustLogCostModel:
    """Score ~ A + B*log(Cost) via quantile regression for early stopping."""

    def __init__(self, quantile: float = 0.3, min_num_samples: int = 30):
        self.quantile = quantile
        self.min_num_samples = min_num_samples
        self.is_fitted = False
        self.A = None
        self.B = None
        self.max_score = None
        self.max_cost = None
        self.upper_cost_threshold = None

    def _quantile_loss(self, params, x, y, q):
        a, b = params
        y_pred = a + b * x
        residuals = y - y_pred
        return np.sum(np.maximum(q * residuals, (q - 1) * residuals))

    def fit(self, observations: list, upper_cost_threshold=None):
        self.is_fitted = False
        scores = np.array([e["output"] for e in observations])
        costs = np.array([e["cost"] for e in observations])
        self.max_score = scores.max()
        self.upper_cost_threshold = upper_cost_threshold or costs.max()

        valid_indices = (costs > EPSILON) & np.isfinite(scores)
        if np.sum(valid_indices) < self.min_num_samples:
            return

        y = scores[valid_indices]
        c = costs[valid_indices]
        x_log_c = np.log(c)

        try:
            b_init, a_init = np.polyfit(x_log_c, y, 1)
        except np.linalg.LinAlgError:
            b_init, a_init = 0.0, np.mean(y)

        res = scipy_minimize(
            self._quantile_loss,
            x0=[a_init, b_init],
            args=(x_log_c, y, self.quantile),
            method="Nelder-Mead",
        )

        self.A, self.B = res.x
        self.is_fitted = True

    def get_threshold(self, cost: float, min_cost_fraction: float = 0.3, abs_min_cost: float = 10):
        if not self.is_fitted or self.upper_cost_threshold is None:
            return -np.inf
        min_allowed_cost = self.upper_cost_threshold * min_cost_fraction + abs_min_cost
        if cost < min_allowed_cost:
            return -np.inf
        if cost > 1.2 * self.upper_cost_threshold:
            return 0.9 * self.max_score
        return self.A + self.B * np.log(cost)


# ---------------------------------------------------------------------------
# Protein sweep (adapted from PufferLib sweep.py:522-966)
# ---------------------------------------------------------------------------

class Protein:
    """GP-based hyperparameter optimizer with Pareto tracking.

    Adapted from PufferLib sweep.py:522-966. PufferLib-specific references
    (env configs, wandb, _C) have been removed. The interface accepts a
    ProofForge-style sweep config dict.
    """

    def __init__(
        self,
        sweep_config: dict,
        max_suggestion_cost: float = 3600,
        num_random_samples: int = 10,
        num_keep_top_obs: int = 5,
        global_search_scale: float = 1,
        suggestions_per_pareto: int = 256,
        expansion_rate: float = 0.1,
        gp_training_iter: int = 50,
        gp_learning_rate: float = 0.001,
        gp_max_obs: int = 750,
        infer_batch_size: int = 4096,
        optimizer_reset_frequency: int = 50,
        use_gpu: bool = True,
        prune_pareto: bool = True,
    ):
        if not _GPYTORCH_AVAILABLE:
            raise ImportError(
                "gpytorch is required for Protein sweep. "
                "Install with: pip install -e 'python/[sweep]'"
            )

        _use_gpu = sweep_config.get("use_gpu", use_gpu)
        _prune_pareto = sweep_config.get("prune_pareto", prune_pareto)
        _max_suggestion_cost = sweep_config.get("max_suggestion_cost", max_suggestion_cost)

        self.device = torch.device(
            "cuda:0" if _use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.hyperparameters = Hyperparameters(sweep_config)
        self.metric_distribution = sweep_config.get("metric_distribution", "linear")
        self.global_search_scale = global_search_scale
        self.suggestions_per_pareto = suggestions_per_pareto
        self.max_suggestion_cost = _max_suggestion_cost
        self.expansion_rate = expansion_rate
        self.gp_training_iter = gp_training_iter
        self.gp_learning_rate = gp_learning_rate
        self.optimizer_reset_frequency = optimizer_reset_frequency
        self.prune_pareto = _prune_pareto

        self.success_observations: list[dict] = []
        self.failure_observations: list[dict] = []
        self.num_keep_top_obs = num_keep_top_obs
        self.top_observations: list[dict] = []

        self.suggestion_idx = 0
        self.min_score, self.max_score = math.inf, -math.inf
        self.log_c_min, self.log_c_max = math.inf, -math.inf

        self.sobol = Sobol(d=self.hyperparameters.num, scramble=True)
        self.num_random_samples = num_random_samples

        self.target_cost_ratio: list[float] = []
        self._running_target_buffer: deque = deque(maxlen=30)

        self.gp_max_obs = gp_max_obs
        self.infer_batch_size = infer_batch_size

        self.use_success_prob = sweep_config.get("downsample", 5) == 1
        if _SKLEARN_AVAILABLE:
            self.success_classifier = LogisticRegression(class_weight="balanced")
        else:
            self.success_classifier = None

        early_stop_q = sweep_config.get("early_stop_quantile", 0.3)
        self.stop_threshold_model = RobustLogCostModel(quantile=early_stop_q)
        self.upper_cost_threshold = -np.inf

        with default_tensor_dtype(torch.float64):
            noise_prior = LogNormalPrior(math.log(1e-2), 0.5)
            dummy_x = torch.ones((1, self.hyperparameters.num), device=self.device)
            dummy_y = torch.zeros(1, device=self.device)

            self.likelihood_score = GaussianLikelihood(
                noise_prior=deepcopy(noise_prior)
            ).to(self.device)
            self.gp_score = ExactGPModel(
                dummy_x, dummy_y, self.likelihood_score, self.hyperparameters.num
            ).to(self.device)
            self.mll_score = ExactMarginalLogLikelihood(
                self.likelihood_score, self.gp_score
            ).to(self.device)
            self.score_opt = torch.optim.Adam(
                self.gp_score.parameters(), lr=self.gp_learning_rate, amsgrad=True
            )

            self.likelihood_cost = GaussianLikelihood(
                noise_prior=deepcopy(noise_prior)
            ).to(self.device)
            self.gp_cost = ExactGPModel(
                dummy_x, dummy_y, self.likelihood_cost, self.hyperparameters.num
            ).to(self.device)
            self.mll_cost = ExactMarginalLogLikelihood(
                self.likelihood_cost, self.gp_cost
            ).to(self.device)
            self.cost_opt = torch.optim.Adam(
                self.gp_cost.parameters(), lr=self.gp_learning_rate, amsgrad=True
            )

            self.gp_params_buffer = torch.empty(
                self.gp_max_obs, self.hyperparameters.num, device=self.device
            )
            self.gp_score_buffer = torch.empty(self.gp_max_obs, device=self.device)
            self.gp_cost_buffer = torch.empty(self.gp_max_obs, device=self.device)
            self.infer_batch_buffer = torch.empty(
                self.infer_batch_size, self.hyperparameters.num, device=self.device
            )

    # -- Internal helpers --

    def _filter_near_duplicates(self, inputs, duplicate_threshold=EPSILON):
        """Remove near-duplicate points. Adapted from PufferLib sweep.py:621-636."""
        if len(inputs) < 2:
            return np.arange(len(inputs))
        tree = KDTree(inputs)
        to_keep = np.ones(len(inputs), dtype=bool)
        for i in range(len(inputs) - 1, -1, -1):
            if to_keep[i]:
                nearby = tree.query_ball_point(inputs[i], r=duplicate_threshold)
                nearby.remove(i)
                if nearby:
                    to_keep[nearby] = False
        return np.where(to_keep)[0]

    def _sample_observations(self, max_size=None, recent_ratio=0.5):
        """Sample observations for GP training. Adapted from PufferLib sweep.py:638-679."""
        if not self.success_observations:
            return []
        observations = self.success_observations.copy()
        y = np.array([e["output"] for e in observations])
        self.min_score, self.max_score = y.min(), y.max()
        c = np.array([e["cost"] for e in observations])
        log_c = np.log(np.maximum(c, EPSILON))
        self.log_c_min = log_c.min()
        self.log_c_max = np.quantile(log_c, 0.97)

        if len(observations) < 100 and self.failure_observations:
            for e in self.failure_observations:
                e["output"] = self.min_score
            observations = self.failure_observations + observations

        params = np.array(
            [np.append(e["input"], [e["output"], e["cost"]]) for e in observations]
        )
        dedup_indices = self._filter_near_duplicates(params)
        observations = [observations[i] for i in dedup_indices]

        if max_size is None:
            max_size = self.gp_max_obs
        if len(observations) <= max_size:
            return observations

        recent_size = int(recent_ratio * max_size)
        recent_obs = observations[-recent_size:]
        older_obs = observations[:-recent_size]
        num_to_sample = max_size - recent_size
        random_sample_obs = random.sample(older_obs, num_to_sample)
        return random_sample_obs + recent_obs

    def _train_gp_models(self):
        """Train score and cost GPs. Adapted from PufferLib sweep.py:681-710."""
        if not self.success_observations:
            return 0, 0
        sampled = self._sample_observations(max_size=self.gp_max_obs)
        num_sampled = len(sampled)

        params = np.array([e["input"] for e in sampled])
        params_tensor = self.gp_params_buffer[:num_sampled]
        params_tensor.copy_(torch.from_numpy(params))

        y = np.array([e["output"] for e in sampled])
        y_norm = (y - self.min_score) / (abs(self.max_score - self.min_score) + EPSILON)
        y_norm_tensor = self.gp_score_buffer[:num_sampled]
        y_norm_tensor.copy_(torch.from_numpy(y_norm))

        c = np.array([e["cost"] for e in sampled])
        log_c = np.log(np.maximum(c, EPSILON))
        log_c_norm = (log_c - self.log_c_min) / (self.log_c_max - self.log_c_min + EPSILON)
        log_c_norm_tensor = self.gp_cost_buffer[:num_sampled]
        log_c_norm_tensor.copy_(torch.from_numpy(log_c_norm))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score_loss = train_gp_model(
                self.gp_score, self.likelihood_score, self.mll_score,
                self.score_opt, params_tensor, y_norm_tensor,
                training_iter=self.gp_training_iter,
            )
            cost_loss = train_gp_model(
                self.gp_cost, self.likelihood_cost, self.mll_cost,
                self.cost_opt, params_tensor, log_c_norm_tensor,
                training_iter=self.gp_training_iter,
            )
        return score_loss, cost_loss

    def _get_top_obs_params(self):
        if not self.top_observations:
            return np.array([])
        return np.array([e["input"] for e in self.top_observations])

    def _sample_target_cost_ratio(
        self, expansion_rate, target_ratios=(0.16, 0.32, 0.48, 0.64, 0.8, 1.0)
    ):
        if not self.target_cost_ratio:
            self.target_cost_ratio = list(target_ratios)
            random.shuffle(self.target_cost_ratio)
        target_ratio = np.clip(self.target_cost_ratio.pop() + 0.1 * np.random.randn(), 0, 1)
        return (1 + expansion_rate) * target_ratio

    # -- Public API --

    def suggest(self, fill=None) -> tuple[dict, dict]:
        """Suggest next hyperparameter configuration.

        Returns (params_dict, info_dict). Adapted from PufferLib sweep.py:739-889.
        """
        info: dict = {}
        self.suggestion_idx += 1

        if self.suggestion_idx <= self.num_random_samples:
            zero_one = self.sobol.random(1)[0]
            suggestion = 2 * zero_one - 1
            return self.hyperparameters.to_dict(suggestion, fill), info

        score_loss, cost_loss = self._train_gp_models()

        if (
            self.optimizer_reset_frequency
            and self.suggestion_idx % self.optimizer_reset_frequency == 0
        ):
            self.score_opt = torch.optim.Adam(
                self.gp_score.parameters(), lr=self.gp_learning_rate, amsgrad=True
            )
            self.cost_opt = torch.optim.Adam(
                self.gp_cost.parameters(), lr=self.gp_learning_rate, amsgrad=True
            )

        pareto_front, _ = pareto_points(self.success_observations)
        pruned_front = prune_pareto_front(pareto_front)
        pareto_observations = pruned_front if self.prune_pareto else pareto_front

        if not pareto_observations:
            suggestion = self.hyperparameters.search_centers
            return self.hyperparameters.to_dict(suggestion, fill), info

        if self.upper_cost_threshold < 0:
            self.upper_cost_threshold = pruned_front[-1]["cost"]
        elif self.upper_cost_threshold < pruned_front[-1]["cost"]:
            self.upper_cost_threshold *= 1.01
        self.stop_threshold_model.fit(self.success_observations, self.upper_cost_threshold)

        search_centers = np.stack([e["input"] for e in pareto_observations])
        if self.top_observations:
            search_centers = np.vstack([search_centers, self._get_top_obs_params()])

        suggestions = self.hyperparameters.sample(
            len(search_centers) * self.suggestions_per_pareto, mu=search_centers
        )

        dedup_indices = self._filter_near_duplicates(suggestions)
        suggestions = suggestions[dedup_indices]
        if len(suggestions) == 0:
            return self.suggest(fill)

        # Batch GP predictions
        gp_y_norm_list, gp_log_c_norm_list = [], []
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(0, len(suggestions), self.infer_batch_size):
                batch_numpy = suggestions[i : i + self.infer_batch_size]
                current_batch_size = len(batch_numpy)
                batch_tensor = self.infer_batch_buffer[:current_batch_size]
                batch_tensor.copy_(torch.from_numpy(batch_numpy))
                try:
                    pred_y_mean = self.likelihood_score(self.gp_score(batch_tensor)).mean.cpu()
                    pred_c_mean = self.likelihood_cost(self.gp_cost(batch_tensor)).mean.cpu()
                except RuntimeError:
                    pred_y_mean = torch.zeros(current_batch_size)
                    pred_c_mean = torch.zeros(current_batch_size)
                gp_y_norm_list.append(pred_y_mean)
                gp_log_c_norm_list.append(pred_c_mean)

        gp_y_norm = torch.cat(gp_y_norm_list).numpy()
        gp_log_c_norm = torch.cat(gp_log_c_norm_list).numpy()

        gp_y = gp_y_norm * (self.max_score - self.min_score) + self.min_score
        gp_log_c = gp_log_c_norm * (self.log_c_max - self.log_c_min) + self.log_c_min
        gp_c = np.exp(gp_log_c)

        suggestion_scores = self.hyperparameters.optimize_direction * gp_y_norm
        max_c_mask = gp_c < self.max_suggestion_cost
        target_cost = self._sample_target_cost_ratio(self.expansion_rate)
        weight = 1 - abs(target_cost - gp_log_c_norm)
        suggestion_scores *= max_c_mask * weight

        best_idx = np.argmax(suggestion_scores)
        info = dict(
            cost=gp_c[best_idx].item(),
            score=gp_y[best_idx].item(),
            rating=suggestion_scores[best_idx].item(),
            score_loss=score_loss,
            cost_loss=cost_loss,
        )

        best = suggestions[best_idx]
        return self.hyperparameters.to_dict(best, fill), info

    def logit_transform(self, value, epsilon=1e-9):
        value = np.clip(value, epsilon, 1 - epsilon)
        logit = math.log(value / (1 - value))
        return np.clip(logit, -5, 100)

    def observe(self, hypers: dict, score: float, cost: float, is_failure: bool = False):
        """Record an observation. Adapted from PufferLib sweep.py:896-935."""
        params = self.hyperparameters.from_dict(hypers)

        if self.metric_distribution == "percentile":
            score = self.logit_transform(score)

        new_observation = dict(input=params, output=score, cost=cost, is_failure=is_failure)

        if is_failure or not np.isfinite(score) or np.isnan(score):
            new_observation["is_failure"] = True
            self.failure_observations.append(new_observation)
            return

        if self.success_observations:
            success_params = np.stack([e["input"] for e in self.success_observations])
            dist = np.linalg.norm(params - success_params, axis=1)
            same = np.where(dist < EPSILON)[0]
            if len(same) > 0:
                self.success_observations[same[0]] = new_observation
                return

        self.success_observations.append(new_observation)

        if len(self.top_observations) < self.num_keep_top_obs:
            self.top_observations.append(new_observation)
            self.top_observations.sort(key=lambda x: x["output"], reverse=True)
        elif score > self.top_observations[-1]["output"]:
            self.top_observations.pop()
            self.top_observations.append(new_observation)
            self.top_observations.sort(key=lambda x: x["output"], reverse=True)

    def get_early_stop_threshold(self, cost: float) -> float:
        return self.stop_threshold_model.get_threshold(cost)

    def should_stop(self, score: float, cost: float) -> bool:
        """Check if a run should be stopped early."""
        threshold = self.get_early_stop_threshold(cost)
        if self.metric_distribution == "percentile":
            score = self.logit_transform(score)
        return score < threshold

    def early_stop(self, logs: dict) -> bool:
        """Check for NaN loss or poor performance. Adapted from PufferLib sweep.py:948-965."""
        if "loss" in logs:
            for v in logs["loss"].values():
                if np.isnan(v):
                    logs["is_loss_nan"] = True
                    return True

        if "cost" not in logs or "score" not in logs:
            return False

        score, cost = logs["score"], logs["cost"]
        self._running_target_buffer.append(score)
        target_running_mean = np.mean(self._running_target_buffer)
        threshold = self.get_early_stop_threshold(cost)
        logs["early_stop_threshold"] = max(threshold, -5)
        if self.should_stop(max(target_running_mean, score), cost):
            logs["is_loss_nan"] = False
            return True
        return False
