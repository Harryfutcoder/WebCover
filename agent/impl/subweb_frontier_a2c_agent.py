"""SubWeb-Frontier-A2C.

This agent uses a frontier coverage objective with an online
actor-critic optimizer. The policy is Markovian over an augmented Web state:
the current Web state, the run coverage summary U_t, and budget/history
features. The web adaptation uses a masked policy over available actions.
"""

from __future__ import annotations

import logging
import os
import hashlib
import math
import copy
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim
from torch.distributions import Categorical

from action.impl.restart_action import RestartAction
from action.impl.random_input_action import RandomInputAction
from action.impl.random_select_action import RandomSelectAction
from action.web_action import WebAction
from agent.agent import Agent
from agent.impl.subweb_action_context_features import context_feature_dim
from agent.impl.subweb_frontier_coverage import (
    FrontierCoverageTracker,
    extract_graph_node_id,
    graph_edge_prefix,
    graph_edge_label_mode_from_env,
    graph_node_mode_from_env,
    unit_class,
)
from agent.impl.subweb_action_coverage_features import (
    FEATURE_NAMES as ACTION_COVERAGE_FEATURE_NAMES,
    SubWebActionCoverageFeatures,
    action_coverage_feature_dim,
)
from agent.impl.subweb_learning_signal_diagnostics import (
    format_learning_signal_diagnostics,
    parameter_grad_norm,
)
from agent.impl.subweb_structural_action_features import structural_feature_dim
from agent.impl.subweb_padded_actions import (
    PaddedActionBatch,
    action_kind,
    action_signature,
    build_padded_action_batch,
)
from agent.impl.subweb_policy_diagnostics import SubWebPolicyDiagnostics, format_policy_brief
from agent.impl.subweb_recent_action_summary import (
    append_recent_action,
    recent_action_summary,
    recent_action_summary_dim,
)
from exceptions import NoActionsException
from model.subweb_frontier_a2c import CandidateQHead, FrontierCritic, MaskedCandidateActor, masked_mean
from observation.observer import Observer
from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState
from state.impl.action_execute_failed_state import ActionExecuteFailedState
from state.impl.out_of_domain_state import OutOfDomainState
from state.impl.same_url_state import SameUrlState
from state.web_state import WebState
from utils import clip_reward_value, instantiate_class_by_module_and_class_name, read_reward_clip_abs

logger = logging.getLogger(__name__)

_COVERAGE_TYPE_BUCKETS = (
    "context",
    "field",
    "submit",
    "main_action",
    "workflow_link",
    "structure",
    "row_action",
    "other",
)
_COVERAGE_SKETCH_SIZE = 16
_PURE_TIME_CONTEXT_DIM = 2


def _read_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _read_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _read_bool_env(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw not in {"0", "false", "no", "off"}


def _read_policy_input_mode(params: dict) -> str:
    raw = (
        os.environ.get("WEBTEST_SUBWEB_POLICY_INPUT_MODE", params.get("policy_input_mode", "augmented"))
        .strip()
        .lower()
        .replace("-", "_")
    )
    if raw in {"", "default", "full", "augmented", "coverage_augmented"}:
        return "augmented"
    if raw in {"coverage", "coverage_summary", "coverage_state"}:
        return "coverage_summary"
    if raw in {"pure", "pure_m", "pure_nm", "state_time", "markov_time", "time"}:
        return "pure"
    if raw in {"state_only", "no_context", "none", "zero"}:
        return "state_only"
    return "augmented"


def _read_reward_mode(params: dict) -> str:
    raw = (
        os.environ.get("WEBTEST_SUBWEB_REWARD_MODE", params.get("reward_mode", "marginal"))
        .strip()
        .lower()
        .replace("-", "_")
    )
    if raw in {"", "default", "main", "marginal", "delta_f", "delta_f_cov", "hard_marginal", "strict_delta_f", "graph_delta_f"}:
        return "marginal"
    if raw in {"count", "count_decay", "local_count_decay", "transition_count_decay", "curiosity"}:
        return "count_decay"
    if raw in {"webqt", "webqt_proxy"}:
        return "webqt"
    if raw in {"webrled", "webrled_proxy"}:
        return "webrled"
    if raw in {"qexplore", "qexplore_proxy", "q_explore"}:
        return "qexplore"
    raise ValueError(f"Unsupported WEBTEST_SUBWEB_REWARD_MODE={raw!r}")


def _read_optimizer_mode(params: dict) -> str:
    raw = (
        os.environ.get("WEBTEST_SUBWEB_OPTIMIZER", params.get("optimizer_mode", "a2c"))
        .strip()
        .lower()
        .replace("-", "_")
    )
    if raw in {"", "default", "a2c", "actor_critic", "masked_a2c", "masked_online_a2c"}:
        return "a2c"
    if raw in {"q", "qlearn", "q_learning", "qlearning", "candidate_q", "candidate_qlearning"}:
        return "qlearning"
    raise ValueError(f"Unsupported WEBTEST_SUBWEB_OPTIMIZER={raw!r}")


def _reward_action_type(action: Optional[WebAction]) -> str:
    if action is None:
        return "none"
    if isinstance(action, RestartAction):
        return "restart"
    if isinstance(action, RandomInputAction):
        return "input"
    if isinstance(action, RandomSelectAction):
        return "select"
    return str(getattr(action, "action_type", "") or type(action).__name__ or "default").lower()


def _is_failure_state(web_state: WebState) -> bool:
    return isinstance(web_state, (OutOfDomainState, ActionExecuteFailedState, SameUrlState))


def _is_qexplore_invalid_state(web_state: WebState) -> bool:
    # QExplore's source reward treats out-of-domain / failed execution as invalid.
    # Same-URL transitions remain valid because repeated in-page actions are still
    # credited through the action execution count.
    return isinstance(web_state, (OutOfDomainState, ActionExecuteFailedState))


class SubWebFrontierA2CAgent(Agent):
    LOG_PREFIX = "SubWeb-Frontier-A2C"
    POLICY_FAMILY = "markov_augmented_state"
    log_prefix = LOG_PREFIX
    policy_family = POLICY_FAMILY

    def __init__(self, params: dict):
        self.log_prefix = getattr(self, "LOG_PREFIX", self.LOG_PREFIX)
        self.policy_family = getattr(self, "POLICY_FAMILY", self.POLICY_FAMILY)
        self.policy_input_mode = _read_policy_input_mode(params)
        if self.policy_input_mode == "pure":
            self.policy_family = f"{self.policy_family}_pure_input"
        elif self.policy_input_mode == "state_only":
            self.policy_family = f"{self.policy_family}_state_only_input"
        self.base_input_dim = int(params.get("input_dim", 52))
        self.use_structural_action_features = _read_bool_env("WEBTEST_SUBWEB_STRUCTURAL_ACTION_FEATURES", True)
        self.structural_feature_dim = structural_feature_dim() if self.use_structural_action_features else 0
        requested_action_coverage_features = _read_bool_env("WEBTEST_SUBWEB_ACTION_COVERAGE_FEATURES", True)
        self.use_action_coverage_features = (
            requested_action_coverage_features
            and self.policy_input_mode == "augmented"
        )
        self.action_coverage_feature_dim = (
            action_coverage_feature_dim() if self.use_action_coverage_features else 0
        )
        self.input_dim = (
            self.base_input_dim
            + context_feature_dim()
            + self.structural_feature_dim
            + self.action_coverage_feature_dim
        )
        self.max_actions = _read_int_env("WEBTEST_MAX_ACTIONS", int(params.get("max_actions", 512)))
        hard_cap = _read_int_env("WEBTEST_MAX_ACTIONS_HARD_CAP", int(params.get("max_actions_hard_cap", 1024)))
        self.max_actions = max(1, min(self.max_actions, hard_cap))
        self.rollout_len = max(1, _read_int_env("WEBTEST_ROLLOUT_LEN", int(params.get("rollout_len", 32))))
        gamma_default = _read_float_env("WEBTEST_GAMMA", float(params.get("gamma", 1.0)))
        self.gamma = max(0.0, min(1.0, _read_float_env("WEBTEST_A2C_GAMMA", gamma_default)))
        self.gae_lambda = max(
            0.0,
            min(1.0, _read_float_env("WEBTEST_A2C_GAE_LAMBDA", float(params.get("gae_lambda", 0.95)))),
        )
        self.entropy_coef = _read_float_env("WEBTEST_A2C_ENTROPY_COEF", float(params.get("entropy_coef", 0.01)))
        self.policy_temperature = max(
            0.05,
            _read_float_env("WEBTEST_A2C_POLICY_TEMPERATURE", float(params.get("policy_temperature", 1.0))),
        )
        self.value_coef = _read_float_env("WEBTEST_A2C_VALUE_COEF", float(params.get("value_coef", 0.5)))
        self.max_grad_norm = _read_float_env("WEBTEST_A2C_MAX_GRAD_NORM", float(params.get("max_grad_norm", 0.5)))
        self.separate_grad_clip = _read_bool_env(
            "WEBTEST_A2C_SEPARATE_GRAD_CLIP",
            bool(params.get("separate_grad_clip", True)),
        )
        self.learning_rate = _read_float_env("WEBTEST_A2C_LR", float(params.get("learning_rate", 1e-3)))
        self.actor_learning_rate = max(
            1e-8,
            _read_float_env("WEBTEST_A2C_ACTOR_LR", self.learning_rate),
        )
        self.critic_learning_rate = max(
            1e-8,
            _read_float_env("WEBTEST_A2C_CRITIC_LR", self.learning_rate),
        )
        self.update_epochs = max(1, _read_int_env("WEBTEST_A2C_UPDATE_EPOCHS", int(params.get("update_epochs", 1))))
        self.advantage_estimator = (
            os.environ.get("WEBTEST_A2C_ADVANTAGE_ESTIMATOR", params.get("advantage_estimator", "n_step"))
            .strip()
            .lower()
            .replace("-", "_")
        )
        if self.advantage_estimator == "one_step":
            self.advantage_estimator = "td"
        if self.advantage_estimator == "return_to_go":
            self.advantage_estimator = "n_step"
        if self.advantage_estimator in {"td_lambda", "gae_lambda"}:
            self.advantage_estimator = "gae"
        if self.advantage_estimator not in {"td", "n_step", "gae"}:
            self.advantage_estimator = "n_step"
        self.reward_mode = _read_reward_mode(params)
        self.reward_mode_is_proxy = self.reward_mode in {"webqt", "webrled"}
        self.optimizer_mode = _read_optimizer_mode(params)
        self.qlearning_epsilon = max(
            0.0,
            min(1.0, _read_float_env("WEBTEST_QLEARNING_EPSILON", float(params.get("qlearning_epsilon", 0.10)))),
        )
        self.qlearning_epsilon_final = max(
            0.0,
            min(
                1.0,
                _read_float_env(
                    "WEBTEST_QLEARNING_EPSILON_FINAL",
                    float(params.get("qlearning_epsilon_final", self.qlearning_epsilon)),
                ),
            ),
        )
        self.qlearning_epsilon_decay_steps = max(
            1,
            _read_int_env("WEBTEST_QLEARNING_EPSILON_DECAY_STEPS", int(params.get("qlearning_epsilon_decay_steps", 1))),
        )
        self.qlearning_target_sync_interval = max(
            1,
            _read_int_env("WEBTEST_QLEARNING_TARGET_SYNC_INTERVAL", int(params.get("qlearning_target_sync_interval", 25))),
        )
        self.qlearning_learning_rate = max(
            1e-8,
            _read_float_env(
                "WEBTEST_QLEARNING_LR",
                float(params.get("qlearning_lr", params.get("candidate_q_aux_lr", self.actor_learning_rate))),
            ),
        )
        if self._qlearning_enabled():
            self.policy_family = f"{self.policy_family}_candidate_q_learning"
        self.count_decay_scale = max(
            0.0,
            _read_float_env("WEBTEST_SUBWEB_COUNT_DECAY_SCALE", float(params.get("count_decay_scale", 1.0))),
        )
        self.count_decay_power = max(
            1e-6,
            _read_float_env("WEBTEST_SUBWEB_COUNT_DECAY_POWER", float(params.get("count_decay_power", 0.5))),
        )
        self.webqt_w_loc = _read_float_env("WEBTEST_SUBWEB_WEBQT_W_LOC", _read_float_env("WEBQT_W_LOC", 10.0))
        self.webqt_w_attention = _read_float_env(
            "WEBTEST_SUBWEB_WEBQT_W_ATTENTION",
            _read_float_env("WEBQT_W_ATTENTION", 50.0),
        )
        self.webqt_w_freq = _read_float_env("WEBTEST_SUBWEB_WEBQT_W_FREQ", _read_float_env("WEBQT_W_FREQ", 5.0))
        self.webqt_w_explore = _read_float_env(
            "WEBTEST_SUBWEB_WEBQT_W_EXPLORE",
            _read_float_env("WEBQT_W_EXPLORE", 5.0),
        )
        self.webrled_w_global = _read_float_env(
            "WEBTEST_SUBWEB_WEBRLED_W_GLOBAL",
            _read_float_env("WEBRLED_W_GLOBAL", 0.6),
        )
        self.webrled_w_episode = _read_float_env(
            "WEBTEST_SUBWEB_WEBRLED_W_EPISODE",
            _read_float_env("WEBRLED_W_EPISODE", 0.3),
        )
        self.webrled_w_transition = _read_float_env(
            "WEBTEST_SUBWEB_WEBRLED_W_TRANSITION",
            _read_float_env("WEBRLED_W_TRANSITION", 0.1),
        )
        self.qexplore_valid_reward_scale = _read_float_env("WEBTEST_SUBWEB_QEXPLORE_VALID_REWARD_SCALE", 1.0)
        self.qexplore_invalid_reward = _read_float_env("WEBTEST_SUBWEB_QEXPLORE_INVALID_REWARD", -1.0)
        # QExplore uses 500 as the initial Q-table value, not as the executed
        # step reward. The reward proxy below follows the source reward:
        # valid action reward = 1 / global action execution count.
        self.qexplore_initial_q = _read_float_env("WEBTEST_SUBWEB_QEXPLORE_INITIAL_Q", 500.0)
        self.policy_baseline_mode = (
            os.environ.get("WEBTEST_A2C_POLICY_BASELINE", params.get("policy_baseline", "critic"))
            .strip()
            .lower()
            .replace("-", "_")
        )
        if self.policy_baseline_mode in {"mean", "batch", "batch_mean", "return_mean"}:
            self.policy_baseline_mode = "batch_mean"
        elif self.policy_baseline_mode in {"none", "zero", "no_baseline"}:
            self.policy_baseline_mode = "none"
        else:
            self.policy_baseline_mode = "critic"
        self.critic_pooling = str(
            os.environ.get("WEBTEST_A2C_CRITIC_POOLING", params.get("critic_pooling", "mean"))
        ).strip().lower()
        if self.critic_pooling not in {"mean", "mean_max", "mean+max"}:
            self.critic_pooling = "mean"
        self.normalize_advantages = _read_bool_env(
            "WEBTEST_A2C_NORMALIZE_ADVANTAGE",
            bool(params.get("normalize_advantages", True)),
        )
        self.advantage_norm_min_std = max(
            0.0,
            _read_float_env(
                "WEBTEST_A2C_ADV_NORM_MIN_STD",
                float(params.get("advantage_norm_min_std", 0.1)),
            ),
        )
        default_min_actor_update_steps = min(
            int(params.get("min_actor_update_steps", 2)),
            self.rollout_len,
        )
        self.min_actor_update_steps = max(
            1,
            _read_int_env(
                "WEBTEST_A2C_MIN_ACTOR_UPDATE_STEPS",
                default_min_actor_update_steps,
            ),
        )

        self.transformer = instantiate_class_by_module_and_class_name(
            params["transformer_module"], params["transformer_class"],
        )
        hidden_dim = int(params.get("hidden_dim", 128))
        self.full_coverage_summary_dim = (
            1
            + len(_COVERAGE_TYPE_BUCKETS)
            + 3
            + 1
            + 3
            + _COVERAGE_SKETCH_SIZE
            + recent_action_summary_dim()
        )
        if self.policy_input_mode == "state_only":
            self.coverage_summary_dim = 0
        elif self.policy_input_mode == "pure":
            self.coverage_summary_dim = _PURE_TIME_CONTEXT_DIM
        else:
            self.coverage_summary_dim = self.full_coverage_summary_dim
        self.actor = MaskedCandidateActor(
            self.input_dim,
            hidden_dim=hidden_dim,
            context_dim=self.coverage_summary_dim,
        )
        self.critic = FrontierCritic(self.critic_input_dim, hidden_dim=hidden_dim)
        self.candidate_q_aux = _read_bool_env(
            "WEBTEST_CANDIDATE_Q_AUX",
            bool(params.get("candidate_q_aux", False)),
        )
        self.candidate_q_aux_coef = max(
            0.0,
            _read_float_env(
                "WEBTEST_CANDIDATE_Q_AUX_COEF",
                float(params.get("candidate_q_aux_coef", 0.05)),
            ),
        )
        self.candidate_q_aux_target = (
            os.environ.get("WEBTEST_CANDIDATE_Q_AUX_TARGET", params.get("candidate_q_aux_target", "critic_target"))
            .strip()
            .lower()
            .replace("-", "_")
        )
        if self.candidate_q_aux_target not in {
            "critic_target",
            "critic_target_norm",
            "return",
            "return_norm",
            "td",
            "td_norm",
            "advantage",
            "advantage_norm",
        }:
            self.candidate_q_aux_target = "critic_target"
        self.candidate_q_learning_rate = max(
            1e-8,
            _read_float_env(
                "WEBTEST_CANDIDATE_Q_AUX_LR",
                float(params.get("candidate_q_aux_lr", self.actor_learning_rate)),
            ),
        )
        self.candidate_q_head = (
            CandidateQHead(
                getattr(self.actor, "embedding_dim", 64),
                hidden_dim=hidden_dim,
                recurrent_hidden_dim=0,
            )
            if self._qlearning_enabled() or (self.candidate_q_aux and self.candidate_q_aux_coef > 0.0)
            else None
        )
        self.target_actor = None
        self.target_candidate_q_head = None
        if self._qlearning_enabled():
            self._sync_qlearning_target()
        self.optimizer = self._make_optimizer()

        self.coverage_tracker = FrontierCoverageTracker("frontier_uniform")
        self.graph_state_list: List[WebState] = []
        self.reward_transition_counts = defaultdict(int)
        self.reward_action_counts = defaultdict(int)
        self.reward_qexplore_action_counts = defaultdict(int)
        self.reward_action_type_counts = defaultdict(int)
        self.reward_node_visit_counts = defaultdict(int)
        self.reward_episode_nodes = set()
        self.rollout_buffer: List[Dict[str, torch.Tensor]] = []
        self.pending_step: Optional[Dict[str, object]] = None
        self.total_steps = 0
        self.total_updates = 0
        self.last_reward = 0.0
        self.last_train_reward = 0.0
        self.last_finalized_transition: Optional[Dict[str, object]] = None
        self.recent_rewards = deque(maxlen=16)
        self.recent_action_history = deque(maxlen=8)
        self.zero_gain_streak = 0
        self.initial_frontier_seeded = False
        self.start_time = datetime.now()
        self.alive_time = params.get("alive_time", 3600)
        self.reward_clip_abs = read_reward_clip_abs(0.0)
        self.graph_residual_aux = _read_bool_env(
            "WEBTEST_GRAPH_RESIDUAL_AUX",
            bool(params.get("graph_residual_aux", False)),
        )
        graph_residual_aux_coef_default = float(
            params.get(
                "graph_residual_aux_coef",
                0.2 if self.graph_residual_aux else 0.0,
            )
        )
        self.graph_residual_aux_coef = max(
            0.0,
            _read_float_env("WEBTEST_GRAPH_RESIDUAL_AUX_COEF", graph_residual_aux_coef_default),
        )
        default_graph_residual_aux_target = params.get(
            "graph_residual_aux_target",
            "uncovered",
        )
        self.graph_residual_aux_target_mode = (
            os.environ.get("WEBTEST_GRAPH_RESIDUAL_AUX_TARGET", default_graph_residual_aux_target)
            .strip()
            .lower()
            .replace("-", "_")
            or str(default_graph_residual_aux_target)
        )
        if self.graph_residual_aux_target_mode in {"residual", "graph_residual", "exact_graph"}:
            self.graph_residual_aux_target_mode = "graph"
        self.graph_residual_aux_target_power = max(
            0.05,
            _read_float_env(
                "WEBTEST_GRAPH_RESIDUAL_AUX_TARGET_POWER",
                float(params.get("graph_residual_aux_target_power", 1.0)),
            ),
        )
        default_target_distribution = params.get(
            "graph_residual_aux_distribution",
            "uniform" if self.graph_residual_aux_target_mode == "uncovered" else "max_only",
        )
        self.graph_residual_aux_distribution = (
            os.environ.get(
                "WEBTEST_GRAPH_RESIDUAL_AUX_DISTRIBUTION",
                default_target_distribution,
            )
            .strip()
            .lower()
            .replace("-", "_")
            or str(default_target_distribution)
        )
        if self.graph_residual_aux_distribution not in {"uniform", "proportional", "max_only", "greedy"}:
            raise ValueError("Unknown graph_residual_aux_distribution")
        default_tie_break_coef = params.get(
            "graph_residual_aux_tie_break_coef",
            0.0,
        )
        self.graph_residual_aux_tie_break_coef = max(
            0.0,
            _read_float_env(
                "WEBTEST_GRAPH_RESIDUAL_AUX_TIE_BREAK_COEF",
                float(default_tie_break_coef),
            ),
        )
        if self.graph_residual_aux_target_mode == "uncovered":
            if (self.graph_residual_aux_distribution != "uniform"
                    or self.graph_residual_aux_target_power != 1.0
                    or self.graph_residual_aux_tie_break_coef != 0.0):
                raise ValueError("Paper UAA requires uniform distribution, power=1 and tie_break=0")
        elif self.graph_residual_aux_distribution == "uniform":
            raise ValueError("Uniform UAA requires target=uncovered")
        self.last_graph_residual_aux_target_debug: Dict[str, object] = {}

        self.observer = Observer(
            agent_name=self.log_prefix,
            log_dir="observation_logs",
            marginal_gain_sample_cap=50,
        )
        self.policy_diagnostics = SubWebPolicyDiagnostics(self.log_prefix)
        self.action_coverage_features = SubWebActionCoverageFeatures(self.coverage_tracker)

        gamma_mode = "main" if abs(self.gamma - 1.0) < 1e-9 else "discounted_variant"
        print(
            f"[{self.log_prefix}][config] "
            f"gamma={self.gamma:.4f} gamma_mode={gamma_mode} "
            f"max_actions={self.max_actions} rollout_len={self.rollout_len} "
            f"update_epochs={self.update_epochs} "
            f"policy_family={self.policy_family} "
            f"policy_input_mode={self.policy_input_mode} "
            f"advantage_estimator={self.advantage_estimator} "
            f"policy_baseline={self.policy_baseline_mode} "
            f"gae_lambda={self.gae_lambda:.3f} "
            f"policy_temperature={self.policy_temperature:.3f} "
            f"value_coef={self.value_coef:.3f} "
            f"entropy_coef={self.entropy_coef:.3f} "
            f"normalize_advantages={1 if self.normalize_advantages else 0} "
            f"adv_norm_min_std={self.advantage_norm_min_std:.4f} "
            f"min_actor_update_steps={self.min_actor_update_steps} "
            f"coverage_summary_dim={self.coverage_summary_dim} "
            f"full_coverage_summary_dim={self.full_coverage_summary_dim} "
            f"recent_action_summary_dim={recent_action_summary_dim()} "
            f"critic_input_dim={self.critic_input_dim} "
            f"critic_pooling={self.critic_pooling} "
            f"structural_features={1 if self.use_structural_action_features else 0} "
            f"structural_feature_dim={self.structural_feature_dim} "
            f"action_coverage_features={1 if self.use_action_coverage_features else 0} "
            f"action_coverage_features_requested={1 if requested_action_coverage_features else 0} "
            f"action_coverage_feature_dim={self.action_coverage_feature_dim} "
            f"input_dim={self.input_dim} max_grad_norm={self.max_grad_norm} "
            f"lr={self.learning_rate:.6f} actor_lr={self.actor_learning_rate:.6f} "
            f"critic_lr={self.critic_learning_rate:.6f} "
            f"separate_grad_clip={1 if self.separate_grad_clip else 0} "
            f"reward_clip_abs={self.reward_clip_abs if self.reward_clip_abs is not None else 'disabled'} "
            f"reward_mode={self.reward_mode} "
            f"reward_proxy={1 if self.reward_mode_is_proxy else 0} "
            f"optimizer_mode={self.optimizer_mode} "
            f"qlearning_epsilon={self.qlearning_epsilon:.3f} "
            f"qlearning_epsilon_final={self.qlearning_epsilon_final:.3f} "
            f"qlearning_epsilon_decay_steps={self.qlearning_epsilon_decay_steps} "
            f"qlearning_target_sync_interval={self.qlearning_target_sync_interval} "
            f"qlearning_lr={self.qlearning_learning_rate:.6f} "
            f"count_decay_scale={self.count_decay_scale:.3f} "
            f"count_decay_power={self.count_decay_power:.3f} "
            f"webqt_weights=({self.webqt_w_loc:.3f},{self.webqt_w_attention:.3f},{self.webqt_w_freq:.3f},{self.webqt_w_explore:.3f}) "
            f"webrled_weights=({self.webrled_w_global:.3f},{self.webrled_w_episode:.3f},{self.webrled_w_transition:.3f}) "
            f"qexplore_source=source_code_inverse_global_action_count "
            f"qexplore_valid_reward_scale={self.qexplore_valid_reward_scale:.3f} "
            f"qexplore_invalid_reward={self.qexplore_invalid_reward:.3f} "
            f"qexplore_initial_q={self.qexplore_initial_q:.3f} "
            "coverage_units=graph_nodes+alpha_graph_edges "
            f"graph_node_mode={graph_node_mode_from_env()} "
            f"graph_node_weight={self.coverage_tracker.graph_node_weight:.3f} "
            f"graph_edge_alpha={self.coverage_tracker.graph_edge_alpha:.3f} "
            f"graph_edge_label_mode={graph_edge_label_mode_from_env()} "
            f"graph_edge_weight_mode={self.coverage_tracker.graph_edge_weight_mode} "
            f"graph_filter_template_edges={1 if _read_bool_env('WEBTEST_GRAPH_FILTER_TEMPLATE_EDGES', False) else 0} "
            f"graph_residual_aux={1 if self.graph_residual_aux else 0} "
            f"graph_residual_aux_coef={self.graph_residual_aux_coef:.3f} "
            f"graph_residual_aux_target={self.graph_residual_aux_target_mode} "
            f"graph_residual_aux_target_power={self.graph_residual_aux_target_power:.3f} "
            f"graph_residual_aux_distribution={self.graph_residual_aux_distribution} "
            f"graph_residual_aux_tie_break_coef={self.graph_residual_aux_tie_break_coef:.3f} "
            f"candidate_q_aux={1 if self._candidate_q_enabled() else 0} "
            f"candidate_q_aux_coef={self.candidate_q_aux_coef:.3f} "
            f"candidate_q_aux_target={self.candidate_q_aux_target} "
            f"candidate_q_aux_lr={self.candidate_q_learning_rate:.6f} "
            f"policy_diag={self.policy_diagnostics.path or 'disabled'}"
        )

    def _make_optimizer(self) -> optim.Optimizer:
        if self._qlearning_enabled():
            param_groups = [{"params": list(self.actor.parameters()), "lr": self.qlearning_learning_rate}]
            if self.candidate_q_head is not None:
                param_groups.append(
                    {"params": list(self.candidate_q_head.parameters()), "lr": self.qlearning_learning_rate}
                )
            return optim.Adam(param_groups)
        param_groups = [
            {"params": list(self.actor.parameters()), "lr": self.actor_learning_rate},
            {"params": list(self.critic.parameters()), "lr": self.critic_learning_rate},
        ]
        if self._candidate_q_enabled():
            param_groups.append(
                {"params": list(self.candidate_q_head.parameters()), "lr": self.candidate_q_learning_rate}
            )
        return optim.Adam(param_groups)

    def _candidate_q_enabled(self) -> bool:
        return (
            (
                self._qlearning_enabled()
                or (
                    bool(getattr(self, "candidate_q_aux", False))
                    and float(getattr(self, "candidate_q_aux_coef", 0.0)) > 0.0
                )
            )
            and getattr(self, "candidate_q_head", None) is not None
        )

    def _qlearning_enabled(self) -> bool:
        return str(getattr(self, "optimizer_mode", "a2c")).lower() == "qlearning"

    def _policy_side_parameters(self) -> List[torch.nn.Parameter]:
        params = list(self.actor.parameters())
        if self._candidate_q_enabled():
            params.extend(list(self.candidate_q_head.parameters()))
        return params

    def _sync_qlearning_target(self) -> None:
        if getattr(self, "candidate_q_head", None) is None:
            return
        self.target_actor = copy.deepcopy(self.actor)
        self.target_candidate_q_head = copy.deepcopy(self.candidate_q_head)
        self.target_actor.eval()
        self.target_candidate_q_head.eval()
        for param in self.target_actor.parameters():
            param.requires_grad_(False)
        for param in self.target_candidate_q_head.parameters():
            param.requires_grad_(False)

    def _target_candidate_q_values(
        self,
        action_mat: torch.Tensor,
        action_mask: torch.Tensor,
        coverage_summary: torch.Tensor,
    ) -> torch.Tensor:
        actor = getattr(self, "target_actor", None) or self.actor
        head = getattr(self, "target_candidate_q_head", None) or self.candidate_q_head
        if head is None:
            raise RuntimeError("Q-learning target requested without candidate Q head")
        embeddings = actor.action_embeddings(action_mat, coverage_summary)
        return head(embeddings, action_mask)

    def _candidate_q_values(
        self,
        action_mat: torch.Tensor,
        action_mask: torch.Tensor,
        coverage_summary: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if not self._candidate_q_enabled():
            return None
        embeddings = self.actor.action_embeddings(action_mat, coverage_summary)
        return self.candidate_q_head(embeddings, action_mask, hidden)

    def _candidate_q_targets(
        self,
        returns_t: torch.Tensor,
        critic_targets_t: torch.Tensor,
        td_targets_t: torch.Tensor,
        policy_advantages_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target_mode = getattr(self, "candidate_q_aux_target", "critic_target")
        normalize = target_mode.endswith("_norm")
        base_mode = target_mode[:-5] if normalize else target_mode
        if base_mode == "advantage":
            base = (
                policy_advantages_t.detach()
                if policy_advantages_t is not None
                else (critic_targets_t - critic_targets_t.mean()).detach()
            )
        elif base_mode == "return":
            base = returns_t.detach()
        elif base_mode == "td":
            base = td_targets_t.detach()
        else:
            base = critic_targets_t.detach()
        if normalize and base.numel() > 1:
            std = base.std(unbiased=False)
            if float(std.detach().item()) > 1e-8:
                base = (base - base.mean()) / std
            else:
                base = base - base.mean()
        return base.detach()

    def _current_qlearning_epsilon(self) -> float:
        start = float(getattr(self, "qlearning_epsilon", 0.1))
        end = float(getattr(self, "qlearning_epsilon_final", start))
        decay_steps = max(1, int(getattr(self, "qlearning_epsilon_decay_steps", 1)))
        frac = min(1.0, max(0.0, float(getattr(self, "total_steps", 0)) / float(decay_steps)))
        return float(start + (end - start) * frac)

    def _qlearning_policy_probs(
        self,
        q_values: torch.Tensor,
        action_mask: torch.Tensor,
        greedy_idx: int,
        epsilon: float,
    ) -> torch.Tensor:
        probs = torch.zeros_like(action_mask, dtype=torch.float32)
        valid = action_mask.detach().bool()
        valid_count = int(valid.sum().item())
        if valid_count <= 0:
            return probs
        eps = max(0.0, min(1.0, float(epsilon)))
        probs[valid] = eps / float(valid_count)
        if 0 <= greedy_idx < int(probs.numel()) and bool(valid[greedy_idx].item()):
            probs[greedy_idx] += 1.0 - eps
        else:
            masked_q = q_values.detach().float().masked_fill(~valid, -1.0e9)
            probs[int(torch.argmax(masked_q).item())] += 1.0 - eps
        return probs

    def _choose_qlearning_action(
        self,
        action_mat: torch.Tensor,
        action_mask: torch.Tensor,
        coverage_summary: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, bool]:
        q_values = self._candidate_q_values(action_mat, action_mask, coverage_summary)
        if q_values is None:
            raise RuntimeError("Q-learning optimizer requires candidate Q values")
        valid = action_mask.detach().bool()
        valid_indices = torch.nonzero(valid, as_tuple=False).view(-1)
        if valid_indices.numel() == 0:
            raise ValueError("Q-learning received no valid actions")
        masked_q = q_values.detach().float().masked_fill(~valid, -1.0e9)
        greedy_idx = int(torch.argmax(masked_q).item())
        epsilon = self._current_qlearning_epsilon()
        explore = bool(torch.rand((), dtype=torch.float32).item() < epsilon)
        if explore:
            random_pos = int(torch.randint(0, int(valid_indices.numel()), ()).item())
            chosen_idx = valid_indices[random_pos].detach().cpu().long()
        else:
            chosen_idx = torch.tensor(greedy_idx, dtype=torch.long)
        probs = self._qlearning_policy_probs(q_values, action_mask, greedy_idx, epsilon)
        return chosen_idx, q_values, probs, epsilon, explore

    def _update_qlearning_if_ready(
        self,
        next_batch: Optional[PaddedActionBatch] = None,
        next_coverage_summary: Optional[torch.Tensor] = None,
        force: bool = False,
    ) -> bool:
        if not self.rollout_buffer:
            return False
        if not force and len(self.rollout_buffer) < 1:
            return False

        step = self.rollout_buffer[-1]
        dropped_extra = max(0, len(self.rollout_buffer) - 1)
        reward = step["reward"].detach().float().view(())
        bootstrap = torch.tensor(0.0, dtype=torch.float32)
        next_max_q = 0.0
        if next_batch is not None and next_coverage_summary is not None and torch.any(next_batch.action_mask):
            with torch.no_grad():
                next_q = self._target_candidate_q_values(
                    next_batch.action_mat,
                    next_batch.action_mask,
                    next_coverage_summary,
                )
                masked_next_q = next_q.detach().float().masked_fill(~next_batch.action_mask.bool(), -1.0e9)
                bootstrap = masked_next_q.max().clamp_min(-1.0e8).view(())
                next_max_q = float(bootstrap.detach().item())
        target = (reward + float(self.gamma) * bootstrap).detach()

        q_values = self._candidate_q_values(
            step["action_mat"],
            step["action_mask"],
            step["coverage_summary"],
        )
        if q_values is None:
            return False
        chosen_idx = int(step["chosen_idx"].long().view(()).item())
        chosen_q = q_values[chosen_idx].view(())
        loss = torch.nn.functional.smooth_l1_loss(chosen_q, target)

        self.optimizer.zero_grad()
        loss.backward()
        policy_side_params = self._policy_side_parameters()
        grad_norm_preclip = parameter_grad_norm(policy_side_params)
        clip_norm = float(torch.nn.utils.clip_grad_norm_(policy_side_params, max_norm=self.max_grad_norm))
        self.optimizer.step()
        self.total_updates += 1
        sync_interval = int(getattr(self, "qlearning_target_sync_interval", 25))
        target_synced = self.total_updates % sync_interval == 0
        if target_synced:
            self._sync_qlearning_target()

        mask = step["action_mask"].detach().bool()
        valid_q = q_values.detach().float()[mask]
        q_mean = float(valid_q.mean().item()) if valid_q.numel() else 0.0
        q_max = float(valid_q.max().item()) if valid_q.numel() else 0.0
        self.last_qlearning_update = {
            "reward": float(reward.item()),
            "target": float(target.item()),
            "bootstrap_next_max_q": next_max_q,
            "chosen_q": float(chosen_q.detach().item()),
            "loss": float(loss.detach().item()),
            "q_mean": q_mean,
            "q_max": q_max,
            "target_synced": target_synced,
        }
        print(
            f"[{self.log_prefix}][update] "
            f"idx={self.total_updates} optimizer=qlearning steps=1 "
            f"gamma={self.gamma:.4f} reward={float(reward.item()):.4f} "
            f"target={float(target.item()):.4f} bootstrap_next_max_q={next_max_q:.4f} "
            f"chosen_q={float(chosen_q.detach().item()):.4f} "
            f"q_mean={q_mean:.4f} q_max={q_max:.4f} "
            f"loss={float(loss.detach().item()):.4f} "
            f"epsilon={self._current_qlearning_epsilon():.4f} "
            f"target_sync_interval={sync_interval} "
            f"target_synced={1 if target_synced else 0} "
            f"grad_norm={grad_norm_preclip:.4f} clip_norm={clip_norm:.4f} "
            f"dropped_extra_rollout={dropped_extra} "
            f"F={self._coverage_value():.3f} "
            f"{self._coverage_counts_text()}"
        )
        self.rollout_buffer.clear()
        return True

    def _bucket_unit_class(self, cls: str) -> str:
        if cls == "context":
            return "context"
        if cls.startswith("field:"):
            return "field"
        if cls.startswith("submit:"):
            return "submit"
        if cls.startswith("main-action:"):
            return "main_action"
        if cls.startswith("workflow"):
            return "workflow_link"
        if cls.startswith("row-action:"):
            return "row_action"
        if cls in {"structure", "dialog", "tab"}:
            return "structure"
        return "other"

    def _coverage_summary(self) -> torch.Tensor:
        budget_remaining = 1.0
        max_transitions = _read_int_env("WEBTEST_MAX_TRANSITIONS", 0)
        if max_transitions > 0:
            budget_remaining = 1.0 - (self.total_steps / float(max_transitions))
        step_norm = min(self.total_steps, 5000) / 5000.0

        if self.policy_input_mode == "state_only":
            return torch.zeros((0,), dtype=torch.float32)
        if self.policy_input_mode == "pure":
            return torch.tensor(
                [
                    float(max(0.0, min(1.0, budget_remaining))),
                    step_norm,
                ],
                dtype=torch.float32,
            )

        covered_units = list(self.coverage_tracker.covered_units)
        total_covered = len(covered_units)
        denom = float(max(1, total_covered))
        type_counts = {bucket: 0 for bucket in _COVERAGE_TYPE_BUCKETS}
        sketch = [0 for _ in range(_COVERAGE_SKETCH_SIZE)]
        for unit in covered_units:
            bucket = self._bucket_unit_class(unit_class(unit))
            type_counts[bucket] = type_counts.get(bucket, 0) + 1
            digest = hashlib.blake2b(str(unit).encode("utf-8"), digest_size=2).digest()
            sketch_idx = int.from_bytes(digest, "big") % _COVERAGE_SKETCH_SIZE
            sketch[sketch_idx] += 1

        rewards = list(self.recent_rewards)
        recent_mean = sum(rewards) / len(rewards) if rewards else 0.0
        last_reward_norm = min(max(self.last_reward, 0.0), 50.0) / 50.0
        recent_mean_norm = min(max(recent_mean, 0.0), 50.0) / 50.0
        recent_max_norm = min(max(max(rewards) if rewards else 0.0, 0.0), 50.0) / 50.0
        rollout_fill = len(self.rollout_buffer) / float(self.rollout_len)
        summary_values = [
            min(self._coverage_value(), 5000.0) / 5000.0,
            *[type_counts[bucket] / denom for bucket in _COVERAGE_TYPE_BUCKETS],
            last_reward_norm,
            recent_mean_norm,
            recent_max_norm,
            min(self.zero_gain_streak, 100) / 100.0,
            float(max(0.0, min(1.0, budget_remaining))),
            step_norm,
            rollout_fill,
            *[count / denom for count in sketch],
            *recent_action_summary(self.recent_action_history).tolist(),
        ]
        return torch.tensor(summary_values, dtype=torch.float32)

    @property
    def critic_summary_dim(self) -> int:
        return self.coverage_summary_dim

    @property
    def critic_input_dim(self) -> int:
        action_pool_dim = 2 * self.input_dim if self.critic_pooling in {"mean_max", "mean+max"} else self.input_dim
        return action_pool_dim + self.critic_summary_dim

    def _masked_max(self, action_mat: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        if not torch.any(action_mask):
            return torch.zeros((self.input_dim,), dtype=torch.float32)
        return action_mat[action_mask.bool()].float().max(dim=0).values

    def _critic_input(self, batch: PaddedActionBatch, coverage_summary: torch.Tensor) -> torch.Tensor:
        action_mean = masked_mean(batch.action_mat, batch.action_mask)
        if self.critic_pooling == "mean":
            return torch.cat([action_mean, coverage_summary]).float()
        action_max = self._masked_max(batch.action_mat, batch.action_mask)
        return torch.cat([action_mean, action_max, coverage_summary]).float()

    def _coverage_value(self) -> float:
        return float(getattr(self.coverage_tracker, "cumulative_F", len(self.coverage_tracker.covered_units)))

    def _coverage_counts_text(self) -> str:
        return (
            f"node_count={len(getattr(self.coverage_tracker, 'covered_nodes', set()))} "
            f"edge_count={len(getattr(self.coverage_tracker, 'covered_edges', set()))} "
            f"graph_state_count={len(getattr(self, 'graph_state_list', []))} "
            f"node_mode={graph_node_mode_from_env()} "
            f"node_weight={getattr(self.coverage_tracker, 'graph_node_weight', 1.0):.3f} "
            f"edge_alpha={getattr(self.coverage_tracker, 'graph_edge_alpha', 1.0):.3f} "
            f"edge_label_mode={graph_edge_label_mode_from_env()} "
            f"edge_weight_mode={getattr(self.coverage_tracker, 'graph_edge_weight_mode', 'home_zero')}"
        )

    def _ensure_graph_state_identity(self, web_state: WebState) -> None:
        """Annotate graph nodes with WebTest's state-list equivalence boundary.

        DataCollector reports unique states from ``Webtest.state_dict`` using
        Python state equality. The graph coverage objective should use the same
        node boundary; otherwise the agent can optimize internal hash states
        that do not correspond to the metric used for baseline comparison.
        """
        if not isinstance(web_state, ActionSetWithExecutionTimesState):
            return
        try:
            index = self.graph_state_list.index(web_state)
            canonical_state = self.graph_state_list[index]
        except ValueError:
            self.graph_state_list.append(web_state)
            index = len(self.graph_state_list) - 1
            canonical_state = web_state
        setattr(web_state, "graph_state_index", index)
        try:
            setattr(canonical_state, "graph_state_index", index)
        except Exception:
            pass

    def _graph_residual_scores(self, web_state: WebState, batch: PaddedActionBatch) -> torch.Tensor:
        scores = torch.zeros_like(batch.action_mask, dtype=torch.float32)
        source_url = str(getattr(web_state, "raw_url", getattr(web_state, "url", "")) or "")
        source_node_id = extract_graph_node_id(web_state)
        covered_edges = set(getattr(self.coverage_tracker, "covered_edges", set()))
        covered_nodes = set(getattr(self.coverage_tracker, "covered_nodes", set()))
        edge_alpha = float(getattr(self.coverage_tracker, "graph_edge_alpha", 1.0))
        node_weight = float(getattr(self.coverage_tracker, "graph_node_weight", 1.0))
        for idx, action in enumerate(getattr(batch, "actions_policy", []) or []):
            if idx >= int(scores.numel()) or not bool(batch.action_mask[idx].detach().cpu().item()):
                continue
            prefix = graph_edge_prefix(source_url, action, source_node_id=source_node_id)
            edge_gain = 0.0
            if prefix and not any(str(edge).startswith(prefix) for edge in covered_edges):
                edge_gain = edge_alpha * float(self.coverage_tracker._graph_edge_weight(prefix))
            if edge_gain <= 0.0:
                continue
            target_route = ""
            try:
                target_route = ActionSetWithExecutionTimesState._functional_action_route_hint(action)
            except Exception:
                target_route = ""
            target_node_gain = 0.0
            if target_route:
                target_exact = f"node:{target_route}"
                target_prefix = f"node:{target_route}|"
                target_seen = any(
                    str(node) == target_exact or str(node).startswith(target_prefix)
                    for node in covered_nodes
                )
                if not target_seen:
                    target_node_gain = node_weight
            scores[idx] = float(edge_gain + target_node_gain)
        return scores * batch.action_mask.detach().float().cpu()

    @staticmethod
    def _parse_functional_family(signature: str) -> str:
        for part in str(signature or "").split("|"):
            if part.startswith("family="):
                return part.split("=", 1)[1] or "other"
        return "other"

    def _uncovered_action_aux_target(
        self,
        batch: PaddedActionBatch,
        web_state: Optional[WebState],
    ) -> torch.Tensor:
        """Paper UAA: uniform over unresolved valid source-action prefixes.

        Unlike the legacy residual target, neither a target-state gain estimate
        nor an edge reward weight changes these binary labels.
        """
        labels = torch.zeros_like(batch.action_mask, dtype=torch.float32, device="cpu")
        mask = batch.action_mask.detach().bool().cpu()
        if isinstance(web_state, ActionSetWithExecutionTimesState):
            source_url = str(getattr(web_state, "raw_url", getattr(web_state, "url", "")) or "")
            source_node = extract_graph_node_id(web_state)
            covered = self.coverage_tracker.covered_edges
            for idx, action in enumerate(batch.actions_policy):
                if idx >= labels.numel() or not bool(mask[idx]):
                    continue
                prefix = graph_edge_prefix(source_url, action, source_node_id=source_node)
                if prefix and not any(edge.startswith(prefix) for edge in covered):
                    labels[idx] = 1.0
        count = float(labels.sum().item())
        target = labels / count if count else labels
        self.last_graph_residual_aux_target_debug = {
            "target_source": "uncovered" if count else "zero",
            "target_distribution": "uniform",
            "uncovered_count": int(count),
            "graph_score_sum": count,
            "coverage_score_sum": 0.0,
            "target_power": 1.0,
            "tie_break_coef": 0.0,
            "pre_power_score_sum": count,
            "post_power_score_sum": count,
        }
        return target

    def _graph_residual_aux_target(
        self,
        batch: PaddedActionBatch,
        web_state: Optional[WebState] = None,
    ) -> torch.Tensor:
        if getattr(self, "graph_residual_aux_target_mode", "graph") == "uncovered":
            return self._uncovered_action_aux_target(batch, web_state)
        scores = self._graph_residual_scores(web_state, batch) if web_state is not None else torch.zeros_like(
            batch.action_mask,
            dtype=torch.float32,
        )
        coverage_scores = self._coverage_feature_marginal_scores(batch)
        graph_sum = float(scores.sum().item())
        coverage_sum = float(coverage_scores.sum().item())
        target_source = "zero"
        if graph_sum > 1e-8 and coverage_sum > 1e-8:
            # The hard one-step graph marginal is the anchor. Coverage-feature
            # scores are only a small residual tie-break among actions that
            # already have positive immediate graph marginal; they must not
            # override a strictly larger F_graph node/edge gain.
            positive_graph = scores > 1e-8
            coverage_tiebreak = torch.zeros_like(scores)
            if bool(positive_graph.any().item()):
                coverage_on_graph = coverage_scores * positive_graph.float()
                coverage_max = float(coverage_on_graph.max().item())
                if coverage_max > 1e-8:
                    min_positive_graph = float(scores[positive_graph].min().item())
                    tie_scale = float(getattr(self, "graph_residual_aux_tie_break_coef", 0.0)) * max(
                        min_positive_graph,
                        1e-8,
                    )
                    coverage_tiebreak = tie_scale * (coverage_on_graph / coverage_max)
            scores = scores + coverage_tiebreak
            target_source = "graph+coverage"
        elif graph_sum > 1e-8:
            target_source = "graph"
        elif (
            getattr(self, "graph_residual_aux_target_mode", "graph")
            in {"graph_bridge", "graph_with_bridge", "bridge_graph"}
            and coverage_sum > 1e-8
        ):
            # When the current source node has no remaining one-step graph
            # marginal, use only objective-derived bridge features: actions
            # that move to already-known contexts with residual graph frontier.
            # This is an auxiliary target, not reward shaping; hard rewards and
            # evaluation still come solely from discrete Delta F_graph.
            scores = coverage_scores
            target_source = "graph_bridge"
        else:
            # Coverage features may sharpen choices only inside the same
            # positive F_graph support. They must not create a standalone
            # teacher target after the graph marginal is already exhausted.
            target_source = "zero"
        scores = scores * batch.action_mask.detach().float().cpu()
        distribution = str(
            getattr(self, "graph_residual_aux_distribution", "max_only") or "max_only"
        ).strip().lower().replace("-", "_")
        if distribution in {"max_only", "greedy"} and float(scores.sum().item()) > 1e-8:
            # Greedy policy-improvement target for the same submodular graph
            # objective: imitate the currently maximal estimated marginal-F set,
            # not every merely-positive residual action.
            valid_scores = scores[batch.action_mask.detach().bool().cpu()]
            if valid_scores.numel():
                max_score = float(valid_scores.max().item())
                if max_score > 1e-8:
                    max_mask = torch.isclose(
                        scores,
                        torch.tensor(max_score, dtype=scores.dtype),
                        rtol=1e-5,
                        atol=1e-8,
                    )
                    scores = torch.where(max_mask, scores, torch.zeros_like(scores))
        pre_power_scores = scores.detach().clone()
        power = max(0.05, float(getattr(self, "graph_residual_aux_target_power", 1.0)))
        if abs(power - 1.0) > 1e-9:
            # Training-only target sharpness. This preserves the same residual
            # F_graph support and mask while making weak marginal preferences
            # easier for the actor to imitate.
            scores = scores.clamp_min(0.0).pow(power)
        total = float(scores.sum().item())
        if total <= 1e-8:
            self.last_graph_residual_aux_target_debug = {
                "target_source": target_source,
                "graph_score_sum": round(graph_sum, 6),
                "coverage_score_sum": round(coverage_sum, 6),
                "target_power": round(power, 6),
                "target_distribution": distribution,
                "tie_break_coef": round(float(getattr(self, "graph_residual_aux_tie_break_coef", 0.0)), 6),
                "pre_power_score_sum": round(float(pre_power_scores.sum().item()), 6),
                "post_power_score_sum": 0.0,
            }
            return torch.zeros_like(scores, dtype=torch.float32)
        target = (scores / total).float()
        pre_top_idx = int(torch.argmax(pre_power_scores).item()) if pre_power_scores.numel() else -1
        post_top_idx = int(torch.argmax(target).item()) if target.numel() else -1
        self.last_graph_residual_aux_target_debug = {
            "target_source": target_source,
            "graph_score_sum": round(graph_sum, 6),
            "coverage_score_sum": round(coverage_sum, 6),
            "target_power": round(power, 6),
            "target_distribution": distribution,
            "tie_break_coef": round(float(getattr(self, "graph_residual_aux_tie_break_coef", 0.0)), 6),
            "pre_power_score_sum": round(float(pre_power_scores.sum().item()), 6),
            "post_power_score_sum": round(total, 6),
            "pre_power_top_idx": pre_top_idx,
            "pre_power_top_score": (
                round(float(pre_power_scores[pre_top_idx].item()), 6)
                if 0 <= pre_top_idx < int(pre_power_scores.numel())
                else 0.0
            ),
            "post_power_top_idx": post_top_idx,
            "post_power_top_prob": (
                round(float(target[post_top_idx].item()), 6)
                if 0 <= post_top_idx < int(target.numel())
                else 0.0
            ),
        }
        return target

    def _coverage_feature_marginal_scores(self, batch: PaddedActionBatch) -> torch.Tensor:
        scores = torch.zeros_like(batch.action_mask, dtype=torch.float32)
        if getattr(self, "graph_residual_aux_target_mode", "graph") not in {
            "coverage",
            "coverage_features",
            "graph_bridge",
            "graph_with_bridge",
            "bridge_graph",
        }:
            return scores
        if not getattr(self, "use_action_coverage_features", False):
            return scores
        feature_dim = int(getattr(self, "action_coverage_feature_dim", 0) or 0)
        if feature_dim <= 0 or batch.action_mat.shape[1] < feature_dim:
            return scores
        start = batch.action_mat.shape[1] - feature_dim
        features = batch.action_mat[:, start:].detach().float().cpu()
        try:
            unseen_idx = ACTION_COVERAGE_FEATURE_NAMES.index("edge_unseen_from_current_node")
            action_count_idx = ACTION_COVERAGE_FEATURE_NAMES.index("action_count_log")
            zero_count_idx = ACTION_COVERAGE_FEATURE_NAMES.index("zero_gain_count_log")
            target_ratio_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_known_frontier_ratio")
            target_count_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_known_frontier_count_log")
            target_seen_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_route_seen")
        except ValueError:
            return scores

        unseen = features[:, unseen_idx].clamp(0.0, 1.0)
        action_count = features[:, action_count_idx].clamp_min(0.0)
        zero_count = features[:, zero_count_idx].clamp_min(0.0)
        target_ratio = features[:, target_ratio_idx].clamp_min(0.0)
        target_count = features[:, target_count_idx].clamp_min(0.0)
        target_seen = features[:, target_seen_idx].clamp(0.0, 1.0)
        tau = 0.5
        count_decay = torch.exp(-(action_count + zero_count) / tau)
        downstream_frontier = target_ratio + target_count
        # Expected known marginal of the action-attempt edge plus a bridge term:
        # even when the current edge was already covered, an action can be useful
        # if it moves the run to a known context with residual graph frontier.
        # This is an optimizer target derived from F_graph, not a reward bonus.
        immediate_scores = unseen * count_decay * (1.0 + downstream_frontier)
        bridge_scores = target_seen * downstream_frontier
        scores = immediate_scores + bridge_scores
        return scores * batch.action_mask.detach().float().cpu()

    def _prepare_batch(self, web_state: WebState, html: str) -> PaddedActionBatch:
        source_node_id = extract_graph_node_id(web_state)
        batch = build_padded_action_batch(
            web_state=web_state,
            html=html,
            transformer=self.transformer,
            max_actions=self.max_actions,
            input_dim=self.input_dim,
            include_structural_features=self.use_structural_action_features,
            coverage_feature_provider=(
                self.action_coverage_features if self.use_action_coverage_features else None
            ),
            source_node_id=source_node_id,
        )
        register_degree = getattr(self.coverage_tracker, "register_graph_source_degree", None)
        if callable(register_degree):
            register_degree(source_node_id, int(getattr(batch, "original_action_count", 0) or 0))
        return batch

    def _coverage_action_diagnostics(
        self,
        web_state: WebState,
        batch: PaddedActionBatch,
        probs: torch.Tensor,
        chosen_idx: int,
        source_node_id: str,
        graph_residual_aux_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        """Observe how much policy mass is on currently unexecuted graph edges."""
        if not self.use_action_coverage_features or not batch.actions_policy:
            return {}

        try:
            unseen_idx = ACTION_COVERAGE_FEATURE_NAMES.index("edge_unseen_from_current_node")
            seen_idx = ACTION_COVERAGE_FEATURE_NAMES.index("edge_seen_from_current_node")
            action_count_idx = ACTION_COVERAGE_FEATURE_NAMES.index("action_count_log")
            zero_count_idx = ACTION_COVERAGE_FEATURE_NAMES.index("zero_gain_count_log")
            target_frontier_ratio_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_known_frontier_ratio")
            target_frontier_count_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_known_frontier_count_log")
        except ValueError:
            return {}

        probs_cpu = probs.detach().float().cpu()
        unseen_rows = []
        target_frontier_rows = []
        chosen_features = None
        chosen_target_frontier_context = ""
        source_url = str(getattr(web_state, "raw_url", getattr(web_state, "url", "")) or "")
        source_context = ActionSetWithExecutionTimesState._normalize_route_identity(source_url)
        coverage_rows = getattr(batch, "coverage_features_policy", None) or []
        for idx, action in enumerate(batch.actions_policy):
            if idx >= len(probs_cpu) or not bool(batch.action_mask[idx].item()):
                continue
            if idx < len(coverage_rows):
                features = coverage_rows[idx].detach().float()
            else:
                features = self.action_coverage_features.features_for(
                    action,
                    web_state,
                    batch.actions_full,
                    source_node_id=source_node_id,
                )
            explicit_target_context = ""
            try:
                explicit_target_context = ActionSetWithExecutionTimesState._normalize_route_identity(
                    ActionSetWithExecutionTimesState._functional_action_route_hint(action)
                )
            except Exception:
                explicit_target_context = ""
            recorded_target_context = ""
            try:
                coverage_key = self.action_coverage_features.key_for(
                    source_node_id,
                    source_url,
                    action,
                )
                recorded_target_context = str(
                    self.action_coverage_features.action_target_contexts.get(coverage_key, "")
                    or ""
                )
            except Exception:
                recorded_target_context = ""
            target_frontier_context = ""
            if explicit_target_context and explicit_target_context != source_context:
                target_frontier_context = explicit_target_context
            elif recorded_target_context and recorded_target_context != source_context:
                target_frontier_context = recorded_target_context
            is_unseen = bool(float(features[unseen_idx].item()) > 0.5)
            row = {
                "idx": idx,
                "prob": float(probs_cpu[idx].item()),
                "unseen": is_unseen,
                "seen": bool(float(features[seen_idx].item()) > 0.5),
                "target_frontier_ratio": float(features[target_frontier_ratio_idx].item()),
                "target_frontier_count_log": float(features[target_frontier_count_idx].item()),
                "target_context": target_frontier_context,
                "kind": action_kind(action),
                "signature": action_signature(action),
            }
            if is_unseen:
                unseen_rows.append(row)
            if row["target_frontier_ratio"] > 0.0 or row["target_frontier_count_log"] > 0.0:
                target_frontier_rows.append(row)
            if idx == int(chosen_idx):
                chosen_features = features
                chosen_target_frontier_context = target_frontier_context

        valid_count = int(batch.action_mask.sum().item())
        unseen_mass = sum(row["prob"] for row in unseen_rows)
        target_frontier_mass = sum(row["prob"] for row in target_frontier_rows)
        unseen_rows_by_prob = sorted(unseen_rows, key=lambda row: row["prob"], reverse=True)
        target_frontier_rows_by_prob = sorted(
            target_frontier_rows,
            key=lambda row: row["prob"],
            reverse=True,
        )
        chosen_unseen_rank = 0
        for rank, row in enumerate(unseen_rows_by_prob, start=1):
            if row["idx"] == int(chosen_idx):
                chosen_unseen_rank = rank
                break
        chosen_target_frontier_rank = 0
        for rank, row in enumerate(target_frontier_rows_by_prob, start=1):
            if row["idx"] == int(chosen_idx):
                chosen_target_frontier_rank = rank
                break
        target_frontier_route_counts: Dict[str, int] = {}
        for row in target_frontier_rows:
            target_context = str(row.get("target_context", "") or "unknown")
            target_frontier_route_counts[target_context] = target_frontier_route_counts.get(target_context, 0) + 1
        top_target_frontier_actions = [
            {
                "idx": int(row["idx"]),
                "prob": round(float(row["prob"]), 6),
                "kind": str(row.get("kind", "")),
                "target_context": str(row.get("target_context", "")),
                "target_frontier_ratio": round(float(row["target_frontier_ratio"]), 6),
                "target_frontier_count_log": round(float(row["target_frontier_count_log"]), 6),
                "signature": str(row.get("signature", ""))[:160],
            }
            for row in target_frontier_rows_by_prob[:5]
        ]

        chosen_edge_unseen = False
        chosen_edge_seen = False
        chosen_action_count_log = 0.0
        chosen_zero_gain_count_log = 0.0
        chosen_target_frontier_ratio = 0.0
        chosen_target_frontier_count_log = 0.0
        if chosen_features is not None:
            chosen_edge_unseen = bool(float(chosen_features[unseen_idx].item()) > 0.5)
            chosen_edge_seen = bool(float(chosen_features[seen_idx].item()) > 0.5)
            chosen_action_count_log = float(chosen_features[action_count_idx].item())
            chosen_zero_gain_count_log = float(chosen_features[zero_count_idx].item())
            chosen_target_frontier_ratio = float(chosen_features[target_frontier_ratio_idx].item())
            chosen_target_frontier_count_log = float(chosen_features[target_frontier_count_idx].item())

        coverage_diag = {
            "unseen_action_count": len(unseen_rows),
            "seen_action_count": max(0, valid_count - len(unseen_rows)),
            "unseen_action_ratio": len(unseen_rows) / float(max(1, valid_count)),
            "unseen_action_mass": round(float(unseen_mass), 6),
            "top_unseen_prob": round(float(unseen_rows_by_prob[0]["prob"]), 6) if unseen_rows_by_prob else 0.0,
            "chosen_edge_unseen_before": bool(chosen_edge_unseen),
            "chosen_edge_seen_before": bool(chosen_edge_seen),
            "chosen_unseen_rank_by_prob": int(chosen_unseen_rank),
            "chosen_action_count_log": round(float(chosen_action_count_log), 6),
            "chosen_zero_gain_count_log": round(float(chosen_zero_gain_count_log), 6),
            "target_frontier_action_count": len(target_frontier_rows),
            "target_frontier_action_ratio": len(target_frontier_rows) / float(max(1, valid_count)),
            "target_frontier_action_mass": round(float(target_frontier_mass), 6),
            "top_target_frontier_prob": (
                round(float(target_frontier_rows_by_prob[0]["prob"]), 6)
                if target_frontier_rows_by_prob
                else 0.0
            ),
            "top_target_frontier_actions": top_target_frontier_actions,
            "target_frontier_route_counts": target_frontier_route_counts,
            "chosen_target_frontier_ratio": round(float(chosen_target_frontier_ratio), 6),
            "chosen_target_frontier_count_log": round(float(chosen_target_frontier_count_log), 6),
            "chosen_target_frontier_rank_by_prob": int(chosen_target_frontier_rank),
            "chosen_target_frontier_context": chosen_target_frontier_context,
        }
        diagnostics = {"coverage_action_diag": coverage_diag}

        if graph_residual_aux_target is not None:
            target = graph_residual_aux_target.detach().float().cpu()
            mask = batch.action_mask.detach().bool().cpu()
            if target.numel() == mask.numel() and bool(mask.any().item()):
                target = target * mask.float()
                target_sum = float(target.sum().item())
                if target_sum > 1e-8:
                    target = target / target_sum
                    probs_masked = probs_cpu * mask.float()
                    target_support = target > 1e-8
                    top_idx = int(torch.argmax(target).item())
                    chosen_target_prob = (
                        float(target[int(chosen_idx)].item())
                        if 0 <= int(chosen_idx) < int(target.numel())
                        else 0.0
                    )
                    chosen_target_rank = 0
                    ranked_target_indices = torch.argsort(target, descending=True)
                    for rank, idx_tensor in enumerate(ranked_target_indices.tolist(), start=1):
                        if float(target[idx_tensor].item()) <= 1e-8:
                            break
                        if int(idx_tensor) == int(chosen_idx):
                            chosen_target_rank = rank
                            break
                    kl = float(
                        (
                            target[target_support]
                            * (
                                torch.log(target[target_support].clamp_min(1e-8))
                                - torch.log(probs_masked[target_support].clamp_min(1e-8))
                            )
                        ).sum().item()
                    )
                    top_action = batch.actions_policy[top_idx] if top_idx < len(batch.actions_policy) else None
                    diagnostics["graph_residual_aux_target_diag"] = {
                        "target_mass": round(target_sum, 6),
                        "target_support_count": int(target_support.sum().item()),
                        "target_entropy": round(float(-(target[target_support] * torch.log(target[target_support].clamp_min(1e-8))).sum().item()), 6),
                        "policy_target_kl": round(kl, 6),
                        "policy_mass_on_target_support": round(float(probs_masked[target_support].sum().item()), 6),
                        "chosen_target_prob": round(chosen_target_prob, 6),
                        "chosen_target_rank": int(chosen_target_rank),
                        "target_top_idx": int(top_idx),
                        "target_top_prob": round(float(target[top_idx].item()), 6),
                        "target_top_policy_prob": round(float(probs_masked[top_idx].item()), 6),
                        "target_top_action_type": action_kind(top_action) if top_action is not None else "",
                        "target_top_action_signature": action_signature(top_action) if top_action is not None else "",
                    }
                    target_debug = getattr(self, "last_graph_residual_aux_target_debug", {})
                    if target_debug:
                        diagnostics["graph_residual_aux_target_source_diag"] = dict(target_debug)
                else:
                    diagnostics["graph_residual_aux_target_diag"] = {
                        "target_mass": 0.0,
                        "target_support_count": 0,
                    }
        return diagnostics

    def _graph_residual_aux_console_brief(self, diagnostics: Dict[str, object]) -> str:
        residual_diag = diagnostics.get("graph_residual_aux_target_diag", {}) if diagnostics else {}
        if not residual_diag:
            return "graph_residual_aux_target=disabled"
        return (
            f"graph_residual_aux_support={int(residual_diag.get('target_support_count', 0) or 0)} "
            f"graph_residual_aux_top_prob={float(residual_diag.get('target_top_prob', 0.0) or 0.0):.3f} "
            f"graph_residual_aux_top_policy={float(residual_diag.get('target_top_policy_prob', 0.0) or 0.0):.3f} "
            f"chosen_residual_aux_prob={float(residual_diag.get('chosen_target_prob', 0.0) or 0.0):.3f} "
            f"chosen_residual_aux_rank={int(residual_diag.get('chosen_target_rank', 0) or 0)} "
            f"policy_residual_aux_kl={float(residual_diag.get('policy_target_kl', 0.0) or 0.0):.3f} "
            f"graph_residual_aux_top_type={residual_diag.get('target_top_action_type', '')} "
            f"graph_residual_aux_source={diagnostics.get('graph_residual_aux_target_source_diag', {}).get('target_source', '')}"
        )

    def _seed_initial_frontier(self, web_state: WebState) -> None:
        if self.initial_frontier_seeded:
            return
        seed_info = self.coverage_tracker.seed_initial_state(web_state)
        self.initial_frontier_seeded = True
        print(
            f"[{self.log_prefix}][seed] "
            f"initial_F_seeded={seed_info.cumulative_F:.3f} "
            f"initial_units_count={seed_info.new_units} "
            f"{self._coverage_counts_text()}"
        )

    def _rnn_feedback_from_transition(self, transition: Optional[Dict]) -> Optional[torch.Tensor]:
        return None

    def _reward_transition_key(self, web_state: WebState) -> Tuple[str, str, str]:
        source_node = str((self.pending_step or {}).get("source_node_id") or "")
        action_sig = str((self.pending_step or {}).get("action_signature") or "")
        if not source_node:
            source_node = str((self.pending_step or {}).get("source_url") or "")
        try:
            target_node = extract_graph_node_id(web_state)
        except Exception:
            target_node = f"state:{type(web_state).__name__}"
        return source_node, action_sig, target_node

    def _compute_training_reward(self, web_state: WebState, reward_info) -> Tuple[float, Dict[str, object]]:
        """Return the scalar reward consumed by the online learner.

        The graph tracker has already updated the run's objective coverage.
        RQ3 reward variants intentionally reuse the same WebCover state/action
        identity and only alter this scalar training reward.
        """
        objective_reward = float(reward_info.reward)
        mode = getattr(self, "reward_mode", "marginal")
        if mode == "marginal":
            return objective_reward, {
                "mode": mode,
                "objective_reward": objective_reward,
            }

        key = self._reward_transition_key(web_state)
        source_node, action_sig, target_node = key
        chosen_action = (self.pending_step or {}).get("chosen_action")
        action_type = _reward_action_type(chosen_action)
        is_failure = _is_failure_state(web_state)

        transition_count = self.reward_transition_counts[key] + 1
        self.reward_transition_counts[key] = transition_count
        action_count_key = (source_node, action_sig)
        action_count = self.reward_action_counts[action_count_key] + 1
        self.reward_action_counts[action_count_key] = action_count
        action_type_count = self.reward_action_type_counts[action_type] + 1
        self.reward_action_type_counts[action_type] = action_type_count
        node_visit_count = self.reward_node_visit_counts[target_node] + 1
        self.reward_node_visit_counts[target_node] = node_visit_count

        transition_decay = self.count_decay_scale / (float(transition_count) ** self.count_decay_power)
        action_decay = 1.0 / math.sqrt(float(action_count))
        action_type_decay = 1.0 / math.sqrt(float(action_type_count))
        node_novelty = 1.0 if int(getattr(reward_info, "node_gain", 0) or 0) > 0 else 0.0
        transition_novelty = 1.0 if int(getattr(reward_info, "edge_gain", 0) or 0) > 0 else 0.0

        if mode == "count_decay":
            return float(0.0 if is_failure else transition_decay), {
                "mode": mode,
                "transition_count": transition_count,
                "transition_decay": transition_decay,
            }

        if mode == "webqt":
            reward = (
                self.webqt_w_loc * node_novelty
                + self.webqt_w_attention * action_type_decay
                + self.webqt_w_freq * action_decay
                + self.webqt_w_explore * transition_decay
            )
            if is_failure:
                reward = -1.0
            return float(reward), {
                "mode": mode,
                "r_loc": node_novelty,
                "r_attention": action_type_decay,
                "r_freq": action_decay,
                "r_explore": transition_decay,
                "transition_count": transition_count,
                "action_count": action_count,
                "action_type_count": action_type_count,
            }

        if mode == "webrled":
            if isinstance(chosen_action, RestartAction) or is_failure:
                self.reward_episode_nodes.clear()
            episode_novelty = 0.0 if target_node in self.reward_episode_nodes else 1.0
            repeat_penalty = 1.0 / math.sqrt(float(node_visit_count))
            reward = repeat_penalty * (
                self.webrled_w_global * node_novelty
                + self.webrled_w_episode * episode_novelty
                + self.webrled_w_transition * transition_decay
            )
            if is_failure:
                reward = -1.0
            else:
                self.reward_episode_nodes.add(target_node)
            return float(reward), {
                "mode": mode,
                "r_global": node_novelty,
                "r_episode": episode_novelty,
                "r_transition": transition_decay,
                "repeat_penalty": repeat_penalty,
                "transition_count": transition_count,
                "node_visit_count": node_visit_count,
            }

        if mode == "qexplore":
            qexplore_action_key = action_sig or action_type or "unknown"
            if _is_qexplore_invalid_state(web_state):
                reward = self.qexplore_invalid_reward
                qexplore_action_count = self.reward_qexplore_action_counts[qexplore_action_key]
            else:
                qexplore_action_count = self.reward_qexplore_action_counts[qexplore_action_key] + 1
                self.reward_qexplore_action_counts[qexplore_action_key] = qexplore_action_count
                reward = self.qexplore_valid_reward_scale / float(qexplore_action_count)
            return float(reward), {
                "mode": mode,
                "action_signature": qexplore_action_key,
                "action_count": qexplore_action_count,
                "valid_reward_scale": self.qexplore_valid_reward_scale,
                "invalid_reward": self.qexplore_invalid_reward,
                "initial_q_table_value": self.qexplore_initial_q,
                "note": "qexplore_source_code_reward_inverse_global_action_count",
            }

        raise ValueError(f"Unsupported reward_mode={mode!r}")

    def _finalize_pending(self, web_state: WebState) -> float:
        if self.pending_step is None:
            self.last_finalized_transition = None
            return 0.0

        reward_info = self.coverage_tracker.compute_strict_marginal_reward(
            web_state,
            chosen_action=self.pending_step.get("chosen_action"),
            source_url=self.pending_step.get("source_url"),
            source_node_id=self.pending_step.get("source_node_id"),
            source_action_count=self.pending_step.get("action_count_full"),
        )
        reward_mode = getattr(self, "reward_mode", "marginal")
        objective_reward = float(reward_info.reward)
        raw_reward, reward_components = self._compute_training_reward(web_state, reward_info)
        train_reward = clip_reward_value(raw_reward, self.reward_clip_abs)
        self.last_reward = objective_reward
        self.last_train_reward = train_reward
        self.recent_rewards.append(objective_reward)
        if objective_reward <= 0.0:
            self.zero_gain_streak += 1
        else:
            self.zero_gain_streak = 0

        clipped = abs(train_reward - raw_reward) > 1e-9
        coverage_feature_stats = None
        if self.pending_step.get("chosen_action") is not None:
            coverage_feature_stats = self.action_coverage_features.update(
                str(self.pending_step.get("source_node_id") or ""),
                str(self.pending_step.get("source_url") or ""),
                self.pending_step["chosen_action"],
                objective_reward,
                target_state=web_state,
            )
        self.last_finalized_transition = {
            "sample_step": self.pending_step.get("step"),
            "action_signature": self.pending_step.get("action_signature"),
            "action_type": self.pending_step.get("action_type"),
            "reward_mode": reward_mode,
            "objective_reward": objective_reward,
            "raw_reward": raw_reward,
            "train_reward": train_reward,
            "train_reward_unshaped": train_reward,
            "clipped": clipped,
            "F": reward_info.cumulative_F,
            "node_gain": reward_info.node_gain,
            "edge_gain": reward_info.edge_gain,
            "node_weight_gain": reward_info.node_weight_gain,
            "edge_weight_gain": reward_info.edge_weight_gain,
            "reward_components": reward_components,
            "action_coverage": coverage_feature_stats or {},
        }
        if not hasattr(self, "recent_action_history"):
            self.recent_action_history = deque(maxlen=8)
        append_recent_action(self.recent_action_history, self.last_finalized_transition)
        print(
            f"[{self.log_prefix}][transition] "
            f"sample_step={self.pending_step.get('step')} "
            f"prev_action_type={self.pending_step.get('action_type')} "
            f"prev_action_signature={self.pending_step.get('action_signature')} "
            f"reward_mode={reward_mode} "
            f"reward_objective={objective_reward:.3f} "
            f"reward_raw={raw_reward:.3f} reward_train={train_reward:.3f} "
            f"reward_clipped={1 if clipped else 0} F={reward_info.cumulative_F:.3f} "
            f"node_gain={reward_info.node_gain} edge_gain={reward_info.edge_gain} "
            f"node_weight_gain={reward_info.node_weight_gain:.3f} "
            f"edge_weight_gain={reward_info.edge_weight_gain:.3f} "
            f"{self._coverage_counts_text()} "
            f"action_coverage={coverage_feature_stats or {}} "
            f"zero_gain_streak={self.zero_gain_streak}"
        )

        rollout_step = {
            "action_mat": self.pending_step["action_mat"],
            "action_mask": self.pending_step["action_mask"],
            "coverage_summary": self.pending_step["coverage_summary"],
            "critic_input": self.pending_step["critic_input"],
            "policy_temperature": self.pending_step.get("policy_temperature", getattr(self, "policy_temperature", 1.0)),
            "graph_residual_aux_target": self.pending_step.get(
                "graph_residual_aux_target",
                torch.zeros_like(self.pending_step["action_mask"], dtype=torch.float32),
            ),
            "chosen_idx": self.pending_step["chosen_idx"],
            "reward": torch.tensor(train_reward, dtype=torch.float32),
        }
        rnn_feedback = self._rnn_feedback_from_transition(self.last_finalized_transition)
        if rnn_feedback is not None:
            rollout_step["rnn_feedback"] = rnn_feedback.detach().float()
        if "hidden_in" in self.pending_step:
            rollout_step["hidden_in"] = self.pending_step["hidden_in"]
        self.rollout_buffer.append(rollout_step)
        self.pending_step = None
        return train_reward

    def _drop_pending(self, reason: str) -> None:
        if self.pending_step is None:
            return
        action = self.pending_step.get("chosen_action")
        print(
            f"[{self.log_prefix}][pending] "
            f"dropped=1 reason={reason} action={action}"
        )
        self.pending_step = None

    def _one_step_td_targets(
        self,
        rewards_t: torch.Tensor,
        values: torch.Tensor,
        bootstrap_scalar: torch.Tensor,
    ) -> torch.Tensor:
        """One-step actor-critic targets for online coverage credit.

        This keeps the learning signal close to the strict marginal gain while
        still allowing the critic to learn repositioning value through V(s').
        """
        if values.numel() > 1:
            next_values = torch.cat([values[1:].detach(), bootstrap_scalar.detach().view(1)])
        else:
            next_values = bootstrap_scalar.detach().view(1)
        return rewards_t + self.gamma * next_values

    def _gae_targets(
        self,
        rewards_t: torch.Tensor,
        values: torch.Tensor,
        bootstrap_scalar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generalized advantage estimates for finite-budget coverage runs.

        GAE is a training-only credit-assignment estimator. The hard objective
        remains the strict graph marginal reward; lambda controls how much
        delayed coverage gain is propagated back to earlier bridge actions.
        """
        values_detached = values.detach()
        if values_detached.numel() > 1:
            next_values = torch.cat([values_detached[1:], bootstrap_scalar.detach().view(1)])
        else:
            next_values = bootstrap_scalar.detach().view(1)
        deltas = rewards_t + self.gamma * next_values - values_detached
        running_advantage = torch.tensor(0.0, dtype=deltas.dtype, device=deltas.device)
        advantages = []
        decay = float(self.gamma) * float(self.gae_lambda)
        for delta in reversed(deltas):
            running_advantage = delta + decay * running_advantage
            advantages.append(running_advantage)
        advantages.reverse()
        advantages_t = torch.stack(advantages).detach()
        targets_t = (advantages_t + values_detached).detach()
        return targets_t, advantages_t

    def _policy_distribution(
        self,
        logits: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> Categorical:
        temp = max(0.05, float(self.policy_temperature if temperature is None else temperature))
        adjusted_logits = logits
        if abs(temp - 1.0) > 1e-9:
            adjusted_logits = adjusted_logits / temp
        return Categorical(logits=adjusted_logits)

    def _update_if_ready(self, bootstrap_value: Optional[torch.Tensor] = None, force: bool = False) -> bool:
        if self._qlearning_enabled():
            return self._update_qlearning_if_ready(force=force)
        if not self.rollout_buffer:
            return False
        if not force and len(self.rollout_buffer) < self.rollout_len:
            return False

        if bootstrap_value is None:
            bootstrap_scalar = torch.tensor(0.0)
        else:
            bootstrap_scalar = bootstrap_value.detach().view(())

        running_return = bootstrap_scalar
        returns = []
        for step in reversed(self.rollout_buffer):
            running_return = step["reward"] + self.gamma * running_return
            returns.append(running_return)
        returns.reverse()

        rewards_t = torch.stack([step["reward"] for step in self.rollout_buffer]).detach()
        returns_t = torch.stack(returns).detach()
        return_std = returns_t.std(unbiased=False) if returns_t.numel() > 1 else torch.tensor(0.0)
        critic_targets_t = returns_t
        td_targets_t = rewards_t.clone()
        nonzero_reward_count = sum(
            1 for step in self.rollout_buffer if abs(float(step["reward"].detach().item())) > 1e-9
        )
        zero_reward_count = len(self.rollout_buffer) - nonzero_reward_count
        actor_loss = torch.tensor(0.0)
        critic_loss = torch.tensor(0.0)
        entropy_loss = torch.tensor(0.0)
        graph_residual_aux_loss = torch.tensor(0.0)
        candidate_q_loss = torch.tensor(0.0)
        advantages = torch.zeros_like(returns_t)
        policy_advantages = torch.zeros_like(returns_t)
        candidate_q_targets_t = returns_t
        candidate_q_pred_mean = 0.0
        candidate_q_target_mean = 0.0
        grad_norm_preclip = 0.0
        actor_grad_norm_preclip = 0.0
        critic_grad_norm_preclip = 0.0
        adv_norm_applied = False
        policy_adv_std_raw = torch.tensor(0.0)
        has_reward_signal = nonzero_reward_count > 0
        has_bootstrap_signal = abs(float(bootstrap_scalar.detach().item())) > 1e-9
        actor_update_allowed = len(self.rollout_buffer) >= getattr(self, "min_actor_update_steps", 2)
        actor_update_has_policy_signal = False
        actor_update_applied = False
        graph_residual_aux_update_applied = False
        candidate_q_update_applied = False
        policy_baseline_mode = getattr(self, "policy_baseline_mode", "critic")

        for _epoch in range(self.update_epochs):
            log_prob_list = []
            entropy_list = []
            value_list = []
            graph_residual_aux_list = []
            candidate_q_list = []
            for step in self.rollout_buffer:
                logits = self.actor(
                    step["action_mat"],
                    step["action_mask"],
                    step["coverage_summary"],
                )
                dist = self._policy_distribution(
                    logits,
                    float(step.get("policy_temperature", self.policy_temperature)),
                )
                log_prob_list.append(dist.log_prob(step["chosen_idx"]))
                entropy_list.append(dist.entropy())
                value_list.append(self.critic(step["critic_input"]).view(()))
                q_values = self._candidate_q_values(
                    step["action_mat"],
                    step["action_mask"],
                    step["coverage_summary"],
                )
                if q_values is not None:
                    chosen_q_idx = int(step["chosen_idx"].long().view(()).item())
                    candidate_q_list.append(q_values[chosen_q_idx].view(()))
                if getattr(self, "graph_residual_aux", False):
                    target = step.get("graph_residual_aux_target")
                    if target is not None:
                        target = target.to(dist.probs.device).float()
                        mask = step["action_mask"].to(dist.probs.device).bool()
                        target = target * mask.float()
                        target_sum = target.sum()
                        if float(target_sum.detach().item()) > 1e-8:
                            target = target / target_sum
                            log_probs_all = torch.log(dist.probs.clamp_min(1e-8))
                            graph_residual_aux_list.append(-(target[mask] * log_probs_all[mask]).sum())

            log_probs = torch.stack(log_prob_list)
            entropies = torch.stack(entropy_list)
            values = torch.stack(value_list)
            if self.advantage_estimator == "td":
                td_targets_t = self._one_step_td_targets(rewards_t, values, bootstrap_scalar)
                critic_targets_t = td_targets_t.detach()
                advantages = critic_targets_t - values
                policy_advantages = critic_targets_t - values.detach()
            elif self.advantage_estimator == "gae":
                critic_targets_t, policy_advantages = self._gae_targets(
                    rewards_t,
                    values,
                    bootstrap_scalar,
                )
                advantages = critic_targets_t - values
            else:
                critic_targets_t = returns_t
                advantages = critic_targets_t - values
                if policy_baseline_mode == "batch_mean":
                    policy_advantages = (returns_t - returns_t.mean()).detach()
                elif policy_baseline_mode == "none":
                    policy_advantages = returns_t.detach()
                else:
                    policy_advantages = advantages.detach()
            policy_advantages = policy_advantages.detach()
            adv_norm_applied = False
            policy_adv_std_raw = torch.tensor(
                0.0,
                dtype=policy_advantages.dtype,
                device=policy_advantages.device,
            )
            if self.normalize_advantages and policy_advantages.numel() > 1:
                policy_adv_std_raw = policy_advantages.std(unbiased=False)
                min_adv_std = max(float(self.advantage_norm_min_std), 1e-8)
                if float(policy_adv_std_raw.detach().item()) > min_adv_std:
                    policy_advantages = (
                        policy_advantages - policy_advantages.mean()
                    ) / policy_adv_std_raw
                    adv_norm_applied = True
            actor_update_has_policy_signal = (
                self.advantage_estimator == "gae"
                and policy_advantages.numel() > 1
                and float(policy_adv_std_raw.detach().item())
                > max(float(self.advantage_norm_min_std), 1e-8)
            )
            if self.advantage_estimator == "gae":
                actor_update_applied = (
                    actor_update_allowed
                    and has_reward_signal
                    and actor_update_has_policy_signal
                )
            else:
                actor_update_applied = actor_update_allowed and has_reward_signal

            if actor_update_applied:
                actor_loss = -(log_probs * policy_advantages).mean()
                entropy_loss = -entropies.mean()
            else:
                actor_loss = torch.tensor(0.0, device=values.device)
                entropy_loss = torch.tensor(0.0, device=values.device)
            critic_loss = torch.mean((critic_targets_t - values) ** 2)
            if graph_residual_aux_list:
                graph_residual_aux_loss = torch.stack(graph_residual_aux_list).mean()
                graph_residual_aux_update_applied = True
            else:
                graph_residual_aux_loss = torch.tensor(0.0, device=values.device)
                graph_residual_aux_update_applied = False
            if candidate_q_list:
                candidate_q_predictions = torch.stack(candidate_q_list)
                candidate_q_targets_t = self._candidate_q_targets(
                    returns_t,
                    critic_targets_t,
                    td_targets_t,
                    policy_advantages,
                ).to(candidate_q_predictions.device)
                candidate_q_loss = torch.nn.functional.smooth_l1_loss(
                    candidate_q_predictions,
                    candidate_q_targets_t.detach(),
                )
                candidate_q_update_applied = True
                candidate_q_pred_mean = float(candidate_q_predictions.detach().mean().item())
                candidate_q_target_mean = float(candidate_q_targets_t.detach().mean().item())
            else:
                candidate_q_loss = torch.tensor(0.0, device=values.device)
                candidate_q_update_applied = False
            total_loss = (
                actor_loss
                + self.value_coef * critic_loss
                + self.entropy_coef * entropy_loss
                + float(getattr(self, "graph_residual_aux_coef", 0.0)) * graph_residual_aux_loss
                + float(getattr(self, "candidate_q_aux_coef", 0.0)) * candidate_q_loss
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            policy_side_params = self._policy_side_parameters()
            actor_grad_norm_preclip = parameter_grad_norm(policy_side_params)
            critic_grad_norm_preclip = parameter_grad_norm(self.critic.parameters())
            if self.separate_grad_clip:
                actor_clip_norm = float(torch.nn.utils.clip_grad_norm_(
                    policy_side_params,
                    max_norm=self.max_grad_norm,
                ))
                critic_clip_norm = float(torch.nn.utils.clip_grad_norm_(
                    list(self.critic.parameters()),
                    max_norm=self.max_grad_norm,
                ))
                grad_norm_preclip = math.sqrt(actor_clip_norm * actor_clip_norm + critic_clip_norm * critic_clip_norm)
            else:
                grad_norm_preclip = float(torch.nn.utils.clip_grad_norm_(
                    self._policy_side_parameters() + list(self.critic.parameters()),
                    max_norm=self.max_grad_norm,
                ))
            self.optimizer.step()

        self.total_updates += 1
        adv_std_after = advantages.detach().std(unbiased=False) if advantages.numel() > 1 else torch.tensor(0.0)
        policy_adv_std = (
            policy_advantages.detach().std(unbiased=False)
            if policy_advantages.numel() > 1
            else torch.tensor(0.0)
        )
        signal_diag = format_learning_signal_diagnostics(
            rewards_t,
            returns_t,
            raw_advantages=advantages,
            policy_advantages=policy_advantages,
            grad_norm=grad_norm_preclip,
            max_grad_norm=self.max_grad_norm,
            actor_grad_norm=actor_grad_norm_preclip,
            critic_grad_norm=critic_grad_norm_preclip,
        )
        print(
            f"[{self.log_prefix}][update] "
            f"idx={self.total_updates} steps={len(self.rollout_buffer)} "
            f"gamma={self.gamma:.4f} update_epochs={self.update_epochs} "
            f"advantage_estimator={self.advantage_estimator} "
            f"policy_baseline={policy_baseline_mode} "
            f"gae_lambda={self.gae_lambda:.3f} "
            f"adv_norm={1 if self.normalize_advantages else 0} "
            f"adv_norm_min_std={self.advantage_norm_min_std:.4f} "
            f"min_actor_update_steps={getattr(self, 'min_actor_update_steps', 2)} "
            f"actor_update_applied={1 if actor_update_applied else 0} "
            f"actor_update_has_reward_signal={1 if has_reward_signal else 0} "
            f"actor_update_has_bootstrap_signal={1 if has_bootstrap_signal else 0} "
            f"actor_update_has_policy_signal={1 if actor_update_has_policy_signal else 0} "
            f"graph_residual_aux_update_applied={1 if graph_residual_aux_update_applied else 0} "
            f"candidate_q_update_applied={1 if candidate_q_update_applied else 0} "
            f"adv_norm_applied={1 if adv_norm_applied else 0} "
            f"policy_adv_std_raw={policy_adv_std_raw.detach().item():.4f} "
            f"actor_loss={actor_loss.item():.4f} critic_loss={critic_loss.item():.4f} "
            f"graph_residual_aux_loss={graph_residual_aux_loss.item():.4f} "
            f"candidate_q_loss={candidate_q_loss.item():.4f} "
            f"candidate_q_target={getattr(self, 'candidate_q_aux_target', 'off')} "
            f"candidate_q_pred_mean={candidate_q_pred_mean:.4f} "
            f"candidate_q_target_mean={candidate_q_target_mean:.4f} "
            f"return_mean={returns_t.mean().item():.4f} return_std={return_std.item():.4f} "
            f"critic_target_mean={critic_targets_t.mean().item():.4f} "
            f"td_target_mean={td_targets_t.mean().item():.4f} "
            f"adv_mean={advantages.mean().item():.4f} "
            f"adv_std={adv_std_after.item():.4f} "
            f"policy_adv_mean={policy_advantages.mean().item():.4f} "
            f"policy_adv_std={policy_adv_std.item():.4f} "
            f"nonzero_reward_count={nonzero_reward_count} zero_reward_count={zero_reward_count} "
            f"separate_grad_clip={1 if self.separate_grad_clip else 0} "
            f"{signal_diag} "
            f"F={self._coverage_value():.3f} "
            f"{self._coverage_counts_text()}"
        )
        self.rollout_buffer.clear()
        return True

    def _flush_episode(self, reason: str = "episode_flush") -> None:
        self._drop_pending(reason)
        self._update_if_ready(bootstrap_value=None, force=True)

    def finalize_run(self, final_state: Optional[WebState] = None, reason: str = "run_end") -> None:
        finalized_pending = 0
        if self.pending_step is not None:
            if final_state is not None:
                self._ensure_graph_state_identity(final_state)
                reward = self._finalize_pending(final_state)
                reward = self.last_train_reward
                finalized_pending = 1
                transition = self.last_finalized_transition or {}
                print(
                    f"[{self.log_prefix}][finalize] "
                    f"reason={reason} final_pending_finalized=1 reward={reward:.3f} "
                    f"reward_raw={float(transition.get('raw_reward', reward)):.3f} "
                    f"reward_train={float(transition.get('train_reward', reward)):.3f} "
                    f"F={self._coverage_value():.3f} {self._coverage_counts_text()}"
                )
            else:
                self._drop_pending(f"{reason}:missing_final_state")
        flushed = self._update_if_ready(bootstrap_value=None, force=True)
        print(
            f"[{self.log_prefix}][finalize] "
            f"reason={reason} final_pending_finalized={finalized_pending} "
            f"final_rollout_flushed={1 if flushed else 0} "
            f"remaining_rollout={len(self.rollout_buffer)} F={self._coverage_value():.3f} "
            f"{self._coverage_counts_text()}"
        )

    def get_action(self, web_state: WebState, html: str) -> WebAction:
        self.total_steps += 1
        self._ensure_graph_state_identity(web_state)
        obs_reward = self._finalize_pending(web_state)
        finalized_transition = self.last_finalized_transition
        self._seed_initial_frontier(web_state)

        batch = self._prepare_batch(web_state, html)
        if batch.kept_action_count == 0:
            raise NoActionsException("No actions available")

        if getattr(self, "graph_residual_aux", False):
            graph_residual_aux_target = self._graph_residual_aux_target(batch, web_state)
        else:
            graph_residual_aux_target = torch.zeros_like(batch.action_mask, dtype=torch.float32)
        coverage_summary = self._coverage_summary()
        critic_input = self._critic_input(batch, coverage_summary)
        if self._qlearning_enabled():
            self._update_qlearning_if_ready(
                next_batch=batch,
                next_coverage_summary=coverage_summary.detach(),
                force=True,
            )
            critic_value = torch.tensor(0.0)
        else:
            with torch.no_grad():
                bootstrap_value = self.critic(critic_input).detach()
            self._update_if_ready(bootstrap_value=bootstrap_value)
            critic_value = self.critic(critic_input)

        if batch.kept_action_count == 1 and isinstance(batch.actions_policy[0], RestartAction):
            chosen_action = batch.actions_policy[0]
            self._flush_episode()
            max_prob = 1.0
            entropy = torch.tensor(0.0)
            chosen_idx = torch.tensor(0, dtype=torch.long)
            probs = torch.zeros_like(batch.action_mask, dtype=torch.float32)
            probs[0] = 1.0
        else:
            logits = self.actor(batch.action_mat, batch.action_mask, coverage_summary)
            q_values = None
            q_epsilon = 0.0
            q_explore = False
            if self._qlearning_enabled():
                chosen_idx, q_values, probs, q_epsilon, q_explore = self._choose_qlearning_action(
                    batch.action_mat,
                    batch.action_mask,
                    coverage_summary,
                )
                chosen_action = batch.actions_policy[int(chosen_idx.item())]
                valid_q = q_values.detach().float()[batch.action_mask.bool()]
                max_prob = float(valid_q.max().item()) if valid_q.numel() else 0.0
                entropy = torch.tensor(0.0)
                critic_value = q_values[int(chosen_idx.item())].detach().view(())
            else:
                dist = self._policy_distribution(logits)
                chosen_idx = dist.sample()
                chosen_action = batch.actions_policy[int(chosen_idx.item())]
                max_prob = float(dist.probs[batch.action_mask].max().detach().item())
                entropy = dist.entropy()
                probs = dist.probs.detach()
            self.pending_step = {
                "action_mat": batch.action_mat.detach(),
                "action_mask": batch.action_mask.detach(),
                "coverage_summary": coverage_summary.detach(),
                "critic_input": critic_input.detach(),
                "policy_temperature": float(self.policy_temperature),
                "chosen_idx": chosen_idx.detach().cpu(),
                "chosen_action": chosen_action,
                "source_url": str(getattr(web_state, "raw_url", getattr(web_state, "url", "")) or ""),
                "source_node_id": extract_graph_node_id(web_state),
                "step": self.total_steps,
                "action_count_full": batch.original_action_count,
                "action_count_policy": batch.kept_action_count,
                "truncated_count": batch.truncated_count,
                "truncated_by_type": dict(batch.truncated_by_type),
                "action_type": action_kind(chosen_action),
                "action_signature": action_signature(chosen_action),
                "graph_residual_aux_target": graph_residual_aux_target.detach(),
                "optimizer_mode": self.optimizer_mode,
                "qlearning_epsilon": q_epsilon,
                "qlearning_explore": q_explore,
            }

        coverage_action_diag = self._coverage_action_diagnostics(
            web_state,
            batch,
            probs,
            int(chosen_idx.item()),
            extract_graph_node_id(web_state),
            graph_residual_aux_target,
        )
        graph_residual_aux_brief = self._graph_residual_aux_console_brief(coverage_action_diag)
        policy_diag_record = self.policy_diagnostics.record_step(
            step=self.total_steps,
            web_state=web_state,
            batch=batch,
            probs=probs,
            chosen_idx=int(chosen_idx.item()),
            entropy=float(entropy.detach().item()),
            max_prob=max_prob,
            finalized_transition=finalized_transition,
            F=self._coverage_value(),
            zero_gain_streak=self.zero_gain_streak,
            value=float(critic_value.detach().item()),
            extra_diagnostics=coverage_action_diag,
        )
        policy_brief = format_policy_brief(policy_diag_record) if policy_diag_record else ""
        if policy_diag_record and self.policy_diagnostics.console_enabled:
            print(f"[{self.log_prefix}][policy] step={self.total_steps} {policy_brief}")

        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.observer.record_step(
            step=self.total_steps,
            web_state=web_state,
            chosen_action=chosen_action,
            actual_reward=obs_reward,
            max_q_value=max_prob,
            history_states=[],
            elapsed_seconds=elapsed,
            marg_style_reward=obs_reward,
            action_global_count=1,
        )

        try:
            print(
                f"[{self.log_prefix}] "
                f"step={self.total_steps} reward={obs_reward:.3f} "
                f"prev_reward_raw={float((finalized_transition or {}).get('raw_reward', 0.0)):.3f} "
                f"prev_reward_train={float((finalized_transition or {}).get('train_reward', 0.0)):.3f} "
                f"prev_action_signature={(finalized_transition or {}).get('action_signature', '')} "
                f"F={self._coverage_value():.3f} "
                f"{self._coverage_counts_text()} "
                f"gamma={self.gamma:.4f} "
                f"optimizer_mode={self.optimizer_mode} "
                f"qlearning_epsilon={float((self.pending_step or {}).get('qlearning_epsilon', 0.0)):.3f} "
                f"qlearning_explore={1 if bool((self.pending_step or {}).get('qlearning_explore', False)) else 0} "
                f"policy_family={self.policy_family} "
                f"policy_temperature={self.policy_temperature:.3f} "
                f"action_count_full={batch.original_action_count} "
                f"action_count_policy={batch.kept_action_count} "
                f"mask_valid={int(batch.action_mask.sum().item())} "
                f"action_batch_ms={float(getattr(batch, 'build_elapsed_ms', 0.0)):.1f} "
                f"action_rank_ms={float(getattr(batch, 'rank_elapsed_ms', 0.0)):.1f} "
                f"action_context_ms={float(getattr(batch, 'context_elapsed_ms', 0.0)):.1f} "
                f"action_tensor_ms={float(getattr(batch, 'tensor_elapsed_ms', 0.0)):.1f} "
                f"truncated_count={batch.truncated_count} "
                f"truncated_by_type={dict(batch.truncated_by_type)} "
                f"entropy={float(entropy.detach().item()):.3f} "
                f"value={float(critic_value.detach().item()):.3f} max_prob={max_prob:.3f} "
                f"type={action_kind(chosen_action)} "
                f"chosen_action_signature={action_signature(chosen_action)} "
                f"zero_gain_streak={self.zero_gain_streak} "
                f"{graph_residual_aux_brief} "
                f"policy_diag=\"{policy_brief}\" action={chosen_action}"
            )
        except UnicodeEncodeError:
            print(
                f"[{self.log_prefix}] "
                f"step={self.total_steps} reward={obs_reward:.3f} "
                f"F={self._coverage_value():.3f} "
                f"gamma={self.gamma:.4f} "
                f"optimizer_mode={self.optimizer_mode} "
                f"action_count_full={batch.original_action_count} "
                f"action_count_policy={batch.kept_action_count} "
                f"mask_valid={int(batch.action_mask.sum().item())}"
                f" action_batch_ms={float(getattr(batch, 'build_elapsed_ms', 0.0)):.1f}"
            )
        return chosen_action
