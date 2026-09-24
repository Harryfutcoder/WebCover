import os
import os.path
import random
import re
import sys
from copy import deepcopy

import yaml

from config.cli_options import cli_options
from fairness import apply_fair_env_defaults


class Settings:
    AUTO_SITE_TEMPLATE_SUFFIXES = (
        "drl-1agent-observation",
        "webexplor-1agent",
        "qlearning-1agent",
        "random-1agent",
        "subweb-frontier-a2c-1agent",
    )

    # New large-site shortcuts: default endpoints can be overridden by env:
    #   WEBTEST_SITE_<SITE>_ENTRY_URL
    #   WEBTEST_SITE_<SITE>_DOMAINS   (comma-separated)
    AUTO_SITE_DEFAULTS = {
        "odoo": {
            "entry_url": "http://localhost:8069",
            "domains": ["localhost:8069", "localhost"],
        },
        "discourse": {
            "entry_url": "http://localhost:4203",
            "domains": ["localhost:4203", "localhost"],
        },
        "nextcloud": {
            "entry_url": "http://localhost:8082",
            "domains": ["localhost:8082", "localhost"],
        },
        "4gaboards": {
            "entry_url": "http://localhost:3000",
            "domains": ["localhost:3000", "localhost"],
        },
        "agilefant": {
            "entry_url": "http://localhost:8084",
            "domains": ["localhost:8084", "localhost"],
        },
    }

    def __init__(self) -> None:
        self.settings_path = cli_options.settings
        self.output_path = cli_options.output
        self.model_path = cli_options.model_path
        self.restart_interval = cli_options.restart_interval
        self.continuous_restart_threshold = cli_options.continuous_restart_threshold
        self.enable_screen_shot = cli_options.enable_screen_shot
        self.profile = cli_options.profile
        self.session = cli_options.session
        self.agent_num = cli_options.agent_num
        self.record_interval = None
        self.alive_time = None
        self.page_load_timeout = None
        self.browser_path = None
        self.browser_data_path = None
        self.driver_path = None
        self.resources_path = None
        self.entry_url = None
        self.domains = None
        self.browser_arguments = None
        self.action_detector = None
        self.state = None
        self.agent = None
        self.agent_module = None
        self.agent_class = None

    @staticmethod
    def _resolve_config_path(path: str, base_dir: str) -> str:
        """Relative paths in YAML are resolved against the directory of settings.yaml."""
        if not path:
            return path
        path = os.path.expandvars(os.path.expanduser(path))
        if os.path.isabs(path):
            return os.path.normpath(path)
        return os.path.normpath(os.path.join(base_dir, path))

    @staticmethod
    def _read_env_int(name: str):
        raw = os.environ.get(name, "").strip()
        if not raw:
            return None
        try:
            value = int(raw)
        except ValueError:
            return None
        return value

    @classmethod
    def _try_bootstrap_profile(cls, settings_data: dict, profile_name: str) -> bool:
        profiles = settings_data.get("profiles", {})
        if profile_name in profiles:
            return True

        # Split as "<site>-<template_suffix>" at the first hyphen.
        # Using a greedy regex here can mis-parse names like
        # "odoo-drl-1agent-observation" into
        # site="odoo-drl-1agent", suffix="observation".
        if "-" not in profile_name:
            return False
        site, suffix = profile_name.split("-", 1)
        site = site.lower().strip()
        suffix = suffix.strip()
        if not site or not suffix:
            return False

        if suffix == "subweb-frontier-a2c-1agent":
            site_template = f"{site}-qlearning-1agent"
            agent_template = "realworld-subweb-frontier-a2c-1agent"
            if site_template in profiles and agent_template in profiles:
                cfg = deepcopy(profiles[site_template])
                cfg["agent"] = deepcopy(profiles[agent_template]["agent"])
                profiles[profile_name] = cfg
                settings_data["profiles"] = profiles
                print(
                    f"Auto-bootstrapped profile '{profile_name}' from '{site_template}' "
                    f"with agent from '{agent_template}'"
                )
                return True

        if site not in cls.AUTO_SITE_DEFAULTS:
            return False
        if suffix not in cls.AUTO_SITE_TEMPLATE_SUFFIXES:
            return False

        template_profile = f"realworld-{suffix}"
        if template_profile not in profiles:
            return False

        cfg = deepcopy(profiles[template_profile])
        default_entry = cls.AUTO_SITE_DEFAULTS[site]["entry_url"]
        default_domains = list(cls.AUTO_SITE_DEFAULTS[site]["domains"])

        env_key = site.upper().replace("-", "_")
        entry_override = os.environ.get(f"WEBTEST_SITE_{env_key}_ENTRY_URL", "").strip()
        domains_override = os.environ.get(f"WEBTEST_SITE_{env_key}_DOMAINS", "").strip()

        cfg["entry_url"] = entry_override or default_entry
        if domains_override:
            cfg["domains"] = [x.strip() for x in domains_override.split(",") if x.strip()]
        else:
            cfg["domains"] = default_domains

        profiles[profile_name] = cfg
        settings_data["profiles"] = profiles
        print(
            f"Auto-bootstrapped profile '{profile_name}' from '{template_profile}' "
            f"(entry_url={cfg['entry_url']}, domains={cfg['domains']})"
        )
        return True

    def load_settings(self) -> None:
        with open(self.settings_path, 'r', encoding='utf-8') as f:
            settings_data = yaml.safe_load(f)
            if self.output_path is None:
                self.output_path = settings_data['default_output_path']
            if self.profile is None:
                self.profile = settings_data['default_profile']
            if self.session is None:
                self.session = settings_data['default_session']
            if self.model_path is None:
                self.model_path = settings_data['default_model_path']
            if self.restart_interval is None:
                self.restart_interval = settings_data['default_restart_interval']
            if self.continuous_restart_threshold is None:
                self.continuous_restart_threshold = settings_data['default_continuous_restart_threshold']
            if self.enable_screen_shot is None:
                self.enable_screen_shot = settings_data['default_enable_screen_shot']
            env_enable_screenshot = os.environ.get("WEBTEST_ENABLE_SCREENSHOT")
            if env_enable_screenshot is not None:
                self.enable_screen_shot = env_enable_screenshot.strip().lower() in ("1", "true", "yes", "on")
            if self.profile not in settings_data['profiles']:
                self._try_bootstrap_profile(settings_data, self.profile)

            if self.profile not in settings_data['profiles']:
                print(f"Profile \"{self.profile}\" not exist", file=sys.stderr)
                sys.exit(1)
            if self.session == settings_data['default_session']:
                folder_name = self.profile + "-" + self.session + "-" + format(random.randint(0, 0xFFFFFF),
                                                                               '06x')
            else:
                folder_name = self.profile + "-" + self.session
            output_path = os.path.join(self.output_path, folder_name)
            while True:
                if os.path.exists(output_path):
                    if self.session == settings_data['default_session']:
                        folder_name = self.profile + "-" + self.session + "-" + format(random.randint(0, 0xFFFFFF),
                                                                                       '06x')
                        output_path = os.path.join(self.output_path, folder_name)
                    else:
                        # 自定义 session：文件夹已存在则直接使用，避免无限循环
                        break
                else:
                    break
            self.output_path = output_path
            self.agent_num = settings_data['profiles'][self.profile]['agent_num']
            if cli_options.agent_num is not None:
                self.agent_num = cli_options.agent_num
            self.record_interval = settings_data['profiles'][self.profile]['record_interval']
            self.alive_time = settings_data['profiles'][self.profile]['alive_time']
            self.page_load_timeout = settings_data['profiles'][self.profile]['page_load_timeout']
            env_record_interval = self._read_env_int("WEBTEST_RECORD_INTERVAL_OVERRIDE")
            if env_record_interval is not None and env_record_interval > 0:
                self.record_interval = env_record_interval
            env_alive_time = self._read_env_int("WEBTEST_ALIVE_TIME_OVERRIDE")
            if env_alive_time is not None and env_alive_time > 0:
                self.alive_time = env_alive_time
            env_page_load_timeout = self._read_env_int("WEBTEST_PAGE_LOAD_TIMEOUT_OVERRIDE")
            if env_page_load_timeout is not None and env_page_load_timeout > 0:
                self.page_load_timeout = env_page_load_timeout
            _cfg_dir = os.path.dirname(os.path.abspath(self.settings_path))
            self.browser_path = self._resolve_config_path(
                settings_data['profiles'][self.profile]['browser_path'], _cfg_dir
            )
            self.browser_data_path = self._resolve_config_path(
                settings_data['profiles'][self.profile]['browser_data_path'], _cfg_dir
            )
            self.driver_path = self._resolve_config_path(
                settings_data['profiles'][self.profile]['driver_path'], _cfg_dir
            )
            self.resources_path = self._resolve_config_path(
                settings_data['profiles'][self.profile]['resources_path'], _cfg_dir
            )
            self.entry_url = settings_data['profiles'][self.profile]['entry_url']
            self.domains = settings_data['profiles'][self.profile]['domains']
            # Allow env var override for entry_url/domains even when profile already exists in YAML.
            # This lets the experiment runner point any site at a public URL without editing settings.yaml.
            _site = self.profile.split("-")[0].lower()
            _env_key = _site.upper().replace("-", "_")
            _entry_override = os.environ.get(f"WEBTEST_SITE_{_env_key}_ENTRY_URL", "").strip()
            _domains_override = os.environ.get(f"WEBTEST_SITE_{_env_key}_DOMAINS", "").strip()
            if _entry_override:
                self.entry_url = _entry_override
            if _domains_override:
                self.domains = [x.strip() for x in _domains_override.split(",") if x.strip()]
            self.browser_arguments = settings_data['profiles'][self.profile]['browser_arguments']
            self.action_detector = settings_data['profiles'][self.profile]['action_detector']
            self.state = settings_data['profiles'][self.profile]['state']
            self.agent = settings_data['profiles'][self.profile]['agent']
            self.agent_module = os.environ.get("WEBTEST_AGENT_MODULE", self.agent["module"])
            self.agent_class = os.environ.get("WEBTEST_AGENT_CLASS", self.agent["class"])
            apply_fair_env_defaults(self.agent_module, self.agent_class)
            _mod = self.agent_module
            _cls = self.agent_class
            if (_mod == "agent.impl.drl_agent" and _cls == "DRLagent") or \
               (_mod == "agent.impl.webexplor_agent" and _cls == "WebExplorAgent"):
                self.load_drl_agent_cli_options()
            elif _mod == "agent.impl.q_learning_agent" and _cls == "QLearningAgent":
                self.load_q_learning_agent_cli_options()
            else:
                self.load_generic_agent_cli_options()

    def load_generic_agent_cli_options(self) -> None:
        self.agent["params"]["alive_time"] = self.alive_time
        self.agent["params"]["agent_num"] = self.agent_num
        self.agent["params"]["entry_url"] = self.entry_url
        if cli_options.model_module is not None:
            self.agent["params"]["model_module"] = cli_options.model_module
        if cli_options.model_class is not None:
            self.agent["params"]["model_class"] = cli_options.model_class
        if cli_options.model_load_type is not None:
            self.agent["params"]["model_load_type"] = cli_options.model_load_type
        if cli_options.model_load_name is not None:
            self.agent["params"]["model_load_name"] = cli_options.model_load_name
        if cli_options.transformer_module is not None:
            self.agent["params"]["transformer_module"] = cli_options.transformer_module
        if cli_options.transformer_class is not None:
            self.agent["params"]["transformer_class"] = cli_options.transformer_class
        if cli_options.reward_function is not None:
            self.agent["params"]["reward_function"] = cli_options.reward_function
        if cli_options.stop_update is not None:
            self.agent["params"]["stop_update"] = cli_options.stop_update
        if cli_options.batch_size is not None:
            self.agent["params"]["batch_size"] = cli_options.batch_size
        if cli_options.learning_rate is not None:
            self.agent["params"]["learning_rate"] = cli_options.learning_rate
        if cli_options.gamma is not None:
            self.agent["params"]["gamma"] = cli_options.gamma
        if cli_options.max_random is not None:
            self.agent["params"]["max_random"] = cli_options.max_random
        if cli_options.min_random is not None:
            self.agent["params"]["min_random"] = cli_options.min_random
        if cli_options.min_random is not None:
            self.agent["params"]["min_random"] = cli_options.min_random
        if cli_options.update_target_interval is not None:
            self.agent["params"]["update_target_interval"] = cli_options.update_target_interval
        if cli_options.update_target_interval is not None:
            self.agent["params"]["update_network_interval"] = cli_options.update_network_interval
        if cli_options.update_mixing_network_interval is not None:
            self.agent["params"]["update_mixing_network_interval"] = cli_options.update_mixing_network_interval


    def load_drl_agent_cli_options(self) -> None:
        self.agent["params"]["alive_time"] = self.alive_time
        if cli_options.model_module is not None:
            self.agent["params"]["model_module"] = cli_options.model_module
        if cli_options.model_class is not None:
            self.agent["params"]["model_class"] = cli_options.model_class
        if cli_options.model_load_type is not None:
            self.agent["params"]["model_load_type"] = cli_options.model_load_type
        if cli_options.model_load_name is not None:
            self.agent["params"]["model_load_name"] = cli_options.model_load_name
        if cli_options.transformer_module is not None:
            self.agent["params"]["transformer_module"] = cli_options.transformer_module
        if cli_options.transformer_class is not None:
            self.agent["params"]["transformer_class"] = cli_options.transformer_class
        if cli_options.reward_function is not None:
            self.agent["params"]["reward_function"] = cli_options.reward_function
        if cli_options.stop_update is not None:
            self.agent["params"]["stop_update"] = cli_options.stop_update
        if cli_options.batch_size is not None:
            self.agent["params"]["batch_size"] = cli_options.batch_size
        if cli_options.learning_rate is not None:
            self.agent["params"]["learning_rate"] = cli_options.learning_rate
        if cli_options.gamma is not None:
            self.agent["params"]["gamma"] = cli_options.gamma
        if cli_options.max_random is not None:
            self.agent["params"]["max_random"] = cli_options.max_random
        if cli_options.min_random is not None:
            self.agent["params"]["min_random"] = cli_options.min_random
        if cli_options.min_random is not None:
            self.agent["params"]["min_random"] = cli_options.min_random
        if cli_options.update_target_interval is not None:
            self.agent["params"]["update_target_interval"] = cli_options.update_target_interval
        if cli_options.update_target_interval is not None:
            self.agent["params"]["update_network_interval"] = cli_options.update_network_interval


    def load_q_learning_agent_cli_options(self) -> None:
        if cli_options.agent_type is not None:
            self.agent["params"]["agent_type"] = cli_options.agent_type
        if cli_options.alpha is not None:
            self.agent["params"]["alpha"] = cli_options.alpha
        if cli_options.gamma is not None:
            self.agent["params"]["gamma"] = cli_options.gamma
        if cli_options.epsilon is not None:
            self.agent["params"]["epsilon"] = cli_options.epsilon
        if cli_options.initial_q_value is not None:
            self.agent["params"]["initial_q_value"] = cli_options.initial_q_value
        if cli_options.r_reward is not None:
            self.agent["params"]["r_reward"] = cli_options.r_reward
        if cli_options.r_penalty is not None:
            self.agent["params"]["r_penalty"] = cli_options.r_penalty
        if cli_options.max_sim_line is not None:
            self.agent["params"]["max_sim_line"] = cli_options.max_sim_line


settings = Settings()
settings.load_settings()
