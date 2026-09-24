import json
import os
import time
import random
import copy
from collections import deque

import joblib
from bs4 import BeautifulSoup

from src import settings
from src.settings import time_limit, get_valid_dict, SVM_MODEL_PATH
from src.webenv.config import WebConfig
from src.webenv.actionvalue.grid import GridActionValue
from src.webenv.action_discriminator import ActionDiscriminator
from src.webenv.input_gen import InputGenerator
from src.webenv.utils import get_app_name, get_state_from_observation
from src.webenv.js import js_get_observation_and_actions, js_form_action, js_get_html
from playwright._impl._errors import TimeoutError, TargetClosedError, Error
from sklearn.metrics.pairwise import cosine_similarity
# zyf
import numpy as np


# zyf

try:
    from scripts.external_canonical_recorder import recorder_from_env
except Exception:
    recorder_from_env = None


try:
    import Levenshtein
except ImportError:
    class Levenshtein:
        @staticmethod
        def distance(str1, str2):
            prev = list(range(len(str2) + 1))
            for i, ch1 in enumerate(str1, 1):
                cur = [i]
                for j, ch2 in enumerate(str2, 1):
                    cur.append(min(
                        cur[-1] + 1,
                        prev[j] + 1,
                        prev[j - 1] + (ch1 != ch2),
                    ))
                prev = cur
            return prev[-1]


class EpisodeRun:
    def __init__(self):
        # config
        self.max_steps_per_episode = 21
        self.csv_file = os.path.join(settings.APP_DIR_PATH, 'data.csv')
        self.start_time = time.time()
        # record
        self.episode_num = 0
        self.step_num = 0
        self.continuous_steps = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        # self.current_episode_dir = settings.EPISODES_DIR_PATH

        self.continuous_episodes = 0
        self.flag_episode_unchanged = True

    def reset(self):
        self.episode_num += 1
        self.step_num = 0
        self.continuous_steps = 0
        self.rall = 0
        # self.current_episode_dir = os.path.join(settings.EPISODES_DIR_PATH, str(self.episode_num))
        # os.makedirs(self.current_episode_dir) if not os.path.exists(self.current_episode_dir) else None

        if self.flag_episode_unchanged:
            self.continuous_episodes += 1
        else:
            self.continuous_episodes = 0
            self.flag_episode_unchanged = True

    def update(self, is_cov_changed: bool = False):
        if is_cov_changed:
            reward = 1
            self.continuous_steps = 0
            self.flag_episode_unchanged = False
        else:
            reward = 0
            self.continuous_steps += 1

        done = 0
        if self.continuous_steps >= self.max_steps_per_episode:
            done = 1
        if (time.time() - self.start_time) >= time_limit:
            settings.logger.info("Reached 1 hours.")
            done = 1
        # done = 1 if self.step_num >= self.max_steps_per_episode else 0
        self.step_num += 1
        self.rall += reward
        return reward, done

    def episode_done(self):
        self.recent_rlist.append(self.rall)


class ActionSpace:
    def __init__(self, config_ban_elem):
        # config
        self.config_ban_elem = config_ban_elem
        raw_deny_substrings = os.environ.get("WEBTEST_WEBRLED_ACTION_DENY_SUBSTR", "")
        self.deny_substrings = [
            item.strip().lower()
            for item in raw_deny_substrings.split(",")
            if item.strip()
        ]

        self.action_space = []
        self.prev_action_space = []
        self.prev_action_info = None

        self.action_history = []  # shtml
        self.action_history_len = 2

        self.new_action_space = []

    @property
    def action_num(self):
        return len(self.action_space)

    # Basic
    def as_step(self, action_info):
        self.prev_action_space = self.action_space
        self.prev_action_info = action_info

        self.action_history.append(action_info['shtml'])
        if len(self.action_history) > self.action_history_len:
            self.action_history.pop(0)

        target_index = -1
        for new_action_index in range(len(self.new_action_space)):
            new_action_info = self.new_action_space[new_action_index]
            if action_info['shtml'] == new_action_info['shtml'] and \
                    action_info['xpath'] == new_action_info['xpath']:
                target_index = new_action_index
                break
        if target_index != -1:
            self.new_action_space.pop(target_index)

    @staticmethod
    def remove_value_and_style(html_input):
        soup = BeautifulSoup(html_input, 'lxml').body
        for element in soup.find_all(True):
            del element['value']
            del element['style']
            if element.string is not None:
                element.string = ''
        return str(soup.contents[0])

    def is_contains_banstr(self, html, action_info):
        if self.config_ban_elem:
            return False
        if 'back' in html:
            index = html.find('back')
            res = html[index + 4] if index != -1 and index + 4 < len(html) else ''
            return not res.isalpha()
        # if 'close' in html:
        #     index = html.find('close') - 1
        #     res = html[index] if 0 <= index < len(html) else ' '
        #     return res != '-'
        banstrs = ['cancel', 'delete', 'logout', 'log out', 'sign out', '关闭', '×']
        for banstr in banstrs:
            if banstr in html:
                return True
        if 'disable' in action_info['innerText'].lower():
            return True
        return False

    def is_denied_action(self, action_info):
        if not self.deny_substrings:
            return False
        haystack = " ".join([
            str(action_info.get('outerHTML', '')),
            str(action_info.get('innerText', '')),
        ]).lower()
        return any(deny in haystack for deny in self.deny_substrings)

    def as_update(self, raw_action_space):
        action_space, back_actions = [], []  # back action 是后退元素
        for action_info in raw_action_space:
            action_info['shtml'] = self.remove_value_and_style(action_info['outerHTML'])  # 新增了shtml属性
            html = action_info['outerHTML'].lower()
            if self.is_denied_action(action_info):
                continue
            if action_info['tagName'] in ['A', 'BUTTON'] and self.is_contains_banstr(html, action_info):
                back_actions.append(action_info)
            else:
                action_space.append(action_info)
        if len(action_space) > 0:
            if len(back_actions) > 0:
                flag = True
                for action_info in action_space:
                    if action_info['shtml'] not in self.action_history:
                        flag = False
                        break
                self.action_space = back_actions if flag else action_space
            else:
                self.action_space = action_space
        else:
            self.action_space = back_actions

    def as_reset(self):
        self.prev_action_space, self.prev_action_info = [], None
        self.action_history.clear()
        self.new_action_space = []

    # ad
    def update_action_space_by_ad(self, potential_actions):
        self.action_space += potential_actions

    ## New Random
    @staticmethod
    def is_same_action(action_info_1: dict, action_info_2: dict, strict: bool = True) -> bool:
        condition_1 = action_info_1['xpath'] == action_info_2['xpath']
        condition_2 = action_info_1['shtml'] == action_info_2['shtml']
        res = condition_1 and condition_2 if strict else condition_2
        return res

    @staticmethod
    def str_similarity(str1, str2):
        distance = Levenshtein.distance(str1, str2)
        similarity = 1 - distance / max(len(str1), len(str2))
        return similarity

    def get_new_actions(self):
        if len(self.prev_action_space) == 0:
            return self.action_space
        prev_action_space = self.prev_action_space.copy()
        old_actions, new_actions = [], []
        for action_info in self.action_space:
            flag = False
            equal_prev_action_index = -1
            for prev_action_index in range(len(prev_action_space)):
                prev_action_info = prev_action_space[prev_action_index]
                shtml_equal = action_info['shtml'] == prev_action_info['shtml']
                xpath_equal = action_info['xpath'] == prev_action_info['xpath']
                if (shtml_equal and xpath_equal) or \
                        (xpath_equal and action_info['tagName'] == prev_action_info['tagName'] and
                         self.str_similarity(action_info['shtml'], prev_action_info['shtml']) >= 0.95):
                    flag = True
                    equal_prev_action_index = prev_action_index
                    break
                elif shtml_equal:
                    flag = True
                    equal_prev_action_index = prev_action_index
            if flag:
                old_actions.append(action_info)
                prev_action_space.pop(equal_prev_action_index)
            else:
                new_actions.append(action_info)
        return new_actions

    def random_new_action_show(self):
        new_actions = self.get_new_actions()
        res = None
        if len(new_actions) == 0:
            if len(self.new_action_space):
                res = self.new_action_space
            else:
                res = self.action_space
        else:
            if len(new_actions) == 1 and new_actions[0]['shtml'] in self.action_history:
                self.new_action_space.clear()
                res = self.action_space
            else:
                self.new_action_space = new_actions
                res = self.new_action_space
        return res


# zyf start
class HistoryState:
    def __init__(self, vector, history_trace, tried_actions, avail_actions):
        self.vector = vector
        self.history_trace = history_trace
        self.tried_actions = tried_actions
        self.avail_actions = avail_actions

    vector = None
    # 抵达这个状态的操作集
    history_trace = []
    # 这个状态下，已经执行的，跳到其他状态的“点击”动作
    tried_actions = []
    # 这个状态下，值得尝试的动作
    avail_actions = []


# zyf end


def _load_state_comparator_model():
    try:
        return joblib.load(SVM_MODEL_PATH)
    except Exception as e:
        print(f"WebRLED official: SVM comparator unavailable, using threshold comparator: {e}")
        return None


class WebEnvironment:
    # zyf start
    # 增加历史状态列表，以便从已探索到的状态继续探索
    history_states: list[HistoryState] = []
    history_trace = []
    clf = _load_state_comparator_model()

    def rollette_random_state(self) -> [HistoryState | None]:
        if len(self.history_states) == 0:
            return None
        # 计算累积分布
        weights = list(map(lambda x: max(0., 1 - len(x.tried_actions) / len(x.avail_actions)), self.history_states))
        total_weight = sum(weights)
        if total_weight == 0:
            return None
        normalized_weights = [w / total_weight for w in weights]
        cumulative_distribution = np.cumsum(normalized_weights)

        # 生成一个随机数
        random_number = np.random.rand()

        # 选择候选
        for index, value in enumerate(cumulative_distribution):
            if random_number < value:
                return self.history_states[index]

    def go_to_state(self, state: HistoryState):
        try:
            for action in state.history_trace:
                self.perform_action(action)
        except Exception as e:
            pass

    def update_state_archive(self, vector, tried_action, avail_actions):

        self.history_trace.append(tried_action)
        # 已有类似状态，更新信息
        for state in self.history_states:
            # sim = cosine_similarity(vec1=state.vector, vec2=vector)
            # settings.logger.info('sim:{}'.format(sim))
            is_same_state = self.is_same_state(state.vector, vector)
            if is_same_state:
                # merge avail actions
                for action1 in avail_actions:
                    flag = False
                    for action2 in state.avail_actions:
                        if self.is_same_action2(action1, action2):
                            flag = True
                            break
                    if not flag:
                        state.avail_actions.append(action1)
                # Optional: randomly overwrite history trace
                if random.random() > 0.6:
                    state.history_trace = copy.deepcopy(self.history_trace)
                # already have tried
                for action in state.tried_actions:
                    if self.is_same_action2(action, tried_action):
                        return
                # not tried
                state.tried_actions.append(tried_action)
                return

        # 新状态
        self.history_states.append(
            HistoryState(vector, copy.deepcopy(self.history_trace), [tried_action], avail_actions)
        )

    def is_same_action2(self, action1, action2):
        return self.action_module.str_similarity(action1['shtml'], action2['shtml']) >= 0.9

    def is_same_state(self, state1, state2):
        sim = cosine_similarity(state1.reshape(1, -1), state2.reshape(1, -1))
        sim_value = float(sim[0][0])
        if self.state_comparator == 'svm' and self.clf is not None:
            svm_output = self.clf.predict(sim).tolist()
            is_same_state = svm_output[0] == 0
        else:
            if self.state_comparator != 'threshold':
                self.state_comparator = 'threshold'
            similarity_threshold = 0.999
            is_same_state = sim_value > similarity_threshold
        # settings.logger.info(
        #     'state_comparator:{}, sim:{}, is_same_state:{}'.format(self.state_comparator, sim, is_same_state)
        # )
        return is_same_state

    # zyf end

    def __init__(self, app_name: str = 'other', url: str = '', domain: str = '', flag_coverage: bool = False,
                 state_comparator: str = 'svm'):
        # config
        self.config_offline = False
        self.config_ban_elem = False  # todo
        self.config_click_top_left = False
        self.is_first_time_init = True
        # web app
        self.app_name = get_app_name(app_name, url)
        self.web_config = WebConfig(app_name, url, domain, flag_coverage)
        self.save_pic = False
        self.data_path = settings.DATA_PATH
        # MDP config
        self.state_dim = 300
        self.action_space_dim = 400
        # RL
        self.observation, self.state = None, None  # Raw HTML Document / [768,]
        self.action_module = ActionSpace(self.config_ban_elem)
        # State Machine
        self.state_comparator = state_comparator
        if self.state_comparator == 'svm' and self.clf is None:
            self.state_comparator = 'threshold'
        # Episode
        self.episode_run = EpisodeRun()
        # Input
        self.input_generator = InputGenerator(get_valid_dict(app_name))
        # Action value
        self.gav = GridActionValue()
        # Action discriminator
        self.ad = ActionDiscriminator()
        self.potential_actions = []
        # Observability only: these counters do not affect WebRLED decisions.
        self.metrics_total_actions = 0
        self.metrics_action_signatures = set()
        self.metrics_urls = set()
        self.metrics_last_action = ""
        self.canonical_recorder = (
            recorder_from_env(
                default_algorithm='webrled-official',
                default_site=os.environ.get('WEBTEST_EXTERNAL_CANONICAL_SITE', self.app_name),
                default_seed=os.environ.get('WEBRLED_SEED_TEXT', os.environ.get('WEBRLED_SEED', '')),
                default_profile='drl-1agent-observation',
            )
            if recorder_from_env is not None
            else None
        )
        # Reset
        self.reset()
        self._record_canonical_state(event='init')
        self._write_metrics(event='init')
        settings.logger.info('Initializing WebEnvironment')

    def __del__(self):
        if hasattr(self, 'web_config'):
            self.web_config.__del__()

    @property
    def action_num(self):
        return self.action_module.action_num

    @property
    def start_time(self):
        return self.episode_run.start_time

    # RL interface
    def reset(self):
        self.web_config.manage_browser_pages()
        self.web_config.reset_page()
        self.episode_run.reset()
        if self.is_first_time_init:
            self.web_config.first_init(self.app_name)
            self.is_first_time_init = False
        # Go To state
        target_state = self.rollette_random_state()
        if target_state is not None:
            self.go_to_state(target_state)
        ### Trick +++
        if self.config_offline and (self.episode_run.episode_num + 1) % 10 == 0:
            self.web_config.context.set_offline(self.web_config.online)
            self.web_config.online = not self.web_config.online
        ### Trick ---
        self.update_environment()
        self.action_module.as_reset()
        self.history_trace.clear()

        if self.ad.flag_on:
            self.ad.check(self.episode_run.episode_num, self.episode_run.continuous_episodes)
        if hasattr(self.input_generator, 'reset'):
            self.input_generator.reset(self.episode_run.episode_num, self.episode_run.continuous_episodes)
        return self.state, {}

    def step(self, action_info: dict):
        canonical_source_url = self._current_url()
        canonical_source_actions = copy.deepcopy(self.action_module.action_space)
        covered_points, nearest_point = self.gav.get_covered_points_from_action(action_info)
        # perform action
        self.perform_action(action_info)
        self.action_module.as_step(action_info)

        # update
        is_changed = self.web_config.update_cov_and_errors()
        reward, done = self.episode_run.update(is_changed)
        self.update_environment()
        # zyf start
        self.update_state_archive(
            vector=copy.deepcopy(self.state), tried_action=copy.deepcopy(action_info),
            avail_actions=copy.deepcopy(self.action_module.action_space)
        )
        # zyf end
        self._record_canonical_transition(
            canonical_source_url,
            canonical_source_actions,
            action_info,
        )
        self.metrics_total_actions += 1
        self.metrics_last_action = self._action_signature(action_info)
        self.metrics_action_signatures.add(self.metrics_last_action)
        current_url = self._current_url()
        if current_url:
            self.metrics_urls.add(current_url)
        self._write_metrics(event='step', reward=reward, done=done, action_info=action_info)
        settings.logger.debug(
            "[Episode {}]  Step: {}  branch_coverage: {}  line_coverage: {}".format(
                self.episode_run.episode_num, self.episode_run.step_num, self.web_config.branch_coverage,
                self.web_config.line_coverage
            ))
        if done:
            self.episode_run.episode_done()
            # AD Explore
            if self.ad.flag_on:
                start_time = time.time()
                self.update_environment()
                self.ad.flag_explore = True
                for _ in range(10):
                    # todo
                    try:
                        if self.ad_explore():
                            break
                    except Error as e:
                        settings.logger.debug('AD Explore stop')
                self.ad.flag_explore = False
                self.ad.time_cost += time.time() - start_time
        info = {'covered_points': covered_points, 'nearest_point': nearest_point}
        return self.state, reward, done, info

    def _current_url(self):
        try:
            return self.web_config.page.url
        except Exception:
            return ''

    def _record_canonical_state(self, event='state'):
        if self.canonical_recorder is None:
            return
        try:
            self.canonical_recorder.observe_state(
                self._current_url(),
                copy.deepcopy(self.action_module.action_space),
            )
        except Exception as e:
            settings.logger.debug(
                'Canonical recorder state update failed: {}'.format(str(e).replace('\n', '\\n'))
            )

    def _record_canonical_transition(self, source_url, source_actions, action_info):
        if self.canonical_recorder is None:
            return
        try:
            self.canonical_recorder.record_transition(
                source_url=source_url,
                source_actions=source_actions,
                selected_action=copy.deepcopy(action_info),
                target_url=self._current_url(),
                target_actions=copy.deepcopy(self.action_module.action_space),
            )
        except Exception as e:
            settings.logger.debug(
                'Canonical recorder transition update failed: {}'.format(str(e).replace('\n', '\\n'))
            )

    @staticmethod
    def _clean_metric_part(value, max_len=160):
        text = str(value or '')
        text = ' '.join(text.split())
        if len(text) > max_len:
            text = text[:max_len] + '...'
        return text

    def _action_signature(self, action_info):
        if not action_info:
            return ''
        parts = [
            'type=' + self._clean_metric_part(action_info.get('actiontype')),
            'tag=' + self._clean_metric_part(action_info.get('tagName')),
            'xpath=' + self._clean_metric_part(action_info.get('xpath')),
            'text=' + self._clean_metric_part(action_info.get('innerText')),
            'html=' + self._clean_metric_part(action_info.get('shtml') or action_info.get('outerHTML'), 240),
        ]
        return '|'.join(parts)

    def _write_metrics(self, event='step', reward=None, done=None, action_info=None):
        try:
            os.makedirs(settings.APP_DIR_PATH, exist_ok=True)
            current_url = self._current_url()
            if current_url:
                self.metrics_urls.add(current_url)
            payload = {
                'schema_version': 1,
                'algorithm': 'webrled-official',
                'app_name': self.app_name,
                'session': os.environ.get('WEBRLED_SESSION', ''),
                'seed': os.environ.get('WEBRLED_SEED_TEXT', os.environ.get('WEBRLED_SEED', '')),
                'seed_value': os.environ.get('WEBRLED_SEED', ''),
                'event': event,
                'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'episode': self.episode_run.episode_num,
                'step_in_episode': self.episode_run.step_num,
                'total_actions': self.metrics_total_actions,
                'unique_actions': len(self.metrics_action_signatures),
                'unique_states': len(self.history_states),
                'unique_urls': len(self.metrics_urls),
                'current_url': current_url,
                'last_action_signature': self.metrics_last_action,
                'action_space_size': self.action_module.action_num,
                'branch_coverage': self.web_config.branch_coverage,
                'line_coverage': self.web_config.line_coverage,
                'reward': reward,
                'done': done,
                'coverage_enabled': bool(getattr(self.web_config, 'flag_coverage', False)),
            }
            metrics_path = os.path.join(settings.APP_DIR_PATH, 'metrics.json')
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
            history_path = os.path.join(settings.APP_DIR_PATH, 'metrics_history.jsonl')
            with open(history_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + '\n')
        except Exception as e:
            settings.logger.debug('Write metrics failed: {}'.format(str(e).replace('\n', '\\n')))

    # Get from environment
    def get_observation_and_actions(self) -> dict:
        """
        Get observation and action space. The observation is the HTML documentation.
        The action space is a list of actions, containing several properties for each action.
        :return:
        """
        observation_and_actions = self.web_config.page.evaluate(
            js_get_observation_and_actions, self.web_config.domain
        )
        return observation_and_actions

    def update_action_space(self, action_space):
        # 将动作判别器识别的动作加入到动作空间中
        self.potential_actions = self.ad.recognize(self.potential_actions)
        for action in self.potential_actions:
            action['actiontype'] = 1
        action_space += self.potential_actions
        return action_space

    def update_environment(self, flag_error: bool = False):
        """
        Including: observation, action_space, prev_action_space
        :return:
        """
        # settings.logger.debug('update_environment')
        # settings.logger.debug('Update Environment +')
        self.web_config.page.wait_for_timeout(2000)
        action_space = []
        i = 5
        while True:
            try:
                self.web_config.page.wait_for_load_state(timeout=180000)

                if self.app_name != 'other' and not self.web_config.is_url_in_scope():
                    self.web_config.reset_page()
                    continue

                if self.web_config.ensure_authenticated():
                    self.web_config.page.wait_for_load_state(timeout=180000)
                    continue

                obs_and_actions = self.get_observation_and_actions()
                self.observation, action_space = obs_and_actions['observation'], obs_and_actions['actionSpace']
                action_space = [
                    action for action in action_space
                    if not self.web_config._is_auth_or_logout_action(action)
                ]
                self.potential_actions = [
                    action for action in obs_and_actions['potentialActions']
                    if not self.web_config._is_auth_or_logout_action(action)
                ]
                if len(action_space):
                    break
                else:
                    i -= 1
                    settings.logger.debug('Update Environment Retry Index: ' + str(i))
                    self.web_config.page.wait_for_timeout(2000)
                if i <= 0:
                    settings.logger.debug('=== Update Environment Reset ===')
                    self.web_config.reset_page()
                    i = 5
            except TimeoutError as timeout_error:
                settings.logger.debug('TimeoutError: Update Environment.')
                self.web_config.reset_page()
            except TargetClosedError as target_closed_error:
                settings.logger.debug('TargetClosedError: Update Environment.')
                self.web_config.new_page()
            except Exception as e:
                msg = str(e).replace("\n", '\\n')
                settings.logger.debug('Update Environment.' + msg)
                self.web_config.page.wait_for_timeout(1000)
        if self.ad.flag_enable and not self.ad.flag_explore:
            action_space = self.update_action_space(action_space)
        self.state = get_state_from_observation(self.observation)
        self.action_module.as_update(action_space)
        if flag_error:
            self.action_module.new_action_space.clear()
        settings.logger.debug('update_environment Page: {}'.format(str(self.web_config.page)))

    # Perform in environment
    def perform_action(self, action_info: dict) -> None:
        # settings.logger.debug('perform_action Page: {}'.format(str(self.web_config.page)))
        action_type = action_info['actiontype']
        assert action_type != 0
        if action_type == 1:  # Click
            action = self.web_config.page.locator('xpath=/{}'.format(action_info['xpath']))
            action.click(timeout=self.web_config.action_timeout)
            if self.config_click_top_left and random.random() < 0.35:
                self.web_config.page.mouse.click(0, 0)
            settings.logger.debug(
                'Clicked: {}'.format(action_info["outerHTML"].replace("\n", '\\n').replace("\t", "\\t")))
        elif action_type == 2:  # Type
            action = self.web_config.page.locator('xpath=/{}'.format(action_info['xpath']))
            content = self.input_generator.get_single_input_value(action_info['outerHTML'])
            action.fill(content, timeout=self.web_config.action_timeout)
            action.press("Enter")
            # action.fill(content)
            settings.logger.debug(
                'Typed [{content}]: {html}'.format(
                    content=content, html=action_info["outerHTML"].replace("\n", '\\n').replace("\t", "\\t"))
            )
        elif action_type == 3:
            self.select_action(action_info)
            settings.logger.debug(
                'Select: {html}'.format(
                    html=action_info["outerHTML"].replace("\n", '\\n').replace("\t", "\\t"))
            )
        elif action_type == 4:
            if (
                self.web_config._normalized_app_name() == "gadael"
                and "#/user/settings" in (self.web_config.page.url or "")
            ):
                settings.logger.debug(
                    "Skipped Gadael user settings form action to avoid mutating the active account/session."
                )
                return
            self.form_action(action_info['xpath'])

    def select_action(self, action_info: dict) -> None:
        select_element = self.web_config.page.locator('xpath=/{}'.format(action_info['xpath']))
        options = select_element.locator('option').all_text_contents()
        if len(options) > 1:
            options = options[1:]
        if options:
            chosen_option_label = random.choice(options).strip()
            select_element.select_option(label=chosen_option_label)

    def form_action_normal(self, form_action_xpath) -> None:
        form_actions = self.web_config.page.evaluate(
            js_form_action, form_action_xpath
        )
        settings.logger.debug('form_action_normal.')
        click_actions, submit_actions = [], []
        click_elems, type_elems, = form_actions['clickActions'], form_actions['typeActions']
        type_actions = []
        for elem in type_elems:
            type_actions.append(elem)
        for elem in type_actions:
            elem['value'] = self.input_generator.get_single_input_value(elem['outerHTML'], False)
            action = self.web_config.page.locator('xpath=/{}'.format(elem['xpath']))
            action.fill(elem['value'], timeout=self.web_config.action_timeout)
            if self.input_generator.type or 'enter' in elem['outerHTML'].lower():
                action.press("Enter")  # todo type
        for elem in click_elems:
            if 'submit' in elem['outerHTML'].lower():
                submit_actions.append(elem)
            else:
                click_actions.append(elem)
        if len(submit_actions) == 0:
            submit_elem_index = -1
            for elem_index in range(len(click_actions)):
                elem = click_actions[elem_index]
                if 'login' in elem['outerHTML'].lower() or 'check' in elem['outerHTML'].lower():
                    submit_actions.append(elem)
                    submit_elem_index = elem_index
                    break
            if submit_elem_index != -1:
                click_actions.pop(submit_elem_index)
        if len(submit_actions) > 0:
            submit_action = random.choice(submit_actions) if len(submit_actions) > 1 else submit_actions[0]
            action = self.web_config.page.locator('xpath=/{}'.format(submit_action['xpath']))
            action.click(timeout=self.web_config.action_timeout)
            if self.web_config._normalized_app_name() == "gadael":
                self.web_config.page.wait_for_timeout(1500)
                self.web_config._fix_gadael_base_url()
            else:
                action.click(timeout=self.web_config.action_timeout)
        else:
            if len(click_actions) > 0:
                click_action = random.choice(click_actions)
                action = self.web_config.page.locator('xpath=/{}'.format(click_action['xpath']))
                action.click(timeout=self.web_config.action_timeout)

    def form_action(self, form_action_xpath) -> None:
        try:
            self.form_action_normal(form_action_xpath)
        except Error as te:
            # settings.logger.error('Form action normal interrupted.')
            pass

    def ad_explore(self):
        # 使用动作判别器进行探索
        # 依次点击当前网页上每个div、span等元素，如果页面发生变化，则认为该元素是可执行元素，否则继续点击直到结束
        flag_unchanged = True
        current_html = self.web_config.page.evaluate(js_get_html)
        data, label = [], []
        for index in range(len(self.potential_actions)):
            action_info = self.potential_actions[index]
            if action_info['outerHTML'] not in self.ad.data_cache.ad_dict:
                data.append(action_info['outerHTML'])
                try:
                    action = self.web_config.page.locator('xpath=/{}'.format(action_info['xpath']))
                    action.click(timeout=self.web_config.action_timeout)
                except Error as te:
                    settings.logger.debug('ad explore perform stop:' + str(te))
                    label.append(0)
                    continue
                self.web_config.page.wait_for_load_state(timeout=180000)
                next_html = self.web_config.page.evaluate(js_get_html)
                if next_html != current_html:
                    label.append(1)
                    flag_unchanged = 0
                    self.update_environment()
                    break
                else:
                    label.append(0)

        assert len(data) == len(label)
        self.web_config.update_cov_and_errors()
        if len(label):
            self.ad.update(data, label)
        return flag_unchanged

    def select_action_by_grid_only(self, point_value):
        # 选择动作空间中价值最高的动作
        target_action, target_action_value = None, float("-inf")
        for action_info in self.action_module.action_space:
            action_value = self.gav.get_action_value_from_point(action_info, point_value)
            if action_value > target_action_value:
                target_action, target_action_value = action_info, action_value
        return target_action

    def select_action_by_random(self):
        # 从动作空间中随机选择一个动作
        res = self.action_module.random_new_action_show()
        return random.choice(res)
