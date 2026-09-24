import json
import os
import logging
from datetime import datetime

import warnings

from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

def _read_env_int(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Time (allow override from runner)
time_limit = _read_env_int("WEBRLED_TIME_LIMIT", 3600)

# Log
logger = None


def init_logger(appname):
    file_name = os.path.join(APP_DIR_PATH, f'{appname}.log')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(file_name)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    # ch.setLevel(logging.INFO)
    ch.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def remove_logger(logger):
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


# PATH
## Run PATH
SETTINGS_FILE_PATH = os.path.abspath(__file__)
SRC_DIR_PATH = os.path.dirname(SETTINGS_FILE_PATH)

WEBNEXT_DIR_PATH = os.path.dirname(SRC_DIR_PATH)
JACOCO_PATH = os.path.join(WEBNEXT_DIR_PATH, 'jacoco')
JACOCO_COV_PATH = os.path.join(WEBNEXT_DIR_PATH, 'jacoco', 'html', 'index.html')
DBSQL_PATH = os.path.join(WEBNEXT_DIR_PATH, 'dbsql')
DATA_PATH = os.path.join(WEBNEXT_DIR_PATH, 'data')

RUN_DIR_PATH = os.path.join(SRC_DIR_PATH, 'run')
CURRENT_RUN_DIR_NAME = 'run-{}'.format(datetime.now().strftime("%m%d-%H%M%S"))
CURRENT_RUN_DIR_PATH = os.path.join(RUN_DIR_PATH, CURRENT_RUN_DIR_NAME)
os.makedirs(CURRENT_RUN_DIR_PATH, exist_ok=True)
APP_DIR_PATH = CURRENT_RUN_DIR_PATH  # its value is None, Change according to the web app
EPISODES_DIR_PATH = CURRENT_RUN_DIR_PATH
## Model PATH
MODEL_DIR_PATH = os.path.join(SRC_DIR_PATH, 'models')
BERT_MODEL_PATH = os.path.join(MODEL_DIR_PATH, 'all-MiniLM-L6-v2')
MARKUPLM_MODEL_PATH = os.path.join(MODEL_DIR_PATH, 'markuplm-base')
WEBEMBED_MODEL_PATH = os.path.join(MODEL_DIR_PATH, 'webembed')
SVM_MODEL_PATH = os.path.join(WEBEMBED_MODEL_PATH, 'webEmbed_content_tag_size300_epoch30_fix_seed_classifier.pkl')
## Config PATH
CONFIG_JSON_PATH = os.path.join(SRC_DIR_PATH, 'config.json')


def update_path_for_new_app(site_name):
    global APP_DIR_PATH, logger, EPISODES_DIR_PATH
    APP_DIR_PATH = os.path.join(CURRENT_RUN_DIR_PATH, site_name)
    os.makedirs(APP_DIR_PATH, exist_ok=True)
    # EPISODES_DIR_PATH = os.path.join(APP_DIR_PATH, 'episodes')
    # os.makedirs(EPISODES_DIR_PATH) if not os.path.exists(EPISODES_DIR_PATH) else None
    if logger is not None:
        remove_logger(logger)
    logger = init_logger(site_name)


# Web Applications
WEB_APPS = [
    'petclinic', 'retroboard', 'dimeshift', 'splittypie', 'phoenix', 'pagekit',
    'realworld', 'timeoff', 'parabank', 'agilefant', 'gadael', '4gaboards',
    'nextcloud', 'github', 'odoo'
]
WEB_INFO = {
    'petclinic': {
        'url': 'http://localhost:8081',
        'domain': 'http://localhost:8081',
        'cov_type': 'nyc',
        'coverage_mode': 'external_instrumentation_required',
        'benchmark': 'simple'
    },
    'retroboard': {
        'url': 'http://localhost:4004',
        'domain': 'http://localhost:4004',
        'cov_type': 'nyc',
        'coverage_url': 'http://localhost:6969',
        'benchmark': 'simple'
    },
    'dimeshift': {
        'url': 'http://localhost:4000/',
        'domain': 'http://localhost:4000/',
        'cov_type': 'nyc',
        'coverage_url': 'http://localhost:6969',
        'benchmark': 'simple'
    },
    'splittypie': {
        'url': 'http://localhost:4200',
        'domain': 'http://localhost:4200',
        'cov_type': 'nyc',
        'coverage_mode': 'external_instrumentation_required',
        'benchmark': 'simple'
    },
    'phoenix': {
        'url': 'http://localhost:4003',
        'domain': 'http://localhost:4003',
        'cov_type': 'nyc',
        'coverage_url': 'http://localhost:6969',
        'benchmark': 'simple'
    },
    'pagekit': {
        'url': 'http://localhost:4001/pagekit/index.php/admin/login',
        'domain': 'http://localhost:4001/pagekit/index.php/admin/',
        'cov_type': 'nyc',
        'coverage_url': 'http://localhost:6969',
        'benchmark': 'simple',
        'username': os.environ.get('WEBTEST_PAGEKIT_USERNAME', ''),
        'password': os.environ.get('WEBTEST_PAGEKIT_PASSWORD', ''),
    },
    'parabank': {
        'url': 'http://localhost:8080/parabank',
        'domain': 'http://localhost:8080/parabank',
        'cov_type': 'jacoco',
        'coverage_url': 'http://localhost:6969',
        'benchmark': 'complex',
    },
    'agilefant': {
        'url': 'http://localhost:8084/agilefant',
        'domain': 'http://localhost:8084/agilefant',
        'cov_type': 'jacoco',
        'coverage_url': 'http://localhost:6969',
        'username': os.environ.get('WEBTEST_AGILEFANT_USERNAME', ''),
        'password': os.environ.get('WEBTEST_AGILEFANT_PASSWORD', ''),
        'benchmark': 'complex',
    },
    'timeoff': {
        'url': 'http://localhost:3002/',
        'domain': 'http://localhost:3002',
        'cov_type': 'jacoco',
        'coverage_url': 'http://localhost:6971',
        'username': os.environ.get('WEBTEST_TIMEOFF_EMAIL', ''),
        'password': os.environ.get('WEBTEST_TIMEOFF_PASSWORD', ''),
        'benchmark': 'complex',
    },
    'gadael': {
        'url': 'http://localhost:3001',
        'domain': 'http://localhost:3001',
        'cov_type': 'jacoco',
        'coverage_url': 'http://localhost:6970',
        'username': os.environ.get('WEBTEST_GADAEL_EMAIL', ''),
        'password': os.environ.get('WEBTEST_GADAEL_PASSWORD', ''),
        'benchmark': 'complex',
    },
	'4gaBoards': {
        'url': 'http://localhost:3000',
        'domain': 'http://localhost:3000',
        'cov_type': 'error',
        'coverage_mode': 'postrun_nyc',
        'coverage_report_path': 'server/coverage/index.html',
        'username': os.environ.get('WEBTEST_4GABOARDS_USERNAME', ''),
        'email': os.environ.get('WEBTEST_4GABOARDS_EMAIL', ''),
        'password': os.environ.get('WEBTEST_4GABOARDS_PASSWORD', ''),
        'benchmark': 'complex',
    },
    'realworld': {
        'url': 'https://demo.realworld.show',
        'domain': 'demo.realworld.show',
        'cov_type': 'error',
        'coverage_mode': 'postrun_nyc',
        'coverage_report_path': 'coverage/index.html',
        'benchmark': 'complex'
    },
    'nextcloud': {
        'url': 'http://localhost:8082/apps/files/',
        'domain': 'localhost:8082',
        'cov_type': 'error',
        'coverage_mode': 'unavailable_current_deployment',
        'username': os.environ.get('WEBTEST_NEXTCLOUD_USERNAME', ''),
        'password': os.environ.get('WEBTEST_NEXTCLOUD_PASSWORD', ''),
        'benchmark': 'complex'
    },
    'github': {
        'url': 'https://github.com',
        'domain': 'github.com',
        'cov_type': 'error',
        'coverage_mode': 'unavailable_remote_site',
        'benchmark': 'complex'
    },
    'odoo': {
        'url': 'https://runbot.odoo.com/',
        'domain': 'odoo.com',
        'cov_type': 'error',
        'coverage_mode': 'unavailable_remote_site',
        'benchmark': 'complex'
    },
    'other': {
        'url': 'https://www.quora.com/',
        'domain': 'https://www.quora.com/',
        'cov_type': 'error'
    },
}
WEB_INFO['4gaboards'] = WEB_INFO['4gaBoards']
REAL_APPS_URL = [
    'https://www.google.com/',
    'https://www.youtube.com/',
    'https://www.baidu.com/',
    'https://www.facebook.com/',
    'https://www.bilibili.com/',
    'https://www.qq.com/',
    'https://www.amazon.com/',
    'https://www.instagram.com/',
    'https://www.wikipedia.org/',
    'https://www.x.com/',
    'https://www.bing.com/',
    'https://www.yahoo.com/',
    'https://www.zhihu.com/',
    'https://www.reddit.com/',
    'https://www.whatsapp.com/',
    'https://www.linkedin.com/',
    'https://www.microsoft.com/',
    'https://www.taobao.com/',
    'https://www.163.com/',
    'https://www.csdn.net/',
    'https://dzen.ru/',
    'https://www.vk.com/',
    'https://www.tiktok.com/',
    'https://www.github.com/',
    'https://www.weibo.com/',
    'https://www.jd.com/',
    'https://www.sina.com.cn',
    'https://www.canva.com/',
    'https://www.netflix.com/',
    'https://www.office.com/',
    'https://www.zoom.us/',
    'https://www.fandom.com/',
    'https://www.aliexpress.com/',
    'https://www.naver.com/',
    'https://www.paypal.com/',
    'https://www.apple.com/',
    'https://www.ebay.com/',
    'https://www.mail.ru/',
    'https://www.openai.com/',
    'https://www.amazon.in/',
    'https://www.sohu.com/',
    'https://www.douban.com/',
    'https://www.imdb.com/',
    'https://www.stackoverflow.com/',
    'https://www.pinterest.com/',
    'https://www.tmall.com/',
    'https://www.adobe.com/',
    'https://www.msn.com/',
    'https://www.spotify.com/',
    'https://www.quora.com/',
]
REAL_APPS_DOMAIN = [
    'google',
    'youtube.com',
    'baidu.com',
    'facebook.com',
    'bilibili.com',
    'qq.com',
    'amazon.com',
    'instagram.com',
    'wikipedia.org',
    'x.com',
    'bing.com',
    'yahoo.com',
    'zhihu.com',
    'reddit.com',
    'whatsapp.com',
    'linkedin.com',
    'microsoft.com',
    'taobao.com',
    '163.com',
    'csdn.net',
    'dzen.ru',
    'vk.com',
    'tiktok.com',
    'github.com',
    'weibo.com',
    'jd.com',
    'sina.com.cn',
    'canva.com',
    'netflix.com',
    'office.com',
    'zoom.us',
    'fandom.com',
    'aliexpress.com',
    'naver.com',
    'paypal.com',
    'apple.com',
    'ebay.com',
    'mail.ru',
    'openai.com',
    'amazon.in',
    'sohu.com',
    'douban.com',
    'imdb.com',
    'stackoverflow.com',
    'pinterest.com',
    'tmall.com',
    'adobe.com',
    'msn.com',
    'spotify.com',
    'quora.com',
]


def get_parameters_from_json(args, json_path=CONFIG_JSON_PATH):
    with open(CONFIG_JSON_PATH, 'r') as config_file:
        config = json.load(config_file)
    for key, value in config.items():
        setattr(args, key, value)

    int_overrides = [
        "num_rollout",
        "burnin_length",
        "unroll_length",
        "n_agent_burnin",
        "n_learner_cycle",
        "update_iter",
        "batch_size",
    ]
    for key in int_overrides:
        raw = os.environ.get(f"WEBRLED_{key.upper()}", "").strip()
        if raw:
            try:
                setattr(args, key, int(raw))
            except ValueError:
                pass

    seed_override = os.environ.get("WEBRLED_SEED", "").strip()
    if seed_override:
        try:
            args.seed = int(seed_override)
        except ValueError:
            pass

    state_comp_override = os.environ.get("WEBRLED_STATE_COMPARATOR", "").strip()
    if state_comp_override:
        args.state_comparator = state_comp_override
    return args


def get_valid_dict(appname):
    web_dict = WEB_INFO.get(appname, WEB_INFO['other'])
    username = web_dict['username'] if 'username' in web_dict else 'secret'
    password = web_dict['password'] if 'password' in web_dict else 'secret'
    valid_dict = {
        'bankname': '123456',
        'routingnumber': '123345678',
        'accountnumber': '123345678',
        'repeatedpassword': password,
        'newpassword': password,
        'password': password,
        'email': 'webcover@example.com',
        'username': username,
        'phone': '123123123',
        'number': '1',
        'startdate': '2024-03-30',
        'enddate': '2024-04-12',
        'date': '2024-01-29',
    }
    return valid_dict
