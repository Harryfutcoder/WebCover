import csv
import math
import os
import re
import numpy as np
from bs4 import BeautifulSoup, Comment
from transformers import MarkupLMProcessor, MarkupLMModel
from playwright.sync_api import sync_playwright, BrowserType, Page
from playwright.sync_api._generated import Playwright as SyncPlaywright
from gensim.models.doc2vec import Doc2Vec
import gensim
from bs4 import NavigableString, Comment, BeautifulSoup
from src import settings


# Web Driver
def init_browser(width: int, height: int) -> [SyncPlaywright, BrowserType, Page]:
    pw = sync_playwright().start()
    headless = os.environ.get("WEBTEST_FORCE_HEADFUL", "0") != "1"
    browser = pw.chromium.launch(headless=headless)
    context = browser.new_context(viewport={'width': width, 'height': height})
    page = context.new_page()
    return pw, browser, context, page


def get_webembed_model():
    embedding_type = ['all']  # ['content', 'tags', 'content_tags', 'all]

    model_common_suffix = '_model_train_setsize300epoch30.doc2vec.model'
    # content_model_path = os.path.join(settings.WEBEMBED_MODEL_PATH, 'content' + model_common_suffix)
    # tags_model_path = os.path.join(settings.WEBEMBED_MODEL_PATH, 'tags' + model_common_suffix)
    content_tags_model_path = os.path.join(settings.WEBEMBED_MODEL_PATH, 'content_tags' + model_common_suffix)

    # models = []
    # model_content = Doc2Vec.load(content_model_path)
    # model_tags = Doc2Vec.load(tags_model_path)
    try:
        model_content_tags = Doc2Vec.load(content_tags_model_path)
    except Exception as e:
        raise RuntimeError(
            "WebRLED official requires the full webembed Doc2Vec model files under "
            f"{settings.WEBEMBED_MODEL_PATH}. Re-extract src/models/webembed or set "
            "WEBTEST_WEBRLED_PYTHON to an environment that can load gensim Doc2Vec."
        ) from e
    # models.append(model_content)
    # models.append(model_tags)
    # models.append(model_content_tags)
    return model_content_tags


# State Representation
# model_bert = SentenceTransformer(settings.BERT_MODEL_PATH)
model_markupLM_processor = MarkupLMProcessor.from_pretrained(settings.MARKUPLM_MODEL_PATH)
model_markupLM = MarkupLMModel.from_pretrained(settings.MARKUPLM_MODEL_PATH)
model_webembed = get_webembed_model()


def simplify_html_doc(html_doc: str) -> str:
    soup = BeautifulSoup(html_doc, 'lxml')
    soup = soup.body
    comments = soup.findAll(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()
    tag_list = ['script', 'link', 'style']
    for tag_name in tag_list:
        scripts = soup.find_all(tag_name)
        for script in scripts:
            script.extract()
    path_tags = soup.find_all('path')
    for path_tag in path_tags:
        path_tag['d'] = ''

    res = re.sub(r"[\n\t]", "", str(soup))
    return res


def html_structure_content(bs, corpus):
    # corpus[3]: content, tags, content+tags
    try:
        if type(bs) == NavigableString:
            tokens = gensim.utils.simple_preprocess(bs.string)
            if len(tokens) > 0:
                corpus[0].extend(tokens)
                corpus[2].extend(tokens)
            return

        bs_has_name = bs.name is not None
        bs_is_single_tag = str(bs)[-2:] == '/>'

        if bs_has_name and not bs_is_single_tag:
            corpus[1].append(f'<{bs.name}>')
            corpus[2].append(f'<{bs.name}>')
        elif bs_has_name and bs_is_single_tag:
            corpus[1].append(f'<{bs.name}/>')
            corpus[2].append(f'<{bs.name}/>')
        try:
            for c in bs.children:
                if type(c) == Comment:
                    continue
                html_structure_content(c, corpus)
        except Exception as e:
            pass
        if bs_has_name and not bs_is_single_tag:
            corpus[1].append(f'</{bs.name}>')
            corpus[2].append(f'</{bs.name}>')
    except Exception as e:
        print('html structure content error', e)
        pass


def embedding_by_markupLM(html_doc: str) -> np.ndarray:
    try:
        processor = model_markupLM_processor(html_doc, return_tensors="pt", max_length=512,
                                             truncation=True)
        embedding = model_markupLM(**(processor))['pooler_output'].detach().numpy()[0]
    except Exception as msg:
        print(msg)
        embedding = ""
    return embedding


def embedding_by_webembed(html_doc: str) -> np.ndarray:
    soup = BeautifulSoup(html_doc, 'html.parser')
    corpus = ([], [], [])
    html_structure_content(soup, corpus)
    # emb_content = models_webembed[0].infer_vector(corpus[0]).reshape(1, -1)
    # emb_tags = models_webembed[1].infer_vector(corpus[1]).reshape(1, -1)
    model_webembed.random.seed(0)
    emb_content_tags = model_webembed.infer_vector(corpus[2]).reshape(1, -1)
    emb_page = emb_content_tags
    return emb_page.squeeze()


def get_state_from_observation(observation: str, model: str = 'markupLM') -> np.ndarray:
    # if model == 'markupLM':
    #     # simplify
    #     html_doc = simplify_html_doc(observation)
    #     # embedding
    #     return embedding_by_markupLM(html_doc)

    # webbed
    return embedding_by_webembed(observation)


# Normalize
def min_max_normalize(distances):
    min_val = min(distances)
    max_val = max(distances)
    normalized = [(x - min_val) / (max_val - min_val) for x in distances]
    return normalized


def z_score_normalize(distances):
    mean_val = sum(distances) / len(distances)
    std_dev = (sum((x - mean_val) ** 2 for x in distances) / len(distances)) ** 0.5
    normalized = [(x - mean_val) / std_dev for x in distances]
    return normalized


# Similarity
def get_xpath_distance(xpath1: str, xpath2: str) -> int:
    xpath1 = xpath1.split('/')
    xpath2 = xpath2.split('/')
    len1, len2 = len(xpath1), len(xpath2)
    len_min = min(len1, len2)
    i = 0
    while (i < len_min) and (xpath1[i] == xpath2[i]):
        i += 1
    distance = len1 - i + len2 - i
    return distance


def get_xpath_distance_by_gradient(xpath1: list, xpath2: list, max_len: int) -> int:
    len1, len2 = len(xpath1), len(xpath2)
    len_min = min(len1, len2)
    i = 0
    while (i < len_min) and (xpath1[i] == xpath2[i]):
        i += 1
    distance = 0
    for j in range(i, len1):
        distance += max_len - j
    for j in range(i, len2):
        distance += max_len - j
    return distance


def get_page_distance(action_info1: dict, action_info2: dict) -> int:
    point1_x = (action_info1['top'] + action_info1['bottom']) / 2
    point1_y = (action_info1['left'] + action_info1['right']) / 2
    point2_x = (action_info2['top'] + action_info2['bottom']) / 2
    point2_y = (action_info2['left'] + action_info2['right']) / 2
    distance = int(math.sqrt((point2_x - point1_x) ** 2 + (point2_y - point1_y) ** 2))
    return distance


def get_app_name(app_name, url):
    if app_name and app_name != 'other':
        return app_name

    ans = app_name
    with sync_playwright() as pw:
        headless = os.environ.get("WEBTEST_FORCE_HEADFUL", "0") != "1"
        browser = pw.chromium.launch(headless=headless)
        context = browser.new_context()
        page = context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=10000)
            doc = page.content().lower()
            for name in settings.WEB_APPS:
                if name in doc:
                    return name
        except Exception as e:
            print(f"get_app_name probe skipped: {e}")
        finally:
            context.close()
            browser.close()
    return app_name


# Write File
def write_csv(filename: str, data_list: list, start_pointer: int = 0):
    with (open(filename, 'a+', encoding='UTF8', newline='') as f):
        writer = csv.writer(f)
        for data_row in data_list:
            data_raw = str(data_row)
            writer.writerow(data_raw)


def write_file(filename: str, data_list: list):
    with open(filename, 'a+') as file:
        for data_row in data_list:
            data_row = data_row.replace("\n", '\\n')
            file.write(f"{data_row}\n")
