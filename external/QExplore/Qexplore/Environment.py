import cryptohash as chash
import json
import os
import random
import re
import numpy as np
import pandas as pd
import wordninja
import nltk
import gensim
import enchant
import sister
from bs4 import BeautifulSoup
import string
from num2words import num2words
import matplotlib.pyplot as plt
import requests
from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.chrome.options import Options as ChromeOptions
try:
    from selenium.webdriver.common.by import By
except Exception:
    By = None
try:
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.firefox.service import Service as FirefoxService
except Exception:
    ChromeService = None
    FirefoxService = None
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.ui import Select
from prettytable import PrettyTable
import requests
import time
import exrex as ex
import js_regex
from sklearn.metrics.pairwise import cosine_similarity
import ast
import datetime



class webEnv:
    
    def __init__(self,url,BaseURL="http://localhost/timeclock/",actionWait=0.5):
        self.url = url
        self.tags_to_find = ['input','button','a','select']
        self.website = self._make_webdriver()
        self._install_selenium4_compat(self.website)
        self.website.get(url)
        self.datalabel = ['zipcode','city','streetname','secondaryaddress',
                'county','country','countrycode','state','stateabbr',
                'latitude','longitude','address','email','username',
                'password','sentence','word','paragraph','firstname',
                'lastname','fullname','age','phonenumber','date']
        self.embedding = sister.MeanEmbedding(lang="en")
        self.edict = enchant.Dict('en_US')
        self.tagAttr = {'a':[''],'button':['value','name'],
                        'select':['name','class'],
                   'input':['placeholder','name','value']}
        self.prev_password = None
        self.BaseURL = BaseURL
        self.currentDepth=0
        self.actionWait = actionWait
        self.coverage_base_url = os.environ.get("QEXPLORE_COVERAGE_URL", "").strip().rstrip("/")
        self.coverage_kind = os.environ.get("QEXPLORE_COVERAGE_KIND", "").strip().lower()
        self.coverage_snapshot = {"branch_coverage": "", "line_coverage": ""}
        self.update_coverage_metric(event="init")

    @staticmethod
    def _install_selenium4_compat(driver):
        if By is None:
            return
        if not hasattr(driver, "find_element_by_xpath"):
            driver.find_element_by_xpath = lambda xpath: driver.find_element(By.XPATH, xpath)
        if not hasattr(driver, "find_elements_by_tag_name"):
            driver.find_elements_by_tag_name = lambda tag: driver.find_elements(By.TAG_NAME, tag)

    def _make_webdriver(self):
        browser = os.environ.get("QEXPLORE_BROWSER", "chrome").strip().lower()
        headless = os.environ.get("QEXPLORE_HEADLESS", "1").strip().lower() not in ("0", "false", "no", "off", "headful")
        if browser == "firefox":
            options = FirefoxOptions()
            firefox_binary = os.environ.get("QEXPLORE_FIREFOX_BINARY", "").strip()
            if firefox_binary:
                options.binary_location = firefox_binary
            if headless:
                options.add_argument("-headless")
            geckodriver = os.environ.get("QEXPLORE_GECKODRIVER", "").strip()
            if geckodriver and FirefoxService is not None:
                return webdriver.Firefox(service=FirefoxService(executable_path=geckodriver), options=options)
            if geckodriver:
                return webdriver.Firefox(executable_path=geckodriver, options=options)
            return webdriver.Firefox(options=options)

        options = ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        chrome_binary = os.environ.get("QEXPLORE_CHROME_BINARY", "").strip()
        if not chrome_binary:
            for candidate in (
                r"C:\Users\SUST\Desktop\webTest\webTest\chrome-win\chrome.exe",
                r"C:\Users\SUST\AppData\Local\ms-playwright\chromium-1217\chrome-win64\chrome.exe",
            ):
                if os.path.isfile(candidate):
                    chrome_binary = candidate
                    break
        if chrome_binary:
            options.binary_location = chrome_binary
        chromedriver = os.environ.get("QEXPLORE_CHROMEDRIVER", "").strip()
        if not chromedriver:
            for candidate in (
                os.path.join(os.getcwd(), "chromedriver.exe"),
                r"C:\Users\SUST\Desktop\webTest\webTest\chromedriver.exe",
            ):
                if os.path.isfile(candidate):
                    chromedriver = candidate
                    break
        if chromedriver and ChromeService is not None:
            return webdriver.Chrome(service=ChromeService(executable_path=chromedriver), options=options)
        if chromedriver:
            return webdriver.Chrome(executable_path=chromedriver, options=options)
        return webdriver.Chrome(options=options)

    def _coverage_from_html(self, html):
        pattern = r'<span class="strong">([\d.]+%) </span>\s*<span class="quiet">(\w+)</span>\s*<span class=[\'"]fraction[\'"]>(\d+/\d+)</span>'
        matches = re.findall(pattern, html or "")
        branch_coverage, line_coverage = "", ""
        for percentage, label, fraction in matches:
            if label == "Branches":
                branch_coverage = percentage
            elif label == "Lines":
                line_coverage = percentage
        return branch_coverage, line_coverage

    def update_coverage_metric(self, event="step"):
        if not self.coverage_base_url:
            return
        branch_coverage, line_coverage = "", ""
        try:
            if self.coverage_kind == "nyc":
                try:
                    cov_raw = self.website.execute_script("return window.__coverage__;")
                    if cov_raw is not None:
                        requests.post(
                            self.coverage_base_url + "/coverage/client",
                            data=json.dumps(cov_raw),
                            headers={"Content-Type": "application/json"},
                            timeout=3,
                        )
                except Exception:
                    pass
            response = requests.get(self.coverage_base_url + "/coverage", timeout=3)
            text = response.text or ""
            try:
                payload = response.json()
                branch_coverage = str(payload.get("branch_coverage", ""))
                line_coverage = str(payload.get("line_coverage", ""))
            except Exception:
                branch_coverage, line_coverage = self._coverage_from_html(text)
            self.coverage_snapshot = {
                "event": event,
                "coverage_url": self.coverage_base_url,
                "coverage_kind": self.coverage_kind,
                "branch_coverage": branch_coverage,
                "line_coverage": line_coverage,
                "current_url": self.website.current_url,
                "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
            }
            with open("qexplore_coverage.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(self.coverage_snapshot, sort_keys=True, indent=2))
            with open("qexplore_coverage_history.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(self.coverage_snapshot, sort_keys=True) + "\n")
        except Exception:
            pass
    
    def get_Actions_OR_state(self,action=False):
        tagstr = []
        for x in self.tags_to_find:
            xtags = self.website.find_elements_by_tag_name(x)
            if xtags!=[]:
                for element in xtags:
                    s = x+"!@!"
                    if x=='input':
                        try:
                            if element.get_attribute('name')is not '' and element.get_attribute('name')is not None:
                                s+=element.get_attribute('name').strip()+'!@!'
                            else:
                                s+='nan!@!'
                        except:
                            s+='nan!@!'
                        try:
                            value_attr = element.get_attribute('value')
                            type_attr = element.get_attribute('type')
                        
                            if value_attr is not '' and value_attr is not None and type_attr not in ["text","password"]:
                                s+=element.get_attribute('value').strip()+'!@!'
                                #s+='nan!@!'
                            else:
                                s+='nan!@!'
                        except:
                            s+='nan!@!'
                        s+='nan'
                        
                    elif x=='button':
                        s+='nan!@!' #name
                        try:
                            if element.get_attribute('value') is not '' and element.get_attribute('value') is not None:
                                s+=element.get_attribute('value').strip()+'!@!'
                            else:
                                s+='nan!@!'
                        except:
                            s+='nan!@!'
                        s+='nan'
                    elif x=='a':
                        s+='nan!@!' #name
                        s+='nan!@!' #value
                        try:
                            if element.get_attribute('href') is not '' and element.get_attribute('href') is not None:
                                s+=element.get_attribute('href').strip()
                            else:
                                s+='nan'
                        except:
                            s+='nan'
                    elif x=='select':
                        try:
                            if element.get_attribute('name') is not '' and element.get_attribute('name') is not None:
                                s+=element.get_attribute('name').strip()+'!@!'
                            else:
                                s+='nan!@!'
                        except:
                            s+='nan!@!'
                        s+='nan!@!' #value
                        s+='nan'    #href
                    tagstr.append(s)
        if action:
            dedup_list = []
            for i in tagstr:
                if i not in dedup_list:
                    dedup_list.append(i)

            #tagstr = list(set(tagstr))
            #tagstr.sort(reverse=True)
            return dedup_list
        else:
            for i in range(len(tagstr)):
                x = tagstr[i].split("!@!")
                if x[0]=="a":
                    x[-1]="#"
                x = "!@!".join(x)
                tagstr[i]=x
            return "\n".join(tagstr)

    def reverseEngineerAction(self,action):
        tag,name,value,href = action.split('!@!')
        xpath='//'+tag+'['
        att = []
        hreflist = []
        XPATH = ""
        if name!='nan':
            att.append('@name='+'"'+name+'"')
        if value!='nan':
            att.append('@value='+'"'+value+'"')
        if href!='nan':
            hreflist.append(href)
            for x in range(1,len(href.split("/"))):
                temphref = "/".join(href.split("/")[x::])
                hreflist.append(temphref)
                hreflist.append("../"+temphref)
                hreflist.append("/"+temphref)
        elem = None
        if hreflist==[]:
            try:
                XPATH = "//"+tag if att==[] else xpath+" and ".join(att)+"]"
                elem = self.website.find_element_by_xpath(XPATH)
            except:
                pass
        else:
            for x in hreflist:
                try:
                    _att = []
                    _att.extend(att)
                    _att.append("@href="+"'"+x+"'")
                    XPATH = xpath+" and ".join(_att)+"]"
                    elem = self.website.find_element_by_xpath(XPATH)
                    break
                except:
                    continue
            if elem==None:
                #print("none")
                try:
                    x="#"
                    _att = []
                    _att.extend(att)
                    _att.append("@href="+"'"+x+"'")
                    XPATH = xpath+" and ".join(_att)+"]"
                    elem = self.website.find_element_by_xpath(XPATH)
                except:
                    pass
        return elem
            

    def get_random_string(self,length):
        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(length))
        return result_str

    def getvectors(self,sentences):
        vector = self.embedding(sentences)
        return vector
    
    def getsimilarity(self, feature_vec_1, feature_vec_2):    
        return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]
    
    def getdistance(self,a,b):
        return np.linalg.norm(a-b)

    def _generated_input_value(self, label):
        service_url = os.environ.get("QEXPLORE_INPUT_SERVICE_URL", "").strip()
        if service_url:
            try:
                r = requests.get(url=service_url, params={"value": label}, timeout=3)
                d = ast.literal_eval((r.text or "").replace("`", ""))
                value = d.get("'d'", [""])[0]
                if value:
                    return value
            except:
                pass
        fallback = {
            "zipcode": "12345",
            "city": "Shanghai",
            "streetname": "Main Street",
            "secondaryaddress": "Apt 1",
            "county": "County",
            "country": "China",
            "countrycode": "CN",
            "state": "Shanghai",
            "stateabbr": "SH",
            "latitude": "31.2304",
            "longitude": "121.4737",
            "address": "123 Main Street",
            "email": "qexplore@example.com",
            "username": "qexplore_user",
            "password": "Qexplore123!",
            "sentence": "qexplore test sentence",
            "word": "qexplore",
            "paragraph": "qexplore test paragraph",
            "firstname": "Q",
            "lastname": "Explore",
            "fullname": "Q Explore",
            "age": "30",
            "phonenumber": "1234567890",
            "date": "01/01/2026",
        }
        return fallback.get(label, "abcd123456")
    
    def getSentence(self,html):
        soup = BeautifulSoup(html,'lxml')
        sentence = ""
        table = str.maketrans('', '', string.punctuation)
        for tag in self.tags_to_find:
            for t in soup.findAll(tag):
                att = t.attrs
                for chose in self.tagAttr[tag]:
                    try:
                        v = att[chose]
                        if type(v)==list:
                            sentence+=" ".join(v)+" "
                        else:
                            sentence+=' '+v
                    except:
                        continue
                
                sentence+=t.text
                #sentence = t.name+" "+sentence
        #print("before: ",sentence)
        #or cls in self.bootclasses:
        #   sentence = sentence.strip().replace(cls,'')
        #or cls in self.stopwords:
        #   sentence = sentence.strip().replace(cls,'')
            
        sentence = sentence.strip().translate(table)
        sentence = sentence.lower().replace('lastname','last-name')
        sentence = sentence.replace('firstname','first-name')
        sentence = sentence.replace('username','user-name')
        sentence = sentence.replace('userid','user-name')
        sentence = sentence.replace('enddate','end-date')
        sentence = sentence.replace('startdate','start-date')
        sentence = sentence.replace('cnic','identitynumber')
        #print("after class: ",sentence)
        sentence_new = ""
        for num in wordninja.split(sentence):
            word = ''.join([i for i in num if not i.isdigit()])
            try:
                if self.edict.check(word) and len(word)>1:
                    sentence_new+=word+" "
                    #print(word)
            except:
                pass
        #print("after ninja: ",sentence_new)
        return " ".join(list(set(sentence_new.split(" "))))
    
    #This method execute each element of the DOM depending on the type of element
    
    def click(self,elem):
        try:
            elem.click()
            time.sleep(self.actionWait)
            return 1
        except:
            try:
                self.website.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", elem)
                time.sleep(0.1)
                self.website.execute_script("arguments[0].click();", elem)
                time.sleep(self.actionWait)
                return 1
            except:
                return 0        
    
    def write(self,elem,login_url,username,password,email):
        html = elem.get_attribute("outerHTML")
        sentence = self.getSentence(html)
        if sentence!="":
            v_sentence = self.getvectors(sentence)
            #print("***********************")
            simL = []
            for x in self.datalabel:
                x_vector = self.getvectors(x)
                sim = self.getsimilarity(x_vector,v_sentence)
                simL.append(sim)
                #print(x+" is "+str(sim)+" similar to '"+sentence+"'")
            mostsim = self.datalabel[np.argmax(simL)]
            #print("'"+sentence+"' is most similar to "+mostsim)
            d = {"'d'": [self._generated_input_value(mostsim)]}
            if mostsim=='date':
                try:
                    date = d["'d'"][0].split("T")[0]
                    date = datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%m/%d/%Y')
                    d["'d'"][0]=date
                except:
                    d["'d'"][0]="01/01/2026"
            if mostsim=="paragraph" or mostsim=="word":
                d["'d'"][0] = "abcd123456"
            if mostsim=="username":
                if self.website.current_url in login_url:
                    if username!=None:
                        d["'d'"][0] = username
                else:
                    if username!=None:
                        d["'d'"][0] = username
                    print(self.website.current_url,login_url,self.website.current_url in login_url)
            if mostsim=="email":
                if self.website.current_url in login_url:
                    if email!=None:
                        d["'d'"][0] = email
                else:
                    print(self.website.current_url,login_url,self.website.current_url in login_url)
            if mostsim=="password":
                if self.website.current_url in login_url:
                    if password!=None:
                        d["'d'"][0] = password
                else:
                    if password!=None:
                        d["'d'"][0] = password
                        
                    if self.prev_password!=None:
                        d["'d'"][0] = self.prev_password
                    else:
                        self.prev_password = d["'d'"][0]

            #print("Generated: ",d["'d'"][0]," mostsim=",mostsim)
            try:
                if elem.get_attribute("value")=="" or elem.get_attribute("value")==None:
                    if elem.get_attribute("value")!=d["'d'"][0]:
                        elem.send_keys(d["'d'"][0])
                return 1
            except:
                return 0
        else:
            d = {"'d'": [self._generated_input_value("word")]}
            try:
                if elem.get_attribute("value")=="" or elem.get_attribute("value")==None:
                    if elem.get_attribute("value")!=d["'d'"][0]:
                        elem.send_keys(d["'d'"][0])
                return 1
                #elem.send_keys(d["'d'"][0])
                #return 1
            except:
                return 0
               
    def checkDone(self,depth):
        if self.currentDepth>=depth:
            return True
        else:
            return False
    
    
    def step(self,elem,login_url="",username=None,password=None,depth=4,email=None):
        
        clickable = ["a","button","submit","select","radio","checkbox","image"]
        writable = ["input","text","password","search"]
        status = 0
        
        if elem.tag_name in clickable or elem.get_attribute('Type') in clickable:
            if elem.tag_name=="select":
                try:
                    select = Select(elem)
                    option = random.choice(select.options)
                    status = self.click(option)
                except:
                    status = 0
            else:
                status = self.click(elem)
                
            if status:
                self.currentDepth+=1
                
        elif elem.tag_name in writable or elem.get_attribute('Type') in writable:
            status = self.write(elem,login_url,username,password,email)
            if status:
                self.currentDepth+=1
        else:
            return 0,self.checkDone(depth)
        
        self.update_coverage_metric(event="step")
        return status,self.checkDone(depth)

    def reset(self,curl=""):
        if curl=="":
            self.website.get(self.url)
            self.currentDepth=0
            self.prev_password = None
        else:
            self.website.get(curl)
            #self.currentDepth=0
            #self.prev_password = None
        self.update_coverage_metric(event="reset")
        
    def close(self):
        try:
            self.website.close()
            self.website.close()
        except:
            pass
