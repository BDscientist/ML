#word count

import requests
from bs4 import BeautifulSoup
from collections import Counter

def word_count(url): # 문자열 수집
    html = requests.get(url).text
    # 단어 리스트로 변환
    soup = BeautifulSoup(html, 'html.parser')
    words = soup.text.split()
    # 단어수 카운트
    counter = Counter(words)
    # 통계 출력
    return counter

word_count('http://nomade.kr/vod/')
print(word_count('http://nomade.kr/vod/'))

#한글단어만 추출

import re
import requests
from bs4 import BeautifulSoup
from collections import Counter

def korean_word_count(url):
    html = requests.get(url).text

    soup = BeautifulSoup(html,'html.parser')
    words = soup.text.split()

    words = [word for word in words if re.match(r'^[ᄀ-힣]+$', word)]
    counter = Counter(words)
    return counter

print(korean_word_count("http://nomade.kr/vod"))
print("=========================================================================")
##대망의 class 버전
import requests
from bs4 import BeautifulSoup
from collections import Counter

class WordCount(object):
    def get_text(self,url):
        html = requests.get(url).text
        soup = BeautifulSoup(html,'html.parser')
        return soup.text
 
    def get_list(self,text):
        return text.split()
    
    def __call__(self, url):
        text = self.get_text(url)
        words = self.get_list(text)
        counter = Counter(words)
        return counter

word_count = WordCount()
print(word_count("http://nomade.kr/vod"))
print("=========================================================================")


