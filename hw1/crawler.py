import requests
from bs4 import BeautifulSoup
import re
import os
import argparse
import urllib.parse
from collections import deque
import time
import random

class WebCrawler:
    def __init__(self, max_pages=50, delay=1):
        self.visited_urls = set()
        self.max_pages = max_pages
        self.delay = delay
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def clean_text(self, text, keep_english=True, keep_chinese=True):
        """清洗文本，去除多余空格，并根据参数保留英文或中文"""
        text = re.sub(r'\s+', ' ', text).strip()
        
        result = ""
        if keep_english and keep_chinese:
            result = text
        elif keep_english:
            result = re.sub(r'[^a-zA-Z\s]', '', text)
        elif keep_chinese:
            result = re.sub(r'[^\u4e00-\u9fff\s]', '', text)
        
        # 输出清洗前后的文本长度
        print(f"清洗前文本长度: {len(text)}, 清洗后文本长度: {len(result)}")

        return result
    
    def get_links(self, url, soup):
        """从网页中提取所有超链接并返回绝对URL"""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urllib.parse.urljoin(url, href)
            # 过滤非http链接
            if absolute_url.startswith('http'):
                links.append(absolute_url)
        return links
    
    def extract_text(self, soup):
        """提取网页中的所有文本内容"""
        for script in soup(["script", "style"]):
            script.decompose()
        
        # 获取所有文本
        news = ''
        title = soup.title.text.strip() if soup.title else ''
        news += title
        for x in soup.find_all('div'):
            for y in x.find_all('p'):
                news += y.text.strip()
        
        return news
    

    def crawl(self, seed_url, output_file, keep_english=True, keep_chinese=True):
        """从种子URL开始爬取网页"""
        queue = deque([seed_url])
        page_count = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            while queue and page_count < self.max_pages:
                url = queue.popleft()
                
                if url in self.visited_urls:
                    continue
                
                blacklist = ['video', 'pdf', 'download', 'doc', 'xls', 'ppt', 'mp3', 'mp4','swahili','italian' , 'kaz', 'thai', 'malay', 'greek','vietnamese','urdu','hindi','app','liuyan','login']
                # 如果url里含有被黑名单字符串，跳过
                if any(black in url for black in blacklist):
                    print(f"跳过黑名单URL: {url}")
                    continue
                
                if not keep_english:
                    whitelist =['people']
                else:
                    whitelist = ['en.people']
                # 如果url里不含有白名单字符串，跳过
                if not any(white in url for white in whitelist):
                    print(f"跳过非白名单URL: {url}")
                    continue

                try:
                    print(f"正在爬取: {url} 已爬取 {page_count}/{self.max_pages} ,已爬取文件大小: {os.path.getsize(output_file)} bytes")
                    response = requests.get(url, headers=self.headers, timeout=10)
                    self.visited_urls.add(url)
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # 提取文本并清洗
                        text = self.extract_text(soup)
                        cleaned_text = self.clean_text(text, keep_english, keep_chinese)
                        
                        # 写入文件
                        f.write(f"{cleaned_text}")
                        print(f"成功写入{len(cleaned_text)}个字符到文件")
                        
                        # 提取链接并加入队列
                        links = self.get_links(url, soup)
                        for link in links:
                            if link not in self.visited_urls:
                                queue.append(link)
                        
                        page_count += 1
                        
                        # 添加延迟，避免过快请求
                        #time.sleep(self.delay + random.random()/500)
                
                except Exception as e:
                    print(f"爬取 {url} 时出错: {str(e)}")
            
            print(f"已完成爬取，共爬取了 {page_count} 个网页")

def main():
    parser = argparse.ArgumentParser(description="网页爬虫工具")
    parser.add_argument("seed_file", help="包含种子URL的文件")
    parser.add_argument("output_file", help="保存爬取结果的输出文件")
    parser.add_argument("--max_pages", type=int, default=50, help="最大爬取页面数量")
    parser.add_argument("--english", action="store_true", help="保留英文内容")
    parser.add_argument("--chinese", action="store_true", help="保留中文内容")
    parser.add_argument("--delay", type=float, default=1.0, help="每次请求之间的延迟时间(秒)")
    
    args = parser.parse_args()
    
    # 如果两个语言选项都没有提供，默认都保留
    keep_english = args.english if (args.english or args.chinese) else True
    keep_chinese = args.chinese if (args.english or args.chinese) else True
    
    # 读取种子URL
    try:
        with open(args.seed_file, 'r', encoding='utf-8') as f:
            seed_urls = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"读取种子文件时出错: {str(e)}")
        return
    
    if not seed_urls:
        print("没有找到有效的种子URL")
        return
    
    # 创建爬虫并开始爬取
    crawler = WebCrawler(max_pages=args.max_pages, delay=args.delay)
    for url in seed_urls:
        crawler.crawl(url, args.output_file, keep_english, keep_chinese)

if __name__ == "__main__":
    main()
