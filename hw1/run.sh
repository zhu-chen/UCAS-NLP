# 爬取中文网页
python crawler.py seed_CH.txt result_CH.txt --chinese --delay 0.01 --max_pages 50000
# 爬取英文网页
python crawler.py seed_EN.txt result_EN.txt --english --delay 0.01 --max_pages 10000
# 分析中文网页
python analysis.py result_CH.txt --step 2 --output analysis_CH
# 分析英文网页
python analysis.py result_EN.txt --step 2 --output analysis_EN