import os
# 下载词向量模型Sentence Transformer
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.system('huggingface-cli download --resume-download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir ./model/sentence-transformer')
# 下载NLTK相关资源
os.system('git clone https://gitee.com/yzy0612/nltk_data.git  --branch gh-pages')
os.system('cd nltk_data \
mv packages/*  ./ \
cd tokenizers \
unzip punkt.zip \
cd ../taggers \
unzip averaged_perceptron_tagger.zip')
os.system('huggingface-cli download --resume-download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir ./model/sentence-transformer')
# 下载模型
# os.system('huggingface-cli download --resume-download internlm/internlm2-chat-7b --local-dir ./model/internlm2-chat-7b')