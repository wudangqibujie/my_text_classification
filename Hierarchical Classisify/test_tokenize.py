import os
from tokenize_jay.pure_tokenize import Corpus


list_files = []
for c in ['体育', '娱乐', '家居', '彩票', '房产', '教育', '时尚', '时政', '星座', '游戏', '社会', '科技', '股票', '财经']:
    base_folder = os.path.join("../../THUCNews", c)
    list_files += [os.path.join(base_folder, i) for i in os.listdir(base_folder)]


corpus = Corpus(list_files, 'util/vocab.json', 'util/stopword.dic')
corpus.build_vocab()
corpus.write_vocab()

