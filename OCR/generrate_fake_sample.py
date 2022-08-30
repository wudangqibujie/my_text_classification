import os
from PIL import Image, ImageDraw, ImageFont
import random

print(os.listdir("C:\Windows\Fonts"))
DATA_DIR = "../../data/ocr/capture"
if_exist = os.path.exists(DATA_DIR)
if not if_exist:
    os.makedirs(DATA_DIR)

# bg_colors = ['#747D9E', '#BFB5B4', '#A1C8CD']
# word_colors = ['#9063A4', '#2F1C32', '#0F1418']
# word = "学编程的小伙伴最大的愿望估计是有一天自己的编程水平秒杀众人，用起来信手拈来，但是怎么能突破原有的舒适圈，真正提高自己的实力，话说罗马不是一日建成的，朝着高水平不断努力，才能潜移默化中离传说中的“大神”又近一步。现在，我们先跟着别人的经验体会一下“晋升”之路，期待你的突破。 "
# # for i in word:
# words = list(word)
# for ix in range(20):
#     # (40, 40)为生成的文字图片尺寸
#     image = Image.new("RGB", (280, 35), color=random.choice(bg_colors))
#     draw_table = ImageDraw.Draw(im=image)
#     # xy=(5, 0)文字在图片中的位置，font为生成文字字体及字符大小
#     draw_table.text(xy=(5, 0), text="".join([random.choice(words) for _ in range(10)]), fill=random.choice(word_colors),
#                     font=ImageFont.truetype('C:\Windows\Fonts\方正粗黑宋简体.ttf', 25))
#     image.save(os.path.join(DATA_DIR, f"{ix}.jpg"))


class FakeDataset:
    def __init__(self, corpus_file_name, im_file_predix="", max_length=20, label_file_name=os.path.join(DATA_DIR, "label.txt")):
        self.corpus_file = open(corpus_file_name, encoding="utf-8")
        self.max_length = max_length
        self.label_file = open(label_file_name, "w", encoding="utf-8")
        self.bg_colors = ['#747D9E', '#BFB5B4', '#A1C8CD']
        self.word_colors = ['#9063A4', '#2F1C32', '#0F1418']
        self.im_file_predix = im_file_predix

    def _parse_line(self, line):
        line = line.strip()
        line = line.split(",")[-1]
        return line

    def _cut_line(self, line):
        lines = []
        cut_line = ""
        cut_length = 0
        for ix, i in enumerate(line):
            if cut_length > self.max_length:
                lines.append(cut_line)
                cut_line = ""
                cut_length = 0
                continue
            cut_line += i
            cut_length += 1
        return lines

    def write_im(self, line, file_name):
        image_length = int(20 * len(line))
        image = Image.new("RGB", (image_length, 35), color=random.choice(self.bg_colors))
        draw_table = ImageDraw.Draw(im=image)
        draw_table.text(xy=(5, 0), text=line, fill=random.choice(self.word_colors),
                        font=ImageFont.truetype('C:\Windows\Fonts\方正粗黑宋简体.ttf', 25))
        im_f = os.path.join(DATA_DIR, file_name)
        image.save(im_f)

    def run(self):
        flg = 0
        for ix, line in enumerate(self.corpus_file):
            line = self._parse_line(line)
            lines = self._cut_line(line)
            for text in lines:
                im_file_name = f"{self.im_file_predix}_{flg}.jpg"
                self.write_im(text, im_file_name)
                self.label_file.write(f"{im_file_name}\t{text.strip()}\n")
                flg += 1
            if flg > 1000:
                break

if __name__ == '__main__':
    fakeDataset = FakeDataset("../../THU/财经.txt", "财经")
    fakeDataset.run()