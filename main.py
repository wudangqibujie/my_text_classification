from pyecharts.charts import Page, Tree
from pyecharts import options as opts


class TreeLabel:
    def __init__(self, label_name):
        self.label_name = label_name
        self.next_cls_labels = dict()


label_root = TreeLabel("root")
node = label_root
f = open("Sample_data/sample_train.txt")
for i in f:
    raw_data = i.strip().split("\t")
    if len(raw_data) < 2:
        continue
    labels, texts = raw_data
    h_labels = labels.split(",")
    for k in h_labels:
        labels_chain = k.split("@")
        if len(labels_chain) == 1:
            if labels_chain[0] in label_root.next_cls_labels:
                node = label_root.next_cls_labels[labels_chain[0]]
            else:
                label_root.next_cls_labels[labels_chain[0]] = TreeLabel(labels_chain[0])
                node = label_root.next_cls_labels[labels_chain[0]]
            continue
        if labels_chain[-1] not in node.next_cls_labels:
            node.next_cls_labels[labels_chain[-1]] = TreeLabel(labels_chain[-1])
        else:
            node = node.next_cls_labels[labels_chain[-1]]


def read_label_tree(tree, plot_contain):
    if len(tree.next_cls_labels) == 0:
        return
    now = dict()
    now["name"] = tree.label_name
    now["children"] = []
    plot_contain["children"].append(now)
    for k, v in tree.next_cls_labels.items():
        read_label_tree(v, now)


def plot_tree_label():
    plot_di = dict()
    plot_di["name"] = "START"
    plot_di["children"] = []
    read_label_tree(label_root, plot_di)
    tree = (Tree().add("", plot_di["children"]).set_global_opts(title_opts=opts.TitleOpts(title="TREE-LABEL")))
    tree.render()



