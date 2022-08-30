import simplejson
import gzip
import tensorflow as tf


cate_set = ["Amazon Instant Video" ,
"Arts",
"Automotive",
"Baby",
"Beauty",
"Books",
"Cell Phones & Accessories",
"Clothing & Accessories",
"Electronics",
"Gourmet Foods",
"Health",
"Home & Kitchen",
"Industrial & Scientific",
"Jewelry",
"Kindle Store",
"Movies & TV",
"Musical Instruments",
"Music",
"Office Products",
"Patio",
"Pet Supplies",
"Shoes",
"Software",
"Sports & Outdoors",
"Tools & Home Improvement",
"Toys & Games",
"Video Games",
"Watches",]


pth = r"E:\all.txt\all.txt"
# f = open(pth)
# cnt = 0
# for i in f:
#     if cnt > 200:
#         break
#     cnt += 1
#     print(i.strip())

def parse(filename):
  f = gzip.open(filename, 'r')
  entry = {}
  for l in f:
    l = l.strip().decode()
    colonPos = l.find(':')
    if colonPos == -1:
      yield entry
      entry = {}
      continue
    eName = l[:colonPos]
    rest = l[colonPos+2:]
    entry[eName] = rest
  yield entry



cnt = 9
cate_pth = r"E:\all.txt\categories.txt\categories.txt"
f = open(cate_pth)
id_set = set()
for i in f:
    raw = i.strip()
    line = i.strip().split(",")
    if line[0] not in cate_set:
        print(raw)
        id_set.add(raw)
    if cnt > 2000:
        break
    cnt += 1

# import json
# for e in parse(r"E:\all.txt.gz"):
#     d = simplejson.dumps(e)
#     d = json.loads(d)
#     id_ = d["product/productId"]
#     print(id_)
