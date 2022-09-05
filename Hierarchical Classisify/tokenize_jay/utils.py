


def read_stopword(file):
    with open(file, encoding="utf-8") as f:
        data = f.readlines()
        data = [i.strip() for i in data]
    return data