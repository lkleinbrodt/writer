import pandas as pd

class Frost:
    """https://www.kaggle.com/archanghosh/robert-frost-collection"""
    def __init__(self):
        self.in_path = '../data/robert_frost_collection.csv'
        self.out_path = '../data/clean_frost.txt'
    
    def process(self):

        df = pd.read_csv(self.in_path)

        x = ''

        for name, row in df.set_index('Name ').iterrows():
            content = row['Content']

            if not pd.isna(name):
                x += name
                x += ': \n\n'
            if not pd.isna(content):
                x += content
            

        with open(self.out_path, 'w') as f:
            f.write(x)

class Tolkien:
    def __init__(self, ):
        """https://github.com/MokoSan/FSharpAdvent/tree/master/Data"""
        pass
    def process(self):
        with open('../data/HobbitBook.txt', 'r') as f:
            hobbit = f.read()

        with open('../data/LotrBook.txt', 'r') as f:
            lotr = f.read()

        tolkien = hobbit + '\n\n' + lotr

        with open('../data/clean_tolkien.txt', 'w') as f:
            f.write(tolkien)

        