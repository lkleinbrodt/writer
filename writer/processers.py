import pandas as pd

class Frost:
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
            f.write(x)'