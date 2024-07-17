import pandas as pd
from unimol_tools import UniMolRepr

class ProcessData:
    def __init__(self, df):
        self.df = df
        pass

    # # check none in the input dataframe
    # try :
    #     if self.df.isnull().values.any():
    #         print('There are some None values in the input dataframe')
    #     else:
    #         print('There are no None values in the input dataframe')
    # except:
    #     print('There are no None values in the input dataframe')

    def get_repr(self):
        """   Get the molecular representation of the input dataframe.
        Prameters:
        -----------
            param df: input dataframe, smiles column names should include 'smi'
            Returns: the dataframe with the molecular representation
        """
        self.df_repr = pd.DataFrame()
        colunms = self.df.columns
        label_col = []
        clf = UniMolRepr(data_type='molecule')
        for i in colunms:
            if 'smi' in i:
                smiles = self.df[i].values.tolist()
                reprs = clf.get_repr(smiles)
                col_name = i + '_repr'
                self.df_repr[col_name] = reprs.tolist()
            else:
                label_col.append(i)
        if len(label_col) ==  0:
            print('No label column in the input dataframe')
        elif len(label_col) == len(self.df.columns):
            print('No smiles column in the input dataframe')
        else:
            print('The label columns are: ', label_col)
            for i in label_col:
                self.df_repr[i] = self.df[i]

        return self.df_repr
    
    def save_csv(self, path):
        self.df_repr.to_csv(path, index=False)


if __name__ == '__main__':
    df = pd.read_csv('test.csv')
    process = ProcessData(df)
    df_repr = process.get_repr()
    process.save_csv('test_repr.csv')