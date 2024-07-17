import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import AllChem
from tqdm import tqdm
import time

IPythonConsole.ipython_useSVG = True


def pep_lib_sep_generate(Candidate_aa):
    """
    Generate a peptide sequence representation library from a candidate amino acid list.
    :param Candidate_aa: candidate amino acid list
    :return: a list of peptide sequence string
    """
    pep_lib = []
    pep_start = ['P', 'p']
    # 考虑四肽序列为 起始为D/L P， 四肽序列为P/p + 三肽序列
    for k in pep_start:
        for i in Candidate_aa:
            for j in Candidate_aa:
                for l in Candidate_aa:
                    pep_lib.append(k + i + j + l)
    return pep_lib


def pep_seq_transform(pep_seq, trans_type='SMILES', Add_H_atom=False):
    """
    Transform sequence type of a peptide to SMILES string ,or 2d mol, or 3d mol.
    Please set Add_H_atom as True when draw a 3d_mol.
    :param pep_seq: the sequence of a peptide
    :param trans_type: could be one of SMILES, 2d_mol, 3d_mol
    :param Add_H_atom: Add H atoms to mol or not
    :return:
    """
    if trans_type == 'SMILES':
        out_put = Chem.MolToSmiles(Chem.MolFromFASTA(pep_seq, flavor=1))  # flavor=1 区分 D,L构型\
    if 'mol' in trans_type:
        pep_mol = Chem.rdmolfiles.MolFromFASTA(pep_seq, flavor=1)
        # pep_mol = Chem.MolFromFASTA(pep_seq, flavor=1)
        if Add_H_atom == True:
            pep_mol = Chem.AddHs(pep_mol)
        if trans_type == '2d_mol':
            out_put = pep_mol
        if trans_type == '3d_mol':
            AllChem.EmbedMolecule(pep_mol, randomSeed=10)
            out_put = pep_mol
    return out_put


# 几何算法+ETKDG+MMFF94构像优化 生成3d构像并保存
def get_structure(mol, n_confs):
    mol = Chem.AddHs(mol)
    new_mol = Chem.Mol(mol)

    AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, useExpTorsionAnglePrefs=True, useBasicKnowledge=True,
                               numThreads=0)
    energies = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000, nonBondedThresh=100.0, numThreads=0)

    energies_list = [e[1] for e in energies]
    min_e_index = energies_list.index(min(energies_list))

    new_mol.AddConformer(mol.GetConformer(min_e_index))

    return new_mol

def class_label(ee):
    low = []
    med = []
    high = []
    for i in range(len(ee)):
        if ee[i] >= 80:
            high.append(i)
            ee.iloc[i] = 'high'
        elif ee[i] <= 20:
            low.append(i)
            ee.iloc[i] = 'low'
        else:
            med.append(i)
            ee.iloc[i] = 'medium'
    return ee

def kennardstonealgorithm(x_variables, k):
    
    x_variables = np.array(x_variables)
    original_x = x_variables
    distance_to_average = ((x_variables - np.tile(x_variables.mean(axis=0), (x_variables.shape[0], 1))) ** 2).sum(axis=1)
    max_distance_sample_number = np.where(distance_to_average == np.max(distance_to_average))
    max_distance_sample_number = max_distance_sample_number[0][0]
    selected_sample_numbers = list()
    selected_sample_numbers.append(max_distance_sample_number)
    remaining_sample_numbers = np.arange(0, x_variables.shape[0], 1)
    x_variables = np.delete(x_variables, selected_sample_numbers, 0)
    remaining_sample_numbers = np.delete(remaining_sample_numbers, selected_sample_numbers, 0)
    for iteration in range(1, k):
        selected_samples = original_x[selected_sample_numbers, :]
        min_distance_to_selected_samples = list()
        for min_distance_calculation_number in range(0, x_variables.shape[0]):
            distance_to_selected_samples = ((selected_samples - np.tile(x_variables[min_distance_calculation_number, :],
                                                                        (selected_samples.shape[0], 1))) ** 2).sum(axis=1)
            min_distance_to_selected_samples.append(np.min(distance_to_selected_samples))
        max_distance_sample_number = np.where(
            min_distance_to_selected_samples == np.max(min_distance_to_selected_samples))
        max_distance_sample_number = max_distance_sample_number[0][0]
        selected_sample_numbers.append(remaining_sample_numbers[max_distance_sample_number])
        x_variables = np.delete(x_variables, max_distance_sample_number, 0)
        remaining_sample_numbers = np.delete(remaining_sample_numbers, max_distance_sample_number, 0)
    return selected_sample_numbers, remaining_sample_numbers

# predict_path = pkl_path
def get_df_results(predict_path): # input path of your .pkl file
    predict = pd.read_pickle(predict_path)
    # print(predict[0])
    smi_list, mol_repr_list, pair_repr_list = [], [], []
    for batch in predict:
        sz = batch["bsz"]
        for i in range(sz):
            smi_list.append(batch["smi_name"][i])
            mol_repr_list.append(batch["mol_repr_cls"][i])
            # pair_repr_list.append(batch["pair_repr"][i])
    predict_df = pd.DataFrame({"SMILES": smi_list, "mol_repr": mol_repr_list})
    return predict_df

if __name__ == "__main__":

    # 建立四肽序列库

    # 选取候选的氨基酸分子， 大写字母表示L构型，小写字母表示D构型
    """
    常用20种氨基酸：
    丙氨酸 Ala A      精氨酸 Arg R    天冬氨酸 Asp D
    半胱氨酸 Cys C   谷氨酰胺 Gln Q    谷氨酸 Glu/Gln E
    组氨酸 His H      甘氨酸 Gly G    天冬酰胺 Asn N
    酪氨酸 Tyr Y      脯氨酸 Pro P     丝氨酸 Ser S
    甲硫氨酸 Met M     赖氨酸 Lys K    缬氨酸 Val V
    异亮氨酸 Ile I    苯丙氨酸 Phe F    亮氨酸 Leu L
    色氨酸 Trp W      苏氨酸 Thr T
    """
    # 这次选取的候选氨基酸： 脯氨酸 Pro P 组氨酸 His H 天冬氨酸 Asp D 天冬酰胺 Asn N 亮氨酸 Leu L 谷氨酸 Glu/Gln E 谷氨酰胺 Gln Q 甘氨酸 Gly G 酪氨酸 Tyr Y
    Candidate_aa = ['P', 'p', 'G', 'H', 'h', 'D', 'd', 'N', 'n', 'L', 'l', 'E', 'e', 'Q', 'q', 'G', 'g', 'Y',
                    'y']  # 区分D,L构型共19种氨基酸

    pep_lib = pep_lib_sep_generate(Candidate_aa)  # lengh = 2*(19**3)=13718

    pep_2d_mols = []
    for i in pep_lib:
        pep_2d_mols.append(pep_seq_transform(i, trans_type='2d_mol'))


    for i in tqdm(range(len(pep_2d_mols))):
        time.sleep(0.1)
        mol_temp = get_structure(pep_2d_mols[i], 50)
        path = 'Data//mol_3d_opt/' + pep_lib[i] + '_3d.sdf'
        # print(i)
        Chem.MolToMolFile(mol_temp, path)


