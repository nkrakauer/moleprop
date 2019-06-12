from rdkit import Chem

def remove_invalid_smiles(data):
    invalid = []
    for index, row in data.iterrows():
        #print(row['smiles'])
        if Chem.MolFromSmiles(row['smiles']) == None:
            invalid.append(row['smiles'])
            #data.drop(row.index[0], inplace=True)
    print('invalid smiles strings')
    print('------------------------------')
    print(invalid)
    print('------------------------------')
    for smi in invalid:
        data.drop([data[data['smiles'] == smi].index[0]], inplace=True)
    return data

def remove_duplicates(data):
    result = data.drop_duplicates(subset='smiles', keep=False)#[~duplicates]
    #for each unique smiles that has duplicates
    for smiles in data[data.duplicated(subset='smiles')]['smiles'].unique():
        dup_rows = data.loc[data['smiles'] == smiles]
        if dup_rows['flashpoint'].unique().shape[0] == 1:
            # remove all but one
            result = result.append(dup_rows.iloc[0], sort=False)
        else:
            if dup_rows['flashpoint'].std() < 5:
                # add 1 back
                result = result.append(dup_rows.iloc[0], sort=False)
    return result  

def canonicalize_smiles(data):
    for idx, row in data.iterrows():
        m = Chem.MolFromSmiles(data.iloc[idx]['smiles'])
        if m != None:
            data.iloc[idx]['smiles'] = Chem.MolToSmiles(m)
        else:
            data.iloc[idx]['smiles'] = None
    return data

def kelvinToCelsius(temp):
    return temp -  273.15

def celsiusToKelvin(temp):
    return temp + 273.15

def get_compound_with_element(data, element):
    return data[data['smiles'].str.contains(element)]
