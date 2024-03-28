import argparse
import re
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--result", help="Results Path")

args = parser.parse_args()

# Read the results file
with open(args.result, "r") as file:
    results = file.read()

errors = re.findall(r"\*\*\*\*\*\*\*\*\*\*\*\*\* Module ([\s\S]*?)\n\n\-\-\-", results)

print("Errors In : ", len(errors)," Files")

data = []
for error in errors:
    error = error.strip()
    error = error.split("\n")
    module, number = re.findall(r"([\s\S]*?)_([\d]*?).py_temp", error[0])[0]
    for er in error[1:]:
        err_list = re.findall(r"Datasets/pairs_data/cleaned_data/code/"+module+r"/"+module+r"_"+str(number)+r".py_temp.py:([\d]*):([\d]*): ([\s\S]*): ([\s\S]*)", er)
        if err_list == []:
            continue
        line_err, ch_err, err, err_mes = err_list[0]
        data.append([module, number, line_err, ch_err, err, err_mes])

df = pd.DataFrame(data, columns=['API', 'FILE_#','LINE_ERR','CH_ERR',"ERR","ERR_MES"]) 
df.to_csv('Datasets/pairs_data/cleaned_data/test_res.csv', index=False)        
    

