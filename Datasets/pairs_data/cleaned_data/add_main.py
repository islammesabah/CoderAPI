import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", help="File Path")

args = parser.parse_args()

# Read the code file
with open(args.file, "r") as file:
    lines = file.readlines()

code = "if __name__ == '__main__':\n"
for line in lines:
    if line.strip() in ["if __name__ == '__main__':","if __name__ == \"__main__\":"]:
        code = lines
        break
    else:
        code += '    '+line

# Write the updated content back to the file
with open(args.file+"_temp.py", 'w') as file:
    file.writelines(code)
