import argparse
import zipfile

parser = argparse.ArgumentParser()

parser.add_argument('--src_file', type=str, required=True, help='path to the file to be unzipped')
parser.add_argument('--dst_path', type=str, required=True, help='path to the folder where the unzipped files should be placed')

args = parser.parse_args()

print(args.src_file)
print(args.dst_path)
with zipfile.ZipFile(args.src_file) as zf:
    zf.extractall(args.dst_path)

