import os

files = os.popen("ls -1 *.ipynb").read().split('\n')[:-1]
print(files)
nb = os.popen(f"head -30 {files[0]}").read()
md_sections = [i.split("""\"source\": [\n""")[-1] for i in nb.split('markdown') if 'source' in i]

for i in md_sections:
    print(i)
