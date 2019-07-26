# FILE_PATH = 'GSE25066-Normal-50-sgx.txt'
FILE_PATH = 'GSE25066-Tumor-50-sgx.txt'
with open(FILE_PATH) as f:
    content = f.read().strip().split('\n')[1:]

cnt = len(content[0].split('\t'))
res = ['['] * cnt
for i in range(len(content)):
    for j, num in enumerate(content[i].split('\t')):
        res[j] += num
        res[j] += ','

with open('CONVERTED_' + FILE_PATH, 'w') as f:
    for j in range(cnt):
        f.write(res[j])
        f.write(']\n')


