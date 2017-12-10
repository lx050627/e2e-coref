import os
import sys
import glob

INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]

# reference
fin = open(os.path.join(INPUT_DIR, 'ref/answer.txt'))
ref = [line.strip() for line in fin.readlines()]

fin = open(os.path.join(INPUT_DIR, 'res/answer.txt'))
res = [line.strip() for line in fin.readlines()]
fin.close()

acc = 0
for f, s in zip(ref, res):
    if f == s: acc += 1

score = 100.0 * acc / len(ref)
print('accuracy: %5.2f (%d/%d)' % (score, acc, len(ref)))
fout = open(os.path.join(OUTPUT_DIR, 'scores.txt'), 'w')
fout.write('accuracy:{0}\n'.format(score))
fout.close()