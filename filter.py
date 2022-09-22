# open 'all.txt' and read each line as sgfFile
# if sgfFile doesn't have 'HA[', append to allValidFile

import os


# find all sgf files in games/
def findSgfFiles(path):
    sgfFiles = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.sgf'):
                sgfFiles.append(os.path.join(root, file))
    return sgfFiles


allValidFile = []

count = 0

allSgfFiles = findSgfFiles('games')
for sgfFile in allSgfFiles:
    try:
        with open(sgfFile, 'r') as sgf:
            data = sgf.read()
        # DT[2017-08-24] date > 2000
        if 'HA[' not in data and 'DT[20' in data:
            allValidFile.append(sgfFile)
    except:
        print('Error: ' + sgfFile)

    count += 1
    if count % 10000 == 0:
        print('Processed ' + str(count) + ' files')

# write allValidFile to 'allValid.txt'
with open('games/allValid.txt', 'w') as allValid:
    for sgfFile in allValidFile:
        allValid.write(sgfFile)
        allValid.write('\n')

print('Total:', len(allValidFile))
