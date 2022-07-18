# open 'all.txt' and read each line as sgfFile
# if sgfFile doesn't have 'HA[', append to allValidFile
with open('jgdb/all.txt', 'r') as allFile:
    allLines = allFile.readlines()

allValidFile = []

count = 0

for line in allLines:
    sgfFile = line.strip()
    try:
        with open(sgfFile, 'r') as sgf:
            data = sgf.read()
        if 'HA[' not in data:
            allValidFile.append(sgfFile)
    except:
        print('Error: ' + sgfFile)

    count += 1
    if count % 10000 == 0:
        print('Processed ' + str(count) + ' files')

# write allValidFile to 'allValid.txt'
with open('jgdb/allValid.txt', 'w') as allValid:
    for sgfFile in allValidFile:
        allValid.write(sgfFile)
        allValid.write('\n')
