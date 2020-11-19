import uuid

def writeBuffer(fileName, buff):
    with open(fileName, 'w') as bf:
        for line in buff:
            bf.write(line)

buff = []
with open('./lichess_db_standard_rated_2020-03.pgn') as f:
    for line in f:
        buff.append(line) 
        if (line[0] == '1'):
            fileName = './games/{}.pgn'.format(uuid.uuid4())
            print(fileName)
            writeBuffer(fileName, buff)
            buff = []


