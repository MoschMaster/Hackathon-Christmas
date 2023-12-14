import sys;

class adreslist:
    def readinput():
        f = open("Documents/GitHub/Hackathon-Christmas/Datasets/1_Geldermalsen.txt", 'r+')
        listinput = []
        for line in f.readlines():
            listinput.append(line)
        f.close()
        print(listinput)
    readinput()
    




#class elf:

#class route: