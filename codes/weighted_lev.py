import numpy as np
from numpy import insert
from weighted_levenshtein import lev


def codeMapping(task):
    sequence = ''
    for question in task:
        if question == 'DC1V':
            sequence += 'A'
        if question == 'DC2V':
            sequence += 'B'
        elif question == 'PTPCI':
            sequence += 'C'
        elif question == 'PT1V':
            sequence += 'D'
        elif question == 'APPCC':
            sequence += 'E'
        elif question == 'APPP':
            sequence += 'F'
        elif question == 'EPI':
            sequence += 'G'
        elif question == 'EPC':
            sequence += 'H'
        elif question == 'RtPICU':
            sequence += 'I'
        elif question == 'RtPPRD':
            sequence += 'J'
        elif question == 'RtPPP':
            sequence += 'K'
        elif question == 'RPPCC':
            sequence += 'L'
        elif question == 'RPPP':
            sequence += 'M'
        elif question == 'RtITQ':
            sequence += 'N'
        elif question == 'RtCTQ':
            sequence += 'O'
        elif question == 'RtTICU':
            sequence += 'P'
        elif question == 'RtT1V':
            sequence += 'Q'
        elif question == 'RtTPP':
            sequence += 'R'
        elif question == 'RTPCI':
            sequence += 'S'
        elif question == 'RTMCPICU':
            sequence += 'T'
    return sequence


array1 = ['DC1V', 'PTPCI', 'APPCC', 'EPI', 'RtPICU', 'RPPCC', 'RtITQ', 'RtTICU', 'RTPCI']  # 'ACEGILNPS'
array2 = ['DC1V', 'PTPCI', 'APPCC', 'EPC', 'RtPPRD', 'RPPCC', 'RtCTQ', 'RtT1V']  #---------- 'ACEHJLOQ'
array3 = ['DC2V', 'PT1V', 'APPP', 'EPC', 'RtPPP', 'RPPP', 'RtCTQ', 'RtTPP', 'RTMCPICU']
substitute_costs = np.ones((128, 128), dtype=np.float64)  # make a 2D array of 1's
# substitute_costs = np.chararray((128, 3))
# substitute_costs[ord('G'), ord('H')] = 1.01  # make substituting 'A' for 'B' cost 1.25 as well
# substitute_costs[ord('I'), ord('J')] = 1.1
# substitute_costs[ord('N'), ord('O')] = 1.2
# substitute_costs[ord('P'), ord('Q')] = 1.3
# substitute_costs[ord('A'), ord('B')] = 1.4
# substitute_costs[ord('C'), ord('D')] = 1.5
# substitute_costs[ord('E'), ord('F')] = 1.6
# substitute_costs[ord('I'), ord('K')] = 1.7
# substitute_costs[ord('P'), ord('R')] = 1.8
# substitute_costs[ord('S'), ord('T')] = 1.99

insert_costs = np.ones(128, dtype=np.float64)
# insert_costs[ord('S')] = 1.9
# insert_costs[ord('D')] = 1.8

delete_costs = np.ones(128, dtype=np.float64)
# delete_costs[ord('S')] = 1.7
# delete_costs[ord('D')] = 1.6
# delete_costs[ord('g')] = 1.9
# substitute_costs[ord('a'), ord('b')] = 1.9
# substitute_costs[ord('H'), ord('G')] = 1.005
# substitute_costs[ord('G'), ord('H')] = 1.05
# insert_costs[ord('h')] = 1.8


# substitute_costs[ord('a'), ord('a')] = 1
# substitute_costs[ord('b'), ord('b')] = 1
# substitute_costs[ord('c'), ord('c')] = 1
# substitute_costs[ord('d'), ord('d')] = 1
# substitute_costs[ord('e'), ord('e')] = 1
# substitute_costs[ord('f'), ord('f')] = 1
# substitute_costs[ord('a'), ord('c')] = 1.25
# substitute_costs[ord('c'), ord('a')] = 0.25
# substitute_costs[ord('b'), ord('e')] = 0.25
# substitute_costs[ord('e'), ord('b')] = 1.25


# print(lev('abcd', 'aecf', substitute_costs=substitute_costs, insert_costs=insert_costs, delete_costs=delete_costs))  # prints '1.25'
# print(lev('aecf', 'abcd', substitute_costs=substitute_costs, insert_costs=insert_costs, delete_costs=delete_costs))  # prints '1.25'


# print(lev('HANANA', 'BANANA', substitute_costs=substitute_costs))  # prints '1.25'


# print(lev('AAAAAH', 'AAAAAG',  substitute_costs=substitute_costs, insert_costs=insert_costs, delete_costs=delete_costs))
# print(lev('BANANASDH', 'BANDANASG',  substitute_costs=substitute_costs, insert_costs=insert_costs, delete_costs=delete_costs))
# print(lev('BANANASDH', 'BANDANASG',  substitute_costs=substitute_costs, insert_costs=insert_costs, delete_costs=delete_costs))
# print(lev('BANANASDH', 'BANDANASG',  insert_costs=insert_costs, delete_costs=delete_costs))
#
# print(lev('abcdefg', 'bbcdefhh',  substitute_costs=substitute_costs, delete_costs=delete_costs, insert_costs=insert_costs))

# string1 = ['DCPG&A', 'PTPCI', 'APPP', 'EPI', 'RtPPRD', 'RP1V', 'NR7', 'NR8', 'RTPCI', '']
# strin2 = ['DCPG&A', 'PTICU', 'APPP', 'NR4', 'RtPPRD', 'RPPP', 'RtCTQ', 'RtTPP', 'RTPP', '']
# {'0': '', '1': 'APPP', '2': 'DCPG&A', '3': 'EPI', '4': 'NR4', '5': 'NR7', '6': 'NR8', '7': 'PTICU', '8': 'PTPCI',
# '9': 'RP1V', 'a': 'RPPP', 'b': 'RTPCI', 'c': 'RTPP', 'd': 'RtCTQ', 'e': 'RtPPRD', 'f': 'RtTPP'}
# ['2813e956b0', '2714eadfc0']


insert_costs[ord('0')] = 0  # ''
insert_costs[ord('1')] = 1.2  # APPP
insert_costs[ord('2')] = 1.5  # DCPG&A
insert_costs[ord('3')] = 1.0  # EPI
insert_costs[ord('4')] = 2.0  # NR4
insert_costs[ord('5')] = 1  # NR7
insert_costs[ord('6')] = 1  # NR8
insert_costs[ord('7')] = 1.5  # PTICU
insert_costs[ord('8')] = 1.1  # PTPCI
insert_costs[ord('9')] = 1  # RP1V
insert_costs[ord('a')] = 1  # RPPP
insert_costs[ord('b')] = 1  # RTPCI
insert_costs[ord('c')] = 1  # RTPP
insert_costs[ord('d')] = 1  # RtCTQ
insert_costs[ord('e')] = 1.3  # RtPPRD
insert_costs[ord('f')] = 1  # RtTPP

delete_costs[ord('0')] = 1  # ''
delete_costs[ord('1')] = 1.1  # APPP
delete_costs[ord('2')] = 1.3  # DCPG&A
delete_costs[ord('3')] = 2.0  # EPI
delete_costs[ord('4')] = 1.5  # NR4
delete_costs[ord('5')] = 1  # NR7
delete_costs[ord('6')] = 1  # NR8
delete_costs[ord('7')] = 1.6  # PTICU
delete_costs[ord('8')] = 1.2  # PTPCI
delete_costs[ord('9')] = 1  # RP1V
delete_costs[ord('a')] = 1  # RPPP
delete_costs[ord('b')] = 1  # RTPCI
delete_costs[ord('c')] = 1  # RTPP
delete_costs[ord('d')] = 1  # RtCTQ
delete_costs[ord('e')] = 1.5  # RtPPRD
delete_costs[ord('f')] = 1  # RtTPP


substitute_costs[ord('8'), ord('7')] = 0.4  # 'PTPCI', 'PTICU'
substitute_costs[ord('7'), ord('8')] = 0.4  # 'PTICU', 'PTPCI'
substitute_costs[ord('3'), ord('4')] = 2    # 'EPI', 'NR4'
substitute_costs[ord('4'), ord('3')] = 2    # 'NR4', 'EPI'
substitute_costs[ord('9'), ord('a')] = 1    # 'RP1V', 'RPPP'
substitute_costs[ord('a'), ord('9')] = 1    # 'RPPP', 'RP1V'
substitute_costs[ord('5'), ord('d')] = 1    # 'NR7', 'RtCTQ'
substitute_costs[ord('d'), ord('5')] = 1    # 'RtCTQ', 'NR7'
substitute_costs[ord('6'), ord('f')] = 1    # 'NR8', 'RtTPP'
substitute_costs[ord('f'), ord('6')] = 1    # 'RtTPP', 'NR8'
substitute_costs[ord('b'), ord('c')] = 1    # 'RTPCI', 'RTPP'
substitute_costs[ord('c'), ord('b')] = 1    # 'RTPP', 'RTPCI'

print(lev('2813e956b0', '2714eadfc0',  substitute_costs=substitute_costs, delete_costs=delete_costs, insert_costs=insert_costs))
print(lev('2714eadfc0', '2813e956b0',  substitute_costs=substitute_costs, delete_costs=delete_costs, insert_costs=insert_costs))


# http://www.let.rug.nl/~kleiweg/lev/
# https://awsm-tools.com/text/levenshtein-distance

#BANANASDH -> BANDNASAH -> BANDANSAH
         # BANDANASG
# print('cost of converting array1 to array2: ', lev(codeMapping(array1), codeMapping(array2),
#                                                                 substitute_costs=substitute_costs,
#                                                                 insert_costs=insert_costs, delete_costs=delete_costs))
#
# print('cost of converting array2 to array1: ', lev(codeMapping(array2), codeMapping(array1),
#                                                                 substitute_costs=substitute_costs,
#                                                                 insert_costs=insert_costs, delete_costs=delete_costs))
#
# print('deletion/insertion cost of converting array1 to array2: ', lev(codeMapping(array1), codeMapping(array2),
#                                                                       insert_costs=insert_costs,
#                                                                       delete_costs=delete_costs))

# print('ord1 to ord3: ', lev(codeMapping(ord_1), codeMapping(ord_3), substitute_costs=substitute_costs))
# print('ord2 to ord3: ', lev(codeMapping(ord_2), codeMapping(ord_3), substitute_costs=substitute_costs))
# print('test: ', lev("123", "456", substitute_costs=substitute_costs))
# print(ord("1"))
# print(ord("a"))
# print(ord("b"))
# print(ord("B"))
# print(ord("%"))
