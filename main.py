
class Fraction:
    def __init__ (self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator
    def convertToInt(self):
        self.numerator = int(self.numerator)
        self.denominator = int(self.denominator)
    def simplify(self):
        done = False
        if self.numerator == 0:     #
            self.denominator = 1    # Avoiding edge cases
        elif self.numerator != 1:   #
            while not done:
                done = True
                higher = max(self.numerator, self.denominator)
                for div in range(int(((higher + (higher % 2)) / 2) + 1)):
                    if div != 0 and div != 1: # quick fix
                        if self.numerator % div == 0 and self.denominator % div == 0:
                            self.numerator /= div
                            self.denominator /= div
                            done = False
        if self.denominator < 0:
            self.numerator *= -1
            self.denominator *= -1
        self.convertToInt()
    def timesInt(self, num):
        if self.denominator % num == 0:
            self.denominator /= num
        else:
            self.numerator *= num
        self.simplify()
    def timesFraction(self, fraction):
        if self.denominator % fraction.numerator == 0:
            self.denominator /= fraction.numerator
        else:
            self.numerator *= fraction.numerator
            
        if self.numerator % fraction.denominator == 0:
            self.numerator /= fraction.denominator
        else:
            self.denominator *= fraction.denominator
        self.simplify()
    def inverse(self):
        return Fraction(self.denominator, self.numerator)
    def divInt(self, num):
        self.denominator *= num
        self.simplify()
    def divFraction(self, fraction):
        self.numerator *= fraction.denominator
        self.denominator *= fraction.numerator
        self.simplify()
    def set(self, newNumerator, newDenominator):
        self.numerator = int(newNumerator)
        self.denominator = int(newDenominator)
    def add(self, frac):
        selfNum = self.numerator
        selfDen = self.denominator
        fracNum = frac.numerator
        fracDen = frac.denominator
        selfNum *= fracDen
        fracNum *= selfDen
        fracDen *= selfDen
        selfDen = fracDen + 0
        self.numerator = selfNum + fracNum
        self.denominator = selfDen #work on simplification later
        self.simplify()
    @staticmethod
    def copy(self, frac):
        frac.numerator = self.numerator
        frac.denominator = self.denominator
    def __str__(self):
        return str(self.numerator) + "/" + str(self.denominator)

inputFile = open("input.txt", "r")
inputStr = inputFile.read()
inputLines = inputStr.split('\n')
inputFile.close()
strMatrix = []
for line in inputLines:
    strMatrix += [line.split(" ")]

print(strMatrix)

matrix = []
copyOfOriginalMatrix = []
matrixHasInverse = True

for i in range(len(strMatrix)):
    line = []
    line2 = [] # to prevent instance linking
    for j in range(len(strMatrix)):
        line += [Fraction(0, 1)]
        line2 += [Fraction(0, 1)]
    matrix += [line]
    copyOfOriginalMatrix += [line2]

def createIdentity(size):
    identityMatrix = []
    for i in range(size):
        idLine = []
        for j in range(size):
            if i == j:
                idLine += [Fraction(1, 1)]
            else:
                idLine += [Fraction(0, 1)]
        identityMatrix += [idLine]
    return identityMatrix

print(matrix)

for i in range(len(strMatrix)):
    for j in range(len(strMatrix[i])):
        numStr = strMatrix[i][j]
        if '/' in numStr:
            numStrSplit = numStr.split('/')
            numerator = int(numStrSplit[0])
            denominator = int(numStrSplit[1])
            matrix[i][j].numerator = numerator
            matrix[i][j].denominator = denominator
        else:
            matrix[i][j].numerator = int(numStr)
            matrix[i][j].denominator = 1
        Fraction.copy(matrix[i][j], copyOfOriginalMatrix[i][j])

def printMatrix(matrixArr, maxLen = -1, newLine = True):
    if maxLen == -1:
        for line in matrixArr:
            numLen = 0
            for num in line:
                if num.denominator == 1:
                    numLen = len(str(num.numerator))
                else:
                    numLen = len(str(num.numerator)) + 1 + len(str(num.denominator))
                if numLen > maxLen:
                    maxLen = numLen
    for line in matrixArr:
        outputLine = ""
        for num in line:
            numStr = ""
            if num.denominator == 1:
                numStr = str(num.numerator)
            else:
                numStr = str(num.numerator) + "/" + str(num.denominator)
            while len(numStr) < maxLen:
                numStr = " " + numStr
            outputLine += numStr + " "
        print(outputLine)
    if newLine:
        print("\n")

printMatrix(matrix)

def matrixToString(matrixArr, maxLen = -1):
    outputString = ""
    if maxLen == -1:
        for line in matrixArr:
            numLen = 0
            for num in line:
                if num.denominator == 1:
                    numLen = len(str(num.numerator))
                else:
                    numLen = len(str(num.numerator)) + 1 + len(str(num.denominator))
                if numLen > maxLen:
                    maxLen = numLen
    for line in matrixArr:
        outputLine = ""
        for num in line:
            numStr = ""
            if num.denominator == 1:
                numStr = str(num.numerator)
            else:
                numStr = str(num.numerator) + "/" + str(num.denominator)
            while len(numStr) < maxLen:
                numStr = " " + numStr
            outputLine += numStr + " "
        outputString += outputLine + "\n"
    return outputString

# inversion functions

def multiplyLine(inputMatrix, lineNum, k):
    if type(k) == int:
        for i in range(len(inputMatrix[lineNum])):
            inputMatrix[lineNum][i].timesInt(k)
    else:
        for i in range(len(inputMatrix[lineNum])):
            inputMatrix[lineNum][i].timesFraction(k)

def addLine(originalMatrix, originalLineNum, matrixToAdd, lineNumToAdd, k = 1):
    newLine = []
    if type(k) == int:
        for i in range(len(matrixToAdd[lineNumToAdd])):
            frac = Fraction(1, 1)
            Fraction.copy(matrixToAdd[lineNumToAdd][i], frac)
            frac.timesInt(k)
            newLine += [frac]
    else:
        for i in range(len(matrixToAdd[lineNumToAdd])):
            frac = Fraction(1, 1)
            Fraction.copy(matrixToAdd[lineNumToAdd][i], frac)
            frac.timesFraction(k)
            newLine += [frac]
    for i in range(len(originalMatrix[originalLineNum])):
        newFrac = Fraction(1, 1)
        newLine[i].add(originalMatrix[originalLineNum][i])
        Fraction.copy(newLine[i], newFrac)
        originalMatrix[originalLineNum][i].set(
            newFrac.numerator,
            newFrac.denominator
        )

def switchLines(originalMatrix, lineNum1, lineNum2):
    for i in range(len(originalMatrix[lineNum1])):
        tempNum = originalMatrix[lineNum2][i].numerator     #to avoid instance linking
        tempDen = originalMatrix[lineNum2][i].denominator   #
        originalMatrix[lineNum2][i].set(originalMatrix[lineNum1][i].numerator, originalMatrix[lineNum1][i].denominator)
        originalMatrix[lineNum1][i].set(tempNum, tempDen)

"""
frac = Fraction(2, 3)
multiplyLine(matrix, 1, frac)
printMatrix(matrix)

addLine(matrix, 2, identity, 1)
printMatrix(matrix)

switchLines(matrix, 1, 2)
printMatrix(matrix)
"""

def rowIsEmpty(originalMatrix, row):
    for col in range(len(originalMatrix[row])):
        if originalMatrix[row][col].numerator != 0:
            return False
    return True

def REFsolo(originalMatrix):
    global matrixHasInverse
    for row in range(len(originalMatrix)):
        for col in range(row + 1): # stop before col == row + 1
            num = originalMatrix[row][col].numerator
            den = originalMatrix[row][col].denominator

            print(num, den)

            if col < row: # left of diagonal, needs to be 0
                if num != 0:
                    k = Fraction(num * -1, den)
                    addLine(originalMatrix, row, originalMatrix, col, k) # assumes that above rows are in ref and thus takes the col# for row (which would yield a 1)
                    print("adding line", col, "to line", row, "k =", k)
            elif col == row:
                if num != 0:
                    if num != den:
                        k = Fraction(den, num) # k * row would yield a 1 in this diagonal element
                        multiplyLine(originalMatrix, row, k)
                        print("multiplying line", row, "by", k)
                else:
                    if rowIsEmpty(originalMatrix, row):
                        print("Matrix has no inverse.")
                        return None
                    else:
                        for rowTrying in range(len(originalMatrix)):
                            if originalMatrix[rowTrying][col].numerator != 0 and rowTrying != row:
                                addLine(originalMatrix, row, originalMatrix, rowTrying)
                                print("adding line", rowTrying, "to line", row, "k = 1")
                        if num != den:
                            if rowIsEmpty(originalMatrix, row):
                                print("Matrix has no inverse.")
                                matrixHasInverse = False
                                return None
                            else:
                                k = Fraction(den, num) # k * row would yield a 1 in this diagonal element
                                multiplyLine(originalMatrix, row, k)
                                print("multiplying line", row, "by", k)
            
            printMatrix(originalMatrix)

    return originalMatrix

def REFtoIdentity(originalMatrix):
    global matrixHasInverse
    identity = createIdentity(len(originalMatrix)) # all row operations will be done to the identity matrix as well.
    for row in range(len(originalMatrix)):
        for col in range(row + 1): # stop before col == row + 1
            num = originalMatrix[row][col].numerator
            den = originalMatrix[row][col].denominator

            print(num, den)

            if col < row: # left of diagonal, needs to be 0
                if num != 0:
                    k = Fraction(num * -1, den)
                    addLine(originalMatrix, row, originalMatrix, col, k) # assumes that above rows are in ref and thus takes the col# for row (which would yield a 1)
                    addLine(identity, row, identity, col, k)
                    print("adding line", col, "to line", row, "k =", k)
            elif col == row:
                if num != 0:
                    if num != den:
                        k = Fraction(den, num) # k * row would yield a 1 in this diagonal element
                        multiplyLine(originalMatrix, row, k)
                        multiplyLine(identity, row, k)
                        print("multiplying line", row, "by", k)
                else:
                    if rowIsEmpty(originalMatrix, row):
                        print("Matrix has no inverse.")
                        matrixHasInverse = False
                        return None
                    else:
                        for rowTrying in range(len(originalMatrix)):
                            if originalMatrix[rowTrying][col].numerator != 0 and rowTrying != row:
                                addLine(originalMatrix, row, originalMatrix, rowTrying)
                                addLine(identity, row, identity, rowTrying)
                                print("adding line", rowTrying, "to line", row, "k = 1")
                        if num != den:
                            k = Fraction(den, num) # k * row would yield a 1 in this diagonal element
                            multiplyLine(originalMatrix, row, k)
                            multiplyLine(identity, row, k)
                            print("multiplying line", row, "by", k)
            print("---------------------------------------------")
            printMatrix(originalMatrix)
            printMatrix(identity)
            print("---------------------------------------------")

    return identity


def RREFfromREFsolo(originalMatrix):
    for i in range(len(originalMatrix)):
        row = (len(originalMatrix) - i) - 1
        for j in range(i + 1):
            col = (len(originalMatrix[row]) - j) - 1
            if col > row:
                print(originalMatrix[row][col])
                num = originalMatrix[row][col].numerator
                den = originalMatrix[row][col].denominator
                k = Fraction(num, den)
                if num != 0:
                    k.timesInt(-1)
                    addLine(originalMatrix, row, originalMatrix, col, k)
                printMatrix(originalMatrix)
    return originalMatrix

def RREFtoPassedMatrix(originalMatrix, modifiedMatrix):
    for i in range(len(originalMatrix)):
        row = (len(originalMatrix) - i) - 1
        for j in range(i + 1):
            col = (len(originalMatrix[row]) - j) - 1
            if col > row:
                print(originalMatrix[row][col])
                num = originalMatrix[row][col].numerator
                den = originalMatrix[row][col].denominator
                k = Fraction(num, den)
                if num != 0:
                    k.timesInt(-1)
                    addLine(originalMatrix, row, originalMatrix, col, k)
                    addLine(modifiedMatrix, row, modifiedMatrix, col, k)
                print("---------------------------------------------")
                printMatrix(originalMatrix)
                printMatrix(modifiedMatrix)
                print("---------------------------------------------")
    return modifiedMatrix

print("\nREF:\n")
REFIdentity = REFtoIdentity(matrix)

outputStr = ""

if matrixHasInverse:

    print("Result from operations on identity:")
    printMatrix(REFIdentity)

    print("\nRREF:\n")
    RREFMatrix = RREFtoPassedMatrix(matrix, REFIdentity) # if a matrix has REF, then it has RREF

    print("Original matrix:")
    printMatrix(copyOfOriginalMatrix)
    print("Final inverse:")
    printMatrix(RREFMatrix)
    outputStr = matrixToString(RREFMatrix)
else:
    outputStr = "Matrix has no inverse."

outputFile = open("output.txt", "w")
outputFile.write(outputStr)
outputFile.close()