import copy
import random

class Matrix(object):
    """ Matrix class with various methods useful for studying linear algebra
    Methods include:
     | method      | parameters           | returns | note                 |
     |-------------|----------------------|---------|----------------------|
     | printMatrix |                      |         | pretty print         |
     | printRow    |                      |         | pretty print         |
     | printCol    |                      |         | pretty print         |
     | getRow      | row(int)             | array   | 0-indexed            |
     | getCol      | col(int)             | array   | 0-indexed            |
     | getVal      | col, row(int, int)   | float   | 0-indexed            |
     | deleteCol   | col(int)             |         | 0-indexed            |
     | deleteRow   | row(int)             |         | 0-indexed            |
     | linComb     | vector  (array)      | array   | Linear combination   |
     | swapCol     | colA, colB(int, int) |         | 0-indexed            |
     | swapRow     | rowA, rowB(int, int) |         | 0-indexed            |
     | mMultedBy   | Matrix               | Matrix  | matrix multiplied by |
     | transpose   |                      | Matrix  |                      |
     | ref         |                      | Matrix  | row echelon form     |
     | rref        |                      | Matrix  | reduced row echelon  |
     | nulspace    |                      | Matrix  | null space basis     |
     | colspace    |                      | Matrix  | column space basis   |
     | rowspace    |                      | Matrix  | row space basis      |
     | leftnul     |                      | Matrix  | leftnull space basis |
     | getRank     | Matrix               | int     | rank of matrix       |
        """

    def __init__(self, W = None, H = None, matrix=[]):
        self.H = H
        self.R = []
        self.W = W
        self.C = []
        if matrix != []:
            self.matrix = matrix
        else:
            self.matrix = self.create_matrix(W, H)
        self.numRows = len(self.matrix)
        self.numVars = len(self.matrix[0])
        self.maxLengthString = self.calcMaxDig()

    def dotProduct(self, v1, v2):
        assert (len(v1) == len(v2)), "vectors must be same length"
        output = 0
        for i in range(len(v1)):
            output += v1[i]*v2[i]
        return(output)

    def calcMaxDig(self):
        curMax = 0
        for i in range(self.numRows):
            self.matrix[i] = [float(x) for x in self.matrix[i]]
            slrow = [len(str(round(x, 3))) for x in self.matrix[i]]
            rowMax = max(slrow)
            curMax = (curMax if curMax > rowMax else rowMax)
        return(curMax)

    def printMatrix(self):
        self.maxLengthString = self.calcMaxDig()
        print("")
        for i in range(self.numRows):
            printrow = "|"
            for j in range(self.numVars):
                printrow += " {:^{width}} ".format(round(self.matrix[i][j], 3),
                                                   width=self.maxLengthString)
            printrow += "|"
            print(printrow)
        print("")

    def printRow(self, rowNum):
        print("")
        printRow = "|"
        for j in range(self.numVars):
            printRow += " {:^{width}} ".format(self.matrix[rowNum][j],
                                               width=self.maxLengthString)
        printRow += "|"
        print(printRow)

    def printCol(self, colNum):
        for i in range(self.numVars):
            print("| {:^{width}} |".format(self.matrix[i][colNum],
                                           width=self.maxLengthString))

    def getColumn(self, colNum):
        column = []
        for i in range(self.numRows):
            column.append(self.matrix[i][colNum])
        return(column)

    def deleteCol(self, colNum):
        assert colNum < self.numVars, "out of range"
        assert colNum >= 0, "negative column number"
        for i in self.matrix:
            i.pop(colNum)
        self.numVars = len(self.matrix[0])

    def getRow(self, rowNum):
        return(self.matrix[rowNum])

    def getVal(self, col, row):
        return(self.matrix[row][col])

    def deleteRow(self, rowNum):
        assert rowNum < self.numRows, "out of range"
        assert rowNum >= 0, "negative row number"
        self.matrix.pop(rowNum)
        self.numRows = len(self.matrix)

    def linComb(self, vector):
        assert len(vector) == self.numVars, "dimensions mis-match"
        output = []
        for row in self.matrix:
            outRow = 0
            for index, value in enumerate(vector):
                outRow += value * row[index]
            output.append(outRow)
        return(output)

    def matMultedBy(self, matrixB):
        assert self.numVars == matrixB.numRows, "dimensions mis-match"
        prodNumRows = self.numRows
        prodNumVars = matrixB.numVars
        output = []
        for i in range(prodNumRows):
            row = []
            for j in range(prodNumVars):
                row.append(self.dotProduct(self.getRow(i),
                                           matrixB.getColumn(j)))
            output.append(row)
        return(Matrix(output))

    def swapCols(self, a, b):
        for row in self.matrix:
            row[a], row[b] = row[b], row[a]

    def swapRows(self, a, b):
        self.matrix[a], self.matrix[b] = self.matrix[b], self.matrix[a]

    def transpose(self):
        output = []
        for n in range(self.numVars):
            col = self.getColumn(n)
            output.append(col)
        return(Matrix(output))

    def findPivots(self):
        pivots = []
        for index, row in enumerate(self.matrix):
            cy = index
            cx = 0
            while(self.matrix[cy][cx] == 0):
                cx += 1
                if(cx >= self.numVars):
                    break
            if(cx >= self.numVars):
                break
            pivots.append([cx, cy])
        return(pivots)

    def ref(self):
        cx = 0
        cy = 0
        output = copy.deepcopy(self)
        for n in range(output.numRows - 1):
            cy = n
            while(output.matrix[cy][cx] == 0):
                cy += 1
                if(cy >= output.numRows):
                    cy = n
                    cx += 1
                    if(cx >= output.numVars):
                        return(output)
                if(output.matrix[cy][cx] != 0):
                    output.swapRows(n, cy)
                    cy = n
            pivot = output.matrix[cy][cx]
            for step in range(cy + 1, output.numRows):
                subpivot = output.matrix[step][cx]
                mult = -(subpivot/pivot)
                for i in range(cx, output.numVars):
                    output.matrix[step][i] += (mult * output.matrix[cy][i])
            cx += 1
            if(cx >= output.numVars):
                return(output)
        return(output)

    def getRank(self):
        self.numRows = len(self.matrix)
        self.numVars = len(self.matrix[0])
        self.maxLengthString = self.calcMaxDig()
        ref = self.ref()
        return(len(ref.findPivots()))

    def refp1s(self):
        output = self.ref()
        cx = 0
        cy = 0
        for index, row in enumerate(output.matrix):
            cy = index
            while(output.matrix[cy][cx] == 0.0):
                cx += 1
                if(cx >= output.numVars):
                    return(output)
            pivot = output.matrix[cy][cx]
            newrow = [i/pivot for i in row]
            output.matrix[index] = newrow
        return(output)

    def rref(self):
        output = self.refp1s()
        for r in range(output.numRows - 1, 0, -1):
            cx = 0
            cy = r
            emptyRow = False
            while(output.matrix[cy][cx] == 0.0):
                cx += 1
                if(cx >= output.numVars):
                    emptyRow = True
                    break
            if(not emptyRow):
                for s in range(cy - 1, -1, -1):
                    mult = -(output.matrix[s][cx])
                    newrow = [x + (mult * output.matrix[cy][ind])
                              for ind, x in
                              enumerate(output.matrix[s])]
                    output.matrix[s] = newrow
        return(output)

    def nulspace(self):
        rref = self.rref()
        pivots = rref.findPivots()
        numPivots = len(pivots)
        numFree = rref.numVars - numPivots
        pivCols = [i[0] for i in pivots]
        sequence = []
        freeColVecs = []
        for i in range(rref.numVars):
            if i not in pivCols:
                freeColVecs.append(rref.getColumn(i))
        for i in range(rref.numVars):
            sequence.append('P' if i in pivCols else 0)
        output = []
        for i in range(numFree):
            temp = copy.deepcopy(sequence)
            counter = -1
            for j in range(len(temp)):
                if temp[j] == 0:
                    counter += 1
                    if counter == i:
                        temp[j] = 1
                if temp[j] == 'P':
                    temp[j] = -(freeColVecs[i].pop(0))
            output.append(temp)
        return(Matrix(output).transpose())

    def colspace(self):
        ref = self.ref()
        pivotCols = [n[0] for n in ref.findPivots()]
        output = copy.deepcopy(self)
        for i in range(self.numVars - 1, -1, -1):
            if i not in pivotCols:
                output.deleteCol(i)
        return(output)

    def rowspace(self):
        return(self.transpose().colspace())

    def leftnul(self):
        return(self.transpose().nulspace())
    def create_matrix(self, H, W):
        matrix = []
        for col in range(H):
            value_col = []
            for row in range(W):
                value_col.append(random.randint(1, 10))
            matrix.append(value_col)
        return matrix
    def get_padding(self, padding_value=0, len_pading=1):
        item = []
        for i in range(len_pading):
            item.append(padding_value)
        return item
    def padding(self, image, padding_value=0):
        new_image = []
        for row in range(len(image)):
            new_image.append(self.get_padding(0, 1)+image[row]+ self.get_padding(0, 1))
        image_padding = [self.get_padding(padding_value, len(image[0])+2)]+ new_image+ [self.get_padding(padding_value, len(image[0])+2)]
        return image_padding
    def sum_conv(self, matrix, kernel):
        sum_item = 0
        for col in range(len(matrix)):
            for row in range(len(matrix[0])):
                    sum_item+=matrix[col][row]*kernel[col][row]
        return sum_item
    def get_matrix(self, image_padding, h_start, h_end, v_start, v_end):
        matrix = []
        for col in range(h_start, h_end):
            temp = []
            for row in range(v_start, v_end):
                temp.append(image_padding[col][row])
            matrix.append(temp)
        return matrix
    def conv2d(self, image, kernel, bi_as = 0, stride = 1, pad =None):
        n_H_old, n_W_old = len(image), len(image[0])
        f, f = len(kernel), len(kernel[0])
        if pad is None:
            image_padding = image
        else:
            image_padding = self.padding(image, padding_value=pad)
        n_H_new = int((n_H_old - f + 2*pad)/stride) + 1
        n_W_new = int((n_W_old - f + 2*pad)/stride) + 1
        con2d_image = self.create_matrix(n_H_new, n_W_new)
        for h in range(n_H_new):
            for v in range(n_W_new):
                h_start = h*stride
                h_end = h_start + f
                v_start = v*stride
                v_end = v_start + f
                matrix = self.get_matrix( image_padding, h_start, h_end, v_start, v_end)
                con2d_image[h][v] = int(self.sum_conv(matrix, kernel))
        return con2d_image

    def transposeMatrix(self, m):
        return list(map(list, zip(*m)))

    def getMatrixMinor(self, m, i, j):
        return [row[:j] + row[j + 1:] for row in (m[:i] + m[i + 1:])]

    def getMatrixDeternminant(self, m):
        if len(m) == 2:
            return m[0][0] * m[1][1] - m[0][1] * m[1][0]
        determinant = 0
        for c in range(len(m)):
            determinant += ((-1) ** c) * m[0][c] * self.getMatrixDeternminant(self.getMatrixMinor(m, 0, c))
        return determinant

    def getMatrixInverse(self, m):
        determinant = self.getMatrixDeternminant(m)
        if len(m) == 2:
            return [[m[1][1] / determinant, -1 * m[0][1] / determinant],
                    [-1 * m[1][0] / determinant, m[0][0] / determinant]]
        cofactors = []
        for r in range(len(m)):
            cofactorRow = []
            for c in range(len(m)):
                minor = self.getMatrixMinor(m, r, c)
                cofactorRow.append(((-1) ** (r + c)) * self.getMatrixDeternminant(minor))
            cofactors.append(cofactorRow)
        cofactors = self.transposeMatrix(cofactors)
        for r in range(len(cofactors)):
            for c in range(len(cofactors)):
                cofactors[r][c] = cofactors[r][c] / determinant
        return cofactors

    def getMatrixMinor(self, m, i, j):
        return [row[:j] + row[j + 1:] for row in (m[:i] + m[i + 1:])]

    def getMatrixDeternminant(self, m):
        if len(m) == 2:
            return m[0][0] * m[1][1] - m[0][1] * m[1][0]
        determinant = 0
        for c in range(len(m)):
            determinant += ((-1) ** c) * m[0][c] * self.getMatrixDeternminant(self.getMatrixMinor(m, 0, c))
        return determinant
if __name__ == '__main__':
    matrix = Matrix(H = 28, W = 28)
    kernel = Matrix( 3, 3)
    rank_stop = 4
    rank = Matrix(matrix = [[171, 266, 281], [217, 417, 401], [206, 337, 392]])
    while True:
        matrix.matrix = matrix.conv2d(matrix.matrix, kernel.matrix, pad=0)
        if len(matrix.matrix)>=2:
            if matrix.getRank() <= rank_stop:
                matrix.printMatrix()
                print(matrix.getRank())
                break
        else:
            break