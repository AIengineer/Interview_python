import random
class MATRIX(object):
    def __init__(self, W, H):
        self.H = H
        self.R = []
        self.W = W
        self.C = []
        self.matrix = self.create_matrix(H, W)

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
    def conv2d(self, image, kernel, bi_as = 0, stride = 1, pad =1):
        n_H_old, n_W_old = len(image), len(image[0])
        f, f = len(kernel), len(kernel[0])
        image_padding = self.padding(image, padding_value=0)
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
                con2d_image[h][v] = self.sum_conv(matrix, kernel)
        return con2d_image
matrix = MATRIX(28,28)
kernel = MATRIX(3,3)
matrix.matrix = matrix.conv2d(matrix.matrix, kernel.matrix, pad=0)