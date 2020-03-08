import tkinter as tk
import numpy as np
import os
import mlp

def load_mnist():  # train and test 
    data_dir = 'mnist_dataset/'

    train_num = 2000
    test_num = 1000

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.int)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.int)

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    perm = np.random.permutation(trY.shape[0])
    trX = trX[perm]
    trY = trY[perm]

    perm = np.random.permutation(teY.shape[0])
    teX = teX[perm]
    teY = teY[perm]

    return trX[:train_num], trY[:train_num], teX[:test_num], teY[:test_num]  # TrainX, TrainY, TestX, TestY



class GridWindow:
    def __init__(self, root, rows, columns):
        self.root = root
        self.root.title("28x28 grid paint program")
        #print(type(root))
        self.rows = rows
        self.columns = columns
        self.cellwidth = 18
        self.cellheight = 18
        self.cells = {}
        self.matrix = [[0 for j in range(self.columns)] for i in range(self.rows)] # matrix of image that will be drawed
        self.old_x, self.old_y = None, None

        self.myCanvas = tk.Canvas(self.root)
        self.myCanvas.configure(width = self.cellheight*rows + 2, height = self.cellwidth*columns + 2)
        self.myCanvas.grid(row = 0, column = 0, padx = 5, pady = 5)
        self.myCanvas.bind('<Button-1>', self.paint)
        self.myCanvas.bind('<B1-Motion>', self.paint)
        self.myCanvas.bind('<ButtonRelease-1>', self.reset)        
        
        self.rightPanel = tk.Frame(self.root)
        self.rightPanel.grid(row = 0, column = 1, sticky="n") # if i dont write sticky by default it will be centered

        self.recognizeBtn = tk.Button(self.rightPanel, text="RECOGNIZE", width=15, height=2, command=self.recognize_image)
        self.recognizeBtn.grid(row = 0, column = 1, sticky="nwe", padx = 5, pady = 5)
        
        self.clearBtn = tk.Button(self.rightPanel, text="Clear", width=15, height=2, command=self.clear_grid)
        self.clearBtn.grid(row = 1, column = 1, sticky="nwe", padx = 5, pady = 5)

        self.mlp_init()


    def draw_grid(self): # draws grid
        for row in range(self.rows):
            for column in range(self.columns):
                x1 = column * self.cellwidth + 4
                y1 = row * self.cellheight + 4
                x2 = x1 + self.cellwidth
                y2 = y1 + self.cellheight
                self.cells[row, column] = self.myCanvas.create_rectangle(x1, y1, x2, y2, fill = "white") # grid cells - rectangle objects
                self.matrix[row][column] = 0 
                

    def fill_grid_cell(self, x, y, col): # x, y - grid cell indexes
        if x >= self.columns or y >= self.rows or x < 0 or y < 0: # check if indexes out of bounds
            return
        self.matrix[y][x] = 1
        self.myCanvas.itemconfig(self.cells[y, x], fill = col) # y - row, x - column
        #p.s we dont create new rectangle, we only change colors of old ones
        # if we will always create new rectangles it will cause memory leak

    def clear_grid_cell(self, x, y, col):
        if x >= self.columns or y >= self.rows or x < 0 or y < 0: # check if indexes out of bounds
            return
        self.matrix[y][x] = 0
        self.myCanvas.itemconfig(self.cells[y, x], fill = col) # y - row, x - column
        #p.s we dont create new rectangle, we only change colors of old ones
        # if we will always create new rectangles it will cause memory leak


    def draw_line_on_grid(self, x1=0, y1=0, x2=0, y2=0): #Bresenham's line algorithm
        # x1, y1, x2, y2 - grid cell coordinates
        dx = x2 - x1
        dy = y2 - y1        
        sign_x = 1 if dx>0 else -1 if dx<0 else 0
        sign_y = 1 if dy>0 else -1 if dy<0 else 0
        if dx < 0: dx = -dx
        if dy < 0: dy = -dy
        if dx > dy:
            pdx, pdy = sign_x, 0
            es, el = dy, dx
        else:
            pdx, pdy = 0, sign_y
            es, el = dx, dy
        x, y = x1, y1
        error, t = el/2, 0        
        self.fill_grid_cell(x, y, "black")
        while t < el:
            error -= es
            if error < 0:
                error += el
                x += sign_x
                y += sign_y
            else:
                x += pdx
                y += pdy
            t += 1
            self.fill_grid_cell(x, y, "black")


    def paint(self, event):
        # we draw line between last and current mouse poisitions
        # because if we'd fill only cell on current mouse positions,
        # there will be breaks in line if you move mouse fast
        cur_cell_x = (event.x-4)//self.cellwidth
        cur_cell_y = (event.y-4)//self.cellheight
        self.fill_grid_cell(cur_cell_x, cur_cell_y, "black")
        if self.old_x and self.old_y:
            self.draw_line_on_grid(self.old_x, self.old_y, cur_cell_x, cur_cell_y)
        self.old_x = (event.x-4)//self.cellwidth
        self.old_y = (event.y-4)//self.cellheight


    def reset(self, event): # it is called on every mouse release (binded to b1 release event)
        # print matrix of image to console
        print('\n'.join([''.join([('.' if self.matrix[i][j] == 0 else '@') for j in range(self.columns)]) for i in range(self.rows)]), end='\n\n')
        self.old_x, self.old_y = None, None

    def clear_grid(self):
        #self.matrix = [[0 for j in range(self.columns)] for i in range(self.rows)]
        for row in range(self.rows):
            for column in range(self.columns):
                self.clear_grid_cell(row, column, "white")

    def mlp_init(self):
        trainX, trainY, testX, testY = load_mnist()
        print("Shapes: ", trainX.shape, trainY.shape, testX.shape, testY.shape)
        
        epochs = 25
        num_hidden_units = 300 
        minibatch_size = 100  
        regularization_rate = 0.01 
        learning_rate = 0.001 

        self.my_model = mlp.MLP(num_hidden_units, minibatch_size, regularization_rate, learning_rate)

        print("Starting training..........")
        self.my_model.train(trainX, trainY, epochs)
        print("Training complete")

        print("Starting testing..........")
        labels = self.my_model.test(testX)
        accuracy = np.mean((labels == testY)) * 100.0
        print("\nTest accuracy: %lf%%" % accuracy)

    
    def recognize_image(self):
        matr = np.asarray([self.matrix])
        print('DIGIT IS : ', self.my_model.test(matr))
        


def runApp(rows, columns):
    root = tk.Tk()
    myapp = GridWindow(root, rows, columns)
    myapp.draw_grid()
    root.mainloop()

if __name__ == '__main__':
    runApp(28, 28)
