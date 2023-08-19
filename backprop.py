import math
import random
import numpy as np
import time

random.seed(time.time())


# מחזיר מס רנדומלי בין a-b
def rand(a, b):
    return random.uniform(a, b)


# פונקציית סיגמונד
def sigmoid(x):
    for xi in np.nditer(x):
        v=math.tanh(xi)
        xi=v
    print(x)
    return (x)


def dsigmoid(y):
    return 1.0 - y ** 2


def makeMatrix(I, J, fill=0.0):
    m=[]
    for i in range(I):
        m.append([fill] * J)
    return m


class NeuralNetwork:
    def __init__(self, ni, nh, no):
        # מחלקה של קלט פלט ושכבות
        self.ni=ni + 1  # +1 לביאס
        self.nh=nh
        self.no=no

        self.ai=[1.0] * self.ni
        self.ah=[1.0] * self.nh
        self.ao=[1.0] * self.no

        # יצירת מטריצות משקלים
        self.wi=makeMatrix(self.ni, self.nh)
        self.wo=makeMatrix(self.nh, self.no)

        # הצבת משתנים רמדומלים
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j]=rand(-2.0, 2.0)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k]=rand(-2.0, 2.0)

        # עדכון ערכים
        self.ci=makeMatrix(self.ni, self.nh)
        self.co=makeMatrix(self.nh, self.no)

    def update(self, inputs):

        if len(inputs) != self.ni - 1:
            raise ValueError('שגיעה בקלטים')

            # קלטים
        for i in range(self.ni - 1):

            try:
                self.ai[i]=inputs[i]
            except IndexError:
                self.ai[i]=0
                # שכבות
        for j in range(self.nh):
            sum=0.0
            for i in range(self.ni):
                sum=sum + (self.ai[i] * self.wi[i][j])
            self.ah[j]=1 / (1 + np.exp(-sum))

        # פלטים
        for k in range(self.no):
            sum=0.0
            for j in range(self.nh):
                sum=sum + self.ah[j] * self.wo[j][k]
            self.ao[k]=1 / (1 + np.exp(-sum))

        return self.ao[:]

    ##########back propogate##########
    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('שגיעה')

        # חישוב תנאי שגיעה לפלט
        output_deltas=[0.0] * self.no
        for k in range(self.no):
            error=targets[k] - self.ao[k]
            output_deltas[k]=(dsigmoid(self.ao[k]) * error)

        # לשכבות
        hidden_deltas=[0.0] * self.nh
        for j in range(self.nh):
            error=0.0
            for k in range(self.no):
                try:
                    error=error + (output_deltas[k] * self.wo[j][k])
                except ValueError:
                    np.squeeze(error)
                    error=error + (output_deltas[k] * self.wo[j])
                    print("error ACURED")
            hidden_deltas[j]=dsigmoid(self.ah[j]) * error

        # עדכון משקלים
        for j in range(self.nh):
            for k in range(self.no):
                change=output_deltas[k] * self.ah[j]
                self.wo[j][k]=self.wo[j][k] + N * change + M * self.co[j][k]
                self.co[j][k]=change

        for i in range(self.ni):
            for j in range(self.nh):
                change=hidden_deltas[j] * self.ai[i]
                self.wi[i][j]=self.wi[i][j] + N * change + M * self.ci[i][j]
                self.ci[i][j]=change

        #################חישוב שגיעה######
        error=0.0
        for k in range(len(targets)):
            error=error + 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    ###########################
    def test(self, patterns):
        print('גודל קבוצה =', len(patterns[0][0]), '|  שכבות חסויות= ', self.nh, )
        print("0=אליפסה, 0.5=משולש ,1=עיגול")
        for p in patterns:
            predict=np.mean(self.update(p[0]))
            precentile=100 - round(abs(p[1][0] - predict) * 100, 2)
            print(p[1], 'תוצאה', predict, 'אחוזי הצלחה:', precentile, '%')

    #####################################
    """
    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])
"""

    ############################אימון#############
    def train(self, patterns, iterations=10000, N=0.5, M=0.1):

        old_error=0
        for i in range(iterations):
            error=0.0
            for p in patterns:
                inputs=p[0]
                targets=p[1]
                self.update(inputs)
                error=error + self.backPropagate(targets, N, M)
            if i % 1000 == 0:
                fix=round(np.mean(error), 6)
                print('אינטרקציות=', i, '| טעות=', fix, '| שיפור= ', round(old_error - fix, 5))
                old_error=fix


############################
def main():
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    from PIL import Image
    import numpy as np
    from os import listdir
    import os

    triangles=list()
    circles=list()
    ellipses=list()
    # המרת התמונות למטריצות, פונקצייה בנוייה . 3 לולאות לשלושת הצורות נכנסות לרשימות למעלה
    kernel=0.42
    curr_dir=os.getcwd()
    triangle_path=os.path.join(curr_dir, 'Triangles')
    for filename in listdir(triangle_path):
        img=Image.open(triangle_path + "\\\\" + filename)
        new_img=img.resize((28, 28))
        mat=np.array(new_img).flatten() / 255.0
        triangles.append(np.convolve(mat, kernel))

    ellipse_path=os.path.join(curr_dir, 'Ellipses')
    for filename in listdir(ellipse_path):
        img=Image.open(ellipse_path + "\\\\" + filename)
        new_img=img.resize((28, 28))
        mat=np.array(new_img).flatten() / 255.0
        ellipses.append(np.convolve(mat, kernel))

    circle_path=os.path.join(curr_dir, 'Circles')
    for filename in listdir(circle_path):
        img=Image.open(circle_path + "\\\\" + filename)
        new_img=img.resize((28, 28))
        mat=np.array(new_img).flatten() / 255.0
        circles.append(np.convolve(mat, kernel))

    # בחירת גודל הקבוצה :
    train_set=[
        [ellipses[0:10], [0]],
        [triangles[0:10], [0.5]],
        [circles[0:10], [1]]
    ]

    # train_set=[
    #     [ellipses[0:5], [0]],
    #     [triangles[0:5], [0.5]],
    #     [circles[0:5], [1]]
    # ]

    # train_set=[
    #     [ellipses[0:19], [0]],
    #     [triangles[0:19], [0.5]],
    #     [circles[0:19], [1]]
    # ]

    # בחירת כמות שכבות רצויות והרצה
    for times in range(1, 5):
        n=NeuralNetwork(len(train_set[0][0]), times, 1)
        # כמות אינטרקציות
        n.train(train_set, iterations=10000)

        print(" ------- בחינת  קבוצה מסודרת(10,000 אינטרקציות)------- ")

        print(" ---- תוצאות בבחינה מסודרת ----")
        n.test(train_set)
        print(" ---- תוצאות בבחינה אקראית ----")

        for i in train_set:
            random.shuffle(i[0])
        n.test(train_set)


if __name__ == '__main__':
    main()

