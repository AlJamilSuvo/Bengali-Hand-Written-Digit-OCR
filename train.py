
import sys
import zipfile
import pandas as pd
from Lib.ocr import OCR_NN
import os
import numpy as np


class NN:

    def __init__(self):
        self.ocr_train_root_dir = os.getcwd()
        self.orc_valid_root_dir=os.getcwd()
        ocrdf = pd.read_csv('annotation.txt')
        self.annotation = ocrdf.as_matrix()
        np.random.shuffle(self.annotation)
        length=self.annotation.shape[0]
        self.ocr_train_annotation=self.annotation[:int(.8*length)]
        self.ocr_valid_annotation=self.annotation[int(.8*length):]


        self.ocr = OCR_NN(32, self.ocr_train_annotation, self.ocr_valid_annotation,
                          self.ocr_train_root_dir, self.ocr_train_root_dir, False)



    def trainOnlyOCR(self):
        self.ocr.build_model()
        validation_loss = self.ocr.train_model()
        return validation_loss

    def test(self, data_path):


        ocrdf = pd.read_csv(os.path.join(data_path, 'number_label.txt'))
        test_annotation = ocrdf.as_matrix()
        test_ds = self.ocr.getTestDataSet(test_annotation, data_path)
        test_loss=self.ocr.test_model(test_ds)
        print('OCR Test Loss:',test_loss)

        return test_loss


def trainFullDataSet():


    nn = NN()
    nn.trainOnlyOCR()


if __name__ == '__main__':
    trainFullDataSet()
