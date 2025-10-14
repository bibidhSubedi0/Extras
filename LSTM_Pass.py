# Inspired from StatQuest's Video on LSTM

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

def Sigmoid(x):
    return 1/(1+np.exp(-x))

def Tanh(x):
    return np.tanh(x)


class Blocks(ABC):
    def __init__(self, InputWeight=0,STMWeight=0, Bias = 0):
        self.InputWeight = InputWeight
        self.STMWeight = STMWeight
        self.Bias = Bias
        

    @abstractmethod
    def Activation(self,p):
        pass

    
    def Process(self):
        return self.Activation(self.Input * self.InputWeight + self.STM * self.STMWeight + self.Bias)
    
class SBlock(Blocks):
    def Activation(self, p):
        return Sigmoid(p)
    
class TBlock(Blocks):
    def Activation(self, p):
        return Tanh(p)
    


class LSTMUnit:
    def __init__(self,input, ltm, stm):
        self.input = input
        self.LTM = ltm
        self.STM = stm
        self.Percentage_LT_To_Rem = SBlock(1.63,2.70,1.62)
        self.Percentage_Potentail_LTM = SBlock(1.65,2.00,0.62)
        self.Potential_LTM = TBlock(0.94,1.41,-0.32)
        self.Percentage_Potentail_STM = SBlock(-0.19,4.38,0.59)
        self.Potential_STM = TBlock(0,0,0)

    def SetInput(self, input):
        self.input = input

    def Pass(self):
        # Step 1
        pr =self.Percentage_LT_To_Rem.Activation(self.input * self.Percentage_LT_To_Rem.InputWeight + self.STM * self.Percentage_LT_To_Rem.STMWeight+self.Percentage_LT_To_Rem.Bias)
        self.LTM = pr * self.LTM
        # print(f'{pr} -> {self.LTM}')

        # Step 2
        pltm = self.Potential_LTM.Activation(self.input * self.Potential_LTM.InputWeight + self.STM * self.Potential_LTM.STMWeight + self.Potential_LTM.Bias)
        ppltm = self.Percentage_Potentail_LTM.Activation(self.input * self.Percentage_Potentail_LTM.InputWeight+ self.STM * self.Percentage_Potentail_LTM.STMWeight+self.Percentage_Potentail_LTM.Bias )
        res = ppltm * pltm
        self.LTM += res
        # print(f'{pltm} -> {ppltm} -> {res} -> {self.LTM}')

        # Step 3
        pstm = self.Potential_STM.Activation(self.LTM)
        ppstm = self.Percentage_Potentail_STM.Activation(self.input * self.Percentage_Potentail_STM.InputWeight + self.STM * self.Percentage_Potentail_STM.STMWeight+self.Percentage_Potentail_STM.Bias)
        res = pstm * ppstm
        self.STM = res
        # print(f'{pstm} -> {ppstm} -> {res}')

lstm = LSTMUnit(0,0,0)


# Assume the data for 4 days is as follow
data = [0,0.5,0.25,1]

for d in data:
    lstm.SetInput(d)
    lstm.Pass()
    print(f'LTM : {lstm.LTM}   |   STM : {lstm.STM}')

print(f"\nValue at the end of the stint {lstm.STM}\n")

# Assume the data for 4 days for some other company is as follow
data = [1,0.5,0.25,1]

for d in data:
    lstm.SetInput(d)
    lstm.Pass()
    print(f'LTM : {lstm.LTM}   |   STM : {lstm.STM}')

print(f"\nValue at the end of the stint {lstm.STM}\n")


'''
Output: 

LTM : -0.20124714108443564   |   STM : -0.1277553207642879
LTM : -0.20219254576038737   |   STM : -0.09652184013248234
LTM : -0.324621451889213   |   STM : -0.16621818785088271
LTM : 0.015269573158389116   |   STM : 0.0063931582408056405

Value at the end of the stint 0.0063931582408056405

LTM : 0.5204924394547513   |   STM : 0.28942793839380054
LTM : 0.9478620030138591   |   STM : 0.6306119794151629
LTM : 1.5309461656071615   |   STM : 0.8783385118141785
LTM : 2.461272080625197   |   STM : 0.9716443798973041

Value at the end of the stint 0.9716443798973041

'''