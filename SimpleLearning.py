import numpy as np
from puissance4 import Game
from collections import defaultdict
import random as rd
import matplotlib.pyplot as plt

def default_value():
    return 0
class SimpleTrainer:
    
    def __init__(self, h = 4, w = 4):
        self.h = h
        self.w = w
        self.Vdic = defaultdict(default_value) #value dic for states
        self.eps0 = 0.5
        self.epoch = 0
        self.decay = 0.0002
    def get_move(self, game, training = True):
        available_moves = game.get_valid_moves()
        if training:
            eps = self.get_randomness()
            r = rd.random()
            if r<eps:
                return rd.choice(available_moves)

        maxval = -np.inf
        best_moves = []
        for move in available_moves:
            w, board = game.simulate_move(move)
            new_state = game.get_hash(board)
            if self.Vdic[new_state]>maxval:
                maxval = self.Vdic[new_state]
                best_moves = [move]
            if self.Vdic[new_state] == maxval:
                best_moves.append(move)
      
        return rd.choice(best_moves)
    def updateV():
        for state in self.Vdic.keys():
            Vdic[state] = Vdic[state]//10
    def get_randomness(self):
        #return self.eps0
        pasacc = self.epoch//(1/self.decay)
        return self.eps0/(1+pasacc)
    def update_vals(self, state_history, winner):

        for i in range(0,len(state_history),2):
            state = state_history[i]
            self.Vdic[state] -= winner
        for i in range(1, len(state_history),2):
            state = state_history[i]
            self.Vdic[state] += winner

    def play_game(self):
        game = Game(self.h, self.w)
        state_history = [game.get_hash()]
        while not game.is_game_over:
            move = self.get_move(game)
            winner = game.play_move(move)
            new_state = game.get_hash()
            state_history.append(new_state)
        
        self.update_vals(state_history, winner)
        self.epoch += 1
    def train(self, n = 100):
        for i in range(n):
            self.play_game()

    def test_against_minmax(self, n = 1000, pmax = 2, player = 1):
        wins = 0
        losses = 0
        draws = 0
        for i in range(n):
            game = Game(self.h,self.w)
            while not game.is_game_over:
                if game.turn == player:
                    move = self.get_move(game, training = False)
                else:
                    move, score = game.minmax(pmax = pmax)
                winner = game.play_move(move)
            
            if winner == player:
                wins += 1
            elif winner == -player:
                losses += 1
            elif winner == 0:
                draws+=1
        if n ==0:
            return 0,0,0
        else:
            return wins/n, losses/n, draws/n
        
if __name__ == "__main__":
    h = 4
    w = 4
    nsteps = 1000 
    ntrain = 10000
    ntest = 100
    trainer = SimpleTrainer(h,w)
    
    epochs = [0]
    wrates0, wrates1, wrates2 = [], [], []
    lrates0, lrates1, lrates2 = [], [], []
    wrates00, wrates11, wrates22 = [], [], []
    lrates00, lrates11, lrates22 = [], [], []
    print("Itération : ", 0)
    w0,l0,d0 = trainer.test_against_minmax(n = ntest, pmax = 0, player = 1)
    w1,l1,d1 = trainer.test_against_minmax(n = ntest, pmax = 1, player = 1)
    w2,l2,d2 = trainer.test_against_minmax(n = ntest, pmax = 2, player = 1)
    w00,l00,d00 = trainer.test_against_minmax(n = ntest, pmax = 0, player = 1)
    w11,l11,d11 = trainer.test_against_minmax(n = ntest, pmax = 1, player = 1)
    w22,l22,d22 = trainer.test_against_minmax(n = ntest, pmax = 2, player = 1)
    print(w0,l0,d0)
    print(w1,l1,d1)
    print(w2,l2,d2)
    wrates0.append(w0),wrates1.append(w1),wrates2.append(w2)
    lrates0.append(l0),lrates1.append(l1),lrates2.append(l2)
    wrates00.append(w00),wrates11.append(w11),wrates22.append(w22)
    lrates00.append(l00),lrates11.append(l11),lrates22.append(l22)
    
    for i in range(nsteps):
        print("Itération : ", i+1)
        epochs.append((i+1)*ntrain)
        trainer.train(ntrain)
        w0,l0,d0 = trainer.test_against_minmax(n = ntest, pmax = 0, player = 1)
        w1,l1,d1 = trainer.test_against_minmax(n = ntest, pmax = 1, player = 1)
        w2,l2,d2 = trainer.test_against_minmax(n = ntest, pmax = 2, player = 1)
        w00,l00,d00 = trainer.test_against_minmax(n = ntest, pmax = 0, player = 1)
        w11,l11,d11 = trainer.test_against_minmax(n = ntest, pmax = 1, player = 1)
        w22,l22,d22 = trainer.test_against_minmax(n = ntest, pmax = 2, player = 1)
        
        print("Joueur 1 :")
        print(w0,l0,d0)
        print(w1,l1,d1)
        print(w2,l2,d2)
        print("Joueur 2")
        print(w00,l00,d00)
        print(w11,l11,d11)
        print(w22,l22,d22)
        
        wrates0.append(w0),wrates1.append(w1),wrates2.append(w2)
        lrates0.append(l0),lrates1.append(l1),lrates2.append(l2)
        wrates00.append(w00),wrates11.append(w11),wrates22.append(w22)
        lrates00.append(l00),lrates11.append(l11),lrates22.append(l22)