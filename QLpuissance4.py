from hashlib import new
from puissance4 import Game
import random as rd
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
import numpy as np
import time as time

import pickle as pickle #sauver et load dico

from savitzky_golay import get_interpol
"""
state/hash : string de h*w 0,1 ou 2 ex : '000000000000010000002000020102101212121221'
action : int entre 0 et w-1
"""
class QAgent:
    def __init__(self, h = 6, w=7, Q = {}) -> None:
        self.h = h
        self.w = w
        self.game = Game(h,w) #sert uniquement pour la fonction "available moves", le vrai jeu sera dans le Trainer
        #hyperparameters
        self.lr = 0.1  #learning rate
        self.gamma = 0.95 #discount factor
        self.randomness = 0.1 #real rand = randomness/epoch
        self.min_randomness = 0.1
        #Q[etat][action]
        self.Q = Q #{etat1 : {0 : val, 1 : val etc...}, etat2 : {0 : val2 etc ...}}
        self.epoch = 1
        
    def real_randomness(self):
        return self.min_randomness + (self.randomness-self.min_randomness)/self.epoch

    def update_Q(self, move, reward, state, new_state):
        if not(new_state in self.Q.keys()):
            self.Q[new_state] = {}
            available_moves = self.game.available_moves(new_state)
            for move in available_moves:
                self.Q[new_state][move] = 0
        if not(state in self.Q.keys()):
            self.Q[state] = {}
            available_moves = self.game.available_moves(state)
            for move in available_moves:
                self.Q[state][move] = 0

        if self.Q[new_state] == {}:
            maxQ = 0
        else:
            maxQ = max(self.Q[new_state].values())
        
        if not(move in self.Q[state].keys()):
            self.Q[state][move] = 0

        self.Q[state][move] = (1-self.lr)*self.Q[state][move]  + self.lr * (reward + self.gamma*maxQ) #formula
    
    def get_random_move(self,state):
        moves = self.game.available_moves(state)
        return rd.choice(moves)
    
    def get_move(self, state, training = True):
        #randomness
        if training:
            r = rd.random() #unif 0,1
            real_randomness = self.real_randomness()
            if r<real_randomness and training:
                return self.get_random_move(state)
        
        if not(state in self.Q.keys()): #état du jeu jamais exploré
            return self.get_random_move(state)
        if self.Q[state] == {}: #aucune action explorée pour l'état
            return self.get_random_move(state)

        #RQ : il n'y aura que des coups autorisés
        move_val_list = [(move, self.Q[state][move]) for move in self.Q[state].keys()]
        best_move, best_val = max(move_val_list, key=lambda item:item[1])

        return best_move


class Trainer:
    def __init__(self, h =6, w = 7, Q = {}):
        self.agent = QAgent(h,w, Q = Q) #ne change pas
        self.h = h
        self.w = w
        self.game = Game(h,w) #change
        self.scores = [] #len(scores) = epoch - 1
        self.reinforced_training = False #refais une maj à la fin de la game avec l'historique

        self.draw_reward = 0
        self.o_lossreward = -10
        self.x_lossreward = -10
        self.x_winreward = 10
        self.o_winreward = 10
    def train_self_one_game(self): #trains against himself
        self.game = Game(self.h,self.w)
        states = [None, None, None, self.game.get_hash()] #historique des 4 derniers états = file
        historique = []
        moves = [None, None, None, None] #coups menant aux états respectifs
        while not(self.game.is_game_over):
            move = self.agent.get_move(states[-1])
            w = self.game.play_move(move)
            new_state = self.game.get_hash()

            #verif si partie finie
            if self.game.is_game_over:
                if w == 0: #matchnul
                    state1 = states[-2]
                    move1 = moves[-1] #coup après state1
                    state2 = states[-1]
                    move2 = move #coup après state2
                    self.agent.update_Q(move1, self.draw_reward, state1, new_state)
                    self.agent.update_Q(move2, self.draw_reward, state2, new_state)
                    historique.append((move1, self.draw_reward, state1, new_state))
                    historique.append((move2, self.draw_reward, state2, new_state))
                if w == 1: #test innutilse si il y avait un loss/winreward commun
                    state1 = states[-2]
                    move1 = moves[-1] #coup après state1
                    state2 = states[-1]
                    move2 = move #will get rewarded for this move

                    self.agent.update_Q(move1, self.o_lossreward, state1, new_state)
                    self.agent.update_Q(move2,self.x_winreward, state2, new_state)
                    historique.append((move1, self.o_lossreward, state1, new_state))
                    historique.append((move2, self.x_winreward, state2, new_state))
                if w == -1:
                    state1 = states[-2]
                    move1 = moves[-1] #coup après state1
                    state2 = states[-1]
                    move2 = move #will get rewarded for this move
                    self.agent.update_Q(move1, self.x_lossreward, state1, new_state)
                    self.agent.update_Q(move2,self.o_winreward, state2, new_state)
                    historique.append((move1, self.x_lossreward, state1, new_state))
                    historique.append((move2,self.o_winreward, state2, new_state))
                    
            else: #game not over, actually updating for old move
                old_state = states[-2]
                move_for_old_state = moves[-1]
                if old_state != None:

                    self.agent.update_Q(move_for_old_state, 0, old_state, new_state)
                    historique.append((move_for_old_state, 0, old_state, new_state))



            states.pop(0) #defiler
            states.append(new_state) #enfiler

            moves.pop(0) #defiler
            moves.append(move) #enfiler

        if self.reinforced_training: #refait tous les coups et update
            for c in historique:
                self.agent.update_Q(*c)
        self.agent.epoch += 1
        return historique
    def self_train(self, n=1000):
        k=max(n//100,1)
        for i in range(n):
            if i%k == 0:
                
                #print(int(100*(i/n)), "%")
                pass
            self.train_self_one_game()
            
            
    def play_Q(self): #play against the Q
        self.game = Game()
        self.game.print_board()
        state = self.game.get_hash()
        move = self.agent.get_move(state)
        self.game.play(move)
        self.game.print_board()
        while self.game.is_game_over == False:
            move = int(input("Your move : "))
            self.game.play(move)
            self.game.print_board()
            print(" ")
            if self.game.is_game_over:
                print("OVER")
            else:
                state = self.game.get_hash()
                move = self.agent.get_move(state)
                self.game.play(move)
            self.game.print_board()
            print(" ")

        b = int(input("play again ? 1 = yes, 0 = no : "))
        if b == 1:
            self.play_Q()
            
    def test_agent_minmax(self, n = 1000, player = 1, pmax = 2):
        agent = self.agent

        wins = 0
        losses = 0
        draws = 0
        for i in range(n):
            game = Game(self.h, self.w, profondeur_max = pmax)
            moves = []
            while not(game.is_game_over):
                turn = game.turn
                #jouer le coup
                if turn == player:
                    #get state and move for DQN
                    state = game.get_hash()
                    move = agent.get_move(state, training = False)
                    #play move
                    winner = game.play_move(move)
                else:
                    move, score = game.minmax(pmax = pmax)
                    winner = game.play_move(move)
                
                moves.append(move)
                #verif du gagnant
                if winner == player:
                    wins += 1
                elif winner == -player:
                    losses += 1
                elif winner == 0: #draw
                    draws += 1
        #print(moves)
        return wins/n, losses/n, draws/n
            

if __name__ == "__main__":
    h = 6
    w = 7
    
    number_of_steps = 5000
    trains_per_step = 10000
    tests_per_step = 100
    trainer = Trainer(h,w)
    
    start = time.time()
    
    wr0, lr0, dr0 = trainer.test_agent_minmax(n=tests_per_step, player = 1, pmax = 0)
    wr1, lr1, dr1 = trainer.test_agent_minmax(n=tests_per_step, player = 1, pmax = 1)
    wr2, lr2, dr2 = trainer.test_agent_minmax(n=tests_per_step, player = 1, pmax = 2)
    wr00, lr00, dr00 = trainer.test_agent_minmax(n=tests_per_step, player = -1, pmax = 0)
    wr11, lr11, dr11 = trainer.test_agent_minmax(n=tests_per_step, player = -1, pmax = 1)
    wr22, lr22, dr22 = trainer.test_agent_minmax(n=tests_per_step, player = -1, pmax = 2)
    
    end = time.time()
    print(" First test (no training) for 6*", tests_per_step, " tests : ", end-start, " seconds")
    print("Profondeur 0, joueur 1 : ", wr0, lr0, dr0)
    print("Profondeur 1, joueur 1 : ", wr1, lr1, dr1)
    print("Profondeur 2, joueur 1 : ", wr2, lr2, dr2)
    print("Profondeur 0, joueur 2 : ", wr00, lr00, dr00)
    print("Profondeur 1, joueur 2 : ", wr11, lr11, dr11)
    print("Profondeur 2, joueur 2 : ", wr22, lr22, dr22)
    wrates0 = [wr0]
    lrates0 = [lr0]
    wrates1 = [wr1]
    lrates1 = [lr1]
    wrates2 = [wr2]
    lrates2 = [lr2]
    wrates00 = [wr00]
    lrates00 = [lr00]
    wrates11 = [wr11]
    lrates11 = [lr11]
    wrates22 = [wr22]
    lrates22 = [lr22]
    
    epochs = [0]
    for i in range(number_of_steps):
        if i%1000 == 0:
            print(100*i/number_of_steps, "%")
            
        start = time.time()
        trainer.self_train(n=trains_per_step)
        end = time.time()
        print(" Time for ", trains_per_step, " training: ", end-start, " seconds")

        epochs.append(trains_per_step*(i+1))
        start = time.time()
        wr0, lr0, dr0 = trainer.test_agent_minmax(n=tests_per_step, player = 1, pmax = 0)
        wr1, lr1, dr1 = trainer.test_agent_minmax(n=tests_per_step, player = 1, pmax = 1)
        wr2, lr2, dr2 = trainer.test_agent_minmax(n=tests_per_step, player = 1, pmax = 2)
        wr00, lr00, dr00 = trainer.test_agent_minmax(n=tests_per_step, player = -1, pmax = 0)
        wr11, lr11, dr11 = trainer.test_agent_minmax(n=tests_per_step, player = -1, pmax = 1)
        wr22, lr22, dr22 = trainer.test_agent_minmax(n=tests_per_step, player = -1, pmax = 2)
        end = time.time()
        print(" i = ", i, "for 3*", tests_per_step, " tests : ", end-start, " seconds")
        print("Profondeur 0, joueur 1 : ", wr0, lr0, dr0)
        print("Profondeur 1, joueur 1 : ", wr1, lr1, dr1)
        print("Profondeur 2, joueur 1 : ", wr2, lr2, dr2)
        print("Profondeur 0, joueur 2 : ", wr00, lr00, dr00)
        print("Profondeur 1, joueur 2 : ", wr11, lr11, dr11)
        print("Profondeur 2, joueur 2 : ", wr22, lr22, dr22)
        wrates0.append(wr0)
        lrates0.append(lr0)
        wrates1.append(wr1)
        lrates1.append(lr1)
        wrates2.append(wr2)
        lrates2.append(lr2)
        wrates00.append(wr00)
        lrates00.append(lr00)
        wrates11.append(wr11)
        lrates11.append(lr11)
        wrates22.append(wr22)
        lrates22.append(lr22)
    plt.plot(epochs, values)
    plt.plot(epochs, wrates)
    plt.plot(epochs, lrates)
    plt.plot(epochs, drates)
    
    X = np.array(epochs).reshape(-1,1)
    #wreg = LinearRegression().fit(X,wrates)
    #lreg = LinearRegression().fit(X,lrates)
    #dreg = LinearRegression().fit(X,drates)
    #vreg = LinearRegression().fit(X,values)
    #yw = wreg.coef_[0] * np.array(epochs) + wreg.intercept_
    #yl = lreg.coef_[0] * np.array(epochs) + lreg.intercept_
    #yd = dreg.coef_[0] * np.array(epochs) + dreg.intercept_
    #yv = vreg.coef_[0] * np.array(epochs) + vreg.intercept_

    #Pour X
    plt.plot(epochs, wrates0, epochs, wrates1, epochs, wrates2)
    plt.title("Taux de victoire en fonction de l'époque, contre différentes IA minmax")
    plt.xlabel("époque")
    plt.ylabel("taux de victoire")
    plt.legend(("vs profondeur 0", "vs profondeur 1", "vs profondeur 2"))
    #Pour O
    plt.plot(epochs, wrates00, epochs, wrates11, epochs, wrates22)
    plt.title("Taux de victoire en fonction de l'époque, contre différentes IA minmax")
    plt.xlabel("époque")
    plt.ylabel("taux de victoire")
    plt.legend(("vs profondeur 0", "vs profondeur 1", "vs profondeur 2"))
    
    values0 = [wrates0[i] - lrates0[i] for i in range(len(lrates0))]
    values1 = [wrates1[i] - lrates1[i] for i in range(len(lrates1))]
    values2 = [wrates2[i] - lrates2[i] for i in range(len(lrates2))]
    
    values00 = [wrates00[i] - lrates00[i] for i in range(len(lrates00))]
    values11 = [wrates11[i] - lrates11[i] for i in range(len(lrates11))]
    values22 = [wrates22[i] - lrates22[i] for i in range(len(lrates22))]
        
    
    #Pour X
    plt.plot(epochs, values0, epochs, values1, epochs, values2)
    #plt.title("Résultat (taux de victoire - taux de défaite) en fonction de l'époque, contre différentes IA minmax")
    plt.xlabel("époque")
    plt.ylabel("Résultat")
    plt.legend(("vs profondeur 0", "vs profondeur 1", "vs profondeur 2"))
    
    #Pour O
    plt.plot(epochs, values00, epochs, values11, epochs, values22)
    #plt.title("Résultat (taux de victoire - taux de défaite) en fonction de l'époque, contre différentes IA minmax")
    plt.xlabel("époque")
    plt.ylabel("Résultat")
    plt.legend(("vs profondeur 0", "vs profondeur 1", "vs profondeur 2"))
    
    values0 = get_interpol(values0)
    values1 = get_interpol(values1)
    values2 = get_interpol(values2)
    
    win = 51
    pol = 3
    values00 = get_interpol(values00, win, pol)
    values11 = get_interpol(values11, win, pol)
    values22 = get_interpol(values22, win, pol)
    """
    s = "Qdic_" + str(h) + str(w) + ".pkl"
    a_file = open(s, "wb")
    pickle.dump(trainer.agent.Q, a_file)
    a_file.close()
    
    a_file = open(s, "rb")
    dic = pickle.load(a_file)"""
    