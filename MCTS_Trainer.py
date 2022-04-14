from matplotlib.style import available
import numpy as np
import tensorflow as tf
import copy as copy
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from collections import deque
import random as rd
from puissance4 import Game
import matplotlib.pyplot as plt
import time as time

import psutil as psutil #cpu usage
import os as os

from MCTS import MCTS

import pickle as pickle #charger le training data

from Policy_value_MCTS_net import Policynet, Valuenet

class Node:
    def __init__(self, state, turn, move = None, parent = None):
        self.state = state
        self.turn = turn #1 or -1, turn for the state
        self.move = move #None pour le root node

        self.val = 0 #prior value (for player 1 always) => neural net

        self.vis = 0 #how many times node has been visited
        self.W = 0 #sum (values)
        #self.Q = 0 #Qval = W/vis
        self.P = None #Prior probability => neural net

        self.parent = parent #parent node
        self.children = [] #child nodes
        
        self.is_expandable = True #is state still expandable
        self.winner = None #winner of state

    def get_Q(self):
        if self.winner == None:
            if self.vis != 0:
                return self.W/self.vis
            else:
                return 0
        else:
            return self.winner
    def get_sum_nb(self, t = 1): #sum of visits for all brothers
        rep = 0
        for child in self.parent.children:
            rep += child.vis**(1/t)
        return rep
    def get_uct(self):
        sumnb = self.get_sum_nb()
        true_Q = -self.turn*self.get_Q()#valeur toujours pour le joueur 1 OK
        cpuct = np.sqrt(2)
        return true_Q + cpuct*self.P * np.sqrt(sumnb)/(1+self.vis)

    def get_result(self):
        return self.val/self.vis


class MCTS_NN: #does the ting for one positions (in game)
    def __init__(self,h = 6, w = 7, game = 0, n_iteration = 1000, modelP = 0, modelV = 0, rootnode = 0):
        self.h = h
        self.w = w
        self.n_iteration = n_iteration
        self.policynet = Policynet(h=h, w=w, model = modelP)
        self.valuenet = Valuenet(h=h, w=w, model = modelV)

        #game only useful to know initial conditions
        self.game = game
        if isinstance(game, int):
            self.game = Game(h = self.h, w= self.w)

        
        state = self.game.get_state_CNN_bitmap()
        if rootnode == 0:
            self.root_node = Node(state, self.game.turn)
        else:
            self.root_node = rootnode

        self.initial_prediction = self.policynet.get_probabilities(state)
        _, self.sum_valid_probabilities = self.policynet.get_probabilities_adjusted(state)

    def get_new_prediction(self):
        #doesnt affect the illigal moves !
        prediction = copy.deepcopy(self.initial_prediction)
        for child in self.root_node.children:
            move = child.move
            pr = child.vis/child.get_sum_nb()
            prediction[move] = pr/self.sum_valid_probabilities

        return prediction
    def get_move_probabilities(self, t = 1):
        #list of length w, illegle moves = 0, legal moves = p, sum(p) = 1
        probabilities = self.w * [0]
        for child in self.root_node.children:
            move = child.move
            pr = child.vis**(1/t)/child.get_sum_nb(t = t)
            probabilities[move] = pr
        return probabilities
    def get_prob_and_pred(self): #optimisation

        #prob : 0 sur les coups invalies, pr réelle sur les vrais coups
        #pred : valeur initiale de la prediction pour les coups invalides, et proba ajustée pour les coups valides
        prediction = copy.deepcopy(self.initial_prediction)
        probabilities = self.w * [0]
        for child in self.root_node.children:
            move = child.move
            pr = child.vis/child.get_sum_nb()
            prediction[move] = pr/self.sum_valid_probabilities
            probabilities[move] = pr
        return probabilities, prediction

    def get_qvals(self):
        values = self.w * [-1.1] #illegle moves is -1.1
        for child in self.root_node.children:
            move = child.move
            values[move] = child.get_Q()
        
        return values
            

    def do_process(self):
        self.expand_node(self.root_node)
        n_iteration = self.n_iteration
        for i in range(n_iteration):
            node = self.select_node()
            node = self.expand_node(node)
            if node.winner == None: 
                self.update(node, node.val)
            else:
                self.update(node, node.winner)
    def select_node(self):
        #select node to expand 
        node = self.root_node
        while not(node.is_expandable):
            if node.children == []:
                return node #terminal state
            node = max(node.children, key = lambda x : x.get_uct())
        #print("NODE SELECTED : ", node.state)
        return node

    def expand_node(self, node):
        #expand : node has no children or node is terminal => get all children with proba,  + give value to node
        if not(node.is_expandable): #peut pas etre étendu, la simulation sera directe
            return node
        if node.children != []:
            print("Error, node has children")
        #print("NODE EXPANDED : ", node.state)
        state, turn = node.state, node.turn #state is bitmap

        #give value to node
        node.val = self.valuenet.get_value(state)

        #get available moves
        board = self.game.convert_state_to_board_bitmap(state)
        available_moves = self.game.get_valid_moves(board = board)

        unexplored_moves = available_moves

        if unexplored_moves == []: #probleme
            print(node.state)
            print(node.is_expandable)
            print(node.parent)
            print([child.state for child in node.children])
        
        #fully expand node and gives probabilities to children
        probas, sum_valid_proba = self.policynet.get_probabilities_adjusted(state) #probabilities for each move
        for move in unexplored_moves:
            winner, new_board = self.game.simulate_move(move, board, turn)
            new_state = self.game.get_state_CNN_bitmap(board = new_board, turn = -turn)
            new_node = Node(new_state, -turn, parent = node, move=move)
        
            #check for winner
            if winner != None:
                new_node.is_expandable = False #terminal state
                new_node.winner = winner
            
            #gives probability of chosing node
            new_node.P = probas[move]
            #add child node
            node.children.append(new_node)

            
        node.is_expandable = False

        return node
    
    def update(self, node, value):
        node.vis += 1
        node.W += value
        if node.parent != None:
            self.update(node.parent, value)

class MCTS_trainer:
    def __init__(self, h = 6, w = 7, modelP = 0, modelV = 0, n_iteration = 100, rootnode = 0)-> None:
        self.h = h
        self.w = w
        self.rootnode = rootnode
        self.n_iteration = n_iteration

        self.policynet = Policynet(h, w, model = modelP)
        self.valuenet = Valuenet(h,w, model = modelV)
    
    def play_game(self):
        statelist = []
        predlist = [] #predictions for moves, for every state in game => used to fit
        valuelist = []
        game = Game(self.h, self.w)
        mcts_nn = 0
        while not game.is_game_over:
            statelist.append(game.get_state_CNN_bitmap())
            move, mcts_nn = self.get_move(game=game, NNmcts=mcts_nn, return_nn_mcts=True, doproc=True, training=True)

            prob, pred = mcts_nn.get_prob_and_pred()
            qvals = mcts_nn.get_qvals()
            if game.turn == 1:
                val = max(qvals)
            else:
                val = min([q for q in qvals if q>=-1])
            valuelist.append([val])
            predlist.append(pred)

            winner = game.play_move(move)

            #update du rootnode
            new_root_node = self.get_new_root_node(move, mcts_nn)
            mcts_nn = MCTS_NN(self.h, self.w, game = game, n_iteration=self.n_iteration,
                            modelP = self.policynet.model, modelV = self.valuenet.model, rootnode = new_root_node)

        #reset du rootnode
        self.rootnode = 0
        print(winner)
        return [statelist, predlist, valuelist]
    def play_game_MCTS(self):
        statelist = []
        predlist = [] #predictions for moves, for every state in game => used to fit
        valuelist = []
        mcts = 0
        game = Game(self.h, self.w)
        while not game.is_game_over:
            state = game.get_state_CNN_bitmap()
            normal_mcts = MCTS(self.h, self.w, game = game, n_iteration = 1000)
            best_move, move_dic = normal_mcts.do_process()
            pred = normal_mcts.get_probabilities(move_dic, invalid_move_value = 0)
            if game.turn == 1:
                val = normal_mcts.get_val(move_dic)
            else:
                val = - normal_mcts.get_val(move_dic)

            valuenet_val = self.valuenet.get_value(state) #valeur
            if abs(val)>0.5:
                #stocker datapoints
                valuelist.append([val])
                predlist.append(pred)
                statelist.append(state)
            elif abs(val-valuenet_val)>0.5:
                valuelist.append([val])
                predlist.append(pred)
                statelist.append(state)
            #jouer coup, Mcts réseau de neuronnes
            mcts = MCTS_NN(self.h, self.w, game = game, n_iteration = self.n_iteration, modelP = self.policynet.model, modelV = self.valuenet.model)
            mcts.do_process()

            r = rd.random()
            if r<0.2:
                move = game.random_IA()
            elif r<0.7:
                move = best_move
            else:
                move = self.get_move(game = game,NNmcts=mcts)
            winner = game.play_move(move)
        print(winner)
        return [statelist, predlist, valuelist]
        
    def update_weights(self, datapoints, batch_size = 1):
        #datapoints = [statelist, predlist, valuelist]
        states = np.array(datapoints[0])
        preds = np.array(datapoints[1])
        vals = np.array(datapoints[2])
        self.policynet.model.fit(states, preds, batch_size=batch_size)
        self.valuenet.model.fit(states, vals, batch_size=batch_size)

    def self_train(self, ntrains = 10):
        datapoints = [[], [], []]
        for i in range(ntrains):
            data = self.play_game()
            datapoints[0] = datapoints[0] + data[0]
            datapoints[1] = datapoints[1] + data[1]
            datapoints[2] = datapoints[2] + data[2]
        if len(datapoints[0])>0:
            self.update_weights(datapoints, batch_size = min(len(datapoints),100))
    def train_mcts_guided(self, ntrains = 10):
        trainsize = 50
        datapoints = [[], [], []]
        for i in range(ntrains):
            datapoints_i = self.play_game_MCTS()
            datapoints[0] = datapoints[0] + datapoints_i[0]
            datapoints[1] = datapoints[1] + datapoints_i[1]
            datapoints[2] = datapoints[2] + datapoints_i[2]
            if len(datapoints[0]) >trainsize:
                print("yes")
                self.update_weights(datapoints, batch_size=25)
                datapoints = [[], [], []]


    def get_move(self, game, verbose = 1, NNmcts = 0, training = False, t = 1, return_nn_mcts = False, doproc = False):
        #si training : renvoit un coups distribué selon les probs
        #sinon renvoi le coup le plus visité
        mcts = NNmcts
        if isinstance(NNmcts,int):
            mcts = MCTS_NN(self.h, self.w, game = game, n_iteration = self.n_iteration, modelP = self.policynet.model, modelV = self.valuenet.model)
            mcts.do_process()
        elif doproc:
            mcts.do_process()
        qvals = mcts.get_qvals()
        if game.turn == 1:
            val = max(qvals)
        else:
            val = min([q for q in qvals if q>=-1])
        prob = mcts.get_move_probabilities(t =t)
        if training:
            move = np.random.choice(list(range(self.w)), p = prob)
        else:
            move = np.argmax(prob)
        if verbose == 1:
            print("NNMCTS : probalities for moves : ", prob)
            print("NNMCTS : Qvalues : ", qvals)
            print("NNMCTS : Value : ", val)
        
        if return_nn_mcts:
            return move, mcts
        else:
            return move

    
    def test_agent_minmax(self, n = 1000, player = 1, pmax = 2):
        wins = 0
        losses = 0
        draws = 0
        for i in range(n):
            self.rootnode = 0
            moves = [] #historique des coups
            game = Game(self.h, self.w, profondeur_max = pmax)
            mcts_nn = 0
            while not(game.is_game_over):
                turn = game.turn
                #jouer le coup
                if turn == player:
                    #
                    move, mcts_nn = self.get_move(game=game, NNmcts=mcts_nn, return_nn_mcts=True, doproc=True)
                    
                    #update du mcts_nn
                    #play move
                    winner = game.play_move(move)

                    #update du rootnode
                    new_root_node = self.get_new_root_node(move, mcts_nn)
                    mcts_nn = MCTS_NN(self.h, self.w, game = game, n_iteration=self.n_iteration,
                                    modelP = self.policynet.model, modelV = self.valuenet.model, rootnode = new_root_node)
                else:
                    move, score = game.minmax(pmax = pmax)
                    winner = game.play_move(move)
                    #update du rootnode
                    new_root_node = self.get_new_root_node(move, mcts_nn)
                    mcts_nn = MCTS_NN(self.h, self.w, game = game, n_iteration=self.n_iteration,
                                    modelP = self.policynet.model, modelV = self.valuenet.model, rootnode = new_root_node)
                moves.append(move)
                #verif du gagnant
                if winner == player:
                    wins += 1
                    print("Moves : ", moves)
                elif winner == -player:
                    losses += 1
                    print("Moves : ", moves)
                elif winner == 0: #draw
                    draws += 1
                    print("Moves : ", moves)

        return wins/n, losses/n, draws/n


    
    def get_new_root_node(self,move,NNmcts):
        #returns new rootnode for move if existant, 0 else
        if NNmcts == 0:
            return 0
        elif NNmcts.root_node == 0:
            return 0
        elif len(NNmcts.root_node.children) == 0:
            return 0
        else:

            for child in NNmcts.root_node.children:
                if child.move == move:
                    newrootnode = child
            #print("got new root : ", newrootnode.vis)
            return newrootnode
if __name__ == "__main__":
    h = 6
    w = 7
    n_iteration = 100

    modelP = tf.keras.models.load_model("policy_crossentropy3"  + str(h) + str(w))
    modelV = tf.keras.models.load_model("value_crossentropy3" + str(h) + str(w))

    trainer = MCTS_trainer(h = h, w = w, n_iteration = n_iteration, modelP = modelP, modelV=modelV)
    ntrain = 0
    trainer.self_train(ntrain)

    ntest = 10
    pmax = 5
    player = 1

    trainer.policynet.model.save("policy_crossentropy3" + str(h) + str(w))
    trainer.valuenet.model.save("value_crossentropy3" + str(h) + str(w))

    wr,lr,dr = trainer.test_agent_minmax(n = ntest, player = player, pmax = pmax)
    #wr,lr,dr = trainer.test_against_MCTS(ntest = ntest, player = player, MCTSnit = 1000, NNMCTSnit = 100)
    print(wr,lr,dr)
