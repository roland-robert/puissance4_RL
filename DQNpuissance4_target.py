import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from collections import deque
import random as rd
from puissance4 import Game
import matplotlib.pyplot as plt
import time as time
from savitzky_golay import get_interpol #interpolation de courbe

from MCTS import MCTS
import os, psutil
#process = psutil.Process(os.getpid())
#print(process.memory_info().rss)


physical_devices = tf.config.experimental.list_physical_devices("GPU")
#print(physical_devices)

"""
state/observation : board applatit (hw,)
ICI plus de memoire grande, update après chaque partie

"""
class Hyperparameters:
    def __init__(self):
        self.lr = 0.01
        self.Qlr = 0
        self.gamma = 0.95

        self.eps = 0.99 #randomness
        self.min_eps = 0.05
        self.decay = 0.01



        self.seed = 10
        self.std = 0.1
        self.mean = 0
        
        #self.loss = tf.keras.losses.MeanSquaredError()
        self.loss = tf.keras.losses.MeanSquaredError()
        #self.loss = tf.keras.losses.Huber()
        
        self.metric=tf.keras.metrics.MeanSquaredError()
        self.kern_initializer = tf.keras.initializers.RandomNormal(mean=self.mean, stddev=self.std, seed=self.seed)
        #self.kern_initializer = tf.keras.initializers.GlorotUniform(seed = self.seed)
        self.bias_initializer = tf.keras.initializers.Zeros()
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr)
        
        self.maxmemlen = 10000 #memoire maxi des coups 
        self.games_before_train = 1
        self.games_before_sync = 4
        self.allow_illegal_moves = False

        #self.MIN_REPLAY_SIZE = 500
        self.batch_size = 16
        self.sample_size = 256
        
    def real_randomness(self, epoch):
         return self.min_eps + (self.eps - self.min_eps)*np.exp(-self.decay*epoch)
class DQNAgent:
    def __init__(self, h=6, w=7, model = 0) -> None:
        self.h = h
        self.w = w
        self.hpp = Hyperparameters()
        self.game = Game(h,w) #just for rules
        if model == 0:
            
            self.model = Sequential()
            self.model.add(layers.Dense(units = 256, input_shape = (h*w,), activation = "selu", 
                  kernel_initializer=self.hpp.kern_initializer,
                  bias_initializer=self.hpp.bias_initializer))
            self.model.add(layers.Dense(units = 128, input_shape = (h*w,), activation = 'tanh', 
                  kernel_initializer=self.hpp.kern_initializer,
                  bias_initializer=self.hpp.bias_initializer))

            self.model.add(layers.Dense(units = w,activation = "tanh"))
        
            self.model.compile(optimizer = self.hpp.optimizer, 
                   loss = self.hpp.loss,
                   metrics=[tf.keras.metrics.MeanSquaredError()])
            #Adam, SGD
        else:
            self.model = model
        
        print(self.model.summary())
        #QL params
        self.Qlr = self.hpp.Qlr #Q learning rate in the Q formula
        self.gamma = self.hpp.gamma #discount rate in the Q formula
        #params
        self.epoch = 1 #number of games played
        self.randomness = self.hpp.eps

    def get_qs(self, X): #get Q valus for input = state X
        L = []
        L.append(X)
        Y = self.model.predict(np.array(L))[0]
        return Y

    def get_move(self, X, training = False):
        board = self.game.convert_state_to_board(X)
        available_moves = self.game.get_valid_moves(board)
        if training: #entrainant => randomness
            r = self.hpp.real_randomness(self.epoch)
            rnd = rd.random()
            if rnd<r:
                return rd.choice(available_moves)
        Y = self.get_qs(X)
        if self.hpp.allow_illegal_moves and training == True:
            return np.argmax(Y)
        
        themax = -np.inf
        imax = -1
        for i in available_moves:
            if Y[i]>themax:
                themax = Y[i]
                imax = i
        return imax


class Trainer:
    def __init__(self,h=6, w=7, modelX = 0, modelO = 0) -> None:
        self.h = h
        self.w = w
        self.game =  Game(h,w)

        self.draw_reward = 0
        self.o_lossreward = -10
        self.x_lossreward = -10
        self.x_winreward = 10
        self.o_winreward = 10

        self.hpp = Hyperparameters()
        self.agent = DQNAgent(h,w, modelX)
        self.target_agent = DQNAgent(h,w, modelX)
        self.target_agent.model.set_weights(self.agent.model.get_weights())
        self.replay_memory = deque(maxlen = self.hpp.maxmemlen) #liste de couples (observation, action, reward, new_observation, done)

        self.agentO = DQNAgent(h,w, modelO)
        self.target_agentO = DQNAgent(h,w, modelX)
        self.target_agentO.model.set_weights(self.agentO.model.get_weights())
        self.replay_memoryO = deque(maxlen = self.hpp.maxmemlen)

    def _reset_deques(self):
        self.replay_memory = deque(maxlen = self.hpp.maxmemlen)
        self.replay_memoryO = deque(maxlen = self.hpp.maxmemlen)
    def train_with_minmax(self, n = 1000, player = 1, profondeur_max = 2):
        games_played_train = 0 #après x parties => training
        games_played_sync = 0 #aprèx x parties => sync de agent et target agent
        agent = self.agent if player == 1 else self.agentO
        for epoch in range(n):
            agent.epoch += 1
            game = Game(self.h, self.w, profondeur_max=profondeur_max)
            turn = 1
            DQNstates = [[], []] #states visited by DQN : FILE
            DQNmoves = [0,0] #move played by DQN : FILE DQNmoves[-1] = last move played (move for states[-1])
            while not(game.is_game_over):
                turn = game.turn
                if turn == player:
                    #get state and move for DQN
                    state = game.get_state()
                    move = agent.get_move(state)
                    #update move and state history
                    DQNmoves.append(move)
                    DQNmoves.pop(0)
                    DQNstates.append(state)
                    DQNstates.pop(0)
                    #add to replay memory : reward 0 (en cours)
                    if len(DQNstates[-2]) >0:
                        self.replay_memory.append([DQNstates[-2], DQNmoves[-2], 0, DQNstates[-1], False])
                    #play move
                    winner = game.play_move(move)
                else:
                    move, score = game.minmax()
                    winner = game.play_move(move)

                #checking for win
                if winner == player:
                    reward = self.x_winreward if player == 1 else self.o_winreward
                    winning_state = game.get_state()
                    if player == 1:
                        self.replay_memory.append([DQNstates[-1], DQNmoves[-1], reward, winning_state, True])
                    else:
                        self.replay_memory.append([DQNstates[-1], DQNmoves[-1], reward, winning_state, True])
                elif winner == -player:
                    reward = self.x_lossreward if player == 1 else self.o_lossreward
                    losing_state = game.get_state()
                    if player == 1:
                        self.replay_memory.append([DQNstates[-1], DQNmoves[-1], reward, losing_state, True])
                    else:
                        self.replay_memory.append([DQNstates[-1], DQNmoves[-1], reward, losing_state, True])
                elif winner == 0: #draw
                    reward = self.draw_reward
                    drawing_state = game.get_state()
                    if player == 1:
                        self.replay_memory.append([DQNstates[-1], DQNmoves[-1], reward, drawing_state, True])
                    else:
                        self.replay_memory.append([DQNstates[-1], DQNmoves[-1], reward, drawing_state, True])
                    
                
            games_played_train += 1
            games_played_sync += 1
            if games_played_train == self.hpp.games_before_train:
                games_played_train = 0
                self.train(player = 1)
            if games_played_sync== self.hpp.games_before_sync:
                games_played_sync = 0
                if player == 1:
                    self.target_agent.model.set_weights(self.agent.model.get_weights())
                else:
                    self.target_agentO.model.set_weights(self.agentO.model.get_weights())
    def train_with_mcts(self, n = 1000, player = 1, n_iteration = 1000):
        games_played_train = 0 #après x parties => training
        games_played_sync = 0 #aprèx x parties => sync de agent et target agent
        agent = self.agent if player == 1 else self.agentO
        for epoch in range(n):
            agent.epoch += 1
            game = Game(self.h, self.w)
            turn = 1
            DQNstates = [[], []] #states visited by DQN : FILE
            DQNmoves = [0,0] #move played by DQN : FIFO DQNmoves[-1] = last move played (move for states[-1])
            while not(game.is_game_over):
                turn = game.turn
                if turn == player:
                    #get state and move for DQN
                    state = game.get_state()
                    move = agent.get_move(state)
                    #update move and state history
                    DQNmoves.append(move)
                    DQNmoves.pop(0)
                    DQNstates.append(state)
                    DQNstates.pop(0)
                    #add to replay memory : reward 0 (en cours)
                    if len(DQNstates[-2]) >0:
                        self.replay_memory.append([DQNstates[-2], DQNmoves[-2], 0, DQNstates[-1], False])
                    #play move
                    winner = game.play_move(move)
                else:
                    mcts = MCTS(h=h,w=w,game = game, n_iteration = n_iteration)
                    move, _ = mcts.do_process()
                    winner = game.play_move(move)

                #checking for win
                if winner == player:
                    reward = self.x_winreward if player == 1 else self.o_winreward
                    winning_state = game.get_state()
                    if player == 1:
                        self.replay_memory.append([DQNstates[-1], DQNmoves[-1], reward, winning_state, True])
                    else:
                        self.replay_memory.append([DQNstates[-1], DQNmoves[-1], reward, winning_state, True])
                elif winner == -player:
                    reward = self.x_lossreward if player == 1 else self.o_lossreward
                    losing_state = game.get_state()
                    if player == 1:
                        self.replay_memory.append([DQNstates[-1], DQNmoves[-1], reward, losing_state, True])
                    else:
                        self.replay_memory.append([DQNstates[-1], DQNmoves[-1], reward, losing_state, True])
                elif winner == 0: #draw
                    reward = self.draw_reward
                    drawing_state = game.get_state()
                    if player == 1:
                        self.replay_memory.append([DQNstates[-1], DQNmoves[-1], reward, drawing_state, True])
                    else:
                        self.replay_memory.append([DQNstates[-1], DQNmoves[-1], reward, drawing_state, True])
                    
                
            games_played_train += 1
            games_played_sync += 1
            if games_played_train == self.hpp.games_before_train:
                games_played_train = 0
                self.train(player = 1)
            if games_played_sync== self.hpp.games_before_sync:
                games_played_sync = 0
                if player == 1:
                    self.target_agent.model.set_weights(self.agent.model.get_weights())
                else:
                    self.target_agentO.model.set_weights(self.agentO.model.get_weights())
    def train(self, player = 1): #uses target nn
        
        replay_memory = self.replay_memory if player == 1 else self.replay_memoryO
        if self.hpp.sample_size < len(replay_memory):
            mini_batch = rd.sample(replay_memory, self.hpp.sample_size)
        else:
            mini_batch = replay_memory
        
        current_states = np.array([transition[0] for transition in mini_batch])
        current_qs_list = self.agent.model.predict(current_states) if player == 1 else self.agentO.model.predict(current_states)
        new_current_states = np.array([transition[3] for transition in mini_batch])
        future_qs_list = self.target_agent.model.predict(new_current_states) if player == 1 else self.target_agentO.model.predict(new_current_states)

        X = [] #entrée
        Y = [] #sorties attendues
        for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
            if self.hpp.allow_illegal_moves:
                maxfutureQ = np.max(future_qs_list[index])
            else:
                if done: #if new observation is end of game
                    maxfutureQ = 0
                else:
                    l=[future_qs_list[index][i] for i in self.game.get_valid_moves_from_state(new_observation)]
                    maxfutureQ = max(l)

            max_future_q = reward + self.hpp.gamma * maxfutureQ #le target


            current_qs = current_qs_list[index]
            current_qs[action] = max_future_q
            #current_qs[action] = (1 - self.hpp.Qlr) * current_qs[action] + self.hpp.Qlr * max_future_q

            X.append(observation)
            Y.append(current_qs) # étiquette
        if player == 1:
            self.agent.model.fit(np.array(X), np.array(Y), batch_size=self.hpp.batch_size, verbose=0, shuffle=True)
        else:
            self.agentO.model.fit(np.array(X), np.array(Y), batch_size=self.hpp.batch_size, verbose=0, shuffle=True)

    def self_train(self, n=1000, player = None):
        if player != None:
            self._reset_deques()
        games_played_train = 0 #games played after last train
        games_played_sync = 0 #games played after last sync
        for episode in range(n):
            self.agent.epoch += 1
            self.agentO.epoch += 1
            self.game = Game(self.h, self.w)
            states = [[], [], [], self.game.get_state()] #historique des 4 derniers états = file
            moves = [[], [], [], []] #coups menant aux états respectifs
            if episode%100 == 0:
                #print(int(100 * episode/n), "%")
                pass
            done = False
            
            while not done:
                turn = self.game.turn
                if turn == 1:
                    action = self.agent.get_move(states[-1], training = True)
                else:
                    action = self.agentO.get_move(states[-1], training = True)
                w = self.game.play_move(action)
                new_observation = self.game.get_state()
                #mese à jours des files
                #print("States : ", states)
                #print("Moves : ", moves)
                states.pop(0) #defiler
                states.append(new_observation) #enfiler

                moves.pop(0) #defiler
                moves.append(action) #enfiler
                done = self.game.is_game_over

                #verif si partie finie
                if self.game.is_game_over:
                    if w == 0:
                        state2 = states[-2]
                        move1 = moves[-2] #coup apres state 1, puis etat final
                        state1 = states[-3]
                        move2 = moves[-1] #coup après state 2, mene a letat final

                        if turn == 1: #alors 1 a gagné
                            self.replay_memoryO.append([state1, move1,self.draw_reward, new_observation,done])
                            self.replay_memory.append([state2, move2,self.draw_reward, new_observation,done])
                        else:
                            self.replay_memory.append([state1, move1,self.draw_reward, new_observation,done])
                            self.replay_memoryO.append([state2, move2,self.draw_reward, new_observation,done])
                    if w == 1: 
                        state1 = states[-3]
                        move1 = moves[-2] #coup après state1
                        state2 = states[-2]
                        move2 = moves[-1] #will get rewarded for this move
                        self.replay_memoryO.append([state1, move1, self.o_lossreward, new_observation,done])
                        self.replay_memory.append([state2, move2, self.x_winreward, new_observation,done])
                       
                    if w == -1:
                        state1 = states[-3]
                        move1 = moves[-2] #coup après state1
                        state2 = states[-2]
                        move2 = moves[-1] #will get rewarded for this move
                        self.replay_memoryO.append([state2, move2,self.o_winreward, new_observation,done])
                        self.replay_memory.append([state1, move1,self.x_lossreward, new_observation,done])

                else: #game not over, actually updating for old move
                    old_state = states[-3]
                    move_for_old_state = moves[-2]
                    if len(old_state) != 0:
                        if turn == -1:
                            self.replay_memory.append([old_state, move_for_old_state,0, new_observation,done])
                        else:
                            self.replay_memoryO.append([old_state, move_for_old_state,0, new_observation,done])
                            

            #entrainement fin de partie
            games_played_train += 1
            games_played_sync += 1
            if games_played_train == self.hpp.games_before_train:
                if player == None:
                    self.train(1)
                    self.train(-1)
                else:
                    self.train(player)
                games_played_train = 0
                #print("THE MEMORY FOR GAME FOR X")
                #print(self.replay_memory)
            if games_played_sync== self.hpp.games_before_sync:
                games_played_sync = 0
                self.target_agent.model.set_weights(self.agent.model.get_weights())
                self.target_agentO.model.set_weights(self.agentO.model.get_weights())
                
            

                
    def test_agent(self, n = 1000):
        #print("Testing QL Agent who has trained", self.agent.epoch, " Times")
        scores = []
        for _ in range(n):
            game = Game(self.h,self.w)
            
            while not(game.is_game_over):
                state = game.get_state()
                move = self.agent.get_move(state, training = False)
                w = game.play_move(move)
                if game.is_game_over: #partie finie
                    if w == 1:
                        scores.append(100)
                    elif w == -1:
                        scores.append(-100)
                    else:
                        scores.append(0)
                        
                else:
                    move = game.IA1()
                    w = game.play_move(move)
                    if game.is_game_over: #partie finie
                        if w == 1:
                            scores.append(100)
                        elif w == -1:
                            scores.append(-100)
                        else:
                            scores.append(0)
                            
        #print("Scores : ", scores)
        wr = len([i for i in scores if i>0])/n
        lr = len([i for i in scores if i<0])/n
        dr = 1-wr-lr
        #print("Winrate, lossrate, drawrate : ", wr, lr, dr)
        return scores, wr, lr, dr
    def test_agent_minmax(self, n = 1000, player = 1, pmax = 2):
        if player == 1:
            agent = self.agent
        else:
            agent = self.agentO

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
                    state = game.get_state()
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
    #modelX = tf.keras.models.load_model("DQN67_CNN_p2_1")
    modelX = 0
    h = 4
    w = 4
    ntest = 100 #tests
    steps = 1000 #epoch step 
    step_size = 1000 #number of trainings per step
    profondeur = 2
    
    #player = 1 #pour le différé
    #modelX = tf.keras.models.load_model("DQN_2X")
    trainer = Trainer(h,w, modelX = modelX)
    start = time.time()
    
    wr0, lr0, dr0 = trainer.test_agent_minmax(n=ntest, player = 1, pmax = 0)
    wr1, lr1, dr1 = trainer.test_agent_minmax(n=ntest, player = 1, pmax = 1)
    wr2, lr2, dr2 = trainer.test_agent_minmax(n=ntest, player = 1, pmax = 2)
    wr00, lr00, dr00 = trainer.test_agent_minmax(n=ntest, player = -1, pmax = 0)
    wr11, lr11, dr11 = trainer.test_agent_minmax(n=ntest, player = -1, pmax = 1)
    wr22, lr22, dr22 = trainer.test_agent_minmax(n=ntest, player = -1, pmax = 2)
    
    end = time.time()
    print(" First test (no training) for 3*", ntest, " tests : ", end-start, " seconds")
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
    for i in range(steps):
        print(100*i/steps, "%")
        start = time.time() #time
        trainer.train_with_minmax(n = step_size, player = 1, profondeur_max = profondeur)
        trainer.train_with_minmax(n = step_size, player = -1, profondeur_max = profondeur)
        #player = -player
        end = time.time()
        print("For step : ", i, "time for ", step_size, " training is : ", end-start, "s")
        epochs.append(step_size*(i+1))
        start = time.time()
        wr0, lr0, dr0 = trainer.test_agent_minmax(n=ntest, player = 1, pmax = 0)
        wr1, lr1, dr1 = trainer.test_agent_minmax(n=ntest, player = 1, pmax = 1)
        wr2, lr2, dr2 = trainer.test_agent_minmax(n=ntest, player = 1, pmax = 2)
        wr00, lr00, dr00 = trainer.test_agent_minmax(n=ntest, player = -1, pmax = 0)
        wr11, lr11, dr11 = trainer.test_agent_minmax(n=ntest, player = -1, pmax = 1)
        wr22, lr22, dr22 = trainer.test_agent_minmax(n=ntest, player = -1, pmax = 2)
        end = time.time()
        print(" i = ", i, "for 3*", ntest, " tests : ", end-start, " seconds")
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
        
    wrates = wrates2
    lrates = lrates2
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