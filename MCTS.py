import numpy as np
import random as rd
import matplotlib.pyplot as plt
from puissance4 import Game
import copy as copy
import pickle as pickle
import time as time

class Node:
    def __init__(self, state, turn, parent = None, coup_menant_au_node = None):

        
        self.state = state
        self.val = 0 #value for player 1
        self.vis = 0 #how many times node has been visited

        self.turn = turn #1 or -1

        self.parent = parent #parent node
        self.children = [] #child nodes

        self.coup_menant_au_node = coup_menant_au_node #pratique
        
        self.is_expandable = True
        self.winner = None

        self.uct = None
    def get_uct(self, player = 1):
        if self.uct != None:
            return self.uct
        if self.vis == 0:
            print("UTC sans être visité")
            return 10
        true_val = -self.turn*player * self.val #valeur toujours pour le joueur 1, la valeur pour le joueur qui a joué c'est true_val
        return true_val/self.vis +np.sqrt(2)*np.sqrt(np.log(self.parent.vis)/self.vis)
    def get_result(self):
        return self.val/self.vis
    
    
class MCTS:
    def __init__(self,h = 6, w = 7, game = 0, n_iteration = 1000):
        self.h = h
        self.w = w
        self.n_iteration = n_iteration
        #game only useful to know initial conditions
        self.game = game #le puissance 4 (ou autre)
        if isinstance(game, int):
            self.game = Game(h = self.h, w= self.w)

        self.player = self.game.turn #pour le player perspective
        self.root_node = Node(self.game.get_state(), self.game.turn)
        self.root_node.uct = 9999 #fully expand root node always
        
        
    def do_process(self, n_iteration = None):
        if n_iteration == None:
            n_iteration = self.n_iteration
        for i in range(n_iteration):
            node = self.select_node()
            new_node = self.expand_node(node)
            winner = self.simulate(new_node)
            self.backpropagate(new_node,winner)

        move_val_dic =  {} #à titre informatif

        root_children = [child for child in self.root_node.children]
        for child in root_children:
            move_val_dic[child.coup_menant_au_node] = (child.val, child.vis)
            move_val_dic = dict(sorted(move_val_dic.items()))
        best_move = self.get_best_move(root_children)
        return best_move, move_val_dic

    def get_best_move(self, root_children): #à appeler à la toute fin
        best_child = max(root_children, key = lambda x : x.val/x.vis) #Robust child
        best_move = best_child.coup_menant_au_node
        return best_move

    def select_node(self):
        #select node to expand 
        node = self.root_node
        while not(node.is_expandable):
            if node.children == []:
                return node #terminal state
            node = max(node.children, key = lambda x : x.get_uct(player = self.player))
        
        #print("NODE SELECTED : ", node.state)
        return node

    def get_qvals(self,move_val_dic, invalid_move_value = 0):
        # returns qvales of valid moves (0 for non valid moves)
        rep = invalid_move_value * np.ones((self.w,))
        total_vis = self.root_node.vis
        for move in move_val_dic.keys():
            (val,vis) = move_val_dic[move]
            rep[move] = val/vis

        return rep
    def get_val(self, move_val_dic):
        q = self.get_qvals(move_val_dic, invalid_move_value=-1000)
        maxq = max(q)
        if maxq == -1000:
            print("problème A4395")
        return maxq

    def get_policyvals(self, move_val_dic, invalid_move_value = 0):
        # returns qvales of valid moves (0 for non valid moves)
        rep = invalid_move_value * np.ones((self.w,))
        total_vis = self.root_node.vis
        for move in move_val_dic.keys():
            (val,vis) = move_val_dic[move]
            rep[move] = vis/total_vis

        return rep
    def get_probabilities(self, move_val_dic, invalid_move_value = 0):
        rep = invalid_move_value * np.ones((self.w,))
        total_vis = self.root_node.vis
        for move in move_val_dic.keys():
            (val, vis) = move_val_dic[move]
            rep[move] = vis/total_vis

        return rep
    def expand_node(self, node):
        #expand : get random move, get new child node return new child node (then simulation and backprop)
        if not(node.is_expandable): #peut pas etre étendu, la simulation sera directe
            return node
            
        #print("NODE EXPANDED : ", node.state)
        state, turn = node.state, node.turn
        board = state.reshape(self.h,self.w)
        available_moves = self.game.get_valid_moves_from_state(state)

        explored_moves = [child.coup_menant_au_node for child in node.children]
        unexplored_moves = [m for m in available_moves if not(m in explored_moves)]

        if unexplored_moves == []:
            print(node.state)
            print(node.is_expandable)
            print(node.parent)
            print([child.state for child in node.children])
            
            
        move = rd.choice(unexplored_moves)
        winner, new_board = self.game.simulate_move(move, board, turn)
        new_state = new_board.reshape((self.h*self.w,))
        new_node = Node(new_state, -turn, parent = node, coup_menant_au_node=move)
        
        if winner != None:
            new_node.is_expandable = False #terminal state
            new_node.winner = winner
            
        node.children.append(new_node)
        if len(node.children)>= len(available_moves):
            node.is_expandable = False

        
        return new_node

    def simulate(self,node):
        state, turn = copy.deepcopy(node.state), node.turn
        game = Game(h = self.h, w= self.w)
        game.board = state.reshape(self.h,self.w)
        game.turn = turn

        if node.winner != None: #terminal state, no sim to be done
            return node.winner

        while not(game.is_game_over):
            move = rd.choice(game.get_valid_moves())
            winner = game.play_move(move)

        return winner #1, 0 ou -1

    def backpropagate(self, node, winner):
        node.vis += 1
        if self.player == 1: #valeur dépend de la perspective
            node.val += winner
        if self.player == -1:
            node.val -= winner
        if node.parent != None:
            self.backpropagate(node.parent,winner)
        
def create_training_batch(h=4, w=4, n_iteraton = 1000, batch_size = 100, invalid_move_value = 0):
    #batch is list : [[state1, state2, ..., staten], [qvals 1, ...]]
    #state is bitmap np array (h,w,3), qvals is np array (w,)
    #returns [states, corresponding q values]
    batch = []
    state_batch = [] #inputs
    qval_batch = [] #etiquettes
    size = 0
    random_play_batches = batch_size//4

    mcts_play_batches = batch_size - random_play_batches
    randomness = 0.05 #randomness for mcts play batches

    game = Game(h = h, w = w)
    for i in range(random_play_batches):
        if i%10 == 0:
            print(int(100*i/batch_size), "%")
        state = game.get_state_CNN_bitmap()
        mcts = MCTS(h = h, w = w, game = game)
        best_move, move_val_dic = mcts.do_process(n_iteration = n_iteration)
        qvals = mcts.get_qvals(move_val_dic, invalid_move_value = invalid_move_value)
        
        batch.append([state, qvals])
        state_batch.append(state)
        qval_batch.append(qvals)
        size += 1

        move = game.random_IA()
        winner = game.play_move(move)

        if winner != None:
            game = Game(h = h, w = w)

    for i in range(mcts_play_batches):
        if i%10 == 0:
            print(int(100*(i+random_play_batches)/batch_size), "%")
        state = game.get_state_CNN_bitmap()
        mcts = MCTS(h = h, w = w, game = game)
        best_move, move_val_dic = mcts.do_process(n_iteration = n_iteration)
        qvals = mcts.get_qvals(move_val_dic, invalid_move_value = invalid_move_value)

        batch.append([state, qvals])
        state_batch.append(state)
        qval_batch.append(qvals)
        size += 1

        r = rd.random()
        if r< randomness:
            move = game.random_IA()
            winner = game.play_move(move)
        else:
            winner = game.play_move(best_move)

        if winner != None:
            game = Game(h = h, w = w)
    return [state_batch, qval_batch]
def create_training_batch_policy_val(h=4, w=4, n_iteraton = 1000, batch_size = 100, invalid_move_value = 0, doublons = False):
    #batch is list : [[state1, state2, ..., staten], [policy1,policy2...], [valu1, value2,....]]
    #state is bitmap np array (h,w,3), policy is probability array (w,), value is [int] (1,)
    #invalid_move_value : for the policy probability, should be 0
    batch = []
    states = [] #inputs
    policies = [] #etiquettes
    values = []
    hashlist = [] #hash des states, pour retirer doublons
    batch_dic = {} #batch_dic[state] = 
    size = 0
    random_play_batches = batch_size//2

    mcts_play_batches = batch_size - random_play_batches
    randomness = 0.3 #randomness for mcts play batches

    game = Game(h = h, w = w)
    start = time.time()
    for i in range(random_play_batches):
        if i == 2:
            end = time.time()
            print("For 2 first : ", end - start, "s")
            estimated_time = (end-start) * batch_size/2
            print("Estimated time : ", estimated_time)
        if i%100 == 0:
            print(int(100*i/batch_size), "%")
        state = game.get_state_CNN_bitmap()
        hashstate = game.get_hash()
        
        mcts = MCTS(h = h, w = w, game = game)
        

        if not(doublons):
            if not(hashstate in hashlist):
                best_move, move_val_dic = mcts.do_process(n_iteration = n_iteration)
                policy = mcts.get_policyvals(move_val_dic, invalid_move_value = invalid_move_value)
                val = game.turn * mcts.get_val(move_val_dic)
                states.append(state)
                hashlist.append(hashstate)
                policies.append(policy)
                values.append([val])
                size += 1
        else:
            best_move, move_val_dic = mcts.do_process(n_iteration = n_iteration)
            policy = mcts.get_policyvals(move_val_dic, invalid_move_value = invalid_move_value)
            val = mcts.get_val(move_val_dic)
            states.append(state)
            policies.append(policy)
            values.append([val])
            size += 1

        move = game.random_IA()
        winner = game.play_move(move)

        if winner != None:
            game = Game(h = h, w = w)

    for i in range(mcts_play_batches):
        if i%10 == 0:
            print(int(100*(i+random_play_batches)/batch_size), "%")
        state = game.get_state_CNN_bitmap()
        hashstate = game.get_hash()
        mcts = MCTS(h = h, w = w, game = game)
        
        best_move = -1
        if not(doublons):
            if not(hashstate in hashlist):
                best_move, move_val_dic = mcts.do_process(n_iteration = n_iteration)
                policy = mcts.get_policyvals(move_val_dic, invalid_move_value = invalid_move_value)
                val = game.turn*mcts.get_val(move_val_dic)
                states.append(state)
                hashlist.append(hashstate)
                policies.append(policy)
                values.append([val])
                size += 1
        else:
            best_move, move_val_dic = mcts.do_process(n_iteration = n_iteration)
            policy = mcts.get_policyvals(move_val_dic, invalid_move_value = invalid_move_value)
            val = game.turn * mcts.get_val(move_val_dic)
            states.append(state)
            policies.append(policy)
            values.append([val])
            size += 1
            
        r = rd.random()
        if r< randomness:
            move = game.random_IA()
            winner = game.play_move(move)
        else:
            if best_move == -1:
                best_move, move_val_dic = mcts.do_process(n_iteration = n_iteration)
            winner = game.play_move(best_move)

        if winner != None:
            #stocker l'état final?
            game = Game(h = h, w = w)
    print("size of batch : ", size)
    return [states, policies, values]
def create_training_batch_policy_val2(h=4, w=4, n_iteraton = 1000, batch_size = 100, invalid_move_value = 0, doublons = False):
    #batch is list : [[state1, state2, ..., staten], [policy1,policy2...], [valu1, value2,....]]
    #state is bitmap np array (h,w,3), policy is probability array (w,), value is [int] (1,)
    #invalid_move_value : for the policy probability, should be 0
    batch = []
    states = [] #inputs
    policies = [] #etiquettes
    values = []
    hashlist = [] #hash des states, pour retirer doublons
    batch_dic = {} #batch_dic[state] = 
    size = 0
    random_play_batches = 0

    mcts_play_batches = batch_size - random_play_batches
    randomness = 0.05 #randomness for mcts play batches

    game = Game(h = h, w = w)
    start = time.time()
    for i in range(random_play_batches):
        if i == 2:
            end = time.time()
            print("For 2 first : ", end - start, "s")
            estimated_time = (end-start) * batch_size/2
            print("Estimated time : ", estimated_time)
        if i%100 == 0:
            print(int(100*i/batch_size), "%")
        state = game.get_state_CNN_bitmap()
        hashstate = game.get_hash()
        
        mcts = MCTS(h = h, w = w, game = game)
        

        if not(doublons):
            if not(hashstate in hashlist):
                best_move, move_val_dic = mcts.do_process(n_iteration = n_iteration)
                policy = mcts.get_policyvals(move_val_dic, invalid_move_value = invalid_move_value)
                val = game.turn * mcts.get_val(move_val_dic)
                states.append(state)
                hashlist.append(hashstate)
                policies.append(policy)
                values.append([val])
                size += 1
        else:
            best_move, move_val_dic = mcts.do_process(n_iteration = n_iteration)
            policy = mcts.get_policyvals(move_val_dic, invalid_move_value = invalid_move_value)
            val = mcts.get_val(move_val_dic)
            states.append(state)
            policies.append(policy)
            values.append([val])
            size += 1

        move = game.random_IA()
        winner = game.play_move(move)

        if winner != None:
            game = Game(h = h, w = w)

    number_of_moves = 0
    for i in range(mcts_play_batches):
        if i%10 == 0:
            print(int(100*(i+random_play_batches)/batch_size), "%")
        state = game.get_state_CNN_bitmap()
        hashstate = game.get_hash()
        mcts = MCTS(h = h, w = w, game = game)
        
        best_move = -1
        if not(doublons):
            if not(hashstate in hashlist):
                if number_of_moves> 6:
                    best_move, move_val_dic = mcts.do_process(n_iteration = n_iteration)
                    policy = mcts.get_policyvals(move_val_dic, invalid_move_value = invalid_move_value)
                    val = game.turn*mcts.get_val(move_val_dic)
                    states.append(state)
                    hashlist.append(hashstate)
                    policies.append(policy)
                    values.append([val])
                    size += 1
        else:
            best_move, move_val_dic = mcts.do_process(n_iteration = n_iteration)
            policy = mcts.get_policyvals(move_val_dic, invalid_move_value = invalid_move_value)
            val = game.turn * mcts.get_val(move_val_dic)
            states.append(state)
            policies.append(policy)
            values.append([val])
            size += 1
            
        number_of_moves += 1
        r = rd.random()
        if r< randomness:
            move = game.random_IA()
            winner = game.play_move(move)
        else:
            if best_move == -1:
                best_move, move_val_dic = mcts.do_process(n_iteration = 1000)
            winner = game.play_move(best_move)

        if winner != None:
            #stocker l'état final?
            number_of_moves == 0
            game = Game(h = h, w = w)
    print("size of batch : ", size)
    return [states, policies, values]
def remove_doublons(h, w, batch):
    hashstatelist = []
    new_batch = [[], [], []]
    game = Game(h,w)

    for i in range(len(batch[0])):
        statei = batch[0][i]
        board = game.convert_state_to_board_bitmap(statei)
        hashstate = game.get_hash(board = board)
        if not(hashstate in hashstatelist):
            hashstatelist.append(hashstate)
            new_batch[0].append(statei)
            new_batch[1].append(batch[1][i])
            new_batch[2].append(batch[2][i])
    return new_batch
def test_MCTS(h =4, w = 4, ntest = 100, n_iteration=2000, MCTS_player = 1, prof = 1):
    wins = 0
    losses = 0
    draws = 0
    for i in range(ntest):
        if i%10 == 0:
            print(int(100*i/ntest),"%")
        game = Game(h = h, w = w)
        while not game.is_game_over:
            if game.turn ==MCTS_player:
                mcts = MCTS(h=h,w=w,game = game, n_iteration = n_iteration)
                move, _ = mcts.do_process()
                winner = game.play_move(move)
            else:
                move, score = game.minmax(pmax = prof)
                winner = game.play_move(move)
                
        if winner == MCTS_player:
            wins += 1
        if winner == -MCTS_player:
            losses += 1
        if winner == 0:
            draws += 1
    return wins/ntest, losses/ntest, draws/ntest
def test_MCTS_self(h = 4, w = 4, ntest = 100, n_iteration = 2000, MCTS_player = 1):
    wins = 0
    losses = 0
    draws = 0
    for i in range(ntest):
        if i%10 == 0:
            print(int(100*i/ntest),"%")
        game = Game(h = h, w = w)
        while not game.is_game_over:

            mcts = MCTS(h=h,w=w,game = game, n_iteration = n_iteration)
            move, _ = mcts.do_process()
            winner = game.play_move(move)

                
        if winner == MCTS_player:
            wins += 1
        if winner == -MCTS_player:
            losses += 1
        if winner == 0:
            draws += 1
    return wins/ntest, losses/ntest, draws/ntest
if __name__ == "__main__":
    h = 6
    w = 7

    n = 10 #itérations de MCTS/précision
    prof = 5
    player = 1 #quel joueur joué par l'algo MCTS
    ntest = 100
    #wr,lr,dr = test_MCTS_self(h =h, w = w, ntest = ntest, n_iteration=n)
    wr,lr,dr = test_MCTS(h =h, w = w, ntest = ntest, n_iteration=n, MCTS_player = player, prof = prof)
    
    print(wr,lr,dr)
    
    
    n_iteration = 10000 #iterations for batch creation
    batch_size = 1 #size of batch creation
    invalid_move_value = 0 #what an invalid move qvalue is replaced with

    batch = create_training_batch_policy_val2(h=h, w=w, n_iteraton = n_iteration,
                                  batch_size = batch_size,
                                  invalid_move_value = invalid_move_value
                                  )
    
    
    
    #file name to save
    file_name = "batchTOT" + str(h)+str(w) + "_" + str(n_iteration) + "_" + str(invalid_move_value) + ".pkl"
    
    #open file to update the batch
    open_file = open(file_name, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()

    #append => careful doublons
    batch[0] = batch[0] + loaded_list[0] #states append
    batch[1] = batch[1] + loaded_list[1] #
    batch[2] = batch[2] + loaded_list[2]
    #save
    open_file = open(file_name, "wb")
    pickle.dump(batch, open_file)
    open_file.close()
    #open
    open_file = open(file_name, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close() 

