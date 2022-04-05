# -*- coding: utf-8 -*-
import random as rd
import tkinter as tk
import time as time
import copy as copy
from puissance4 import Game

import DQNpuissance4_target as DQN
import DQNpuissance4_CNN as DQNCNN
import Policy_value_MCTS_net as PolicyVal

from MCTS_Trainer import MCTS_trainer
from MCTS import MCTS

import tensorflow as tf
import time as time #pour sleep
import QLpuissance4 as QL
import pickle as pickle #load and save dics



H_MAX = 600 #pixels
#WMAX = w/h Hmax

class Play_moves_dialog: #fenetre de dialogue
    def __init__(self, parent):
        self.top = tk.Toplevel(parent)

        tk.Label(self.top, text="Coups : ", fg="green").pack(side="top", anchor="nw")
        b = tk.Button(self.top, text="OK", command=self.get_input)
        b.pack(pady=5, side="right")
        self.entry_mtp = tk.Entry(self.top, bg='green')
        self.entry_mtp.pack(padx=5,side="left")
        self.parent=parent

    def get_input(self):
        mtp=self.entry_mtp.get()
        self.parent.set_moves_to_play(mtp, self.top)
        self.top.destroy()
class Simulate_game_dialog: #fenetre de dialogue
    def __init__(self, parent):
        self.top = tk.Toplevel(parent)
        self.parent=parent
        #bouton submit
        b = tk.Button(self.top, text="OK", command=self.get_input)
        b.pack(pady=5, side="bottom")
        #fonction 1
        self.framef1 = tk.Frame(self.top)
        self.framef1.pack(side = "top")
        label1 = tk.Label(self.framef1,
                    text = "Fonction 1  :  ",
                    font = ("Times New Roman", 10), 
                    padx = 0, pady = 0, bg = "white")
        label1.pack(side = "left")
        self.f1_input_variable = tk.StringVar(self.top)
        self.f1_input_variable.set("MCTS")
        self.IA_menu_deroulant1=tk.OptionMenu(self.framef1 , self.f1_input_variable, "MCTS","minmax", "QL", "random", "MCTSDQN", "DQNDense", "DQNCNN")
        self.IA_menu_deroulant1.config(bg='green', width=8, height=1)
        self.IA_menu_deroulant1.pack(side = "left")
        #fonction 2
        self.framef2 = tk.Frame(self.top)
        self.framef2.pack(side = "top")
        label2 = tk.Label(self.framef2,
                    text = "Fonction 2  :  ",
                    font = ("Times New Roman", 10), 
                    padx = 0, pady = 0, bg = "white")
        label2.pack(side = "left")
        self.f2_input_variable = tk.StringVar(self.top)
        self.f2_input_variable.set("minmax")
        self.IA_menu_deroulant2=tk.OptionMenu(self.framef2 , self.f2_input_variable, "MCTS","minmax", "QL", "random", "MCTSDQN", "DQNDense", "DQNCNN")
        self.IA_menu_deroulant2.config(bg='green', width=8, height=1)
        self.IA_menu_deroulant2.pack(side = "left")

        #combien de fois
        self.framew = tk.Frame(self.top)
        self.framew.pack(side = "top")
        labelw = tk.Label(self.framew,
                    text = "Combien de simulations : ",
                    font = ("Times New Roman", 10), 
                    padx = 0, pady = 0, bg = "white")
        labelw.pack(side = "left")
        width_default= tk.StringVar(self.top)
        width_default.set("1")
        self.width_input = tk.Spinbox(self.framew, from_=1, to=200,textvariable=width_default)
        self.width_input.pack(side = "bottom")


    def get_input(self):
        self.parent.sim_game_name_func1 = self.f1_input_variable.get() 
        self.parent.sim_game_name_func2 = self.f2_input_variable.get()
        self.parent.simulate_game_ngames = int(self.width_input.get())
        self.parent.simulate_game_1()
        self.top.destroy()
        
class Interface(tk.Tk):
    def __init__(self): #page de menu
    #si aucun param, on ouvre la fenetre menu et on laisse choisir
        tk.Tk.__init__(self)

        
        self.Main_Frame = tk.Frame(self)
        self.Main_Frame.pack()
          
        #HAUTEUR
        self.frameh = tk.Frame(self.Main_Frame)
        self.frameh.pack(side = "top")
        labelh = tk.Label(self.frameh,
                    text = "Entrez la hauteur        :  ",
                    font = ("Times New Roman", 10), 
                    padx = 0, pady = 0, bg = "white")
        labelh.pack(side = "left")
        height_default= tk.StringVar(self)
        height_default.set("6")
        self.height_input = tk.Spinbox(self.frameh, from_=4, to=20,textvariable=height_default)
        self.height_input.pack(side = "bottom")
        
        #LARGEUR
        self.framew = tk.Frame(self.Main_Frame)
        self.framew.pack(side = "top")
        labelw = tk.Label(self.framew,
                    text = "Entrez la largeur         :  ",
                    font = ("Times New Roman", 10), 
                    padx = 0, pady = 0, bg = "white")
        labelw.pack(side = "left")
        width_default= tk.StringVar(self)
        width_default.set("7")
        self.width_input = tk.Spinbox(self.framew, from_=4, to=20,textvariable=width_default)
        self.width_input.pack(side = "bottom")
        
        #PROFONDEUR
        self.framep = tk.Frame(self.Main_Frame)
        self.framep.pack(side = "top")
        labelp = tk.Label(self.framep,
                    text = "Entrez la profondeur :  ",
                    font = ("Times New Roman", 10), 
                    padx = 0, pady = 0, bg = "white")
        labelp.pack(side = "left")
        profondeur_default= tk.StringVar(self)
        profondeur_default.set("4")
        self.profondeur_input = tk.Spinbox(self.framep, from_=0, to=12,textvariable=profondeur_default)
        self.profondeur_input.pack(side = "bottom")
        #ITERATIONS MONTECARLO
        self.framenit = tk.Frame(self.Main_Frame)
        self.framenit.pack(side = "top")
        labelnit = tk.Label(self.framenit,
                    text = "MCTS iterations       :  ",
                    font = ("Times New Roman", 10), 
                    padx = 0, pady = 0, bg = "white")
        labelnit.pack(side = "left")
        nit_default= tk.StringVar(self)
        nit_default.set("10")
        self.nit_input = tk.Spinbox(self.framenit, from_=10, to=100000,textvariable=nit_default)
        self.nit_input.pack(side = "bottom")


        
        #BOUTON JOUER
        playbutton = tk.Button(self.Main_Frame, text = "Jouer", command = self.jouer, bg = "red", width = 10, height = 1)
        playbutton.pack(side = "bottom")



    def jouer(self): #lance le vrai jeu avec les param du menu en entré
        self.h = int(self.height_input.get())
        self.w = int(self.width_input.get())
        self.profondeur = int(self.profondeur_input.get()) #profondeur minmax
        self.mctsnit = int(self.nit_input.get())
        self.IA_mode = "Sandbox" #Sandbox ou MCTS
        print("Prof : ",self.profondeur, " MCTS it : ", self.mctsnit )
        self.initgame()
        
    def initgame(self): #init du vrai jeu
        #destruction menu principal
        self.Main_Frame.destroy()
        self.puissance4 = Game(self.h,self.w, profondeur_max = self.profondeur)

        self.dh = H_MAX/self.h
        #W_MAX = H_MAX * self.w/self.h
        self.dw = self.dh

        #POUR LE BATCH
        file_name = 'batch67_10000_0.pkl'
        open_file = open(file_name, "rb")
        self.batch = pickle.load(open_file) #le batch qui peut être parcouru
        open_file.close()
        self.i_batch = 0 #indice du batch à 
        
        #MCTS policy and value net
        modelP, modelV = self.get_models_MCTS_NN()

        nit = 100
        self.MCTStrainer = MCTS_trainer(h = self.h, w = self.w, n_iteration = nit, modelP = modelP, modelV=modelV)#used to play and iterate
        self.trainer_policyval = PolicyVal.Trainer(self.h, self.w, modelP =modelP, modelV =modelV ) #used to get predictions
       
        #IA DQN
        modelX, modelO = self.get_models_DQN_normal()
        #IA DQN CNN
        modelXcnn, modelOcnn = self.get_models_DQN_CNN()
        #IA QL
        Q_dic = self.get_dic_QL()

        
        self.trainerDQL = DQN.Trainer(self.h, self.w, modelX = modelX, modelO = modelO)
        self.trainerCNN = DQNCNN.Trainer(self.h, self.w, modelX = modelXcnn, modelO = modelOcnn)
        self.trainerQL = QL.Trainer(self.h, self.w, Q = Q_dic)

        
        #barre d'outil
        self.barreOutils=tk.Frame(self)
        self.barreOutils.pack(side="top")
        self.barreOutils.config(bg='green')
        boutonNewgame = tk.Button(self.barreOutils, text='Nouvelle partie', width=15, command=self.nouvellePartie)
        boutonNewgame.pack(side="left",padx=5,pady=5)
        boutonQuitter = tk.Button(self.barreOutils, text='Quitter', width=15,command=self.destroy)
        boutonQuitter.pack(side="left",padx=5,pady=5)
        #recuperer l'historique des coups
        boutonHistorique = tk.Button(self.barreOutils, text = "Afficher Historique", width = 20, command = self.afficher_historique)
        boutonHistorique.pack(side="left",padx=5,pady=5)
        #jouer les coups moves
        boutonPlaymoves = tk.Button(self.barreOutils, text = "Play Moves", width = 15, command = self.play_moves)
        boutonPlaymoves.pack(side="left",padx=5,pady=5)
        #simuler un 1v1
        boutonSimulategame = tk.Button(self.barreOutils, text = "Simulate Game", width = 15, command = self.simulate_game)
        boutonSimulategame.pack(side="left",padx=5,pady=5)
        #observation du batch
        #boutonViewbatch = tk.Button(self.barreOutils, text = "View Batch", width = 15, command = self.view_batch)
        #boutonViewbatch.pack(side="left",padx=5,pady=5)
        #Frame + canvas du jeu lui meme
        self.frameCan = tk.Frame(self)
        self.frameCan.pack(side='top')
        self.canvas = tk.Canvas(self.frameCan,width=H_MAX * self.w/self.h,height=H_MAX,bg='white')
        self.canvas.bind("<Button-1>",self.onClick_souris) # <Button-1> : Bouton gauche de la souris 
        self.canvas.pack()
        
        for i in range(1,self.h):
            self.canvas.create_line(0, i*self.dh, H_MAX * self.w/self.h, i*self.dh )
        for j in range(1,self.w):
            self.canvas.create_line(j*self.dw, 0, j*self.dw, H_MAX)
        
        self.frameButton = tk.Frame(self)
        self.frameButton.pack(side='bottom')
        self.liste_cases = [self.w*[0] for _ in range(self.h)]
        #Création de la liste des cases pour y tracer les formes
        for i in range(self.h):
            for j in range(self.w):
                self.liste_cases[i][j] = (j*self.dw + 1, i*self.dh + 1, (j+1)*self.dw-1, (i+1) * self.dh-1)

        #BOUTONS DU BAS : IAS
        if self.IA_mode == "Sandbox":
            boutonIArd = tk.Button(self.frameButton, text='IA random', width=12, command=self.IArandom)
            boutonIArd.pack(side="left",padx=5,pady=5)
            boutonMCTS = tk.Button(self.frameButton, text='MCTS', width=12,command=self.play_MCTS)
            boutonMCTS.pack(side="left",padx=5,pady=5)
            boutonIAminmax = tk.Button(self.frameButton, text='IA minmax', width=12,command=self.IA_minmax)
            boutonIAminmax.pack(side="left",padx=5,pady=5)
            
            
            boutonDQN= tk.Button(self.frameButton, text='DQN', width=12,command=self.DQN_normal)
            boutonDQN.pack(side="left",padx=5,pady=5)
            boutonDQN1= tk.Button(self.frameButton, text='DQN CNN', width=12,command=self.DQN_CNN)
            boutonDQN1.pack(side="left",padx=5,pady=5)
            boutonDQNapprentice= tk.Button(self.frameButton, text='Policy Val net', width=12,command=self.DQN_Policyval)
            boutonDQNapprentice.pack(side="left",padx=5,pady=5)
            
            boutonQL= tk.Button(self.frameButton, text='QL', width=12, command=self.play_QL)
            boutonQL.pack(side="left",padx=5,pady=5)
            #PLAY MCTS
            boutonMCTSpolicyval= tk.Button(self.frameButton, text='MCTS Policy/Value NN', width=20,command=self.MCTS_Policyval)
            boutonMCTSpolicyval.pack(side="left",padx=5,pady=5)

        elif self.IA_mode == "MCTS":
            if self.human_player == "B":
                self.play_MCTS()
        #pour la simulation de parties
        self.play_func_dic = {"minmax" : self.IA_minmax, "random" : self.IArandom,
                                 "MCTS" : self.play_MCTS, "QL" : self.play_QL, "MCTSDQN" : self.MCTS_Policyval,
                                 "DQNDense" :self.DQN_normal, "DQNCNN" : self.DQN_CNN
                                 }
    def tracer(self,forme,case):
        #Trace la forme dans la case, rond ou croix
        if forme=='rond':
            self.canvas.create_oval(*(self.liste_cases[case]))
        else:
            self.canvas.create_line(*(self.liste_cases[case]))
            self.canvas.create_line(*(self.liste_cases_opposee[case]))
        self.update()
    
    def nouvellePartie(self):
        print("Partie recommence...")
        self.puissance4 = Game(self.h, self.w, profondeur_max = self.profondeur)
        self.repaint()
    def win(self):
        self.canvas.create_rectangle(0,0,600,480, fill = "green", outline = "blue")
    def effacer(self,i,j):
        #vide la case
        self.canvas.create_rectangle(*(self.liste_cases[i][j]),fill='white',outline='white')
    
    def onClick_souris(self,event):
        x=event.x
        # Sur quelle case a-t-on cliqué ?
        if self.puissance4.is_game_over:
            return "Partie finie"
        move = int(x//self.dw)
        if move >= self.w:
            print("too far")
            return "too far"
        
        w = self.puissance4.play_move(move)
        if w != None:
            if w == 1:
                print("Red wins")
            elif w == -1:
                print("Blue wins")
            else:
                print("Draw")
        self.repaint()
        self.update()
        if self.IA_mode == "MCTS":
            if w==None:
                self.play_MCTS()
        


        
                          
    def repaint(self):
        board = self.puissance4.board
        for i in range(self.h):
            for j in range(self.w):
                if board[i][j] == 1:
                    self.canvas.create_oval(*(self.liste_cases[i][j]), fill = "red")
                elif board[i][j] == -1:
                    self.canvas.create_oval(*(self.liste_cases[i][j]), fill = "blue")
                else:
                    self.canvas.create_oval(*(self.liste_cases[i][j]), fill = "white", outline="")
                        

    def afficher_historique(self):
        print("Historique des coups : ", self.puissance4.moves_played)
        print("Board : ", self.puissance4.board)
        print("Board flattened ", self.puissance4.get_state())
    
    #LES IAS
    def IArandom(self):
        move = self.puissance4.random_IA()
        w = self.puissance4.play_move(move)
        self.repaint()
        if self.puissance4.is_game_over:
            print("Winner is ", w)
            self.repaint()
    def IA_minmax(self):
        move,score = self.puissance4.minmax(pmax = self.profondeur)
        print("Le score (minmax) : ", score)
    
        w = self.puissance4.play_move(move)
        self.repaint()
        if self.puissance4.is_game_over:
            print("Winner is ", w)
            self.repaint()
    def play_MCTS(self, n_iteration = None):
        if n_iteration == None:
            n_iteration = self.mctsnit
        mcts = MCTS(self.h, self.w, self.puissance4)
        best_move, move_dic = mcts.do_process(n_iteration = n_iteration)
        
        print("MCTS values : ", move_dic)
        
        w = self.puissance4.play_move(best_move)
        self.repaint()
        if self.puissance4.is_game_over:
            print("Winner is ", w)
            self.repaint()
    #PARTIEDQN
    def DQN_CNN(self):
        state = self.puissance4.get_state_CNN()
        turn = self.puissance4.turn

        
        if turn ==1:
            move = self.trainerCNN.agent.get_move(state, training = False)
            print("Qs for CNN net (player 1) : ", self.trainerCNN.agent.get_qs(state))
        else:
            move = self.trainerCNN.agentO.get_move(state, training = False)
            print("Qs for CNN net (player 2) : ", self.trainerCNN.agentO.get_qs(state))
        w = self.puissance4.play_move(move)
        self.repaint()
        if self.puissance4.is_game_over:
            print("Winner is ", w)
            self.repaint()

    def DQN_Policyval(self):
        state = self.puissance4.get_state_CNN_bitmap()
        val = self.trainer_policyval.get_value(state)
        pol = self.trainer_policyval.get_policy(state)
        print("Value is : ", val)
        print("Policy is : ", pol)
        
    def show_info_DQN3(self, state = []):
        for layer in self.trainerV3.agent.model.layers:
            weights = layer.get_weights()
            print(weights)# list of numpy arrays

    def DQN_normal(self):
        state = self.puissance4.get_state()
        turn = self.puissance4.turn
        if turn ==1:
            print(self.trainerDQL.agent.get_qs(state))
            move = self.trainerDQL.agent.get_move(state, training = False)
        else:
            print(self.trainerDQL.agentO.get_qs(state))
            move = self.trainerDQL.agentO.get_move(state, training = False)
        w = self.puissance4.play_move(move)
        self.repaint()
        if self.puissance4.is_game_over:
            print("Winner is ", w)
            self.repaint()
    def play_QL(self):
        state = self.puissance4.get_hash()
        if not(state in self.trainerQL.agent.Q.keys()): #état du jeu jamais exploré
            print("Etat : ", state, " jamais exploré")
        else:
            print("Q vals : ", self.trainerQL.agent.Q[state])
        
        move = self.trainerQL.agent.get_move(state, training = False)
        
        w = self.puissance4.play_move(move)
        
        self.repaint()
        if self.puissance4.is_game_over:
            print("Winner is ", w)
            self.repaint()
    
    def MCTS_Policyval(self):
        move = self.MCTStrainer.get_move(self.puissance4, verbose = 1) #verbose to print qvals
        winner = self.puissance4.play_move(move)
        self.repaint()
        if self.puissance4.is_game_over:
            print("Winner is ", winner)
            self.repaint()


    def trainminmax_1(self, player = 2): #entraine le self.trainer3 contre l'ia minmax
        n = 1000
        print(n, " entrainements contre IA minmax, pronfondeur : ", self.profondeur)
        print("joueur entrainé : ", player)
        start = time.time()
        self.trainerV3.train_with_minmax(n = n, player = player, profondeur_max = self.profondeur)
        end = time.time()
        print("Après : ", end - start, "secondes, entrainement fini")
        
    def save_model_1(self, name = "QUICKSAVE"):
        self.trainerV3.agent.model.save(name + "X")
        self.trainerV3.agentO.model.save(name + "O")
        
        
    def play_moves(self):
        d = Play_moves_dialog(self)
        
    def set_moves_to_play(self, mtp, top, sleep_time = 0.3):
        top.destroy()
        mtp = mtp.replace("[", "")
        mtp = mtp.replace("]", "")
        mtp = mtp.replace(" ", "")
        moves_to_play = mtp.split(",")
        moves_to_play = [int(s) for s in moves_to_play]
        self.moves_to_play = moves_to_play
        for move in moves_to_play:
            self.puissance4.play_move(move)
            self.repaint()
            self.update()
            time.sleep(sleep_time)
            self.update()
            
        self.repaint()

    def set_game_board(self, board, turn):
        self.puissance4.board = board
        self.puissance4.turn = turn
        self.repaint()
    def view_batch(self, i = -1):
        #pops element from batch
        if i == - 1:
            i = self.i_batch
            self.i_batch += 1
        state = self.batch[0][i] #CNN bitmap state
        policy = self.batch[1][i]
        value = self.batch[2][i]
        turn = state[0][0][2]
        board = self.puissance4.convert_state_to_board_bitmap(state)
        print(board, turn)
        self.set_game_board(board,turn)
        
        print("Policy : ", policy)
        print("Value :", value)

    def simulate_game(self):
        d = Simulate_game_dialog(self)
    def simulate_game_1(self, function1 = 0, function2 = 0, sleep_time = 0.1, ntimes = 0):
        if ntimes == 0:
            ntimes = self.simulate_game_ngames
        #function = plays a move when called
        if function1 ==0:
            name1 = self.sim_game_name_func1
            function1 = self.play_func_dic[name1]
        if function2 == 0:
            name2 = self.sim_game_name_func2
            function2 = self.play_func_dic[name2]
        for i in range(ntimes):
            while not self.puissance4.is_game_over:
                turn = self.puissance4.turn
                time.sleep(sleep_time)
                if turn == 1:
                    function1()
                else:
                    function2()
                self.update()
            
            time.sleep(5*sleep_time)
            self.puissance4 = Game(self.h, self.w, profondeur_max = self.profondeur)

    def get_models_MCTS_NN(self):
        h = self.h
        w = self.w
        if h == 6 and w == 7:
            modelP = tf.keras.models.load_model("policy_crossentropy367")
            modelV = tf.keras.models.load_model("value_crossentropy367")
        elif h == 4 and w == 4:
            modelP = 0
            modelV = 0
        else:
            modelP = 0
            modelV = 0
            
        return modelP, modelV
    
    def get_models_DQN_normal(self):
        w = self.w
        h = self.h
        if h == 4 and w == 4:
            modelX = tf.keras.models.load_model("DQL44\Dense_bon\modelX")
            modelO = tf.keras.models.load_model("DQL44\temp44O")
            #modelX = tf.keras.models.load_model("DQL44\Dense_2\modelX")
            #modelO = tf.keras.models.load_model("DQL44\Dense_2\modelO")
        elif h == 6 and w == 7:
            modelX = tf.keras.models.load_model("DQL67/target_self_1/modelX")
            modelO = tf.keras.models.load_model("DQL67/target_self_1/modelO")
        else:
            modelX = 0
            modelO = 0
        return modelX, modelO
    def get_models_DQN_CNN(self):
        w = self.w
        h = self.h
        if h == 4 and w == 4:
            modelX = tf.keras.models.load_model("DQL44/CNN1/modelX")
            modelO = tf.keras.models.load_model("DQL44/CNN1/modelO")
        else:
            modelX = tf.keras.models.load_model("DQL67/CNN_p2_4/playerX")
            modelO = tf.keras.models.load_model("DQL67/CNN_p2_4/playerO")
        return modelX, modelO

    def get_dic_QL(self):
        h = self.h
        w = self.w
        if h == 4 and w == 4:
            a_file = open("QL44/44_1/Qdic_44_1.pkl", "rb")
            dic = pickle.load(a_file)
            return dic
        elif h == 6 and w == 7:
            #a_file = open("QL67/67_1/Qdic_67.pkl", "rb")
            #dic = pickle.load(a_file)
            #ATTENTION CRASH CAR FICHIER DE 2GB
            return {}

        return {}
if __name__ == "__main__" :
    jeu = Interface()
    jeu.mainloop()
    
