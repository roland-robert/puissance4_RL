import numpy as np
import random as rd
import copy as copy
"""

1 : Rouge
-1 : Bleu
0 : vide
board : toujours un np array h*w (hauteur x largeur)
etat,state : le board de shape (h*w,) (applatit)
move : toujours un entier entre 0 et h-1
player : 1 ou -1
"""
class Game:
    def __init__(self, h = 6, w = 7, profondeur_max = 2) -> None:
        self.h = h
        self.w = w
        self.is_game_over = False
        self.moves_played = []
        if w < 4 and h<4:
            print("/!\ Injouable")
        
        self.turn = 1
        self.board = np.zeros((h,w))
        
        self.profondeurmax = profondeur_max #POUR LE MINMAX
        self.penalite_nombre_coup = 1 #POUR MINMAX
        
    def get_hash(self, board = []): #POUR LE QLEARNING
        if len(board) == 0:
            board = self.board
        lehash = ""
        for i in board.reshape((self.h*self.w,)):
            if i==-1:
                lehash = lehash + "2"
            elif i == 1:
                lehash = lehash + "1"
            else:
                lehash = lehash + "0"
        return lehash
    def convert_hash_to_board(self, hashh): #POUR LE QLEARNING
        rep = list(hashh)
        for i in range(len(rep)):
            if rep[i] == "2":
                rep[i] = -1
            else:
                rep[i] = int(rep[i])
        rep = np.array(rep)
        return rep.reshape((self.h, self.w))
    def available_moves(self, hashh): #POUR LE QLEARNING
        return self.get_valid_moves(self.convert_hash_to_board(hashh))
    
    def get_state_CNN(self,board = []):
        if len(board) == 0:
            board = self.board
        return copy.deepcopy(board.reshape(self.h,self.w,1)) #np array h x w
    def get_state_CNN_bitmap(self,board = [], turn = []):
        if len(board) == 0:
            board = self.board
            turn = self.turn
        board1 = copy.deepcopy(board.reshape(self.h,self.w,1)) #np array h x w
        board2 = copy.deepcopy(board.reshape(self.h,self.w,1))
        board1 = np.where(board1 == 1, board1, 0) #remplace les -1 par 0
        board2 = -np.where(board2 == -1, board2, 0) #remplace les 1 par 0
        
        turnboard = turn * np.ones((self.h,self.w,1))
        bitmap = np.concatenate((board1, board2, turnboard), axis=2)
        if bitmap.shape != (self.h, self.w, 3):
            print("Shape error : ", bitmap.shape)
        return bitmap
    def get_state(self, board = []):
        if len(board) == 0:
            board = self.board
        return copy.deepcopy(board.reshape((self.h*self.w,))) #turns board into line
    def convert_state_to_board(self, state):
        return copy.deepcopy(state.reshape((self.h, self.w)))
    def convert_state_to_board_bitmap(self,state): #pour CNN bitmap
        st = state[:,:,0] - state[:,:,1]
        return st

    def get_valid_moves_from_state(self, state): #pour le DQL
        board = self.convert_state_to_board(state)
        return self.get_valid_moves(board)
    
    def get_valid_moves(self, board = []):
        if len(board) == 0:
            board = self.board
        rep = []
        for j in range(self.w):
            if board[0,j] == 0:
                rep.append(j)
        return rep

    def _get_winner(self, move, board = [], current_player = 0): #à activer après le move "move"
        #returns 1,-1, 0 or None
        if len(board) == 0:
            board = self.board
            current_player = self.turn

        player = -current_player #joueur qui a joué, potentiel vainqueur
        #récupérer les coordonnées i,j du coup
        j = move
        i = 0
        while board[i][j] == 0:
            i = i+1
        #vérif ligne
        left_space = min(3, j-0)
        right_space = min(3, self.w - j-1)
        line = board[i][j-left_space: j+ right_space+1]
        for k in range(len(line) - 3):
            if line[k] == line[k+1] == line[k+2] == line[k+3]:
                return player
        #verif colonne
        if i < self.h-3:
            if board[i][j] == board[i+1][j] == board[i+2][j] == board[i+3][j]:
                return player
        #verif diag
        diag1 = [board[i+k][j+k] for k in range(-3,4) if i+k>=0 and i+k<self.h and j+k>=0 and j+k<self.w]
        diag2 = [board[i+k][j-k] for k in range(-3,4) if i+k>=0 and i+k<self.h and j-k>=0 and j-k<self.w]
        for k in range(len(diag1) - 3):
            if diag1[k] == diag1[k+1] == diag1[k+2] == diag1[k+3]:
                return player
        for k in range(len(diag2) - 3):
            if diag2[k] == diag2[k+1] == diag2[k+2] == diag2[k+3]:
                return player
        #
        if 0 in board:
            return None#partie en cours
        return 0#match nul
    
    def play_move(self, move):
        #renvoie le gagnant 1, -1 ou 0 si partie finie, renvoie False si coup invalide, None si la partie continue
        #validité du coup
        board = self.board
        player = self.turn
        available_moves = self.get_valid_moves(board)
        if self.is_game_over:
            print("Game Over")
            return False
        if not(move in available_moves):
            print("COUP INVALIDE")
            print("board : ", board)
            print("coup : ", move)
            return False
        self.moves_played.append(move)
        #modification du board
        i = 0
        while board[i][move] == 0:
            i += 1
            if i== self.h:
                break
        i -= 1
        board[i][move] = player
        #update joueur
        self.turn = -self.turn
        #verif winner
        w = self._get_winner(move)
        if w != None:
            self.is_game_over = True
            return w
        return None
        
    def print_board(self,board = []):
        if len(board) == 0:
            board = self.board
        for line in range(len(board)):
            s = '|'
            for j  in range(len(board[0])):
                if board[line][j] == 0:
                    s = s + " " + "|"
                elif board[line][j] == 1:
                    s = s + "X" + "|"
                else:
                    s = s + "O" + "|"
            print(s)
    
    
    #PARTIE IA
    def simulate_move(self, move, board = [], turn = []):
        #prend un coup, un board et un turn et renvoit le gagnant et le board après coup (gagnant, board)
        #gagnant = 0,1,-1 ou None
        if len(board) == 0:
            board = copy.deepcopy(self.board)
            turn = self.turn
        else:
            board = copy.deepcopy(board)
        available_moves = self.get_valid_moves(board)
        if not(move in available_moves):
            print("COUP INVALIDE SIMULATION")
            print("board : ", board)
            print("coup : ", move)
            return False, False
        #modification du board
        i = 0
        while board[i][move] == 0:
            i += 1
            if i== self.h:
                break
        i -= 1
        board[i][move] = turn
        #verif winner
        w = self._get_winner(move, board, -turn)

        return w,board
    
    def random_IA(self):
        return rd.choice(self.get_valid_moves())
    def IA1(self):
        available_moves = self.get_valid_moves()
        for move in available_moves:
            w, board = self.simulate_move(move)
            if w == self.turn:
                return move
        return rd.choice(available_moves)
     #MINMAX
    def minmax(self, profondeur = 0, maxcol = None, col = None, etat = [], winner = None, pmax = None): #etat = (VX,VO,VV), joueur = "X" ou "O"
        if pmax == None:
            pmax = self.profondeurmax
        #col = 1 ou -1 (turn)
        #ici état est un board (numpy h*w)
        if pmax == 0:
            return self.random_IA(), 0
        if len(etat) == 0:
            etat = copy.deepcopy(self.board)
        else:
            etat = copy.deepcopy(etat)
        if col == None:
            col = self.turn
        if maxcol == None: #maxcol = la couleur qui doit jouer, reste constant dans la récursion
            maxcol = col
        
        colcomp = -1 * col #couleur complémentaire
        
        if winner == 1 or winner == -1:
            valeur = 10000-profondeur*self.penalite_nombre_coup if winner == maxcol else -10000 + profondeur*self.penalite_nombre_coup
            return None, valeur
        if winner == 0:
            valeur = 0
            return None, valeur
        if profondeur >= pmax:
            valeur = 0 #valeur de l'état... pas implémenté
            return None, valeur
        
        min_score = np.inf
        max_score = -np.inf
        best_moves = []
        best_score = None
        valid_moves = self.get_valid_moves(board = etat)
        liste_plateaux_successeurs = []
        for move in valid_moves:
            w, newetat = self.simulate_move(move, board = etat, turn = col)
            nextmove, score = self.minmax(profondeur +1, maxcol, colcomp, newetat, winner = w, pmax = pmax)
            if maxcol == col:
                if score>max_score:
                    max_score = score
                    best_moves = [move]
                    best_score = max_score
                if score == max_score:
                    best_moves.append(move)
            else:
                if score < min_score:
                    min_score = score
                    best_moves = [move]
                    best_score = min_score
                if score == min_score:
                    best_moves.append(move)

        best_move = rd.choice(best_moves)
        
        return best_move, best_score
    


            
def play_game():
    game = Game()
    while not(game.is_game_over):
        game.print_board()
        w = False
        while w == False:
            move = int(input("Your move : "))
            w = game.play_move(move)
        
        if w != None : 
            print("Winner is : ", w)
            game.print_board()
            
def test_randomvsbasic(h = 4, w = 4, n =1000):
    wins = 0 #for random IA who is 1
    losses = 0
    for i in range(n):
        game = Game(h,w)
        while not(game.is_game_over):
            rmove = game.random_IA()
            wi = game.play_move(rmove)

            if wi == None:
                move1 = game.IA1()
                wi = game.play_move(move1)
            if wi != None:
                if wi == 1:
                    wins += 1
                if wi == -1:
                    losses += 1
    return wins/n, losses/n, (n-wins-losses)/n
def test_minmaxvsminmax(h =4, w = 4, n=1000, p1 = 0, p2 = 0):
    wins = 0 #for random IA who is 1
    losses = 0
    for i in range(n):
        if i%10 == 0:
            print(int(100*i/n), "%")
        game = Game(h,w)
        while not(game.is_game_over):
            move1, score1 = game.minmax(pmax = p1)
            wi = game.play_move(move1)

            if wi == None:
                move2, score2 = game.minmax(pmax = p2)
                wi = game.play_move(move2)
            if wi != None:
                if wi == 1:
                    wins += 1
                if wi == -1:
                    losses += 1
    return wins/n, losses/n, (n-wins-losses)/n

if __name__ == "__main__":
    #play_game()
    h = 6
    w = 7
    p1 = 3
    p2 = 2
    n = 200
    a,b,c = test_minmaxvsminmax(h = h, w = w, n = n, p1 = p1, p2 = p2)
    print("Result for p1 = ", p1, " and p2 = ", p2, " for n = ", n, " games")
    print(a,b,c, 100*(a-b))
    print(int(100*a),"%, ", int(100*b),"%", int(100*(a-b)))