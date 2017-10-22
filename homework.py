# Author - Lakshya Kejriwal
# Date   - 31st August, 2017

import numpy as np
import math
import random
import itertools
import copy
from compiler.ast import flatten
import time

start_time = time.time()

# Checking if the current board position is valid or not
def isValidPosition(solution, row, column):
    for position in range(row):
        if (solution[position] == column) or (position + solution[position] == row + column) or (position - solution[position] == row - column):  # Checking for column, row and diagnol queens
               return False
    return True

#Check the max no of lizards that can be placed
def maxLizards(board):
    total = 0
    for row in board:
        row = row.tolist()
        trees = row.count(2)
        if trees == 0:
            total += 1
        else:
            check = [x[0] for x in itertools.groupby(row)]
            total += check.count(0)
    
    return total

#To check if a lizard can attack another in the presence of trees
def checkCostWithTrees(values, count):
    index = [index for index, j in enumerate(values) if j==2]
    if count > len(index)+1:
        return True
    index.append(len(values)-1)
    index.insert(0, 0)
    cost = 0
    for idx in range(len(index)-1):
        if(values[index[idx]:index[idx+1]+1].count(1) == 2):
            cost = cost + 1
        elif(values[index[idx]:index[idx+1]+1].count(1) > 2):
            cost += values[index[idx]:index[idx+1]+1].count(1)
    
    return cost
    
#To check if current board position is valid or not when trees are present
def getHeuristicCostForTrees(board):
    temp_board = board[:]
    
    total_cost = 0

    for row in temp_board:
        row = row.tolist()
        count = row.count(1)
        trees = row.count(2)
        if(trees>0):
            total_cost += checkCostWithTrees(row, count)
        elif(count==2):
            total_cost += 1
        elif(count>2):
            total_cost += count
    
    for cols in temp_board.T:
        cols = cols.tolist()
        count = cols.count(1)
        trees = cols.count(2)
        if(trees>0):
            total_cost += checkCostWithTrees(cols, count)
        elif(count==2):
            total_cost += 1
        elif(count>2):
            total_cost += count
    
    #Checking main diagnol of the board
    left_to_right = np.diag(temp_board, 0).tolist()
        
    count = left_to_right.count(1)
    trees = left_to_right.count(2)
    if(trees>0):
        total_cost += checkCostWithTrees(left_to_right, count)
    elif(count==2):
        total_cost += 1
    elif(count>2):
        total_cost += count       
    
    right_to_left = np.diag(np.fliplr(temp_board), 0).tolist()
    count = right_to_left.count(1)
    trees = right_to_left.count(2)
    if(trees>0):
        total_cost += checkCostWithTrees(right_to_left, count)
    elif(count==2):
        total_cost += 1
    elif(count>2):
        total_cost += count
    
    #Checking upper and lower diagnols
    for diagnol in range(1, len(temp_board)-1):
        left_to_right_up = np.diag(temp_board, diagnol).tolist()
        
        count = left_to_right_up.count(1)
        trees = left_to_right_up.count(2)
        if(trees>0):
            total_cost += checkCostWithTrees(left_to_right_up, count)
        elif(count==2):
            total_cost += 1
        elif(count>2):
            total_cost += count   
        
        left_to_right_down = np.diag(temp_board, (-1 * diagnol)).tolist()
        
        count = left_to_right_down.count(1)
        trees = left_to_right_down.count(2)
        if(trees>0):
            total_cost += checkCostWithTrees(left_to_right_down, count)
        elif(count==2):
            total_cost += 1
        elif(count>2):
            total_cost += count
            
        right_to_left_up = np.diag(np.fliplr(temp_board), diagnol).tolist()
        count = right_to_left_up.count(1)
        trees = right_to_left_up.count(2)
        if(trees>0):
            total_cost += checkCostWithTrees(right_to_left_up, count)
        elif(count==2):
            total_cost += 1
        elif(count>2):
            total_cost += count
        
        right_to_left_down = np.diag(np.fliplr(temp_board), (-1 * diagnol)).tolist()
        count = right_to_left_down.count(1)
        trees = right_to_left_down.count(2)
        if(trees>0):
            total_cost += checkCostWithTrees(right_to_left_down, count)
        elif(count==2):
            total_cost += 1
        elif(count>2):
            total_cost += count   
        
    return total_cost

#To check if a lizard can attack another in the presence of trees
def checkWithTrees(values, count):
    index = [index for index, j in enumerate(values) if j==2]
    if count > len(index)+1:
        return True
    index.append(len(values)-1)
    index.insert(0, 0)
    for idx in range(len(index)-1):
        if(values[index[idx]:index[idx+1]+1].count(1) > 1):
            return True
    
#To check if current board position is valid or not when trees are present
def isValidPositionForTrees(solution, tree_pos, row, column):
    #temp_board = makeBoard(solution, tree_pos)
    
    if(isinstance(column,(int))):
        column = [column]
    
    for position in range(row):
        for col in column:
            if(type(solution[position]) is int) :
                if (solution[position] == col):   # Checking for column, row and diagnol queens
                    if(col in list(itertools.chain.from_iterable(tree_pos[position+1:row]))):
                        continue
                    else:
                        return False
                if (position + solution[position] == row + col) or (position - solution[position] == row - col):
                    if(any(solution[position] < t < col for t in list(itertools.chain.from_iterable(tree_pos[position+1:row])))):
                        continue
                    elif(any(col < t < solution[position] for t in list(itertools.chain.from_iterable(tree_pos[position+1:row])))):
                        continue
                    else:
                        return False
            elif (solution[position] == None):
                continue
            else:
                for sol in solution[position]:
                    if (sol == col):   # Checking for column, row and diagnol queens
                        if(col in list(itertools.chain.from_iterable(tree_pos[position+1:row]))):
                            continue
                        else:
                            return False
                    if (position + sol == row + col) or (position - sol == row - col):
                        if(any(sol < t < col for t in list(itertools.chain.from_iterable(tree_pos[position+1:row])))):
                            continue
                        elif(any(col < t < sol for t in list(itertools.chain.from_iterable(tree_pos[position+1:row])))):
                            continue
                        else:
                            return False
            
    return True

#Generate positions for the next layer of the graph
def generateLizardPositions(solution, size, row):
    positions = []
    for pos in range(size):
        if(isValidPosition(solution, row, pos)):
            positions.append([row, pos])
    return positions

#Generating permutations for lizard positions in between trees
def generatePermutations(combination_list):
    permutation_list = []
    
    for num in range(2, len(combination_list)+1):
        permutation_list += list(itertools.combinations(combination_list, num))
    
    position_list = []
    for permutations in permutation_list:
        permutations = list(permutations)
        position_list += list(itertools.product(*permutations))
        
    return position_list

#Generate positions for the next layer of the graph with obstacles
def generateLizardPositionsForTreesDfs(solution, tree_pos, row):
    positions = []
    board = makeBoard(solution, tree_pos, 0)
    row_val = board[row].tolist()
    index_0 = [index_0 for index_0, j in enumerate(row_val) if j==0]
    for pos in index_0:
        if(isValidPositionForTrees(solution, tree_pos, row, pos)):
            positions.append(pos)
    
    if(len(index_0) == len(board)):
        return positions
    
    index_2 = [index_2 for index_2, j in enumerate(row_val) if j==2]
    
    combination_list = []
    
    if(index_2[0]!=0):
        if(index_2[0]) == 1:
            combination_list.append([0])
        else:
            combination_list.append(range(0, index_2[0]))
    
    for idx in range(len(index_2)-1):
        if(index_2[idx+1] - index_2[idx]) == 1:
            continue
        elif(index_2[idx+1] - index_2[idx]) == 2:
            combination_list.append([index_2[idx]+1])
        else:
            combination_list.append(range(index_2[idx]+1,index_2[idx+1]))
    
    if(index_2[-1]!=len(board)-1):
        if(len(board)-index_2[-1]) == 2:
            combination_list.append([index_2[-1]+1])
        else:
            combination_list.append(range(index_2[-1]+1, len(board)))
    
    if([item for item in positions] == combination_list):
        return positions
    else:
        permutation_positions = generatePermutations(combination_list[:])
#        for permutation in permutation_positions:
#            if(isValidPositionForTrees(solution, tree_pos, row, list(permutation))):
#                positions.append([row, list(permutation)])
        positions += permutation_positions
        return positions
    
#Generate positions for the next layer of the graph with obstacles
def generateLizardPositionsForTrees(solution, tree_pos, row):
    positions = []
    board = makeBoard(solution, tree_pos, 0)
    row_val = board[row].tolist()
    index_0 = [index_0 for index_0, j in enumerate(row_val) if j==0]
    for pos in index_0:
        if(isValidPositionForTrees(solution, tree_pos, row, pos)):
            positions.append([row, pos])
    
    if(len(index_0) == len(board)):
        return positions
    
    index_2 = [index_2 for index_2, j in enumerate(row_val) if j==2]
    
    combination_list = []
    
    if(index_2[0]!=0):
        if(index_2[0]) == 1:
            combination_list.append([0])
        else:
            combination_list.append(range(0, index_2[0]))
    
    for idx in range(len(index_2)-1):
        if(index_2[idx+1] - index_2[idx]) == 1:
            continue
        elif(index_2[idx+1] - index_2[idx]) == 2:
            combination_list.append([index_2[idx]+1])
        else:
            combination_list.append(range(index_2[idx]+1,index_2[idx+1]))
    
    if(index_2[-1]!=len(board)-1):
        if(len(board)-index_2[-1]) == 2:
            combination_list.append([index_2[-1]+1])
        else:
            combination_list.append(range(index_2[-1]+1, len(board)))
    
    if([item[1] for item in positions] == combination_list):
        return positions
    else:
        permutation_positions = generatePermutations(combination_list[:])
        for permutation in permutation_positions:
            if(isValidPositionForTrees(solution, tree_pos, row, list(permutation))):
                positions.append([row, list(permutation)])
        return positions

    

#Check if we have reached a solution or not
def isValidSolution(solution, row, column, lizards, size):
    if(isValidPosition(solution, row, column)):            #Check if this solution is valid or not
        if(row == size-1) or (row == lizards-1):           #Check if we reached the last layer or placed the last queen
            solution[row] = column
            return True
    return False


def removeLizards(solution, remove):
    idx = 0
    for row in solution:
        if remove == 0:
            return solution
        if(type(row) is int):
            solution[idx] = None
            remove -= 1
        elif(type(row) is list):
            if remove < len(row):
                row = row[0:len(row)-remove]
                solution[idx] = row
                remove = 0
            else:
                solution[idx] = None
                remove -= len(row)

#Check if we have reached a solution or not with Trees
def isValidSolutionForTrees(solution, row, column, lizards, size):
    if(isValidPositionForTrees(solution, tree_pos, row, column)):            #Check if this solution is valid or not
        solution[row] = column
        total_placed = sum(x is not None for x in flatten(solution))
        if(row == size-1):                                      #Check if we reached the last layer or placed the last queen
            if total_placed == lizards:
                return True
            else:
                solution[row] = None
                return False
        if total_placed == lizards:
            return True
        elif total_placed >= lizards:
            remove = total_placed - lizards
            solution = removeLizards(solution, remove)
            return True
    solution[row] = None
    return False
        
#Cost function for Simulated Annealing
def getHeuristicCost(solution):
    cost = 0
    for position in range(0, len(solution)):
        for next_position in range(position+1, len(solution)):
            if (solution[position] == solution[next_position]) or abs(position - next_position) == abs(solution[position] - solution[next_position]):
                cost = cost + 1
    return cost

#Random selection of lizards on the board
def randomGenerator(board, lizards):
    positions = []
    y_val = range(len(board))
    chosen = [None] * len(board)
    row_val = 0
    skip_row = [0] * len(board)
    placed = [0] * len(board)
    for row in board:
        row = row.tolist()
        index = [idx for idx, item in enumerate(row) if item == 0]
        row_positions = []
        for idx in index:
            row_positions.append(idx)
        positions.append(row_positions)
        row_val += 1
    
    total_lizards = lizards
    
    for idx, pos in enumerate(positions):
        if not pos:
            skip_row[idx] = 0
            placed[idx] = 0
            continue
        else:
            num = maxLizards(board[idx:idx+1])
            skip_row[idx] = num
            if (num>1):
                rem = maxLizards(board[idx+1:len(board)])
                if (total_lizards > (len(board)-idx)) or (rem+1 < total_lizards):
                    num=num
                else:
                    num=1
            
            if(not(total_lizards == 0)):
                placed[idx] = num
                chose_y = []
                
                if (not(y_val)):
                    chose_y = random.sample(pos, num)
                elif(num > len(y_val)):
                    chose_y = y_val
                    chose_y = random.sample(pos, num-len(y_val))
                else:
                    chose_y = random.sample(y_val, num)
                    while(not(set(chose_y) < set(pos))):
                        if(chose_y == pos):
                            break
                        if(pos not in y_val):
                            chose_y = random.sample(pos, num)
                            break
                        else:
                            chose_y = []
                            chose_y = random.sample(y_val, num)
    
                y_val = [y for y in y_val if y not in chose_y]
                
                board[idx][chose_y] = 1
                total_lizards -= num
                chosen[idx] = chose_y
                positions[idx] = [x for x in positions[idx] if x not in chose_y]
            
    return board, positions, chosen, skip_row, placed

#Find position of trees in board
def findTrees(board):
    tree_pos = []
    i = 0
    for row in board:
        tree_pos += [(i,idx) for idx, item in enumerate(row) if item == 2]
        i += 1
    
    return tree_pos

#Find position of trees in board
def findTreesPos(board):
    tree_pos = [None] * len(board)
    i = 0
    for row in board:
        tree_pos[i] = [idx for idx, item in enumerate(row) if item == 2]
        i += 1
    
    return tree_pos

def nextRow(board, temp_position, temp_chosen, skip_row, placed, lizards):
    temp_board = copy.deepcopy(board)
    
    index_1 = random.randint(0, len(skip_row)-1)
    
    if(skip_row == placed):
        return temp_board, temp_position, temp_chosen
    
    iteration = 0
    while(skip_row[index_1]==0 or not(temp_position[index_1]) or skip_row[index_1] <= placed[index_1]):
        index_1 = random.randint(0, len(skip_row)-1)
        iteration += 1
        if(iteration>10):
            return temp_board, temp_position, temp_chosen
    
    index_2 = random.randint(0, len(temp_chosen)-1)
    iteration = 0
    while(index_1 == index_2 or not temp_chosen[index_2]):
        index_2 = random.randint(0, len(temp_chosen)-1)
        iteration += 1
        if(iteration>10):
            return temp_board, temp_position, temp_chosen
    x_1 = 0
    
    if(len(temp_chosen[index_2])>1):
        x_1 = random.randint(0, len(temp_chosen[index_2])-1)
    
    selected_1 = temp_chosen[index_2][x_1]
    if(len(temp_position[index_1]) == 1):
        selected_pos = temp_position[index_1][0]
    else:
        selected_pos = temp_position[index_1][random.randint(0, len(temp_position[index_1])-1)]
    
    temp_chosen[index_2].remove(selected_1)
    temp_position[index_2] += [selected_1]
    
    temp_chosen[index_1] += [selected_pos]
    temp_position[index_1].remove(selected_pos)
    
    temp_board[index_2][selected_1] = 0
    temp_board[index_1][selected_pos] = 1
    
    return temp_board, temp_position, temp_chosen

def nextColumn(board, positions, chosen, tree_pos, new_board, lizards):
    temp_position = copy.deepcopy(positions)
    temp_board = copy.deepcopy(board)
    temp_chosen = copy.deepcopy(chosen)
    x_1 = 0

    index_1 = random.randint(0, len(temp_chosen)-1)
    
    iteration = 0
    while(not temp_chosen[index_1] or not temp_position[index_1]):
        index_1 = random.randint(0, len(temp_chosen)-1)
        iteration += 1
        if(iteration>10):
            return temp_board, temp_position, temp_chosen
        
    if(len(temp_chosen[index_1])>1):
        x_1 = random.randint(0, len(temp_chosen[index_1])-1)
        
    selected_1 = temp_chosen[index_1][x_1]
    selected_2 = random.randint(0, len(temp_position[index_1])-1)
    selected_2 = temp_position[index_1][selected_2]
        
    iteration = 0
    while((selected_1 == selected_2)):
        index_1 = random.randint(0, len(temp_chosen)-1)

        while(not temp_chosen[index_1]):
            index_1 = random.randint(0, len(temp_chosen)-1)
        
        if(len(temp_chosen[index_1])>1):
            x_1 = random.randint(0, len(temp_chosen[index_1])-1)
            selected_1 = temp_chosen[index_1][x_1]
            
        else:
            x_1 = 0
            selected_1 = temp_chosen[index_1][x_1]
        
        iteration += 1
        if(iteration>10):
            return temp_board, temp_position, temp_chosen

    if(len(temp_chosen[index_1]) > 1):
 
        temp_position[index_1] += [temp_chosen[index_1][x_1]]
        temp_position[index_1].remove(selected_2)
    
        temp_chosen[index_1][x_1] = selected_2
    else:

        temp_position[index_1] += [temp_chosen[index_1][x_1]]
        temp_position[index_1].remove(selected_2)
        
        temp_chosen[index_1][x_1] = selected_2
    
    temp_board[index_1][selected_1] = 0
    temp_board[index_1][selected_2] = 1
    
    return temp_board, temp_position, temp_chosen

# Random selection of next position by swapping two lizards on the board
def randomStateGenerator(next_s):
    x = random.randint(0, len(next_s)-1)
    y = random.randint(0, len(next_s)-1)
    next_s[x], next_s[y] = next_s[y], next_s[x]
    return next_s

#The recursive dfs function
def dfs(solution, row, lizards):
    size = len(solution)
    all_positions = generateLizardPositions(solution, size, row)
    stack = []
    stack += all_positions
    while stack:
        position = stack.pop()
        if(isValidSolution(solution, position[0], position[1], lizards, size)):
            return True

        solution[position[0]] = position[1]
        explored = dfs(solution, row+1, lizards)
        if explored:
            return True
        else:
            solution[position[0]] = None
    return False

def makeBoard(solution, tree_pos, r):
    board = np.zeros((size, size), dtype = np.int)
    idx = 0
    for row in board:
        val = solution[idx]
        if(type(val) is list) or (type(val) is int):
            row[solution[idx]] = 1
        if(not(tree_pos[idx])==None):
            row[tree_pos[idx]] = 2
        idx += 1
    return board[r:len(board)]

#The recursive dfs function for obstacles
def dfsForTrees(solution, tree_pos, row, lizards):
    size = len(solution)                          
    all_positions = generateLizardPositionsForTreesDfs(solution, tree_pos, row)
    stack = []
    stack += all_positions
    position = None
    if not stack:
        if getHeuristicCostForTrees(makeBoard(solution, tree_pos, 0))==0 and lizards == sum(x is not None for x in flatten(solution)):
            return True
        if row > len(board)-2:
            return False
        stack = generateLizardPositionsForTreesDfs(solution, tree_pos, row+1)
        row = row + 1
    while stack:
        position = stack.pop()
        if type(position) is int:
            position = [row, position]
        else:
            position = [row, list(position)]
        if(isValidSolutionForTrees(solution, position[0], position[1], lizards, size)):
            solution[row] = position[1]                                             #To keep the last location on board when returned otherwise no last location on board
            return True
        if(isValidPositionForTrees(solution, tree_pos, position[0], position[1])):
            if(row > len(board)-2):                                                             #To keep the dfs from going beyond the rows of the board and allow it to backtrack to another solution
                solution[position[0]] = position[1]                                                  #To remove the last location in placed of lizard
                return False
            solution[position[0]] = position[1]
            rem_placed = lizards - sum(x is not None for x in flatten(solution))
            if(maxLizards(makeBoard(solution, tree_pos, row+1)) >= rem_placed): 
                explored = dfsForTrees(solution, tree_pos, row+1, lizards)
                if explored:
                    return True
                else:
                    solution[position[0]] = None  
            else:
                solution[position[0]] = None  
                return False
        
        if(time.time()-start_time > 290):
            return False
                        
    rem_placed = lizards - sum(x is not None for x in flatten(solution))
    if(maxLizards(makeBoard(solution, tree_pos, row+1)) >= rem_placed):                     #To skip rows with no possible placement in order to try for other rows or else dfs will fail
        if(row > len(board)-2):                                                             #To keep the dfs from going beyond the rows of the board and allow it to backtrack to another solution
            if type(position) is None:
                return False
            solution[position[0]] = None                                                    #To remove the last location in placed of lizard
            return False
        explored = dfsForTrees(solution, tree_pos, row+1, lizards)
        if explored:
                return True
        else:
            if position is not None:
                solution[position[0]] = None  
            return False
        
#    board[position[0]][position[1]] = 0
    return False

#The bfs function
def bfs(board, solution, row, lizards):
    size = len(solution)
    list_of_sol = []
    list_of_sol.append(solution[:])
    while list_of_sol:
        layer_i = list_of_sol[:]
        list_of_sol = []
        for solution in layer_i:
            queue = generateLizardPositions(solution, size, row)
            for position in queue:
                if(isValidSolution(solution, position[0], position[1], lizards, size)):
                    return solution
#                elif(isValidPosition(solution, position[0], position[1])):
                solution[position[0]] = position[1]
                list_of_sol.append(solution[:])
        row = row+1
        
#The bfs function for obstacles
def bfsForTrees(solution, tree_pos, row, lizards):
    size = len(board)
    list_of_sol = []
    list_of_sol.append(solution[:])
    while list_of_sol:
        layer_i = list_of_sol[:]
        temp_sol = []
        for sol in layer_i:
            queue = generateLizardPositionsForTrees(sol, tree_pos, row)
            for position in queue:
                if(isValidSolutionForTrees(sol, position[0], position[1], lizards, size)):
                    return sol
#                elif(isValidPositionForTrees(solution, position[0], position[1])):
                sol[position[0]] = position[1]
                rem_placed = lizards - sum(x is not None for x in flatten(sol))
                rem_rows = size - row - 1
                if(rem_placed <= rem_rows):
                    temp_sol.append(copy.deepcopy(sol))
                elif(maxLizards(makeBoard(solution, tree_pos, row+1)) >= rem_placed):
                    temp_sol.append(copy.deepcopy(sol))
                sol[position[0]] = None
                
                if(time.time()-start_time > 290):
                    return False

            if not queue:
                rem_placed = lizards - sum(x is not None for x in flatten(sol))             
                if(maxLizards(makeBoard(sol, tree_pos, row+1)) >= rem_placed):
                        temp_sol.append(copy.deepcopy(sol))
  
        if queue:
            rem_placed = lizards - sum(x is not None for x in flatten(sol))             
            if(maxLizards(makeBoard(sol, tree_pos, row+1)) >= rem_placed):
                        temp_sol.append(copy.deepcopy(sol))
                        
        row = row + 1
        if(row > len(board)-1):                                     #To keep the bfs from going beyond the rows of the board
            sol[position[0]] = None                                 #To remove the last location in placed of lizard
            return False
        
        if temp_sol:
            list_of_sol = []
            list_of_sol = copy.deepcopy(temp_sol)
            
    
    return False
            
    
def simulatedAnnealing(size, temperature, alpha, lizards):
    num_iter = 170000
    current = range(lizards)
    current += [float("Nan")] * (size-lizards)
    old_cost = getHeuristicCost(current)

    for i in range(num_iter):
        #print(current, old_cost, temperature)
        temperature = temperature * alpha

        successor = randomStateGenerator(current[:])
        new_cost = getHeuristicCost(successor[:])
        delta_E = new_cost - getHeuristicCost(current)
        if delta_E<0:
           current = successor[:]
           old_cost = new_cost
        elif math.exp(-delta_E/temperature) > random.uniform(0,1):
           current = successor[:]
           old_cost = new_cost
        
        if old_cost == 0:
           return current

    return None

def simulatedAnnealingForTrees(board, temperature, alpha, lizards):
    num_iter = 170000
    no_sol = 0
    new_board = copy.deepcopy(board)
    board, positions, chosen, skip_row, placed = randomGenerator(board[:], lizards)
#    print board
    tree_pos = findTrees(board)
    
    old_cost = getHeuristicCostForTrees(board)
    i = 0
    restarts = 0

    while i < num_iter:
#        print(i, old_cost, temperature)
        temperature = temperature * alpha
        
        if(time.time()-start_time > 290):
            return False
        
        if(i > 2000):
            i = 0
            #print time.time()-start_time
            if restarts > 5:
                restarts = 0
                board, positions, chosen, skip_row, placed = randomGenerator(board[:], lizards)
                temperature = 2
                
                if no_sol > 20:
                    return False
                no_sol += 1
            else:
                #board = copy.deepcopy(new_board)
                #board, positions, chosen, skip_row, placed = randomGenerator(board[:], lizards)
                board, positions, chosen = nextRow(board, positions, chosen, skip_row, placed, lizards)
                old_cost = getHeuristicCostForTrees(board)
            temperature = 2
            restarts += 1
            continue

        successor, new_positions, new_chosen = nextColumn(board[:], positions[:], chosen[:], tree_pos, new_board, lizards)
        new_cost = getHeuristicCostForTrees(successor[:])
        delta_E = new_cost - getHeuristicCostForTrees(board)
        
        if delta_E<0:
           board = successor[:]
           old_cost = new_cost
           positions = copy.deepcopy(new_positions)
           chosen = copy.deepcopy(new_chosen)

        elif math.exp(-delta_E/temperature) > random.uniform(0,1):
           board = successor[:]
           old_cost = new_cost
           positions = copy.deepcopy(new_positions)
           chosen = copy.deepcopy(new_chosen)
        
        if old_cost == 0:
           return board
     
        i += 1
    

    return board

data = [];

with open("Test cases/1/input.txt") as file:
    data = file.read().splitlines()

method = data.pop(0)
size = int(data.pop(0))
lizards = int(data.pop(0))

values = [];
for val in data:
    values.append(list(val))

board = np.array(values, dtype=int)

tree_count = np.array(board).flatten().tolist().count(2)
ans = ""

solution = [None] * len(board)
tree_pos = findTreesPos(board)

if (tree_count == 0) and (lizards > len(board)):
    ans = "FAIL"
    with open("output.txt", 'w+') as file:
        file.write(ans)
     
elif (tree_count > 0 ) and (lizards > maxLizards(board)):
    ans = "FAIL"
    with open("output.txt", 'w+') as file:
        file.write(ans)
    
else:
    if (tree_count == 0):
        if (method == "BFS"):
            solution = bfs(board, solution, 0, lizards)

        elif (method == "DFS"):
            dfs(solution, 0, lizards)

        elif (method == "SA"):
            solution = simulatedAnnealing(len(board), 2, 0.99, lizards)
        
        if(solution is None):
            ans = "FAIL"
            with open("output.txt", 'w+') as file:
                file.write(ans)
        elif (solution.count(None) == len(solution)):
            ans = "FAIL"
            with open("output.txt", 'w+') as file:
                file.write(ans)
        else:
            ans = "OK"
            with open("output.txt", 'w+') as file:
                file.write(ans + "\n")
                idx = 0
                for row in board:
                    if(not(solution[idx]==None or math.isnan(solution[idx]))):
                        row[solution[idx]] = 1
                    file.write(''.join(str(r) for r in row))
                    file.write("\n")
                    idx += 1

    
    elif (tree_count > 0):
        if (method == "BFS"):
            solution = bfsForTrees(solution, tree_pos, 0, lizards)
            
            if(solution is False):
                ans = "FAIL"
                with open("output.txt", 'w+') as file:
                    file.write(ans)
            else:
                ans = "OK"
                with open("output.txt", 'w+') as file:
                    file.write(ans + "\n")
                    idx = 0
                    for row in board:
                        val = solution[idx]
                        if(type(val) is list) or (type(val) is int):
                            row[solution[idx]] = 1
                        if(not(tree_pos[idx])==None):
                            row[tree_pos[idx]] = 2
                        file.write(''.join(str(r) for r in row))
                        file.write("\n")
                        idx += 1

        elif (method == "DFS"):
            dfsForTrees(solution, tree_pos, 0, lizards)
            
            if(solution is None):
                ans = "FAIL"
                with open("output.txt", 'w+') as file:
                    file.write(ans)
            elif (solution.count(None) == len(solution)):
                ans = "FAIL"
                with open("output.txt", 'w+') as file:
                    file.write(ans)
            else:
                ans = "OK"
                with open("output.txt", 'w+') as file:
                    file.write(ans + "\n")
                    idx = 0
                    for row in board:
                        val = solution[idx]
                        if(type(val) is list) or (type(val) is int):
                            row[solution[idx]] = 1
                        if(not(tree_pos[idx])==None):
                            row[tree_pos[idx]] = 2
                        file.write(''.join(str(r) for r in row))
                        file.write("\n")
                        idx += 1

        elif (method == "SA"):
            board = simulatedAnnealingForTrees(board, 2, 0.99, lizards)

            if (np.array(board).flatten().tolist().count(1) == 0):
                ans = "FAIL"
                with open("output.txt", 'w+') as file:
                    file.write(ans)
            else:
                ans = "OK"
                with open("output.txt", 'w+') as file:
                    file.write(ans + "\n")
                    idx = 0
                    for row in board:
                        file.write(''.join(str(r) for r in row))
                        file.write("\n")
                        idx += 1

 #   print("--- %s seconds ---" % (time.time() - start_time))

    
