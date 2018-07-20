#NOTE: We added a field called "plan" to the Player class. We did nothing to change the functionality of the game


from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import math
from itertools import repeat
from itertools import permutations
import copy
from datetime import datetime
### GLOBALS

SETTLEMENT = 0
CARD = 1
CITY = 2
ROAD = 3
MAX_POINTS = 10
MAX_RESOURCES = 6
START_RESOURCES = 3
LIMIT = 7

costs = np.array([[2, 1, 1],
                  [1, 2, 2],
                  [0, 3, 3],
                  [1, 1, 0]])

class CatanException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class Catan:
    def __init__(self, dice, resources, settlements = [], cities = [], roads = []):
        self.width = dice.shape[1]
        self.height = dice.shape[0]
        self.dice = dice
        self.resources = resources
        self.settlements = settlements
        self.cities = cities
        self.roads = roads
        self.max_vertex = (self.width+1)*(self.height+1) - 1

    def is_port(self, vertex):
        return vertex == 0 or vertex == self.width or vertex == self.max_vertex or vertex == self.max_vertex - self.width

    ## 0 - 2:1 wood
    ## 1 - 2:1 brick
    ## 2 - 2:1 grain
    ## 3 - 3:1 general
    def which_port(self, vertex):
        if vertex == 0:
            return 0
        elif vertex == self.width:
            return 1
        elif vertex == self.max_vertex - self.width:
            return 2
        elif vertex == self.max_vertex:
            return 3
        else:
            raise CatanException("{0} is not a port".format(vertex))

    def get_vertex_number(self, x, y):
        return (self.height + 1) * y + x
    
    def get_vertex_location(self, n):
        return (n % (self.height+1), n // (self.height+1))
    
    def is_tile(self, x, y):
        """returns whether x,y is a valid tile"""
        return x >= 0 and x < self.width and y >= 0 and y < self.width

    def build_road(self, c0, c1):
        v0 = self.get_vertex_number(c0[0], c0[1])
        v1 = self.get_vertex_number(c1[0], c1[1])
        if self.if_can_build_road(v0, v1):
            self.roads.append((v0, v1))
        else:
            raise CatanException("({0},{1}) is an invalid road".format(c0, c1))
            
    def if_can_build_road(self, start, end):
        ##order the road vertices
        temp = max(start, end)
        v1 = min(start, end)
        v2 = temp
        """returns true if road is valid, false otherwise"""
        #check if road vertices are on the map
        if v1 < 0 or v2 < 0 or v1 > self.max_vertex or v2 > self.max_vertex:
            raise CatanException("({0},{1}) is an invalid road".format(v1, v2))
        if v1 == v2: return False
        #first let's check that the spot is empty:
        if (v1, v2) in self.roads or (v2, v1) in self.roads:
            return False
        
        #now let's check if the proposed road is valid.
        #CORNER CASES
        if v1 == 0 or v2 == 0:
            if not (v1 + v2 == 1 or v1 + v2 == self.width+1):
                return False
        if v1 == self.width or v2 == self.width:
            if not (v1 + v2 == 2*self.width - 1 or v1 + v2 == 3*self.width+ 1):
                return False
        if v1 == (self.width + 1)*self.height or v2 == (self.width + 1)*self.height:
            if not (v1 + v2 == 2*(self.width + 1)*self.height + 1 or v1 + v2 == (self.width + 1)*(2*self.height - 1)):
                return False
        if v1 == self.max_vertex or v2 == self.max_vertex:
            if not (v1 + v2== 2*self.max_vertex - 1 or v1 + v2== (2 * self.max_vertex - (self.width + 1))):
                return False
        #EDGE CASES... literally --
        ## left edge
        if v1%(self.width + 1) == 0 or v2%(self.width + 1) == 0:
            if not (v2 - v1 == self.width + 1 or v2 - v1 == 1):
                return False
        ## bottom edge
        if v1 in range(1, self.width + 1) or v2 in range(1, self.width + 1):
            if not (v2 - v1 == self.width + 1 or v2 - v1 == 1):
                return False
        ## right edge
        if v1 in range(self.width, self.max_vertex + 1, self.width + 1) or v2 in range(self.width, self.max_vertex + 1, self.width + 1):
            if not (v2 - v1 == self.width + 1 or (v2 - v1 and v2%(self.width + 1) != 0) == 1):
                return False
        ## top edge
        if v1 in range(self.max_vertex - self.width + 1, self.max_vertex) or v2 in range(self.max_vertex - self.width + 1, self.max_vertex):
            if not (v2 - v1 == self.width + 1 or v2 - v1 == 1):
                return False
        #GENERAL CASE
        if not (v2 - v1 == self.width + 1 or v2 - v1 == 1): return False
        
        #If there are no roads, it must be connected to a settlement or a city
        if len(self.roads) == 0:
            if v1 not in self.settlements and v2 not in self.settlements and v1 not in self.cities and v2 not in self.cities:
                return False

        #Otherwise, it must be connected to another road
        elif len(self.roads) != 0:
            if v1 not in set([element for tupl in self.roads for element in tupl]) and v2 not in set([element for tupl in self.roads for element in tupl]):
                return False
        return True
        
    
    def build(self, x, y, building):
        """build either a city or a settlement"""
        if self.if_can_build(building, x, y):
            vertex = self.get_vertex_number(x, y)
            if building == "settlement":
                self.settlements.append(vertex)
            elif building == "city":
                if vertex not in self.settlements:
                    raise CatanException("A settlement must be built first.")
                self.cities.append(vertex)
                self.settlements.remove(vertex)
            else:
                raise CatanException("{0} is an unknown building. Please use 'city' or 'settlement'.".format(building))
        else:
            raise CatanException("Cannot build {0} here. Please check if_can_build before building".format(building))
            
    
    def if_can_build(self, building, x, y):
        """returns true if spot (x,y) is available, false otherwise"""
        if x< 0 or y<0 or x > self.width+1 or y > self.height + 1:
            raise CatanException("({0},{1}) is an invalid vertex".format(x,y))
        #first let's check that the spot is empty:
        if self.get_vertex_number(x,y) in self.cities:
            return False

        ## upgrading first settlment into a city
        if (building == "city"):
            return self.get_vertex_number(x, y) in self.settlements
        ## If no cities, or settlements, build for freebies, otherwise need road connecting.
        if (len(self.settlements) + len(self.cities) != 0 and self.get_vertex_number(x, y) not in set([element for tupl in self.roads for element in tupl])):
            return False
        for x1 in range(x-1,x+2):
            for y1 in range(y-1,y+2):
                if x1+y1 < x+y-1 or x1+y1 > x+y+1 or y1-x1 < y-x-1 or y1-x1 > y-x+1: ## only interested in up, down, left, and right
                    pass
                elif x1 < 0 or x1 > self.width or y1 < 0 or y1 > self.height: ## only interested in valid tiles
                    pass
                elif self.get_vertex_number(x1, y1) in self.settlements or self.get_vertex_number(x1, y1) in self.cities:
                    return False
        return True

    def get_resources(self):
        """Returns array r where:
        r[i, :] = resources gained from throwing a (i+2)"""
        r = np.zeros((11, 3)) 
        for vertex in self.settlements:
            x, y = self.get_vertex_location(vertex)
            for dx in [-1, 0]:
                for dy in [-1, 0]:
                    xx = x + dx
                    yy = y + dy
                    if self.is_tile(xx, yy):
                        die = self.dice[yy, xx]
                        resource = self.resources[yy, xx]
                        r[die - 2, resource] += 1
        for vertex in self.cities:
            x, y = self.get_vertex_location(vertex)
            for dx in [-1, 0]:
                for dy in [-1, 0]:
                    xx = x + dx
                    yy = y + dy
                    if self.is_tile(xx, yy):
                        die = self.dice[yy, xx]
                        resource = self.resources[yy, xx]
                        r[die - 2, resource] += 2
        return r
    
    def draw(self):
        print("Drawing...")
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.set_xlim(-0.02,self.width+0.02)
        ax.set_ylim(-0.02,self.height+0.02)
        ax.set_frame_on(False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        for x in range(self.width):
            for y in range(self.height):
                color = ["brown", "red", "green"][self.resources[y, x]]
                ax.add_patch(patches.Rectangle((x, y),1,1, 
                                               facecolor=color))
                if self.dice[y,x] != 0:
                    ax.text(x+0.5, y+0.5, str(self.dice[y, x]), fontsize=15)
        ## draw roads
        for road in self.roads:
            x0, y0 = self.get_vertex_location(road[0])
            x1, y1 = self.get_vertex_location(road[1])
            #vertical road
            if x0 == x1:
                ax.add_patch(patches.Rectangle((x0 - 0.05, (y0 + y1)/2 + 0.05), 0.1, 0.9,
                                               facecolor = "white"))
            #horizontal road
            elif y0 == y1:
                ax.add_patch(patches.Rectangle(((x0 + x1)/2 + 0.05, y0 - 0.05), 0.9, 0.1,
                                                facecolor = "white"))
        for vertex in self.settlements:
            x, y = self.get_vertex_location(vertex)
            ax.add_patch(patches.Rectangle((x-0.1, y-0.1),0.2,0.2, 
                                           facecolor="purple"))
            ax.text(x-0.05, y-0.09, "1", fontsize=15, color="white")
        for vertex in self.cities:
            x, y = self.get_vertex_location(vertex)
            ax.add_patch(patches.Rectangle((x-0.1, y-0.1),0.2,0.2, 
                                           facecolor="blue")) 
            ax.text(x-0.05, y-0.09, "2", fontsize=15, color="white")
            
            

class Player:    
    def __init__(self, action, board, resources, points = 0, turn_counter = 0):
        self.board = board
        self.action = action
        self.resources = resources
        self.points = points
        self.turn_counter = turn_counter
        self.plan = [] #WE ADDED THIS FIELD
        
    def if_can_buy(self, item):
        if item == "card":
            return np.all(self.resources >= costs[CARD,:])
        elif item == "settlement":
            return np.all(self.resources >= costs[SETTLEMENT,:])
        elif item == "city":
            return np.all(self.resources >= costs[CITY,:])
        elif item == "road":
            return np.all(self.resources >= costs[ROAD,:])
        else:
            raise CatanException("Unknown item: {0}".format(item))

    def buy(self, item, x=-1,y=-1):
        if item == "card":
            self.points += 1
            self.resources = np.subtract(self.resources,costs[1])
        elif item == "road": #input should be of format board.buy("road", (1,1), (1,2))
            v0 = self.board.get_vertex_number(x[0], x[1])
            v1 = self.board.get_vertex_number(y[0], y[1])
            if self.board.if_can_build_road(v0, v1):
                self.board.build_road(x, y)
                self.resources = np.subtract(self.resources, costs[ROAD,:])
        elif (item == "settlement" or item == "city") and self.board.if_can_build(item,x,y):
            self.board.build(x,y,item)
            if item == "settlement":
                self.points += 1
                self.resources = np.subtract(self.resources,costs[SETTLEMENT,:])
            else:
                self.points += 1
                self.resources = np.subtract(self.resources,costs[CITY,:])

    #Trading
    def trade(self, r_in, r_out):
        required = 4
        ports = []
        for e in self.board.settlements:
            if self.board.is_port(e):
                ports.append(self.board.which_port(e))
        for e in self.board.cities:
            if self.board.is_port(e):
                ports.append(self.board.which_port(e))
        if r_in in ports:
            required = 2
        if self.resources[r_in] < required or self.resources[r_out] == MAX_RESOURCES:
            raise CatanException("Invalid trade.")
        if 3 in ports:
            required = min(required, 3)
        self.resources[r_in] -= required
        self.resources[r_out] += 1
    
    def play_round(self):
        dice_roll = np.random.randint(1,7)+np.random.randint(1,7)

        # collect resources
        collected_resources = self.board.get_resources()[dice_roll-2,:]
        self.resources = np.add(self.resources,collected_resources)
        self.resources = np.minimum(self.resources, MAX_RESOURCES) # LIMIT IS MAX # OF RESOURCES
            
        # perform action
        self.action(self, self.resources, costs)
        assert np.max(self.resources) < LIMIT
        
        # update the turn counter
        self.turn_counter += 1
        
        return dice_roll

def simulate_game(action, board, num_trials):
    """Simulates 'num_trials' games with policy 'action' and returns average length of games"""
    results = list()
    for _ in xrange(num_trials):
        resources = np.array([START_RESOURCES, START_RESOURCES, START_RESOURCES])
        live_board = Catan(board.dice, board.resources, [], [], [])
        player = Player(action, live_board, resources)

        while player.points < MAX_POINTS:
            if player.turn_counter > 1000000:
                raise CatanException("possible infinite loop (over 1M turns)")
                break
            player.play_round()
        results.append(player.turn_counter)
    
    return np.sum(results)/float(num_trials)

def simulate_game_and_save(action, board):
    """Simulates 'num_trials' games with policy 'action' and returns average length of games"""
    results = list()
    
    resources = np.array([START_RESOURCES, START_RESOURCES, START_RESOURCES])
    live_board = Catan(board.dice, board.resources, [], [], [])
    player = Player(action, live_board, resources)
    
    settlements = []
    cities = []
    hands = []
    live_points = []
    dice_rolls = []

    while player.points < MAX_POINTS:
        if player.turn_counter > 1000000:
            raise CatanException("possible infinite loop (over 1M turns)")
            break
        dice_roll = player.play_round()
        dice_rolls.append(dice_roll)
        settlements.append(live_board.settlements[:])
        cities.append(live_board.cities[:])
        hands.append(player.resources[:])
        live_points.append(player.points)
    
    return settlements, cities, hands, live_points, dice_rolls

def get_random_dice_arrangement(width, height):
    """returns a random field of dice"""
    ns = range(2, 13) * (width * height // 10 + 1)
    ns = ns[:width*height]
    np.random.shuffle(ns)
    ns = np.reshape(ns, (height, width))
    return ns

##############################################################################################
################################### Code we wrote ############################################
##############################################################################################
def action(self, resources, costs): #aka my_action_more_selective
    resource_from_rolls = rolls_to_receive_each_resource(self)

    #buy starting settlement, if is first turn
    if self.board.settlements == [] and self.board.cities == []:
        best_starting_vertex = choose_best_starting_settlement(self.board)
        x, y = self.board.get_vertex_location(best_starting_vertex)
        self.buy("settlement", x, y)
        return

    num_villages = min(2, 10-self.points)
    if not self.plan or len(self.plan) == 0:
        selectivity = 2
        self.plan, min_ttw = compute_n_length_goal_more_selective(self, resources, costs, num_villages, selectivity)
        rpt = np.array(total_resources_per_turn(self.board))
        if np.count_nonzero(rpt) == 3:
            card_rate = np.min(np.array(costs[CARD,:]) / rpt)
            win_with_cards_time = (10-self.points)/card_rate
            if win_with_cards_time < min_ttw:
               self.plan = ["card" for _ in range(10-self.points + 1) ]
    next_build = self.plan[0]
    if next_build == "card":
        goal_state = costs[CARD,:]
        if self.if_can_buy("card"):
            self.buy("card")
            action(self, self.resources, costs)
            return
    elif type(next_build) == tuple:
        #road
        item = "road"
        goal_state = costs[ROAD,:]
    elif next_build < 0 or (next_build == 0 and next_build in self.board.settlements):
        item = "city"
        goal_state = costs[CITY,:]
        x, y = self.board.get_vertex_location(-next_build)
    else:
        item = "settlement"
        goal_state = costs[SETTLEMENT,:]
        x, y = self.board.get_vertex_location(next_build)
    if reached_goal(resources, goal_state):
        if item == "road":
            v1, v2 = next_build
            x1, y1 = self.board.get_vertex_location(v1)
            x2, y2 = self.board.get_vertex_location(v2)
            self.buy("road", (x1,y1), (x2,y2))
        else:
            self.buy(item, x, y)
        self.plan = self.plan[1:]
        action(self, self.resources, costs)
        return
    for i in range(len(resource_from_rolls)):
        curr_res_count = self.resources[i]
        if curr_res_count == 5 or curr_res_count == 6:
            make_trade_for_most_lacking(self, i)
        res = resource_from_rolls[i]
        if len(res) == 0:
            #never receive particular resource
            #need to trade for resource
            make_best_trade_for_desired_res(self, i)

# def my_action(self, resources, costs):
#     resource_from_rolls = rolls_to_receive_each_resource(self)
    
#     #buy starting settlement, if is first turn
#     if self.board.settlements == [] and self.board.cities == []:
#         best_starting_vertex = choose_best_starting_settlement(self.board)
#         x, y = self.board.get_vertex_location(best_starting_vertex)
#         self.buy("settlement", x, y)
#         return
#     # choose n = points_needed-2 best purchases possible(settlements/roads, cities), the settlements need to be connected
#     # will buy 10-n cards at the end
#     num_villages = 1#4 - (self.points//2 + 1)
#     num_villages = max(1, num_villages)
#     if not self.plan or len(self.plan) == 0:
#         self.plan, min_ttw = compute_n_length_goal_more_selective(self, resources, costs, num_villages)
#     next_build = self.plan[0]
#     if next_build == "card":
#         goal_state = costs[CARD,:]
#         if self.if_can_buy("card"):
#             self.buy("card")
#             my_action(self, self.resources, costs)
#             return

#     elif type(next_build) == tuple:
#         #road
#         item = "road"
#         goal_state = costs[ROAD,:]
#     elif next_build < 0 or (next_build == 0 and next_build in self.board.settlements):
#         item = "city"
#         goal_state = costs[CITY,:]
#         x, y = self.board.get_vertex_location(-next_build)
#     else:
#         item = "settlement"
#         goal_state = costs[SETTLEMENT,:]
#         x, y = self.board.get_vertex_location(next_build)
#     if reached_goal(resources, goal_state):
#         if item == "road":
#             v1, v2 = next_build
#             x1, y1 = self.board.get_vertex_location(v1)
#             x2, y2 = self.board.get_vertex_location(v2)
#             self.buy("road", (x1,y1), (x2,y2))
#         else:
#             self.buy(item, x, y)
#         self.plan = self.plan[1:]
#         my_action(self, self.resources, costs)
#         return

#     for i in range(len(resource_from_rolls)):
#         curr_res_count = self.resources[i]
#         if curr_res_count == 5 or curr_res_count == 6:
#             make_trade_for_most_lacking(self, i)
#         res = resource_from_rolls[i]
#         if len(res) == 0:
#             #never receive particular resource
#             #need to trade for resource
#             make_best_trade_for_desired_res(self, i)

# def my_action_greedy(self, resources, costs):
#     resource_from_rolls = rolls_to_receive_each_resource(self)
    
#     #buy starting settlement, if is first turn
#     if self.board.settlements == [] and self.board.cities == []:
#         best_starting_vertex = choose_best_starting_settlement(self.board)
#         x, y = self.board.get_vertex_location(best_starting_vertex)
#         self.buy("settlement", x, y)
#         return
#     # choose n = points_needed-2 best purchases possible(settlements/roads, cities), the settlements need to be connected
#     # will buy 10-n cards at the end
#     num_villages = 1#4 - (self.points//2 + 1)
#     self.plan, min_ttw = compute_n_length_goal_more_selective(self, resources, costs, num_villages)
#     rpt = np.array(total_resources_per_turn(self.board))
#     if np.count_nonzero(rpt) == 3:
#         card_rate = np.min(np.array(costs[CARD,:]) / rpt)
#         win_with_cards_time = (10-self.points)/card_rate
#         if win_with_cards_time < min_ttw:
#             self.plan = ["card" for _ in range(10-self.points + 1) ]

#     next_build = self.plan[0]
#     if next_build == "card":
#         goal_state = costs[CARD,:]
#         if self.if_can_buy("card"):
#             self.buy("card")
#             my_action(self, self.resources, costs)
#             return
#     elif type(next_build) == tuple:
#         #road
#         item = "road"
#         goal_state = costs[ROAD,:]
#     elif next_build < 0 or (next_build == 0 and next_build in self.board.settlements):
#         item = "city"
#         goal_state = costs[CITY,:]
#         x, y = self.board.get_vertex_location(-next_build)
#     else:
#         item = "settlement"
#         goal_state = costs[SETTLEMENT,:]
#         x, y = self.board.get_vertex_location(next_build)
#     if reached_goal(resources, goal_state):
#         if item == "road":
#             v1, v2 = next_build
#             x1, y1 = self.board.get_vertex_location(v1)
#             x2, y2 = self.board.get_vertex_location(v2)
#             self.buy("road", (x1,y1), (x2,y2))
#         else:
#             self.buy(item, x, y)
#         self.plan = self.plan[1:]
#         my_action(self, self.resources, costs)
#         return

#     for i in range(len(resource_from_rolls)):
#         curr_res_count = self.resources[i]
#         if curr_res_count == 5 or curr_res_count == 6:
#             make_trade_for_most_lacking(self, i)
#         res = resource_from_rolls[i]
#         if len(res) == 0:
#             #never receive particular resource
#             #need to trade for resource
#             make_best_trade_for_desired_res(self, i)

##############################################################################################
###################################### Helpers ###############################################
##############################################################################################

def rolls_to_receive_each_resource(player):
    gettable = [[], [], []]
    for roll in range(12):
        resources = player.board.get_resources()[roll-2,:]
        for i in range(len(resources)):
            r = resources[i]
            if r != 0:
                gettable[i].append(roll)
    return gettable
def make_best_trade_for_desired_res(player, desired_res):
    p_copy = copy.deepcopy(player)
    best_res_to_trade = None
    best_loss = 6
    for held_res in range(3):
        try:
            p_copy.trade(held_res, desired_res)
            for res_i in range(3):
                if p_copy.resources[res_i] < player.resources[res_i]:
                    lost = player.resources[res_i] - p_copy.resources[res_i]
                    if lost < best_loss:
                        best_loss = lost
                        best_res_to_trade = held_res
        except:
            pass
    if best_res_to_trade:
        player.trade(best_res_to_trade, desired_res)

def choose_best_starting_settlement(board):
    all_three = []
    just_two = []
    just_one = []
    for vertex in range(board.max_vertex):
        r_per_turn = resources_per_turn_from_tile(board, vertex)
        if np.count_nonzero(r_per_turn) == 3:
            all_three.append((vertex, r_per_turn))
        elif np.count_nonzero(r_per_turn) == 2:
            just_two.append((vertex, r_per_turn))
        else:
            just_one.append((vertex, r_per_turn))
    max_total_rate = 0
    best_location = None
    if all_three:
        for vertex in all_three:
            total_rate = sum(vertex[1])
            if total_rate > max_total_rate:
                max_total_rate = max(total_rate, max_total_rate)
                best_location = vertex
        return best_location[0]
    if just_two:
        for vertex in just_two:
            total_rate = sum(vertex[1])
            if total_rate > max_total_rate:
                max_total_rate = max(total_rate, max_total_rate)
                best_location = vertex
        return best_location[0]
    for vertex in just_one:
        total_rate = sum(vertex[1])
        if total_rate > max_total_rate:
            max_total_rate = max(total_rate, max_total_rate)
            best_location = vertex
    return best_location[0]

def choose_n_best_starting_settlement(board, vertices, n):
    all_three = []
    just_two = []
    just_one = []
    for vertex in vertices: #range(board.max_vertex):
        r_per_turn = resources_per_turn_from_tile(board, vertex)
        if np.count_nonzero(r_per_turn) == 3:
            all_three.append((vertex, r_per_turn))
        elif np.count_nonzero(r_per_turn) == 2:
            just_two.append((vertex, r_per_turn))
        else:
            just_one.append((vertex, r_per_turn))
    best_locations3 = [(None,0) for _ in range(min(len(all_three), n))] #vertex, rate
    worst_of_best = (0, 0) #(rate,idx in best_locations)
    if all_three:
        for vertex in all_three:
            total_rate = sum(vertex[1])
            if total_rate > worst_of_best[0]:
                worst_rate = total_rate
                worst_idx = worst_of_best[1]
                best_locations3[worst_idx] = (vertex[0], total_rate) 
                for i in range(len(best_locations3)):
                    rate_of_ith = best_locations3[i][1]
                    if rate_of_ith < worst_rate:
                        worst_rate = rate_of_ith
                        worst_idx = i
                worst_of_best = (worst_rate, worst_idx)
    best_locations3 = [vertex for vertex, rate in best_locations3]
    if len(all_three) == n:
        return best_locations3
    else:
        n -= len(all_three)
    best_locations2 = [(None,0) for _ in range(min(len(just_two), n))] #vertex, rate
    worst_of_best = (0, 0) #(rate,idx in best_locations)
    if just_two:
        for vertex in just_two:
            total_rate = sum(vertex[1])
            if total_rate > worst_of_best[0]:
                worst_rate = total_rate
                worst_idx = worst_of_best[1]
                best_locations2[worst_idx] = (vertex[0], total_rate) 
                for i in range(len(best_locations2)):
                    rate_of_ith = best_locations2[i][1]
                    if rate_of_ith < worst_rate:
                        worst_rate = rate_of_ith
                        worst_idx = i
                worst_of_best = (worst_rate, worst_idx)
    best_locations2 = [vertex for vertex, rate in best_locations2]
    if len(just_two) == n:
        return best_locations3 + best_locations2
    else:
        n -= len(just_two)
    best_locations1 = [(None,0) for _ in range(min(len(just_one), n))] #vertex, rate
    worst_of_best = (0, 0) #(rate,idx in best_locations)
    if just_one:
        for vertex in just_one:
            total_rate = sum(vertex[1])
            if total_rate > worst_of_best[0]:
                worst_rate = total_rate
                worst_idx = worst_of_best[1]
                best_locations1[worst_idx] = (vertex[0], total_rate) 
                for i in range(len(best_locations1)):
                    rate_of_ith = best_locations1[i][1]
                    if rate_of_ith < worst_rate:
                        worst_rate = rate_of_ith
                        worst_idx = i
                worst_of_best = (worst_rate, worst_idx)
    best_locations1 = [vertex for vertex, rate in best_locations1]
    return best_locations3 + best_locations2 + best_locations1


def total_resources_per_turn(board):
    total = np.array([0,0,0])
    for vertex in board.settlements:
        np.add(total, resources_per_turn_from_tile(board, vertex))
    for vertex in board.cities:
        np.add(total, resources_per_turn_from_tile(board, vertex))
    return total

def resources_per_turn_from_tile(board, vertex):
    r = np.zeros(3)
    x, y = board.get_vertex_location(vertex)
    for dx in [-1, 0]:
        for dy in [-1, 0]:
            xx = x + dx
            yy = y + dy
            if board.is_tile(xx, yy):
                die = board.dice[yy, xx]
                resource = board.resources[yy, xx]
                r[resource] += 1 * prob_rolling(die)
    return r

def prob_rolling(roll):
    return (6-abs(7-roll))/36

def make_trade_for_most_lacking(player, resource_to_give_away):
    min_num = 6
    min_idx = resource_to_give_away
    for res_idx in range(3):
        if res_idx == resource_to_give_away:
            continue
        res_count = player.resources[res_idx]
        if res_count < min_num:
            min_num = res_count
            min_idx = res_idx
    player.trade(resource_to_give_away, min_idx)

def compute_n_length_goal_more_selective(self, resources, costs, n, selectivity = 100000):
    current_settlements = self.board.settlements
    current_cities = self.board.cities
    perms = valid_permutations_more_selective(board, current_settlements, current_cities, n, selectivity)
    # for every permutation of purchases possible: 
    # calculate E[ttw] = sum(E[ttt_i,i+1] for i in range(n))
    best_perm = None
    min_ttw = 100000
    for perm in perms:
        # a state i is [r_1, r_2, r_3]
        # curr_player = copy.deepcopy(self)
        curr_resources = copy.copy(resources) #curr_player.resources#
        curr_board = copy.deepcopy(self.board) #curr_player.board
        curr_resources = tuple([int(r) for r in curr_resources])
        ttw = 0
        for i in range(len(perm)):
            if type(perm[i]) == tuple:
                item = "road"
                v1, v2 = perm[i]
                if (v1, v2) in curr_board.roads or (v2, v1) in curr_board.roads:
                    #don't need to build again
                    continue
                goal_state = costs[ROAD,:]
            elif perm[i] < 0 or (perm[i] == 0 and perm[i] in curr_board.settlements):
                item = "city"
                goal_state = costs[CITY,:]
                x, y = self.board.get_vertex_location(-perm[i])
            else:
                item = "settlement"
                goal_state = costs[SETTLEMENT,:]
                x, y = self.board.get_vertex_location(perm[i])
            if reached_goal(curr_resources, goal_state):
                ttw += 0
                curr_resources = np.add(curr_resources,-goal_state)
                curr_resources = tuple([int(r) for r in curr_resources])
                if item == "city" or item == "settlement":
                    curr_board.build(x,y,item) 
                else:
                    v1, v2 = perm[i]
                    x1, y1 = self.board.get_vertex_location(v1)
                    x2, y2 = self.board.get_vertex_location(v2)
                    # if (v1, v2) in curr_board.roads or (v2, v1) in curr_board.roads:
                    #     continue
                    curr_board.build_road((x1,y1), (x2,y2))
                continue
            # enumerate states between each pair of purchases
            int_states = get_intermediate_states(curr_resources, goal_state)
            goal_state = tuple([int(r) for r in goal_state])
            num_states = max(int_states.values()) + 1
            mat = -1 * np.identity(num_states, dtype = float)
            b = np.mat([-1 for _ in range(num_states)]).T
            # E[ttt_ij] found by solving system of equations E[ttt_ij] = 1+ sum(E[ttt_kj] * p_ik)
            # p_ik = sum(pr(roll) if val(roll) + i == k)
            for int_state in int_states:
                state_idx = int_states[int_state]
                for roll in range(2,13):
                    collected_resources = curr_board.get_resources()[roll-2,:]
                    temp_next_state = np.add(int_state,collected_resources)
                    temp_next_state = np.minimum(temp_next_state, MAX_RESOURCES) # LIMIT IS MAX # OF RESOURCES
                    res_idx = 0
                    changed = False
                    while(not changed and res_idx in range(len(temp_next_state))):
                        res = temp_next_state[res_idx]
                        if res == 5 or res == 6:
                            changed = True
                            temp_next_state[res_idx] -= 4
                            if res_idx == 0:
                                j = 1
                                k = 2
                            if res_idx == 1:
                                j = 0
                                k = 2
                            if res_idx == 2:
                                j = 0
                                k = 1
                            if temp_next_state[k] < temp_next_state[j]:
                                temp_next_state[k] += 1
                            else:
                                temp_next_state[j] += 1
                        res_idx += 1
                    temp_next_state = tuple([int(r) for r in temp_next_state])
                    if temp_next_state in int_states:
                        temp_next_state_idx = int_states[temp_next_state]
                        mat[state_idx, temp_next_state_idx] += prob_rolling(roll)
            ttts = np.linalg.solve(mat, b)
            # pinv = np.linalg.pinv(mat)
            # ttts = pinv * b#np.dot(pinv, b)
            ttt = ttts[int_states[curr_resources]]
            ttw += ttt
            if item == "city" or item == "settlement":
                curr_board.build(x,y,item)
            else:
                v1, v2 = perm[i]
                x1, y1 = self.board.get_vertex_location(v1)
                x2, y2 = self.board.get_vertex_location(v2)
                curr_board.build_road((x1,y1), (x2,y2))
            curr_resources =  np.add(np.add(curr_resources, total_resources_per_turn(curr_board)), -np.array(goal_state)) #(0,0,0)
            curr_resources = tuple([max(r, 0) for r in curr_resources])
        # use permuation with min E[ttw]
        if ttw < min_ttw and ttw >= 0:
            best_perm = perm
            min_ttw = ttw
    if best_perm:
        return list(best_perm), min_ttw
    else:
        return ["card" for _ in range(10-self.points + 1) ], min_ttw

def reached_goal(curr_state, goal_state):
    for i in range(len(curr_state)):
        if curr_state[i] < goal_state[i]:
            return False
    return True

def get_intermediate_states(curr_resources, goal_state):
    #returns dictionary which maps state (3 tuple) to index in matrix
    int_states = {}
    idx = 0
    for i in range(int(curr_resources[0]), MAX_RESOURCES + 1):
        for j in range(int(curr_resources[1]), MAX_RESOURCES + 1):
            for k in range(int(curr_resources[2]), MAX_RESOURCES + 1):
                if not reached_goal((i, j, k), goal_state) and (i, j, k) not in int_states:
                    int_states[(i, j, k)] = idx
                    idx += 1
    for i in range(int(curr_resources[0]), MAX_RESOURCES + 1):
        for k in range(int(curr_resources[2]), MAX_RESOURCES + 1):
            for j in range(int(curr_resources[1]), MAX_RESOURCES + 1):
                if not reached_goal((i, j, k), goal_state) and (i, j, k) not in int_states:
                    int_states[(i, j, k)] = idx
                    idx += 1
    for j in range(int(curr_resources[1]), MAX_RESOURCES + 1):
        for k in range(int(curr_resources[2]), MAX_RESOURCES + 1):
            for i in range(int(curr_resources[0]), MAX_RESOURCES + 1):
                if not reached_goal((i, j, k), goal_state) and (i, j, k) not in int_states:
                    int_states[(i, j, k)] = idx
                    idx += 1
    for j in range(int(curr_resources[1]), MAX_RESOURCES + 1):
        for i in range(int(curr_resources[0]), MAX_RESOURCES + 1):
            for k in range(int(curr_resources[2]), MAX_RESOURCES + 1):
                if not reached_goal((i, j, k), goal_state) and (i, j, k) not in int_states:
                    int_states[(i, j, k)] = idx
                    idx += 1
    for k in range(int(curr_resources[2]), MAX_RESOURCES + 1):
        for i in range(int(curr_resources[0]), MAX_RESOURCES + 1):
            for j in range(int(curr_resources[1]), MAX_RESOURCES + 1):
                if not reached_goal((i, j, k), goal_state) and (i, j, k) not in int_states:
                    int_states[(i, j, k)] = idx
                    idx += 1
    for k in range(int(curr_resources[2]), MAX_RESOURCES + 1):
        for j in range(int(curr_resources[1]), MAX_RESOURCES + 1):
            for i in range(int(curr_resources[0]), MAX_RESOURCES + 1):
                if not reached_goal((i, j, k), goal_state) and (i, j, k) not in int_states:
                    int_states[(i, j, k)] = idx
                    idx += 1
    return int_states

def valid_permutations_more_selective(board, current_settlements, current_cities, n, selectivity = 100000):
    if n==0:
        return [[]]
    valids = []
    #all permuations which start with upgrading current settlement
    for vertex in current_settlements:
        if current_cities:
            new_current_cities = copy.copy(current_cities)
            new_current_cities.append(vertex)
        else:
            new_current_cities = [vertex]
        new_current_settlements = copy.copy(current_settlements)
        new_current_settlements.remove(vertex)
        short_permutations = valid_permutations_more_selective(board, new_current_settlements, new_current_cities, n-1, selectivity)
        for short_permutation in short_permutations:
            valids.append([-vertex] + short_permutation)
    
    #all permutations which start with building road and then new settlement
    if current_cities:
        current_both = set(current_cities + current_settlements)
    else:
        current_both = set(current_settlements)
    tried = set()
    for vertex in current_both:
        next_picks = [v for v in vertices_two_away(board, vertex) if v not in tried and v not in current_both]
        next_picks = choose_n_best_starting_settlement(board, next_picks, selectivity)
        for next_pick in next_picks:
            new_current_settlements = copy.copy(current_settlements)
            new_current_settlements.append(next_pick)
            roads_between = find_roads_between(board, vertex, next_pick)
            short_permutations = valid_permutations_more_selective(board, new_current_settlements, current_cities, n-1, selectivity)
            for short_permutation in short_permutations:
                for road in roads_between:
                    if road in short_permutation:
                        short_permutation.remove(road)
                    elif (road[1], road[0]) in short_permutation:
                        short_permutation.remove((road[1], road[0]))
                valids.append(roads_between + [next_pick] + short_permutation)
            
            tried.add(next_pick)
    return valids

def find_roads_between(board, v1, v2):
    x1, y1 = board.get_vertex_location(v1)
    x2, y2 = board.get_vertex_location(v2)
    if x1 == x2 or y1 == y2:
        #only one way
        x_between = (x1+x2)/2
        y_between = (y1+y2)/2        
        v_between = board.get_vertex_number(x_between, y_between)
        roads = [(v1, v_between), (v_between, v2)]
        return roads
    vb1 = board.get_vertex_number(x1, y2) #move y first
    vb2 = board.get_vertex_number(x2, y1) #move x first
    if (v1, vb1) in board.roads or (vb1, v1) in board.roads:
        return [(vb1, v2)]
    if (vb1, v2) in board.roads or (v2, vb1) in board.roads:
        return [(v1, vb1)]
    if (v1, vb2) in board.roads or (vb2, v1) in board.roads:
        return [(vb2, v2)]
    if (vb2, v2) in board.roads or (v2, vb2) in board.roads:
        return [(v1, vb2)]
    else:
        return [(v1, vb2), (vb2, v2)]

def vertices_two_away(board, vertex):
    x, y = board.get_vertex_location(vertex)
    coords = [(x-2,y), (x-1, y+1), (x-1, y-1), (x, y-2), (x,y+2), (x+1, y-1), (x+1, y+1), (x+2, y)]
    coords = [(x,y) for x,y in coords if x > 0 and x < board.width and y > 0 and y < board.height]
    return [board.get_vertex_number(x, y) for x,y in coords]



# if __name__ == "__main__":
#     num_trials = 10
#     width, height = 4, 4
#     dice = get_random_dice_arrangement(width, height)
#     resources = np.random.randint(0, 3, (height, width))
#     board = Catan(dice, resources)
    
#     print(datetime.now())
#     print simulate_game(my_action, board, num_trials)

#     print(datetime.now())
#     print simulate_game(my_action_greedy, board, num_trials)

    # print(datetime.now())
    # print simulate_game(action, board, num_trials)


    # print(datetime.now())
#     print simulate_game(dumb_action, board, num_trials)
#     print(datetime.now())