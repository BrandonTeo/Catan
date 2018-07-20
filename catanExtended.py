import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import math
import random

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
        #Which spot will have a robber
        self.robberX = random.randint(0,self.width-1) 
        self.robberY = random.randint(0,self.height-1)
        self.robberDice = self.dice[self.robberX][self.robberY]
        self.currTurn = 0
        self.changeRobber = 5  # Move the robber after this number of steps

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

        ## upgrading first settlement into a city
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
    
    def get_resources1(self, dice_roll):
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
                        #Account for the fact that resources don't spawn on the tile the robber is on
                        if not (die == self.robberDice and dice_roll == self.robberDice and self.robberX == xx and self.robberY == yy):
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
                        #Account for the fact that resources don't spawn on the tile the robber is on
                        if not (die == self.robberDice and dice_roll == self.robberDice and self.robberX == xx and self.robberY == yy):
                            r[die - 2, resource] += 2
        return r


    def rob(self, myResource):
        #When the robber randomly moves onto a spot he will randomly destroy half of the resources of players
        #that have settlements beside the robber's location with more than 'cap' resources
        cap = 5
        for vertex in self.settlements:
            x, y = self.get_vertex_location(vertex)
            for dx in [-1, 0]:
                for dy in [-1, 0]:
                    xx = x + dx
                    yy = y + dy
                    if self.is_tile(xx, yy):
                        die = self.dice[yy, xx]
                        if xx == self.robberX and yy == self.robberY:
                            summ = myResource[0] + myResource[1] + myResource[2]
                            if summ > cap:
                                newResource = [0,0,0]
                                rList = []
                                for i in range(3):
                                    numResource = myResource[i]
                                    for j in range(int(numResource)):
                                        rList.append(i)
                                random.shuffle(rList)
                                remainList = rList[:int(summ/2)]
                                for k in remainList:
                                    newResource[k] += 1
                                return newResource
        for vertex in self.cities:
            x, y = self.get_vertex_location(vertex)
            for dx in [-1, 0]:
                for dy in [-1, 0]:
                    xx = x + dx
                    yy = y + dy
                    if self.is_tile(xx, yy):
                        die = self.dice[yy, xx]
                        if xx == self.robberX and yy == self.robberY:
                            summ = myResource[0] + myResource[1] + myResource[2]
                            if summ > cap:
                                rList = []
                                newResource = [0,0,0]
                                for i in range(3):
                                    numResource = myResource[i]
                                    for j in range(int(numResource)):
                                        rList.append(i)
                                random.shuffle(rList)
                                remainList = rList[:int(summ/2)]
                                for k in remainList:
                                    newResource[k] += 1
                                return newResource
        return myResource

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
        # debts = [# wood owed, # brick owed, # grain owed]
        self.debts = np.array([0, 0, 0])
        
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

    def advance_pay(self, resource):
        """ resource: 'wood, 'brick', or 'grain'
        If a given resource is accessible from current settlements/cities,
        advance_pay(resource) may be used once per turn, when valid,
        to gain one resource in advance. In exchange, 2 of that resource
        that were originally supposed to be gained in the next 1 or 2 turns
        will be collected and not rewarded. Also, until the 2 resources have
        been repaid, player cannot advance_pay for that resource type.
        """
        accessible_resources = self.board.get_resources().sum(axis=0)
        if (resource == 'wood'):
            if (self.debts[0] == 0 and accessible_resources[0] > 0):
                self.resources = np.add(self.resources,np.array([1, 0, 0]))
                self.debts[0] = 2
        elif (resource == 'brick'):
            if (self.debts[1] == 0 and accessible_resources[1] > 0):
                self.resources = np.add(self.resources, np.array([0, 1, 0]))
                self.debts[1] = 2
        elif (resource == 'grain'):
            if (self.debts[2] == 0 and accessible_resources[2] > 0):
                self.resources = np.add(self.resources, np.array([0, 0, 1]))
                self.debts[2] = 2
        else:
            print('Cannot advance pay for:', resource)

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
        #Move rubber if self.board.changeRobber steps have passed since last move
        if self.board.currTurn % self.board.changeRobber == 0:
            #Robber moves and does cap check on the spot it lands
            self.board.robberX = random.randint(0,self.board.width-1)
            self.board.robberY = random.randint(0,self.board.height-1)
            self.board.robberDice = self.board.dice[self.board.robberX][self.board.robberY]
            resourceCopy = np.copy(self.resources)
            replacementResource = self.board.rob(resourceCopy)
            self.resources = np.array(replacementResource)

        dice_roll = np.random.randint(1,7)+np.random.randint(1,7)
        # collect resources
        collected_resources = self.board.get_resources1(dice_roll)[dice_roll-2,:]
        ##### Additional: also pay debt if there is debt
        debt_after_pay = np.subtract(self.debts, collected_resources)
        for i in range(0,3):
            if debt_after_pay[i] >= 0:
                collected_resources[i] = 0
            else:
                collected_resources[i] = np.negative(debt_after_pay[i])
        self.debts = debt_after_pay.clip(min=0) # keep unpaid debts
        #####
        self.resources = np.add(self.resources,collected_resources)
        self.resources = np.minimum(self.resources, MAX_RESOURCES) # LIMIT IS MAX # OF RESOURCES

        # perform action
        self.action(self, self.resources, costs)
        assert np.max(self.resources) < LIMIT
        
        # update the turn counter
        self.turn_counter += 1
        self.board.currTurn += 1

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
    ns = range(2, 13) * (width * height / 10 + 1)
    ns = ns[:width*height]
    np.random.shuffle(ns)
    ns = np.reshape(ns, (height, width))
    return ns


# sample action function utilizing features
def sample_action(self, resources, costs):
    # inputs:
    # resources - an array of resources
    # costs - an array of costs, 0 - settlement, 1 - card, 2 - city
    # basic strategy: Once we get 4 of one resource, we make a trade.
    # Then we try to buy development cards
    if self.board.settlements == []:
        x = np.random.randint(1, self.board.width)
        y = np.random.randint(1, self.board.height)
        self.buy("settlement", x, y)
#         self.board.draw()
    elif self.if_can_buy("card"):
        self.advance_pay('wood')
        self.buy("card")
    elif self.resources[np.argmax(self.resources)] >= 4:
        rmax, rmin = np.argmax(self.resources), np.argmin(self.resources)
        self.trade(rmax,rmin)
    self.advance_pay('wood')
    return
