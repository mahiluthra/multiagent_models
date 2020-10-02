# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 09:37:44 2020

@author: mahi
"""
from mesa import Model, Agent
import random
from mesa.time import RandomActivation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from mesa.time import BaseScheduler
import math
import pandas as pd
import sys
from mesa import space
import copy
from scipy.ndimage.filters import gaussian_filter

class A2(Agent):  # second level agent-- predator agent   
    
    """ starts at energy level 3 (unless parent has very high energy)
        eats 5 energy points
        tires 3 energy points
        reproduces at 10 energy points, splitting energy level with offspring
    """
    
    def __init__(self, unique_id, model, start_energy, cognition):
        super().__init__(unique_id, model)  # creates an agent in the world with a unique id
        
        # energy parameters
        self.energy = start_energy
        self.eat_energy = self.model.eat_energy
        self.tire_energy = self.model.tire_energy
        self.reproduction_energy = self.model.reproduction_energy
        self.cognition_energy = self.model.cognition_energy
        
        # other initiializations
        self.cognition = self.model.cognition
        self.dead = False
        self.identity = 2
        self.age = 0
        
        # movement parameters
        self.velocity = np.random.normal(1, 0.2)
        self.direction = np.random.uniform(-1, 1, 2)
        self.direction /= np.linalg.norm(self.direction)
                
    def step(self): # this iterates at every step
        if not self.dead:  # the agent moves on ev ery step   
            self.move()
        if not self.dead: # have to repeat because they might have died through cognition and move (e.g., combat)
            self.tire_die()
        if not self.dead:
            self.reproduce()
            self.age+=1
            self.model.age.append(self.age)
            
    def introduce(self, x, y, energy, cog):
        a = A2(self.model.unique_id, self.model, start_energy = energy, cognition = cog)
        self.model.unique_id += 1
        self.model.grid.place_agent(a, (x,y))
        self.model.schedule.add(a)
        
    def kill(self):
        self.dead=True
        x,y = self.pos
        self.model.grid.remove_agent(self) 
        self.model.schedule.remove(self)
        self.model.death += 1
                
    def eat(self, coord, eaten_energy = 0):
        avail_food = [agent for agent in self.model.grid.get_neighbors([coord], radius = 1) if agent.identity==1]
        if len(avail_food)==0:
            return
        hungry_energy = self.eat_energy - eaten_energy
        food = random.choice(avail_food)
        if food.energy>hungry_energy:
            food.energy-=hungry_energy
            self.energy += hungry_energy
            # print("fooood")
            return
        if food.energy==hungry_energy:
            self.energy += hungry_energy
            food.dead = True
            self.model.grid.remove_agent(food)
            self.model.schedule.remove(food)   
            #print("fooood")
            return
        if food.energy<hungry_energy:
            food_energy = food.energy
            self.energy += food_energy    
            food.dead = True
            self.model.grid.remove_agent(food)
            self.model.schedule.remove(food)
            self.eat(coord, eaten_energy = eaten_energy+food_energy)
            #print("fooood")

    def reproduce(self): # reproduce function
        if self.energy >= self.reproduction_energy:
            self.model.reprod += 1
            if self.model.disp_rate == 1:
                x = random.randrange(self.model.grid.width)
                y = random.randrange(self.model.grid.height)
                new_position = (x,y)
            elif self.model.disp_rate == 0:
                new_position = self.model.grid.torus_adj(np.array(self.pos) + np.random.choice([1, -1], 2) * np.random.exponential(0.5, 2))
            
            self.energy -= 10
            energy_own = math.ceil(self.energy/2)
            energy_off = self.energy - energy_own
            self.energy = energy_own
            
            cog = self.cognition
            # add mutuation function
                                    
            x,y = new_position                
            self.introduce(x,y, energy_off, cog)
            
    def tire_die(self): 
        x,y = self.pos
        self.energy-=self.tire_energy # + (self.cognition[0]/10)
        if self.energy<=0:
            self.kill()
        
    def cogdecision(self):
      #  radius = 10
      #  food, locs, dists = self.model.grid.get_neighbors_locs(self.pos, radius)
      #  dists = dists/radius
        if self.model.cognition==1:
            ls = self.model.food_ls
            cir = np.array(self.model.points_on_circumference(self.pos, self.velocity))%100
            in_ = np.round(cir*2).astype(int)%199
            out = ls[in_[:,0], in_[:,1] ]
            move = cir[np.argmax(out)]
            return(tuple(move))
            
        if self.model.cognition==0:
            self.direction = np.random.uniform(-1, 1, 2)
            self.direction /= np.linalg.norm(self.direction)
            move = self.model.grid.torus_adj(np.array(self.pos) + self.direction * self.velocity)
            return(move)
        
    def convert_to_ls(self, point):
        point = np.array(point)
        return(np.round(point*2).astype(int)%199)
    
    def move(self):  
        self.energy-=self.cognition_energy  
        newx, newy = self.cogdecision()
        x,y = self.pos
        self.model.grid.move_agent(self, (newx, newy) )
        self.eat((newx, newy)) 
    
class A1(Agent):
    
    """ plants agent functions
    """
    
    def __init__(self, unique_id, model, start_energy):
        super().__init__(unique_id, model)

        self.energy = start_energy # agent starts at energy level 10
        self.eat_energy =  self.model.eat_energy
        self.tire_energy = self.model.tire_energy
        self.reproduction_energy = self.model.reproduction_energy
        self.dead = False
        self.identity = 1
        
    def step(self): # this iterates at every step
        self.eat()
        self.tire_die()
        if not self.dead:
            self.reproduce() 
            
    def reproduce(self):
        if self.energy >= self.reproduction_energy:
            if self.model.disp_rate == 1:
                x = random.randrange(self.model.grid.width)
                y = random.randrange(self.model.grid.height)
                new_position = (x,y)
            elif self.model.disp_rate == 0:
                new_position = self.model.grid.torus_adj(np.array(self.pos) + np.random.choice([1, -1], 2) * np.random.exponential(0.5, 2))
            
            self.energy -= 10
            energy_own = math.ceil(self.energy/2)
            energy_off = self.energy - energy_own
            self.energy = energy_own
            
            x,y = new_position
            if np.size(self.model.grid.get_neighbors([(x, y)], radius = 0.3))==0:                
                a = A1(self.model.unique_id, self.model, energy_off)
                self.model.unique_id += 1
                self.model.grid.place_agent(a, new_position)
                self.model.schedule.add(a)
            
    def eat(self): # agent eats at every step and thus depeletes resources          
        self.energy += self.eat_energy # nutrition is added to agent's nutrition
            
    def tire_die(self): # agent loses energy at every step. if it fails to eat regularly, it dies due to energy loss
        x,y = self.pos
        self.energy-=self.tire_energy
        if self.energy<=0:
            self.dead=True
            self.model.grid[x][y].remove(self) 
            self.model.schedule.remove(self)
            
class model(Model):
    
    def __init__(self, introduce_time, skip_300 = True):
        
        # initializations of 
        self.start_energy = 10
        self.eat_energy = 5
        self.tire_energy = 3
        self.reproduction_energy = 10
        self.cognition_energy = 1
        (self.a1num, self.a2num) = (50, 50)
        self.skip_300 = skip_300
        self.intro_time = introduce_time
        self.schedule = RandomActivation(self) # agents take a step in random order 
        self.cognition=0
        
        self.grid = space.ContinuousSpace(100, 100, True) # the world is a grid with specified height and width
        self.disp_rate = 0
        # data storage initialization
        self.age = []
        (self.nstep, self.unique_id, self.reprod, self.food, self.death, self.combat) = (0, ) * 6
        self.history = pd.DataFrame(columns = ["nA1", "nA2", "age", "neigh5","neigh10","neightanim5", "neightanim10", "reprod", "food", "death", 
                                       "combat", "dist", "det"])
        
        # initializations of plotting parameters
        self.cmap = colors.ListedColormap(['midnightblue', 'mediumseagreen', 'white', 'white', 'white', 'white', 'white'])#'yellow', 'orange', 'red', 'brown'])
        bounds=[0,1,2,3,4,5,6,7]
        self.norm = colors.BoundaryNorm(bounds, self.cmap.N)    
        
        # initializations for calculating patchiness of world
        self.distances1 = np.array([])
        self.distances2 = np.array([])
        self.expect_NN = []
        self.neigh = [5, 10]
        for i in self.neigh:
            self.expect_NN.append((math.factorial(2*i) * i)/(2**i * math.factorial(i))**2)

        # initialize resource agent
        for i in range(self.a1num):
            self.introduce_agents("A1")
            
    def introduce_agents(self, which_agent):
        x = random.randrange(self.grid.width)
        y = random.randrange(self.grid.height)
            
        if which_agent == "A1":
            a = A1(self.unique_id, self, self.start_energy)
            self.unique_id += 1
            self.grid.place_agent(a, (x, y) )
            self.schedule.add(a)
              
        elif which_agent == "A2":
            c = ()
            a = A2(self.unique_id, self, self.start_energy, cognition = c)
            self.unique_id += 1 
            self.grid.place_agent(a, (x,y))
            self.schedule.add(a)    
            
    def return_zero(self, num, denom):
        if self.nstep == 1:
            return(0)
        if denom == "old_nA2":
            denom = self.history["nA2"][self.nstep-2]
        if denom == 0.0:
            return 0
        return(num/denom)
        
    def nearest_neighbor(self, agent):
        if agent == "a1":
            x = np.argwhere(self.agentgrid==1)
            if len(x)<=10:
                return([-1]*len(self.neigh))
            elif len(x) > 39990:
                return([0.97, 0.99])
            if self.nstep<300 and self.skip_300:
                return([-1,-1] )
        else:
            x = np.argwhere(self.agentgrid==2)
            if len(x)<=10:
                return([-1]*len(self.neigh))
        density = len(x)/ (self.grid.width)**2
        expect_neigh_ = self.expect_neigh
        expect_dist = np.array(expect_neigh_) /(density ** 0.5)
        distances = [0, 0]
        for i in x:   # calculates pairwise distances in a toroid
            distx = abs(x[:,0]-i[0])
            distx[distx>100] = 200-distx[distx>100]
            disty = abs(x[:,1]-i[1])
            disty[disty>100] = 200-disty[disty>100]
            dist = (distx**2+disty**2)**0.5
            distances[0] += (np.partition(dist, 5)[5])
            distances[1] += 0#np.partition(dist, 10)[10])
        mean_dist = np.array(distances)/len(x)
        out = mean_dist/expect_dist
        return(out)
    
    def collect_hist(self):
        neigh_calc = [0,0]#self.nearest_neighbor("a1") 
        neigh_animcalc = [0,0]#self.nearest_neighbor("a2")
        dat = { "nA1" : self.nA1, "nA2" : self.nA2,
               "age" : self.return_zero(sum(self.age), self.nA2),
               "neigh5": neigh_calc[0],"neigh10": neigh_calc[1],
               "neighanim5": neigh_animcalc[0],"neighanim10": neigh_animcalc[1],
               "reprod" : self.return_zero(self.reprod, "old_nA2" ), "food": self.return_zero(self.food, self.nA2),
               "death" : self.return_zero(self.death, "old_nA2"), "combat" : self.return_zero(self.combat, "old_nA2")}
        self.history = self.history.append(dat, ignore_index = True)
        self.age = []
        (self.reprod, self.food, self.death, self.combat) = (0, ) * 4
  
    def step(self):
        self.nstep +=1 # step counter
        self.food_ls = self.get_food_ls()
        if self.nstep == self.intro_time:
            for i in range(self.a2num):
                self.introduce_agents("A2")  
        self.schedule.step()  
        self.nA1 = np.sum(self.grid._agent_ids==1)            
        self.nA2 = np.sum(self.grid._agent_ids==2)
        self.collect_hist()
        if self.nstep%10 == 0:
            sys.stdout.write( (str(self.nstep) +" "  +str(self.nA1) + " " + str(self.nA2) + "\n") )
        
    def animate(self):
        colors = ['midnightblue', 'mediumseagreen', 'white']
        plot_c = [colors[aid] for aid in self.grid._agent_ids]
        n = str(self.nstep)
        fig = plt.scatter(self.grid._agent_points[:, 0], self.grid._agent_points[:, 1], \
                          c = plot_c, s = self.grid._agent_ids*0.8)
        ax = plt.gca()
        ax.set_facecolor(colors[0])
        plt.title("Step #" + n, loc = "right")
       # plt.axis("off")
        return(fig)
        
    def get_food_ls(self):  # code taken from:       
        data = self.grid._agent_points[self.grid._agent_ids == 1]
        # Generate 2D data.
        x_data, y_data = data[:, 0], data[:, 1]
        xmin, xmax = (0,self.grid.width) #min(x_data), max(x_data)
        ymin, ymax = (0,self.grid.width) #min(y_data), max(y_data)
        
        # Define grid density.
        gd = 200
        # Define bandwidth
        bw = 1.
        
        # Using gaussian_filter
        # Obtain 2D histogram.
        rang = [[xmin, xmax], [ymin, ymax]]
        binsxy = [gd, gd]
        hist1, xedges, yedges = np.histogram2d(x_data, y_data, range=rang, bins=binsxy)
        # Gaussian filtered histogram.
        h_g = gaussian_filter(hist1, bw)
        return h_g
    
    def visualize_food_ls(self):
        h_g = self.get_food_ls()
        # Make plots.
        fig, ax1 = plt.subplots(1, 1)
        # Gaussian filtered 2D histograms.
        ax1.imshow(h_g.transpose(), origin='lower')
        
    def points_on_circumference(self, center_tup, r, n=20):
        return [
            [center_tup[0]+(math.cos(2 * math.pi / n * x) * r),  # x
             center_tup[1] + (math.sin(2 * math.pi / n * x) * r)  # y
            ] for x in np.arange(0, n + 1)]
            
    def visualize(self):
        #f, ax = plt.subplots(1)
        plt.figure()
        fig = plt.scatter(self.grid._agent_points[:, 0], self.grid._agent_points[:, 1], \
                          c = self.grid._agent_ids, s = self.grid._agent_ids*0.8)
       # plt.axis("off")
        return(fig)