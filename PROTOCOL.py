import numpy as np 
import pickle
from sklearn.preprocessing import MinMaxScaler
import networkx as nx 
from scipy.spatial import ConvexHull 
import sys 
import GPy
import uuid
import os 

class HyperRect:  
    def __init__(self,xmin,xmax,rescale=False,num_decimals=None,scaler=None,disallowed_configs=None,data=None):
        self.rescale = rescale
        self.scaler = scaler
        if self.rescale:
            assert self.scaler is not None 
            self.xmin = self.scaler.transform(xmin.reshape(1,-1)).flatten() 
            self.xmax = self.scaler.transform(xmax.reshape(1,-1)).flatten()
        else:
            self.xmin = xmin
            self.xmax = xmax 
        self.rescale = False #only need to rescale first time
        self.center = np.array((self.xmax+self.xmin)/2).reshape(1,-1)
        self.center_y = None 
        self.closest_to_center = None
        self.num_decimals = num_decimals
        self.disallowed_configs = disallowed_configs
        self.id = uuid.uuid4()
        self.data = data
        self.X_in_interval = []
        eps = 0.0001

        if self.data is not None:
            for x in self.data:
                if (xmin <= x).all() and (xmax > x).all():
                    self.X_in_interval.append(x)
            
            if len(self.X_in_interval) > 0:
                closest_to_center_idx = np.argmin(
                    [np.linalg.norm(self.center-x) for x in self.X_in_interval]
                )

                self.closest_to_center = np.array(self.X_in_interval[closest_to_center_idx]).reshape(1,-1)

        if self.num_decimals is not None:
            assert data is None
            assert self.scaler is not None
            center = self.scaler.inverse_transform(self.center).flatten() + eps #addresses floating point shenanigans cause problem with scaler, ie. 2.5 will round to 2 instead of 3. 
            closest_to_center = []
            for i in range(len(num_decimals)):
                rounded = np.around(center[i],self.num_decimals[i])
                closest_to_center.append(rounded)
            c = self.scaler.transform(np.array(closest_to_center).reshape(1,-1))
            if (self.xmin <= c).all() and (self.xmax > c-eps).all():
                self.closest_to_center = self.scaler.transform(np.array(closest_to_center).reshape(1,-1))

    def divide(self):
        longest_idx = np.argmax(
            np.square(self.xmax-self.xmin)
        )

        longest_length = np.linalg.norm(
            self.xmax[longest_idx]-self.xmin[longest_idx]
        )
        #print(longest_idx)
        #print(self.xmin)
        x1, x2 = self.xmin[longest_idx] + np.array([1/3,2/3])*longest_length
        dmin = np.delete(self.xmin,longest_idx)
        dmax = np.delete(self.xmax,longest_idx)

        new_coord_min_1 = np.insert(dmin,longest_idx,x1)
        new_coord_max_1 = np.insert(dmax,longest_idx,x1)
        new_coord_min_2 = np.insert(dmin,longest_idx,x2)
        new_coord_max_2 = np.insert(dmax,longest_idx,x2)
        
        h1 = HyperRect(self.xmin,new_coord_max_1,
            rescale=self.rescale,
            num_decimals=self.num_decimals,
            scaler=self.scaler,
            disallowed_configs=self.disallowed_configs,
            data=self.data
        )

        h2 = HyperRect(new_coord_min_1,new_coord_max_2,
            rescale=self.rescale,
            num_decimals=self.num_decimals,
            scaler=self.scaler,
            disallowed_configs=self.disallowed_configs,
            data=self.data
        )

        h3 = HyperRect(new_coord_min_2,self.xmax,
            rescale=self.rescale,
            num_decimals=self.num_decimals,
            scaler=self.scaler,
            disallowed_configs=self.disallowed_configs,
            data=self.data
        )

        return h1,h2,h3
        
class HierarchicalTree:
    def __init__(self,root,continuous):
        self.root = root 
        self.continuous = continuous 
        self.tree = nx.Graph()
        self.depth = 0 
        self.tree.add_nodes_from([
            (self.root,{"children":[],"level":0,"leaf":True,"GP-based":False,"ID":self.root.id})
        ])

    def grow(self,node):
        new_nodes = node.divide()
        for n in new_nodes:
            if self.continuous:
                self.tree.add_edge(node,n)
                self.tree.add_nodes_from([
                    (n,{"children":[],"level":self.tree.nodes[node]["level"] + 1,
                        "leaf":True,"ID":n.id}
                )])
                self.tree.nodes[node]["children"].append(n)
                if np.array_equal(node.center,n.center):
                    self.tree.nodes[n]["GP-based"] = False
                    n.center_y = node.center_y
                else:
                    self.tree.nodes[n]["GP-based"] = True 
            else:
                if n.closest_to_center is not None:
                    self.tree.add_edge(node,n)
                    self.tree.nodes[n]["ID"] = n.id
                    self.tree.nodes[n]["level"] = self.tree.nodes[node]["level"] + 1
                    self.tree.nodes[n]["leaf"] = True
                    self.tree.nodes[n]["children"] = []
                    self.tree.nodes[node]["children"].append(n)
                    if np.array_equal(node.closest_to_center.flatten(),n.closest_to_center.flatten()): 
                        self.tree.nodes[n]["GP-based"] = False
                        n.center_y = node.center_y 
                    else:
                        self.tree.nodes[n]["GP-based"] = True 

        self.tree.nodes[node]["leaf"] = False
        if self.depth == self.tree.nodes[node]["level"]:
            self.depth += 1

    def find_leaves(self):
        return [node for node in self.tree.nodes if self.tree.nodes[node]["leaf"] == True]  

class OptimizerHandler:
    def __init__(self,max_evals,M,eta,Xi,Xi_max, \
                 continuous,func=None,batch_size=1, \
                 cur_depth=0,cur_iter=0,logging_dir="log/"):
        self.batch_size = batch_size
        self.max_evals = max_evals
        self.cur_depth = cur_depth
        self.M = M 
        self.eta = eta 
        self.Xi = Xi 
        self.Xi_max = Xi_max  
        self.Xi_n = 0
        self.continuous = continuous
        self.func = func 
        self.logging_dir = logging_dir
        self.num_divisions = 0
        self.num_iterations = 0
        self.num_evals = 0
        self.ns = []
        self.rhos = []
        self.f_plusses = []
        self.pre_division = True
        self.xbest = None
        self.ucb_X = []
        self.ucb_y = []
        self.num_requests = []
        self.used_frontier = []
        self.initialized = False
        
        if self.continuous:
            assert (self.func is not None),"A function must be provided with continuous mode"

    def initialize_tree(self,kernel,X=None,X_min=None,X_max=None,y=None,num_decimals=None,normalize_y=False,dim=3):
        assert (self.cur_depth==0),"Tree has nonzero depth, do not reinitialize"
        self.normalize_y = normalize_y
        self.kernel = kernel
        self.dim = dim 

        #initialize for continuous use case, ie. have function to provide
        if self.continuous:

            xmin = np.zeros(dim)
            xmax = np.ones(dim)

            root = HyperRect(xmin,xmax)
            X_init = root.center
            y_init = self.func(X_init).reshape(1,-1)

            print("Root initialized at {}".format(X_init))
            print("Function value at root: {}".format(y_init))
            root.center_y = y_init
            self.T = HierarchicalTree(root,True)
            self.gp = GPy.models.GPRegression(X_init,y_init,self.kernel,normalizer=normalize_y,noise_var=0)
            self.gp.optimize()
            self.f_plus = y_init.flatten()[0]
            self.f_plusses.append(self.f_plus)
            self.X_evaluated = X_init 
            self.y_evaluated = y_init
            self.num_evals += 1
            self.ns.append(self.num_evals)
            self.initialized = True
        
        #initialize based on an input xrange
        elif not self.continuous and X_min is not None and X_max is not None and num_decimals is not None and X is None and y is None:
            #assert (X is not None),"Please provide all possible evaluation points using the \"X\" argument"

            x = np.array([X_min,X_max])
            scaler = MinMaxScaler((0,0.999)).fit(x)
            self.scaler = scaler

            root = HyperRect(X_min,X_max,rescale=True,num_decimals=num_decimals,scaler=scaler)
            X_init = root.closest_to_center.reshape(1,-1)
            T = HierarchicalTree(root,self.continuous)
            self.T = T
            self.X_start = self.scaler.inverse_transform(X_init)
            
            loc = self.logging_dir+"initial_data.csv"
            print("Please evaluate the following data, then re-run initialize_tree with the ground truth label(s)")
            print()
            print(self.X_start)
            print()
            print("The above data has been saved at {}".format(loc))
            print()
            np.savetxt(loc,self.X_start,delimiter=",")

            obj_fname = self.logging_dir+"initialize_optimizer.pkl"
            with open(obj_fname,"wb") as obj:
                pickle.dump(self,obj)
            print("The save state for the optimizer has been saved to {}".format(obj_fname))
            print()

        #initialize the tree when providing a data set as input
        elif X is not None and y is None and not self.continuous:
            xmin = np.zeros(X.shape[1])
            xmax = np.ones(X.shape[1])

            scaler = MinMaxScaler((0,0.999)).fit(X)
            X_scaled = scaler.transform(X)
            self.initial_data = X_scaled
            self.scaler = scaler

            root = HyperRect(xmin,xmax,data=X_scaled)
            X_init = root.closest_to_center.reshape(1,-1)
            T = HierarchicalTree(root,self.continuous)
            self.T = T
            self.X_start = self.scaler.inverse_transform(X_init)
            self.X_init_raw = self.X_start
            loc = self.logging_dir+"initial_data.csv"
            print("Please evaluate the following data, then re-run initialize_tree with the ground truth label(s)")
            print()
            print(self.X_start)
            print()
            print()
            print("The above data has been saved at {}".format(loc))
            print()
            np.savetxt(loc,self.X_start,delimiter=",")

            obj_fname = self.logging_dir+"initialize_optimizer.pkl"
            with open(obj_fname,"wb") as obj:
                pickle.dump(self,obj)
            print("The save state for the optimizer has been saved to {}".format(obj_fname))
            print()

        #provide the label for the root node.  To use this, will have to have previously called initialize_tree with one of the above settings.
        elif X is not None and y is not None and not self.continuous:
            assert (len(X) == len(y)),"Please provide the data points that were evaluated using the \"X\" argument, as well as their observed labels using the \"y\" argument"
            y = y.reshape(-1,1)
            X = self.scaler.transform(X)
            self.gp = GPy.models.GPRegression(X,y,self.kernel,normalizer=normalize_y,noise_var=0)
            list(self.T.tree.nodes)[0].center_y = y
            self.gp.optimize()
            self.f_plus = np.max(y) 
            self.X_evaluated = X
            self.y_evaluated = y 
            self.f_plusses.append(self.f_plus)
            self.num_evals += len(X)
            self.ns.append(self.num_evals)
            self.initialized = True 

        else:
            print("Invalid input, try again")
            sys.exit()


    def save_state(self):
        #assert (hasattr(self,"centers_to_eval")),"Please run an optimization algo before saving."

        loc = self.logging_dir
        obj_fname = loc+"{0}_{1}_optimizer.pkl".format(str(self.num_iterations),str(self.num_evals))
        eval_fname = loc+"{0}_{1}_evaluation_points.csv".format(str(self.num_iterations),str(self.num_evals))
        history_fname = loc+"history.csv"

        if self.num_evals == self.max_evals:
            self.ns.append(self.num_evals)

        self.centers_to_eval = []

        self.num_requests.append(len(self.to_eval))

        assert len(self.X_evaluated) <= self.max_evals
        assert len(self.X_evaluated) == self.num_evals
        
        for c in self.to_eval:
            if isinstance(c,HyperRect):
                if self.continuous:
                    self.centers_to_eval.append(c.center.flatten())
                else:
                    d = self.scaler.inverse_transform(c.closest_to_center).flatten()
                    self.centers_to_eval.append(d)
            else:
                if not self.continuous:
                    d = self.scaler.inverse_transform(c.reshape(1,-1)).flatten()
                    self.centers_to_eval.append(d)
                else:
                    self.centers_to_eval.append(c)
 
        to_eval_set = set([tuple(x) for x in self.centers_to_eval])

        assert len(to_eval_set) == len(self.centers_to_eval)

        #GPy kernels not picklable, so need workaround
        self.params = self.gp.param_array
        self.gp = None 
        self.kernel = None 

        assert len(self.centers_to_eval) <= self.batch_size
        with open(obj_fname,"wb") as obj:
            pickle.dump(self,obj)
        
        print()
        print("Total function evaluations made thus far: {}".format(len(self.X_evaluated)))
        print()
        print("The save state for the optimizer has been saved to {}".format(obj_fname))
        print()
        
        np.savetxt(eval_fname,self.centers_to_eval,delimiter=",")
        
        print("Please evaluate the following data points:")
        print()
        print(np.array(self.centers_to_eval))
        print()
        print("The data points to evaluate have been saved to {}".format(eval_fname))
        print("Please obtain ground truth labels for these data points.")
        print("Afterwards, load the optimizer save state and run update with the ground truth labels.")
        print("Then, run the next iteration if desired.")
        print()

        print("A log of the history up to this point has been written to {}".format(history_fname))
        print("The final column corresponds to the label, and the preceeding columns correspond to each parameter")

        history = np.hstack((self.scaler.inverse_transform(self.X_evaluated),self.y_evaluated))
        np.savetxt(history_fname,history,delimiter=",")
        
        return self.T, self.gp

    def update(self,X_update,y_update):
        #assert (len(updates)==len(self.centers_to_eval)),"Incorrect number of ground truth labels provided. \
        #    Expected {0} but received {1}".format(len(self.centers_to_eval),len(updates))
        
        var, ls, noise = self.params 
        self.kernel = GPy.kern.src.sde_matern.sde_Matern52(input_dim=self.dim, variance=var, lengthscale=ls)
        self.y_evaluated = y_update #np.vstack((self.y_evaluated,y_update.reshape(-1,1)))

        if self.continuous:
            self.X_evaluated = self.X_evaluated #np.vstack((self.X_evaluated,X_update))
            for node in self.T.tree.nodes:
                for i,x in enumerate(X_update):
                    if np.array_equal(node.center.flatten(),x):
                        node.center_y = y_update[i]
                        self.T.tree.nodes[node]["GP-based"] = False
        else:
            #account for case where only one value was analyzed
            try:
                X_update = self.scaler.transform(X_update)
            except:
                X_update = self.scaler.transform(X_update.reshape(1,-1))
            self.X_evaluated = X_update #np.vstack((self.X_evaluated,X_update))
            for node in self.T.tree.nodes:
                for i,x in enumerate(X_update):
                    if np.array_equal(node.closest_to_center.flatten(),x):
                        node.center_y = y_update[i]
                        self.T.tree.nodes[node]["GP-based"] = False

        self.gp = GPy.models.GPRegression(self.X_evaluated,self.y_evaluated,self.kernel,normalizer=self.normalize_y,noise_var=noise)
        new_f_plus = np.max(self.y_evaluated)
        self.num_evals = len(self.X_evaluated) 

        if new_f_plus > self.f_plus:
            self.f_plus = new_f_plus
            
        if self.num_evals == self.max_evals:
            self.ns.append(self.num_evals)
            self.f_plusses.append(self.f_plus)
  
        self.to_eval = None 

    def PROTOCOL(self):
        while True:

            #steps I and II
            if self.pre_division:
                self.prev_fplus = self.f_plus
                leaves = self.T.find_leaves()
                candidates = {} #candidate intervals to divide
                vmax = -np.inf
                first_half_queue = []
                for h in range(self.T.depth+1):
                    i_leaves = [leaf for leaf in leaves if self.T.tree.nodes[leaf]["level"]==h]
                    if i_leaves:
                       while True:
                            center_vals = [leaf.center_y for leaf in i_leaves]
                            i_star = np.argmax(center_vals)
                            max_interval = i_leaves[i_star]
                            max_center = center_vals[i_star]
                            if np.less(max_center,vmax):
                                break
                            elif not self.T.tree.nodes[max_interval]["GP-based"]:
                                candidates[h] = max_interval
                                vmax = max_center
                                break
                            else: # difference- add to queue, use frontier to fill
                                first_half_queue.append(max_interval)
                                t,g = self.obtain_eval_points(h,first_half_queue,leaves,[max_interval.center_y])
                                return t,g 


                #step III
                #identical to imgpo
                for h in list(candidates):
                    if candidates[h]:
                        xi_limit = np.min([self.Xi,self.Xi_max])
                        cur_xi = 1
                        while True:
                            if cur_xi > xi_limit:
                                cur_xi = 0
                                break
                            higher_levels = {hh:candidates[hh] for hh in candidates if hh > h and candidates[hh]}
                            if h + cur_xi in higher_levels:
                                break
                            else:
                                cur_xi += 1
                        if cur_xi > 0:
                            xi_leaf = candidates[h+cur_xi]
                            g = xi_leaf.center_y
                            z, self.M = self._find_zs(candidates[h],cur_xi,g)
                            if not z:
                                if cur_xi > self.Xi_n:
                                    self.Xi_n = cur_xi
                                candidates[h] = None 
                self.candidates = candidates

            #steps IV-VI
            leaves = self.T.find_leaves()
            self.pre_division = False
            vmax = -np.inf 
            for h in self.candidates:
                candidate = self.candidates[h]
                if candidate:
                    if np.greater_equal(candidate.center_y,vmax):
                        self.num_divisions += 1
                        self.T.grow(candidate)
                        if len(self.T.tree.nodes[candidate]["children"]) == 1:
                            continue
                        assert len(self.T.tree.nodes[candidate]["children"]) != 1
                        second_half_queue = []
                        ucbs = []
                        for child in self.T.tree.nodes[candidate]["children"]:
                            if not self.continuous:
                                for i,arr in enumerate(self.X_evaluated):
                                    if np.array_equal(child.closest_to_center.flatten(),arr):
                                        if not child.center_y:
                                            child.center_y = self.y_evaluated[i]
                                        self.T.tree.nodes[child]["GP-based"] = False
                            if self.T.tree.nodes[child]["GP-based"]: #difference- check both children, then fill queue using frontier 
                                if self.continuous:
                                    ucb, M_cur = self._compute_UCB(child.center,self.M)
                                else:
                                    ucb, M_cur = self._compute_UCB(child.closest_to_center,self.M)
                                if np.greater_equal(ucb,self.f_plus): 
                                    child.center_y = ucb
                                    second_half_queue.append(child)
                                    ucbs.append(ucb)
                                else:
                                    child.center_y = ucb
                                    self.M = M_cur
                        if second_half_queue:
                            leaves = self.T.find_leaves()
                            self.candidates = {hh:self.candidates[hh] for hh in self.candidates if hh > h}
                            t,g = self.obtain_eval_points(h,second_half_queue,leaves,ucbs)
                            return t,g 
            
            #update parameters
            if self.f_plus != self.prev_fplus:
                self.Xi += 4
            else:
                self.Xi = np.max([self.Xi-0.5,1])
            self.pre_division = True
            self.num_iterations += 1
            if self.num_divisions < 4:
                self.gp.optimize()
            self.ns.append(self.num_evals)
            self.f_plusses.append(self.f_plus)
            if self.num_evals >= self.max_evals:
                return self.T, self.gp 

    def obtain_eval_points(self,h,init_queue,leaves,ucbs):
        #init queue always has length 1 or 2
        init_keep = len(init_queue)
        if init_keep > 2:
            print("why")
            sys.exit()
        init_queue_ids = [q.id for q in init_queue]

        #when the queue matches the batch size or remaining eval size, just request the queue
        if len(init_queue) == np.min([self.batch_size,self.max_evals-len(self.X_evaluated)]):#self.batch_size or len(init_queue) == self.max_evals-len(self.X_evaluated):
            self.to_eval = init_queue
            self.used_frontier.append(0)
            t,g = self.save_state()
            return t,g

        #if init_queue is larger than allowable, choose one with larger UCB. Set center value of remainder to be its UCB
        if init_keep > self.max_evals-len(self.X_evaluated) or init_keep > self.batch_size:
            larger_ucb_idx = np.argmax(ucbs)
            smaller_ucb_idx = larger_ucb_idx - 1 
            self.to_eval = [np.array(init_queue)[larger_ucb_idx]]
            init_queue[smaller_ucb_idx].center_y = ucbs[smaller_ucb_idx]
            self.M += 1 
            self.used_frontier.append(0)
            t,g = self.save_state()
            return t,g 

        #when the queue does not equal the batch size, determine how much needed to meet batch size
        #account for max amount of total evaluations
        allowable = np.min([self.batch_size-len(init_queue),self.max_evals-len(self.X_evaluated)-len(init_queue)])
        assert allowable > 0

        #find leaves at greater depths than current interval for building frontier
        frontier_queue = init_queue[:]
        for leaf in leaves:
            if self.T.tree.nodes[leaf]["GP-based"] and self.T.tree.nodes[leaf]["ID"] not in init_queue_ids:
                frontier_queue.append(leaf)
        center_ys = [i.center_y for i in frontier_queue]

        #calculate the convex hull if possible- there are three or more in the frontier queue
        if len(center_ys) > 2: 
            self.used_frontier.append(1)
            frontier_pairs = []
            levels_in_frontier = set()
            for i,center in enumerate(center_ys):
                hh = self.T.tree.nodes[frontier_queue[i]]["level"]
                frontier_pairs.append([hh,center])
                levels_in_frontier.add(hh)

            #must be two dimensional to calculate convex hull
            if len(levels_in_frontier) > 1:
                #problem arises if UCB happens to be same for two nodes on same level
                con_hull = ConvexHull(frontier_pairs,qhull_options="QJ")
                con_hull_idxs = [ch for ch in con_hull.vertices if ch > init_keep-1]

                #among convex hull, choose highest ucb at each level
                ch_best_ucb = {}
                for i_ch in con_hull_idxs:
                    ch_pair_i = frontier_pairs[i_ch]
                    ch_interval = frontier_queue[i_ch]
                    ch_level = ch_pair_i[0]
                    ch_value = ch_pair_i[1]
                    if ch_level in ch_best_ucb:
                        if ch_value > ch_best_ucb[ch_level][1]:
                            ch_best_ucb[ch_level] = (i_ch,ch_value)
                    else:
                        ch_best_ucb[ch_level] = (i_ch,ch_value)

                ch_idxs = [ch[0] for ch in list(ch_best_ucb.values())]
                on_frontier_values = np.array(center_ys)[ch_idxs].flatten()

            else:
                on_frontier_values = np.array(center_ys).flatten()[init_keep:]

        #if unable to use convex hull, choose best from single dimension
        else:
            on_frontier_values = np.array(center_ys).flatten()[init_keep:]


        #proceed choosing from frontier
        max_idx = np.argsort(on_frontier_values)[::-1][:allowable]
        on_frontier_intervals = np.array(frontier_queue[init_keep:])[max_idx].flatten()
        for q in init_queue:
            assert q not in on_frontier_intervals
        self.to_eval = init_queue + list(on_frontier_intervals) 
        t,g = self.save_state()
        return t,g

    def _update_data(self,X,y):
        self.X_evaluated = np.vstack((self.X_evaluated,X))
        self.y_evaluated = np.vstack((self.y_evaluated,y))
        self.gp.set_XY(self.X_evaluated,self.y_evaluated) 

    def _find_zs(self,interval,cur_xi,g):
        children = [interval]
        M = self.M 
        while cur_xi > 0:
            new_children = []
            for child in children:
                new_nodes = child.divide()
                for node in new_nodes:
                    new_children.append(node)
                    if self.continuous:
                        ucb, M_update = self._compute_UCB(node.center,M)
                    else:
                        if node.closest_to_center is not None:
                            ucb, M_update = self._compute_UCB(node.closest_to_center,M)
                        else: 
                            continue
                    if np.greater(ucb,g):
                        return True, self.M
                    M = M_update
            children = new_children
            cur_xi -= 1 
        return False, M 

    def _compute_UCB(self,X,M):
        sigmaM = np.sqrt(2*np.log((np.pi**2)*(M**2)/(12*self.eta)))+0.2
        mean,var = self.gp.predict(X)
        ucb = mean + sigmaM*np.sqrt(var)
        M += 1
        return ucb, M 

def sin1(x):
    return (np.sin(13*x)*np.sin(27*x)+1)/2

def initialize(logging_dir,continuous,batch_size,max_evals, \
    xmin=None,xmax=None,num_decimals=None,data=None,M=1,eta=0.05,Xi=1,Xi_max=4):

    assert((xmin is not None and xmax is not None) or (data is not None)),"Please either specify a min and max X range or provide a data set as input"

    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    experiment = OptimizerHandler(
        max_evals=max_evals,
        M=M,
        eta=eta,
        Xi=Xi,
        Xi_max=Xi_max,
        continuous=continuous,
        batch_size=batch_size,
        logging_dir=logging_dir
    )

    if data is not None:
        dim = data.shape[1]
        kernel = GPy.kern.src.sde_matern.sde_Matern52(input_dim=dim, variance=1., lengthscale=0.25)
        experiment.initialize_tree(kernel,X=data,dim=dim)

    else:
        if num_decimals is None and not continuous:
            print("No level of precision specified, rounding to 3 decimal places")
            num_decimals = [3]*len(xmin)
        scaler = MinMaxScaler((0,0.999)).fit(np.array([xmin,xmax]))
        dim = len(xmin)

        kernel = GPy.kern.src.sde_matern.sde_Matern52(input_dim=dim, variance=1., lengthscale=0.25)
        experiment.initialize_tree(kernel,X_min=xmin,X_max=xmax,num_decimals=num_decimals,dim=dim)

    eval_data = np.loadtxt(logging_dir+"/initial_data.csv",delimiter=",")
    if eval_data.shape == (dim,):
        eval_data = np.array([eval_data])

    return eval_data 

def update_PROTOCOL(X,y,optimizer_file,batch_size=None):

    with open(optimizer_file,"rb") as experiment:
        experiment = pickle.load(experiment)

    if batch_size is not None:
        experiment.batch_size = batch_size

    if experiment.initialized:
        #assert y_updates is not None 
        #y_updates = np.array(y_updates).reshape(-1,1)
        experiment.update(X,y) 
        experiment.PROTOCOL()
    else:
        kernel = GPy.kern.src.sde_matern.sde_Matern52(input_dim=experiment.dim, variance=1., lengthscale=0.25)
       # assert y_updates is not None 
        experiment.initialize_tree(kernel,X=X,y=y,dim=experiment.dim)
        experiment.PROTOCOL()
    
    #obtain the data points just requested by the algorithm
    eval_fname = experiment.logging_dir+"{0}_{1}_evaluation_points.csv".format(str(experiment.num_iterations),str(experiment.num_evals))
    eval_data = np.loadtxt(eval_fname,delimiter=",")
    if eval_data.shape == (experiment.dim,):
        eval_data = np.array([eval_data])

    return eval_data 


if __name__ == '__main__':
    
    
    ### EXAMPLE 1- initialize HPLC tree
    ### uses xmax and xmin initialization
    
    #initial input space
    xmin = np.array([1,1,0.2,5,25,260])
    xmax = np.array([4,5,1.8,45,45,285])
    num_decimals = [0,0,1,1,1,0]

    #intial input values
    logging_dir = "test_log/"
    continuous = False 
    batch_size = 3
    max_evals = 25

    #initialize the tree- will request evaluation of root node
    d1 = initialize(logging_dir,continuous,batch_size,max_evals,
        xmin=xmin,
        xmax=xmax,
        num_decimals=num_decimals
    )
    
    #made up y update, just meant to be illustrative
    y1 = np.array([4.90])
    
    #update PROTOCOL by passing the X array, y array, and path to the optimizer.  You may specify a batch size.  
    #The X and y arrays should have all data points previously evaluated
    #The optimizer was made by the initialize function.
    d2 = update_PROTOCOL(d1,y1,"test_log/initialize_optimizer.pkl",batch_size=batch_size)

    #made up second y update, just meant to be illustrative
    y_requested = np.array([3,5,0]).reshape(-1,1)
 
    #This time (and all times after the initial update), we refer to a history file made by PROTOCOL to obtain previously evaluated points.
    #make sure the requested data points and their y values are index-matched
    data = np.loadtxt("test_log/history.csv",delimiter=",").reshape(1,-1)
    x2 = np.vstack((data[:,:-1],d2))
    y2 = np.vstack((data[:,-1],y_requested))

    #Update PROTOCOL as before.  Make sure to refer to correct X and y arrays, as well as the most recent optimizer.
    d3 = update_PROTOCOL(x2,y2,"test_log/1_1_optimizer.pkl",batch_size=batch_size)
    


















