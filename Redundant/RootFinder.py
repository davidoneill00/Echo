import numpy as np
from scipy.optimize import brentq as sob
from scipy.optimize import minimize, Bounds
import warnings
if hasattr(np, "ComplexWarning"):
    warnings.simplefilter("ignore", np.ComplexWarning)

"""
A root finding algorithm required for Method.py. The class takes a function as an input, followed by a number of 
solver specific parameters (for accuracy and speed). The 'Roots' function returns the numerical values of the roots.
"""

class GodTierRootFinder():
	def __init__(self, func, BrentTol, BrentIter, Tol, dx, Offset, MaxIter, rhobeg, Max_Number_of_Roots, *args):
		self.func                = func
		self.BrentTol            = BrentTol
		self.BrentIter           = BrentIter
		self.Tol                 = Tol
		self.dx                  = dx
		self.Offset              = Offset
		self.MaxIter             = MaxIter
		self.args                = args
		self.rhobeg              = rhobeg
		self.Max_Number_of_Roots = Max_Number_of_Roots

	def RootFunction(self, var):
		if isinstance(var, np.ndarray):
			return self.func(var[0], *self.args)
		return self.func(var, *self.args)

	def MinusFunction(self, var):
		if isinstance(var, np.ndarray):
			return -self.func(var[0], *self.args)
		return -self.func(var,*self.args)

	def Derivative(self, var):
		xl = var - 0.5 * self.dx
		xr = var + 0.5 * self.dx

		fl = self.func(xl,*self.args)
		fr = self.func(xr,*self.args)

		df = fr - fl

		return df / self.dx

	def StopCondition(self, a, b, Turn_a_pos, Turn_b_pos):
		if Turn_a_pos > Turn_b_pos:
			self.Stop = True
		else:
			if np.abs(Turn_a_pos - Turn_b_pos)< 2 * self.Offset:#self.StopCond:
				self.Stop = True
			else:
				self.Stop = False


	def BrentIteration(self,endpts,updated_endpts,endvals,updated_endvals):
		a_old,b_old     = endpts
		a_new,b_new     = updated_endpts
		f_a, f_b        = endvals
		f_a_new,f_b_new = updated_endvals

		roots = []
	    
		if f_a * f_a_new < 0:
			brent_root_a = sob(self.func, a_old, a_new, xtol=self.BrentTol, rtol=self.BrentTol, maxiter=self.BrentIter, args=self.args)
			roots.append(brent_root_a)

		if f_b * f_b_new < 0:
			brent_root_b = sob(self.func, b_new, b_old, xtol=self.BrentTol, rtol=self.BrentTol, maxiter=self.BrentIter, args=self.args)
			roots.append(brent_root_b)
			
		return roots

	def UniqueRoots(self,a):
		lnrts     = len(a)
		if lnrts>1:
			DiffRoots = np.diff(a)
			Unique    = []
			for rt in range(lnrts-1):
				if np.abs(DiffRoots[rt]) > 1e-5:
					Unique.append(a[rt])
			Unique.append(a[lnrts-1])
			return Unique
		else:
			return a


	def InitialStep(self, endpts, endvals):
		a, b     = endpts
		f_a, f_b = endvals
		df_a     = self.Derivative(a)
		df_b     = self.Derivative(b)

		def constraint_a(x):
			return x - a

		def constraint_b(x):
			return b - x

		constraints = [
			{'type': 'ineq', 'fun': constraint_a},
			{'type': 'ineq', 'fun': constraint_b}
		]
		roots = []

		if df_a <= 0:
			Turn_a      = minimize(self.RootFunction ,a, method = 'COBYLA', constraints=constraints, tol = self.Tol, options = {'maxiter':self.MaxIter, 'rhobeg':self.rhobeg}) 
			Turn_Type_a = 'Minimum'

		elif df_a > 0:
			Turn_a      = minimize(self.MinusFunction,a, method = 'COBYLA', constraints=constraints, tol = self.Tol, options = {'maxiter':self.MaxIter, 'rhobeg':self.rhobeg}) 
			Turn_a.fun  = -Turn_a.fun
			Turn_Type_a = 'Maximum'

		if df_b >= 0: 
			Turn_b      = minimize(self.RootFunction ,b, method = 'COBYLA', constraints=constraints, tol = self.Tol, options = {'maxiter':self.MaxIter, 'rhobeg':self.rhobeg})  
			Turn_Type_b = 'Minimum'

		elif df_b < 0: 
			Turn_b      = minimize(self.MinusFunction,b, method = 'COBYLA', constraints=constraints, tol = self.Tol, options = {'maxiter':self.MaxIter, 'rhobeg':self.rhobeg})  
			Turn_b.fun  = -Turn_b.fun
			Turn_Type_b = 'Maximum'

		roots = self.BrentIteration([a,b],[Turn_a.x,Turn_b.x],[f_a,f_b],[Turn_a.fun, Turn_b.fun])
		Stop  = self.StopCondition(a, b, Turn_a.x, Turn_b.x)

		#print('First step is',[a,b], '--->',[Turn_a.x[0],Turn_b.x[0]])
		RootStep = {
		"InitialBounds":[a,b],
		"UpdatedBounds":[Turn_a.x[0],Turn_b.x[0]],
		"ValuesOnBounds":[Turn_a.fun,Turn_b.fun],
		"Roots":roots,
		"TurnType_a":Turn_Type_a,
		"TurnType_b":Turn_Type_b,
		"Finished":self.Stop
		}

		return RootStep


	def Offset_Compression(self, endpts, endvals, Turn_Types):
		a, b                   = endpts
		f_a, f_b               = endvals
		TurnType_a, TurnType_b = Turn_Types

		a_in   = a + self.Offset
		b_in   = b - self.Offset
		if a_in > b_in:
			return {
			"InitialBounds":[a,b],
			"UpdatedBounds":[a,b],
			"ValuesOnBounds":[f_a,f_b],
			"Roots":[],
			"TurnType_a":np.nan,
			"TurnType_b":np.nan,
			"Finished":True
			}

		def constraint_a(x):
			return x - a_in

		def constraint_b(x):
			return b_in - x

		constraints = [
			{'type': 'ineq', 'fun': constraint_a},
			{'type': 'ineq', 'fun': constraint_b}
		]

		if TurnType_a == 'Minimum':
			Turn_a     = minimize(self.MinusFunction,a_in, method = 'COBYLA', constraints=constraints, tol = self.Tol, options = {'maxiter':self.MaxIter, 'rhobeg':self.rhobeg}) 
			Turn_a.fun = -Turn_a.fun

		elif TurnType_a == 'Maximum': 
			Turn_a     = minimize(self.RootFunction ,a_in, method = 'COBYLA', constraints=constraints, tol = self.Tol, options = {'maxiter':self.MaxIter, 'rhobeg':self.rhobeg})  

		if TurnType_b == 'Minimum':
			Turn_b     = minimize(self.MinusFunction,b_in, method = 'COBYLA', constraints=constraints, tol = self.Tol, options = {'maxiter':self.MaxIter, 'rhobeg':self.rhobeg}) 
			Turn_b.fun = -Turn_b.fun

		elif TurnType_b == 'Maximum':
			Turn_b     = minimize(self.RootFunction ,b_in, method = 'COBYLA', constraints=constraints, tol = self.Tol, options = {'maxiter':self.MaxIter, 'rhobeg':self.rhobeg}) 

		
		roots = self.BrentIteration([a,b],[Turn_a.x,Turn_b.x],[f_a,f_b],[Turn_a.fun, Turn_b.fun]) ######### a_in, b_in here?
		Stop  = self.StopCondition(a, b, Turn_a.x, Turn_b.x)


		RootStep = {
		"InitialBounds":[a,b],
		"UpdatedBounds":[Turn_a.x[0],Turn_b.x[0]],
		"ValuesOnBounds":[Turn_a.fun,Turn_b.fun],
		"Roots":roots,
		"TurnType_a":TurnType_a,
		"TurnType_b":TurnType_b,
		"Finished":self.Stop
		}

		return RootStep

	def IterateCompressions(self, endpts, endvals):
		a, b     = endpts
		f_a, f_b = endvals

		Constraining_Bounds = self.InitialStep(endpts, [f_a, f_b])
		roots               = Constraining_Bounds["Roots"]
		bracket_chain       = [endpts,Constraining_Bounds["UpdatedBounds"]]

		iterations = 1     
		while Constraining_Bounds["Finished"] == False:
		#for _____ in range(0,10):
			Iteration = self.Offset_Compression(
				Constraining_Bounds["UpdatedBounds"],
				Constraining_Bounds["ValuesOnBounds"],
				[Constraining_Bounds["TurnType_a"], Constraining_Bounds["TurnType_b"]],
				)

			if Iteration["Roots"]!= []:
				[roots.append(i) for i in Iteration["Roots"]]
				if len(roots) >= self.Max_Number_of_Roots:
					Constraining_Bounds["Finished"] = True

			bracket_chain.append(Iteration["UpdatedBounds"])
			Constraining_Bounds = Iteration
			iterations         +=iterations
	
		#print('[initial],[first step], [iterations]')
		#print(bracket_chain)
		if Constraining_Bounds['ValuesOnBounds'][0] * Constraining_Bounds['ValuesOnBounds'][1] <0:
			roots.append(sob(self.func, Constraining_Bounds['UpdatedBounds'][0], Constraining_Bounds['UpdatedBounds'][1], xtol=self.BrentTol, rtol=self.BrentTol, maxiter=self.BrentIter, args=self.args))
		
		return {'Roots':roots,
		'chain':bracket_chain,
		'iter':iterations
		}


	def Roots(self,a,b,f_a,f_b):
		IC    =  self.IterateCompressions([a,b], [f_a, f_b])
		Roots = list(self.UniqueRoots(np.sort(IC['Roots'])))
		return Roots + [np.nan for _ in range(self.Max_Number_of_Roots - len(Roots))]





class BisectionMethod():
	"""
	Redundant funtion
	"""
	def __init__(self, func, BrentTol, BrentIter, dx, a, b, *args):
		self.func      = func
		self.BrentTol  = BrentTol
		self.BrentIter = BrentIter
		#self.epsilon   = epsilon
		self.dx        = dx
		self.a         = a
		self.b         = b
		self.args      = args
		

	def RootFunction(self, var):
		return self.func(var, *self.args)

	def rootsearch(self):                                                                  
		x1 = self.a
		f1 = self.RootFunction(x1)
		x2 = x1 + self.dx
		f2 = self.RootFunction(x2)

		while f1 * f2 > 0.0:  
		    if x1 >= self.b:
		        return np.nan, np.nan  
		    x1 = x2
		    f1 = f2
		    x2 = x1 + self.dx
		    f2 = self.RootFunction(x2)

		return x1, x2 
    

	def bisect(self, x1, x2, switch=0):
	    f1 = self.RootFunction(x1)
	    if f1 == 0.0:    
	        return x1
	    f2 = self.RootFunction(x2)
	    if f2 == 0.0:
	        return x2

	    if f1 * f2 > 0.0:
	        return np.nan  # No root in this interval

	    try:
	        brent_root = sob(self.func, x1, x2, xtol=self.BrentTol, rtol=self.BrentTol, maxiter=self.BrentIter, args=self.args)
	        return brent_root
	    except ValueError:
	        return np.nan  # Brentq failed


	def Roots(self):
	    arr = []
	    while True:
	        x1, x2 = self.rootsearch()
	        if np.isfinite(x1):
	            break  # Stop if no more roots found

	        self.a = x2  # Move search interval forward
	        root = self.bisect(x1, x2, 1)
	        if not np.isfinite(rot):
	            arr.append(root)

	    return arr  # Return the list of roots





