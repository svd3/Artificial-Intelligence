import collections, util, copy
from operator import itemgetter

############################################################
# Problem 3.1a

def create_nqueens_csp(n = 8):
    """
    Return an N-Queen problem on the board of size |n| * |n|.
    You should call csp.add_variable() and csp.add_binary_potential().

    @param n: number of queens, or the size of one dimension of the board.

    @return csp: A CSP problem with correctly configured potentials
        such that it can be solved by a weighted CSP solver.
    """
    csp = util.CSP()
    # BEGIN_YOUR_CODE (around 7 lines of code expected)
    varNames = ['X'+str(i) for i in range(n)]
    for var in varNames:
        csp.add_variable(var, range(n))
    for i in range(n):
        for j in range(i+1, n):
            if i != j:
                csp.add_binary_potential(varNames[i], varNames[j], lambda x, y : x != y)
                csp.add_binary_potential(varNames[i], varNames[j], lambda x, y : x-i != y-j)
                csp.add_binary_potential(varNames[i], varNames[j], lambda x, y : x+i != y+j)
    #raise Exception("Not implemented yet")
    # END_YOUR_CODE
    return csp


# create_nqueens_csp()
############################################################
# Problem 3.1b

# A backtracking algorithm that solves weighted CSP.
# Usage:
#   search = BacktrackingSearch()
#   search.solve(csp)
class BacktrackingSearch():

    def reset_results(self):
        """
        This function resets the statistics of the different aspects of the
        CSP sovler. We will be using the values here for grading, so please
        do not make any modification to these variables.
        """
        # Keep track of the best assignment and weight found.
        self.optimalAssignment = {}
        self.optimalWeight = 0

        # Keep track of the number of optimal assignments and assignments. These
        # two values should be identical when the CSP is unweighted or only has binary
        # weights.
        self.numOptimalAssignments = 0
        self.numAssignments = 0

        # Keep track of the number of times backtrack() gets called.
        self.numOperations = 0

        # Keey track of the number of operations to get to the very first successful
        # assignment (doesn't have to be optimal).
        self.firstAssignmentNumOperations = 0

        # List of all solutions found.
        self.allAssignments = []
        self.allWeights = []

    def print_stats(self):
        """
        Prints a message summarizing the outcome of the solver.
        """
        if self.optimalAssignment:
            print "Found %d optimal assignments with weight %f in %d operations" % \
                (self.numOptimalAssignments, self.optimalWeight, self.numOperations)
            print "First assignment took %d operations" % self.firstAssignmentNumOperations
        else:
            print "No solution was found."

    def get_delta_weight(self, assignment, var, val):
        """
        Given a CSP, a partial assignment, and a proposed new value for a variable,
        return the change of weights after assigning the variable with the proposed
        value.

        @param assignment: A list of current assignment. len(assignment) should
            equal to self.csp.numVars. Unassigned variables have None values, while an
            assigned variable has the index of the value with respect to its
            domain. e.g. if the domain of the first variable is [5,6], and 6
            was assigned to it, then assignment[0] == 1.
        @param var: Index of an unassigned variable.
        @param val: Index of the proposed value with resepct to |var|'s domain.

        @return w: Change in weights as a result of the proposed assignment. This
            will be used as a multiplier on the current weight.
        """
        assert assignment[var] is None
        w = 1.0
        if self.csp.unaryPotentials[var]:
            w *= self.csp.unaryPotentials[var][val]
            if w == 0: return w
        for var2, potential in self.csp.binaryPotentials[var].iteritems():
            if assignment[var2] == None: continue  # Not assigned yet
            w *= potential[val][assignment[var2]]
            if w == 0: return w
        return w

    def solve(self, csp, mcv = False, lcv = False, mac = False):
        """
        Solves the given weighted CSP using heuristics as specified in the
        parameter. Note that unlike a typical unweighted CSP where the search
        terminates when one solution is found, we want this function to find
        all possible assignments. The results are stored in the variables
        described in reset_result().

        @param csp: A weighted CSP.
        @param mcv: When enabled, Monst Constrained Variable heuristics is used.
        @param lcv: When enabled, Least Constraining Value heuristics is used.
        @param mac: When enabled, AC-3 will be used after each assignment of an
            variable is made.
        """
        # CSP to be solved.
        self.csp = csp

        # Set the search heuristics requested asked.
        self.mcv = mcv
        self.lcv = lcv
        self.mac = mac

        # Reset solutions from previous search.
        self.reset_results()

        # The list of domains of every variable in the CSP. Note that we only
        # use the indices of the values. That is, if the domain of a variable
        # A is [2,3,5], then here, it will be stored as [0,1,2]. Original domain
        # name/value can be obtained from self.csp.valNames[A]
        self.domains = [list(range(len(domain))) for domain in self.csp.valNames]
        # Perform backtracking search.
        self.backtrack([None] * self.csp.numVars, 0, 1)

        # Print summary of solutions.
        self.print_stats()

    def backtrack(self, assignment, numAssigned, weight):
        """
        Perform the back-tracking algorithms to find all possible solutions to
        the CSP.

        @param assignment: A list of current assignment. len(assignment) should
            equal to self.csp.numVars. Unassigned variables have None values, while an
            assigned variable has the index of the value with respect to its
            domain. e.g. if the domain of the first variable is [5,6], and 6
            was assigned to it, then assignment[0] == 1.
        @param numAssigned: Number of currently assigned variables
        @param weight: The weight of the current partial assignment.
        """

        self.numOperations += 1
        assert weight > 0
        if numAssigned == self.csp.numVars:
            # A satisfiable solution have been found. Update the statistics.
            self.numAssignments += 1
            newAssignment = {}
            for var in range(self.csp.numVars):
                newAssignment[self.csp.varNames[var]] = self.csp.valNames[var][assignment[var]]
            self.allAssignments.append(newAssignment)
            self.allWeights.append(weight)

            if len(self.optimalAssignment) == 0 or weight >= self.optimalWeight:
                if weight == self.optimalWeight:
                    self.numOptimalAssignments += 1
                else:
                    self.numOptimalAssignments = 1
                self.optimalWeight = weight

                self.optimalAssignment = newAssignment
                if self.firstAssignmentNumOperations == 0:
                    self.firstAssignmentNumOperations = self.numOperations
            return

        # Select the index of the next variable to be assigned.
        var = self.get_unassigned_variable(assignment)

        # Obtain the order of which a variable's values will be tried. Note that
        # this stores the indices of the values with respect to |var|'s domain.
        ordered_values = self.get_ordered_values(assignment, var)
        # Continue the backtracking recursion using |var| and |ordered_values|.
        if not self.mac:
            # When arc consistency check is not enabled.
            for val in ordered_values:
                deltaWeight = self.get_delta_weight(assignment, var, val)
                if deltaWeight > 0:
                    assignment[var] = val
                    self.backtrack(assignment, numAssigned + 1, weight * deltaWeight)
                    assignment[var] = None
        else:
            # Problem 3.1d
            # When arc consistency check is enabled.
            # BEGIN_YOUR_CODE (around 10 lines of code expected)
            #for val in ordered_values:

            raise Exception("Not implemented yet")
            # END_YOUR_CODE

    def get_unassigned_variable(self, assignment):
        """
        Given a partial assignment, return the index of a currently unassigned
        variable.

        @param assignment: A list of current assignment. This is the same as
            what you've seen so far.

        @return var: Index of a currently unassigned variable.
        """

        if not self.mcv:
            # Select a variable without any heuristics.
            for var in xrange(len(assignment)):
                if assignment[var] is None: return var
        else:
            # Problem 3.1b
            # Heuristic: most constrained variable (MCV)
            # Select a variable with the least number of remaining domain values.
            # BEGIN_YOUR_CODE (around 7 lines of code expected)
            domain = []
            for var in xrange(self.csp.numVars):
                if assignment[var] is None:
                    domainSize = 0
                    for val in self.domains[var]:
                        if self.get_delta_weight(assignment, var, val) > 0:
                            domainSize += 1
                    domain.append((var, domainSize))
            return min(domain, key = itemgetter(1))[0]
            #raise Exception("Not implemented yet")
            # END_YOUR_CODE

    def get_ordered_values(self, assignment, var):
        """
        Given an unassigned variable and a partial assignment, return an ordered
        list of indices of the variable's domain such that the backtracking
        algorithm will try |var|'s values according to this order.

        @param assignment: A list of current assignment. This is the same as
            what you've seen so far.
        @param var: The variable that's going to be assigned next.

        @return ordered_values: A list of indeces of |var|'s domain values.
        """
        if not self.lcv:
            # Return an order of value indices without any heuristics.
            # if var == None:
            #     import pdb; pdb.set_trace()
            return self.domains[var]
        else:
            # Problem 3.1c
            # Heuristic: least constraining value (LCV)
            # Return value indices in ascending order of the number of additional
            # constraints imposed on unassigned neighboring variables.
            # BEGIN_YOUR_CODE (around 17 lines of code expected)
            # Will update the domains! The unary constraint on var, val was already checked by backtrack before calling this method
            val_domain_constraints = []
            i = 0
            for val in self.domains[var]:
                var2_domain_size = 0
                for var2 in xrange(self.csp.numVars):
                    if var2 != var and assignment[var2] is None:
                        for val2 in self.domains[var2]:
                            if self.get_delta_weight(assignment, var2, val2) == 1:
                                    if self.csp.binaryPotentials[var][var2][val][val2] == 1:
                                        var2_domain_size += 1
                val_domain_constraints.append((i, val, var2_domain_size))
                i += 1
            ordered_pairs = sorted(val_domain_constraints, key=itemgetter(2), reverse = True)
            ordered_indices = [ele[0] for ele in ordered_pairs]
            return ordered_indices
            #raise Exception("Not implemented yet")
            # END_YOUR_CODE

    def arc_consistency_check(self, var):
        """
        Perform the AC-3 algorithm. The goal is to reduce the size of the
        domain values for the unassigned variables based on arc consistency.

        @param var: The variable whose value has just been set.

        While not required, you can also choose to add return values in this
        function if there's a need.
        '''
            csp.binaryPotentials =
            [{1: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              5: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]},

             {0: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              2: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              5: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]},

             {1: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              3: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              5: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]},

             {2: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              4: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              5: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]},

             {3: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              5: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]},

             {0: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              1: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              2: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              3: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              4: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]},

             {}]
        '''
        """
        # Problem 3.1d
        # BEGIN_YOUR_CODE (around 17 lines of code expected)

        #unassigned_vars = self.get_unassigned_variable(assignment)
        domain = copy.deepcopy(self.domains)
        for val in domain[var]
        for v in self.csp.varNames:
            for val in domain[v]:
                if self.csp.binaryPotentials[var][v][val][val] == 1:
        #raise Exception("Not implemented yet")
        # END_YOUR_CODE

############################################################
