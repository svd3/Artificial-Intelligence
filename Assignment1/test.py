import graderUtil, util, submission
import time


t0 = time.time()
if True:
    ## 2b
    scenario = util.deliveryScenario1
    ucs = util.UniformCostSearch()
    ucs.solve(submission.DeliveryProblem(scenario))
    #scenario.simulate(ucs.actions, True)  # Visualize the solution
    print ucs.numStatesExplored, 'number of states explored.'
    
    # 2c
    scenario = util.deliveryScenario1
    problem = submission.DeliveryProblem(scenario)
    astar = submission.AStarSearch(submission.createHeuristic1(scenario))
    astar.solve(problem)
    print astar.numStatesExplored

    # 2d
    scenario = util.deliveryScenario2
    problem = submission.DeliveryProblem(scenario)
    astar = submission.AStarSearch(submission.createHeuristic2(scenario, 0))
    astar.solve(problem)
    print astar.numStatesExplored

    # 2e
    scenario = util.deliveryScenario3
    problem = submission.DeliveryProblem(scenario)
    astar = submission.AStarSearch(submission.createHeuristic3(scenario))
    astar.solve(problem)
    print astar.numStatesExplored
t1 = time.time()

print t1-t0
