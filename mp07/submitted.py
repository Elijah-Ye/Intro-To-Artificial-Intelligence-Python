'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import copy, queue

def standardize_variables(nonstandard_rules):
    '''
    @param nonstandard_rules (dict) - dict from ruleIDs to rules
        Each rule is a dict:
        rule['antecedents'] contains the rule antecedents (a list of propositions)
        rule['consequent'] contains the rule consequent (a proposition).
   
    @return standardized_rules (dict) - an exact copy of nonstandard_rules,
        except that the antecedents and consequent of every rule have been changed
        to replace the word "something" with some variable name that is
        unique to the rule, and not shared by any other rule.
    @return variables (list) - a list of the variable names that were created.
        This list should contain only the variables that were used in rules.
    '''
    standardized_rules = copy.deepcopy(nonstandard_rules)
    original_var = 0
    variables = []
    for id in standardized_rules:
      for value in standardized_rules[id]['antecedents']:
        for i in range(len(value)):
          if value[i] == 'something':
            value[i] = str(original_var)
            variables.append(value[i])
      for i in range(len(standardized_rules[id]['consequent'])):
        if standardized_rules[id]['consequent'][i] == 'something':
          standardized_rules[id]['consequent'][i] = str(original_var)
      original_var += 1
    
    return standardized_rules, variables

def unify(query, datum, variables):
    '''
    @param query: proposition that you're trying to match.
      The input query should not be modified by this function; consider deepcopy.
    @param datum: proposition against which you're trying to match the query.
      The input datum should not be modified by this function; consider deepcopy.
    @param variables: list of strings that should be considered variables.
      All other strings should be considered constants.
    
    Unification succeeds if (1) every variable x in the unified query is replaced by a 
    variable or constant from datum, which we call subs[x], and (2) for any variable y
    in datum that matches to a constant in query, which we call subs[y], then every 
    instance of y in the unified query should be replaced by subs[y].

    @return unification (list): unified query, or None if unification fails.
    @return subs (dict): mapping from variables to values, or None if unification fails.
       If unification is possible, then answer already has all copies of x replaced by
       subs[x], thus the only reason to return subs is to help the calling function
       to update other rules so that they obey the same substitutions.

    Examples:

    unify(['x', 'eats', 'y', False], ['a', 'eats', 'b', False], ['x','y','a','b'])
      unification = [ 'a', 'eats', 'b', False ]
      subs = { "x":"a", "y":"b" }
    unify(['bobcat','eats','y',True],['a','eats','squirrel',True], ['x','y','a','b'])
      unification = ['bobcat','eats','squirrel',True]
      subs = { 'a':'bobcat', 'y':'squirrel' }
    unify(['x','eats','x',True],['a','eats','a',True],['x','y','a','b'])
      unification = ['a','eats','a',True]
      subs = { 'x':'a' }
    unify(['x','eats','x',True],['a','eats','bobcat',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],
      subs = {'x':'a', 'a':'bobcat'}
      When the 'x':'a' substitution is detected, the query is changed to 
      ['a','eats','a',True].  Then, later, when the 'a':'bobcat' substitution is 
      detected, the query is changed to ['bobcat','eats','bobcat',True], which 
      is the value returned as the answer.
    unify(['a','eats','bobcat',True],['x','eats','x',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],
      subs = {'a':'x', 'x':'bobcat'}
      When the 'a':'x' substitution is detected, the query is changed to 
      ['x','eats','bobcat',True].  Then, later, when the 'x':'bobcat' substitution 
      is detected, the query is changed to ['bobcat','eats','bobcat',True], which is 
      the value returned as the answer.
    unify([...,True],[...,False],[...]) should always return None, None, regardless of the 
      rest of the contents of the query or datum.
    '''
    unification = []
    subs = {}
    
    if query[-1] != datum[-1]:
      return None, None
    
    unification = copy.deepcopy(query)
    datum_cp = copy.deepcopy(datum)
    
    for i in range(len(unification) - 1):
      temp_u = copy.deepcopy(unification[i])
      temp_d = copy.deepcopy(datum_cp[i])
      if unification[i] in variables and datum_cp[i] not in variables:
        for j in range(len(unification) - 1):
          if unification[j] == temp_u:
            unification[j] = datum_cp[i]
        subs[temp_u] = datum_cp[i]
      if unification[i] not in variables and datum_cp[i] in variables:
        for j in range(len(datum_cp) - 1):
          if datum_cp[j] == temp_d:
            datum_cp[j] = temp_u
          if unification[j] == temp_d:
            unification[j] = temp_u
        subs[temp_d] = unification[i]
      if unification[i] in variables and datum_cp[i] in variables:
        for j in range(len(unification) - 1):
          if unification[j] == temp_u:
            unification[j] = datum_cp[i]
        subs[temp_u] = datum_cp[i]
      if unification[i] not in variables and datum_cp[i] not in variables and unification[i] != datum_cp[i]:
        return None, None
    
    return unification, subs


def apply(rule, goals, variables):
  
    applications = []
    goalsets = []
    
    for goal in goals:
      rule_cpy = copy.deepcopy(rule)
      unif, subs = unify(rule_cpy['consequent'], goal, variables) 
      if unif != None and subs != None:
        goals_cpy = copy.deepcopy(goals)
        rule_cpy['consequent'] = unif
        goals_cpy.remove(goal)
        for i in range(len(rule_cpy['antecedents'])): 
          for j in range(len(rule_cpy['antecedents'][i])):
            cur_word = rule_cpy['antecedents'][i][j]
            while  cur_word in subs:           
              cur_word = subs[cur_word]
            rule_cpy['antecedents'][i][j] = cur_word
          goals_cpy.append(rule_cpy['antecedents'][i])         
        applications.append(rule_cpy)      
        goalsets.append(goals_cpy)
          
    return applications, goalsets


def backward_chain(query, rules, variables):
    '''
    @param query: a proposition, you want to know if it is true
    @param rules: dict mapping from ruleIDs to rules
    @param variables: list of strings that should be treated as variables

    @return proof (list): a list of rule applications
      that, when read in sequence, conclude by proving the truth of the query.
      If no proof of the query was found, you should return proof=None.
    '''
    proof = []
    b_queue = queue.Queue()
    
    rules_cp = copy.deepcopy(rules)
    
    b_queue.put([query])
    
    while not b_queue.empty():
      temp = b_queue.get()
      if len(temp) == 0: break
      for rule in rules:
        if rule not in rules_cp: continue
        applications, goalsets = apply(rules[rule], temp, variables)
        if len(applications) > 0: 
          del rules_cp[rule]
          for app in applications:
            proof.insert(0, app)
          for goal in goalsets:
            b_queue.put(goal)
    if len(proof) > 0: return proof
          
    return None
