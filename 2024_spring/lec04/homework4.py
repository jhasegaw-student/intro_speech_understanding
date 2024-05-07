
def list_to_dict(input_list):
    '''
    This function should return a dictionary in which each element of 
    `input_list` is a value, and the corresponding key is the numerical 
    index of that element in `input_list`. 
    '''
    mydict = {}
    for k, v in enumerate(input_list):
        mydict[k] = v
    return mydict
    




