
def next_birthday(date, birthdays):
    '''
    Find the next birthday after the given date.

    @param:
    date - a tuple of two integers specifying (month, day)
    birthdays - a dict mapping from date tuples to lists of names, for example,
      birthdays[(1,10)] = list of all people with birthdays on January 10.

    @return:
    birthday - the next day, after given date, on which somebody has a birthday
    list_of_names - list of all people with birthdays on that date
    '''
    birthday = (1,1)
    list_of_names = []
    for testdate in sorted(birthdays.keys()):
        if testdate >= date:
            return testdate, birthdays[testdate]
            
    # If we reach this point, there was no testdate after the input date
    # so we should go back to the start of the year and return the first date in the dictionary
    for testdate in sorted(birthdays.keys()):
        return testdate, birthdays[testdate]
    
    
