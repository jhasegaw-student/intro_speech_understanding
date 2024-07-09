import bs4, gtts

def extract_stories_from_NPR_text(text):
    '''
    Extract a list of stories from the text of the npr.com webpage.
    
    @params: 
    text (string): the text of a webpage
    
    @returns:
    stories (list of tuples of strings): a list of the news stories in the web page.
      Each story should be a tuple of (title, teaser), where the title and teaser are
      both strings.  If the story has no teaser, its teaser should be an empty string.
    '''
    #raise RuntimeError('You need to write this part!')
    soup = bs4.BeautifulSoup(text, "html.parser")
    stories = []
    for div_tag in soup.find_all('div', 'story-text'):
        titletag = div_tag.find('h3', 'title')
        teasertag = div_tag.find('p', 'teaser')
        
        if teasertag != None:
            stories.append((titletag.text, teasertag.text))
        else:
            stories.append((titletag.text, ""))
    return stories
    
def read_nth_story(stories, n, filename):
    '''
    Read the n'th story from a list of stories.
    
    @params:
    stories (list of tuples of strings): a list of the news stories from a web page
    n (int): the index of the story you want me to read
    filename (str): filename in which to store the synthesized audio

    Output: None
    '''
    #raise RuntimeError('You need to write this part!')
    gtts.gTTS(text=stories[n][0]+" "+stories[n][1], lang="en").save(filename)
