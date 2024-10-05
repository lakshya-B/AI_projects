import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages

#The corpus is a Python dictionary mapping a page name to a set of all pages linked to by that page.
#The page is a string representing which page the random surfer is currently on.

def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    probability_dict = {}

    num_links = len(corpus[page])
    num_pages = len(corpus)

    # Check if the page has no links
    if num_links < 1:
        # Assign equal probability to all pages
        for name in corpus.keys():
            probability_dict[name] = 1 / num_pages
    else:
        # Assign probability based on damping factor and links
        for name in corpus:
            probability_dict[name] = (1 - damping_factor) / num_pages
            if name in corpus[page]:
                probability_dict[name] += damping_factor / num_links

    return probability_dict


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    #n is an integer representing the number of samples that should be generated to estimate PageRank values.
    page_rank = {}

    # Initialize the dictionary
    for name in corpus:
        page_rank[name] = 0.0

    random_page = random.choice(list(corpus))
    page_rank[random_page] = 1 / n

    for _ in range(n):
        model = transition_model(corpus, random_page, damping_factor)
        random_page = random.choices(list(model), weights=model.values(), k=1)[0]
        page_rank[random_page] += 1 / n

    return page_rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_rank = {}
    temp = {}

    N = len(corpus)

    RANDOM_FACTOR = (1 - damping_factor) / N

    # Initializing page values
    for name in corpus:
        page_rank[name] = 1 / N

    # Iterate until the value changes by more than 0.001
    while True:
        for name in page_rank:
            total = 0
            for page in corpus:
                if name in corpus[page]:
                    total += page_rank[page] / len(corpus[page])
                if not corpus[page]:
                    total += page_rank[page] / N

            # Update temp with new value
            temp[name] = RANDOM_FACTOR + total * damping_factor

        difference = max(abs(page_rank[page] - temp[page]) for page in page_rank)
        if difference < 0.001:
            break
        else:
            page_rank = temp.copy()

    return page_rank


if __name__ == "__main__":
    main()
