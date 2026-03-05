import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"


def get_links(page_title):
    """Retrieve links from a Wikipedia page using the MediaWiki API."""
    
    params = {
        "action": "query",
        "format": "json",
        "titles": page_title,
        "prop": "links",
        "pllimit": "max"
    }

    response = requests.get(WIKI_API, params=params)
    data = response.json()

    pages = data["query"]["pages"]

    links = []

    for page in pages.values():
        if "links" in page:
            for link in page["links"]:
                links.append(link["title"])

    return links


def main():
    print("Wikipedia Navigation AI")
    print("-----------------------")

    start_page = input("Start page: ")
    target_page = input("Target page: ")

    print("\nFetching links from:", start_page)

    links = get_links(start_page)

    print("Found", len(links), "links.")
    print("First 10 links:")

    for link in links[:10]:
        print("-", link)

    print("\nSearch algorithm will be implemented next.")


if __name__ == "__main__":
    main()
