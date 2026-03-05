import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"


def get_links(page_title):
    """Retrieve links from a Wikipedia page using the MediaWiki API."""

    params = {
        "action": "query",
        "format": "json",
        "titles": page_title,
        "prop": "links",
        "pllimit": "max",
        "redirects": 1
    }

    headers = {
        "User-Agent": "WikiNavigationAI/1.0 (student project)"
    }

    response = requests.get(WIKI_API, params=params, headers=headers)

    # check if request succeeded
    if response.status_code != 200:
        print("HTTP error:", response.status_code)
        return []

    data = response.json()

    pages = data["query"]["pages"]

    links = []

    for page in pages.values():
        if "links" in page:
            for link in page["links"]:
                title = link["title"]

                if ":" not in title:
                    links.append(title)

    return links


def main():
    print("Wikipedia Navigation AI")
    print("-----------------------")

    start_page = input("Start page: ").strip()
    target_page = input("Target page: ").strip()

    # normalize capitalization for Wikipedia titles
    start_page = start_page[:1].upper() + start_page[1:]
    target_page = target_page[:1].upper() + target_page[1:]

    print("\nFetching links from:", start_page)

    links = get_links(start_page)

    print("Found", len(links), "links.")
    print("First 10 links:")

    for link in links[:10]:
        print("-", link)

    print("\nSearch algorithm will be implemented next.")


if __name__ == "__main__":
    main()
