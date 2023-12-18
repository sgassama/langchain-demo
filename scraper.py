import requests
from bs4 import BeautifulSoup

def scrape_youtube_comments(url):
    response = requests.get(url)
    print(f"response ----> {response}")
    soup = BeautifulSoup(response.text, 'html.parser')
    comments = soup.find_all('div')
    print(f"comments ----> {comments}")

    for comment in comments:
        print(comment.text)

if __name__ == '__main__':
    url = input("Enter URL:")
    print(f"url ----> {url}")
    if url:
        scrape_youtube_comments(url)