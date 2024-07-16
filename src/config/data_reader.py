# from llama_index.readers.web import FireCrawlWebReader
from llama_index.readers.web import SimpleWebPageReader
from llama_parse import LlamaParse
import nest_asyncio

def load_local_data():
    nest_asyncio.apply()
    return LlamaParse(result_type="markdown").load_data("./src/data/knowledge.pdf")

# def load_web_data(url: str):
#     firecrawl_reader = FireCrawlWebReader(
#         mode="scrape",
#         params={"additional": "parameters"},
#     )
#     return firecrawl_reader.load_data(url=url)


# load data using SimpleWebPageReader
def load_web_data(url: str):
    return SimpleWebPageReader(html_to_text=True).load_data([url])
