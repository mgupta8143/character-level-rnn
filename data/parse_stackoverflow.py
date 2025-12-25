import xml.etree.ElementTree as ET
import os

"""
Script to parse Stack Overflow Posts.xml dump.
This script is NOT committed to the repo.
It reads a 'Posts.xml' file (downloaded separately) and extracts 
text from questions and answers into 'input.txt'.
"""

INPUT_XML = 'Posts.xml'  # Expects Posts.xml in this data/ folder
OUTPUT_TXT = 'input.txt' # Outputs input.txt in this data/ folder
LIMIT_COUNT = 10000      # Limit to first N posts for sanity

def parse_posts():
    if not os.path.exists(INPUT_XML):
        print(f"Error: {INPUT_XML} not found.")
        print("Please download the Stack Overflow data dump and extract Posts.xml here.")
        return

    print(f"Parsing {INPUT_XML}...")
    
    count = 0
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f_out:
        # Iterative parsing to handle large XML files
        context = ET.iterparse(INPUT_XML, events=('end',))
        
        for event, elem in context:
            if elem.tag == 'row':
                # Extract Body (and maybe Title)
                body = elem.get('Body', '')
                title = elem.get('Title', '')
                
                # Basic cleaning (very rough)
                text = title + "\n" + body + "\n"
                
                # Remove HTML tags (quick and dirty way, better to use BeautifulSoup if allowed)
                # But to keep it simple and dependency-free, we'll leave it or do minimal cleanup
                # For a character RNN, sometimes keeping HTML is fine (it learns to generate HTML!)
                # Or we can strip:
                # text = ''.join(ET.fromstring(text).itertext()) # This fails on fragments
                
                f_out.write(text)
                
                count += 1
                if count >= LIMIT_COUNT:
                    break
                
                elem.clear() # Free memory
                
    print(f"Finished. Extracted {count} posts to {OUTPUT_TXT}")

if __name__ == "__main__":
    parse_posts()
