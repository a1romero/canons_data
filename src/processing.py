import pytesseract
import fitz
from PIL import Image
import pandas as pd
from pandas import DataFrame as df
from io import StringIO
from pathlib import Path
import csv
import re
from tqdm import tqdm

def pdf_to_data(pdf_path: str, output_folder: str, tesseract_path: str, include_pngs= False):
    '''Chop a given PDF into individual pages, then convert each PDF into an image (saved to the pngs folder). Convert OCR data about each page into a .csv.
        This function requires that you have a folder in the same level as your pdf for outputs, and then two folders within that folder titled 'pngs' and 'tsv_data'.
        For example:
            your pdf
            output folder
                pngs ** Only necessary if you want to see the processed pages-- set include_pngs to True'''
    
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

    open_pdf = fitz.open(pdf_path)

    sum_string = ''
    tsv_total = pd.DataFrame()
    for page_num in tqdm(range(open_pdf.page_count)): # iterate through individual pages
        page = open_pdf[page_num]

        img = page.get_pixmap()
        

        # make the image
        pil_img = Image.frombytes("RGB", [img.width, img.height], img.samples) # convert to PIL Image
        # improve resolution
        scale_factor = 3 # try changing this to improve resolution
        new_size = img.width * scale_factor, img.height * scale_factor
        resize = pil_img.resize(new_size, Image.LANCZOS)
        

        # process image into tsv and clean some values
        png_to_data = pytesseract.image_to_data(resize, config = r'--psm 6') 
        data_to_tsv = StringIO(png_to_data)
        data_read = pd.read_csv(data_to_tsv, sep='\t', quoting=csv.QUOTE_NONE)
        tsv_clean = data_read[['line_num', 'word_num', 'left', 'top', 'text', 'conf']]
        tsv_total = pd.concat([tsv_total, tsv_clean])

        img_to_string = pytesseract.image_to_string(resize, config = r'--psm 6', lang = 'eng')
        sum_string += img_to_string
        
        if include_pngs:
            output_png_path = Path(f"{output_folder}/pngs/page_{page_num + 1}.png")
            resize.save(output_png_path)

    output_tsv_total = Path(f"{output_folder}/location.csv")
    tsv_total.to_csv(output_tsv_total, sep= ',', index=False)
    output_str_path = Path(f"{output_folder}/str_data.txt")
    with open(output_str_path, 'w') as file:
        file.write(sum_string)
    print("Done!")

import csv
import spacy
import re

def data_to_csv(input_folder: str)->None:
    nlp = spacy.load('en_core_web_sm')
    input_path = f'{input_folder}/shortened_str.txt'
    output_path = f'{input_folder}/output_data.csv'

    title_page_pattern = re.compile(r'^(.*?)\s+(\d+)$')

    parsed_data = []
    last_author = None

    with open(input_path, 'r', encoding = 'utf8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            doc = nlp(line)

            is_author_line = False
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    last_author = ent.text.strip()
                    is_author_line = True
                    break
            
            if is_author_line:
                continue

            title_page_match = title_page_pattern.match(line)
            if title_page_match:
                work = title_page_match.group(1).strip()
                page_number = title_page_match.group(2)

                parsed_data.append([work, last_author or 'Unknown', page_number])
                continue

            if last_author and not title_page_match:
                parsed_data.append([line, last_author, 'N/A'])

            with open(output_path, mode = 'w', newline = '') as file:
                writer = csv.writer(file)
                writer.writerow(['Work', 'Author', 'Page Number'])
                writer.writerows(parsed_data)

    print(f'Data successfully written to {output_path}')