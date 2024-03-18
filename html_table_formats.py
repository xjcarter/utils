
import pandas as pd
from bs4 import BeautifulSoup

def create_html_table(dfi, title):

    HARD_BLUE = '#0E2A52'
    SOFT_BLUE = '#315BA5'
    INDENT = '&nbsp;' * 2 

    # Convert the DataFrame to an HTML string with no border
    html_string = df.to_html(index=False, border=0)

    # Parse the HTML content
    soup = BeautifulSoup(html_string, 'html.parser')

    # Add the title above the table with the specified styles
    title_html = f'<div style="text-align: center; color: {HARD_BLUE}; font-family: Verdana;'
    title_html += f'font-size: 16pt; border: none; text-align: left;'
    title_html += f'margin-top: 20px; margin-bottom: 20px;">{INDENT}{title}</div>'
    title_soup = BeautifulSoup(title_html, 'html.parser')

    # Insert the styled title above the table and add a blank line (margin) for spacing
    soup.table.insert_before(title_soup)

    # Find the table header row and apply Helvetica font style, bold font weight, and padding
    header_row = soup.find('tr')
    if header_row:
        for th in header_row.find_all('th'):
            #th_style = 'font-family: Helvetica; font-size: 14pt; font-weight: bold; padding: 5px;'
            th_style = 'font-family: Verdana; font-size: 11pt; padding: 8px;'
            th_style += f' text-align: center; background-color: {HARD_BLUE}; color: white;'
            if th.text == 'Age':  # Center the 'Age' column header
                th_style += 'text-align: center;'
            if th.text == 'Occupation':  # Right-justify the "Occupation" column header
                th_style += 'text-align: right;'
            if th.text == 'Name':  # Right-justify the "Occupation" column header
                th_style += 'text-align: left;'
            th['style'] = th_style

    # Style all non-header rows
    rows = soup.find_all('tr')[1:]
    for index, row in enumerate(rows):
        # Apply general styles and specific styles for even rows
        for td in row.find_all('td'):
            #td_style = 'padding: 5px; font-family: Courier New; font-size: 14pt;'
            td_style = 'padding: 5px; font-family: Verdana; font-size: 12pt;'
            if index % 2 == 0:  # Style even rows differently
                td_style += f'background-color: {SOFT_BLUE}; color: white;'
            else:
                td_style += f'background-color: white; color: {HARD_BLUE};'
            
            # Center the 'Age' column values
            index_of_age = df.columns.get_loc('Age')
            if td.find_previous_siblings('td') is not None and len(td.find_previous_siblings('td')) == index_of_age:
                td_style += ' text-align: center;'

            # Right-justify the "Occupation" column values
            index_of_occupation = df.columns.get_loc('Occupation')
            if td.find_previous_siblings('td') is not None and len(td.find_previous_siblings('td')) == index_of_occupation:
                td_style += ' text-align: right;'

            td['style'] = td_style

    # Return modified HTML as a string
    return str(soup)


## test
if __name__ == '__main__':
    import pandas
    import random

    tbl = list()
    for i in range(15):
        names = ['Alice', 'Tom', 'Charlie', 'Diana', 'Evan', 'Barbie', 'Buddy', 'TJ', 'Cameron', 'Cole']
        jobs = ['Plumber', 'Racer', 'Stripper', 'Boxer', 'Driver', 'Sales', 'Cop', 'Fireman', 'Athlete', 'Doctor']

        name = random.choice(names)
        job = random.choice(jobs)
        age = random.randint(20,90)

        m = dict(Name=name, Occupation=job, Age=age)
        tbl.append(m)

    df = pandas.DataFrame(tbl)
    new_html = create_html_table(df, 'Job Title')
    print(new_html)


