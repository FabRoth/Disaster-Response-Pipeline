import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load and merge Messages and Categories.
    
    Args:
    messages_filepath (str): Path of massages.csv
    categories_filepath (str): Path of categories.csv

    Returns:
    df (DataFrame): merged Messages and Categories 
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(messages, categories, on='id', how='left')
    return df
    


def clean_data(df):
    '''
    Convert categories into dummy coded variables and removes duplicates.
    The original column gets dropped.
    
    Args:
    df (DataFrame): Dataframe with messeges and categories

    Returns:
    df (DataFrame): cleaned Dataframe   
    '''
    
    categories = df['categories'].str.split(pat=';',expand=True)
    
    row = categories.iloc[0]
    category_colnames = [col[:-2] for col in row]
    
    categories.columns = category_colnames

    # converting values and types
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1] 
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')

    # drop the original categories column from `df`
    df.drop(columns='categories', inplace =True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    # drop duplicates
    df = df[~df.duplicated()]

    return df




def save_data(df, database_filename):
    '''
    Save Dataframe as SQL Database in a Table called MessagesAndCategories. 
    If the table exists it'll be replaced with the new data.

    Args:
    df (DataFrame) : Dataframe to be safed
    database_filename (str) : filename of the SQL Database (with file extension .db)
    '''

    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('MessagesAndCategories', engine, index=False, if_exists='replace')



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()