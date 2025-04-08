import pandas as pd

def concatenate_csv(file1, file2, output_file, header_row_file1=6, skiprows_file2=7):
    """
    Concatenates two CSV files, adjusting for headers and formats specific to each file.

    Parameters:
    - file1 (str): Path to the first CSV file, which includes headers on the 7th row.
    - file2 (str): Path to the second CSV file, which does not have a header.
    - output_file (str): Path where the concatenated CSV will be saved.
    - header_row_file1 (int): Row index of the header in the first file (0-indexed).

    Returns:
    - None: The function saves the concatenated file directly to the specified path.
    """

    # Read the first file, specifying which row to use as the header
    df1 = pd.read_csv(file1, header=header_row_file1)
    
    # Read the second file without headers and skipping metadata rows
    df2 = pd.read_csv(file2, header=None, skiprows=skiprows_file2)

    # Rename columns in df2 to match df1
    df2.columns = df1.columns

    # Adjust the 'Sample' column in df2 to continue from the last value of df1
    last_sample = df1['Sample'].iloc[-1]
    df2['Sample'] = df2['Sample'] - df2['Sample'].iloc[0] + last_sample + 1
    
    # Concatenate the dataframes, ensuring the order is maintained
    concatenated_df = pd.concat([df1, df2], ignore_index=True)

    # Save the concatenated dataframe to the specified output file
    concatenated_df.to_csv(output_file, index=False)

    print(f"Concatenated file saved to {output_file}")

# Example usage:
# concatenate_csv('/path/to/first_file.csv', '/path/to/second_file.csv', '/path/to/your/concatenated_file.csv')