import dask.dataframe as dd
from dask import delayed

# Step 1: Define column names
column_names = ['id', 'timestamp', 'value', 'property', 'plug_id', 'household_id', 'house_id']

# Step 2: Read the large CSV file using Dask
df = dd.read_csv('/home/duongtm/Documents/DATN/sorted.csv', header=None, names=column_names, dtype={
    'id': 'int32', 
    'timestamp': 'int32',  
    'value': 'float64', 
    'property': 'int32', 
    'plug_id': 'int32', 
    'household_id': 'int32', 
    'house_id': 'int32'
})

# Step 3: List of house_ids to filter
house_ids = [0, 5, 10, 15, 20, 25, 30, 35]

# Step 4: Define a function to filter and save each house_id to a CSV
# @delayed
# def save_filtered_csv(df, house_id):
for house_id in house_ids:
    # Filter data for the current house_id
    filtered_df = df[df['house_id'] == house_id]
    
    # Define output filename
    output_filename = f'house_{house_id}.csv'
    
    # Write filtered data to CSV
    filtered_df.to_csv(output_filename, single_file=True, index=False)
    
    # return output_filename

# # Step 5: Create delayed tasks for each house_id
# tasks = [save_filtered_csv(df, house_id) for house_id in house_ids]

# # Step 6: Trigger parallel execution of all tasks
# dd.compute(*tasks)

print("Filtered data saved to separate files for each house_id.")
