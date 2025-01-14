import csv 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FFT = False
Sac_tracing = False

Fibril = True
option = ["numb", "Number of Fibrils", "Total number of Fibrils"]
# option = ["l", "Fibril lenght (µm)", "Average Fibril lenght"]
# option = ["w", "Fibril width (µm)", "Average Fibril width"]

All = False
Or = True

groups = [("MH21","MH5"), ("MH25", "MH26"), ("MH28", "MH27"), ("MH7", "MH8.1", "MH8.2")]


#--------------------------------------FFT----------------------------------------------#

if FFT:
    
    file_path = 'Statistics/Results_Sacromere_FFT.csv'
    fft_lengths = []

    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)  

        for row in reader:
            sample_name = row[0].split(" ")[2].split("_")[0]  
            length = float(row[2].replace(',', '.'))  
            condition = row[1]  
                
            fft_lengths.append([sample_name, condition, length])

    stimulated = [row for row in fft_lengths if row[1] == 'Stimulated']
    control = [row for row in fft_lengths if row[1] == 'Control']
    ordered_data = control + stimulated

    data_for_boxplot = []
    for i,j, value in ordered_data:
        data_for_boxplot.append((i, value))
                
    df = pd.DataFrame(data_for_boxplot, columns=['Group', 'Average Sacromere lenghts'])


    if All:
        # Create a color palette
        # The first four boxes will be 'Control' and the rest 'Stimulated'
        palette = ['#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4',  # Control colors
                '#ff7f0e', '#ff7f0e', '#ff7f0e',  '#ff7f0e','#ff7f0e']  # Stimulated colors

        # Create the box plot
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Group', y='Average Sacromere lenghts', data=df, palette=palette)

        # Add legend
        control_patch = plt.Line2D([0], [0], color='#1f77b4', lw=4, label='Control')
        stimulated_patch = plt.Line2D([0], [0], color='#ff7f0e', lw=4, label='Stimulated')
        plt.legend(handles=[control_patch, stimulated_patch], title="Group Type")

        # Customize plot
        plt.title('FFT - Average Sacromere lenght')
        plt.ylabel('Sacromere lenght (µm)')
        plt.xlabel('Groups')
        plt.grid(True)
        plt.tight_layout()  # Adjust layout
        plt.show()


    else:
        print(df)
        palette = ['#1f77b4', '#ff7f0e', '#ff7f0e'] 

        # Create the figure with 3 subplots
        fig, axes = plt.subplots(1, 4, figsize=(18, 6))

        for i, group_pair in enumerate(groups):
            # Filter the DataFrame for the group pair
            group_data = df[df['Group'].isin(group_pair)]
            # Create a boxplot on the corresponding axis
            sns.boxplot(ax=axes[i], x='Group', y='Average Sacromere lenghts', data=group_data, palette=palette) 
            axes[i].grid(True)
            axes[i].set_ylabel('Sacromere Length (µm)')

        # Add a single legend for all subplots
        control_patch = plt.Line2D([0], [0], color='#1f77b4', lw=4, label='Control')
        stimulated_patch = plt.Line2D([0], [0], color='#ff7f0e', lw=4, label='Stimulated')
        fig.legend(handles=[control_patch, stimulated_patch], title="Group Type", loc='upper center', ncol=2)

        plt.tight_layout(pad=3)
        plt.show()



#--------------------------------------Sac_tracing----------------------------------------------#

if Sac_tracing:
    
    file_path = 'Statistics/Results_Sacromere_analysis.csv'
    fft_lengths = []

    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)  

        for row in reader:
            sample_name = row[0].split(" ")[3].split("_")[0]  
            length = float(row[2].replace(',', '.'))  
            condition = row[1]  
                
            fft_lengths.append([sample_name, condition, length])
    print(fft_lengths)

    stimulated = [row for row in fft_lengths if row[1] == 'Stimulated']
    control = [row for row in fft_lengths if row[1] == 'Control']
    ordered_data = control + stimulated

    data_for_boxplot = []
    for i,j, value in ordered_data:
        data_for_boxplot.append((i, value))
                
    df = pd.DataFrame(data_for_boxplot, columns=['Group', 'Average Sacromere lenghts'])


    if All:
        # Create a color palette
        # The first four boxes will be 'Control' and the rest 'Stimulated'
        palette = ['#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4',  # Control colors
                '#ff7f0e', '#ff7f0e', '#ff7f0e',  '#ff7f0e','#ff7f0e']  # Stimulated colors

        # Create the box plot
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Group', y='Average Sacromere lenghts', data=df, palette=palette)

        # Add legend
        control_patch = plt.Line2D([0], [0], color='#1f77b4', lw=4, label='Control')
        stimulated_patch = plt.Line2D([0], [0], color='#ff7f0e', lw=4, label='Stimulated')
        plt.legend(handles=[control_patch, stimulated_patch], title="Group Type")

        # Customize plot
        plt.title('Sacromere Tracing - Average Sacromere lenght')
        plt.ylabel('Sacromere lenght (µm)')
        plt.xlabel('Groups')
        plt.grid(True)
        plt.tight_layout()  # Adjust layout
        plt.show()


    else:
        print(df)
        palette = ['#1f77b4', '#ff7f0e','#ff7f0e'] 

        # Create the figure with 3 subplots
        fig, axes = plt.subplots(1, 4, figsize=(18, 6))

        for i, group_pair in enumerate(groups):
            # Filter the DataFrame for the group pair
            group_data = df[df['Group'].isin(group_pair)]
            # Create a boxplot on the corresponding axis
            sns.boxplot(ax=axes[i], x='Group', y='Average Sacromere lenghts', data=group_data, palette=palette) 
            axes[i].grid(True)
            axes[i].set_ylabel('Sacromere Length (µm)')



        # Add a single legend for all subplots
        control_patch = plt.Line2D([0], [0], color='#1f77b4', lw=4, label='Control')
        stimulated_patch = plt.Line2D([0], [0], color='#ff7f0e', lw=4, label='Stimulated')
        fig.legend(handles=[control_patch, stimulated_patch], title="Group Type", loc='upper center', ncol=2)

        plt.tight_layout(pad=3)
        plt.show()



#--------------------------------------Fibril----------------------------------------------#


if Fibril:
    
    info = []
    
    file_path = 'Statistics/Results_Fibril.csv'

    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)  

        for row in reader:
            sample_name = row[0].split(" ")[3].split("_")[0]  
            length = float(row[2].replace(',', '.'))  
            condition = row[1]  
            
            number = int(row[2])
            lenght = float(row[3].replace(',', '.'))  
            lenght_std = float(row[4].replace(',', '.'))  
            
            width = float(row[5].replace(',', '.'))  
            width_std = float(row[6].replace(',', '.'))  
            
            or1 = float(row[9].replace(',', '.'))  
            or2 = float(row[10].replace(',', '.'))  
            or3 = float(row[11].replace(',', '.'))  
            or4 = float(row[12].replace(',', '.'))  
            
                
            info.append([sample_name, condition, number, lenght, lenght_std, width, width_std, or1, or2, or3, or4])

    stimulated = [row for row in info if row[1] == 'Stimulated']
    control = [row for row in info if row[1] == 'Control']
    ordered_data = control + stimulated
    
    data_for_boxplot = []
    for sample_name, condition, number, lenght, lenght_std, width, width_std, or1, or2, or3, or4 in ordered_data:
        data_for_boxplot.append((sample_name, condition, number, lenght, lenght_std, width, width_std, or1, or2, or3, or4))
                
    df = pd.DataFrame(data_for_boxplot, columns=['Group', "con", "numb", "l", "lstd", "w", "wstd", "or1", "or2","or3", "or4"])
    print(df)
    
    if All:
        palette = ['#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4',  # Control colors
                    '#ff7f0e', '#ff7f0e', '#ff7f0e',  '#ff7f0e','#ff7f0e']  # Stimulated colors

        # Create the box plot
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Group', y=option[0], data=df, palette=palette)

        # Add legend
        control_patch = plt.Line2D([0], [0], color='#1f77b4', lw=4, label='Control')
        stimulated_patch = plt.Line2D([0], [0], color='#ff7f0e', lw=4, label='Stimulated')
        plt.legend(handles=[control_patch, stimulated_patch], title="Group Type")

        # Customize plot
        plt.title(option[2])
        plt.ylabel(option[1])
        plt.xlabel('Groups')
        plt.grid(True)
        plt.tight_layout()  # Adjust layout
        plt.show()
    
    else:
        palette = ['#1f77b4', '#ff7f0e','#ff7f0e'] 

        # Create the figure with 3 subplots
        fig, axes = plt.subplots(1, 4, figsize=(18, 6))

        for i, group_pair in enumerate(groups):
            # Filter the DataFrame for the group pair
            group_data = df[df['Group'].isin(group_pair)]
            # Create a boxplot on the corresponding axis
            sns.boxplot(ax=axes[i], x='Group', y=option[0], data=group_data, palette=palette) 
            axes[i].grid(True)
            axes[i].set_ylabel(option[1])



        # Add a single legend for all subplots
        control_patch = plt.Line2D([0], [0], color='#1f77b4', lw=4, label='Control')
        stimulated_patch = plt.Line2D([0], [0], color='#ff7f0e', lw=4, label='Stimulated')
        fig.legend(handles=[control_patch, stimulated_patch], title="Group Type", loc='upper center', ncol=2)

        plt.tight_layout(pad=3)
        plt.show()
        
        
        
    if Or:
        df = pd.DataFrame(data_for_boxplot, columns=['Group', "con", "numb", "l", "lstd", "w", "wstd", "or1", "or2","or3", "or4"])
        averages = df.groupby("Group")[["or1", "or2", "or3", "or4"]].mean().reset_index()

        palette = ['#1f77b4', '#ff7f0e','#ff7f0e'] 
        

        melted_df = df.melt(id_vars=["Group", "con"], value_vars=["or1", "or2", "or3", "or4"],
                            var_name="Orientation", value_name="Value")

        # Create a separate figure for each pair of groups
        for pair in groups:
            # Create a figure with subplots for each group in the pair
            num_groups = len(pair)
            fig, axes = plt.subplots(1, num_groups, figsize=(5 * num_groups, 5), constrained_layout=True)
            
            for ax, group in zip(axes, pair):
                group_data = melted_df[melted_df["Group"] == group]
                
                vis = ["|","/","-","\\"]*9
                
                # sns.boxplot(data=group_data, x="Orientation", y="Value", ax=ax,hue= "con", palette={"Control": "#1f77b4", "Stimulated": "#ff7f0e"})
                sns.barplot(data=group_data, x=vis, y="Value", ax=ax,hue= "con", palette={"Control": "#1f77b4", "Stimulated": "#ff7f0e"},
                            capsize=.1, legend=False)
                ax.set_title(f"Group {group}")
                ax.set_xlabel("Orientation")
                ax.set_ylabel("Orientation Frequency")
            
            # Set a common title for the figure
            fig.legend(handles=[control_patch, stimulated_patch], title="Group Type")
            # fig.suptitle(f"Distribution of Orientations for Groups {', '.join(pair)}", fontsize=16)
            plt.show()
