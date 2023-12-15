import matplotlib.pyplot as plt
# import mpld3

def createTable(data):
        fig, ax = plt.subplots()

        # Create a table and add data to it
        table = ax.table(cellText=data, loc='center', cellLoc='center')

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)  # Adjust cell height for better readability

        # Remove axis labels and ticks
        ax.axis('off')

        plt.show()
