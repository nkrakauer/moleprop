import matplotlib.pyplot as plt
from  matplotlib import use, cm

class visualization:
    def parity_plot(X, y, z, verbose=False):
        """
        X: array of actual values
        y: array of predicted values
        z: array of encoded sources for the data point
        verbose: true if not using Euler 
        """
        if ~verbose:
            use('Agg')

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        # scatter with colormap mapping to z value
        ax.scatter(X,y,s=20,c=z, marker = 'o')
        max1 = max(X)   
        min1 = min(X)
        max2 = max(y)
        min2 = min(y)

        ax.plot([min1, min2], [max1, max2], color='black')
        
        ax.xlim(min1, max1)
        ax.ylim(min2, max2)
        ax.set_xlabel('Experimental Values (Kelvin)')
        ax.set_ylabel('Predicted Values (Kelvin')
        ax.set_title('Parity Plot')

        fig.savefig('parity_plot.png', bbox='tight')

