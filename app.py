from flask import Flask, render_template, request, url_for
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import io
import base64

app = Flask(__name__)

def generate_plots(N, mu, sigma2, S):

    # STEP 1
    # TODO 1: Generate a random dataset X of size N with values between 0 and 1
    # and a random dataset Y with normal additive error (mean mu, variance sigma^2).
    # Hint: Use numpy's random's functions to generate values for X and Y
    X = np.random.rand(N)  # Generates random values between 0 and 1
    Y = mu + np.random.randn(N) * np.sqrt(sigma2)  # Normal distribution with mean mu and variance sigma2
    X_reshaped = X[:, np.newaxis]  # Reshape to 2D array with one column

    # TODO 2: Fit a linear regression model to X and Y
    # Hint: Use Scikit Learn
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(X_reshaped, Y)
    slope = reg.coef_[0]
    intercept = reg.intercept_

    # TODO 3: Generate a scatter plot of (X, Y) with the fitted regression line
    # Hint: Use Matplotlib
    # Label the x-axis as "X" and the y-axis as "Y".
    # Add a title showing the regression line equation using the slope and intercept values.
    # Finally, save the plot to "static/plot1.png" using plt.savefig()
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, alpha=0.5, label="Data Points")
    Y_pred = reg.predict(X_reshaped)
    plt.plot(X, Y_pred, 'r-', linewidth=2, label="Fitted Line")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Linear Regression: Y = {slope:.2f}X + {intercept:.2f}")
    plt.legend()
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.close()

    # Step 2: Run S simulations and create histograms of slopes and intercepts

    # TODO 1: Initialize empty lists for slopes and intercepts
    # Hint: You will store the slope and intercept of each simulation's linear regression here.
    slopes = []  # Initialize empty list for slopes
    intercepts = []  # Initialize empty list for intercepts

    # TODO 2: Run a loop S times to generate datasets and calculate slopes and intercepts
    # Hint: For each iteration, create random X and Y values using the provided parameters
    for _ in range(S):
        X_sim = np.random.rand(N)
        Y_sim = mu + np.random.randn(N) * np.sqrt(sigma2)
        
        X_sim_reshaped = X_sim[:, np.newaxis]
        sim_model = LinearRegression()
        sim_model.fit(X_sim_reshaped, Y_sim)
        
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    # Plot histograms of slopes and intercepts
    plt.figure(figsize=(12, 6))  
    plt.hist(slopes, bins=25, alpha=0.6, color="purple", label="Slopes")  #
    plt.hist(intercepts, bins=25, alpha=0.6, color="green", label="Intercepts")  # 
    plt.axvline(slope, color="darkblue", linestyle=":", linewidth=1.5, label=f"Slope: {slope:.2f}")  # 
    plt.axvline(intercept, color="darkgreen", linestyle=":", linewidth=1.5, label=f"Intercept: {intercept:.2f}")  # 
    plt.title("Distribution of Slopes and Intercepts from Simulations", fontsize=14)  # 
    plt.xlabel("Value", fontsize=12)  # 
    plt.ylabel("Frequency", fontsize=12)  # 
    plt.legend(loc="upper right")  # 
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path, dpi=150)  # 
    plt.close()

    # Below code is already provided
    # Calculate proportions of more extreme slopes and intercepts
    # For slopes, we will count how many are greater than the initial slope; for intercepts, count how many are less.
    slope_more_extreme = sum(s > slope for s in slopes) / S  # Already provided
    intercept_more_extreme = sum(i < intercept for i in intercepts) / S  # Already provided

    return plot1_path, plot2_path, slope_more_extreme, intercept_more_extreme

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        S = int(request.form["S"])

        # Generate plots and results
        plot1, plot2, slope_extreme, intercept_extreme = generate_plots(N, mu, sigma2, S)

        return render_template("index.html", plot1=plot1, plot2=plot2,
                               slope_extreme=slope_extreme, intercept_extreme=intercept_extreme, N=N, mu=mu, sigma2=sigma2, S=S)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
