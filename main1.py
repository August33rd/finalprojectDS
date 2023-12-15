import streamlit as st
from PIL import Image

st.title(" :blue[Final Project :] ")
st.title(" :blue[Optiver - Trading at the Close] ")

#image1 = Image.open("img1.png")


st.header("Introduction")
st.write(""" 
#         
        Trading at the close
Closing prices serve as a key indicator in evaluating the performance of individual stocks. 
#         
The Kaggle competition is all about closing price discovery since the closing auction determines the benchmark for investment strategies. Let’s consider some financial concepts related to trading at the close. In an auction, multiple buyers and sellers interact directly in a controlled, regulated environment. In a closing auction the exchange begins accepting orders at the start of the day but does not publish the state of the auction until 3:50pm eastern time for 10 minutes before the market closes at 4pm eastern time. Currently orders are matched instantly at a single price. 
#
Let’s compare a closing auction to an Ebay auction. In Ebay auctions the current highest bid is displayed in real-time and bidders can see the leading bid at any moment allowing them to choose to place a higher bid. There is more transparency to allow bidders to adjust their bids. Whereas in a closing auction, the state of the auction is not published until 10min before the close. 
#
         How is the closing price determined?
The price is determined by which the maximum number of shares can be matched. NASDAQ uses a proprietary algorithm that considers the last traded price, price-time priority of orders and the available liquidity at different price levels. The price at which the maximum shares are matched is unique and usually does not have an event with an equal number of matched lots at different levels. In simple terms it checks to see the last price of stocks that were matched, then it looks at the line of people, the next person in line will get matched if they are willing to pay the amount compared to the remaining people, if it determines someone else may be willing to pay the amount instead for a greater quantity then it matches them.
#
With the data we are trying to predict the target, that is we are trying to predict the future move in the form of basis points that are equivalent to a 0.01% price move.
Some other things to consider is what is the true average. Since we are looking at a narrow window (last 10 mins of the closing market) should the average be 16 day moving average, 1 day moving average, or 200 second moving average?
#    
        Can we find trends in the closing price?
    Can we find trends in the shape of the market price in a certain bucket?


        """
)
#st.image(image2, caption=None)


st.header("About the Data")
st.write(""" 
In this project the training data comes from Kaggle competition **Optiver – Trading at the close** in the form of a single csv file with column definitions. """)
image1 = Image.open("desc.png")
st.image(image1, caption=None)


st.write("""
#
We have 199 stock ids, and we have 540 seconds for each. This is a narrow window. We are only given 9 minutes of data to look at. So, we have 107k data points. So, let’s explore the data. We are dealing with order book data, which doesn’t include your high and low prices that are expected to be in an auction. We have multiple prices since a buyer must wait for a competitive price.
#
The data does not include other forms of technical analysis such as exponential moving average, moving average convergence, volatility, relative strength index, volume moving average or statistical analysis. These types of statistics are commonly used in macroeconomic time series as a type of convolution to express how the shape of one function is modified by another function. Since the objective of our model is to predict a convolution, it would be reasonable to use other types of convolutions to summarize different data points. 
        """
)
image20 = Image.open("price.png")
st.image(image20, caption=None)

image33 = Image.open("matchedsize.png")
st.image(image33, caption=None)

st.header("Methods")
st.write(""" 
#         
        Feature Generation
This is a time series problem where we are trying to predict a function result around two features that shape a third feature. It would be good to see how a basic model performs with just the standard features compared to a model that includes both standard and engineered features.
         """)
image2 = Image.open("inspect.png")
st.image(image2, caption=None)


st.write("""
#         
We assume that the model will have a better evaluation score after more dimensions are added with the engineered features. 
First to clean the data we need to analyze the target data points since this is what our model will be predicting. We will remove all the columns where the target is NA or NULL. 
Next, we will need to use statistics to engineer new features. To keep the dimension scalable, we will look first at the features that shape the target. These are stock WAP, Index WAP and seconds. WAP is made up of Bid/ask price and bid/ask size. Another key to note is the target is shaped by seconds and not days or months. So, using seconds as the selected moving average period should return better results. 
We will have three sets of features: imbalance features, averages, and others. 
Volume, mid-price, stock weights, weighted WAP, WAP momentum, imbalance momentum, price spread, price pressure, market urgency, moving averages and RSI. These allow for smoothing the data and de-nosing the data.
         """)
image3 = Image.open("eng.png")
st.image(image3, caption=None)

st.write("""
#         
Some of the TA analysis can be found in python and R libraries but fundamentally the technical analysis offered is not great and we are only looking at a narrow window with less features. We will not import these libraries but rather create our own methods to generate these features.
#
         Feature Clustering
After generating new features, our dataset dimensions increase substantially. Since the target is a function that relates to other features, a correlation heatmap will not be ideal to use. It is beneficial to use a hierarchical cluster to see the subclusters of the features and see which is closest to the target basis point. 
This method is explained excellently in a blog post by P. Sanchez. Suppose that you work at an emergency center, and your job is to tell the pilots of firefighter helicopters to take off. You receive an emergency call because there’s a point in the city on fire and a helicopter needs to put it out. You need to choose which pilot to do this work. The farther helicopter will obviously arrive later than the closer, but you need to know exactly which is closest. This type of evaluation can be seen doing clustering such as k-means. After we find the features closest to the target or center of the cluster, we can standardize the criterion to determine the distance and select the best features to represent the data. After we increase the dimensions of the dataset, we summarize the data using clustering. See the dendrogram plot that represents this technique. This plot shows a link between each feature and links to the cluster themselves. You can draw a horizontal line to see the divided data in one cluster. We can also use clustering to see characteristics in the time series, such as the price change over seconds in relation to volume or volatility.  
""")
image4 = Image.open("clusterf.png")
st.image(image4, caption=None)


st.write(""" 
#        
        Feature Importance"""
)
image54 = Image.open("heatmap.png")
st.image(image54, caption=None)
image50 = Image.open("corr4.png")
st.image(image50, caption=None)
image51 = Image.open("corr3.png")
st.image(image51, caption=None)
image52 = Image.open("corr2.png")
st.image(image52, caption=None)
image53 = Image.open("corr1.png")
st.image(image53, caption=None)
st.write("Heatmap was not great because of the non linear relation to the Target")

st.write("""
#
         Model 1 and 2
In terms of the pipeline, a few feature engineering is done, then it is split into 3 models (GBM, LGB). The models are combined with regression that learns the optimal weights and the final prediction creates an optimal result. Gradient Boosting is great since it is easy to train and performance increases while using tabular data. Neural networks are good for large data after we increase the dimension size, but it is harder to tune.


        """
)
st.write("""
#         
         Standard Features""")
image7 = Image.open("scores.png")
st.image(image7, caption=None)

st.write("""
#         
         Generated Features""")
image5 = Image.open("catscore.png")
st.image(image5, caption=None)
image6 = Image.open("scores.png")
st.image(image6, caption=None)


st.header("Evaluation")
st.write(""" 
The model is evaluated on the Mean Absolute Error between the predicted return and the observed target.""")
image8 = Image.open("mae.png")
st.image(image8, caption=None)

st.write(""" 
#
         Test Set
The data used to train the model should not be centered around the past data. We propose using a central moving dataset that is equally spaced on either side of the point in the series where the mean is calculated. Since there are 480 dates. The training set would include day 45 to day 144 and the testing set would include day 145 to day 244 as well as day 1 to day 45. We will visualize the data to ensure there are no outliers. 


        """
)
image9 = Image.open("topdays.png")
st.image(image9, caption=None)


st.header("Story Telling and Conclusion")
st.write(""" 
#         
        We observe a somewhat skewed distribution of results.
         
 Given the MAE it seems that an optimal model is created by chance and cannot usually be repeated. This could mean that there is not enough training data for the model or there is too much noise in the dataset.""")
image44 = Image.open("skew1.png")
st.image(image44, caption=None)

image45 = Image.open("skew2.png")
st.image(image45, caption=None)

st.write("""" 
#
We see that the more dimensions the better the result. Another way to evaluate whether a successful model is achieved randomly due to lack of data or outliers such as significant external events, is to try time clustering. That is if a bust exists it will appear on a given date due to a significant external event like a GameStop saga. 
""")
st.write("""
#         
        Top performing days by volume""")
image46 = Image.open("topdays.png")
st.image(image46, caption=None)

st.write("""
#         
        Low performing days by volume""")
image47 = Image.open("low.png")
st.image(image47, caption=None)

st.write("""
#
The benchmark used mean data, but another strategy could be to use the mode, or more populous values to train the model and not the means that can be influenced by outliers. The dataset used anonymous stock id’s, the dataset size can be increased by obtaining current data to further split and train the model. This could be done by finding the tick size expressed in the WAP returns and converting the price in USD then comparing it to the actual NASDAQ data close price. This would allow us to map the IDs that are a good match, making a visual comparison and increase the dataset for each bucket to include over 200K data points. On the other hand, the prices of the competition are normalized so having the correct prices may create a more accurate result. Having the actual stock id would be great for this when pulling the accurate price with Yahoo API.
Consider where a low scoring model is found at random. A deep learning model that is trained off the errors to create new features and concatenating them with the original features should create an optimal model.
#
There are research papers around finding the best time bucket to use, like 100 seconds, 200 seconds or 300 seconds in the bucket or even using 1 day, 3-day 30-day, 40-day averages. If time and GPU capacity permitted, it would be worth running a loop on each to create more dimensions and test each to see how the MAE performs.
 


        """
)

st.header("Impact")
st.write(""" 
#
         Fix orders and Market Manipulation
We can derive the optimal execution strategy to fix orders to define various forms of unethical market manipulation. 
#         
There are 3 forms of market manipulation that can be exploited with a well-developed model: Banging on the fix, Information sharing, and profit sharing. If we were to model an unethical trading practice to close a large amount of profit, it will have an impact on the market that can be temporary or artificial. 
The strategy of banging the close is when a trader finds the best time to execute an order and sends an order to transact units of an asset at a specific time to insert a wedge between the fair price and the market price. 
#         
In other words, the trader attempts to move an asset price by executing a large volume of deals during the final moments of trading.
The strategy of information sharing is during the last minutes of the trading day a trader tells a broker to sell a specific amount of assets and gives explicit instructions that the order not to be executed until the last eight minutes of trading. This can influence the trading price when you move a large volume in a short window.
Profiting off the expense of customers is illegal. 
#
A cartel-like strategy would involve manipulating benchmark rates to benefit the position of certain traders.  Traders at different firms can manipulate the auction rates by disclosing confidential customer order information and trading positions to accommodate the interest of the collective group and agree to split the profits. They can engage in a bout of selling to push down the price to buy it back at a profit. 
#
         Risk Neutral Trader
The case of a risk neutral where the trader finds optimal execution strategy that spreads trades out evenly over time to minimize cost but is indifferent to the risk. The trader can use the model target points to insert the price impact and use it in maximizing his expected profits.
#
         Risk Averse Trader
Instead of an execution strategy with preferences to maximizing profits, risk averse traders avoid risk. The trader can use the model to execute a larger share of the assets at a fixed time. 
#
They sell more in earlier periods and ramp up as the fix is getting closer. 
We see that the model impact varies based on the trader’s motivation. We examined different motivations. Understanding the features effects on the model the trader can compare the asset holdings, time in bucket, volatility, per-period volume, risk aversion, and price impact (target basis points) to execute the optimal strategy. 
#         
The procedure would involve starting with all features and looking at eligible features that can be manipulated such as time in bucket and asset holdings. 
Next test the model with the manipulated features until there is a significant change. The method can be adjusted to the trader’s motivation as illustrated above. All strategies have an impact on the real-world market, and it can create a temporary, distorted, or artificial price.


        """
)

st.header("References")
st.write(""" 
#         
            https://www.kaggle.com/code/lognorm/de-anonymizing-stock-id
            https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/250324
            https://www.kaggle.com/code/tomforbes/optiver-trading-at-the-close-introduction
            https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970
            https://www.investopedia.com/terms/m/macd.asp
            https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/RSI
            https://www.marketvolume.com/analysis/volume_ma.asp
            https://c.mql5.com/forextsd/forum/174/banging_the_close_-_fix_orders_and_price_manipulation.pdf


        """
)

st.header("Code")
st.write("https://github.com/August33rd/finalprojectDS/blob/main/FinalProject.ipynb ")
