library(arules)
library(arulesViz)
library(readr)


datatr <- read.transactions("C:\\Users\\ms-msi\\Desktop\\Data Scientist\\Course2\\Task4\\ElectronidexTransactions2017.csv", format = 'basket', sep = ',', rm.duplicates = TRUE)

#explore dataset
inspect(sample(datatr, 10)) #each transaction consists of set with unique products purchased
length(datatr) #9835
size(datatr)
LIST(datatr)

itemFrequencyPlot(datatr, topN = 20, type = 'absolute') # absolute support vector of top 20 most frequent products
itemFrequencyPlot(datatr, topN = 20, type = 'relative') # relative support vector of top 20 most frequent products

image(datatr[1:100]) #another way to visualise transaction dataset
image(sample(datatr, 100)) # the same, but data is randomly chosen

#building out-of-the-box rules
outofthebox <- apriori(datatr, parameter = list(supp = 0.1, conf = 0.8))

#evaluation
inspect(outofthebox) #outputs no rules. this means we need to play with parameters.

#model tuning
#create an empty vector and put all rules inside
x <- vector(mode="list", length = 100)

x <- apriori(datatr, parameter = list(minlen = 2, supp = 0.0005, conf = 0.0005)

# total number of rules is 217'996

#top 2 strongest and popular rules:
strongest_005 <- subset(x, subset = lift > 2 & confidence > 0.65 & support > 0.0065)

#write to strongest.csv
write(strongest_005, file = "strongest.csv", sep = ",", append = TRUE, quote = TRUE,  row.names = FALSE, col.names = FALSE)

#lower support requirements subset: 2582 rules
top_lift_conf2 <- subset(x, subset = lift > 2 & confidence > 0.65 & support > 0.001)

#write to strong.csv
write(z[[i]], file = "strong.csv", sep = ",", append = TRUE, quote = TRUE,  row.names = FALSE, col.names = FALSE)

#plot the data:
plot(strongest_005, method="graph", control=list(type="items"))
plot(top_lift_conf2[1:5], method="graph", control=list(type="items"))
